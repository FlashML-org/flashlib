"""IVF-PQ fine scan via **decode-once workspace + tensor-core GEMM** (Triton).

The fused decode+GEMM kernels (Triton ``"gemm"`` / CuTe ``"cute_gemm"``)
re-decode every inverted list once *per query tile*: with ``qpl`` queries
probing a list, the same PQ codes are gathered and reconstructed
``ceil(qpl / BN)`` times, and that SIMT decode sits on the critical path
of every tensor-core tile (measured: the WGMMA itself is <10% of the
kernel -- the fine scan is decode-latency-bound, which is exactly the
tensor-core-utilisation gap this module closes).

The fix exploits the ADC algebra. With residual encoding the candidate
reconstruction ``xhat_i`` depends only on the stored codes -- *not* on
the query -- and

    ||(q - c_L) - xhat_i||^2
        = ||q - c_L||^2 + (||xhat_i||^2 + 2<c_L, xhat_i>) - 2<q, xhat_i>
        =   rqsq(q, L)  +            w_i                 - 2<q, xhat_i>

so the *only* per-(query, candidate) term is ``<q, xhat_i>`` -- a GEMM
between the **raw** queries and a **query-independent** decoded matrix.
That factorisation splits the fine scan into three clean stages:

  1. **Decode once** (:func:`_pq_decode_ws_kernel`). Each *probed* list's
     codes are decoded to bf16 sub-vectors in a transient workspace
     ``ws (rows, Dp)`` -- one gather per code per *batch* instead of one
     per code per query tile -- along with the fused per-row scalar
     ``w_i``. Coalesced HBM writes; the tiny codebook stays L2-resident.
     Only lists probed by >=1 query occupy workspace rows (CSR
     ``ws_offsets``), and the workspace is freed when the search returns.
  2. **GEMM scan** (:func:`_ivf_pq_ws_gemm_kernel`). Per ``(list,
     query-tile)``: a plain, dense ``(BN, Dp) x (Dp, BM)`` bf16
     ``tl.dot`` against the workspace -- a standard pipelined GEMM
     mainloop, so the tensor cores actually stay fed -- plus the two
     bias terms and an on-chip top-k whose per-chunk cost is one packed
     int32 min-reduction (score and column resolved together; see the
     kernel docstring). Partials land directly at their natural
     ``(query, probe-slot)`` rows.
  3. **Reduce.** One ``topk`` over each query's pooled partials and an
     exact-ADC re-rank of the oversampled pool (bf16 ranking never leaks
     into returned distances). No scatter -- stage 2 already wrote
     natural order.

Cost model vs the fused kernels: decode traffic collapses from
``O(list_len * Dp)`` gathers *per query tile* to ``O(list_len * Dp)``
total, and the scan becomes a dense bf16 GEMM at full cache-line
efficiency. On SIFT-1M (nq=10k, nprobe=32, m=32, H100) the fine scan
drops from ~4.3 ms (fused CuTe decode+WGMMA) to ~2.3 ms end-to-end; at
m=64 / GIST-960 geometries the gap widens to 2.5-3.3x because the fused
kernels' per-tile decode grows with ``m`` while this scan stays a GEMM.

The workspace is transient (``rows * Dp`` bf16, e.g. 256 MB for the
whole of SIFT-1M) and the driver only takes this road when it fits
:data:`_WS_BUDGET_BYTES`; otherwise it returns ``None`` and the caller
falls back to the fused kernels. ``by_residual=False`` is the same
algebra with ``rqsq = ||q||^2`` and ``w_i = ||xhat_i||^2``.
"""
from __future__ import annotations

from typing import Optional, Tuple

import torch
import triton
import triton.language as tl

from flashlib.primitives.knn.triton._common import _next_pow2


# Transient decoded-workspace cap. 2 GiB holds ~8M rows at Dp=128 (bf16)
# while staying a small fraction of any modern accelerator's HBM; larger
# touched sets decline (return None) and route back to the fused kernels.
_WS_BUDGET_BYTES = 2 << 30

# Default candidate-pool oversample for the exact re-rank. The scan ranks
# with bf16-truncated scores (~3 significant digits), so near-boundary
# candidates can swap ranks; a 2x pool absorbs that (same margin the fused
# bf16/tf32 GEMM kernels use) and the exact ADC re-rank restores true
# distances/order.
_WS_OVER = 2


@triton.jit
def _pq_decode_ws_kernel(
    codes_ptr, cb_ptr, cent_ptr,
    list_off_ptr, ws_off_ptr,
    ws_ptr, w_ptr,
    stride_codes_m, stride_codes_s,
    stride_cb_m, stride_cb_j, stride_cb_d,
    stride_cent_c, stride_cent_d,
    stride_ws_r, stride_ws_d,
    BY_RESIDUAL: tl.constexpr,
    DSUB: tl.constexpr, DP: tl.constexpr,
    D_TILE: tl.constexpr, BR: tl.constexpr,
):
    """Grid ``(max_row_chunks, nlist)``: decode one ``BR``-row chunk of one
    probed list into the workspace.

    Writes ``ws[row] = concat_s cb[s, code[row, s]]`` (bf16, coalesced) and
    ``w[row] = ||xhat||^2 (+ 2<c_list, xhat> if residual)`` (fp32). The
    codebook gather is L2-resident; only the ``ws`` write touches HBM.
    Untouched lists have ``ws_offsets[c+1] == ws_offsets[c]`` and exit.

    The workspace CSR is **BM-aligned** (each list's segment is padded to
    the scan tile): padding rows get ``xhat = 0`` and ``w = +inf`` so the
    scan kernel needs *no* per-chunk row masking -- a padded row scores
    ``+inf`` and never inserts. This kernel covers the padded length.
    """
    pid_r = tl.program_id(0)
    pid_c = tl.program_id(1)

    ws_start = tl.load(ws_off_ptr + pid_c)
    ws_end = tl.load(ws_off_ptr + pid_c + 1)
    n_pad = ws_end - ws_start                 # padded segment length
    if pid_r * BR >= n_pad:
        return
    c_start = tl.load(list_off_ptr + pid_c)
    c_end = tl.load(list_off_ptr + pid_c + 1)
    n = c_end - c_start                       # true list length

    r = pid_r * BR + tl.arange(0, BR)
    pmask = r < n_pad                         # rows to write (incl. padding)
    rmask = r < n                             # rows holding real candidates
    orig = (c_start + r).to(tl.int64)         # stored-row (codes) position
    wsrow = (ws_start + r).to(tl.int64)       # workspace row

    # One code load per (row, sub-quantizer), broadcast to its DSUB columns
    # (vs one per (row, column) -- a DSUB-fold load-instruction cut) when
    # the tile is a whole number of sub-quantizers. Only worth it for short
    # sub-vectors: at large DSUB the (BR, S_TILE) load is too narrow and
    # the broadcast's cross-thread shuffle costs more than the saved loads
    # (measured: dsub=4 2x faster, dsub=16 1.7x slower).
    EXPAND: tl.constexpr = (
        (D_TILE % DSUB == 0) and (DP % DSUB == 0) and (DSUB <= 8)
    )
    S_TILE: tl.constexpr = D_TILE // DSUB if EXPAND else 1

    wacc = tl.zeros([BR], dtype=tl.float32)
    for d0 in range(0, DP, D_TILE):
        d = d0 + tl.arange(0, D_TILE)
        dmask = d < DP
        s_of_d = d // DSUB
        o_of_d = d % DSUB
        if EXPAND:
            s_off = d0 // DSUB + tl.arange(0, S_TILE)
            smask = s_off < (DP // DSUB)
            code8 = tl.load(
                codes_ptr + orig[:, None] * stride_codes_m
                + s_off[None, :] * stride_codes_s,
                mask=rmask[:, None] & smask[None, :], other=0,
            )                                                 # (BR, S_TILE)
            code = tl.reshape(
                tl.broadcast_to(code8[:, :, None], (BR, S_TILE, DSUB)),
                (BR, D_TILE),
            ).to(tl.int64)
        else:
            code = tl.load(
                codes_ptr + orig[:, None] * stride_codes_m
                + s_of_d[None, :] * stride_codes_s,
                mask=rmask[:, None] & dmask[None, :], other=0,
            ).to(tl.int64)                                    # (BR, D_TILE)
        cbv = tl.load(
            cb_ptr + s_of_d[None, :] * stride_cb_m + code * stride_cb_j
            + o_of_d[None, :] * stride_cb_d,
            mask=rmask[:, None] & dmask[None, :], other=0.0,
        )                                                     # (BR, D_TILE) bf16
        cbv = tl.where(rmask[:, None], cbv, 0.0)              # zero padding rows
        tl.store(
            ws_ptr + wsrow[:, None] * stride_ws_r + d[None, :] * stride_ws_d,
            cbv, mask=pmask[:, None] & dmask[None, :],
        )
        xf = cbv.to(tl.float32)
        wacc += tl.sum(xf * xf, axis=1)
        if BY_RESIDUAL:
            cent = tl.load(
                cent_ptr + pid_c * stride_cent_c + d * stride_cent_d,
                mask=dmask, other=0.0,
            )                                                 # (D_TILE,)
            wacc += 2.0 * tl.sum(xf * cent[None, :], axis=1)
    wacc = tl.where(rmask, wacc, float("inf"))                # pad rows -> +inf
    tl.store(w_ptr + wsrow, wacc, mask=pmask)


@triton.jit
def _ws_sample_floor_kernel(
    q_ptr, sorted_qid_ptr, rqsq_ptr,
    ws_ptr, w_ptr,
    q_offsets_ptr, ws_off_ptr,
    out_ptr,
    stride_q_n, stride_q_d,
    WS_RD: tl.constexpr,
    DP: tl.constexpr, D_INNER: tl.constexpr,
    BN: tl.constexpr, BM: tl.constexpr,
):
    """Grid ``(MAX_QTILES, nlist)``: per pair, the min score over the
    list's *first* ``BM`` candidates -- a cheap upper bound used to seed
    the scan kernel's insert threshold. Writes ``out[pair_pos]`` (sorted
    order; the driver permutes). One GEMM per program, no top-k. The
    workspace segment is BM-aligned, so the first chunk is full (padding
    rows carry ``w=+inf``); empty segments write ``+inf``."""
    pid_qt = tl.program_id(0)
    pid_c = tl.program_id(1)
    qstart = tl.load(q_offsets_ptr + pid_c)
    qend = tl.load(q_offsets_ptr + pid_c + 1)
    qcount = qend - qstart
    if pid_qt * BN >= qcount:
        return
    i_range = tl.arange(0, BN)
    q_local = pid_qt * BN + i_range
    q_mask = q_local < qcount
    pair_pos = (qstart + q_local).to(tl.int64)
    qid = tl.load(sorted_qid_ptr + pair_pos, mask=q_mask, other=0).to(tl.int64)
    rqsq = tl.load(rqsq_ptr + pair_pos, mask=q_mask, other=0.0)
    ws_start = tl.load(ws_off_ptr + pid_c)
    ws_end = tl.load(ws_off_ptr + pid_c + 1)
    if ws_start >= ws_end:                    # untouched/empty list
        tl.store(out_ptr + pair_pos, tl.full([BN], float("inf"), tl.float32),
                 mask=q_mask)
        return

    d_offs = tl.arange(0, D_INNER).to(tl.int64)
    d_mask = d_offs < DP
    x_tile = tl.load(
        q_ptr + qid[:, None] * stride_q_n + d_offs[None, :] * stride_q_d,
        mask=q_mask[:, None] & d_mask[None, :], other=0.0,
    ).to(tl.bfloat16)

    bm_range = tl.arange(0, BM)
    rows = ws_start + bm_range.to(tl.int64)
    if D_INNER >= DP:
        xhat = tl.load(
            ws_ptr + rows[:, None] * WS_RD + d_offs[None, :],
            mask=d_mask[None, :], other=0.0,
        )
        cross = tl.dot(x_tile, tl.trans(xhat))
    else:
        cross = tl.zeros([BN, BM], dtype=tl.float32)
        for d_start in range(0, DP, D_INNER):
            d_offs = (d_start + tl.arange(0, D_INNER)).to(tl.int64)
            d_mask = d_offs < DP
            x_sub = tl.load(
                q_ptr + qid[:, None] * stride_q_n + d_offs[None, :] * stride_q_d,
                mask=q_mask[:, None] & d_mask[None, :], other=0.0,
            ).to(tl.bfloat16)
            xhat_sub = tl.load(
                ws_ptr + rows[:, None] * WS_RD + d_offs[None, :],
                mask=d_mask[None, :], other=0.0,
            )
            cross += tl.dot(x_sub, tl.trans(xhat_sub))
    wv = tl.load(w_ptr + rows)
    dist = rqsq[:, None] + wv[None, :] - 2.0 * cross
    dist = tl.maximum(dist, 0.0)
    dist = tl.where(q_mask[:, None], dist, float("inf"))
    tl.store(out_ptr + pair_pos, tl.min(dist, axis=1), mask=q_mask)


@triton.jit
def _ivf_pq_ws_gemm_kernel(
    q_ptr, sorted_qid_ptr, sorted_slot_ptr, rqsq_ptr,
    ws_ptr, w_ptr, thr_ptr,
    q_offsets_ptr, ws_off_ptr, list_off_ptr,
    pv_ptr, pi_ptr,
    nprobe, ws_rows,
    stride_q_n, stride_q_d,
    WS_RD: tl.constexpr,
    USE_THR: tl.constexpr,
    DP: tl.constexpr, D_INNER: tl.constexpr,
    BN: tl.constexpr, BM: tl.constexpr,
    TOPK_PAD: tl.constexpr, MAX_STEPS: tl.constexpr,
):
    """Grid ``(MAX_QTILES, nlist)``: score one query tile against one probed
    list's *decoded* workspace rows with a pipelined bf16 ``tl.dot``.

    ``dist = rqsq[pair] + w[row] - 2 <q, ws[row]>`` -- both biases exact
    fp32, the cross term bf16 (the driver's oversampled exact re-rank makes
    the returned distances ADC-exact).

    The on-chip top-k is engineered so the *unconditional* per-chunk cost
    is a single packed reduction (the measured wall for every guarded
    epilogue at BN=64 -- with 64 queries sharing a program, any per-row
    work fires almost every chunk, so it must be cheap):

      * scores are clamped to ``>= 0`` and packed as
        ``(fp32_bits & 0xFFFF0000) | column`` into one int32 -- an
        order-preserving key (positive floats compare as unsigned ints;
        the low mantissa bits are traded for the column id), so one
        ``tl.min`` resolves the chunk's best value *and* its column,
      * the running top-k stays (value, index) register pairs; an insert
        replaces the current worst slot via ``tl.argmax`` (exactly one
        slot, so bf16-truncation ties never collapse two candidates),
      * pops repeat while any row still inserts (masking each popped
        column), so per-pair partials are exact under the truncated-score
        order; partial *values* are bf16-truncated, which the driver's
        oversampled exact re-rank absorbs.

    Partials are written at the pair's **natural** ``(query, probe-slot)``
    row (``sorted_slot`` carries the probe slot through the list-grouped
    ordering), so the host reduce needs no scatter.
    """
    pid_qt = tl.program_id(0)
    pid_c = tl.program_id(1)

    qstart = tl.load(q_offsets_ptr + pid_c)
    qend = tl.load(q_offsets_ptr + pid_c + 1)
    qcount = qend - qstart
    if pid_qt * BN >= qcount:
        return

    i_range = tl.arange(0, BN)
    q_local = pid_qt * BN + i_range
    q_mask = q_local < qcount                              # (BN,)
    pair_pos = (qstart + q_local).to(tl.int64)             # (BN,) sorted-pair rows
    qid = tl.load(sorted_qid_ptr + pair_pos, mask=q_mask, other=0).to(tl.int64)
    slot = tl.load(sorted_slot_ptr + pair_pos, mask=q_mask, other=0).to(tl.int64)
    rqsq = tl.load(rqsq_ptr + pair_pos, mask=q_mask, other=0.0)   # (BN,) fp32
    if USE_THR:
        # Per-QUERY upper bound on its true k-th distance (from the sample
        # floors): candidates above it can never reach the final top-k, so
        # it seeds the insert bar and kills most early-chunk insert storms.
        thr = tl.load(thr_ptr + qid, mask=q_mask, other=float("inf"))
    else:
        thr = tl.full([BN], float("inf"), dtype=tl.float32)

    ws_start = tl.load(ws_off_ptr + pid_c)
    ws_end = tl.load(ws_off_ptr + pid_c + 1)
    c_start = tl.load(list_off_ptr + pid_c).to(tl.int32)

    # Raw-query tile (bf16 GEMM A operand). No centroid subtract -- the
    # residual algebra lives entirely in the rqsq / w biases.
    NO_DMASK: tl.constexpr = D_INNER == DP
    if D_INNER >= DP:
        d_offs = tl.arange(0, D_INNER).to(tl.int64)
        d_mask = d_offs < DP
        x_tile = tl.load(
            q_ptr + qid[:, None] * stride_q_n + d_offs[None, :] * stride_q_d,
            mask=q_mask[:, None] & d_mask[None, :], other=0.0,
        ).to(tl.bfloat16)                                  # (BN, D_INNER)

    I32_MAX: tl.constexpr = 2147483647
    topk_vals = tl.full([BN, TOPK_PAD], float("inf"), dtype=tl.float32)
    topk_idxs = tl.full([BN, TOPK_PAD], -1, dtype=tl.int32)
    topk_max = tl.full([BN], float("inf"), dtype=tl.float32)
    k_range = tl.arange(0, TOPK_PAD)
    bm_range = tl.arange(0, BM)

    # The workspace segment is BM-aligned (decode pads with w=+inf rows),
    # so every chunk is full: no row masks anywhere in the mainloop. While
    # ``Dp`` fits one K tile the ``ws`` operand is read through a **block
    # pointer** (exact shape/stride/alignment -> vectorised, pipelined
    # loads). At high ``Dp`` the K loop would make the pipeliner buffer
    # every K slice at once (SMEM blow-up), so that path keeps plain
    # incremented pointer-tensor loads instead.
    n_chunks = (ws_end - ws_start) // BM
    if D_INNER >= DP:
        xh_bp = tl.make_block_ptr(
            base=ws_ptr, shape=(ws_rows, DP), strides=(WS_RD, 1),
            offsets=(ws_start.to(tl.int32), 0), block_shape=(BM, D_INNER),
            order=(1, 0),
        )
    else:
        xh_ptrs = ws_ptr + (ws_start + bm_range)[:, None].to(tl.int64) * WS_RD \
            + tl.arange(0, D_INNER)[None, :]
    w_ptrs = w_ptr + ws_start + bm_range
    NK: tl.constexpr = (DP + D_INNER - 1) // D_INNER
    for ci in range(n_chunks):
        if D_INNER >= DP:
            if NO_DMASK:
                xhat = tl.load(xh_bp)                      # (BM, D_INNER) bf16
            else:
                xhat = tl.load(xh_bp, boundary_check=(1,), padding_option="zero")
            cross = tl.dot(x_tile, tl.trans(xhat))         # (BN, BM) fp32
            xh_bp = tl.advance(xh_bp, (BM, 0))
        else:
            # Dynamic K loop -> the software pipeliner treats it as a
            # standard GEMM mainloop (num_stages-deep operand buffers)
            # instead of materialising every unrolled K slice in SMEM.
            cross = tl.zeros([BN, BM], dtype=tl.float32)
            for kk in range(NK):
                d_offs = (kk * D_INNER + tl.arange(0, D_INNER)).to(tl.int64)
                d_mask = d_offs < DP
                x_sub = tl.load(
                    q_ptr + qid[:, None] * stride_q_n
                    + d_offs[None, :] * stride_q_d,
                    mask=q_mask[:, None] & d_mask[None, :], other=0.0,
                ).to(tl.bfloat16)
                xh_sub = tl.load(
                    xh_ptrs + kk * D_INNER, mask=d_mask[None, :], other=0.0)
                cross += tl.dot(x_sub, tl.trans(xh_sub))
            xh_ptrs += BM * WS_RD

        wv = tl.load(w_ptrs)                                       # (BM,) fp32
        w_ptrs += BM
        dist = rqsq[:, None] + wv[None, :] - 2.0 * cross
        # Clamp (bf16 rounding can push tiny distances negative) so the
        # sign bit is clear and int32 compare == float compare. Padding
        # rows carry w=+inf -> dist=+inf -> never below any insert bar.
        dist = tl.maximum(dist, 0.0)
        # Packed key: high 16 bits = bf16-truncated score, low 16 = column
        # (MASK_HI is 0xFFFF0000 as a signed int32). Query-masked rows are
        # forced to I32_MAX so they never insert (their stores are masked).
        MASK_HI: tl.constexpr = -65536
        MASK_LO: tl.constexpr = 65535
        packed = (dist.to(tl.int32, bitcast=True) & MASK_HI) | bm_range[None, :]
        packed = tl.where(q_mask[:, None], packed, I32_MAX)

        # Guarded pop loop; every iteration is ONE int32 min-reduction plus
        # elementwise updates. Masking a popped column unconditionally is
        # safe: if a row didn't insert, everything after its min is worse.
        # The insert bar is min(running worst, per-query threshold): a
        # candidate above the threshold can never be in the final top-k,
        # so skipping it loses nothing (the merge drops it anyway).
        rm = tl.min(packed, axis=1)                        # (BN,) best (val|col)
        bar = tl.minimum(topk_max, thr)
        need = (rm & MASK_HI) < (bar.to(tl.int32, bitcast=True) & MASK_HI)
        if tl.max(need.to(tl.int32)) > 0:
            for _step in range(MAX_STEPS):
                if tl.max(need.to(tl.int32)) > 0:
                    val = (rm & MASK_HI).to(tl.float32, bitcast=True)
                    col = rm & MASK_LO
                    worst_slot = tl.argmax(topk_vals, axis=1)          # (BN,)
                    repl = (k_range[None, :] == worst_slot[:, None]) & need[:, None]
                    topk_vals = tl.where(repl, val[:, None], topk_vals)
                    cand = (ci * BM + col).to(tl.int32)
                    topk_idxs = tl.where(repl, cand[:, None], topk_idxs)
                    topk_max = tl.max(topk_vals, axis=1)
                    packed = tl.where(packed == rm[:, None], I32_MAX, packed)
                    rm = tl.min(packed, axis=1)
                    bar = tl.minimum(topk_max, thr)
                    need = (rm & MASK_HI) < (
                        bar.to(tl.int32, bitcast=True) & MASK_HI)

    # list-local candidate -> original stored row (codes CSR and ws CSR are
    # aligned per list, so the offset is just the list's codes start).
    out_pos = qid * nprobe + slot                          # natural pair rows
    valid_idx = topk_idxs >= 0
    glob = tl.where(valid_idx, topk_idxs + c_start, -1).to(tl.int32)
    tl.store(
        pv_ptr + (out_pos[:, None] * TOPK_PAD + k_range[None, :]),
        topk_vals, mask=q_mask[:, None],
    )
    tl.store(
        pi_ptr + (out_pos[:, None] * TOPK_PAD + k_range[None, :]),
        glob, mask=q_mask[:, None],
    )


def ivf_pq_fine_scan_ws(
    Qp: torch.Tensor,
    centroids: torch.Tensor,
    codebooks: torch.Tensor,
    codes: torch.Tensor,
    probed: torch.Tensor,
    list_offsets: torch.Tensor,
    k: int,
    *,
    by_residual: bool,
    max_list_len: int = 0,
    over: Optional[int] = None,
    BN: int = 64,
    BM: int = 128,
    BR: int = 64,
    num_stages: int = 3,
    num_warps: int = 4,
    thresh: bool = True,
    ws_budget_bytes: Optional[int] = None,
) -> Optional[Tuple[torch.Tensor, torch.Tensor]]:
    """Decode-once workspace + GEMM fine scan, then exact-ADC re-rank.

    Args:
        Qp: ``(nq, Dp)`` padded fp32 queries.
        centroids: ``(nlist, Dp)`` coarse centroids (fp32).
        codebooks: ``(m, ksub, dsub)`` PQ sub-centroids (fp32).
        codes: ``(M, m)`` uint8 codes, cell-contiguous.
        probed: ``(nq, nprobe)`` int32 probed list ids.
        list_offsets: ``(nlist + 1,)`` int64 CSR offsets.
        k: neighbours per query.
        by_residual: residual vs direct PQ encoding.
        max_list_len: longest list (sizes the decode grid without a sync);
            0 derives it from ``list_offsets`` (one extra D2H).
        over: candidate-pool oversample factor for the exact re-rank.
        thresh: seed the scan's insert bar with a per-query threshold
            derived from a cheap sample pass (``T_q`` = k-th smallest of
            the pair-wise first-chunk minima -- a valid upper bound on the
            true k-th distance, so pruning with it is lossless). Cuts the
            insert-loop work ~2x on uniform data; disable to A/B.
        ws_budget_bytes: decoded-workspace cap (default
            :data:`_WS_BUDGET_BYTES`).

    Returns ``(vals, pos)`` like the other fine-scan drivers -- ``vals``
    ``(nq, k)`` ADC-exact squared-L2 (fp32), ``pos`` ``(nq, k)`` int64
    stored-row positions (``-1`` pad) -- or ``None`` when the touched
    workspace exceeds the budget (caller falls back to fused kernels).
    """
    assert Qp.is_cuda and codes.is_cuda
    nq, Dp = Qp.shape
    nprobe = probed.shape[1]
    nlist = list_offsets.shape[0] - 1
    dsub = codebooks.shape[2]
    device = Qp.device
    budget = ws_budget_bytes or _WS_BUDGET_BYTES
    if over is None:
        over = _WS_OVER

    Qp = Qp.contiguous()
    centroids = centroids.contiguous()
    codes = codes.contiguous()
    list_offsets = list_offsets.contiguous().to(torch.int64)
    cb_bf16 = codebooks.contiguous().to(torch.bfloat16)     # (m, ksub, dsub)

    # ── inverse map + compact workspace CSR (single D2H sync, hidden
    #    behind the decode/rqsq launches queued before it) ──────────────
    probed = probed.contiguous()
    if probed.dtype != torch.int32:
        probed = probed.to(torch.int32)
    flat = probed.view(-1)                                   # (P,) int32
    P = flat.numel()
    sorted_vals, perm = torch.sort(flat, stable=True)        # int32 radix
    sorted_qid = perm.div(nprobe, rounding_mode="floor").to(torch.int32)
    sorted_slot = (perm - sorted_qid.to(torch.int64) * nprobe).to(torch.int32)
    # CSR offsets straight off the sorted keys: one searchsorted instead of
    # bincount + cumsum + fill.
    q_offsets = torch.searchsorted(
        sorted_vals,
        torch.arange(nlist + 1, device=device, dtype=torch.int32),
    )
    qcounts = q_offsets[1:] - q_offsets[:-1]                 # (nlist,)

    lengths = list_offsets[1:] - list_offsets[:-1]           # (nlist,)
    # BM-aligned segments: the decode kernel writes w=+inf padding rows so
    # the scan mainloop runs full, mask-free chunks (see decode docstring).
    ws_lens = torch.where(
        qcounts > 0, ((lengths + (BM - 1)) // BM) * BM, torch.zeros_like(lengths))
    ws_offsets = torch.zeros(nlist + 1, dtype=torch.int64, device=device)
    ws_offsets[1:] = ws_lens.cumsum(0)

    # ── per-pair exact bias rqsq (queued before the stats sync) ────────
    if by_residual:
        from flashlib.kernels.distance.triton.knn_gather_l2sq import (
            triton_knn_gather_sqdist,
        )
        coarse_d = triton_knn_gather_sqdist(
            Qp.unsqueeze(0), centroids.unsqueeze(0), probed.unsqueeze(0),
        )[0]                                                 # (nq, nprobe) fp32
        rqsq_pair = coarse_d.reshape(-1)[perm].contiguous()
    else:
        qsq = (Qp * Qp).sum(dim=1)                           # (nq,)
        rqsq_pair = qsq[sorted_qid.to(torch.int64)].contiguous()

    stats = torch.stack([qcounts.max(), ws_offsets[-1]]).cpu()
    max_qcount, ws_rows = int(stats[0]), int(stats[1])
    if ws_rows * Dp * 2 > budget:
        return None                                          # caller falls back
    MAX_QTILES = max(1, (max_qcount + BN - 1) // BN)

    # ── stage 1: decode the probed lists once ──────────────────────────
    ws = torch.empty((max(ws_rows, 1), Dp), device=device, dtype=torch.bfloat16)
    w = torch.empty((max(ws_rows, 1),), device=device, dtype=torch.float32)
    if max_list_len <= 0:
        max_list_len = int(lengths.max().item())
    max_seg = triton.cdiv(max_list_len, BM) * BM             # padded longest
    D_TILE = min(_next_pow2(Dp), 128)
    grid_dec = (triton.cdiv(max(max_seg, 1), BR), nlist)
    _pq_decode_ws_kernel[grid_dec](
        codes, cb_bf16, centroids,
        list_offsets, ws_offsets,
        ws, w,
        codes.stride(0), codes.stride(1),
        cb_bf16.stride(0), cb_bf16.stride(1), cb_bf16.stride(2),
        centroids.stride(0), centroids.stride(1),
        ws.stride(0), ws.stride(1),
        BY_RESIDUAL=bool(by_residual),
        DSUB=dsub, DP=Dp,
        D_TILE=D_TILE, BR=BR,
        num_warps=4,
    )

    # ── stage 1.5 (optional): per-query threshold from a sample pass ───
    TOPK_PAD = _next_pow2(k)
    D_INNER = _next_pow2(Dp) if Dp <= 256 else 128
    if thresh:
        samp = torch.full((P,), float("inf"), device=device, dtype=torch.float32)
        _ws_sample_floor_kernel[(MAX_QTILES, nlist)](
            Qp, sorted_qid, rqsq_pair,
            ws, w,
            q_offsets, ws_offsets,
            samp,
            Qp.stride(0), Qp.stride(1),
            WS_RD=Dp,
            DP=Dp, D_INNER=D_INNER, BN=BN, BM=BM,
            num_warps=num_warps, num_stages=num_stages,
        )
        # k-th smallest pair floor upper-bounds the query's k-th distance
        # (k probed lists are disjoint candidate sets, each contributing a
        # candidate <= its floor). Inflate by one bf16 ulp so truncated-
        # score compares at the boundary stay inclusive.
        samp_nat = torch.empty_like(samp)
        samp_nat[perm] = samp
        kk = min(k, nprobe)
        T = samp_nat.view(nq, nprobe).topk(kk, dim=1, largest=False).values[:, -1]
        thr_q = (T + T.abs() * 2.0 ** -8).contiguous()
    else:
        thr_q = torch.empty((1,), device=device, dtype=torch.float32)

    # ── stage 2: pipelined bf16 GEMM scan (writes natural pair order) ──
    MAX_STEPS = min(k, BM)
    pv = torch.full((P, TOPK_PAD), float("inf"), device=device,
                    dtype=torch.float32)
    pi = torch.full((P, TOPK_PAD), -1, device=device, dtype=torch.int32)

    _ivf_pq_ws_gemm_kernel[(MAX_QTILES, nlist)](
        Qp, sorted_qid, sorted_slot, rqsq_pair,
        ws, w, thr_q,
        q_offsets, ws_offsets, list_offsets,
        pv, pi,
        nprobe, max(ws_rows, 1),
        Qp.stride(0), Qp.stride(1),
        WS_RD=Dp,
        USE_THR=bool(thresh),
        DP=Dp, D_INNER=D_INNER,
        BN=BN, BM=BM,
        TOPK_PAD=TOPK_PAD, MAX_STEPS=MAX_STEPS,
        num_warps=num_warps, num_stages=num_stages,
    )

    # ── stage 3: pool + exact-ADC re-rank (partials already natural) ───
    from flashlib.primitives.ivf_pq.cutedsl.fine_scan_host import reduce_rerank

    return reduce_rerank(
        pv, pi, None, nq, nprobe, k,
        Qp=Qp, centroids=centroids, codebooks=codebooks.contiguous(),
        codes=codes, list_offsets=list_offsets, by_residual=by_residual,
        over=over,
    )


__all__ = ["ivf_pq_fine_scan_ws"]
