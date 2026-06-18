"""Efficient shared-memory ADC-LUT IVF-PQ fine scan (CuTe DSL, SM90).

The classic cuVS / FAISS asymmetric-distance LUT algorithm, executed so it
is actually fast at high ``D`` (where decode+GEMM loses because its
per-candidate cost scales with ``D`` while a LUT scan costs only ``m``
lookups, independent of ``D``) **and** at tiny batch (where cuVS wins by
keeping the whole pipeline to a couple of launches).

Design (each item fixed a measured bottleneck):

  1. **One CTA per ``(query, probe)`` -- no inverse map.** Unlike
     decode+GEMM, the LUT shares nothing across the queries probing a
     list (each query has its own table), so the cluster-centric grouping
     (host argsort + bincount + cumsum + a ``max().item()`` D2H sync) was
     pure overhead. The kernel is launched on a ``(nq, nprobe)`` grid and
     reads its ``(query, list)`` pair straight from ``probed`` -- the
     partials land in natural ``(nq, nprobe, k)`` order, so the host side
     is just one ``topk`` merge (no scatter, no sync). This is what cuVS
     does and it removes ~0.5 ms of fixed overhead at small batch.
  2. **Precomputed-table LUT (kills the build).** With residual encoding
     the naive LUT rebuilds ``‖(q-c_L)[s]-cb[s,j]‖²`` -- an ``m·256·dsub``
     SIMT loop that re-reads the codebook -- for *every* probed list,
     which dominated runtime. Instead decompose (FAISS precomputed tables):
     ``‖(q-c_L)-xhat‖² = ‖q-c_L‖² + Σ_s(‖cb[s,j]‖² + 2⟨c_L[s],cb[s,j]⟩
     - 2⟨q[s],cb[s,j]⟩)``. The ``dsub`` cross terms become a per-query GEMM
     (``qterm2``) and a cached per-index GEMM (``cterm2``) on tensor cores,
     so the device build is a single subtract per ``(s,j)``. The per-list
     offset ``‖q-c_L‖²`` is added back so partials rank across lists.
  3. **fp32 SMEM LUT -> exact, no re-rank.** The precomputed-table
     decomposition subtracts two ``O(dsub)`` cross terms, so storing the
     LUT in fp16 lost ~1e-3 relative precision and forced an oversampled
     exact-decode re-rank to restore ADC-exactness. Storing the LUT in
     fp32 (accumulated in fp32) keeps ~1e-6, which *is* ADC-exact to fp
     tolerance, so the re-rank is dropped entirely and the per-probe
     top-k is final. (fp16 + re-rank is still available via ``lut_dtype``
     for the rare large-``m`` LUT where the fp32 table costs occupancy.)
  4. **One query per CTA + warp-shuffle top-k.** Each thread keeps a
     register top-k over its strided candidates with no scan-time
     barriers, then an iterative ``shuffle_sync_bfly`` block arg-min
     merges the ``T`` local lists.
  5. **Warp-coalesced interleaved codes.** The cell-contiguous ``(M, m)``
     codes are read uncoalesced (a warp's 32 lanes hit 32 rows at stride
     ``m``), which ncu showed was the dominant high-D stall (L1TEX
     scoreboard). A one-time interleaved copy (groups of 32 candidates,
     ``il[L] + g*m*32 + s*32 + lane``) makes each warp read one contiguous
     32-byte run per sub-quantizer. Built + cached per index; the canonical
     codes layout and the other scan kernels are untouched.
"""
from __future__ import annotations

import weakref
from typing import Optional, Tuple

import torch

import cutlass
import cutlass.cute as cute
import cutlass.cute.runtime as cute_rt
import cutlass.utils as utils
import cuda.bindings.driver as cuda

from flashlib.primitives.knn.triton._common import _next_pow2
from flashlib.primitives.ivf_pq.cutedsl.fine_scan_host import reduce_rerank


_INF = 3.4e38


class HopperIvfPqSharedLut:
    """Precomputed-table ADC-LUT fine-scan kernel (SM90, SIMT).

    One CTA owns one ``(query, probe)`` pair: it builds that pair's LUT in
    shared memory (a per-(s,j) subtract of the cached cterm / per-query
    qterm tensors), scans the probed list's PQ codes with ``m`` SMEM
    gathers per candidate, and reduces a per-CTA top-k with a
    warp-shuffle block arg-min. Partials are written in natural
    ``(query, probe)`` order, so no host-side inverse map / scatter is
    needed.

    Config (baked at compile time, so the kernel is reused across every
    ``nq`` / ``nprobe`` -- those are launch-grid dims read from the
    tensors, never compile constants):
        M: number of sub-quantizers (``m``).
        DSUB: sub-vector dimension; KSUB: sub-centroids (256).
        DP: padded working dim (``M * DSUB``); TOPK_PAD: padded top-k width.
        by_residual: residual vs direct PQ encoding.
        lut_dtype: SMEM LUT storage dtype (fp32 = exact, fp16 = compact).
        threads: CTA width.
    """

    def __init__(
        self, *, M, DSUB, KSUB, DP, TOPK_PAD,
        by_residual, threads, lut_dtype=cutlass.Float32,
    ):
        self.M = M
        self.DSUB = DSUB
        self.KSUB = KSUB
        self.DP = DP
        self.TOPK_PAD = TOPK_PAD
        self.by_residual = bool(by_residual)
        self.threads = threads
        self.lut_dtype = lut_dtype

    @cute.jit
    def __call__(
        self,
        mQ: cute.Tensor, mCent: cute.Tensor, mProbed: cute.Tensor,
        mListOff: cute.Tensor, mCodesIl: cute.Tensor, mIlOff: cute.Tensor,
        mCT: cute.Tensor, mQT: cute.Tensor, mPV: cute.Tensor, mPI: cute.Tensor,
        stream: cuda.CUstream,
    ):
        M, KSUB, DP = self.M, self.KSUB, self.DP
        TOPK_PAD, T = self.TOPK_PAD, self.threads
        POOL = T * TOPK_PAD
        NWARP = T // 32

        @cute.struct
        class SharedStorage:
            sRQ: cute.struct.MemRange[cutlass.Float32, DP]
            sLUT: cute.struct.MemRange[self.lut_dtype, M * KSUB]
            sQRSQ: cute.struct.MemRange[cutlass.Float32, 1]
            sPoolV: cute.struct.MemRange[cutlass.Float32, POOL]
            sPoolI: cute.struct.MemRange[cutlass.Int32, POOL]
            sWarpV: cute.struct.MemRange[cutlass.Float32, NWARP]
            sWarpP: cute.struct.MemRange[cutlass.Int32, NWARP]
            sOutV: cute.struct.MemRange[cutlass.Float32, TOPK_PAD]
            sOutI: cute.struct.MemRange[cutlass.Int32, TOPK_PAD]

        self.shared_storage = SharedStorage

        # Grid is read from the (dynamic) probed shape so one compiled
        # kernel serves every batch size: x = query, y = probe slot.
        nq = mProbed.shape[0]
        nprobe = mProbed.shape[1]
        grid = (nq, nprobe, 1)
        self.kernel(
            mQ, mCent, mProbed, mListOff, mCodesIl, mIlOff, mCT, mQT, mPV, mPI,
        ).launch(grid=grid, block=[self.threads, 1, 1], stream=stream)

    @cute.kernel
    def kernel(
        self,
        mQ: cute.Tensor, mCent: cute.Tensor, mProbed: cute.Tensor,
        mListOff: cute.Tensor, mCodesIl: cute.Tensor, mIlOff: cute.Tensor,
        mCT: cute.Tensor, mQT: cute.Tensor, mPV: cute.Tensor, mPI: cute.Tensor,
    ):
        M = self.M
        KSUB, DP, TOPK_PAD = self.KSUB, self.DP, self.TOPK_PAD
        T = self.threads
        M4 = (M // 4) * 4

        pid_q, pid_p, _ = cute.arch.block_idx()
        tid, _, _ = cute.arch.thread_idx()

        c = cutlass.Int32(mProbed[pid_q, pid_p])
        c_start = mListOff[c]
        c_end = mListOff[c + 1]

        smem = utils.SmemAllocator()
        storage = smem.allocate(self.shared_storage)
        sRQ = storage.sRQ.get_tensor(cute.make_layout(DP))
        sLUT = storage.sLUT.get_tensor(cute.make_layout(M * KSUB))
        sQRSQ = storage.sQRSQ.get_tensor(cute.make_layout(1))
        sPoolV = storage.sPoolV.get_tensor(cute.make_layout(T * TOPK_PAD))
        sPoolI = storage.sPoolI.get_tensor(cute.make_layout(T * TOPK_PAD))
        sWarpV = storage.sWarpV.get_tensor(cute.make_layout(T // 32))
        sWarpP = storage.sWarpP.get_tensor(cute.make_layout(T // 32))
        sOutV = storage.sOutV.get_tensor(cute.make_layout(TOPK_PAD))
        sOutI = storage.sOutI.get_tensor(cute.make_layout(TOPK_PAD))

        # ── residual query sRQ (DP,) ───────────────────────────────────
        i = tid
        while i < DP:
            val = mQ[pid_q, i]
            if cutlass.const_expr(self.by_residual):
                val = val - mCent[c, i]
            sRQ[i] = val
            i += T
        cute.arch.sync_threads()

        # ── per-query offset ‖q-c_L‖² (added back to every score) ───────
        # The precomputed-table LUT drops this constant, so we re-add it
        # so partials are comparable across the query's probed lists.
        if tid == 0:
            acc = cutlass.Float32(0.0)
            d = 0
            while d < DP:
                v = sRQ[d]
                acc = acc + v * v
                d += 1
            sQRSQ[0] = acc
        cute.arch.sync_threads()

        # ── build the SMEM LUT from precomputed tables ─────────────────
        # sLUT[s,j] = cterm[list,s,j] - qterm[q,s,j]
        #          = (‖cb[s,j]‖²+2⟨c_L[s],cb[s,j]⟩) - 2⟨q[s],cb[s,j]⟩
        # so Σ_s sLUT[s,code_s] + ‖q-c_L‖² == ‖(q-c_L)-xhat‖² (ADC).
        i = tid
        while i < M * KSUB:
            j = i % KSUB
            s = i // KSUB
            val = mCT[c, s, j] - mQT[pid_q, s, j]
            sLUT[i] = val.to(self.lut_dtype)
            i += T
        cute.arch.sync_threads()

        # ── per-thread local top-k over strided candidates ─────────────
        lv = cute.make_rmem_tensor(cute.make_layout(TOPK_PAD), cutlass.Float32)
        li = cute.make_rmem_tensor(cute.make_layout(TOPK_PAD), cutlass.Int32)
        for t in cutlass.range_constexpr(TOPK_PAD):
            lv[t] = cutlass.Float32(_INF)
            li[t] = cutlass.Int32(-1)
        worst = cutlass.Float32(_INF)
        qrsq0 = sQRSQ[0]

        # Interleaved (group=32) codes -> coalesced loads. With T a multiple
        # of 32, a thread's lane = (r-c_start) % 32 is invariant, so a warp's
        # 32 lanes read codes_il[il_base + g*M32 + s*32 + lane] as one 32-byte
        # transaction instead of 32 stride-m scattered byte loads (the
        # measured L1TEX-scoreboard stall that dominated the high-D scan).
        M32 = M * 32
        il_base = cutlass.Int32(mIlOff[c])
        r = c_start + tid
        while r < c_end:
            p = r - c_start
            g = p // 32
            lane = p % 32
            cbase = il_base + g * M32 + lane
            a0 = cutlass.Float32(0.0)
            a1 = cutlass.Float32(0.0)
            a2 = cutlass.Float32(0.0)
            a3 = cutlass.Float32(0.0)
            s = 0
            while s < M4:
                c0 = cutlass.Int32(mCodesIl[cbase + s * 32])
                c1 = cutlass.Int32(mCodesIl[cbase + (s + 1) * 32])
                c2 = cutlass.Int32(mCodesIl[cbase + (s + 2) * 32])
                c3 = cutlass.Int32(mCodesIl[cbase + (s + 3) * 32])
                a0 = a0 + sLUT[s * KSUB + c0].to(cutlass.Float32)
                a1 = a1 + sLUT[(s + 1) * KSUB + c1].to(cutlass.Float32)
                a2 = a2 + sLUT[(s + 2) * KSUB + c2].to(cutlass.Float32)
                a3 = a3 + sLUT[(s + 3) * KSUB + c3].to(cutlass.Float32)
                s += 4
            while s < M:
                cr = cutlass.Int32(mCodesIl[cbase + s * 32])
                a0 = a0 + sLUT[s * KSUB + cr].to(cutlass.Float32)
                s += 1
            d = a0 + a1 + a2 + a3 + qrsq0
            if d < worst:
                pend_d = d
                pend_i = cutlass.Int32(r)
                for t in cutlass.range_constexpr(TOPK_PAD):
                    cur_d = lv[t]
                    cur_i = li[t]
                    take = cur_d > pend_d
                    lv[t] = pend_d if take else cur_d
                    li[t] = pend_i if take else cur_i
                    pend_d = cur_d if take else pend_d
                    pend_i = cur_i if take else pend_i
                worst = lv[TOPK_PAD - 1]
            r += T

        # ── publish local top-k to the SMEM merge pool ─────────────────
        for t in cutlass.range_constexpr(TOPK_PAD):
            sPoolV[tid * TOPK_PAD + t] = lv[t]
            sPoolI[tid * TOPK_PAD + t] = li[t]
        cute.arch.sync_threads()

        # ── iterative block arg-min: extract TOPK_PAD best from pool ───
        lane = tid % 32
        wid = tid // 32
        POOL = T * TOPK_PAD
        for it in cutlass.range_constexpr(TOPK_PAD):
            bv = cutlass.Float32(_INF)
            bp = cutlass.Int32(-1)
            j = tid
            while j < POOL:
                pv = sPoolV[j]
                sm = pv < bv
                bv = pv if sm else bv
                bp = cutlass.Int32(j) if sm else bp
                j += T
            off = 16
            while off > 0:
                ov = cute.arch.shuffle_sync_bfly(bv, off)
                op = cute.arch.shuffle_sync_bfly(bp, off)
                sm = ov < bv
                bv = ov if sm else bv
                bp = op if sm else bp
                off //= 2
            if lane == 0:
                sWarpV[wid] = bv
                sWarpP[wid] = bp
            cute.arch.sync_threads()
            if tid == 0:
                fv = sWarpV[0]
                fp = sWarpP[0]
                for w in cutlass.range_constexpr(1, T // 32):
                    wv = sWarpV[w]
                    sm = wv < fv
                    fv = wv if sm else fv
                    fp = sWarpP[w] if sm else fp
                sOutV[it] = fv
                if fp >= 0:
                    sOutI[it] = sPoolI[fp]
                    sPoolV[fp] = cutlass.Float32(_INF)
                else:
                    sOutI[it] = cutlass.Int32(-1)
            cute.arch.sync_threads()

        # ── write the (TOPK_PAD,) partial for this (query, probe) ──────
        if tid < TOPK_PAD:
            mPV[pid_q, pid_p, tid] = sOutV[tid]
            mPI[pid_q, pid_p, tid] = sOutI[tid]


_kernel_cache: dict = {}
_cterm_cache: dict = {}
_interleave_cache: dict = {}

_IL_GROUP = 32  # interleave group = warp width, so a warp's lanes coalesce


def _compute_qterm2(Qp: torch.Tensor, codebooks: torch.Tensor) -> torch.Tensor:
    """Per-query LUT cross term ``qterm2[q,s,j] = 2·⟨q[s], cb[s,j]⟩``.

    A batched (over sub-quantizers) GEMM ``(m, nq, dsub) @ (m, dsub, ksub)`` --
    the expensive ``dsub`` reduction runs once per query on tensor cores
    instead of once per (query, probe) in a SIMT loop. Returns ``(nq, m, ksub)``
    fp32.
    """
    m, ksub, dsub = codebooks.shape
    nq = Qp.shape[0]
    q3 = Qp.view(nq, m, dsub).permute(1, 0, 2).contiguous()      # (m, nq, dsub)
    cb_t = codebooks.permute(0, 2, 1).contiguous()               # (m, dsub, ksub)
    qcb = torch.bmm(q3, cb_t)                                     # (m, nq, ksub)
    return qcb.mul_(2.0).permute(1, 0, 2).contiguous()           # (nq, m, ksub)


def _get_cterm2(
    centroids: torch.Tensor, codebooks: torch.Tensor, nlist: int, by_residual: bool,
) -> torch.Tensor:
    """Per-index LUT term ``cterm2[L,s,j] = ‖cb[s,j]‖² + 2·⟨c_L[s], cb[s,j]⟩``
    (residual) or just ``‖cb[s,j]‖²`` (direct). Index-only, so cache it.

    Keyed by storage address but *validated* by weakref identity: a freed
    index whose address is later reused by a new (same-shape) index must not
    return the stale table, so we recompute unless the cached entry's centroid
    and codebook objects are still the exact tensors passed in. Returns
    ``(nlist, m, ksub)`` fp32."""
    key = (centroids.data_ptr(), codebooks.data_ptr(), nlist,
           int(codebooks.shape[0]), bool(by_residual))
    ent = _cterm_cache.get(key)
    if ent is not None:
        out, cent_ref, cb_ref = ent
        if cent_ref() is centroids and cb_ref() is codebooks:
            return out
    m, ksub, dsub = codebooks.shape
    base = (codebooks * codebooks).sum(-1)                        # (m, ksub)
    if by_residual:
        c3 = centroids.view(nlist, m, dsub).permute(1, 0, 2).contiguous()
        cb_t = codebooks.permute(0, 2, 1).contiguous()           # (m, dsub, ksub)
        ccb = torch.bmm(c3, cb_t)                                 # (m, nlist, ksub)
        out = (base[:, None, :] + ccb.mul_(2.0)).permute(1, 0, 2).contiguous()
    else:
        out = base[None].expand(nlist, m, ksub).contiguous()
    try:
        _cterm_cache[key] = (out, weakref.ref(centroids), weakref.ref(codebooks))
    except TypeError:
        pass
    return out


def _build_interleaved(codes: torch.Tensor, list_offsets: torch.Tensor, m: int):
    """Interleave the (cell-contiguous) PQ codes for warp-coalesced loads.

    Lays each inverted list out in groups of ``_IL_GROUP`` (=32) candidates:
    element ``(group g, sub-quantizer s, lane)`` of list ``L`` lives at
    ``il_offsets[L] + g*(m*32) + s*32 + lane`` in the flat ``codes_il``
    buffer (lists padded up to a multiple of 32; padding rows are never read
    because the scan stops at the true list length). A warp scanning 32
    consecutive candidates then reads one contiguous 32-byte run per ``s``.

    Returns ``(codes_il (uint8, 1-D), il_offsets (int32, nlist+1))`` in
    *elements*. Built once and cached (this is a one-time index transform).
    """
    G = _IL_GROUP
    device = codes.device
    M = codes.shape[0]
    nlist = list_offsets.shape[0] - 1
    lo = list_offsets.to(torch.int64)
    lengths = lo[1:] - lo[:-1]                              # (nlist,)
    num_groups = (lengths + (G - 1)) // G                   # (nlist,)
    group_off = torch.zeros(nlist + 1, dtype=torch.int64, device=device)
    group_off[1:] = num_groups.cumsum(0)
    total_groups = int(group_off[-1].item())                # one-time build sync
    il_offsets = (group_off * (m * G)).to(torch.int32)      # (nlist+1,) elements
    codes_il = torch.zeros(max(total_groups, 1) * m * G,
                           dtype=torch.uint8, device=device)
    if M > 0:
        rows = torch.arange(M, device=device, dtype=torch.int64)
        row_list = torch.searchsorted(lo, rows, right=True) - 1   # list of each row
        p = rows - lo[row_list]                             # local position in list
        dest_group = group_off[row_list] + (p // G)         # group index in buffer
        lane = p % G
        codes_il.view(total_groups, m, G)[dest_group, :, lane] = codes
    return codes_il, il_offsets


def _get_interleaved(codes: torch.Tensor, list_offsets: torch.Tensor, m: int):
    """Cached :func:`_build_interleaved` (weakref-validated, like cterm)."""
    key = (codes.data_ptr(), list_offsets.data_ptr(), m)
    ent = _interleave_cache.get(key)
    if ent is not None:
        codes_il, il_offsets, codes_ref, lo_ref = ent
        if codes_ref() is codes and lo_ref() is list_offsets:
            return codes_il, il_offsets
    codes_il, il_offsets = _build_interleaved(codes, list_offsets, m)
    try:
        _interleave_cache[key] = (codes_il, il_offsets,
                                  weakref.ref(codes), weakref.ref(list_offsets))
    except TypeError:
        pass
    return codes_il, il_offsets


def _to_cute(t: torch.Tensor):
    mt = cute_rt.from_dlpack(t, assumed_align=16)
    return mt.mark_layout_dynamic(leading_dim=t.ndim - 1)


_LUT_DTYPE = {"fp32": cutlass.Float32, "fp16": cutlass.Float16}


def _merge_topk(pv: torch.Tensor, pi: torch.Tensor, nq: int, k: int):
    """Merge the per-(query, probe) partials into the global per-query top-k.

    ``pv``/``pi`` are ``(nq, nprobe, TOPK_PAD)``; the fp32 LUT makes each
    partial ADC-exact, so a single ``topk`` over the flattened probe axis is
    the final answer (no oversampled re-rank). Returns ``(vals, pos)``.
    """
    pvf = pv.reshape(nq, -1)
    pif = pi.reshape(nq, -1)
    vals, sel = pvf.topk(k, dim=-1, largest=False, sorted=True)
    pos = pif.gather(-1, sel).to(torch.int64)
    pos = torch.where(vals.isinf(), torch.full_like(pos, -1), pos)
    return vals, pos


def ivf_pq_fine_scan_shared_lut(
    Qp: torch.Tensor,
    centroids: torch.Tensor,
    codebooks: torch.Tensor,
    codes: torch.Tensor,
    probed: torch.Tensor,
    list_offsets: torch.Tensor,
    k: int,
    *,
    by_residual: bool,
    threads: int = 128,
    over: int = 2,
    lut_dtype: str = "auto",
    rerank: Optional[bool] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Precomputed-table ADC-LUT fine scan (CuTe DSL), one CTA per
    ``(query, probe)``.

    The PQ codes are scanned from a warp-coalesced interleaved copy (groups
    of 32 candidates; built + cached once per index) so each warp reads one
    32-byte run per sub-quantizer instead of stride-``m`` byte gathers.

    ``lut_dtype="auto"`` (default) keeps the exact fp32 SMEM LUT while it
    fits (``m <= 32``) -- the per-probe top-k is then final and the host side
    is a single ``topk`` merge (no inverse map, scatter, or re-rank) -- and
    switches to a compact fp16 table (auto-enabling an exact-ADC ``rerank``)
    for larger ``m``, where the fp32 table would otherwise cap occupancy.
    Pass ``"fp32"`` / ``"fp16"`` to force one.

    Returns ``(vals, pos)`` -- ``vals`` ``(nq, k)`` ADC squared-L2 (fp32),
    ``pos`` ``(nq, k)`` int64 stored-row positions (``-1`` padded).
    """
    assert Qp.is_cuda and codes.is_cuda
    nq, Dp = Qp.shape
    nprobe = probed.shape[1]
    nlist = list_offsets.shape[0] - 1
    m = codes.shape[1]
    ksub, dsub = codebooks.shape[1], codebooks.shape[2]
    device = Qp.device
    if lut_dtype == "auto":
        # fp32 LUT is exact (no re-rank) but costs m*256*4 B of SMEM; past
        # ~32 KB (m>32) that caps occupancy, so switch to the compact fp16
        # table (+ exact re-rank) where it pays for itself.
        lut_dtype = "fp32" if m * ksub * 4 <= 32 * 1024 else "fp16"
    if rerank is None:
        rerank = (lut_dtype == "fp16")
    cute_lut_dtype = _LUT_DTYPE[lut_dtype]

    Qp = Qp.contiguous()
    centroids = centroids.contiguous()
    codebooks = codebooks.contiguous()
    codes = codes.contiguous()
    # Warp-coalesced interleaved copy of the codes for the scan (built once).
    codes_il, il_offsets = _get_interleaved(codes, list_offsets, m)

    # Precomputed-table LUT terms (FAISS-style): the dsub cross term is a
    # per-query GEMM (qterm2) + a cached per-index GEMM (cterm2), so the
    # device build is a cheap subtract instead of an m·256·dsub SIMT loop.
    qterm2 = _compute_qterm2(Qp, codebooks)                  # (nq, m, ksub) fp32
    cterm2 = _get_cterm2(centroids, codebooks, nlist, by_residual)

    probed_i32 = probed.contiguous().to(torch.int32)
    list_off_i32 = list_offsets.contiguous().to(torch.int32)

    TOPK_PAD = _next_pow2(k)
    pv = torch.full((nq, nprobe, TOPK_PAD), _INF, device=device, dtype=torch.float32)
    pi = torch.full((nq, nprobe, TOPK_PAD), -1, device=device, dtype=torch.int32)

    # Compile key omits nq / nprobe (launch-grid dims read from the dynamic
    # tensors) so one compiled kernel serves every batch size.
    key = (m, dsub, ksub, Dp, TOPK_PAD, bool(by_residual), threads, lut_dtype)
    compiled = _kernel_cache.get(key)
    stream = cuda.CUstream(torch.cuda.current_stream().cuda_stream)
    if compiled is None:
        kernel = HopperIvfPqSharedLut(
            M=m, DSUB=dsub, KSUB=ksub, DP=Dp, TOPK_PAD=TOPK_PAD,
            by_residual=by_residual, threads=threads, lut_dtype=cute_lut_dtype,
        )
        compiled = cute.compile(
            kernel,
            _to_cute(Qp), _to_cute(centroids), _to_cute(probed_i32),
            _to_cute(list_off_i32), _to_cute(codes_il), _to_cute(il_offsets),
            _to_cute(cterm2), _to_cute(qterm2), _to_cute(pv), _to_cute(pi),
            stream,
        )
        _kernel_cache[key] = compiled

    compiled(
        _to_cute(Qp), _to_cute(centroids), _to_cute(probed_i32),
        _to_cute(list_off_i32), _to_cute(codes_il), _to_cute(il_offsets),
        _to_cute(cterm2), _to_cute(qterm2), _to_cute(pv), _to_cute(pi),
        stream,
    )

    if not rerank:
        return _merge_topk(pv, pi, nq, k)

    # fp16 LUT: ranking is approximate, so oversample + exact-ADC re-rank.
    # perm=None -> partials already in natural (query, probe) order.
    return reduce_rerank(
        pv.reshape(nq * nprobe, TOPK_PAD), pi.reshape(nq * nprobe, TOPK_PAD),
        None, nq, nprobe, k,
        Qp=Qp, centroids=centroids, codebooks=codebooks, codes=codes,
        list_offsets=list_offsets, by_residual=by_residual, over=over,
    )


__all__ = ["ivf_pq_fine_scan_shared_lut", "HopperIvfPqSharedLut"]
