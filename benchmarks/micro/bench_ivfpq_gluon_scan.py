"""Micro-benchmark: Gluon (Hopper) prototype of the IVF-PQ ws-scan kernel.

Re-implements the decode-once GEMM scan
(:mod:`flashlib.primitives.ivf_pq.triton.fine_scan_ws`) in Triton's Gluon
dialect with explicit control: NVMMA shared layouts, cp.async double
buffering, and (in the ``pipe`` variant) async WGMMA issued one chunk
ahead so the top-k epilogue overlaps the tensor cores.

Measured finding (H100, SIFT-1M m=32): both Gluon versions land within
noise of the Triton kernel (~1.2 ms) -- the per-chunk top-k epilogue, not
tensor-core feeding, is the critical path, and it cannot overlap with
itself. Kept as the reference point for that conclusion and as a Gluon
idiom example; the integrated kernel stays the portable Triton one.

Run:
    python -m benchmarks.micro.bench_ivfpq_gluon_scan --m 32 --nprobe 32
"""
from __future__ import annotations

import argparse

import torch
import triton
from triton.experimental import gluon
from triton.experimental.gluon import language as gl
from triton.experimental.gluon.language.nvidia import hopper
from triton.experimental.gluon.language.nvidia.hopper import async_copy, fence_async_shared

from benchmarks.vs_cuml._common import load_ann_dataset, time_gpu
from flashlib.primitives.knn.triton._common import _next_pow2


@gluon.jit
def _gluon_ws_scan(
    q_ptr, sorted_qid_ptr, sorted_slot_ptr, rqsq_ptr,
    ws_ptr, w_ptr, thr_ptr,
    q_offsets_ptr, ws_off_ptr, list_off_ptr,
    pv_ptr, pi_ptr,
    nprobe,
    DP: gl.constexpr, BN: gl.constexpr, BM: gl.constexpr,
    TOPK_PAD: gl.constexpr, MAX_STEPS: gl.constexpr,
):
    pid_qt = gl.program_id(0)
    pid_c = gl.program_id(1)
    qstart = gl.load(q_offsets_ptr + pid_c)
    qend = gl.load(q_offsets_ptr + pid_c + 1)
    qcount = qend - qstart
    if pid_qt * BN >= qcount:
        return

    # ---- layouts -----------------------------------------------------
    blk: gl.constexpr = gl.BlockedLayout([1, 8], [2, 16], [4, 1], [1, 0])
    mma: gl.constexpr = gl.NVMMADistributedLayout(
        version=[3, 0], warps_per_cta=[4, 1], instr_shape=[16, BM, 16])
    row1d: gl.constexpr = gl.SliceLayout(1, mma)     # (BN,) rows of acc
    col1d: gl.constexpr = gl.SliceLayout(0, mma)     # (BM,) cols of acc
    smem_l: gl.constexpr = gl.NVMMASharedLayout(
        swizzle_byte_width=128, element_bitwidth=16, rank=2)

    i_range = gl.arange(0, BN, layout=row1d)
    q_local = pid_qt * BN + i_range
    q_mask = q_local < qcount
    pair_pos = (qstart + q_local).to(gl.int64)
    qid = gl.load(sorted_qid_ptr + pair_pos, mask=q_mask, other=0).to(gl.int64)
    slot = gl.load(sorted_slot_ptr + pair_pos, mask=q_mask, other=0).to(gl.int64)
    rqsq = gl.load(rqsq_ptr + pair_pos, mask=q_mask, other=0.0)
    thr = gl.load(thr_ptr + qid, mask=q_mask, other=float("inf"))
    ws_start = gl.load(ws_off_ptr + pid_c)
    ws_end = gl.load(ws_off_ptr + pid_c + 1)
    c_start = gl.load(list_off_ptr + pid_c).to(gl.int32)

    # ---- stage the query tile (A operand) into SMEM ------------------
    qid_b = gl.convert_layout(qid, gl.SliceLayout(1, blk))
    qm_b = gl.convert_layout(q_mask, gl.SliceLayout(1, blk))
    d_off = gl.arange(0, DP, layout=gl.SliceLayout(0, blk))
    x_tile = gl.load(
        q_ptr + qid_b[:, None] * DP + d_off[None, :],
        mask=qm_b[:, None], other=0.0,
    ).to(gl.bfloat16)                                   # (BN, DP) blocked
    a_smem = gl.allocate_shared_memory(gl.bfloat16, [BN, DP], smem_l, x_tile)

    # ---- double-buffered B operand (ws chunk) -------------------------
    b_smem = gl.allocate_shared_memory(gl.bfloat16, [2, BM, DP], smem_l)
    bm_b = gl.arange(0, BM, layout=gl.SliceLayout(1, blk))

    n_chunks = ((ws_end - ws_start) // BM).to(gl.int32)

    # prefetch chunk 0 into buffer 0
    rows0 = (ws_start + bm_b).to(gl.int64)
    async_copy.async_copy_global_to_shared(
        b_smem.index(0), ws_ptr + rows0[:, None] * DP + d_off[None, :])
    async_copy.commit_group()

    I32_MAX: gl.constexpr = 2147483647
    MASK_HI: gl.constexpr = -65536
    MASK_LO: gl.constexpr = 65535
    kr_l: gl.constexpr = gl.BlockedLayout([1, TOPK_PAD], [32, 1], [4, 1], [1, 0])
    k_range = gl.arange(0, TOPK_PAD, layout=gl.SliceLayout(0, kr_l))
    topk_vals = gl.full([BN, TOPK_PAD], float("inf"), gl.float32, layout=kr_l)
    topk_idxs = gl.full([BN, TOPK_PAD], -1, gl.int32, layout=kr_l)
    topk_max = gl.full([BN], float("inf"), gl.float32,
                       layout=gl.SliceLayout(1, kr_l))
    thr_k = gl.convert_layout(thr, gl.SliceLayout(1, kr_l))
    rqsq_m = gl.convert_layout(rqsq, row1d)
    qmask_m = gl.convert_layout(q_mask, row1d)
    bm_cols = gl.arange(0, BM, layout=col1d)
    # insert bar maintained in BOTH layouts: the common (no-insert) path
    # tests in the acc's row layout with no cross-layout conversion; the
    # bar is re-converted only after an actual insert (rare).
    bar_m = gl.convert_layout(gl.minimum(topk_max, thr_k), row1d)

    for ci in range(n_chunks):
        cur = ci % 2
        # prefetch next chunk into the other buffer
        if ci + 1 < n_chunks:
            rows_n = (ws_start + (ci + 1) * BM + bm_b).to(gl.int64)
            async_copy.async_copy_global_to_shared(
                b_smem.index(1 - cur),
                ws_ptr + rows_n[:, None] * DP + d_off[None, :])
            async_copy.commit_group()
            async_copy.wait_group(1)     # current buffer's copy done
        else:
            async_copy.wait_group(0)
        fence_async_shared()

        acc0 = gl.zeros([BN, BM], gl.float32, layout=mma)
        acc = hopper.warpgroup_mma(
            a_smem, b_smem.index(cur).permute((1, 0)), acc0, is_async=True)
        acc = hopper.warpgroup_mma_wait(num_outstanding=0, deps=[acc])

        wv = gl.load(w_ptr + (ws_start + ci * BM + bm_cols).to(gl.int64))
        dist = rqsq_m[:, None] + wv[None, :] - 2.0 * acc
        dist = gl.maximum(dist, 0.0)
        packed = (dist.to(gl.int32, bitcast=True) & MASK_HI) | bm_cols[None, :]
        packed = gl.where(qmask_m[:, None], packed, I32_MAX)

        rm_m = gl.min(packed, axis=1)                     # (BN,) row1d layout
        need_m = (rm_m & MASK_HI) < (bar_m.to(gl.int32, bitcast=True) & MASK_HI)
        if gl.max(need_m.to(gl.int32), axis=0) > 0:
            rm = gl.convert_layout(rm_m, gl.SliceLayout(1, kr_l))
            bar = gl.minimum(topk_max, thr_k)
            need = (rm & MASK_HI) < (bar.to(gl.int32, bitcast=True) & MASK_HI)
            for _step in range(MAX_STEPS):
                if gl.max(need.to(gl.int32), axis=0) > 0:
                    val = (rm & MASK_HI).to(gl.float32, bitcast=True)
                    col = rm & MASK_LO
                    # replace the worst slot: pack (val, slot) so one max
                    # reduction resolves the slot to overwrite
                    pk = (topk_vals.to(gl.int32, bitcast=True) & MASK_HI) \
                        | k_range[None, :]
                    worst = gl.max(pk, axis=1)
                    repl = (k_range[None, :] == (worst & MASK_LO)[:, None]) \
                        & need[:, None]
                    topk_vals = gl.where(repl, val[:, None], topk_vals)
                    cand = (ci * BM + col).to(gl.int32)
                    topk_idxs = gl.where(repl, cand[:, None], topk_idxs)
                    topk_max = gl.max(topk_vals, axis=1)
                    # pop from packed (needs rm in mma-row layout)
                    rm_pop = gl.convert_layout(rm, row1d)
                    packed = gl.where(packed == rm_pop[:, None], I32_MAX, packed)
                    rm = gl.convert_layout(gl.min(packed, axis=1),
                                           gl.SliceLayout(1, kr_l))
                    bar = gl.minimum(topk_max, thr_k)
                    need = (rm & MASK_HI) < (
                        bar.to(gl.int32, bitcast=True) & MASK_HI)
            bar_m = gl.convert_layout(gl.minimum(topk_max, thr_k), row1d)

    out_pos = qid * nprobe + slot
    qm_k = gl.convert_layout(q_mask, gl.SliceLayout(1, kr_l))
    out_pos_k = gl.convert_layout(out_pos, gl.SliceLayout(1, kr_l))
    valid_idx = topk_idxs >= 0
    glob = gl.where(valid_idx, topk_idxs + c_start, -1).to(gl.int32)
    gl.store(pv_ptr + (out_pos_k[:, None] * TOPK_PAD + k_range[None, :]),
             topk_vals, mask=qm_k[:, None])
    gl.store(pi_ptr + (out_pos_k[:, None] * TOPK_PAD + k_range[None, :]),
             glob, mask=qm_k[:, None])


@gluon.jit
def _epilogue(
    acc, ci, rqsq_m, qmask_m, bm_cols, w_ptr, wbase,
    topk_vals, topk_idxs, topk_max, thr_k, k_range,
    BM: gl.constexpr, TOPK_PAD: gl.constexpr, MAX_STEPS: gl.constexpr,
    mma: gl.constexpr, kr_l: gl.constexpr, row1d: gl.constexpr,
):
    I32_MAX: gl.constexpr = 2147483647
    MASK_HI: gl.constexpr = -65536
    MASK_LO: gl.constexpr = 65535
    wv = gl.load(w_ptr + (wbase + bm_cols).to(gl.int64))
    dist = rqsq_m[:, None] + wv[None, :] - 2.0 * acc
    dist = gl.maximum(dist, 0.0)
    packed = (dist.to(gl.int32, bitcast=True) & MASK_HI) | bm_cols[None, :]
    packed = gl.where(qmask_m[:, None], packed, I32_MAX)

    rm_m = gl.min(packed, axis=1)
    rm = gl.convert_layout(rm_m, gl.SliceLayout(1, kr_l))
    bar = gl.minimum(topk_max, thr_k)
    need = (rm & MASK_HI) < (bar.to(gl.int32, bitcast=True) & MASK_HI)
    if gl.max(need.to(gl.int32), axis=0) > 0:
        for _step in range(MAX_STEPS):
            if gl.max(need.to(gl.int32), axis=0) > 0:
                val = (rm & MASK_HI).to(gl.float32, bitcast=True)
                col = rm & MASK_LO
                pk = (topk_vals.to(gl.int32, bitcast=True) & MASK_HI) \
                    | k_range[None, :]
                worst = gl.max(pk, axis=1)
                repl = (k_range[None, :] == (worst & MASK_LO)[:, None]) \
                    & need[:, None]
                topk_vals = gl.where(repl, val[:, None], topk_vals)
                cand = (ci * BM + col).to(gl.int32)
                topk_idxs = gl.where(repl, cand[:, None], topk_idxs)
                topk_max = gl.max(topk_vals, axis=1)
                rm_pop = gl.convert_layout(rm, row1d)
                packed = gl.where(packed == rm_pop[:, None], I32_MAX, packed)
                rm = gl.convert_layout(gl.min(packed, axis=1),
                                       gl.SliceLayout(1, kr_l))
                bar = gl.minimum(topk_max, thr_k)
                need = (rm & MASK_HI) < (
                    bar.to(gl.int32, bitcast=True) & MASK_HI)
    return topk_vals, topk_idxs, topk_max


@gluon.jit
def _gluon_ws_scan_pipe(
    q_ptr, sorted_qid_ptr, sorted_slot_ptr, rqsq_ptr,
    ws_ptr, w_ptr, thr_ptr,
    q_offsets_ptr, ws_off_ptr, list_off_ptr,
    pv_ptr, pi_ptr,
    nprobe,
    DP: gl.constexpr, BN: gl.constexpr, BM: gl.constexpr,
    TOPK_PAD: gl.constexpr, MAX_STEPS: gl.constexpr,
    NBUF: gl.constexpr,
):
    """Software-pipelined: wgmma(ci+1) issued async BEFORE the epilogue of
    chunk ci runs, so the CUDA-core top-k overlaps the tensor-core GEMM."""
    pid_qt = gl.program_id(0)
    pid_c = gl.program_id(1)
    qstart = gl.load(q_offsets_ptr + pid_c)
    qend = gl.load(q_offsets_ptr + pid_c + 1)
    qcount = qend - qstart
    if pid_qt * BN >= qcount:
        return

    blk: gl.constexpr = gl.BlockedLayout([1, 8], [2, 16], [4, 1], [1, 0])
    mma: gl.constexpr = gl.NVMMADistributedLayout(
        version=[3, 0], warps_per_cta=[4, 1], instr_shape=[16, BM, 16])
    row1d: gl.constexpr = gl.SliceLayout(1, mma)
    col1d: gl.constexpr = gl.SliceLayout(0, mma)
    smem_l: gl.constexpr = gl.NVMMASharedLayout(
        swizzle_byte_width=128, element_bitwidth=16, rank=2)
    kr_l: gl.constexpr = gl.BlockedLayout([1, TOPK_PAD], [32, 1], [4, 1], [1, 0])

    i_range = gl.arange(0, BN, layout=row1d)
    q_local = pid_qt * BN + i_range
    q_mask = q_local < qcount
    pair_pos = (qstart + q_local).to(gl.int64)
    qid = gl.load(sorted_qid_ptr + pair_pos, mask=q_mask, other=0).to(gl.int64)
    slot = gl.load(sorted_slot_ptr + pair_pos, mask=q_mask, other=0).to(gl.int64)
    rqsq_m = gl.load(rqsq_ptr + pair_pos, mask=q_mask, other=0.0)
    thr = gl.load(thr_ptr + qid, mask=q_mask, other=float("inf"))
    ws_start = gl.load(ws_off_ptr + pid_c)
    ws_end = gl.load(ws_off_ptr + pid_c + 1)
    c_start = gl.load(list_off_ptr + pid_c).to(gl.int32)

    qid_b = gl.convert_layout(qid, gl.SliceLayout(1, blk))
    qm_b = gl.convert_layout(q_mask, gl.SliceLayout(1, blk))
    d_off = gl.arange(0, DP, layout=gl.SliceLayout(0, blk))
    x_tile = gl.load(
        q_ptr + qid_b[:, None] * DP + d_off[None, :],
        mask=qm_b[:, None], other=0.0,
    ).to(gl.bfloat16)
    a_smem = gl.allocate_shared_memory(gl.bfloat16, [BN, DP], smem_l, x_tile)

    b_smem = gl.allocate_shared_memory(gl.bfloat16, [NBUF, BM, DP], smem_l)
    bm_b = gl.arange(0, BM, layout=gl.SliceLayout(1, blk))

    n_chunks = ((ws_end - ws_start) // BM).to(gl.int32)

    # Prologue: prefetch chunks 0..NBUF-1. Reads past the segment are
    # allowed (the ws allocation carries NBUF*BM slack rows) and their
    # chunks are never consumed; issuing exactly one commit group per slot
    # keeps every wait_group count STATIC (a hardware requirement).
    for pf in gl.static_range(NBUF):
        rows = (ws_start + pf * BM + bm_b).to(gl.int64)
        async_copy.async_copy_global_to_shared(
            b_smem.index(pf), ws_ptr + rows[:, None] * DP + d_off[None, :])
        async_copy.commit_group()

    k_range = gl.arange(0, TOPK_PAD, layout=gl.SliceLayout(0, kr_l))
    topk_vals = gl.full([BN, TOPK_PAD], float("inf"), gl.float32, layout=kr_l)
    topk_idxs = gl.full([BN, TOPK_PAD], -1, gl.int32, layout=kr_l)
    topk_max = gl.full([BN], float("inf"), gl.float32,
                       layout=gl.SliceLayout(1, kr_l))
    thr_k = gl.convert_layout(thr, gl.SliceLayout(1, kr_l))
    qmask_m = gl.convert_layout(q_mask, row1d)
    bm_cols = gl.arange(0, BM, layout=col1d)

    # issue wgmma for chunk 0 (copy 0 done once <= NBUF-1 groups remain)
    acc0 = gl.zeros([BN, BM], gl.float32, layout=mma)
    async_copy.wait_group(NBUF - 1)
    fence_async_shared()
    acc_cur = hopper.warpgroup_mma(
        a_smem, b_smem.index(0).permute((1, 0)), acc0, is_async=True)

    for ci in range(n_chunks):
        nxt = ci + 1
        # issue the NEXT wgmma before consuming the current acc
        if nxt < n_chunks:
            # groups issued: NBUF + ci; <= NBUF-2 outstanding => copy(ci+1)
            # complete (copies land in issue order)
            async_copy.wait_group(NBUF - 2)
            fence_async_shared()
            acc_nxt = hopper.warpgroup_mma(
                a_smem, b_smem.index(nxt % NBUF).permute((1, 0)), acc0,
                is_async=True)
            # wait for wgmma(ci) only; wgmma(ci+1) keeps the TCs busy
            acc = hopper.warpgroup_mma_wait(num_outstanding=1, deps=[acc_cur])
        else:
            acc_nxt = acc_cur
            acc = hopper.warpgroup_mma_wait(num_outstanding=0, deps=[acc_cur])

        # wgmma(ci) has drained -> buffer ci%NBUF is free; refill it with
        # chunk ci+NBUF (past-the-end reads hit the slack rows and are
        # never consumed; the unconditional commit keeps the group count
        # at NBUF + ci + 1 so the static waits stay valid)
        rf = ci + NBUF
        rows = (ws_start + gl.minimum(rf * BM, ws_end - ws_start) + bm_b).to(gl.int64)
        async_copy.async_copy_global_to_shared(
            b_smem.index(rf % NBUF),
            ws_ptr + rows[:, None] * DP + d_off[None, :])
        async_copy.commit_group()

        topk_vals, topk_idxs, topk_max = _epilogue(
            acc, ci, rqsq_m, qmask_m, bm_cols, w_ptr, ws_start + ci * BM,
            topk_vals, topk_idxs, topk_max, thr_k, k_range,
            BM, TOPK_PAD, MAX_STEPS, mma, kr_l, row1d,
        )
        acc_cur = acc_nxt

    out_pos = qid * nprobe + slot
    qm_k = gl.convert_layout(q_mask, gl.SliceLayout(1, kr_l))
    out_pos_k = gl.convert_layout(out_pos, gl.SliceLayout(1, kr_l))
    valid_idx = topk_idxs >= 0
    glob = gl.where(valid_idx, topk_idxs + c_start, -1).to(gl.int32)
    gl.store(pv_ptr + (out_pos_k[:, None] * TOPK_PAD + k_range[None, :]),
             topk_vals, mask=qm_k[:, None])
    gl.store(pi_ptr + (out_pos_k[:, None] * TOPK_PAD + k_range[None, :]),
             glob, mask=qm_k[:, None])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--m", type=int, default=32)
    ap.add_argument("--nprobe", type=int, default=32)
    ap.add_argument("--nlist", type=int, default=1024)
    ap.add_argument("--k", type=int, default=10)
    args = ap.parse_args()

    from flashlib import flash_ivf_pq_build
    from flashlib.primitives.knn import flash_knn
    from flashlib.primitives.ivf_pq.torch_fallback import _pad_features
    from flashlib.primitives.ivf_pq.triton import fine_scan_ws as m_ws
    from flashlib.kernels.distance.triton.knn_gather_l2sq import (
        triton_knn_gather_sqdist,
    )

    train, test, _ = load_ann_dataset("sift-128-euclidean")
    Xc = torch.as_tensor(train, device="cuda")
    Xq = torch.as_tensor(test, device="cuda")
    nq = Xq.shape[0]
    nlist, nprobe, m, k = args.nlist, args.nprobe, args.m, args.k
    BN, BM = 64, 128

    index = flash_ivf_pq_build(Xc, nlist, m=m, nprobe=nprobe, niter=20, seed=0)
    Qp = _pad_features(Xq.to(torch.float32), index.Dp).contiguous()
    cent = index.centroids.to(torch.float32)
    cb = index.pq_codebooks.to(torch.float32)
    Dp = index.Dp
    device = Qp.device

    probed = flash_knn(Qp.unsqueeze(0), cent.unsqueeze(0), nprobe,
                       return_distances=False)[0].to(torch.int32)
    list_offsets = index.list_offsets.contiguous().to(torch.int64)
    codes = index.codes.contiguous()
    cb_bf16 = cb.contiguous().to(torch.bfloat16)

    flat = probed.reshape(-1).contiguous().to(torch.int64)
    perm = torch.argsort(flat, stable=True)
    sorted_qid = (perm // nprobe).to(torch.int32)
    sorted_slot = (perm % nprobe).to(torch.int32)
    qcounts = torch.bincount(flat, minlength=nlist)
    q_offsets = torch.zeros(nlist + 1, dtype=torch.int64, device=device)
    q_offsets[1:] = qcounts.cumsum(0)
    lengths = list_offsets[1:] - list_offsets[:-1]
    ws_lens = torch.where(qcounts > 0, ((lengths + BM - 1) // BM) * BM,
                          torch.zeros_like(lengths))
    ws_offsets = torch.zeros(nlist + 1, dtype=torch.int64, device=device)
    ws_offsets[1:] = ws_lens.cumsum(0)
    stats = torch.stack([qcounts.max(), ws_offsets[-1]]).cpu()
    max_qcount, ws_rows = int(stats[0]), int(stats[1])
    MAX_QTILES = max(1, (max_qcount + BN - 1) // BN)

    cdists = triton_knn_gather_sqdist(
        Qp.unsqueeze(0), cent.unsqueeze(0), probed.unsqueeze(0))[0]
    rqsq_pair = cdists.reshape(-1)[perm].contiguous()
    SLACK = 4 * BM   # prefetch-past-the-end slack for the pipelined kernel
    ws = torch.empty((ws_rows + SLACK, Dp), device=device, dtype=torch.bfloat16)
    w = torch.empty((ws_rows,), device=device, dtype=torch.float32)
    max_seg = triton.cdiv(index.max_list_len, BM) * BM
    m_ws._pq_decode_ws_kernel[(triton.cdiv(max_seg, 64), nlist)](
        codes, cb_bf16, cent, list_offsets, ws_offsets, ws, w,
        codes.stride(0), codes.stride(1),
        cb_bf16.stride(0), cb_bf16.stride(1), cb_bf16.stride(2),
        cent.stride(0), cent.stride(1), ws.stride(0), ws.stride(1),
        BY_RESIDUAL=True, DSUB=index.dsub, DP=Dp, D_TILE=128, BR=64,
        num_warps=4)

    TOPK_PAD = _next_pow2(k)
    P = nq * nprobe
    D_INNER = _next_pow2(Dp)

    samp = torch.full((P,), float("inf"), device=device, dtype=torch.float32)
    m_ws._ws_sample_floor_kernel[(MAX_QTILES, nlist)](
        Qp, sorted_qid, rqsq_pair, ws, w, q_offsets, ws_offsets, samp,
        Qp.stride(0), Qp.stride(1), WS_RD=Dp,
        DP=Dp, D_INNER=D_INNER, BN=BN, BM=BM, num_warps=4, num_stages=3)
    samp_nat = torch.empty_like(samp)
    samp_nat[perm] = samp
    T = samp_nat.view(nq, nprobe).topk(min(k, nprobe), dim=1,
                                       largest=False).values[:, -1]
    thr_q = (T + T.abs() * 2.0 ** -8).contiguous()

    pv = torch.full((P, TOPK_PAD), float("inf"), device=device, dtype=torch.float32)
    pi = torch.full((P, TOPK_PAD), -1, device=device, dtype=torch.int32)
    pv_ref = torch.full_like(pv, float("inf"))
    pi_ref = torch.full_like(pi, -1)

    def run_gluon():
        _gluon_ws_scan[(MAX_QTILES, nlist)](
            Qp, sorted_qid, sorted_slot, rqsq_pair, ws, w, thr_q,
            q_offsets, ws_offsets, list_offsets, pv, pi, nprobe,
            DP=Dp, BN=BN, BM=BM, TOPK_PAD=TOPK_PAD, MAX_STEPS=min(k, BM),
            num_warps=4)

    def run_triton():
        m_ws._ivf_pq_ws_gemm_kernel[(MAX_QTILES, nlist)](
            Qp, sorted_qid, sorted_slot, rqsq_pair, ws, w, thr_q,
            q_offsets, ws_offsets, list_offsets, pv_ref, pi_ref, nprobe,
            ws_rows,
            Qp.stride(0), Qp.stride(1), WS_RD=Dp,
            USE_THR=True, DP=Dp, D_INNER=D_INNER, BN=BN, BM=BM,
            TOPK_PAD=TOPK_PAD, MAX_STEPS=min(k, BM),
            num_warps=4, num_stages=3)

    def run_gluon_pipe(nbuf):
        _gluon_ws_scan_pipe[(MAX_QTILES, nlist)](
            Qp, sorted_qid, sorted_slot, rqsq_pair, ws, w, thr_q,
            q_offsets, ws_offsets, list_offsets, pv, pi, nprobe,
            DP=Dp, BN=BN, BM=BM, TOPK_PAD=TOPK_PAD, MAX_STEPS=min(k, BM),
            NBUF=nbuf, num_warps=4)

    run_gluon()
    run_triton()
    ok_v = torch.allclose(pv.sort(1).values, pv_ref.sort(1).values,
                          rtol=1e-5, atol=1e-2)
    same = (pi.sort(1).values == pi_ref.sort(1).values).float().mean().item()
    t_g = time_gpu(run_gluon, repeat=20, warmup=5)
    t_t = time_gpu(run_triton, repeat=20, warmup=5)
    print(f"gluon      : {t_g:7.3f} ms   vals_ok={ok_v} id_match={same:.4f}")
    print(f"triton     : {t_t:7.3f} ms")

    # Finding (H100, SIFT-1M m=32): the manual wgmma/copy pipeline does NOT
    # beat the plain version (or the Triton kernel) -- the per-chunk top-k
    # epilogue, not tensor-core feeding, is the critical path, and it cannot
    # overlap with itself. NBUF > 3 additionally showed nondeterministic
    # value drift (unchased -- the pipeline is not integrated).
    for nbuf in (2, 3):
        try:
            pv.fill_(float("inf")); pi.fill_(-1)
            run_gluon_pipe(nbuf)
            ok_v = torch.allclose(pv.sort(1).values, pv_ref.sort(1).values,
                                  rtol=1e-5, atol=1e-2)
            same = (pi.sort(1).values == pi_ref.sort(1).values).float().mean().item()
            t_p = time_gpu(lambda: run_gluon_pipe(nbuf), repeat=20, warmup=5)
            print(f"gluon pipe{nbuf}: {t_p:7.3f} ms   vals_ok={ok_v} id_match={same:.4f}")
        except Exception:
            import traceback
            traceback.print_exc()
            print(f"gluon pipe{nbuf}: ERR")
            break


if __name__ == "__main__":
    main()
