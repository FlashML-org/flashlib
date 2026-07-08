"""Micro-benchmark: stage split of the IVF-PQ decode-once (``ws``) fine scan.

Times each stage of :mod:`flashlib.primitives.ivf_pq.triton.fine_scan_ws`
in isolation -- host prep (sort + CSR), rqsq gather, workspace decode,
threshold sample pass, GEMM scan, reduce+rerank -- so a change to any one
kernel can be attributed. Pokes module-private kernels, so it tracks
their signatures.

Run:
    python -m benchmarks.micro.bench_ivfpq_ws_stages --m 32 --nprobe 32
    python -m benchmarks.micro.bench_ivfpq_ws_stages \
        --dataset gist-960-euclidean --m 60 --nprobe 32
"""
from __future__ import annotations

import argparse

import torch
import triton

from benchmarks.vs_cuml._common import load_ann_dataset, time_gpu


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="sift-128-euclidean")
    ap.add_argument("--m", type=int, default=32)
    ap.add_argument("--nprobe", type=int, default=32)
    ap.add_argument("--nlist", type=int, default=1024)
    ap.add_argument("--k", type=int, default=10)
    ap.add_argument("--nq", type=int, default=None)
    ap.add_argument("--BN", type=int, default=64)
    ap.add_argument("--BM", type=int, default=128)
    ap.add_argument("--num-stages", type=int, default=3)
    ap.add_argument("--num-warps", type=int, default=4)
    ap.add_argument("--no-thresh", action="store_true")
    args = ap.parse_args()

    from flashlib import flash_ivf_pq_build
    from flashlib.primitives.knn import flash_knn
    from flashlib.primitives.ivf_pq.torch_fallback import _pad_features
    from flashlib.primitives.ivf_pq.triton import fine_scan_ws as m_ws
    from flashlib.primitives.knn.triton._common import _next_pow2
    from flashlib.kernels.distance.triton.knn_gather_l2sq import (
        triton_knn_gather_sqdist,
    )
    from flashlib.primitives.ivf_pq.cutedsl.fine_scan_host import reduce_rerank

    train, test, _ = load_ann_dataset(args.dataset)
    if args.nq:
        test = test[: args.nq]
    Xc = torch.as_tensor(train, device="cuda")
    Xq = torch.as_tensor(test, device="cuda")
    nq = Xq.shape[0]
    nlist, nprobe, m, k = args.nlist, args.nprobe, args.m, args.k
    BN, BM = args.BN, args.BM

    index = flash_ivf_pq_build(Xc, nlist, m=m, nprobe=nprobe, niter=20, seed=0)
    Qp = _pad_features(Xq.to(torch.float32), index.Dp).contiguous()
    cent = index.centroids.to(torch.float32)
    cb = index.pq_codebooks.to(torch.float32)
    Dp = index.Dp
    dsub = index.dsub
    device = Qp.device

    probed = flash_knn(Qp.unsqueeze(0), cent.unsqueeze(0), nprobe,
                       return_distances=False)[0].to(torch.int32)
    list_offsets = index.list_offsets.contiguous().to(torch.int64)
    codes = index.codes.contiguous()
    cb_bf16 = cb.contiguous().to(torch.bfloat16)

    BM_pad = args.BM

    def prep():
        flat = probed.reshape(-1).contiguous().to(torch.int64)
        perm = torch.argsort(flat, stable=True)
        sorted_qid = (perm // nprobe).to(torch.int32)
        sorted_slot = (perm % nprobe).to(torch.int32)
        qcounts = torch.bincount(flat, minlength=nlist)
        q_offsets = torch.zeros(nlist + 1, dtype=torch.int64, device=device)
        q_offsets[1:] = qcounts.cumsum(0)
        lengths = list_offsets[1:] - list_offsets[:-1]
        ws_lens = torch.where(qcounts > 0,
                              ((lengths + BM_pad - 1) // BM_pad) * BM_pad,
                              torch.zeros_like(lengths))
        ws_offsets = torch.zeros(nlist + 1, dtype=torch.int64, device=device)
        ws_offsets[1:] = ws_lens.cumsum(0)
        stats = torch.stack([qcounts.max(), ws_offsets[-1]]).cpu()
        return (perm, sorted_qid, sorted_slot, q_offsets, ws_offsets,
                int(stats[0]), int(stats[1]))

    t_prep = time_gpu(prep, repeat=20, warmup=5)
    (perm, sorted_qid, sorted_slot, q_offsets, ws_offsets,
     max_qcount, ws_rows) = prep()
    MAX_QTILES = max(1, (max_qcount + BN - 1) // BN)
    print(f"ws_rows={ws_rows:,} / M={index.M:,}   MAX_QTILES={MAX_QTILES}")

    def rqsq_fn():
        cd = triton_knn_gather_sqdist(
            Qp.unsqueeze(0), cent.unsqueeze(0), probed.unsqueeze(0))[0]
        return cd.reshape(-1)[perm].contiguous()

    t_rqsq = time_gpu(rqsq_fn, repeat=20, warmup=5)
    rqsq_pair = rqsq_fn()

    ws = torch.empty((ws_rows, Dp), device=device, dtype=torch.bfloat16)
    w = torch.empty((ws_rows,), device=device, dtype=torch.float32)
    D_TILE = min(_next_pow2(Dp), 128)
    max_seg = triton.cdiv(index.max_list_len, BM_pad) * BM_pad

    def decode():
        m_ws._pq_decode_ws_kernel[(triton.cdiv(max_seg, 64), nlist)](
            codes, cb_bf16, cent,
            list_offsets, ws_offsets, ws, w,
            codes.stride(0), codes.stride(1),
            cb_bf16.stride(0), cb_bf16.stride(1), cb_bf16.stride(2),
            cent.stride(0), cent.stride(1),
            ws.stride(0), ws.stride(1),
            BY_RESIDUAL=True, DSUB=dsub, DP=Dp, D_TILE=D_TILE, BR=64,
            num_warps=4,
        )

    t_dec = time_gpu(decode, repeat=20, warmup=5)

    TOPK_PAD = _next_pow2(k)
    D_INNER = _next_pow2(Dp) if Dp <= 256 else 128
    P = nq * nprobe

    samp = torch.full((P,), float("inf"), device=device, dtype=torch.float32)

    def sample():
        m_ws._ws_sample_floor_kernel[(MAX_QTILES, nlist)](
            Qp, sorted_qid, rqsq_pair, ws, w,
            q_offsets, ws_offsets, samp,
            Qp.stride(0), Qp.stride(1), WS_RD=Dp,
            DP=Dp, D_INNER=D_INNER, BN=BN, BM=BM,
            num_warps=args.num_warps, num_stages=args.num_stages,
        )

    def make_thr():
        samp_nat = torch.empty_like(samp)
        samp_nat[perm] = samp
        kk = min(k, nprobe)
        T = samp_nat.view(nq, nprobe).topk(kk, dim=1, largest=False).values[:, -1]
        return (T + T.abs() * 2.0 ** -8).contiguous()

    use_thr = not args.no_thresh
    if use_thr:
        t_samp = time_gpu(sample, repeat=20, warmup=5)
        sample()
        t_thr = time_gpu(make_thr, repeat=20, warmup=5)
        thr_q = make_thr()
    else:
        t_samp = t_thr = 0.0
        thr_q = torch.empty((1,), device=device, dtype=torch.float32)

    pv = torch.full((P, TOPK_PAD), float("inf"), device=device, dtype=torch.float32)
    pi = torch.full((P, TOPK_PAD), -1, device=device, dtype=torch.int32)

    def gemm():
        m_ws._ivf_pq_ws_gemm_kernel[(MAX_QTILES, nlist)](
            Qp, sorted_qid, sorted_slot, rqsq_pair, ws, w, thr_q,
            q_offsets, ws_offsets, list_offsets, pv, pi, nprobe,
            max(ws_rows, 1),
            Qp.stride(0), Qp.stride(1),
            WS_RD=Dp,
            USE_THR=use_thr,
            DP=Dp, D_INNER=D_INNER, BN=BN, BM=BM,
            TOPK_PAD=TOPK_PAD, MAX_STEPS=min(k, BM),
            num_warps=args.num_warps, num_stages=args.num_stages,
        )

    t_gemm = time_gpu(gemm, repeat=20, warmup=5)

    def reduce():
        return reduce_rerank(
            pv, pi, None, nq, nprobe, k,
            Qp=Qp, centroids=cent, codebooks=cb, codes=codes,
            list_offsets=list_offsets, by_residual=True, over=4)

    t_red = time_gpu(reduce, repeat=20, warmup=5)

    tot = t_prep + t_rqsq + t_dec + t_samp + t_thr + t_gemm + t_red
    print(f"prep(argsort etc): {t_prep:7.3f} ms")
    print(f"rqsq gather      : {t_rqsq:7.3f} ms")
    print(f"decode ws        : {t_dec:7.3f} ms")
    print(f"sample floor     : {t_samp:7.3f} ms")
    print(f"make thr         : {t_thr:7.3f} ms")
    print(f"gemm scan        : {t_gemm:7.3f} ms")
    print(f"reduce+rerank    : {t_red:7.3f} ms")
    print(f"sum              : {tot:7.3f} ms")


if __name__ == "__main__":
    main()
