"""IVF-PQ fine-scan variants vs cuVS on real + synthetic data.

Finer-grained companion to :mod:`benchmarks.vs_cuml.ivf_pq`: per
(dataset, nlist, nprobe, m) it times **each fine-scan variant** (``ws`` /
``cute_gemm`` / ``cute_lut`` / ``gemm`` / ``online`` / ``auto``) on the
same built index (identical candidate sets), plus cuVS IVF-PQ as the
reference, and splits out the coarse stage -- the tool used to calibrate
the ``auto`` routing crossovers in
:mod:`flashlib.primitives.ivf_pq.triton.search`.

Run:
    python -m benchmarks.vs_cuml.ivf_pq_variants --dataset sift-128-euclidean
    python -m benchmarks.vs_cuml.ivf_pq_variants --dataset gist-960-euclidean \
        --configs "1024,32,60,10;1024,32,120,10"
    python -m benchmarks.vs_cuml.ivf_pq_variants --variants ws,cute_lut \
        --configs "1024,32,32,10,256"          # trailing value = nq cap
"""
from __future__ import annotations

import argparse

import numpy as np
import torch

from benchmarks.vs_cuml._common import (
    load_ann_dataset,
    recall_at_k,
    time_gpu,
    title,
)


def _blobs(M, D, n_centers, seed):
    rng = np.random.RandomState(seed)
    centers = rng.randn(n_centers, D).astype(np.float32) * 4.0
    lab = rng.randint(0, n_centers, size=M)
    return (centers[lab] + rng.randn(M, D).astype(np.float32)).astype(np.float32)


def bench_variants(Xc_t, Xq_t, gt, nlist, nprobe, m, k, variants, repeat=20):
    from flashlib import flash_ivf_pq_build, flash_ivf_pq_search

    index = flash_ivf_pq_build(Xc_t, nlist, m=m, nprobe=nprobe, niter=20, seed=0)
    rows = []
    for v in variants:
        try:
            ids = flash_ivf_pq_search(index, Xq_t, k, nprobe=nprobe, variant=v)[1]
            t = time_gpu(
                lambda: flash_ivf_pq_search(index, Xq_t, k, nprobe=nprobe, variant=v),
                repeat=repeat, warmup=5,
            )
            r = recall_at_k(ids.cpu().numpy(), gt, k)
            qps = Xq_t.shape[0] / (t / 1000.0)
            rows.append((v, t, qps, r))
        except Exception as e:
            rows.append((v, float("nan"), 0.0, float("nan")))
            print(f"    {v}: ERR {type(e).__name__}: {str(e)[:120]}")
    return index, rows


def bench_cuvs(Xc_np, Xq_np, gt, nlist, nprobe, m, k, repeat=20):
    import cupy as cp
    from cuvs.neighbors import ivf_pq

    Xc = cp.asarray(Xc_np); Xq = cp.asarray(Xq_np)
    ip = ivf_pq.IndexParams(n_lists=nlist, metric="sqeuclidean", pq_dim=m, pq_bits=8)
    index = ivf_pq.build(ip, Xc)
    sp = ivf_pq.SearchParams(n_probes=nprobe)
    _, I = ivf_pq.search(sp, index, Xq, k)
    ids = cp.asarray(I).get()
    t = time_gpu(lambda: ivf_pq.search(sp, index, Xq, k), repeat=repeat, warmup=5)
    r = recall_at_k(ids, gt, k)
    qps = Xq_np.shape[0] / (t / 1000.0)
    return t, qps, r


def coarse_time(index, Xq_t, nprobe, repeat=20):
    from flashlib.primitives.knn import flash_knn
    from flashlib.primitives.ivf_pq.torch_fallback import _pad_features

    Qp = _pad_features(Xq_t.to(torch.float32), index.Dp).contiguous()
    cent = index.centroids.to(torch.float32)
    t = time_gpu(
        lambda: flash_knn(Qp.unsqueeze(0), cent.unsqueeze(0), nprobe,
                          return_distances=False),
        repeat=repeat, warmup=5,
    )
    return t


def run_case(label, Xc_np, Xq_np, gt, nlist, nprobe, m, k, variants, nq=None,
             with_cuvs=True):
    if nq is not None:
        Xq_np = Xq_np[:nq]
        gt = gt[:nq]
    M, D = Xc_np.shape
    title(f"{label}  (M={M:,}, nq={Xq_np.shape[0]:,}, D={D}, nlist={nlist}, "
          f"nprobe={nprobe}, m={m}, k={k})")
    Xc_t = torch.as_tensor(Xc_np, device="cuda")
    Xq_t = torch.as_tensor(Xq_np, device="cuda")

    index, rows = bench_variants(Xc_t, Xq_t, gt, nlist, nprobe, m, k, variants)
    tc = coarse_time(index, Xq_t, nprobe)
    print(f"  coarse (flash_knn over centroids): {tc:8.3f} ms")
    for v, t, qps, r in rows:
        print(f"  {v:10s}  {t:8.3f} ms   {qps:12,.0f} QPS   recall@{k} {r:.4f}   fine~{t - tc:7.3f} ms")

    if with_cuvs:
        try:
            t, qps, r = bench_cuvs(Xc_np, Xq_np, gt, nlist, nprobe, m, k)
            print(f"  {'cuvs':10s}  {t:8.3f} ms   {qps:12,.0f} QPS   recall@{k} {r:.4f}")
        except Exception as e:
            print(f"  cuvs: SKIP ({str(e)[:80]})")

    del Xc_t, Xq_t
    torch.cuda.empty_cache()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="sift-128-euclidean")
    ap.add_argument("--no-cuvs", action="store_true")
    ap.add_argument("--variants", default="auto,ws,cute_gemm,cute_lut,gemm,online")
    ap.add_argument("--synthetic", action="store_true")
    ap.add_argument("--configs", default=None,
                    help="semicolon list of nlist,nprobe,m,k[,nq]")
    args = ap.parse_args()
    variants = args.variants.split(",")

    print(f"torch {torch.__version__}   GPU {torch.cuda.get_device_name(0)}")

    if args.synthetic:
        M, D = 1_000_000, 128
        Xc_np = _blobs(M, D, 128, seed=0)
        Xq_np = _blobs(10_000, D, 128, seed=1)
        Xc_t = torch.as_tensor(Xc_np, device="cuda")
        Xq_t = torch.as_tensor(Xq_np, device="cuda")
        from flashlib.primitives.knn import flash_knn
        gt = flash_knn(Xq_t.unsqueeze(0), Xc_t.unsqueeze(0), 10)[1][0].cpu().numpy()
        del Xc_t, Xq_t
        torch.cuda.empty_cache()
        cfgs = [(1024, 32, 16, 10, None), (1024, 32, 32, 10, None)]
        for nlist, nprobe, m, k, nq in cfgs:
            run_case("synthetic", Xc_np, Xq_np, gt, nlist, nprobe, m, k,
                     variants, nq=nq, with_cuvs=not args.no_cuvs)
        return

    train, test, gt_full = load_ann_dataset(args.dataset)
    print(f"loaded: base {train.shape}, query {test.shape}")

    if args.configs:
        cfgs = []
        for part in args.configs.split(";"):
            vals = [int(x) for x in part.split(",")]
            if len(vals) == 4:
                vals.append(None)
            cfgs.append(tuple(vals))
    else:
        cfgs = [
            (1024, 16, 16, 10, None),
            (1024, 32, 32, 10, None),
            (1024, 64, 64, 10, None),
            (1024, 32, 32, 10, 100),   # small batch
        ]
    for nlist, nprobe, m, k, nq in cfgs:
        run_case(args.dataset, train, test, gt_full[:, :k], nlist, nprobe, m, k,
                 variants, nq=nq, with_cuvs=not args.no_cuvs)


if __name__ == "__main__":
    main()
