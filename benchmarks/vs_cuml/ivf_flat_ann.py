"""IVF-Flat on ann-benchmarks datasets: flashlib vs cuVS, nprobe sweep.

Companion to :mod:`benchmarks.vs_cuml.ivf_flat_real` (which uses the
TexMex fvecs corpus): this one uses the **ann-benchmarks HDF5 datasets**
-- the same files and exact ground truth as the IVF-PQ benchmarks
(:mod:`benchmarks.vs_cuml.ivf_pq` / ``ivf_pq_variants``) -- so IVF-Flat
and IVF-PQ rows are directly comparable, and supports arbitrary
dataset / nlist / nprobe sweeps from the command line.

Run:
    python -m benchmarks.vs_cuml.ivf_flat_ann --dataset sift-128-euclidean
    python -m benchmarks.vs_cuml.ivf_flat_ann --dataset gist-960-euclidean
"""
from __future__ import annotations

import argparse

import torch

from benchmarks.vs_cuml._common import (
    fmt_table,
    load_ann_dataset,
    recall_at_k,
    time_gpu,
    title,
)


def _flashlib_rows(Xc_t, Xq_t, gt, nq, nlist, nprobes, k):
    from flashlib import flash_ivf_flat_build, flash_ivf_flat_search

    t_build = time_gpu(lambda: flash_ivf_flat_build(Xc_t, nlist, niter=20, seed=0),
                       repeat=1, warmup=0)
    index = flash_ivf_flat_build(Xc_t, nlist, niter=20, seed=0)
    rows = []
    for nprobe in nprobes:
        ids = flash_ivf_flat_search(index, Xq_t, k, nprobe=nprobe)[1].cpu().numpy()
        t = time_gpu(lambda: flash_ivf_flat_search(index, Xq_t, k, nprobe=nprobe),
                     repeat=10, warmup=3)
        rows.append(("flashlib", str(nprobe), f"{t_build:7.0f}", f"{t:8.3f}",
                     f"{nq / (t / 1000.0):11,.0f}",
                     f"{recall_at_k(ids, gt, k):.4f}"))
    return rows


def _cuvs_rows(Xc_np, Xq_np, gt, nq, nlist, nprobes, k):
    import cupy as cp
    from cuvs.neighbors import ivf_flat

    Xc = cp.asarray(Xc_np); Xq = cp.asarray(Xq_np)
    ip = ivf_flat.IndexParams(n_lists=nlist, metric="sqeuclidean")
    t_build = time_gpu(lambda: ivf_flat.build(ip, Xc), repeat=1, warmup=0)
    index = ivf_flat.build(ip, Xc)
    rows = []
    for nprobe in nprobes:
        sp = ivf_flat.SearchParams(n_probes=nprobe)
        _, I = ivf_flat.search(sp, index, Xq, k)
        ids = cp.asarray(I).get()
        t = time_gpu(lambda: ivf_flat.search(sp, index, Xq, k), repeat=10, warmup=3)
        rows.append(("cuvs", str(nprobe), f"{t_build:7.0f}", f"{t:8.3f}",
                     f"{nq / (t / 1000.0):11,.0f}",
                     f"{recall_at_k(ids, gt, k):.4f}"))
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="sift-128-euclidean")
    ap.add_argument("--nlist", type=int, default=1024)
    ap.add_argument("--k", type=int, default=10)
    ap.add_argument("--nprobes", default="8,16,32,64")
    ap.add_argument("--no-cuvs", action="store_true")
    args = ap.parse_args()
    nprobes = [int(x) for x in args.nprobes.split(",")]

    print(f"torch {torch.__version__}   GPU {torch.cuda.get_device_name(0)}")
    train, test, gt_full = load_ann_dataset(args.dataset)
    gt = gt_full[:, : args.k]
    M, D = train.shape
    nq = test.shape[0]
    title(f"IVF-Flat {args.dataset}  (M={M:,}, nq={nq:,}, D={D}, "
          f"nlist={args.nlist}, k={args.k})")

    Xc_t = torch.as_tensor(train, device="cuda")
    Xq_t = torch.as_tensor(test, device="cuda")

    rows = []
    rows += _flashlib_rows(Xc_t, Xq_t, gt, nq, args.nlist, nprobes, args.k)
    del Xc_t, Xq_t
    torch.cuda.empty_cache()
    if not args.no_cuvs:
        try:
            rows += _cuvs_rows(train, test, gt, nq, args.nlist, nprobes, args.k)
        except Exception as e:
            rows.append(("cuvs", "-", " SKIP", str(e).splitlines()[0][:24], "", ""))

    rows.sort(key=lambda r: (int(r[1]) if r[1].isdigit() else 0, r[0]))
    print(fmt_table(rows, ["engine", "nprobe", "build(ms)", "search(ms)",
                           "QPS", "recall@k"]))


if __name__ == "__main__":
    main()
