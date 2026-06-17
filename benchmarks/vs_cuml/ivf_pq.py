"""IVF-PQ: ``flash_ivf_pq`` vs cuVS / cuML IVF-PQ at matched (nlist, nprobe, m).

Like IVF-Flat, the candidate set at a fixed ``(nlist, nprobe)`` matches any
reference IVF-PQ, so the comparison that matters is **search throughput
(QPS) at equal recall**. IVF-PQ additionally stores each vector as ``m``
1-byte codes, so we also report the compression ratio -- the reason
IVF-PQ scales to billions of vectors on a single GPU.

Two data regimes (mirrors how ANN systems are evaluated):

  * **synthetic** -- Gaussian blobs; recall ground truth is exact brute
    force (``flash_knn`` fp32). Good for shape/throughput sweeps.
  * **real** -- ann-benchmarks datasets (default SIFT1M, 1M x 128) with the
    dataset's own exact top-100 neighbours as ground truth. This is where
    the recall / compression trade-off actually matters.

Per row we report, for flashlib and (when installed) cuVS:

  * build(ms)   -- one-time index construction (coarse + PQ codebooks + encode).
  * search(ms)  -- per-batch query latency (warm, cuda-synced).
  * QPS         -- nq / search_time.
  * recall@k    -- vs exact ground truth.
  * B/vec       -- PQ code size (``m`` bytes for nbits=8); fp32 baseline 4*D.
  * cmpr        -- compression vs fp32 (4*D / m).

cuVS / cuML are optional. Install out-of-band, e.g.::

    pip install --extra-index-url https://pypi.nvidia.com cuvs-cu13 cupy-cuda13x

Real datasets need ``h5py`` (``pip install h5py``); the file is cached under
``~/.cache/flashlib_bench`` (override with ``FLASHLIB_BENCH_CACHE``).

Run::

    python -m benchmarks.vs_cuml.ivf_pq                 # synthetic + real
    python -m benchmarks.vs_cuml.ivf_pq --synthetic-only
    python -m benchmarks.vs_cuml.ivf_pq --real-only --dataset sift-128-euclidean
"""
from __future__ import annotations

import argparse

import numpy as np
import torch

from benchmarks.vs_cuml._common import (
    fmt_table,
    load_ann_dataset,
    recall_at_k,
    time_gpu,
    title,
)


# (label,             M,        nq,    D,   nlist, nprobe, m,   k)
SYNTH_SHAPES = [
    ("1M  D=64",       1_000_000, 10_000, 64,  1024,  32,   16,  10),
    ("1M  D=128",      1_000_000, 10_000, 128, 1024,  48,   32,  10),
    ("online D=64",    1_000_000,    100, 64,  1024,  32,   16,  10),
    ("500K D=96",        500_000,  5_000, 96,   512,  32,   24,  10),
]

# Real data (SIFT1M, D=128): sweep m -> trade compression for recall.
# (nlist, nprobe, m,  k)
REAL_CONFIGS = [
    (1024, 16, 16, 10),   # 8 B/vec  -> 64x compression
    (1024, 32, 32, 10),   # 16 B/vec -> 32x compression
    (1024, 64, 64, 10),   # 32 B/vec -> 16x compression
]

COLUMNS = ["engine", "build(ms)", "search(ms)", "QPS", "recall@k", "B/vec", "cmpr"]


def _blobs(M, D, n_centers, seed):
    rng = np.random.RandomState(seed)
    centers = rng.randn(n_centers, D).astype(np.float32) * 4.0
    lab = rng.randint(0, n_centers, size=M)
    return (centers[lab] + rng.randn(M, D).astype(np.float32)).astype(np.float32)


def _brute_ids(Xc_t, Xq_t, k):
    """Exact top-k ids (squared L2) via flash_knn fp32 -- recall ground truth."""
    from flashlib.primitives.knn import flash_knn
    idx = flash_knn(Xq_t.unsqueeze(0), Xc_t.unsqueeze(0), k)[1][0]
    return idx.cpu().numpy()


def _flashlib_row(Xc_t, Xq_t, nq, D, nlist, nprobe, m, k, gt):
    from flashlib import flash_ivf_pq_build, flash_ivf_pq_search

    t_build = time_gpu(
        lambda: flash_ivf_pq_build(Xc_t, nlist, m=m, nprobe=nprobe, niter=20, seed=0),
        repeat=1, warmup=0,
    )
    index = flash_ivf_pq_build(Xc_t, nlist, m=m, nprobe=nprobe, niter=20, seed=0)

    ids = flash_ivf_pq_search(index, Xq_t, k, nprobe=nprobe)[1].cpu().numpy()
    t_search = time_gpu(
        lambda: flash_ivf_pq_search(index, Xq_t, k, nprobe=nprobe),
        repeat=10, warmup=3,
    )
    qps = nq / (t_search / 1000.0)
    bvec = index.code_size_bytes()
    return ("flashlib", f"{t_build:8.1f}", f"{t_search:8.3f}", f"{qps:11,.0f}",
            f"{recall_at_k(ids, gt, k):.4f}", f"{bvec:5d}", f"{4*D/bvec:4.0f}x")


def _cuvs_row(Xc_np, Xq_np, nq, D, nlist, nprobe, m, k, gt):
    import cupy as cp
    from cuvs.neighbors import ivf_pq

    Xc = cp.asarray(Xc_np); Xq = cp.asarray(Xq_np)
    ip = ivf_pq.IndexParams(n_lists=nlist, metric="sqeuclidean", pq_dim=m, pq_bits=8)
    t_build = time_gpu(lambda: ivf_pq.build(ip, Xc), repeat=1, warmup=0)
    index = ivf_pq.build(ip, Xc)
    sp = ivf_pq.SearchParams(n_probes=nprobe)

    _, I = ivf_pq.search(sp, index, Xq, k)
    ids = cp.asarray(I).get()
    t_search = time_gpu(lambda: ivf_pq.search(sp, index, Xq, k),
                        repeat=10, warmup=3)
    qps = nq / (t_search / 1000.0)
    return ("cuvs", f"{t_build:8.1f}", f"{t_search:8.3f}", f"{qps:11,.0f}",
            f"{recall_at_k(ids, gt, k):.4f}", f"{m:5d}", f"{4*D/m:4.0f}x")


def run_case(label, Xc_np, Xq_np, gt, nlist, nprobe, m, k, with_cuvs=True):
    """One benchmark row-group given base/query numpy arrays + exact GT ids."""
    M, D = Xc_np.shape
    nq = Xq_np.shape[0]
    title(f"IVF-PQ  {label}  (M={M:,}, nq={nq:,}, D={D}, nlist={nlist}, "
          f"nprobe={nprobe}, m={m}, k={k})")

    Xc_t = torch.as_tensor(Xc_np, device="cuda")
    Xq_t = torch.as_tensor(Xq_np, device="cuda")

    rows = []
    try:
        rows.append(_flashlib_row(Xc_t, Xq_t, nq, D, nlist, nprobe, m, k, gt))
    except Exception as e:  # pragma: no cover - bench-only
        rows.append(("flashlib", "  ERR", str(e).splitlines()[0][:24], "", "", "", ""))

    if with_cuvs:
        try:
            rows.append(_cuvs_row(Xc_np, Xq_np, nq, D, nlist, nprobe, m, k, gt))
        except Exception as e:  # pragma: no cover - optional dep
            rows.append(("cuvs", "  SKIP", str(e).splitlines()[0][:24], "", "", "", ""))

    print(fmt_table(rows, COLUMNS))
    del Xc_t, Xq_t
    torch.cuda.empty_cache()


def run_synthetic(with_cuvs=True):
    print("\n### synthetic (Gaussian blobs, brute-force GT) ###")
    for label, M, nq, D, nlist, nprobe, m, k in SYNTH_SHAPES:
        Xc_np = _blobs(M, D, max(8, nlist // 8), seed=0)
        Xq_np = _blobs(nq, D, max(8, nlist // 8), seed=1)
        Xc_t = torch.as_tensor(Xc_np, device="cuda")
        Xq_t = torch.as_tensor(Xq_np, device="cuda")
        gt = _brute_ids(Xc_t, Xq_t, k)
        del Xc_t, Xq_t
        torch.cuda.empty_cache()
        run_case(label, Xc_np, Xq_np, gt, nlist, nprobe, m, k, with_cuvs=with_cuvs)


def run_real(dataset="sift-128-euclidean", with_cuvs=True):
    print(f"\n### real ({dataset}, dataset exact GT) ###")
    try:
        train, test, gt_full = load_ann_dataset(dataset)
    except Exception as e:  # pragma: no cover - optional / network
        print(f"  SKIP real data ({dataset}): {e}")
        return
    print(f"  loaded: base {train.shape}, query {test.shape}, gt {gt_full.shape}")
    for nlist, nprobe, m, k in REAL_CONFIGS:
        run_case(dataset, train, test, gt_full[:, :k], nlist, nprobe, m, k,
                 with_cuvs=with_cuvs)


def main():
    ap = argparse.ArgumentParser(description="flash_ivf_pq vs cuVS IVF-PQ")
    ap.add_argument("--synthetic-only", action="store_true")
    ap.add_argument("--real-only", action="store_true")
    ap.add_argument("--dataset", default="sift-128-euclidean",
                    help="ann-benchmarks dataset for the real regime")
    ap.add_argument("--no-cuvs", action="store_true", help="skip the cuVS rows")
    args = ap.parse_args()

    print(f"torch {torch.__version__}   GPU {torch.cuda.get_device_name(0)}")
    with_cuvs = not args.no_cuvs
    if not args.real_only:
        run_synthetic(with_cuvs=with_cuvs)
    if not args.synthetic_only:
        run_real(args.dataset, with_cuvs=with_cuvs)
    print()


if __name__ == "__main__":
    main()
