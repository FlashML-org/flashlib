"""CAGRA: ``flash_cagra`` vs cuVS CAGRA at matched (graph_degree, itopk, search_width).

CAGRA is a *graph* ANN method, so -- unlike IVF -- there is no
``(nlist, nprobe)`` that reproduces an exact candidate set. The contract is
**search throughput (QPS) at equal recall**, with both engines configured at
the same ``graph_degree`` / ``intermediate_degree`` (graph build) and the
same ``itopk_size`` / ``search_width`` (traversal budget). ``itopk_size`` is
the dominant accuracy/speed knob on both sides.

Two data regimes (mirrors how ANN systems are evaluated):

  * **synthetic** -- Gaussian blobs; recall ground truth is exact brute
    force (``flash_knn`` fp32). Good for shape/throughput sweeps.
  * **real** -- ann-benchmarks datasets (default SIFT1M, 1M x 128) with the
    dataset's own exact top-100 neighbours as ground truth.

Per row we report, for flashlib and (when installed) cuVS:

  * build(ms)   -- one-time graph construction (initial kNN graph + optimize).
  * search(ms)  -- per-batch query latency (warm, cuda-synced).
  * QPS         -- nq / search_time.
  * recall@k    -- vs exact ground truth.
  * deg         -- graph out-degree.
  * itopk       -- internal candidate-buffer size used for the search.

cuVS is optional. Install out-of-band, e.g.::

    pip install --extra-index-url https://pypi.nvidia.com cuvs-cu13 cupy-cuda13x

Real datasets need ``h5py`` (``pip install h5py``); the file is cached under
``~/.cache/flashlib_bench`` (override with ``FLASHLIB_BENCH_CACHE``).

Run::

    python -m benchmarks.vs_cuml.cagra                 # synthetic + real
    python -m benchmarks.vs_cuml.cagra --synthetic-only
    python -m benchmarks.vs_cuml.cagra --real-only --dataset gist-960-euclidean
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


# (label,           M,         nq,     D,   graph_degree, itopk, sw, k)
SYNTH_SHAPES = [
    ("1M  D=64",     1_000_000, 10_000, 64,  64,  64,  1, 10),
    ("1M  D=128",    1_000_000, 10_000, 128, 64,  128, 1, 10),
    ("online D=64",  1_000_000,    100, 64,  64,  64,  1, 10),
    ("500K D=96",      500_000,  5_000, 96,  32,  64,  1, 10),
]

# Real data: sweep itopk -> trade search latency for recall at fixed degree.
# (graph_degree, intermediate_degree, itopk, search_width, k)
REAL_CONFIGS = [
    (32, 64,  64,  1, 10),
    (64, 128, 128, 1, 10),
    (64, 128, 256, 1, 10),
]

COLUMNS = ["engine", "build(ms)", "search(ms)", "QPS", "recall@k", "deg", "itopk"]


def _blobs(M, D, n_centers, seed):
    """Gaussian data. ``n_centers==1`` is a single connected cloud.

    Unlike IVF, graph ANN recall depends on graph *connectivity*: widely
    separated blobs build a disconnected graph the traversal cannot cross,
    so the synthetic regime uses one cloud (the real datasets carry their
    own natural structure).
    """
    rng = np.random.RandomState(seed)
    if n_centers <= 1:
        return rng.randn(M, D).astype(np.float32)
    centers = rng.randn(n_centers, D).astype(np.float32) * 4.0
    lab = rng.randint(0, n_centers, size=M)
    return (centers[lab] + rng.randn(M, D).astype(np.float32)).astype(np.float32)


def _brute_ids(Xc_t, Xq_t, k):
    """Exact top-k ids (squared L2) via flash_knn fp32 -- recall ground truth."""
    from flashlib.primitives.knn import flash_knn
    idx = flash_knn(Xq_t.unsqueeze(0), Xc_t.unsqueeze(0), k)[1][0]
    return idx.cpu().numpy()


def _flashlib_row(Xc_t, Xq_t, nq, gd, inter, itopk, sw, k, gt):
    from flashlib import flash_cagra_build, flash_cagra_search

    def _build():
        return flash_cagra_build(Xc_t, graph_degree=gd, intermediate_degree=inter,
                                 seed=0)

    t_build = time_gpu(_build, repeat=1, warmup=0)
    index = _build()

    def _search():
        return flash_cagra_search(index, Xq_t, k, itopk_size=itopk,
                                  search_width=sw)

    ids = _search()[1].cpu().numpy()
    t_search = time_gpu(_search, repeat=10, warmup=3)
    qps = nq / (t_search / 1000.0)
    return ("flashlib", f"{t_build:8.1f}", f"{t_search:8.3f}", f"{qps:11,.0f}",
            f"{recall_at_k(ids, gt, k):.4f}", f"{index.graph_degree:3d}",
            f"{itopk:5d}")


def _cuvs_row(Xc_np, Xq_np, nq, gd, inter, itopk, sw, k, gt):
    import cupy as cp
    from cuvs.neighbors import cagra

    Xc = cp.asarray(Xc_np); Xq = cp.asarray(Xq_np)
    # cuVS's ivf_pq graph build raises on some datasets ("too many invalid
    # or duplicated neighbor nodes"); nn_descent is its faster default and
    # builds cleanly. The search comparison is matched on (graph_degree,
    # itopk, search_width) regardless of which build produced the graph.
    def _build(algo):
        ip = cagra.IndexParams(metric="sqeuclidean", graph_degree=gd,
                               intermediate_graph_degree=inter, build_algo=algo)
        return cagra.build(ip, Xc)

    try:
        index = _build("nn_descent")
        algo = "nn_descent"
    except Exception:
        index = _build("ivf_pq")
        algo = "ivf_pq"
    t_build = time_gpu(lambda: _build(algo), repeat=1, warmup=0)
    sp = cagra.SearchParams(itopk_size=itopk, search_width=sw)

    _, I = cagra.search(sp, index, Xq, k)
    ids = cp.asarray(I).get()
    t_search = time_gpu(lambda: cagra.search(sp, index, Xq, k),
                        repeat=10, warmup=3)
    qps = nq / (t_search / 1000.0)
    return ("cuvs", f"{t_build:8.1f}", f"{t_search:8.3f}", f"{qps:11,.0f}",
            f"{recall_at_k(ids, gt, k):.4f}", f"{gd:3d}", f"{itopk:5d}")


def run_case(label, Xc_np, Xq_np, gt, gd, inter, itopk, sw, k, with_cuvs=True):
    """One benchmark row-group given base/query numpy arrays + exact GT ids."""
    M, D = Xc_np.shape
    nq = Xq_np.shape[0]
    title(f"CAGRA  {label}  (M={M:,}, nq={nq:,}, D={D}, deg={gd}, "
          f"inter={inter}, itopk={itopk}, sw={sw}, k={k})")

    Xc_t = torch.as_tensor(Xc_np, device="cuda")
    Xq_t = torch.as_tensor(Xq_np, device="cuda")

    rows = []
    try:
        rows.append(_flashlib_row(Xc_t, Xq_t, nq, gd, inter, itopk, sw, k, gt))
    except Exception as e:  # pragma: no cover - bench-only
        rows.append(("flashlib", "  ERR", str(e).splitlines()[0][:24], "", "", "", ""))

    if with_cuvs:
        try:
            rows.append(_cuvs_row(Xc_np, Xq_np, nq, gd, inter, itopk, sw, k, gt))
        except Exception as e:  # pragma: no cover - optional dep
            rows.append(("cuvs", "  SKIP", str(e).splitlines()[0][:24], "", "", "", ""))

    print(fmt_table(rows, COLUMNS))
    del Xc_t, Xq_t
    torch.cuda.empty_cache()


def run_synthetic(with_cuvs=True):
    print("\n### synthetic (Gaussian blobs, brute-force GT) ###")
    for label, M, nq, D, gd, itopk, sw, k in SYNTH_SHAPES:
        # One connected cloud (separated blobs would split the graph).
        Xc_np = _blobs(M, D, 1, seed=0)
        Xq_np = _blobs(nq, D, 1, seed=1)
        Xc_t = torch.as_tensor(Xc_np, device="cuda")
        Xq_t = torch.as_tensor(Xq_np, device="cuda")
        gt = _brute_ids(Xc_t, Xq_t, k)
        del Xc_t, Xq_t
        torch.cuda.empty_cache()
        run_case(label, Xc_np, Xq_np, gt, gd, 2 * gd, itopk, sw, k,
                 with_cuvs=with_cuvs)


def run_real(dataset="sift-128-euclidean", with_cuvs=True):
    print(f"\n### real ({dataset}, dataset exact GT) ###")
    try:
        train, test, gt_full = load_ann_dataset(dataset)
    except Exception as e:  # pragma: no cover - optional / network
        print(f"  SKIP real data ({dataset}): {e}")
        return
    print(f"  loaded: base {train.shape}, query {test.shape}, gt {gt_full.shape}")
    for gd, inter, itopk, sw, k in REAL_CONFIGS:
        run_case(dataset, train, test, gt_full[:, :k], gd, inter, itopk, sw, k,
                 with_cuvs=with_cuvs)


def main():
    ap = argparse.ArgumentParser(description="flash_cagra vs cuVS CAGRA")
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
