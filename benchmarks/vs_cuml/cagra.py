"""CAGRA: ``flash_cagra`` vs cuVS CAGRA -- the recall/QPS frontier.

CAGRA has no iso-recall parameter pairing (the graph and the traversal
order both shape recall), so the comparison that matters is the **Pareto
frontier**: for each recall level, who serves more queries per second?
Each engine builds a ``graph_degree=32`` and a ``graph_degree=64`` index
and sweeps its search knob (``itopk``); per row we report:

  * build(s)    -- one-time index construction.
  * gd / itopk  -- the graph degree + recall-knob value for the row.
  * search(ms)  -- per-batch query latency (warm, cuda-synced).
  * QPS         -- nq / search_time.
  * recall@k    -- vs exact ground truth (brute force / dataset GT).

Then the *iso-recall speedup* table: for every point on cuVS's combined
(gd32 + gd64) frontier, the best flashlib QPS at equal-or-better recall
divided by the cuVS QPS.

cuVS is optional. Install out-of-band, e.g.::

    pip install cuvs-cu12

Run::

    python -m benchmarks.vs_cuml.cagra                # synthetic shapes
    python -m benchmarks.vs_cuml.cagra --sift         # + SIFT-1M (download)
"""
from __future__ import annotations

import sys
import time

import numpy as np
import torch

from benchmarks.vs_cuml._common import time_gpu, title, fmt_table

# (label,            M,        nq,     D,  k)
SHAPES = [
    ("1M D=128",      1_000_000, 10_000, 128, 10),
    ("100K D=256",      100_000, 10_000, 256, 10),
    ("online 1M",     1_000_000,    100, 128, 10),
]

# flashlib (gd, itopk, search_width) sweep. sw pairs measured on H100:
# small windows want sw=2 (keeps CW=64 fed); gd=64 runs sw=1 up to
# itopk 192 to keep the buffer at 256 lanes, then sw=2 (an sw=1 512-lane
# buffer spills registers). itopk >= 160 auto-enables the bitonic-merge
# fast path (compact_every=8).
FLASH_GRID = [
    (32, 32, 2), (32, 64, 2), (32, 96, 2), (32, 128, 4),
    (32, 160, 2), (32, 192, 2),
    (64, 48, 1), (64, 64, 1), (64, 96, 2), (64, 128, 2),
    (64, 160, 1), (64, 192, 1), (64, 256, 2), (64, 384, 2),
]
CUVS_GRID = [
    (32, 32), (32, 64), (32, 128), (32, 256),
    (64, 64), (64, 128), (64, 256),
]


def _blobs(M, D, n_centers, seed):
    rng = np.random.RandomState(seed)
    centers = rng.randn(n_centers, D).astype(np.float32) * 4.0
    lab = rng.randint(0, n_centers, size=M)
    return (centers[lab] + rng.randn(M, D).astype(np.float32)).astype(np.float32)


def _recall(pred, ref, k):
    hit = 0
    for p, r in zip(pred, ref):
        hit += len(set(p[:k].tolist()) & set(r[:k].tolist()))
    return hit / (len(pred) * k)


def _brute_ids(Xc_t, Xq_t, k):
    from flashlib.primitives.knn import flash_knn
    idx = flash_knn(Xq_t.unsqueeze(0), Xc_t.unsqueeze(0), k)[1][0]
    return idx.cpu()


def _flash_rows(Xc_t, Xq_t, k, gt, nq):
    from flashlib import flash_cagra_build, flash_cagra_search

    rows = []
    for gd in sorted({g for g, _, _ in FLASH_GRID}):
        torch.cuda.synchronize()
        t0 = time.time()
        index = flash_cagra_build(Xc_t, graph_degree=gd)
        torch.cuda.synchronize()
        t_build = time.time() - t0
        for g, itopk, sw in FLASH_GRID:
            if g != gd:
                continue
            _, ids = flash_cagra_search(index, Xq_t, k, itopk_size=itopk,
                                        search_width=sw)
            r = _recall(ids.cpu(), gt, k)
            t = time_gpu(lambda: flash_cagra_search(
                index, Xq_t, k, itopk_size=itopk, search_width=sw),
                repeat=10, warmup=3)
            rows.append(("flashlib", f"{t_build:7.2f}", gd, itopk,
                         f"{t:8.3f}", nq / t * 1000.0, r))
        del index
    return rows


def _cuvs_rows(Xc_t, Xq_t, k, gt, nq):
    from cuvs.neighbors import cagra

    rows = []
    for gd in sorted({g for g, _ in CUVS_GRID}):
        params = cagra.IndexParams(graph_degree=gd,
                                   intermediate_graph_degree=2 * gd,
                                   build_algo="nn_descent",
                                   metric="sqeuclidean")
        torch.cuda.synchronize()
        t0 = time.time()
        index = cagra.build(params, Xc_t)
        torch.cuda.synchronize()
        t_build = time.time() - t0
        for g, itopk in CUVS_GRID:
            if g != gd:
                continue
            sp = cagra.SearchParams(itopk_size=itopk)
            _, I_ = cagra.search(sp, index, Xq_t, k)
            ids = torch.as_tensor(I_, device="cuda").cpu()
            r = _recall(ids, gt, k)
            t = time_gpu(lambda: cagra.search(sp, index, Xq_t, k),
                         repeat=10, warmup=3)
            rows.append(("cuvs", f"{t_build:7.2f}", gd, itopk,
                         f"{t:8.3f}", nq / t * 1000.0, r))
        del index
    return rows


def _frontier(rows):
    """Pareto frontier (recall asc, QPS desc) of (engine rows)."""
    pts = sorted(((r[6], r[5]) for r in rows), key=lambda p: (-p[0], -p[1]))
    front = []
    best_qps = 0.0
    for rec, qps in pts:
        if qps > best_qps:
            front.append((rec, qps))
            best_qps = qps
    return sorted(front)


def _iso_recall_speedup(flash_rows, cuvs_rows):
    """For each cuVS frontier point: best flashlib QPS at >= recall."""
    out = []
    for c_rec, c_qps in _frontier(cuvs_rows):
        cands = [r for r in flash_rows if r[6] >= c_rec]
        if not cands:
            out.append((c_rec, c_qps, None, None))
            continue
        best = max(cands, key=lambda r: r[5])
        out.append((c_rec, c_qps, best[5], best[5] / c_qps))
    return out


def run_one(label, M, nq, D, k, Xc_np=None, Xq_np=None, gt=None):
    title(f"CAGRA  {label}  (M={M:,}, nq={nq:,}, D={D}, k={k})")

    if Xc_np is None:
        Xc_np = _blobs(M, D, 128, seed=0)
        Xq_np = _blobs(nq, D, 128, seed=1)
    Xc_t = torch.tensor(Xc_np, device="cuda")
    Xq_t = torch.tensor(Xq_np, device="cuda")
    if gt is None:
        gt = _brute_ids(Xc_t, Xq_t, k)

    rows = []
    try:
        flash_rows = _flash_rows(Xc_t, Xq_t, k, gt, nq)
        rows += flash_rows
    except Exception as e:  # pragma: no cover - bench-only
        flash_rows = []
        rows.append(("flashlib", "  ERR", "", "",
                     str(e).splitlines()[0][:32], 0.0, 0.0))
    try:
        cuvs_rows = _cuvs_rows(Xc_t, Xq_t, k, gt, nq)
        rows += cuvs_rows
    except Exception as e:  # pragma: no cover - optional dep
        cuvs_rows = []
        rows.append(("cuvs", "  SKIP", "", "",
                     str(e).splitlines()[0][:32], 0.0, 0.0))

    print(fmt_table(
        [(r[0], r[1], str(r[2]), str(r[3]), r[4], f"{r[5]:12,.0f}",
          f"{r[6]:.4f}") for r in rows],
        ["engine", "build(s)", "gd", "itopk", "search(ms)", "QPS",
         f"recall@{k}"],
    ))

    if flash_rows and cuvs_rows:
        print("\n  iso-recall speedup (flashlib QPS at >= cuVS-frontier "
              "recall):")
        for c_rec, c_qps, f_qps, sp in _iso_recall_speedup(flash_rows,
                                                           cuvs_rows):
            if sp is None:
                print(f"    recall>={c_rec:.4f}: cuvs {c_qps:11,.0f} QPS   "
                      f"flashlib: not reached")
            else:
                print(f"    recall>={c_rec:.4f}: cuvs {c_qps:11,.0f} QPS   "
                      f"flashlib {f_qps:11,.0f} QPS   {sp:5.2f}x")


def run_dataset(name, k=10):
    """Run the frontier sweep on an ann-benchmarks dataset (exact GT)."""
    from benchmarks.vs_cuml._common import load_ann_dataset
    train, test, gt_np = load_ann_dataset(name)
    gt = torch.tensor(gt_np[:, :k])
    run_one(name, train.shape[0], test.shape[0], train.shape[1], k,
            Xc_np=train, Xq_np=test, gt=gt)


def main():
    print(f"torch {torch.__version__}   GPU {torch.cuda.get_device_name(0)}")
    ran_real = False
    if "--sift" in sys.argv:
        run_dataset("sift-128-euclidean")
        ran_real = True
    if "--gist" in sys.argv:
        run_dataset("gist-960-euclidean")
        ran_real = True
    if "--dataset" in sys.argv:
        run_dataset(sys.argv[sys.argv.index("--dataset") + 1])
        ran_real = True
    if "--real-only" in sys.argv:
        return
    if ran_real and "--synthetic" not in sys.argv:
        return
    for s in SHAPES:
        run_one(*s)
    print()


if __name__ == "__main__":
    main()
