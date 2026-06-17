"""Shared utilities for the ``benchmarks/vs_cuml`` suite.

Each per-primitive script (``knn.py``, ``kmeans.py``, ``dbscan.py``,
``hdbscan.py``) imports from here:

* ``cap_threads()`` -- caps OpenBLAS / OMP / MKL threads BEFORE numpy /
  sklearn are touched. Without this, sklearn brute-force segfaults on
  big SKU's because OpenBLAS was built with NUM_THREADS=128.
* ``cuml_shim()`` -- patches a missing ``BaseEstimator._get_default_requests``
  alias so cuml 25.10 can import on top of scikit-learn 1.8.
* ``time_gpu()`` -- standard GPU timer (warmup + repeats).
* ``hr()``, ``title()`` -- small print helpers.
* Convenience metric helpers (``ari``, ``recall_at_k``, ``cluster_count``).

cuML is intentionally NOT in ``flashlib/pyproject.toml`` -- install it
out-of-band with::

    pip install --extra-index-url https://pypi.nvidia.com 'cuml-cu12==25.10.*'
"""
from __future__ import annotations
import os
import time


def cap_threads(n: int = 8) -> None:
    """Limit BLAS / OpenMP threads before numpy is imported.

    sklearn's brute KNN otherwise crashes on machines with > 128 cores
    because OpenBLAS hits its compile-time MAX_THREADS cap.
    """
    os.environ.setdefault("OPENBLAS_NUM_THREADS", str(n))
    os.environ.setdefault("OMP_NUM_THREADS", str(n))
    os.environ.setdefault("MKL_NUM_THREADS", str(n))


def cuml_shim() -> None:
    """Make cuml 25.10 importable on top of sklearn 1.8.

    sklearn 1.8 dropped the private ``_get_default_requests`` alias that
    ``cuml.accel.estimator_proxy`` decorates with ``functools.wraps``.
    The new public name is ``_get_metadata_request``; we forward.
    """
    from sklearn.base import BaseEstimator
    if not hasattr(BaseEstimator, "_get_default_requests"):
        BaseEstimator._get_default_requests = BaseEstimator._get_metadata_request


def time_gpu(fn, repeat: int = 5, warmup: int = 2) -> float:
    """Time a CUDA-bound callable in milliseconds (median-style mean)."""
    import torch
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    t0 = time.time()
    for _ in range(repeat):
        fn()
    torch.cuda.synchronize()
    return (time.time() - t0) / repeat * 1000.0


def time_cpu(fn, repeat: int = 1, warmup: int = 0) -> float:
    """Time a CPU-bound callable in milliseconds."""
    for _ in range(warmup):
        fn()
    t0 = time.time()
    for _ in range(repeat):
        fn()
    return (time.time() - t0) / repeat * 1000.0


def title(s: str) -> None:
    print(f"\n{s}\n" + "=" * len(s))


def hr(width: int = 60) -> None:
    print("  " + "-" * width)


# ── metric helpers ──────────────────────────────────────────────────────
def ari(ref, pred) -> float:
    from sklearn.metrics import adjusted_rand_score
    import numpy as np
    return float(adjusted_rand_score(ref, np.asarray(pred)))


def recall_at_k(pred_idx, ref_idx, K: int) -> float:
    """Average overlap of pred's top-K and ref's top-K per query."""
    import numpy as np
    return float(np.mean([
        len(set(p) & set(r)) / K for p, r in zip(pred_idx, ref_idx)
    ]))


def cluster_count(labels) -> int:
    """Number of non-noise clusters (excludes label -1)."""
    s = set(int(x) for x in labels)
    return len(s) - (1 if -1 in s else 0)


# ── real ANN datasets (ann-benchmarks HDF5: train / test / exact GT) ─────────
_ANN_DATASETS = {
    # name -> (url, metric). HDF5 layout: train (N,D), test (nq,D),
    # neighbors (nq,100) exact top-100 ids, distances (nq,100).
    "sift-128-euclidean": ("http://ann-benchmarks.com/sift-128-euclidean.hdf5", "l2"),
    "glove-100-angular": ("http://ann-benchmarks.com/glove-100-angular.hdf5", "cosine"),
    "fashion-mnist-784-euclidean":
        ("http://ann-benchmarks.com/fashion-mnist-784-euclidean.hdf5", "l2"),
}


def ann_cache_dir() -> str:
    """Directory for cached datasets (override via ``FLASHLIB_BENCH_CACHE``)."""
    import os
    d = os.environ.get(
        "FLASHLIB_BENCH_CACHE", os.path.expanduser("~/.cache/flashlib_bench")
    )
    os.makedirs(d, exist_ok=True)
    return d


def load_ann_dataset(name: str = "sift-128-euclidean"):
    """Download (once, cached) an ann-benchmarks dataset.

    Returns ``(train, test, gt)`` -- ``train`` ``(N, D)`` f32 base vectors,
    ``test`` ``(nq, D)`` f32 queries, ``gt`` ``(nq, 100)`` int64 exact
    nearest-neighbour ids (the recall ground truth). Raises a helpful
    error if ``h5py`` is missing or the download fails so callers can SKIP.
    """
    import os
    import numpy as np
    try:
        import h5py
    except ImportError as e:  # pragma: no cover - optional dep
        raise RuntimeError("real datasets need h5py (`pip install h5py`)") from e
    if name not in _ANN_DATASETS:
        raise KeyError(f"unknown dataset {name!r}; known: {list(_ANN_DATASETS)}")

    url, _metric = _ANN_DATASETS[name]
    path = os.path.join(ann_cache_dir(), name + ".hdf5")
    if not os.path.exists(path):
        import urllib.request
        tmp = path + ".part"

        def _hook(blk, bs, total):
            done = blk * bs
            if total > 0:
                print(f"\r  downloading {name}: {100*done/total:5.1f}% "
                      f"({done/1e6:6.0f}/{total/1e6:.0f} MB)", end="", flush=True)

        print(f"  fetching {url}")
        # ann-benchmarks.com is behind Cloudflare, which 403s the default
        # urllib User-Agent -- present a browser-like one.
        opener = urllib.request.build_opener()
        opener.addheaders = [("User-Agent", "Mozilla/5.0 (flashlib-bench)")]
        urllib.request.install_opener(opener)
        urllib.request.urlretrieve(url, tmp, _hook)
        print()
        os.rename(tmp, path)

    with h5py.File(path, "r") as f:
        train = np.ascontiguousarray(f["train"][:], dtype=np.float32)
        test = np.ascontiguousarray(f["test"][:], dtype=np.float32)
        gt = np.ascontiguousarray(f["neighbors"][:], dtype=np.int64)
    return train, test, gt


def header() -> None:
    """Print a one-line environment header."""
    import torch, cuml
    print(f"torch  {torch.__version__}   cuml {cuml.__version__}   "
          f"GPU {torch.cuda.get_device_name(0)}")


def fmt_table(rows, columns) -> str:
    """rows: list of (engine, time_ms, *metrics).
    columns: list of column header names."""
    widths = [max(len(c), max(len(str(r[i])) for r in rows)) for i, c in enumerate(columns)]
    fmt = "  " + "  ".join(f"{{:>{w}s}}" for w in widths)
    out = [fmt.format(*columns), "  " + "  ".join("-" * w for w in widths)]
    for r in rows:
        out.append(fmt.format(*[str(x) for x in r]))
    return "\n".join(out)
