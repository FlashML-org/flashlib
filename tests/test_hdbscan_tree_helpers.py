"""HDBSCAN CPU tree helpers: the C ports must match the pure-Python reference bit for bit."""
from __future__ import annotations

import numpy as np
import pytest

from flashlib.primitives.hdbscan.triton import _tree_helpers as T
from flashlib.primitives.hdbscan.triton import _tree_helpers_c as TC


def _random_mst(n: int, seed: int, weights: str) -> np.ndarray:
    """Random spanning tree on ``n`` points as sorted ``[u, v, w]`` rows."""
    rng = np.random.default_rng(seed)
    u = np.arange(1, n)
    v = rng.integers(0, u)
    if weights == "ties":
        w = rng.choice([0.0, 0.5, 1.0, 2.0], size=n - 1)  # 0 -> lambda = inf
    else:
        w = rng.exponential(size=n - 1)
    order = np.argsort(w, kind="stable")
    return np.stack([u[order], v[order], w[order]], 1).astype(np.float64)


def _pipeline(label, condense, stability, get_clusters, mst, min_cluster_size):
    slt = label(mst)
    cond = condense(slt, min_cluster_size)
    ids, stab = stability(*cond)
    labels = get_clusters(*cond, ids, stab, slt.shape[0] + 1)
    return [slt, *cond, ids, stab, labels]


@pytest.mark.skipif(TC.library() is None, reason=f"C helpers unavailable: {TC.build_problem()}")
@pytest.mark.parametrize("n,seed,weights,mcs", [
    (2, 0, "ties", 2), (3, 1, "ties", 2), (500, 2, "ties", 5), (3000, 3, "ties", 3),
    (5000, 4, "continuous", 10), (20000, 5, "continuous", 25), (20000, 6, "ties", 100),
])
def test_c_helpers_match_python(n, seed, weights, mcs):
    mst = _random_mst(n, seed, weights)
    c_out = _pipeline(T._fast_label, T._fast_condense_tree, T._fast_compute_stability,
                      T._fast_get_clusters, mst, mcs)
    py_out = _pipeline(T._py_fast_label, T._py_fast_condense_tree, T._py_fast_compute_stability,
                       T._py_fast_get_clusters, mst, mcs)
    for c, py in zip(c_out, py_out):
        assert c.dtype == py.dtype
        assert np.array_equal(c, py)


def test_falls_back_to_python_without_compiler(monkeypatch, tmp_path):
    monkeypatch.setenv("CC", str(tmp_path / "no-such-cc"))
    monkeypatch.setenv("FLASHLIB_CACHE_DIR", str(tmp_path))
    TC._load.cache_clear()
    T._use_c.cache_clear()
    try:
        mst = _random_mst(1000, 7, "continuous")
        with pytest.warns(RuntimeWarning, match="pure Python"):
            slt = T._fast_label(mst)
        assert np.array_equal(slt, T._py_fast_label(mst))
    finally:
        TC._load.cache_clear()
        T._use_c.cache_clear()
