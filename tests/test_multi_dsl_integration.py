"""Multi-DSL integration tests — exercise the new Triton + CuteDSL backends
through the dispatcher / info contract WITHOUT requiring a GPU.

These tests verify:
1. Every primitive's ``triton`` and ``cutedsl`` backend module imports cleanly
   on a CPU-only host (CUTLASS imports are lazy / guarded).
2. Every primitive lists its variants via ``info.variants(...)`` and each
   variant returns a sane Estimate.
3. The flat top-level aliases (``flash_<x>_cutedsl``, ``gemm_fp16_x9``, …)
   are all addressable on ``flashlib.*``.
4. Pareto filtering produces a non-empty subset.

Run via:  pytest tests/test_multi_dsl_integration.py -q
"""
from __future__ import annotations

import pytest


CUTEDSL_PRIMITIVES = [
    "kmeans",
    "knn",
    "pca",
    "truncated_svd",
    "linear_regression",
    "ridge",
    "logistic_regression",
    "dbscan",
    "hdbscan",
    "umap",
    "tsne",
    "multinomial_nb",
    "random_forest",
    "spectral_clustering",
    "standard_scaler",
]


GEMM_VARIANTS = [
    "gemm_fp32", "gemm_tf32", "gemm_3xtf32",
    "gemm_bf16", "gemm_3xbf16",
    "gemm_fp16", "gemm_3xfp16",
    "gemm_fp16_x9", "gemm_fp16_x3_kahan",
    "gemm_tf32_x6", "gemm_ozaki2_int8",
]


# Default shapes used to drive the cost estimators per primitive.
DEFAULT_SHAPES = {
    "kmeans":               ((1, 1_000_000, 128), {"K": 100}),
    "knn":                  ((1, 50_000, 50_000, 128), {"k": 10}),
    "pca":                  ((100_000, 256), {"K": 50}),
    "truncated_svd":        ((100_000, 256), {"K": 50}),
    "linear_regression":    ((100_000, 64), {}),
    "ridge":                ((100_000, 64), {}),
    "logistic_regression":  ((100_000, 64), {"n_iter": 100}),
    "dbscan":               ((20_000, 32), {"eps": 0.5, "min_samples": 5}),
    "hdbscan":              ((20_000, 32), {"min_cluster_size": 10}),
    "umap":                 ((20_000, 64), {"n_components": 2}),
    "tsne":                 ((10_000, 64), {"n_components": 2}),
    "multinomial_nb":       ((50_000, 1024), {"n_classes": 10}),
    "random_forest":        ((50_000, 64), {"n_estimators": 100}),
    "spectral_clustering":  ((10_000, 32), {"n_clusters": 8}),
    "standard_scaler":      ((100_000, 256), {}),
}


@pytest.mark.parametrize("primitive", CUTEDSL_PRIMITIVES)
def test_triton_backend_imports(primitive):
    """Every primitive's Triton backend module is importable on CPU."""
    import importlib
    importlib.import_module(f"flashlib.primitives.{primitive}.triton")


@pytest.mark.parametrize("primitive", CUTEDSL_PRIMITIVES)
def test_cutedsl_backend_imports(primitive):
    """Every primitive's CuteDSL backend module is importable on CPU.

    The actual CUTLASS/cute kernels are imported lazily inside callable
    functions, so the module body must not fail at import time even when
    cutlass is missing.
    """
    import importlib
    importlib.import_module(f"flashlib.primitives.{primitive}.cutedsl")


@pytest.mark.parametrize("primitive", CUTEDSL_PRIMITIVES)
def test_info_variants_per_primitive(primitive):
    """info.variants(<primitive>) returns the registered Triton + CuteDSL pair."""
    import flashlib.info as info
    shape, params = DEFAULT_SHAPES[primitive]
    variants = info.variants(primitive, shape=shape, params=params, device="H100")
    assert len(variants) >= 2, (
        f"{primitive}: expected ≥ 2 variants, got {[v.name for v in variants]}"
    )
    # Triton and CuteDSL should both appear.
    names = {v.name for v in variants}
    assert any("triton" in n for n in names), f"{primitive}: no triton variant"
    assert any("cutedsl" in n for n in names), f"{primitive}: no cutedsl variant"
    for v in variants:
        assert v.estimate.runtime_ms > 0


def test_info_variants_gemm():
    """gemm family lists all 11 precision variants."""
    import flashlib.info as info
    variants = info.variants("gemm", shape=(8192, 8192, 8192), device="H100")
    names = {v.name for v in variants}
    for expected in GEMM_VARIANTS:
        assert expected in names, f"missing variant: {expected}"


def test_info_pareto_gemm():
    """Pareto filtering produces a non-trivial subset of the 11 GEMM variants."""
    import flashlib.info as info
    pareto = info.pareto("gemm", shape=(8192, 8192, 8192), device="H100")
    assert 1 <= len(pareto) <= len(GEMM_VARIANTS)


def test_info_pareto_per_primitive():
    """Pareto on each primitive family returns at least one variant."""
    import flashlib.info as info
    for primitive in CUTEDSL_PRIMITIVES:
        shape, params = DEFAULT_SHAPES[primitive]
        pareto = info.pareto(primitive, shape=shape, params=params, device="H100")
        assert len(pareto) >= 1, f"{primitive}: empty pareto"


@pytest.mark.parametrize("alias", [
    "flash_kmeans_triton", "flash_kmeans_cutedsl", "flash_kmeans_cake",
    "flash_knn_cutedsl",
    "flash_pca_cutedsl",
    "flash_dbscan_cutedsl",
    "flash_hdbscan_cutedsl",
    "flash_umap_cutedsl",
    "flash_truncated_svd_cutedsl",
    "flash_linear_regression_cutedsl",
    "flash_ridge_cutedsl",
    "flash_logistic_regression_cutedsl",
    "flash_multinomial_nb_cutedsl",
    "flash_random_forest_cutedsl",
    "flash_spectral_clustering_cutedsl",
    "flash_standard_scaler_cutedsl",
    "gemm_fp16_x9", "gemm_fp16_x3_kahan", "gemm_tf32_x6", "gemm_ozaki2_int8",
])
def test_top_level_alias_resolvable(alias):
    """Each new alias is reachable as ``flashlib.<alias>``."""
    import flashlib
    fn = getattr(flashlib, alias)
    assert callable(fn), f"{alias}: not callable, got {type(fn).__name__}"


def test_kernels_cute_helpers_importable():
    """flashlib.kernels.cute_helpers is importable."""
    from flashlib.kernels.cute_helpers import is_cutedsl_available
    assert isinstance(is_cutedsl_available(), bool)


def test_kmeans_dispatcher_routes_torch_no_cuda():
    """Without CUDA, flash_kmeans should route to the torch fallback."""
    import torch
    if torch.cuda.is_available():
        pytest.skip("CUDA available; this test asserts the no-CUDA route.")
    from flashlib.primitives.kmeans import flash_kmeans
    x = torch.randn(50, 8)
    cluster_ids, centroids, n_iter = flash_kmeans(x, n_clusters=3, max_iters=5)
    assert cluster_ids.shape == (50,)
    assert centroids.shape == (3, 8)


def test_knn_dispatcher_route_function():
    """KNN ``_route()`` is pure-Python and returns a single backend name.

    The dispatcher returns a single backend identifier; the within-backend
    kernel split is a shape-driven decision inside
    :func:`flashlib.primitives.knn.triton.flash_knn_triton`.
    """
    from flashlib.primitives.knn.impl import _route as route
    backend = route(B=1, N=1024, D=768, k=10)
    assert backend in ("triton", "cutedsl", "torch")
