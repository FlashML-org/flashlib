"""flashlib — high-performance ML primitives + applications + informative cost API.

All primitives are first-class citizens: importable directly at the top level.

    from flashlib import flash_kmeans, flash_knn, flash_pca, flash_dbscan
    from flashlib import flash_truncated_svd, flash_linear_regression, flash_ridge
    from flashlib import flash_logistic_regression, flash_hdbscan, flash_umap, flash_tsne
    from flashlib import flash_multinomial_nb, flash_random_forest, flash_spectral_clustering
    from flashlib import flash_standard_scaler

Per-primitive backend variants are also flat-discoverable, e.g.:

    from flashlib import flash_kmeans_cutedsl, flash_knn_cutedsl_build_fa3
    from flashlib import flash_dbscan_cutedsl, flash_pca_cutedsl, ...

Other top-level entries:

    from flashlib import batch_kmeans_Euclid, pairwise_l2, cov_gemm, eigh
    from flashlib import KMeans, NearestNeighbors, PCA, StandardScaler, DBSCAN
    from flashlib import gemm, gemm_fp16_x9, gemm_ozaki2_int8           # multi-precision

The submodule `flashlib.info` is intentionally lazy: importing it (or
`flashlib.diagnose`) does NOT load torch / triton — agents can call it in
GPU-less environments. Primitives load lazily on first attribute access.
"""
from __future__ import annotations

import importlib
from importlib.metadata import PackageNotFoundError, version as _pkg_version

try:
    __version__ = _pkg_version("flashlib")
except PackageNotFoundError:
    __version__ = "0.0.0+unknown"

# Eager — pure stdlib + no GPU work.
from flashlib.diagnose import diagnose
from flashlib import info

# Lazy attribute map: <flashlib.attr> -> (module_path, attr_name).
_LAZY_ATTRS: dict[str, tuple[str, str]] = {
    # === algorithm primitives ===
    # kmeans
    "batch_kmeans_Euclid":       ("flashlib.primitives.kmeans", "batch_kmeans_Euclid"),
    "batch_kmeans_Cosine":       ("flashlib.primitives.kmeans", "batch_kmeans_Cosine"),
    "batch_kmeans_Dot":          ("flashlib.primitives.kmeans", "batch_kmeans_Dot"),
    "kmeans_largeN":             ("flashlib.primitives.kmeans", "kmeans_largeN"),
    "kmeans_largeN_assign":      ("flashlib.primitives.kmeans", "kmeans_largeN_assign"),
    "flash_kmeans":              ("flashlib.primitives.kmeans", "flash_kmeans"),
    "flash_kmeans_triton":       ("flashlib.primitives.kmeans", "flash_kmeans_triton"),
    "flash_kmeans_cutedsl":      ("flashlib.primitives.kmeans", "flash_kmeans_cutedsl"),
    # knn -- single fused entry point (Triton default, CuteDSL FA3 opt-in).
    "flash_knn":                  ("flashlib.primitives.knn", "flash_knn"),
    "flash_knn_triton":           ("flashlib.primitives.knn", "flash_knn_triton"),
    "flash_knn_cutedsl":          ("flashlib.primitives.knn", "cutedsl_flash_knn"),
    "flash_knn_dispatch":         ("flashlib.primitives.knn", "flash_knn_dispatch"),
    # ivf-flat -- GPU ANN (inverted file); build once, search many.
    "flash_ivf_flat":             ("flashlib.primitives.ivf_flat", "flash_ivf_flat"),
    "flash_ivf_flat_build":       ("flashlib.primitives.ivf_flat", "flash_ivf_flat_build"),
    "flash_ivf_flat_search":      ("flashlib.primitives.ivf_flat", "flash_ivf_flat_search"),
    "IvfFlatIndex":               ("flashlib.primitives.ivf_flat", "IvfFlatIndex"),
    # ivf-pq -- GPU ANN (inverted file + product quantization); compressed.
    "flash_ivf_pq":               ("flashlib.primitives.ivf_pq", "flash_ivf_pq"),
    "flash_ivf_pq_build":         ("flashlib.primitives.ivf_pq", "flash_ivf_pq_build"),
    "flash_ivf_pq_search":        ("flashlib.primitives.ivf_pq", "flash_ivf_pq_search"),
    "IvfPqIndex":                 ("flashlib.primitives.ivf_pq", "IvfPqIndex"),
    # decomposition
    "flash_pca":                 ("flashlib.primitives.pca", "flash_pca"),
    "flash_pca_cutedsl":         ("flashlib.primitives.pca", "flash_pca_cutedsl"),
    "cutedsl_pca":               ("flashlib.primitives.pca", "cutedsl_pca"),
    "flash_truncated_svd":       ("flashlib.primitives.truncated_svd", "flash_truncated_svd"),
    "flash_truncated_svd_cutedsl":("flashlib.primitives.truncated_svd", "flash_truncated_svd_cutedsl"),
    "cutedsl_truncated_svd":     ("flashlib.primitives.truncated_svd", "cutedsl_truncated_svd"),
    # regression
    "flash_linear_regression":   ("flashlib.primitives.linear_regression", "flash_linear_regression"),
    "flash_linear_regression_cutedsl": ("flashlib.primitives.linear_regression", "cutedsl_linear_regression"),
    "flash_ridge":               ("flashlib.primitives.ridge", "flash_ridge"),
    "flash_ridge_regression":    ("flashlib.primitives.ridge", "flash_ridge_regression"),
    "flash_ridge_cutedsl":       ("flashlib.primitives.ridge", "cutedsl_ridge_regression"),
    "flash_logistic_regression": ("flashlib.primitives.logistic_regression", "flash_logistic_regression"),
    "flash_logistic_regression_cutedsl": ("flashlib.primitives.logistic_regression", "cutedsl_logistic_regression"),
    # density-based clustering
    "flash_dbscan":              ("flashlib.primitives.dbscan", "flash_dbscan"),
    "flash_dbscan_cutedsl":      ("flashlib.primitives.dbscan", "cutedsl_dbscan"),
    "flash_hdbscan":             ("flashlib.primitives.hdbscan", "flash_hdbscan"),
    "flash_hdbscan_cutedsl":     ("flashlib.primitives.hdbscan", "cutedsl_hdbscan"),
    # manifold learning
    "flash_umap":                ("flashlib.primitives.umap", "flash_umap"),
    "flash_umap_cutedsl":        ("flashlib.primitives.umap", "cutedsl_flash_umap"),
    "flash_tsne":                ("flashlib.primitives.tsne", "flash_tsne"),
    "flash_tsne_perplex_cutedsl":("flashlib.primitives.tsne", "cutedsl_tsne_perplex_bisect"),
    # classification
    "flash_multinomial_nb":      ("flashlib.primitives.multinomial_nb", "flash_multinomial_nb"),
    "flash_multinomial_nb_cutedsl":("flashlib.primitives.multinomial_nb", "cutedsl_multinomial_nb"),
    "flash_random_forest":       ("flashlib.primitives.random_forest", "flash_random_forest"),
    "flash_random_forest_cutedsl":("flashlib.primitives.random_forest", "cutedsl_predict_classifier"),
    # graph clustering
    "flash_spectral_clustering": ("flashlib.primitives.spectral_clustering", "flash_spectral_clustering"),
    "flash_spectral_clustering_cutedsl": ("flashlib.primitives.spectral_clustering", "cutedsl_spectral_clustering"),
    # preprocessing
    "flash_standard_scaler":     ("flashlib.primitives.standard_scaler", "flash_standard_scaler"),
    "flash_standard_scaler_cutedsl": ("flashlib.primitives.standard_scaler", "cutedsl_standard_scaler_fit_transform"),
    # === linalg primitives ===
    "cov_gemm":                  ("flashlib.linalg.cov_gemm", "cov_gemm"),
    "gram_gemm":                 ("flashlib.linalg.gram_gemm", "gram_gemm"),
    "ab_gemm":                   ("flashlib.linalg.ab_gemm", "ab_gemm"),
    "eigh":                      ("flashlib.linalg.eigh", "eigh"),
    # === kernel primitives ===
    "pairwise_l2":               ("flashlib.kernels.distance", "pairwise_l2"),
    "pairwise_l2sq":             ("flashlib.kernels.distance", "pairwise_l2sq"),
    "connected_components":      ("flashlib.kernels.connected_components", "connected_components"),
    "flash_rmsnorm":             ("flashlib.kernels.norm", "flash_rmsnorm"),
    "flash_layernorm":           ("flashlib.kernels.norm", "flash_layernorm"),
    # === applications ===
    "KMeans":            ("flashlib.applications", "KMeans"),
    "FlashKMeans":       ("flashlib.applications", "FlashKMeans"),
    "NearestNeighbors":  ("flashlib.applications", "NearestNeighbors"),
    "IVFFlat":           ("flashlib.applications", "IVFFlat"),
    "IVFPQ":             ("flashlib.applications", "IVFPQ"),
    "PCA":               ("flashlib.applications", "PCA"),
    "StandardScaler":    ("flashlib.applications", "StandardScaler"),
    "DBSCAN":            ("flashlib.applications", "DBSCAN"),
    "TruncatedSVD":      ("flashlib.applications", "TruncatedSVD"),
    "LinearRegression":  ("flashlib.applications", "LinearRegression"),
    "Ridge":             ("flashlib.applications", "Ridge"),
    "LogisticRegression":("flashlib.applications", "LogisticRegression"),
    "HDBSCAN":           ("flashlib.applications", "HDBSCAN"),
    "UMAP":              ("flashlib.applications", "UMAP"),
    "TSNE":              ("flashlib.applications", "TSNE"),
    "MultinomialNB":     ("flashlib.applications", "MultinomialNB"),
    "RandomForestClassifier": ("flashlib.applications", "RandomForestClassifier"),
    "SpectralClustering":("flashlib.applications", "SpectralClustering"),
    # === multi-precision GEMM variants (Pareto frontier on rel-err vs runtime) ===
    "gemm":              ("flashlib.linalg.gemm", "gemm"),
    "gemm_fp32":         ("flashlib.linalg.gemm", "gemm_fp32"),
    "gemm_tf32":         ("flashlib.linalg.gemm", "gemm_tf32"),
    "gemm_3xtf32":       ("flashlib.linalg.gemm", "gemm_3xtf32"),
    "gemm_bf16":         ("flashlib.linalg.gemm", "gemm_bf16"),
    "gemm_3xbf16":       ("flashlib.linalg.gemm", "gemm_3xbf16"),
    "gemm_fp16":         ("flashlib.linalg.gemm", "gemm_fp16"),
    "gemm_3xfp16":       ("flashlib.linalg.gemm", "gemm_3xfp16"),
    "gemm_fp16_x9":      ("flashlib.linalg.gemm", "gemm_fp16_x9"),
    "gemm_fp16_x3_kahan":("flashlib.linalg.gemm", "gemm_fp16_x3_kahan"),
    "gemm_tf32_x6":      ("flashlib.linalg.gemm", "gemm_tf32_x6"),
    "gemm_ozaki2_int8":  ("flashlib.linalg.gemm", "gemm_ozaki2_int8"),
    # === eigh variants ===
    "eigh_cusolver":     ("flashlib.linalg.eigh", "eigh_cusolver"),
    "eigh_qdwh":         ("flashlib.linalg.eigh", "eigh_qdwh"),
    "eigh_qdwh_ns":      ("flashlib.linalg.eigh", "eigh_qdwh_ns"),
    "eigh_jacobi":       ("flashlib.linalg.eigh", "eigh_jacobi"),
    # === polar / matrix-sign variants ===
    "polar":             ("flashlib.linalg.polar", "polar"),
    "msign":             ("flashlib.linalg.polar", "msign"),
    "polar_qdwh_hybrid": ("flashlib.linalg.polar", "polar_qdwh_hybrid"),
    "polar_express":     ("flashlib.linalg.polar", "polar_express"),
    "polar_express_warm":("flashlib.linalg.polar", "polar_express_warm"),
    "polar_zolo":        ("flashlib.linalg.polar", "polar_zolo"),
    # === orthonormalization ===
    "cholqr2":           ("flashlib.linalg.orthonormalize", "cholqr2"),
    "split_basis":       ("flashlib.linalg.orthonormalize", "split_basis"),
}


def __getattr__(name: str):
    if name in _LAZY_ATTRS:
        modpath, attr = _LAZY_ATTRS[name]
        mod = importlib.import_module(modpath)
        value = getattr(mod, attr)
        globals()[name] = value
        return value
    raise AttributeError(f"module 'flashlib' has no attribute {name!r}")


def __dir__() -> list[str]:
    return sorted(set(globals()) | set(_LAZY_ATTRS) | {"info", "diagnose", "__version__"})


__all__ = [
    "diagnose", "info",
    *list(_LAZY_ATTRS.keys()),
]
