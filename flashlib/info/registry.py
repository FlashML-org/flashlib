"""Registry of ops and their variants for the precision/performance Pareto API.

Two registries:

  _OP_REGISTRY     — op_name -> dotted module path of cost.py (or
                     ``module:funcname`` for a specific function).
                     Used by info.estimate() and info.recommend().

  _VARIANTS        — op_family -> list of variant op_names.
                     Used by info.variants() and info.pareto().

Op families that have multiple precision/performance variants list each
variant separately. Single-variant ops omit a families entry.
"""
from __future__ import annotations

_OP_REGISTRY: dict[str, str] = {
    # === algorithm primitives (smart dispatchers) =============================
    "kmeans":               "flashlib.primitives.kmeans.cost",
    "knn":                  "flashlib.primitives.knn.cost",
    "pca":                  "flashlib.primitives.pca.cost",
    "truncated_svd":        "flashlib.primitives.truncated_svd.cost",
    "linear_regression":    "flashlib.primitives.linear_regression.cost",
    "ridge":                "flashlib.primitives.ridge.cost",
    "logistic_regression":  "flashlib.primitives.logistic_regression.cost",
    "ivf_flat":             "flashlib.primitives.ivf_flat.cost",
    "ivf_pq":               "flashlib.primitives.ivf_pq.cost",
    "dbscan":               "flashlib.primitives.dbscan.cost",
    "hdbscan":              "flashlib.primitives.hdbscan.cost",
    "umap":                 "flashlib.primitives.umap.cost",
    "tsne":                 "flashlib.primitives.tsne.cost",
    "multinomial_nb":       "flashlib.primitives.multinomial_nb.cost",
    "random_forest":        "flashlib.primitives.random_forest.cost",
    "spectral_clustering":  "flashlib.primitives.spectral_clustering.cost",
    "standard_scaler":      "flashlib.primitives.standard_scaler.cost",

    # === per-backend primitive variants (Triton vs CuteDSL) ===================
    "kmeans_triton":            "flashlib.primitives.kmeans.cost:estimate_kmeans_triton",
    "kmeans_cutedsl":           "flashlib.primitives.kmeans.cost:estimate_kmeans_cutedsl",
    "knn_triton":               "flashlib.primitives.knn.cost:estimate_knn_triton",
    "knn_cutedsl_fa3":          "flashlib.primitives.knn.cost:estimate_knn_cutedsl_fa3",
    "ivf_flat_triton":          "flashlib.primitives.ivf_flat.cost:estimate_ivf_flat_triton",
    "ivf_pq_triton":            "flashlib.primitives.ivf_pq.cost:estimate_ivf_pq_triton",
    "pca_triton":               "flashlib.primitives.pca.cost:estimate_pca_triton",
    "pca_cutedsl":              "flashlib.primitives.pca.cost:estimate_pca_cutedsl",
    "truncated_svd_triton":     "flashlib.primitives.truncated_svd.cost:estimate_truncated_svd_triton",
    "truncated_svd_cutedsl":    "flashlib.primitives.truncated_svd.cost:estimate_truncated_svd_cutedsl",
    "linear_regression_triton":  "flashlib.primitives.linear_regression.cost:estimate_linear_regression_triton",
    "linear_regression_cutedsl": "flashlib.primitives.linear_regression.cost:estimate_linear_regression_cutedsl",
    "ridge_triton":             "flashlib.primitives.ridge.cost:estimate_ridge_triton",
    "ridge_cutedsl":            "flashlib.primitives.ridge.cost:estimate_ridge_cutedsl",
    "logistic_regression_triton":  "flashlib.primitives.logistic_regression.cost:estimate_logistic_regression_triton",
    "logistic_regression_cutedsl": "flashlib.primitives.logistic_regression.cost:estimate_logistic_regression_cutedsl",
    "dbscan_triton":            "flashlib.primitives.dbscan.cost:estimate_dbscan_triton",
    "dbscan_cutedsl":           "flashlib.primitives.dbscan.cost:estimate_dbscan_cutedsl",
    "hdbscan_triton":           "flashlib.primitives.hdbscan.cost:estimate_hdbscan_triton",
    "hdbscan_cutedsl":          "flashlib.primitives.hdbscan.cost:estimate_hdbscan_cutedsl",
    "umap_triton":              "flashlib.primitives.umap.cost:estimate_umap_triton",
    "umap_cutedsl":             "flashlib.primitives.umap.cost:estimate_umap_cutedsl",
    "tsne_triton":              "flashlib.primitives.tsne.cost:estimate_tsne_triton",
    "tsne_cutedsl":             "flashlib.primitives.tsne.cost:estimate_tsne_cutedsl",
    "multinomial_nb_triton":    "flashlib.primitives.multinomial_nb.cost:estimate_multinomial_nb_triton",
    "multinomial_nb_cutedsl":   "flashlib.primitives.multinomial_nb.cost:estimate_multinomial_nb_cutedsl",
    "random_forest_triton":     "flashlib.primitives.random_forest.cost:estimate_random_forest_triton",
    "random_forest_cutedsl":    "flashlib.primitives.random_forest.cost:estimate_random_forest_cutedsl",
    "spectral_clustering_triton":  "flashlib.primitives.spectral_clustering.cost:estimate_spectral_clustering_triton",
    "spectral_clustering_cutedsl": "flashlib.primitives.spectral_clustering.cost:estimate_spectral_clustering_cutedsl",
    "standard_scaler_triton":   "flashlib.primitives.standard_scaler.cost:estimate_standard_scaler_triton",
    "standard_scaler_cutedsl":  "flashlib.primitives.standard_scaler.cost:estimate_standard_scaler_cutedsl",

    # === linalg ===
    "cov_gemm":             "flashlib.linalg.cov_gemm.cost",
    "gram_gemm":            "flashlib.linalg.gram_gemm.cost",
    "ab_gemm":              "flashlib.linalg.ab_gemm.cost",
    # generic GEMM with precision variants — each variant module has its own estimate().
    "gemm":                 "flashlib.linalg.gemm",
    "gemm_fp32":            "flashlib.linalg.gemm.fp32",
    "gemm_tf32":            "flashlib.linalg.gemm.tf32",
    "gemm_3xtf32":          "flashlib.linalg.gemm.tf32_x3",
    "gemm_bf16":            "flashlib.linalg.gemm.bf16",
    "gemm_3xbf16":          "flashlib.linalg.gemm.bf16_x3",
    "gemm_fp16":            "flashlib.linalg.gemm.fp16",
    "gemm_3xfp16":          "flashlib.linalg.gemm.fp16_x3",
    "gemm_fp16_x9":         "flashlib.linalg.gemm.fp16_x9",
    "gemm_fp16_x3_kahan":   "flashlib.linalg.gemm.fp16_x3_kahan",
    "gemm_tf32_x6":         "flashlib.linalg.gemm.tf32_x6",
    "gemm_ozaki2_int8":     "flashlib.linalg.gemm.ozaki2_int8",
    "gemm_ozaki2_cute":     "flashlib.linalg.gemm.ozaki2_portable:estimate_cute",
    "gemm_ozaki2_triton":   "flashlib.linalg.gemm.ozaki2_portable:estimate_triton",
    # eigh — smart dispatcher + 5 variants (full + truncated).
    "eigh":                 "flashlib.linalg.eigh.cost",
    "eigh_cusolver":        "flashlib.linalg.eigh.cusolver",
    "eigh_qdwh":            "flashlib.linalg.eigh.cost:qdwh",
    "eigh_qdwh_ns":         "flashlib.linalg.eigh.cost:qdwh_ns",
    "eigh_jacobi":          "flashlib.linalg.eigh.jacobi",
    "eigh_halko":           "flashlib.linalg.eigh.halko",
    # polar (matrix sign).
    "polar":                "flashlib.linalg.polar.cost",
    "polar_qdwh_hybrid":    "flashlib.linalg.polar.cost:qdwh_hybrid",
    "polar_express":        "flashlib.linalg.polar.cost:polar_express",
    "polar_express_warm":   "flashlib.linalg.polar.cost:polar_express_warm",
    "polar_zolo":           "flashlib.linalg.polar.cost:zolo",
    # orthonormalization.
    "cholqr2":              "flashlib.linalg.orthonormalize.cost",

    # === shared kernels ===
    "pairwise_l2":          "flashlib.kernels.distance.cost",
    "connected_components": "flashlib.kernels.connected_components.cost",
    "flash_mst": "flashlib.kernels.flash_mst.cost",
}


# Variant families: which set of variants belongs to each op family.
# When the op_family entry exists, info.variants(<family>) walks each.
_VARIANTS: dict[str, list[str]] = {
    "gemm": [
        "gemm_fp32", "gemm_tf32", "gemm_3xtf32",
        "gemm_bf16", "gemm_3xbf16",
        "gemm_fp16", "gemm_3xfp16",
        "gemm_fp16_x9", "gemm_fp16_x3_kahan",
        "gemm_tf32_x6", "gemm_ozaki2_int8",
        "gemm_ozaki2_cute", "gemm_ozaki2_triton",
    ],
    "eigh": [
        "eigh_cusolver", "eigh_qdwh", "eigh_qdwh_ns", "eigh_jacobi", "eigh_halko",
    ],
    "polar": [
        "polar_qdwh_hybrid", "polar_express", "polar_express_warm", "polar_zolo",
    ],
    # Per-primitive Triton-vs-CuteDSL families.
    "kmeans":               ["kmeans_triton", "kmeans_cutedsl"],
    # Triton (default-routed) + CuteDSL FA3 (opt-in via backend='cutedsl').
    "knn":                  ["knn_triton", "knn_cutedsl_fa3"],
    # ivf_flat / ivf_pq are single-GPU-backend (Triton only; torch is a CPU
    # reference, not a Pareto variant) so they omit a variant-family entry.
    "pca":                  ["pca_triton", "pca_cutedsl"],
    "truncated_svd":        ["truncated_svd_triton", "truncated_svd_cutedsl"],
    "linear_regression":    ["linear_regression_triton", "linear_regression_cutedsl"],
    "ridge":                ["ridge_triton", "ridge_cutedsl"],
    "logistic_regression":  ["logistic_regression_triton", "logistic_regression_cutedsl"],
    "dbscan":               ["dbscan_triton", "dbscan_cutedsl"],
    "hdbscan":              ["hdbscan_triton", "hdbscan_cutedsl"],
    "umap":                 ["umap_triton", "umap_cutedsl"],
    "tsne":                 ["tsne_triton", "tsne_cutedsl"],
    "multinomial_nb":       ["multinomial_nb_triton", "multinomial_nb_cutedsl"],
    "random_forest":        ["random_forest_triton", "random_forest_cutedsl"],
    "spectral_clustering":  ["spectral_clustering_triton", "spectral_clustering_cutedsl"],
    "standard_scaler":      ["standard_scaler_triton", "standard_scaler_cutedsl"],
}


def list_ops() -> list[str]:
    return sorted(_OP_REGISTRY.keys())


def list_variant_families() -> list[str]:
    return sorted(_VARIANTS.keys())


def list_variants(op_family: str) -> list[str]:
    if op_family not in _VARIANTS:
        raise KeyError(
            f"{op_family!r} has no registered variant family. "
            f"Available variant families: {', '.join(sorted(_VARIANTS))}"
        )
    return list(_VARIANTS[op_family])


def resolve(op: str) -> str:
    if op not in _OP_REGISTRY:
        raise KeyError(
            f"unknown op {op!r}. Available: {', '.join(sorted(_OP_REGISTRY))}"
        )
    return _OP_REGISTRY[op]
