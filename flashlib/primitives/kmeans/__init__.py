"""K-Means primitive -- multi-backend (Triton + CuteDSL + Cake).

Public API
----------
Smart dispatcher (recommended):
    flash_kmeans(x, n_clusters, *, metric, max_iters, backend, variant)

Backend-explicit:
    flash_kmeans_triton, flash_kmeans_cutedsl, flash_kmeans_cake
    batch_kmeans_Euclid, batch_kmeans_Cosine, batch_kmeans_Dot   (Triton).
    cutedsl_assign_euclid, cutedsl_kmeans_Euclid                  (CuteDSL).
    flash_kmeans_assign_cake, select_flash_kmeans_route           (Cake).
    kmeans_largeN, kmeans_largeN_assign                           (CPU streaming).

Lower-level Triton kernels are exported for power users:
    euclid_assign_triton, cosine_assign_triton,
    triton_centroid_update_*, triton_centroid_finalize,
    triton_lloyd_centroid_step_euclid.
"""
from flashlib._lazy import lazy_attr
from flashlib.primitives.kmeans import cost
from flashlib.primitives.kmeans.impl import flash_kmeans
from flashlib.primitives.kmeans.large import kmeans_largeN, kmeans_largeN_assign


def flash_kmeans_triton(x, n_clusters, **kw):
    """Force the Triton backend (``flash_kmeans(..., backend='triton')``)."""
    return flash_kmeans(x, n_clusters, backend="triton", **kw)


def flash_kmeans_cutedsl(x, n_clusters, **kw):
    """Force the CuteDSL FA3-style fused-assign backend.

    Constraints: B=1, fp16/bf16 input, D in [16, 512] with D % 16 == 0,
    Hopper SM90 (H100/H200). Falls back to Triton if any constraint is
    violated or CUTLASS DSL is unavailable.
    """
    return flash_kmeans(x, n_clusters, backend="cutedsl", **kw)


def flash_kmeans_cake(x, n_clusters, **kw):
    """Force the Cake backend (``flash_kmeans(..., backend='cake')``)."""
    return flash_kmeans(x, n_clusters, backend="cake", **kw)


batch_kmeans_Euclid = lazy_attr(
    "flashlib.primitives.kmeans.triton",
    "batch_kmeans_Euclid",
)
batch_kmeans_Cosine = lazy_attr(
    "flashlib.primitives.kmeans.triton",
    "batch_kmeans_Cosine",
)
batch_kmeans_Dot = lazy_attr(
    "flashlib.primitives.kmeans.triton",
    "batch_kmeans_Dot",
)
euclid_assign_triton = lazy_attr(
    "flashlib.primitives.kmeans.triton",
    "euclid_assign_triton",
)
cosine_assign_triton = lazy_attr(
    "flashlib.primitives.kmeans.triton",
    "cosine_assign_triton",
)
triton_centroid_update_cosine = lazy_attr(
    "flashlib.primitives.kmeans.triton",
    "triton_centroid_update_cosine",
)
triton_centroid_update_euclid = lazy_attr(
    "flashlib.primitives.kmeans.triton",
    "triton_centroid_update_euclid",
)
triton_centroid_update_sorted_cosine = lazy_attr(
    "flashlib.primitives.kmeans.triton",
    "triton_centroid_update_sorted_cosine",
)
triton_centroid_update_sorted_euclid = lazy_attr(
    "flashlib.primitives.kmeans.triton",
    "triton_centroid_update_sorted_euclid",
)
triton_centroid_finalize = lazy_attr(
    "flashlib.primitives.kmeans.triton",
    "triton_centroid_finalize",
)
triton_lloyd_centroid_step_euclid = lazy_attr(
    "flashlib.primitives.kmeans.triton",
    "triton_lloyd_centroid_step_euclid",
)
cutedsl_assign_euclid = lazy_attr(
    "flashlib.primitives.kmeans.cutedsl",
    "cutedsl_assign_euclid",
)
cutedsl_kmeans_Euclid = lazy_attr(
    "flashlib.primitives.kmeans.cutedsl",
    "cutedsl_kmeans_Euclid",
)
flash_kmeans_assign_cake = lazy_attr(
    "flashlib.primitives.kmeans.cake",
    "flash_kmeans_assign",
)
select_flash_kmeans_route = lazy_attr(
    "flashlib.primitives.kmeans.cake",
    "select_flash_kmeans_route",
)


__all__ = [
    "flash_kmeans",
    "flash_kmeans_triton",
    "flash_kmeans_cutedsl",
    "flash_kmeans_cake",
    "flash_kmeans_assign_cake",
    "batch_kmeans_Euclid",
    "batch_kmeans_Cosine",
    "batch_kmeans_Dot",
    "euclid_assign_triton",
    "cosine_assign_triton",
    "triton_centroid_update_cosine",
    "triton_centroid_update_euclid",
    "triton_centroid_update_sorted_cosine",
    "triton_centroid_update_sorted_euclid",
    "triton_centroid_finalize",
    "triton_lloyd_centroid_step_euclid",
    "kmeans_largeN",
    "kmeans_largeN_assign",
    "cutedsl_assign_euclid",
    "cutedsl_kmeans_Euclid",
    "select_flash_kmeans_route",
    "cost",
]
