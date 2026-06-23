"""kmeans cutedsl backend.

Re-exports the public Python wrappers from each component file.
``@cute.jit`` kernels stay private to their file.
"""
from flashlib.primitives.kmeans.cutedsl.assign import (
    _CUTEDSL_AVAILABLE,
    _CUTE_IMPORT_ERROR,
    _try_init_cutedsl,
    _pick_tile,
    _kernel_cache,
    _dlpack_cache,
    _cached_from_dlpack,
    cutedsl_assign_euclid,
    cutedsl_kmeans_Euclid,
    cutedsl_finalize,
    cutedsl_info,
)
from flashlib.primitives.kmeans.cutedsl.assign_kernel import (
    HopperFlashKmeansAssign,
)
from flashlib.primitives.kmeans.cutedsl.blackwell_assign import (
    blackwell_assign_available,
    blackwell_assign_euclid,
    blackwell_assign_supported,
)

__all__ = [
    "cutedsl_assign_euclid",
    "cutedsl_kmeans_Euclid",
    "cutedsl_finalize",
    "cutedsl_info",
    "HopperFlashKmeansAssign",
    "blackwell_assign_available",
    "blackwell_assign_euclid",
    "blackwell_assign_supported",
]
