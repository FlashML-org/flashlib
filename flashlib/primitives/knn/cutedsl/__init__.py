"""knn cutedsl backend.

Re-exports the public FA3 entry point and a handful of utilities used by
the dispatcher / cost model. The only kernel here is the x^2-free fused
FA3 path (see :mod:`flashlib.primitives.knn.cutedsl.impl` for the design
rationale).
"""
from flashlib.primitives.knn.cutedsl.fused_kernel import (
    _cmp_swap_asc_packed_ptx,
    HopperFlashKnnFused,
)
from flashlib.primitives.knn.cutedsl.impl import (
    _CUTEDSL_AVAILABLE,
    _CUTE_IMPORT_ERROR,
    _try_init_cutedsl,
    _kernel_cache,
    _dlpack_cache,
    _cached_from_dlpack,
    cutedsl_available,
    cutedsl_flash_knn,
)
from flashlib.primitives.knn.cutedsl.blackwell_fused import (
    blackwell_available,
    blackwell_supported,
    blackwell_flash_knn,
    knn_build_cutedsl,
    knn_search_cutedsl,
)

__all__ = [
    "HopperFlashKnnFused",
    "cutedsl_available",
    "cutedsl_flash_knn",
    "blackwell_available",
    "blackwell_supported",
    "blackwell_flash_knn",
    "knn_build_cutedsl",
    "knn_search_cutedsl",
]
