"""KNN primitive -- fused (Triton + CuteDSL FA3) brute-force exact top-K.

Public API
----------
Single canonical entry point:

    flash_knn(x, c, k)                       -- ``(vals, idxs)`` (default)
                                                or just ``idxs`` when
                                                ``return_distances=False``.
                                                Routes Triton/CuteDSL by
                                                shape, never materialises
                                                an N x M cross.

Direct backend entry points (callable but normally accessed through
:func:`flash_knn`):

    flash_knn_triton(x, c, k)                -- Triton dispatcher (small-N
                                                M-split + large-N single-
                                                pass auto-picked from shape).
                                                Returns indices only.
    cutedsl_flash_knn(x, c, k)               -- Hopper FA3 fused (Hopper-
                                                only; opt-in). Returns
                                                indices only.

Torch fallback
--------------

    knn_torch_naive, knn_torch_chunked.
"""
from __future__ import annotations

from flashlib._lazy import lazy_attr
from flashlib.primitives.knn import cost
from flashlib.primitives.knn.impl import (
    flash_knn,
    flash_knn_dispatch,
    route_op_name,
)
from flashlib.primitives.knn.torch_fallback import knn_torch_naive, knn_torch_chunked
from flashlib.primitives.knn.triton.dispatch import flash_knn_triton


cutedsl_flash_knn = lazy_attr(
    "flashlib.primitives.knn.cutedsl",
    "cutedsl_flash_knn",
)

# Trainium (NKI) backend -- lazy so the package imports on a CUDA-less box and
# ``neuronxcc`` is only pulled in when the kernel actually runs.
trainium_knn = lazy_attr(
    "flashlib.primitives.knn.trainium",
    "trainium_knn",
)


__all__ = [
    "flash_knn",
    "flash_knn_dispatch",
    "flash_knn_triton",
    "cutedsl_flash_knn",
    "trainium_knn",
    "knn_torch_naive",
    "knn_torch_chunked",
    "cost",
    "route_op_name",
]
