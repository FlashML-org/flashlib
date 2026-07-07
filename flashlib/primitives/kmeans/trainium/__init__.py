"""kmeans Trainium (NKI) backend.

Re-exports the public Python wrappers. The ``@nki`` kernels in
``assign_kernel`` / ``update_kernel`` import ``neuronxcc`` at module load,
so they are imported lazily (only when a kernel actually runs); importing
this package on a non-Neuron box only needs torch + numpy.
"""
from flashlib.primitives.kmeans.trainium.assign import (
    trainium_assign_euclid,
    trainium_available,
    trainium_info,
    _try_init_nki,
)
from flashlib.primitives.kmeans.trainium.kmeans import trainium_kmeans_Euclid

__all__ = [
    "trainium_assign_euclid",
    "trainium_kmeans_Euclid",
    "trainium_available",
    "trainium_info",
    "_try_init_nki",
]
