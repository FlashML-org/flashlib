"""KMeans NKI backend for AWS Trainium.

Re-exports the public Python wrappers. The ``@nki`` kernels in
``assign_kernel`` / ``update_kernel`` import ``neuronxcc`` at module load,
so they are imported lazily (only when a kernel actually runs); importing
this package on a non-Neuron box only needs torch + numpy.
"""
from flashlib.primitives.kmeans.nki.assign import (
    nki_assign_euclid,
    nki_available,
    nki_info,
    _try_init_nki,
)
from flashlib.primitives.kmeans.nki.kmeans import (
    nki_kmeans_Euclid,
    nki_kmeans_Euclid_distributed,
)

__all__ = [
    "nki_assign_euclid",
    "nki_kmeans_Euclid",
    "nki_kmeans_Euclid_distributed",
    "nki_available",
    "nki_info",
    "_try_init_nki",
]
