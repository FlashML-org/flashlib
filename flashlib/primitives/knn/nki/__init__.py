"""KNN NKI backend for AWS Trainium.

Fused exhaustive k-NN on AWS Trainium: the cross term is a
``nc_matmul`` on the PE array and the top-k reduction uses the NeuronCore DVE
``max8`` / ``nc_match_replace8`` primitives, never materialising the
``(Nq, M)`` distance matrix. See :mod:`knn_kernel` (the NKI kernel) and
:mod:`knn` (the driver).

``neuronxcc`` is imported lazily (only when a kernel actually runs), so this
package imports cleanly on a CUDA-less box.
"""
from __future__ import annotations

from flashlib.primitives.knn.nki.knn import (
    nki_knn,
    nki_knn_available,
)

__all__ = ["nki_knn", "nki_knn_available"]
