"""Legacy flash-hdbscan: end-to-end HDBSCAN using flash_knn + Triton.

Pre-refactor pipeline (kept for parity testing):
  1. Core distances via ``flash_knn(X, k=min_samples)``.
  2. Dense MRD matrix via :func:`triton_pairwise_mrd`.
  3. MST via dense-graph Boruvka (:func:`flash_mst`).
  4. MST -> single-linkage tree on CPU.
  5. Condense + extract on CPU.
"""

import numpy as np
import torch

from flashlib.primitives.hdbscan.triton._tree_helpers import (
    _bfs_into_buffer,
    _bfs_from_hierarchy,
    _fast_condense_tree,
    _fast_compute_stability,
    _fast_get_clusters,
    _fast_label,
)

from flashlib.kernels.distance.triton import triton_pairwise_mrd
from flashlib.kernels.flash_mst import flash_mst
from flashlib.primitives.knn import flash_knn


def _fast_tree_to_labels(slt, min_cluster_size):
    """CPU pipeline: SLT -> condensed -> stability -> labels."""
    parents, children, lambdas, sizes = _fast_condense_tree(slt, min_cluster_size)
    cluster_ids, stab = _fast_compute_stability(parents, children, lambdas, sizes)
    num_points = slt.shape[0] + 1
    labels = _fast_get_clusters(parents, children, lambdas, sizes,
                                 cluster_ids, stab, num_points)
    return labels.astype(np.int32)


def _core_distances(X: torch.Tensor, min_samples: int,
                    *, tol=None) -> torch.Tensor:
    """k-th-nearest-neighbor distance per point (k = min_samples).

    ``tol`` is forwarded directly to :func:`flash_knn`.
    """
    dists_sq, _ = flash_knn(X[None], X[None], k=min_samples + 1, tol=tol)
    cd_sq = dists_sq[0, :, min_samples].clamp(min=0.0)
    return torch.sqrt(cd_sq)


def flash_hdbscan_legacy(X: torch.Tensor,
                         min_cluster_size: int = 25,
                         min_samples: int = 5,
                         *, tol=None):
    """Pre-refactor end-to-end Triton HDBSCAN.

    Args:
        X: (N, D) float32 CUDA tensor.
        min_cluster_size: min cluster size for tree condensing.
        min_samples: k for core distance.
        tol: residual tolerance forwarded to :func:`flash_knn` and
            :func:`triton_pairwise_mrd`. ``None`` (default) keeps both
            stages exact; pass ``tol=1e-3`` to opt into bf16 throughout.

    Returns:
        labels: (N,) numpy int32 -- cluster label per point (-1 = noise).
    """
    assert X.is_cuda
    N, D = X.shape

    core_dists = _core_distances(X, min_samples, tol=tol)
    MRD = triton_pairwise_mrd(X, core_dists, tol=tol)

    mst_gpu = flash_mst(MRD)
    mst = mst_gpu.cpu().numpy().astype(np.float64)

    del MRD
    torch.cuda.empty_cache()

    slt = _fast_label(mst)
    return _fast_tree_to_labels(slt, min_cluster_size)


def triton_hdbscan_mrd_legacy(X: torch.Tensor, min_samples: int = 5,
                              *, tol=None) -> torch.Tensor:
    """Compute mutual reachability distance matrix (kept for legacy bench)."""
    core = _core_distances(X, min_samples, tol=tol)
    return triton_pairwise_mrd(X, core, tol=tol)
