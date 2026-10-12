"""flash-hdbscan: end-to-end HDBSCAN using flash_knn + Triton kernels.

Two pipelines:
  • Default `flash_hdbscan` — dense MRD path:
      1. Core distances via flash_knn(X, k=min_samples+1)
      2. Dense MRD matrix (N×N bf16) via triton_pairwise_mrd
      3. MST via dense-graph Boruvka (flash_mst)
      4. MST → single-linkage tree (numba)
      5. Condense + extract (numba)
    Kept as the default for correctness compatibility on real-data
    workloads (BERTopic-style fine-grained density clustering on
    UMAP-reduced embeddings) where the bf16 dense MRD quantization
    happens to merge near-degenerate clusters cleanly.

  • Optional `flash_hdbscan_sparse` — sparse-kNN MST path:
      1. flash_knn(X_bf16, k=K+1) — single kNN call covers core + edges.
      2. Fused Triton MRD-edge kernel (`triton_fused_mrd_edges`).
      3. Sparse Boruvka MST on N×K edges (`sparse_mst.py`).
      4. Bridge synthesis (residual components → fake weight 1e10).
      5. SLT label + condense + extract on CPU (numba).
    Intended for very large N (≥150K) and high-D data; opt in via
    `prefer="sparse"` or call `flash_hdbscan_sparse(X, ...)` directly.

  • Auto dispatch (`flash_hdbscan(..., prefer="auto")`): picks the dense
    path unless N≥150K and D≥40 where sparse is decisively faster.
"""
import math
import numpy as np
import torch

from flashlib.kernels.distance.triton import (
    triton_pairwise_mrd, triton_fused_mrd_edges,
)
from flashlib.kernels.flash_mst import flash_mst
from flashlib.primitives.knn import flash_knn

from flashlib._optional import require_numba


def _fast_label(mst):
    """Single-linkage tree from sorted MST edges (numba CPU, imported on first use)."""
    require_numba("flash_hdbscan")
    from flashlib.primitives.hdbscan.triton._tree_helpers import _fast_label as _label

    return _label(mst)

def _fast_tree_to_labels(slt, min_cluster_size):
    """E2E numba pipeline: SLT -> condensed -> stability -> labels."""
    require_numba("flash_hdbscan")
    from flashlib.primitives.hdbscan.triton._tree_helpers import (
        _fast_compute_stability,
        _fast_condense_tree,
        _fast_get_clusters,
    )

    parents, children, lambdas, sizes = _fast_condense_tree(slt, min_cluster_size)
    cluster_ids, stab = _fast_compute_stability(parents, children, lambdas, sizes)
    num_points = slt.shape[0] + 1
    labels = _fast_get_clusters(parents, children, lambdas, sizes,
                                 cluster_ids, stab, num_points)
    return labels.astype(np.int32)

def _core_distances(X: torch.Tensor, min_samples: int,
                    *, tol=None) -> torch.Tensor:
    """k-th-nearest-neighbor distance per point (k = min_samples).

    ``tol`` is forwarded directly to :func:`flash_knn`; ``None`` (default)
    keeps the input dtype intact (exact). HDBSCAN can pass ``tol=1e-3``
    to opt into bf16 KNN storage for a ~3x speedup at the cost of ~1e-3
    distance noise.
    """
    dists_sq, _ = flash_knn(X[None], X[None], k=min_samples + 1, tol=tol)
    cd_sq = dists_sq[0, :, min_samples].clamp(min=0.0)
    return torch.sqrt(cd_sq)

def _flash_hdbscan_dense_impl(X: torch.Tensor,
                               min_cluster_size: int = 25,
                               min_samples: int = 5,
                               *, tol=None):
    """Dense MRD pipeline (the default)."""
    assert X.is_cuda
    N, D = X.shape

    core_dists = _core_distances(X, min_samples, tol=tol)
    MRD = triton_pairwise_mrd(X, core_dists, tol=tol)

    # Stage 3: MST via flash-mst (full GPU dense Boruvka)
    mst_gpu = flash_mst(MRD)        # (N-1, 3) float32 on GPU
    mst = mst_gpu.cpu().numpy().astype(np.float64)

    # Free MRD (largest tensor) before CPU work
    del MRD
    torch.cuda.empty_cache()

    # Stage 4: MST -> single-linkage tree -> condensed -> labels (numba CPU)
    slt = _fast_label(mst)
    return _fast_tree_to_labels(slt, min_cluster_size)

def _flash_knn_mrd_edges(X: torch.Tensor, K: int, min_samples: int,
                          *, tol=None):
    """Fused: flash_knn + triton_fused_mrd_edges in one stage.

    Args:
        X: (N, D) fp32 CUDA tensor.
        K: number of nearest neighbors (kNN edges per row).
        min_samples: index of NN whose distance is the core distance.
        tol: residual tolerance forwarded to :func:`flash_knn`. ``None``
            (default) keeps the input dtype intact.

    Returns:
        rows, cols: (N*K,) int32 directed edges (i -> its k-th NN).
        weights:    (N*K,) fp32 MRD weight per edge.
        core:       (N,)   fp32 core distances.
    """
    N, D = X.shape
    k_use = max(min_samples + 1, K + 1)
    dists_sq, idxs = flash_knn(X[None], X[None], k=k_use, tol=tol)
    cd_sq = dists_sq[0, :, min_samples].clamp(min=0.0)
    core = torch.sqrt(cd_sq)

    nn_dists_sq = dists_sq[0, :, 1:K + 1].contiguous()
    nn_idxs = idxs[0, :, 1:K + 1].to(torch.int32).contiguous()

    # Fused MRD per edge (Triton): max(sqrt(d²), core[i], core[partner])
    mrd = triton_fused_mrd_edges(nn_dists_sq, nn_idxs, core)

    rows = torch.arange(N, dtype=torch.int32, device=X.device).unsqueeze(1) \
        .expand(-1, K).contiguous().view(-1)
    cols = nn_idxs.view(-1)
    weights = mrd.view(-1)
    return rows, cols, weights, core

def flash_hdbscan_sparse(X: torch.Tensor,
                          min_cluster_size: int = 25,
                          min_samples: int = 5,
                          k: int = 32,
                          *, tol=None):
    """Sparse-kNN MST HDBSCAN. Faster than dense at large N (>=50K) but
    correctness can diverge from the dense path on fine-grained density
    clusterings.

    Args:
        X: (N, D) float32 CUDA tensor.
        min_cluster_size, min_samples: standard HDBSCAN params.
        k: kNN edges per row (default 32).
        tol: residual tolerance forwarded to :func:`flash_knn`. ``None``
            (default) keeps the input dtype intact (exact).
    """
    require_numba("flash_hdbscan_sparse")
    from flashlib.primitives.hdbscan.triton.sparse_mst import sparse_boruvka_mst

    assert X.is_cuda
    N, D = X.shape

    rows, cols, weights, core = _flash_knn_mrd_edges(
        X, K=k, min_samples=min_samples, tol=tol,
    )

    # Stage 3: symmetrize and run sparse Boruvka MST
    rows_sym = torch.cat([rows, cols])
    cols_sym = torch.cat([cols, rows])
    weights_sym = torch.cat([weights, weights])
    mst_src, mst_dst, mst_w, unique_roots, n_cc = sparse_boruvka_mst(
        rows_sym, cols_sym, weights_sym, N
    )

    # Stage 4: bridge synthesis between residual components.
    # Bridge weights only matter for cluster identity at the very top of the
    # dendrogram (above every real cluster's death lambda); using 1e10 yields
    # the same labels as the unknown true bridge MRD weight.
    if n_cc > 1:
        roots = unique_roots.to(torch.int32)
        n_extra = n_cc - 1
        extra_w = torch.full((n_extra,), 1e10, dtype=torch.float32, device=X.device)
        mst_src = torch.cat([mst_src, roots[:-1]])
        mst_dst = torch.cat([mst_dst, roots[1:]])
        mst_w = torch.cat([mst_w, extra_w])

    sort_idx = torch.argsort(mst_w)
    mst = torch.stack([mst_src[sort_idx].to(torch.float32),
                        mst_dst[sort_idx].to(torch.float32),
                        mst_w[sort_idx]], dim=1).cpu().numpy().astype(np.float64)

    # Stage 5+6: SLT + condense + extract (numba CPU)
    slt = _fast_label(mst)
    return _fast_tree_to_labels(slt, min_cluster_size)

def flash_hdbscan(X: torch.Tensor,
                  min_cluster_size: int = 25,
                  min_samples: int = 5,
                  *, approximate: bool = True,
                  prefer: str = "auto",
                  k: int = 32,
                  tol=None):
    """End-to-end Triton HDBSCAN — two algorithmic paths.

    approximate=True (sparse-kNN MST, default):
        • Sparse kNN-MRD edge list (k=32) instead of dense N×N MRD
        • Sparse Boruvka MST + sentinel-weight (1e10) bridges to splice
          disconnected components
        • What rapids-singlecell / BERTopic do for scaling
        • NOT bit-exact equivalent to dense HDBSCAN — relies on the kNN
          graph covering all intra-cluster edges. ARI≈1.0 vs cuML on
          well-clustered data (BERTopic 20news, scRNA PBMC, fraud);
          adversarial sparse / multi-component data may diverge.
    approximate=False (dense MRD, strict equivalent):
        • Dense N×N mutual-reachability matrix
        • Dense Boruvka MST on the complete graph
        • Bit-faithful to canonical HDBSCAN (McInnes 2017) up to
          tie-breaking; matches cuML's HDBSCAN algorithm='generic'.
        • Use this for unit tests / iso-result claims.

    Args:
        X: (N, D) float32 CUDA tensor.
        min_cluster_size: min cluster size for tree condensing.
        min_samples: k for core distance.
        approximate: if True (default), prefer the sparse path; auto
            dispatch falls back to dense at small N / low D where dense
            is faster anyway. If False, force the strict dense path.
        prefer: explicit dispatch override — "auto" / "sparse" / "dense".
            Only consulted when `approximate=True`.
        k: kNN size for sparse path (default 32; ignored on dense path).

    Returns:
        labels: (N,) numpy int32 — cluster label per point (-1 = noise).
    """
    require_numba("flash_hdbscan")
    assert X.is_cuda
    N, D = X.shape

    if not approximate:
        return _flash_hdbscan_dense_impl(X, min_cluster_size, min_samples, tol=tol)

    if prefer == "sparse":
        use_sparse = True
    elif prefer == "dense":
        use_sparse = False
    else:
        use_sparse = (N >= 150_000 and D >= 40)

    if use_sparse:
        return flash_hdbscan_sparse(X,
                                     min_cluster_size=min_cluster_size,
                                     min_samples=min_samples, k=k, tol=tol)
    return _flash_hdbscan_dense_impl(X, min_cluster_size, min_samples, tol=tol)

def triton_hdbscan_mrd(X: torch.Tensor, min_samples: int = 5,
                        *, tol: "float | None" = 1e-3) -> torch.Tensor:
    """Compute mutual reachability distance matrix (kept for legacy bench)."""
    core = _core_distances(X, min_samples, tol=tol)
    return triton_pairwise_mrd(X, core, tol=tol)
