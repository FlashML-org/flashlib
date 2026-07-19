"""Mutual-Reachability-Distance kernels (for HDBSCAN).

MRD[i,j] = max(core_dist[i], core_dist[j], dist(i,j)).
Fused per-edge variant for the sparse-kNN HDBSCAN path.
"""
import torch
import triton
import triton.language as tl

from flashlib.kernels.distance.triton._common import _round_to_bucket



# =============================================================================
# Kernel 2c: Pairwise Mutual Reachability Distance (for HDBSCAN)
# MRD[i,j] = max(core_dist[i], core_dist[j], dist(i,j))
# =============================================================================

_PAIRWISE_MRD_CONFIGS = [
    triton.Config({"BLOCK_I": 64, "BLOCK_J": 64, "BLOCK_D": 32}, num_stages=2, num_warps=4),
    triton.Config({"BLOCK_I": 128, "BLOCK_J": 64, "BLOCK_D": 32}, num_stages=2, num_warps=8),
    triton.Config({"BLOCK_I": 64, "BLOCK_J": 128, "BLOCK_D": 64}, num_stages=2, num_warps=8),
    triton.Config({"BLOCK_I": 128, "BLOCK_J": 128, "BLOCK_D": 32}, num_stages=1, num_warps=8),
]


@triton.autotune(configs=_PAIRWISE_MRD_CONFIGS, key=["N_KEY", "D_KEY"])
@triton.jit
def _pairwise_mrd_kernel(
    X_ptr, X_sq_ptr, CORE_ptr, OUT_ptr,
    N, D,
    stride_xn, stride_xd, stride_on, stride_om,
    N_KEY: tl.constexpr, D_KEY: tl.constexpr,
    BLOCK_I: tl.constexpr, BLOCK_J: tl.constexpr, BLOCK_D: tl.constexpr,
):
    """MRD[i,j] = max(core[i], core[j], sqrt(||xi-xj||^2)).

    Symmetric optimization: tiles entirely inside the strict lower triangle
    are skipped (no tl.dot at all); each surviving tile stores its
    upper-triangle (i <= j) elements directly and mirrors its strictly-upper
    (i < j) elements to OUT[j, i]. All triangle tests are element-wise on
    row/col coordinates so rectangular tiles (BLOCK_I != BLOCK_J) stay
    exact — comparing pid_i against pid_j is only valid for square tiles.
    """
    pid_i = tl.program_id(0)
    pid_j = tl.program_id(1)

    i_start = pid_i * BLOCK_I
    j_start = pid_j * BLOCK_J

    # Skip tiles entirely inside the strict lower triangle. A tile is fully
    # below the diagonal iff its largest column (j_start + BLOCK_J - 1) is
    # smaller than its smallest row (i_start).
    if j_start + BLOCK_J <= i_start:
        return

    i_offs = (i_start + tl.arange(0, BLOCK_I)).to(tl.int64)
    j_offs = (j_start + tl.arange(0, BLOCK_J)).to(tl.int64)
    i_mask = i_offs < N
    j_mask = j_offs < N

    xi_sq = tl.load(X_sq_ptr + i_offs, mask=i_mask, other=0.0)
    xj_sq = tl.load(X_sq_ptr + j_offs, mask=j_mask, other=0.0)

    core_i = tl.load(CORE_ptr + i_offs, mask=i_mask, other=0.0)
    core_j = tl.load(CORE_ptr + j_offs, mask=j_mask, other=0.0)

    cross = tl.zeros((BLOCK_I, BLOCK_J), dtype=tl.float32)
    for d_start in tl.range(0, D_KEY, BLOCK_D, num_stages=2):
        d_offs = d_start + tl.arange(0, BLOCK_D)
        d_mask = d_offs < D
        xi = tl.load(X_ptr + i_offs[:, None] * stride_xn + d_offs[None, :] * stride_xd,
                      mask=i_mask[:, None] & d_mask[None, :], other=0.0)
        xj = tl.load(X_ptr + j_offs[:, None] * stride_xn + d_offs[None, :] * stride_xd,
                      mask=j_mask[:, None] & d_mask[None, :], other=0.0)
        cross += tl.dot(xi, tl.trans(xj))

    dist_sq = xi_sq[:, None] + xj_sq[None, :] - 2.0 * cross
    dist = tl.sqrt(tl.maximum(dist_sq, 0.0))

    # MRD = max(core_i, core_j, dist)
    mrd = tl.maximum(tl.maximum(core_i[:, None], core_j[None, :]), dist)

    out_ptrs = OUT_ptr + i_offs[:, None] * stride_on + j_offs[None, :] * stride_om

    if j_start >= i_start + BLOCK_I:
        # Fast path: tile lies entirely in the strict upper triangle (every
        # cell has i < j), so both the direct store and the mirror need only
        # bounds masks. This covers all but the O(N / BLOCK) diagonal-band
        # tiles, so the kernel stays store-bandwidth-optimal.
        tl.store(out_ptrs, mrd, mask=i_mask[:, None] & j_mask[None, :])
        out_ptrs_t = (OUT_ptr + j_offs[:, None] * stride_on
                      + i_offs[None, :] * stride_om)
        # Transpose mrd: (BLOCK_I, BLOCK_J) -> (BLOCK_J, BLOCK_I)
        tl.store(out_ptrs_t, tl.trans(mrd), mask=j_mask[:, None] & i_mask[None, :])
    else:
        # Diagonal-straddling tile: store the i <= j cells directly and
        # mirror the i < j cells to OUT[j, i]. Together every cell of the
        # band is written exactly once — no diagonal double-write and no
        # duplicate/racing writes between neighbouring tiles (rectangular
        # configs always produce straddling tiles, so tile-level pid checks
        # cannot express this).
        upper = i_offs[:, None] <= j_offs[None, :]
        tl.store(out_ptrs, mrd, mask=i_mask[:, None] & j_mask[None, :] & upper)
        out_ptrs_t = (OUT_ptr + j_offs[:, None] * stride_on
                      + i_offs[None, :] * stride_om)
        strict_upper_t = i_offs[None, :] < j_offs[:, None]
        tl.store(out_ptrs_t, tl.trans(mrd),
                 mask=j_mask[:, None] & i_mask[None, :] & strict_upper_t)


def triton_pairwise_mrd(X: torch.Tensor, core_dists: torch.Tensor,
                         *, tol: "float | None" = None,
                         dtype: "torch.dtype | None" = None) -> torch.Tensor:
    """Compute pairwise mutual reachability distance.

    Args:
        X: (N, D) float32 tensor.
        core_dists: (N,) core distances.
        tol: residual tolerance for the output. ``None`` -> fp32; any
            positive ``tol`` -> bf16 (2x memory savings for the downstream
            HBM-bound MST argmin scan; edges with <1% relative weight diff
            may flip in tie-breaking). Mirrors the precision-by-tol
            convention used elsewhere in flashlib.
        dtype: explicit override for the output dtype. Kept for backward
            compatibility; new callers should pass ``tol`` instead.

    Returns:
        (N, N) MRD matrix.
    """
    assert X.is_cuda and X.ndim == 2
    N, D = X.shape
    X = X.contiguous()
    core_dists = core_dists.contiguous()

    if dtype is None:
        out_dtype = torch.bfloat16 if (tol is not None and tol > 0) else torch.float32
    else:
        out_dtype = dtype

    x_sq = (X * X).sum(dim=1)
    out = torch.empty(N, N, device=X.device, dtype=out_dtype)

    D_KEY = _round_to_bucket(D)
    grid = lambda META: (
        triton.cdiv(N, META["BLOCK_I"]),
        triton.cdiv(N, META["BLOCK_J"]),
    )
    _pairwise_mrd_kernel[grid](
        X, x_sq, core_dists, out,
        N, D,
        X.stride(0), X.stride(1),
        out.stride(0), out.stride(1),
        N_KEY=_round_to_bucket(N), D_KEY=D_KEY,
    )
    return out


# =============================================================================
# Kernel 2d: Fused per-edge MRD transform (for sparse-knn HDBSCAN path)
# For each kNN edge (i, j) with squared distance d_sq, compute
#   mrd[i, k] = max(sqrt(d_sq), core[i], core[j])
# This replaces 4 separate torch ops (sqrt + 2 gathers + 2 maximums) with a
# single fused Triton kernel. Single-pass over (N, K) edges.
# =============================================================================

@triton.jit
def _fused_mrd_edges_kernel(
    NN_DISTS_SQ_ptr,    # (N, K) fp32 — squared L2 from flash_knn
    NN_IDXS_ptr,        # (N, K) int32 — partner indices
    CORE_ptr,           # (N,) fp32 — core distance per row (sqrt of k-th NN squared)
    OUT_MRD_ptr,        # (N, K) fp32 — output MRD per edge
    N, K: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    pid = tl.program_id(0)
    n_offs = pid * BLOCK_N + tl.arange(0, BLOCK_N)
    n_mask = n_offs < N

    core_i = tl.load(CORE_ptr + n_offs, mask=n_mask, other=0.0)

    k_offs = tl.arange(0, K)
    base_offs = n_offs[:, None] * K + k_offs[None, :]
    dist_sq = tl.load(NN_DISTS_SQ_ptr + base_offs,
                      mask=n_mask[:, None], other=0.0)
    idxs = tl.load(NN_IDXS_ptr + base_offs,
                   mask=n_mask[:, None], other=0)
    d = tl.sqrt(tl.maximum(dist_sq, 0.0))

    core_j = tl.load(CORE_ptr + idxs.to(tl.int64),
                     mask=n_mask[:, None], other=0.0)

    mrd = tl.maximum(tl.maximum(d, core_i[:, None]), core_j)

    tl.store(OUT_MRD_ptr + base_offs, mrd, mask=n_mask[:, None])


def triton_fused_mrd_edges(nn_dists_sq: torch.Tensor,
                            nn_idxs: torch.Tensor,
                            core: torch.Tensor) -> torch.Tensor:
    """Fused MRD edge weight: mrd[i,k] = max(sqrt(d), core[i], core[partner]).

    Args:
        nn_dists_sq: (N, K) fp32 — squared L2 distances per kNN edge.
        nn_idxs:     (N, K) int32 — partner indices per edge.
        core:        (N,)   fp32 — core distances (sqrt of k-th NN).

    Returns:
        (N, K) fp32 MRD per edge.
    """
    assert nn_dists_sq.is_cuda and nn_dists_sq.dtype == torch.float32
    assert nn_idxs.is_cuda and nn_idxs.dtype == torch.int32
    assert core.is_cuda and core.dtype == torch.float32
    N, K = nn_dists_sq.shape
    out = torch.empty(N, K, dtype=torch.float32, device=core.device)
    BLOCK_N = 64
    grid = (triton.cdiv(N, BLOCK_N),)
    _fused_mrd_edges_kernel[grid](
        nn_dists_sq.contiguous(), nn_idxs.contiguous(), core.contiguous(), out,
        N, K=K, BLOCK_N=BLOCK_N, num_warps=4,
    )
    return out

