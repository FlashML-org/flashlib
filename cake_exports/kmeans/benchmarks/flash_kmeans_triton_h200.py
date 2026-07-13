from __future__ import annotations

from typing import Any

import torch
import triton
import triton.language as tl


def _next_pow2(v: int) -> int:
    if v <= 1:
        return 1
    return 1 << (v - 1).bit_length()


def _pad_d(dim: int) -> int:
    return max(16, _next_pow2(int(dim)))


def h200_small_d_config(n_points: int, n_clusters: int, dim: int, dtype: Any) -> dict[str, int]:
    """Flash-KMeans H200 fp16/bf16 small-D heuristic.

    This is copied from flash-kmeans' H200 heuristic and intentionally ignores
    the local GPU name. The benchmark uses it as the default Triton baseline
    even on B200/B300 so the comparison is stable and matches the MR audit.
    """

    if dtype not in {torch.float16, torch.bfloat16}:
        raise ValueError(f"H200 baseline in this benchmark expects fp16/bf16, got {dtype}")
    if dim > 512:
        raise ValueError(f"H200 small-D Triton baseline supports D <= 512, got D={dim}")

    block_n = 128
    block_k = 64
    num_warps = 4
    num_stages = 1

    if dim >= 512:
        block_n = 128
        block_k = 64
        num_warps = 8
        num_stages = 1
    elif dim >= 256:
        block_n = 128
        block_k = 64
        num_warps = 4
        num_stages = 2
    else:
        if n_clusters >= 4096:
            block_k = 128
            if dim >= 128:
                num_warps = 8
                num_stages = 2
            else:
                num_warps = 4
                num_stages = 4
        else:
            block_k = 64
            num_warps = 4
            num_stages = 1

    if dim <= 64 and n_clusters >= 4096:
        block_n = 64
        block_k = 128
        num_warps = 4
        num_stages = 4

    if n_points < 65536:
        block_n = 64

    return {
        "kernel": "triton_h200_small_d",
        "BLOCK_N": int(block_n),
        "BLOCK_K": int(block_k),
        "D_PAD": int(_pad_d(dim)),
        "num_warps": int(num_warps),
        "num_stages": int(num_stages),
    }


@triton.jit
def _euclid_assign_kernel(
    x_ptr,
    c_ptr,
    x_sq_ptr,
    c_sq_ptr,
    out_ptr,
    B: tl.constexpr,
    N: tl.constexpr,
    K: tl.constexpr,
    D: tl.constexpr,
    stride_x_b: tl.constexpr,
    stride_x_n: tl.constexpr,
    stride_x_d: tl.constexpr,
    stride_c_b: tl.constexpr,
    stride_c_k: tl.constexpr,
    stride_c_d: tl.constexpr,
    stride_xsq_b: tl.constexpr,
    stride_xsq_n: tl.constexpr,
    stride_csq_b: tl.constexpr,
    stride_csq_k: tl.constexpr,
    stride_out_b: tl.constexpr,
    stride_out_n: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    D_PAD: tl.constexpr,
):
    pid_n = tl.program_id(0)
    pid_b = tl.program_id(1).to(tl.int64)

    n_start = pid_n * BLOCK_N
    n_offsets = (n_start + tl.arange(0, BLOCK_N)).to(tl.int64)
    n_mask = n_offsets < N
    offs_d = tl.arange(0, D_PAD).to(tl.int64)
    d_mask = offs_d < D

    x_ptrs = (
        x_ptr
        + pid_b * stride_x_b
        + n_offsets[:, None] * stride_x_n
        + offs_d[None, :] * stride_x_d
    )
    x_tile = tl.load(x_ptrs, mask=n_mask[:, None] & d_mask[None, :], other=0.0)

    xsq_ptrs = x_sq_ptr + pid_b * stride_xsq_b + n_offsets * stride_xsq_n
    x_sq_tile = tl.load(xsq_ptrs, mask=n_mask, other=0.0).to(tl.float32)

    best_dist = tl.full((BLOCK_N,), 3.4e38, tl.float32)
    best_idx = tl.zeros((BLOCK_N,), tl.int32)

    for k_start in range(0, K, BLOCK_K):
        k_offsets = (k_start + tl.arange(0, BLOCK_K)).to(tl.int64)
        k_mask = k_offsets < K

        c_ptrs = (
            c_ptr
            + pid_b * stride_c_b
            + k_offsets[None, :] * stride_c_k
            + offs_d[:, None] * stride_c_d
        )
        c_tile = tl.load(c_ptrs, mask=k_mask[None, :] & d_mask[:, None], other=0.0)

        csq_ptrs = c_sq_ptr + pid_b * stride_csq_b + k_offsets * stride_csq_k
        c_sq_tile = tl.load(csq_ptrs, mask=k_mask, other=0.0).to(tl.float32)

        cross = tl.dot(x_tile, c_tile).to(tl.float32)
        dist = x_sq_tile[:, None] + c_sq_tile[None, :] - 2.0 * cross
        dist = tl.maximum(dist, 0.0)
        dist = tl.where(k_mask[None, :], dist, 3.4e38)

        curr_min = tl.min(dist, axis=1)
        curr_idx = tl.argmin(dist, axis=1)
        update = curr_min < best_dist
        best_dist = tl.where(update, curr_min, best_dist)
        best_idx = tl.where(update, k_start + curr_idx, best_idx)

    out_ptrs = out_ptr + pid_b * stride_out_b + n_offsets * stride_out_n
    tl.store(out_ptrs, best_idx, mask=n_mask)


def euclid_assign_triton_h200(
    x: torch.Tensor,
    centroids: torch.Tensor,
    x_sq: torch.Tensor,
    c_sq: torch.Tensor,
    *,
    out: torch.Tensor | None = None,
) -> tuple[torch.Tensor, dict[str, int]]:
    """Run Flash-KMeans' Triton Euclidean assign with forced H200 heuristic."""

    if x.ndim != 3 or centroids.ndim != 3:
        raise ValueError("x and centroids must have shapes [B, N, D] and [B, K, D]")
    if x.dtype not in {torch.float16, torch.bfloat16}:
        raise ValueError(f"Triton H200 baseline expects fp16/bf16 inputs, got {x.dtype}")
    if x.device != centroids.device:
        raise ValueError("x and centroids must be on the same device")
    if not x.is_cuda or not centroids.is_cuda:
        raise ValueError("x and centroids must be CUDA tensors")
    if not x.is_contiguous() or not centroids.is_contiguous():
        raise ValueError("x and centroids must be contiguous")

    bsz, n_points, dim = (int(v) for v in x.shape)
    n_clusters = int(centroids.shape[1])
    if int(centroids.shape[0]) != bsz or int(centroids.shape[2]) != dim:
        raise ValueError("x and centroids must have matching B and D dimensions")
    if out is None:
        out = torch.empty((bsz, n_points), dtype=torch.int32, device=x.device)
    if tuple(out.shape) != (bsz, n_points) or out.dtype is not torch.int32:
        raise ValueError("out must have shape [B, N] and dtype torch.int32")

    cfg = h200_small_d_config(n_points, n_clusters, dim, x.dtype)
    stride_x_b, stride_x_n, stride_x_d = x.stride()
    stride_c_b, stride_c_k, stride_c_d = centroids.stride()
    stride_xsq_b, stride_xsq_n = x_sq.stride()
    stride_csq_b, stride_csq_k = c_sq.stride()
    stride_out_b, stride_out_n = out.stride()

    grid = lambda meta: (triton.cdiv(n_points, meta["BLOCK_N"]), bsz)
    _euclid_assign_kernel[grid](
        x,
        centroids,
        x_sq,
        c_sq,
        out,
        bsz,
        n_points,
        n_clusters,
        dim,
        stride_x_b,
        stride_x_n,
        stride_x_d,
        stride_c_b,
        stride_c_k,
        stride_c_d,
        stride_xsq_b,
        stride_xsq_n,
        stride_csq_b,
        stride_csq_k,
        stride_out_b,
        stride_out_n,
        BLOCK_N=cfg["BLOCK_N"],
        BLOCK_K=cfg["BLOCK_K"],
        D_PAD=cfg["D_PAD"],
        num_warps=cfg["num_warps"],
        num_stages=cfg["num_stages"],
    )
    return out, cfg
