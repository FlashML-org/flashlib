"""Single-launch CuTeDSL bf16x3 — production path.

The deadlock at tile (128, 256+) cooperative was root-caused: with 4 SMEM
operands instead of the upstream's 2, `_compute_stages` computed
ab_stage=1 (no producer/consumer overlap), and the degenerate single-stage
PipelineTmaAsync hangs on Hopper. Fix: shrink epi_stage 4→2 to free the
SMEM needed for 2 ab_stages, and refuse to ship anything below 2 stages.

Best config at 8192³ (multi-trial): tile (128, 128), cluster (1, 2),
swizzle 8 → **263 TF median / 279 TF max effective FP32**, at the 280 TF
advertised-peak target (= 85% of 989/3 BF16 dense peak). Mean rel err
~1.9e-4, comparable to other bf16x3 emulation paths.

Algorithm (same as all our bf16x3 paths):
    a ≈ a_hi + a_lo, b ≈ b_hi + b_lo (all BF16), output FP32.
    out = a_hi·b_hi + a_hi·b_lo + a_lo·b_hi   (Markidis-style, drops lo·lo)
"""

from __future__ import annotations

from functools import lru_cache
from typing import Tuple

import torch

import cuda.bindings.driver as cuda

import cutlass
import cutlass.cute as cute
import cutlass.torch as cutlass_torch
from cutlass.cute.runtime import from_dlpack

from flashlib.linalg.gemm.triton.split import split_fp32_bf16_pair
from flashlib.linalg.gemm.cutedsl.lib.hopper_gemm_bf16x3 import (
    HopperWgmmaGemmBf16x3PersistentKernel,
)


_TileShape = Tuple[int, int]
_ClusterShape = Tuple[int, int]


@lru_cache(maxsize=16)
def _compile_for(
    M: int, N: int, K: int, L: int,
    tile_shape_mn: _TileShape,
    cluster_shape_mn: _ClusterShape,
    swizzle_size: int,
    raster_along_m: bool,
):
    """Compile the patched kernel for these shape/cluster knobs (cached)."""
    # Build 4 distinct tensors so cute.compile doesn't alias them.
    a_hi_cpu = cutlass_torch.matrix(L, M, K, False, cutlass.BFloat16)
    a_lo_cpu = cutlass_torch.matrix(L, M, K, False, cutlass.BFloat16)
    b_hi_cpu = cutlass_torch.matrix(L, N, K, False, cutlass.BFloat16)
    b_lo_cpu = cutlass_torch.matrix(L, N, K, False, cutlass.BFloat16)
    c_cpu = cutlass_torch.matrix(L, M, N, False, cutlass.Float32)
    a_hi_t, _ = cutlass_torch.cute_tensor_like(a_hi_cpu, cutlass.BFloat16, is_dynamic_layout=True, assumed_align=16)
    a_lo_t, _ = cutlass_torch.cute_tensor_like(a_lo_cpu, cutlass.BFloat16, is_dynamic_layout=True, assumed_align=16)
    b_hi_t, _ = cutlass_torch.cute_tensor_like(b_hi_cpu, cutlass.BFloat16, is_dynamic_layout=True, assumed_align=16)
    b_lo_t, _ = cutlass_torch.cute_tensor_like(b_lo_cpu, cutlass.BFloat16, is_dynamic_layout=True, assumed_align=16)
    c_t, _ = cutlass_torch.cute_tensor_like(c_cpu, cutlass.Float32, is_dynamic_layout=True, assumed_align=16)
    gemm = HopperWgmmaGemmBf16x3PersistentKernel(
        cutlass.Float32, tile_shape_mn, cluster_shape_mn, swizzle_size, raster_along_m
    )
    hw = cutlass.utils.HardwareInfo()
    max_active_clusters = hw.get_max_active_clusters(cluster_shape_mn[0] * cluster_shape_mn[1])
    stream = cuda.CUstream(torch.cuda.current_stream().cuda_stream)
    compiled = cute.compile(gemm, a_hi_t, a_lo_t, b_hi_t, b_lo_t, c_t,
                              max_active_clusters, stream)
    return compiled


def _wrap(t: torch.Tensor, dtype):
    ct = from_dlpack(t, assumed_align=16)
    ct.element_type = dtype
    leading_dim = cutlass_torch.get_leading_dim(t)
    return ct.mark_layout_dynamic(leading_dim=leading_dim)


def _round_up(x: int, m: int) -> int:
    return (x + m - 1) // m * m


def matmul_bf16x3_cute_fused_presplit(
    a_hi: torch.Tensor, a_lo: torch.Tensor,
    b_hi: torch.Tensor, b_lo: torch.Tensor,
    tile_shape_mn: _TileShape = (128, 256),
    cluster_shape_mn: _ClusterShape = (2, 1),
    swizzle_size: int = 8,
) -> torch.Tensor:
    """Single-launch BF16x3. Pre-split BF16 components in, FP32 out."""
    assert a_hi.dtype == torch.bfloat16 and b_hi.dtype == torch.bfloat16
    assert a_hi.shape == a_lo.shape and b_hi.shape == b_lo.shape
    assert a_hi.shape[1] == b_hi.shape[1]
    M, K = a_hi.shape
    N, _ = b_hi.shape

    # TMA needs 16-byte-aligned rows: K % 8 for the bf16 operands, N % 4 for the fp32 output.
    K_pad, N_pad = _round_up(K, 8), _round_up(N, 4)
    if (K_pad, N_pad) != (K, N):
        pad = torch.nn.functional.pad
        a_hi, a_lo = pad(a_hi, (0, K_pad - K)), pad(a_lo, (0, K_pad - K))
        b_hi, b_lo = pad(b_hi, (0, K_pad - K, 0, N_pad - N)), pad(b_lo, (0, K_pad - K, 0, N_pad - N))
        out = matmul_bf16x3_cute_fused_presplit(
            a_hi, a_lo, b_hi, b_lo, tile_shape_mn, cluster_shape_mn, swizzle_size,
        )
        return out[:, :N].contiguous()

    a_hi3 = a_hi.unsqueeze(-1).contiguous()
    a_lo3 = a_lo.unsqueeze(-1).contiguous()
    b_hi3 = b_hi.unsqueeze(-1).contiguous()
    b_lo3 = b_lo.unsqueeze(-1).contiguous()
    c = torch.empty((M, N, 1), device=a_hi.device, dtype=torch.float32)

    compiled = _compile_for(M, N, K, 1, tile_shape_mn, cluster_shape_mn,
                              swizzle_size, raster_along_m=True)
    stream = cuda.CUstream(torch.cuda.current_stream().cuda_stream)

    a_hi_ct = _wrap(a_hi3, cutlass.BFloat16)
    a_lo_ct = _wrap(a_lo3, cutlass.BFloat16)
    b_hi_ct = _wrap(b_hi3, cutlass.BFloat16)
    b_lo_ct = _wrap(b_lo3, cutlass.BFloat16)
    c_ct = _wrap(c, cutlass.Float32)

    compiled(a_hi_ct, a_lo_ct, b_hi_ct, b_lo_ct, c_ct, stream)
    return c.squeeze(-1)


def matmul_bf16x3_cute_fused(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """Convenience: split FP32 → BF16 pairs, then single CuTeDSL launch."""
    assert a.dtype == torch.float32 and b.dtype == torch.float32
    a_hi, a_lo = split_fp32_bf16_pair(a)
    b_hi, b_lo = split_fp32_bf16_pair(b)
    return matmul_bf16x3_cute_fused_presplit(a_hi, a_lo, b_hi, b_lo)
