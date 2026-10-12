"""FP16x9 / BF16x9 fused: 3 components per operand × 9 partial products.

Single-launch CuTeDSL kernel emits 9 cute.gemm calls per K-block, with 6
TMA loads (3 A components + 3 B components) sharing one barrier. Mantissa
precision: 3 × component_mantissa bits. With FP16 components → ~30 bits.

Pareto position (compared with our existing modes):
- bf16x3 fused: ~14-bit / ~280 TF (FP32 emul, fast but coarse)
- **fp16x9 fused: ~25-30 bit / ~110 TF** — fills the 14→17 bit gap
- cuBLAS FP32 native: ~17-bit / ~50 TF (precise but slow)
- cuBLAS FP64 native: 52-bit / ~56 TF (full FP64 precision)
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

from flashlib.linalg.gemm.cutedsl.lib.hopper_gemm_x9 import HopperWgmmaGemmX9PersistentKernel


_TileShape = Tuple[int, int]
_ClusterShape = Tuple[int, int]


@lru_cache(maxsize=8)
def _compile_for(M: int, N: int, K: int, L: int,
                  dtype_name: str,
                  tile_shape_mn: _TileShape,
                  cluster_shape_mn: _ClusterShape,
                  swizzle_size: int):
    """Compile the patched kernel. dtype_name in {'fp16', 'bf16'}."""
    dt = cutlass.Float16 if dtype_name == "fp16" else cutlass.BFloat16
    a_cpu = [cutlass_torch.matrix(L, M, K, False, dt) for _ in range(3)]
    b_cpu = [cutlass_torch.matrix(L, N, K, False, dt) for _ in range(3)]
    c_cpu = cutlass_torch.matrix(L, M, N, False, cutlass.Float32)
    a_t = [cutlass_torch.cute_tensor_like(t, dt, is_dynamic_layout=True,
                                            assumed_align=16)[0] for t in a_cpu]
    b_t = [cutlass_torch.cute_tensor_like(t, dt, is_dynamic_layout=True,
                                            assumed_align=16)[0] for t in b_cpu]
    c_t, _ = cutlass_torch.cute_tensor_like(c_cpu, cutlass.Float32,
                                              is_dynamic_layout=True, assumed_align=16)
    gemm = HopperWgmmaGemmX9PersistentKernel(
        cutlass.Float32, tile_shape_mn, cluster_shape_mn, swizzle_size, True
    )
    hw = cutlass.utils.HardwareInfo()
    max_active_clusters = hw.get_max_active_clusters(cluster_shape_mn[0] * cluster_shape_mn[1])
    stream = cuda.CUstream(torch.cuda.current_stream().cuda_stream)
    compiled = cute.compile(gemm, *a_t, *b_t, c_t, max_active_clusters, stream)
    return compiled


def _wrap(t: torch.Tensor, dtype):
    ct = from_dlpack(t, assumed_align=16)
    ct.element_type = dtype
    leading_dim = cutlass_torch.get_leading_dim(t)
    return ct.mark_layout_dynamic(leading_dim=leading_dim)


def _round_up(x: int, m: int) -> int:
    return (x + m - 1) // m * m


def _split_3comp_fp16(x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """fp32/fp64 → (fp16_0, fp16_1, fp16_2) where x ≈ x0 + x1 + x2 with
    each component representable in fp16 directly (no scaling — relies
    on residuals having decreasing magnitude).
    """
    fp16_max = 65504.0
    work = x.float() if x.dtype == torch.float64 else x
    work = work.clamp(-fp16_max, fp16_max).contiguous()
    x0 = work.to(torch.float16)
    r1 = work - x0.float()
    x1 = r1.clamp(-fp16_max, fp16_max).to(torch.float16)
    r2 = r1 - x1.float()
    x2 = r2.clamp(-fp16_max, fp16_max).to(torch.float16)
    return x0, x1, x2


def _split_3comp_bf16(x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    work = x.float() if x.dtype == torch.float64 else x
    x0 = work.to(torch.bfloat16)
    r1 = work - x0.float()
    x1 = r1.to(torch.bfloat16)
    r2 = r1 - x1.float()
    x2 = r2.to(torch.bfloat16)
    return x0, x1, x2


def matmul_x9_cute_fused_presplit(
    a0, a1, a2, b0, b1, b2,
    tile_shape_mn: _TileShape = (128, 128),
    cluster_shape_mn: _ClusterShape = (1, 2),
    swizzle_size: int = 8,
) -> torch.Tensor:
    """Inputs: 6 width-16 tensors (3 components per operand). Output: FP32."""
    assert a0.dtype == b0.dtype and a0.dtype in (torch.float16, torch.bfloat16)
    M, K = a0.shape
    N, _ = b0.shape

    # TMA needs 16-byte-aligned rows: K % 8 for the 16-bit operands, N % 4 for the fp32 output.
    K_pad, N_pad = _round_up(K, 8), _round_up(N, 4)
    if (K_pad, N_pad) != (K, N):
        pad = torch.nn.functional.pad
        a_p = [pad(t, (0, K_pad - K)) for t in (a0, a1, a2)]
        b_p = [pad(t, (0, K_pad - K, 0, N_pad - N)) for t in (b0, b1, b2)]
        out = matmul_x9_cute_fused_presplit(
            *a_p, *b_p, tile_shape_mn=tile_shape_mn,
            cluster_shape_mn=cluster_shape_mn, swizzle_size=swizzle_size,
        )
        return out[:, :N].contiguous()

    a3 = [t.unsqueeze(-1).contiguous() for t in (a0, a1, a2)]
    b3 = [t.unsqueeze(-1).contiguous() for t in (b0, b1, b2)]
    c = torch.empty((M, N, 1), device=a0.device, dtype=torch.float32)

    dtype_name = "fp16" if a0.dtype == torch.float16 else "bf16"
    cute_dt = cutlass.Float16 if dtype_name == "fp16" else cutlass.BFloat16
    compiled = _compile_for(M, N, K, 1, dtype_name, tile_shape_mn,
                              cluster_shape_mn, swizzle_size)
    stream = cuda.CUstream(torch.cuda.current_stream().cuda_stream)

    a_ct = [_wrap(t, cute_dt) for t in a3]
    b_ct = [_wrap(t, cute_dt) for t in b3]
    c_ct = _wrap(c, cutlass.Float32)
    compiled(*a_ct, *b_ct, c_ct, stream)
    return c.squeeze(-1)


def matmul_fp16x9_cute_fused(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """FP32 in, FP32 out — 3 FP16 components per operand, 9 partials.
    Inputs must fit in FP16 range (|x| ≤ 65504), else clamped.
    """
    assert a.dtype == torch.float32 and b.dtype == torch.float32
    a0, a1, a2 = _split_3comp_fp16(a)
    b0, b1, b2 = _split_3comp_fp16(b)
    return matmul_x9_cute_fused_presplit(a0, a1, a2, b0, b1, b2)


def matmul_fp16x9_cute_fused_fp64(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """FP64 in, FP64 out via FP16x9. Inputs must fit in FP16 range."""
    assert a.dtype == torch.float64 and b.dtype == torch.float64
    a0, a1, a2 = _split_3comp_fp16(a)
    b0, b1, b2 = _split_3comp_fp16(b)
    out = matmul_x9_cute_fused_presplit(a0, a1, a2, b0, b1, b2)
    return out.to(torch.float64)


def matmul_bf16x9_cute_fused(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """FP32 in, FP32 out — 3 BF16 components per operand, 9 partials.
    Full FP32 dynamic range (no clamping). Combined ~21-bit mantissa.
    """
    assert a.dtype == torch.float32 and b.dtype == torch.float32
    a0, a1, a2 = _split_3comp_bf16(a)
    b0, b1, b2 = _split_3comp_bf16(b)
    return matmul_x9_cute_fused_presplit(a0, a1, a2, b0, b1, b2)
