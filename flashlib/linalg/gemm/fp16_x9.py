"""3-component FP16 split × 9 partial products via fused CuTeDSL kernel.

Pareto-optimal at K ≤ 1024 (~17-18 effective bits, ~110 TFLOPS on H200).
Above K=2048 the FP32 wgmma accumulator caps both fp16x9 and bf16x3 at the
same ~14 bit ceiling — bf16x3 wins on speed there. Use this variant when
you specifically need < 14-bit-wall accuracy at small K (e.g., refinement
inner loop, ill-conditioned solves).
"""
import torch

from flashlib._optional import require_cutedsl
from flashlib.info.estimate import Estimate
from flashlib.info.roofline import roofline
from flashlib.linalg.gemm.fp32 import _shape_mkn


def gemm(A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
    """3-component FP16 × 9-MMA fused matmul. Standard PyTorch (K, N) convention.

    Inputs FP32 / FP16 / BF16 (any combo cast to fp32 internally before the
    3-way mantissa split). Output FP32. ~25-30 effective bits at K ≤ 1024.

    Note: the underlying fast-gemm cuTe kernel expects ``B`` shaped ``(N, K)``;
    we transpose at the boundary so callers can pass ``B`` in PyTorch's
    standard ``(K, N)`` shape (i.e. computes ``A @ B``).
    """
    require_cutedsl("gemm_fp16_x9")
    from flashlib.linalg.gemm.cutedsl.fp16x9 import matmul_fp16x9_cute_fused
    A32 = A.to(torch.float32).contiguous()
    B32 = B.to(torch.float32).t().contiguous()  # (K,N) -> kernel-native (N,K)
    return matmul_fp16x9_cute_fused(A32, B32)


def estimate(shape, params=None, tol=None, dtype="float32", device="H100", **_):
    M, K, N = _shape_mkn(shape)
    flops = 2 * M * K * N
    bytes_moved = (M * K + K * N) * 4 + M * N * 4
    rt, bound = roofline(flops, bytes_moved, "fp16", device, op_type="gemm")
    # 9 fp16 MMAs per output replace 1 fp32 op → effective TFLOPS = peak/9.
    rt = rt * 9.0
    # K-dependent residual: precision floor is set by FP32-WGMMA-acc round-off
    # × √K. Measured on H200: 1.0e-6 @ K=256, 3.7e-6 @ K=1024, 1.5e-5 @ K=4096.
    if K <= 256:
        residual = 1e-6
    elif K <= 1024:
        residual = 4e-6
    elif K <= 4096:
        residual = 1.5e-5
    else:
        residual = 5e-5
    return Estimate(
        op_name="gemm_fp16_x9",
        runtime_ms=rt, flops=flops, bytes_moved=bytes_moved,
        memory_peak_gb=(M * K + K * N + M * N) * 4 / 1e9,
        bound=bound, confidence="roofline", n_kernel_launches=1,
        suggested_config={"variant": "fused_x9"}, subops=[],
        notes=[
            f"M={M} K={K} N={N}",
            "FP16x9 fused: 3 components × 9 partials; ~17-18 bits at K≤1024.",
            "Measured RMS rel-err: 1e-6 @ K=256, 4e-6 @ K=1024, 1.5e-5 @ K=4096.",
        ],
        expected_residual=residual, precision_tier="mixed", tol=tol,
    )


def recommend(shape, params=None, tol=None, dtype="float32", device="H100", **_):
    return {"variant": "fused_x9"}
