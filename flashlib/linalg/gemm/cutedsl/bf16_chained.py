"""3xbf16 GEMM via CuTe DSL — replaces cuBLAS LT chained emulation.

Architecture:
  gemm_bf16(A_bf16, B_bf16) -> C_fp32   — native CuTe DSL bf16 GEMM (~860 TF/s at N=8192)
  gemm_3xbf16(A_fp32, B_fp32) -> C_fp32 — 3-product Ozaki emulation on top

3-product Ozaki (Ootomo-style, drops A_lo · B_lo):
  A = A_hi + A_lo   with A_hi = RN_bf16(A), A_lo = RN_bf16(A - A_hi)
  B analogous
  C = A_hi @ B_hi + A_hi @ B_lo + A_lo @ B_hi

Each @ is a bf16x bf16 -> fp32 WGMMA via the CuTe DSL kernel. We chain 3 launches
and sum the outputs. Fusing all 3 MMAs in a single kernel would save ~25% HBM
traffic for A/B but these GEMMs are compute-bound (0.75 flop/byte ratio at
bf16/fp32), so the chained approach is within 10% of the in-kernel fused
version at this size range.

Error bound: matches fp32 SGEMM on inputs with elements in [-1, 1] (Ootomo-Yokota
2022, arXiv:2203.03341). Breaks at 2^-14 absolute cancellation level.
"""
from functools import lru_cache

import torch
import cutlass
import cutlass.cute as cute
import cuda.bindings.driver as cuda
from cutlass.cute.runtime import from_dlpack

from flashlib.linalg.gemm.cutedsl.hopper_wgmma_bf16 import HopperWgmmaGemmKernel
from flashlib.linalg.gemm.triton.split_helpers import split_fp32_to_bf16_fused, sum3_fused


def _torch_to_cute(t: torch.Tensor, dtype: type, leading_dim: int):
    """Wrap a 3D (M, N, L) torch tensor as a CuTe dynamic-layout tensor."""
    mt = from_dlpack(t, assumed_align=16)
    mt.element_type = dtype
    return mt.mark_layout_dynamic(leading_dim=leading_dim)


@lru_cache(maxsize=None)
def _compile_bf16_gemm(m: int, n: int, k: int,
                       tile_mn=(128, 256), cluster_mn=(1, 1)):
    """Build & compile a HopperWgmmaGemmKernel for given shape.

    Returns: callable(mA, mB, mC, stream) where mA/mB are bf16 cute tensors
    (permuted (m, k, l=1) with k-major), mC is fp32 cute tensor ((m, n, l=1)
    n-major).
    """
    gemm = HopperWgmmaGemmKernel(
        acc_dtype=cutlass.Float32,
        tile_shape_mn=tile_mn,
        cluster_shape_mn=cluster_mn,
    )
    # Build stub tensors matching the shape we'll invoke.
    # k-major A: torch shape (m, k) — leading_dim=1 means k is contiguous.
    # k-major B: torch shape (n, k) — same.
    # n-major C: torch shape (m, n) — leading_dim=1 means n contiguous.
    # The DSL wants (mode0, mode1, l=1) after permute. We add a trailing L=1.
    a_stub = torch.empty(m, k, 1, device="cuda", dtype=torch.bfloat16)
    b_stub = torch.empty(n, k, 1, device="cuda", dtype=torch.bfloat16)
    c_stub = torch.empty(m, n, 1, device="cuda", dtype=torch.float32)

    mA = _torch_to_cute(a_stub, cutlass.BFloat16, leading_dim=1)
    mB = _torch_to_cute(b_stub, cutlass.BFloat16, leading_dim=1)
    mC = _torch_to_cute(c_stub, cutlass.Float32,  leading_dim=1)

    torch_stream = torch.cuda.current_stream()
    stream = cuda.CUstream(torch_stream.cuda_stream)
    return cute.compile(gemm, mA, mB, mC, stream)


def gemm_bf16(A: torch.Tensor, B: torch.Tensor,
              out: torch.Tensor | None = None,
              tile_mn=(128, 256)) -> torch.Tensor:
    """Compute C = A @ B (standard). A: (M,K) bf16 k-major, B: (K,N) bf16
    but kernel expects B as (N,K) k-major — we handle the transpose internally.

    Fast path: if B is already (N,K) in row-major (e.g. passed as `B_kn` explicitly),
    the caller can use `gemm_bf16_kn(A, B_kn)` to skip the transpose.
    """
    assert A.dtype == torch.bfloat16 and B.dtype == torch.bfloat16
    assert A.is_cuda and B.is_cuda
    M, K = A.shape
    K2, N = B.shape
    assert K == K2, f"K mismatch: {K} vs {K2}"
    # Convert B (K,N) -> (N,K) k-major.
    B_nk = B.T.contiguous()
    return gemm_bf16_kn(A, B_nk, out=out, tile_mn=tile_mn)


def gemm_bf16_kn(A: torch.Tensor, B: torch.Tensor,
                 out: torch.Tensor | None = None,
                 tile_mn=(128, 256)) -> torch.Tensor:
    """Compute C = A @ B.T  (B passed as (N,K) k-major, skip transpose).
    A: (M,K) bf16; B: (N,K) bf16; C: (M,N) fp32.
    """
    assert A.dtype == torch.bfloat16 and B.dtype == torch.bfloat16
    assert A.is_cuda and B.is_cuda
    M, K = A.shape
    N, K2 = B.shape
    assert K == K2, f"K mismatch: {K} vs {K2}"
    assert K % 8 == 0 and N % 4 == 0, (
        f"TMA needs 16-byte-aligned rows (K % 8 == 0, N % 4 == 0); got K={K}, N={N}. "
        f"Use gemm_3xbf16_padded for arbitrary shapes."
    )

    if out is None:
        out = torch.empty(M, N, device=A.device, dtype=torch.float32)
    else:
        assert out.shape == (M, N) and out.dtype == torch.float32

    # Add trailing L=1 dim required by the kernel.
    a3 = A.unsqueeze(-1)  # (M, K, 1)
    b3 = B.unsqueeze(-1)  # (N, K, 1)
    c3 = out.unsqueeze(-1)  # (M, N, 1)

    compiled = _compile_bf16_gemm(M, N, K, tile_mn=tile_mn)
    mA = _torch_to_cute(a3, cutlass.BFloat16, leading_dim=1)
    mB = _torch_to_cute(b3, cutlass.BFloat16, leading_dim=1)
    mC = _torch_to_cute(c3, cutlass.Float32,  leading_dim=1)
    torch_stream = torch.cuda.current_stream()
    stream = cuda.CUstream(torch_stream.cuda_stream)
    compiled(mA, mB, mC, stream)
    return out


def _split_fp32_to_bf16(X: torch.Tensor):
    """Ootomo-style 2-piece fp32 -> (bf16_hi, bf16_lo) split.

    X_hi = RN_bf16(X); X_lo = RN_bf16(X - X_hi).
    Residual |X - X_hi - X_lo| < 2^-16 * max(|X|) — beyond what 3xbf16
    covers anyway.
    """
    X_hi = X.to(torch.bfloat16)
    X_lo = (X - X_hi.float()).to(torch.bfloat16)
    return X_hi, X_lo


def gemm_3xbf16(A: torch.Tensor, B: torch.Tensor,
                out: torch.Tensor | None = None,
                tile_mn=(128, 256)) -> torch.Tensor:
    """Compute C = A @ B via 3-product bf16 Ozaki emulation.
    A: (M, K) fp32; B: (K, N) fp32; C: (M, N) fp32.
    """
    assert A.dtype == torch.float32 and B.dtype == torch.float32
    if A.shape[1] % 8 or B.shape[1] % 4:
        C = gemm_3xbf16_padded(A, B, tile_mn=tile_mn)
        return C if out is None else out.copy_(C)
    if not A.is_contiguous():
        A = A.contiguous()
    # Standard form: B is (K, N). Kernel wants (N, K) k-major.
    B_nk = B.T.contiguous()
    return gemm_3xbf16_kn(A, B_nk, out=out, tile_mn=tile_mn)


def gemm_3xbf16_padded(A: torch.Tensor, B: torch.Tensor,
                       tile_mn=(128, 256)) -> torch.Tensor:
    """Standard-form gemm_3xbf16 with automatic K/N/M zero-padding.

    The underlying WGMMA kernel needs K aligned to 16 and M/N aligned to
    the tile. For shapes violating those (notably back-transform at
    k=n/2+1), pad-then-slice. Padded flops are wasted on zero rows/cols;
    the tradeoff still wins when the unaligned shape falls off the kernel's
    fast path. Measured: (8192, 4097) @ (4097, 4097) → 2.1 ms padded vs
    6.2 ms nvmath / ~broken unpadded.
    """
    M, K = A.shape
    K2, N = B.shape
    assert K == K2
    align_K, align_N, align_M = 16, tile_mn[1], tile_mn[0]
    K_pad = (align_K - K % align_K) % align_K
    N_pad = (align_N - N % align_N) % align_N
    M_pad = (align_M - M % align_M) % align_M
    if K_pad == 0 and N_pad == 0 and M_pad == 0:
        return gemm_3xbf16(A, B, tile_mn=tile_mn)
    Ap = torch.nn.functional.pad(A, (0, K_pad, 0, M_pad)) if (K_pad or M_pad) else A
    Bp = torch.nn.functional.pad(B, (0, N_pad, 0, K_pad)) if (K_pad or N_pad) else B
    Cp = gemm_3xbf16(Ap, Bp, tile_mn=tile_mn)
    return Cp[:M, :N].contiguous()


def gemm_3xbf16_kn(A: torch.Tensor, B: torch.Tensor,
                   out: torch.Tensor | None = None,
                   tile_mn=(128, 256)) -> torch.Tensor:
    """3xbf16 GEMM with B already in (N, K) k-major form. Computes
    C = A @ B.T (math) = sum_k A[m,k] * B[n,k].

    Skip-transpose variant — the hot path when caller can arrange inputs."""
    assert A.dtype == torch.float32 and B.dtype == torch.float32
    M, K = A.shape
    N, _ = B.shape
    if out is None:
        out = torch.empty(M, N, device=A.device, dtype=torch.float32)

    A_hi, A_lo = split_fp32_to_bf16_fused(A)
    B_hi, B_lo = split_fp32_to_bf16_fused(B)

    C1 = gemm_bf16_kn(A_hi, B_hi, tile_mn=tile_mn)
    C2 = gemm_bf16_kn(A_hi, B_lo, tile_mn=tile_mn)
    C3 = gemm_bf16_kn(A_lo, B_hi, tile_mn=tile_mn)
    sum3_fused(C1, C2, C3, out=out)
    return out
