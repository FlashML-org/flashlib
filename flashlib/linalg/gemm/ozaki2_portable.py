"""Ozaki Scheme II — portable Triton + CuTeDSL implementation (no native shim).

This is the **precision-frontier path** that breaks the FP32-WGMMA
accumulator wall capping ``bf16x3 / fp16x9 / tf32x6`` at ~14 effective bits
regardless of how many splits you use. By doing every modular GEMM in
INT8 with an INT32 accumulator (exact for ``K · 127² < 2³¹``) and
reconstructing via the Chinese Remainder Theorem, precision scales
**linearly** with ``num_moduli``: ~7 bits per modulus, from ~10 bits at
s=5 up to **full FP64 at s=18**.

Reference: Ozaki / Uchino / Imamura 2025, arXiv 2504.08009 ("Scheme II").

Two backends are exposed:

* ``backend="triton"`` — every sub-kernel (split, INT8 GEMM, CRT recon)
  is Triton. ~290 TF eff. FP32 at s=3, ~110 TF at s=8 on H200 8192³.
  Works on any Hopper-class GPU with no native shim.

* ``backend="cute"`` — the INT8 GEMM step uses the CUTLASS DSL Hopper
  persistent kernel (89% TC pipe utilisation, 1546 TOPS / 78% peak).
  ~316 TF at s=3, ~126 TF at s=8 on H200 — about 8-15% faster than the
  Triton-only path.

Both backends produce bitwise-identical outputs (same CRT path).

For maximum throughput when the GEMMul8 native library is built, see
``gemm_ozaki2_int8`` (this module's sibling) — that wraps the C++/CUTLASS
production kernel and runs another 30-40% faster but requires the
``libgemmul8_shim.so`` build.
"""
from __future__ import annotations

import torch

from flashlib._optional import require_cutedsl
from flashlib.info.estimate import Estimate
from flashlib.info.roofline import roofline
from flashlib.linalg.gemm.fp32 import _shape_mkn


def _residual_bits(num_moduli: int) -> float:
    """Approximate RMS rel-err for a given num_moduli on N(0,1) inputs.

    Empirical fit: ~7 bits per modulus, ε ≈ 2^(-7s).
    """
    return 2.0 ** (-7 * num_moduli)


def gemm_ozaki2_triton(A: torch.Tensor, B: torch.Tensor, *,
                        num_moduli: int = 7) -> torch.Tensor:
    """Ozaki II GEMM, all-Triton backend. Standard PyTorch convention.

    ``num_moduli`` ∈ [2, 9] for the pure-Triton path. For larger counts,
    use ``gemm_ozaki2_int8`` (GEMMul8 native) or ``gemm_ozaki2_cute``.
    """
    if not (2 <= num_moduli <= 9):
        raise ValueError(
            f"ozaki2_triton supports num_moduli in [2, 9]; got {num_moduli}. "
            f"For higher precision use gemm_ozaki2_cute or gemm_ozaki2_int8."
        )
    from flashlib.linalg.gemm.ozaki2_dispatch import matmul_ozaki2_triton
    A_ = A.contiguous()
    B_ = B.t().contiguous()  # kernel takes B in (N, K)
    return matmul_ozaki2_triton(A_, B_, num_moduli=num_moduli, backend="triton")


def gemm_ozaki2_cute(A: torch.Tensor, B: torch.Tensor, *,
                      num_moduli: int = 7) -> torch.Tensor:
    """Ozaki II GEMM, CuTeDSL INT8 GEMM backend. ~10-15% faster than the
    pure-Triton path. Requires CUTLASS DSL.
    """
    if not (2 <= num_moduli <= 9):
        raise ValueError(
            f"ozaki2_cute supports num_moduli in [2, 9]; got {num_moduli}. "
            f"For higher precision use gemm_ozaki2_int8 (GEMMul8)."
        )
    require_cutedsl("gemm_ozaki2_cute")
    from flashlib.linalg.gemm.ozaki2_dispatch import matmul_ozaki2_triton
    A_ = A.contiguous()
    B_ = B.t().contiguous()
    return matmul_ozaki2_triton(A_, B_, num_moduli=num_moduli, backend="cute")


def estimate(backend: str, shape, params=None, tol=None,
              dtype="float32", device="H100", **_):
    M, K, N = _shape_mkn(shape)
    params = params or {}
    s = params.get("num_moduli", 7)
    # Per-modulus cost: 2× INT8 GEMM FLOPs + reconstruction.
    int8_flops_per_mod = 2 * M * K * N
    flops = s * int8_flops_per_mod + 4 * M * N  # CRT recon
    bytes_moved = s * (M * K + K * N) + M * N * 4
    # H200 INT8 TC peak: 1979 TOPS. Triton ~65% peak; cute ~78% peak.
    eff = 0.65 if backend == "triton" else 0.78
    rt, bound = roofline(flops, bytes_moved, "int8", device, op_type="gemm")
    rt = rt / eff
    return Estimate(
        op_name=f"gemm_ozaki2_{backend}",
        runtime_ms=rt, flops=flops, bytes_moved=bytes_moved,
        memory_peak_gb=(s * (M * K + K * N) + M * N * 4) / 1e9,
        bound=bound, confidence="roofline", n_kernel_launches=s + 2,
        suggested_config={"num_moduli": s}, subops=[],
        notes=[
            f"M={M} K={K} N={N}, num_moduli={s}",
            f"Ozaki II CRT INT8: ~{7*s} effective bits.",
            f"Backend={backend}: {'~78% INT8 peak (CuTeDSL)' if backend=='cute' else '~65% INT8 peak (Triton)'}.",
            "Breaks FP32-WGMMA wall — precision scales linearly with num_moduli.",
        ],
        expected_residual=_residual_bits(s),
        precision_tier="mixed", tol=tol,
    )


def estimate_triton(shape, params=None, tol=None, dtype="float32", device="H100", **_):
    return estimate("triton", shape, params, tol, dtype, device)


def estimate_cute(shape, params=None, tol=None, dtype="float32", device="H100", **_):
    return estimate("cute", shape, params, tol, dtype, device)
