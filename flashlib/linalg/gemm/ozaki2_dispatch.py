"""Pure-Python/Triton (and CuTeDSL) implementation of Ozaki Scheme II.

Pipeline (per matmul):
  1. Split A (M,K) and B (N,K) into INT8 modular slices via
     ``triton.ozaki2_split.split_fp_to_int8_moduli`` — also returns per-row
     inverse scales.
  2. For each modulus t, compute P_t = A_int[t] @ B_int[t]^T.  Two GEMM
     backends are available (selected via the ``backend`` argument):
        - ``"triton"`` — :func:`ozaki1_dispatch._triton_int_mm`
                       (~1290 TOPS, 65% of 1979 TOPS H200 INT8 peak)
        - ``"cute"``   — :func:`cutedsl.int8_gemm.cute_int8_mm`
                       (~1546 TOPS, 78% of peak — TMA + WGMMA + cluster)
  3. CRT-reconstruct P_stack -> floating-point output via
     ``triton.crt_reconstruct.crt_reconstruct``.

This is the linear-cost variant promised by Ozaki/Uchino/Imamura 2025
(arXiv 2504.08009). Compared with this repo's existing ``ozaki_int8`` (which
implements the older Ozaki Scheme I with S^2 INT8 GEMMs), this scheme uses
just S GEMMs while delivering the same ~7 bits per slice of CRT precision
budget — i.e. ~24-bit precision at S=7, ~53-bit at S=18.

Range of supported ``num_moduli``:
  2..9 in the pure-Python Triton path. For higher S use ``mode="ozaki2_int8"``
  (the GEMMul8-wrapped path), which has codegen for s up to 20 and handles
  the multi-word CRT internally.
"""

from __future__ import annotations

from typing import Literal

import torch

from flashlib.linalg.gemm.ozaki_constants import choose_kA_kB, crt_constants_tensor
from flashlib.linalg.gemm.triton.ozaki1_dispatch import _triton_int_mm
from flashlib.linalg.gemm.triton.crt_reconstruct import crt_reconstruct
from flashlib.linalg.gemm.triton.ozaki2_split import split_fp_to_int8_moduli


_MAX_S_TRITON = 9
Backend = Literal["triton", "cute"]


def matmul_ozaki2_triton(
    a: torch.Tensor,
    b: torch.Tensor,
    num_moduli: int = 7,
    backend: Backend = "triton",
) -> torch.Tensor:
    """C = A @ B^T via Ozaki-II (pure Triton/CuTeDSL path, num_moduli in [2, 9]).

    ``backend="cute"`` swaps the per-modulus GEMM in step 2 from the Triton
    INT8 kernel to the CuTeDSL Hopper persistent INT8 GEMM.  The split and
    recon steps are unchanged.
    """
    if a.dtype != b.dtype:
        raise TypeError("dtype mismatch")
    if a.dtype not in (torch.float32, torch.float64):
        raise TypeError(f"ozaki2_triton needs fp32/fp64, got {a.dtype}")
    if not (2 <= num_moduli <= _MAX_S_TRITON):
        raise ValueError(
            f"num_moduli {num_moduli} out of [2, {_MAX_S_TRITON}]; "
            f"use mode='ozaki2_int8_<k>' for k>{_MAX_S_TRITON}"
        )
    if a.shape[1] != b.shape[1]:
        raise ValueError("K mismatch")
    M, K = a.shape
    N, _ = b.shape

    kA, kB = choose_kA_kB(num_moduli, K)
    A_int, inv_a = split_fp_to_int8_moduli(a, num_moduli, kA)  # (S, M, K)
    B_int, inv_b = split_fp_to_int8_moduli(b, num_moduli, kB)  # (S, N, K)

    # The CuTe GEMM's TMA needs 16-byte-aligned rows (K % 16 for int8, N % 4 for the int32 output).
    if backend == "cute" and K % 16 == 0 and N % 4 == 0:
        from flashlib.linalg.gemm.cutedsl.int8_gemm import cute_int8_mm
        gemm_fn = cute_int8_mm
    elif backend in ("cute", "triton"):
        gemm_fn = _triton_int_mm
    else:
        raise ValueError(f"unknown backend {backend!r}; use 'triton' or 'cute'")

    # Per-modulus INT8 GEMM into (S, M, N) INT32.
    P_stack = torch.empty((num_moduli, M, N), device=a.device, dtype=torch.int32)
    for t in range(num_moduli):
        gemm_fn(A_int[t], B_int[t], P_stack[t])

    moduli_t, ys_t, _alphas, M_hi, M_lo = crt_constants_tensor(num_moduli, a.device)
    out = crt_reconstruct(P_stack, inv_a, inv_b, moduli_t, ys_t,
                           M_hi, M_lo, out_dtype=a.dtype)
    return out
