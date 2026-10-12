"""GEMM with multiple precision/performance variants — tol-routed.

Measured RMS relative error on H200 with N(0,1) inputs (see per-file
``cost.py`` notes for K-dependent values, plus
``benchmarks/results/boundaries_gemm.md`` for full Pareto sweeps):

    variant            RMS rel err   typical TF  note
    ──────────────────────────────────────────────────────────────────────
    ozaki2_int8(s=18)  ~1e-15        51 (cuBLAS-FP64-grade) needs gemmul8.so
    ozaki2_int8(s=14)  ~3e-13        60          near-cuBLAS-FP64 throughput
    ozaki2_cute(s=8)   ~3e-7         126         **2× cuBLAS-FP32, +6 bits**
    ozaki2_triton(s=8) ~3e-7         110         portable (no native shim)
    fp16_x3_kahan      ~5e-7         135         Triton+Kahan, K-independent
    fp32               ~3e-7         50          cuBLAS native
    3xtf32             ~1e-6         120
    fp16_x9            ~4e-6 (K≤1k)  ~180        CuTeDSL fused
    tf32_x6            ~3e-6 (K≤1k)  120         FP64-input emulation
    ozaki2_*(s=5)      ~6e-3         170         coarse-precision Ozaki
    3xbf16             ~3e-5         228         **CuTeDSL fused (was ~1.7e-3)**
    3xfp16             ~2e-4         210
    tf32               ~3e-4         386 (1 TC GEMM)
    fp16               ~4e-4         500
    bf16               ~3e-3         816 (cuBLAS / 720 Triton)

Routing logic (per user spec):

  1. **No tradeoff (one variant strictly dominates)** -> use the dominant one.
     Examples: ``3xbf16`` is now CuTeDSL-fused (faster AND tighter than the
     old Python 3-call); ``ozaki2_cute`` >= ``ozaki2_triton`` always.
  2. **Same DSL, one strictly dominates** -> single canonical impl, no
     versioning needed.
  3. **Cross-DSL with both available** -> pick the better-performing one;
     fall back to the available one if only one DSL is installed.
  4. **True precision/throughput tradeoff** -> ``gemm(A, B, tol=t)`` picks
     the strictest residual ≤ ``t``.

Examples::

    flashlib.gemm(A, B)                # tol=None -> fp32 (exact)
    flashlib.gemm(A, B, tol=1e-3)      # bf16
    flashlib.gemm(A, B, tol=1e-5)      # 3xbf16 (cute fused, 228 TF, 3e-5 RMS)
    flashlib.gemm(A, B, tol=1e-7)      # ozaki2_cute(s=8) at 126 TF —
                                       # 2x faster than fp32 with MORE precision!
    flashlib.gemm(A, B, tol=1e-12)     # ozaki2_int8(s=14) FP64-grade
"""
import torch

from flashlib.linalg.gemm import (
    fp32, tf32, bf16, fp16,
    bf16_x3, fp16_x3, tf32_x3,
    fp16_x9, fp16_x3_kahan, tf32_x6, ozaki2_int8,
    ozaki2_portable,
)


# ──────────────────────────────────────────────────────────────────────────
# Top-level kernel handles. Each is a stable public name.
# ──────────────────────────────────────────────────────────────────────────
gemm_fp32           = fp32.gemm
gemm_tf32           = tf32.gemm
gemm_bf16           = bf16.gemm
gemm_fp16           = fp16.gemm
gemm_3xbf16         = bf16_x3.gemm           # auto-routes cute_fused / triton
gemm_3xfp16         = fp16_x3.gemm
gemm_3xtf32         = tf32_x3.gemm
gemm_fp16_x9        = fp16_x9.gemm
gemm_fp16_x3_kahan  = fp16_x3_kahan.gemm
gemm_tf32_x6        = tf32_x6.gemm

# Ozaki-II family: linearly-scalable precision via INT8 CRT. Three impls
# differing only in throughput (no precision tradeoff between them):
#   * gemm_ozaki2_triton  — pure Triton (s ≤ 9), no native shim required
#   * gemm_ozaki2_cute    — CuTeDSL INT8 GEMM (s ≤ 9), ~10-15% faster
#   * gemm_ozaki2_int8    — GEMMul8 native (s ≤ 18 / FP64-grade), needs .so
gemm_ozaki2_triton  = ozaki2_portable.gemm_ozaki2_triton
gemm_ozaki2_cute    = ozaki2_portable.gemm_ozaki2_cute
gemm_ozaki2_int8    = ozaki2_int8.gemm


# ──────────────────────────────────────────────────────────────────────────
# Capability detection: which backends are usable on THIS install?
# ──────────────────────────────────────────────────────────────────────────


def _has_cute() -> bool:
    from flashlib._optional import has_cutedsl

    try:
        return has_cutedsl() and torch.cuda.is_available() and torch.cuda.get_device_properties(0).major >= 9
    except Exception:
        return False


def _has_gemmul8() -> bool:
    try:
        from flashlib.linalg.gemm.native.gemmul8 import _load_lib
        _load_lib()
        return True
    except Exception:
        return False


# ──────────────────────────────────────────────────────────────────────────
# Pareto-ordered residual table — DEDUPED of dominated variants.
#
# Variants that another variant strictly Pareto-dominates (faster AND tighter
# at the same shape regime) are NOT eligible by tol-routing. They remain
# accessible by explicit ``backend="..."``.
#
# Ordering: tightest residual first; dispatcher walks bottom-up to find the
# fastest variant whose residual ≤ tol.
# ──────────────────────────────────────────────────────────────────────────
_RESIDUAL_PREFERENCE = [
    # name,             expected_residual,   note
    ("ozaki2_int8",     1e-15),  # native (.so), s=14 -> FP64 grade
    ("ozaki2_cute",     1e-7),   # CuTeDSL INT8, s=8 -> ~24 bits (better than fp32!)
    ("ozaki2_triton",   1e-7),   # Triton-only, s=8
    ("fp16_x3_kahan",   5e-7),   # Triton + Kahan, K-independent
    ("fp32",            3e-7),   # cuBLAS FP32 (no TC) — slow baseline
    ("3xtf32",          1.4e-6),
    ("tf32_x6",         3e-6),   # K ≤ 1024 typical
    ("fp16_x9",         4e-6),   # K ≤ 1024 typical
    ("3xbf16",          3e-5),   # **Updated**: CuTeDSL fused now hits 3e-5, not 1.7e-3
    ("3xfp16",          2.1e-4),
    ("tf32",            3e-4),
    ("fp16",            4e-4),
    ("bf16",            3e-3),
]


# Approximate TFLOPS at 8192³ on H200 — used for tie-breaking when multiple
# variants meet a tol. Higher = faster.
_THROUGHPUT_TF = {
    "bf16":           816,
    "fp16":           892,
    "tf32":           386,
    "3xbf16":         228,
    "3xfp16":         210,
    "3xtf32":         120,
    "fp16_x9":        180,
    "tf32_x6":        120,
    "fp16_x3_kahan":  135,
    "ozaki2_cute":    126,
    "ozaki2_triton":  110,
    "fp32":            50,
    "ozaki2_int8":     80,  # native, s=14
}


# Native low-precision storage dtypes that some primitives pre-cast their
# operand to once and reuse across many matmuls (e.g. logistic regression
# L-BFGS, DBSCAN's high-D KNN, flash KMeans assignment). All other variants
# (ozaki2, 3xbf16, fp16_x9, ...) consume fp32 inputs and cast internally,
# so their callers should keep fp32 storage and dispatch per call via
# ``gemm(A, B, tol=...)``.
_STORAGE_DTYPE = {
    "fp16":  "float16",
    "bf16":  "bfloat16",
}


def storage_dtype_for(tol):
    """Return the low-precision ``torch.dtype`` a primitive may pre-cast to.

    Used by primitives that cache one operand in low precision once and
    reuse it over many matmuls (LR L-BFGS for example).

      * ``tol`` is ``None`` or ``<= 0`` (exact) -> ``None`` -- the caller
        MUST NOT cast and should keep the user's input dtype as-is.
      * ``tol`` selects ``bf16`` -> ``torch.bfloat16``.
      * ``tol`` selects ``fp16`` -> ``torch.float16``.
      * Any other variant (ozaki, 3xbf16, ...) casts internally per call,
        so the caller should also keep the input dtype -> ``None``.
    """
    import torch as _torch
    if tol is None or tol <= 0:
        return None
    chosen = _pick_by_tol(tol)
    name = _STORAGE_DTYPE.get(chosen)
    if name is None:
        return None
    return getattr(_torch, name)


def _pick_by_tol(tol):
    """Pick the FASTEST variant whose declared residual ≤ ``tol``.

    Implements the user-spec routing logic:
      * tol=None -> the strictest available variant (fp32 by default; or
        ozaki2_* if ``flashlib.gemm.set_default_exact("ozaki2_cute")`` was
        called).
      * Otherwise: among variants with residual ≤ tol, pick the one with
        the highest measured throughput on the current hardware. Skip
        variants whose backend is unavailable on this install.
    """
    if tol is None or tol <= 0:
        return _DEFAULT_EXACT
    eligible = [
        (name, _THROUGHPUT_TF.get(name, 0))
        for name, res in _RESIDUAL_PREFERENCE
        if res <= tol and _is_available(name)
    ]
    if not eligible:
        return _DEFAULT_EXACT
    eligible.sort(key=lambda x: -x[1])  # highest throughput first
    return eligible[0][0]


_DEFAULT_EXACT = "fp32"


def set_default_exact(name: str) -> None:
    """Override the variant used for ``tol=None``.

    By default ``tol=None`` -> ``fp32``. Power users can flip this to
    ``ozaki2_cute`` to get +6 bits of precision at 2× the throughput,
    when their inputs are within INT8-CRT splitting range. The cost
    model still reports the chosen variant accurately.
    """
    global _DEFAULT_EXACT
    if name not in _VARIANT_MODULES:
        raise ValueError(f"unknown gemm variant: {name}")
    _DEFAULT_EXACT = name


def _is_available(name: str) -> bool:
    """Per-variant capability gate."""
    if name == "ozaki2_int8":
        return _has_gemmul8()
    if name in ("ozaki2_cute", "fp16_x9", "3xbf16"):
        return _has_cute() or name == "3xbf16"  # 3xbf16 has a Triton fallback
    return True


# ──────────────────────────────────────────────────────────────────────────
# Public dispatcher
# ──────────────────────────────────────────────────────────────────────────
_VARIANT_FNS = {
    "fp32":            gemm_fp32,
    "tf32":            gemm_tf32,
    "bf16":            gemm_bf16,
    "fp16":            gemm_fp16,
    "3xtf32":          gemm_3xtf32,
    "3xbf16":          gemm_3xbf16,
    "3xfp16":          gemm_3xfp16,
    "fp16_x9":         gemm_fp16_x9,
    "fp16_x3_kahan":   gemm_fp16_x3_kahan,
    "tf32_x6":         gemm_tf32_x6,
    "ozaki2_int8":     gemm_ozaki2_int8,
    "ozaki2_cute":     gemm_ozaki2_cute,
    "ozaki2_triton":   gemm_ozaki2_triton,
}


def gemm(A, B, *, tol=None, backend=None, num_moduli=None):
    """Multi-precision GEMM dispatcher.

    Args:
        A, B: matmul operands. Standard PyTorch ``A @ B`` convention.
        tol: residual tolerance (relative RMS).

            * ``None`` (default) **-> EXACT in input dtype**: just
              ``torch.matmul(A, B)``. No casting, no approximation. If
              ``A``/``B`` are bf16 it stays bf16; if fp32 it stays fp32.
              The "exact" semantics in this codebase always means "do
              not lose any precision the user already gave us".
            * Otherwise: pick the fastest variant whose declared
              residual <= ``tol`` and run it (may cast inputs).
        backend: explicit variant name; overrides ``tol`` and the
            input-dtype-preserving default.
        num_moduli: only for ``ozaki2_*`` variants. Higher -> tighter
            precision (~7 bits per modulus). Defaults: 8 for
            ozaki2_cute / ozaki2_triton, 14 for ozaki2_int8.
    """
    if backend is not None:
        chosen = backend
        fn = _VARIANT_FNS.get(chosen)
        if fn is None:
            raise ValueError(f"unknown gemm backend {chosen!r}")
        if chosen.startswith("ozaki2") and num_moduli is not None:
            return fn(A, B, num_moduli=num_moduli)
        return fn(A, B)
    if tol is None or tol <= 0:
        # Exact in input dtype. PyTorch >= 1.12 defaults to
        # ``torch.backends.cuda.matmul.allow_tf32 = False`` so fp32 inputs
        # already go through strict IEEE here. We do NOT defensively flip
        # the global flag -- callers that explicitly opted into TF32
        # globally are respected.
        return torch.matmul(A, B)
    chosen = _pick_by_tol(tol)
    fn = _VARIANT_FNS.get(chosen)
    if fn is None:
        raise ValueError(f"unknown gemm backend {chosen!r}")
    if chosen.startswith("ozaki2") and num_moduli is not None:
        return fn(A, B, num_moduli=num_moduli)
    return fn(A, B)


# ──────────────────────────────────────────────────────────────────────────
# Cost-model shims (for flashlib.info)
# ──────────────────────────────────────────────────────────────────────────
class _Ozaki2PortableCostShim:
    """Adapter so ``info.estimate('gemm_ozaki2_cute', ...)`` works."""
    def __init__(self, backend):
        self.backend = backend
    def estimate(self, shape, params=None, tol=None, dtype="float32",
                 device="H100", **_):
        return ozaki2_portable.estimate(
            self.backend, shape, params, tol, dtype, device
        )
    def recommend(self, shape, params=None, tol=None, dtype="float32",
                  device="H100", **_):
        return {"backend": self.backend}


_VARIANT_MODULES = {
    "fp32": fp32, "tf32": tf32, "3xtf32": tf32_x3,
    "bf16": bf16, "3xbf16": bf16_x3,
    "fp16": fp16, "3xfp16": fp16_x3,
    "fp16_x9": fp16_x9, "fp16_x3_kahan": fp16_x3_kahan,
    "tf32_x6": tf32_x6, "ozaki2_int8": ozaki2_int8,
    "ozaki2_cute":   _Ozaki2PortableCostShim("cute"),
    "ozaki2_triton": _Ozaki2PortableCostShim("triton"),
}


def estimate(shape, params=None, tol=None, dtype="float32", device="H100", **_):
    """gemm dispatcher cost — routes by tol, sets op_name to the chosen variant."""
    chosen = _pick_by_tol(tol)
    est = _VARIANT_MODULES[chosen].estimate(shape=shape, params=params, tol=tol,
                                             dtype=dtype, device=device)
    est.op_name = f"gemm_{chosen}"
    est.tol = tol
    return est


def recommend(shape, params=None, tol=None, dtype="float32", device="H100", **_):
    return {"variant": f"gemm_{_pick_by_tol(tol)}"}


__all__ = [
    "gemm",
    "gemm_fp32", "gemm_tf32", "gemm_3xtf32",
    "gemm_bf16", "gemm_3xbf16",
    "gemm_fp16", "gemm_3xfp16",
    "gemm_fp16_x9", "gemm_fp16_x3_kahan",
    "gemm_tf32_x6", "gemm_ozaki2_int8",
    "gemm_ozaki2_cute", "gemm_ozaki2_triton",
    "set_default_exact",
    "storage_dtype_for",
    "estimate", "recommend",
]
