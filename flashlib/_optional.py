"""Optional dependency (CuTe DSL): availability checks and the error an op raises when it is missing.

Install with ``pip install 'flashlib[cutedsl]'`` (``nvidia-cutlass-dsl``, NVIDIA only).
"""
from __future__ import annotations

import functools
import importlib.metadata
import re
import warnings
from typing import Optional

# Keep in sync with the ``cutedsl`` extra in pyproject.toml.
_CUTLASS_DSL_RANGE = ((4, 4), (4, 7))
_CUTEDSL_NOT_INSTALLED = "nvidia-cutlass-dsl is not installed"


@functools.lru_cache(maxsize=None)
def _cutedsl_problem() -> Optional[str]:
    """Why the CuTe DSL backend cannot be used here, or None if it can."""
    import torch

    if getattr(torch.version, "hip", None) is not None:
        return "rocm"
    try:
        version = importlib.metadata.version("nvidia-cutlass-dsl")
    except importlib.metadata.PackageNotFoundError:
        return _CUTEDSL_NOT_INSTALLED
    m = re.match(r"(\d+)\.(\d+)", version)
    lo, hi = _CUTLASS_DSL_RANGE
    if m is None or not lo <= (int(m.group(1)), int(m.group(2))) < hi:
        return f"nvidia-cutlass-dsl {version} is outside the supported range >=4.4,<4.7"
    try:
        import cutlass.cute  # noqa: F401
    except Exception as e:  # noqa: BLE001  (a broken install counts as missing)
        return f"importing cutlass failed: {type(e).__name__}: {e}"
    return None


@functools.lru_cache(maxsize=None)
def _warn_cutedsl_unusable(problem: str) -> None:
    warnings.warn(
        f"flashlib: nvidia-cutlass-dsl is installed but cannot be used ({problem}). "
        f"CuTe DSL backends are disabled and ops use their Triton paths. "
        f"Reinstall it with: pip install 'flashlib[cutedsl]'",
        RuntimeWarning,
        stacklevel=3,
    )


def has_cutedsl() -> bool:
    problem = _cutedsl_problem()
    # Not installed is the normal Triton-only setup; installed-but-broken is a misconfiguration worth surfacing.
    if problem is not None and problem not in ("rocm", _CUTEDSL_NOT_INSTALLED):
        _warn_cutedsl_unusable(problem)
    return problem is None


@functools.lru_cache(maxsize=None)
def has_cutedsl_sm90() -> bool:
    """True iff the CuTe DSL is usable and device 0 is Hopper (the WGMMA kernels are sm_90-only)."""
    if not has_cutedsl():
        return False
    import torch

    try:
        return torch.cuda.is_available() and torch.cuda.get_device_properties(0).major == 9
    except Exception:  # noqa: BLE001
        return False


def require_cutedsl(op: str) -> None:
    """Raise ImportError naming ``op`` when the CuTe DSL backend cannot be used."""
    problem = _cutedsl_problem()
    if problem is None:
        return
    if problem == "rocm":
        raise ImportError(f"flashlib: {op} requires the CuTe DSL backend, which runs on NVIDIA GPUs only.")
    raise ImportError(
        f"flashlib: {op} requires the CuTe DSL backend ({problem}). "
        f"Install it with: pip install 'flashlib[cutedsl]'"
    )


__all__ = ["has_cutedsl", "has_cutedsl_sm90", "require_cutedsl"]
