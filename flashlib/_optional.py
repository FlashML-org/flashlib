"""Optional dependencies: availability checks and the error an op raises when one is missing.

* CuTe DSL (``nvidia-cutlass-dsl``, NVIDIA only): ``pip install 'flashlib[cutedsl]'``
* numba (HDBSCAN tree helpers): ``pip install 'flashlib[numba]'``
"""
from __future__ import annotations

import functools
import importlib.metadata
import re
from typing import Optional

# Keep in sync with the ``cutedsl`` extra in pyproject.toml.
_CUTLASS_DSL_RANGE = ((4, 4), (4, 7))


@functools.lru_cache(maxsize=None)
def _cutedsl_problem() -> Optional[str]:
    """Why the CuTe DSL backend cannot be used here, or None if it can."""
    import torch

    if getattr(torch.version, "hip", None) is not None:
        return "rocm"
    try:
        version = importlib.metadata.version("nvidia-cutlass-dsl")
    except importlib.metadata.PackageNotFoundError:
        return "nvidia-cutlass-dsl is not installed"
    m = re.match(r"(\d+)\.(\d+)", version)
    lo, hi = _CUTLASS_DSL_RANGE
    if m is None or not lo <= (int(m.group(1)), int(m.group(2))) < hi:
        return f"nvidia-cutlass-dsl {version} is outside the supported range >=4.4,<4.7"
    try:
        import cutlass.cute  # noqa: F401
    except Exception as e:  # noqa: BLE001  (a broken install counts as missing)
        return f"importing cutlass failed: {type(e).__name__}: {e}"
    return None


def has_cutedsl() -> bool:
    return _cutedsl_problem() is None


@functools.lru_cache(maxsize=None)
def has_numba() -> bool:
    try:
        import numba  # noqa: F401
    except Exception:  # noqa: BLE001
        return False
    return True


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


def require_numba(op: str) -> None:
    """Raise ImportError naming ``op`` when numba is unavailable."""
    if not has_numba():
        raise ImportError(f"flashlib: {op} requires numba. Install it with: pip install 'flashlib[numba]'")


__all__ = ["has_cutedsl", "has_numba", "require_cutedsl", "require_numba"]
