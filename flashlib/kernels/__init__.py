"""Shared low-level GPU kernels used by multiple primitives.

Submodules:
    distance              — pairwise / streaming distance kernels (Triton)
    connected_components  — edge-list union-find CC (Triton)
    flash_mst             — GPU-resident dense / sparse Boruvka MST (Triton)
    norm                  — forward LayerNorm / RMSNorm (Triton)
Top-level helpers:
    cute_helpers          — small CuTeDSL utilities (dlpack wrap, jit cache,
                             stream wrap). Cross-cutting; not tied to one op.
"""
from __future__ import annotations

import importlib

_SUBMODULES = {"distance", "connected_components", "flash_mst", "norm", "cute_helpers"}


def __getattr__(name: str):
    if name in _SUBMODULES:
        mod = importlib.import_module(f"{__name__}.{name}")
        globals()[name] = mod
        return mod
    raise AttributeError(f"module 'flashlib.kernels' has no attribute {name!r}")


def __dir__() -> list[str]:
    return sorted(set(globals()) | _SUBMODULES)


__all__ = sorted(_SUBMODULES)
