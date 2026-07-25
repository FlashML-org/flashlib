"""Shared low-level GPU kernels used by multiple primitives.

Submodules:
    distance              — pairwise / streaming distance kernels (Triton)
    connected_components  — edge-list union-find CC (Triton)
    flash_mst             — GPU-resident dense / sparse Boruvka MST (Triton)
    norm                  — forward LayerNorm / RMSNorm (Triton)
    slot_cache            — device-side LRU admission for an out-of-core
                             table: which ids are resident, what to copy in

Top-level helpers:
    cute_helpers          — small CuTeDSL utilities (dlpack wrap, jit cache,
                             stream wrap). Cross-cutting; not tied to one op.
"""
from __future__ import annotations

from flashlib.kernels import distance
from flashlib.kernels import connected_components
from flashlib.kernels import flash_mst
from flashlib.kernels import norm
from flashlib.kernels import slot_cache


def __dir__() -> list[str]:
    eager = {"distance", "connected_components", "flash_mst", "norm", "slot_cache",
             "cute_helpers"}
    return sorted(set(globals()) | eager)


__all__ = ["distance", "connected_components", "flash_mst", "norm", "slot_cache",
           "cute_helpers"]
