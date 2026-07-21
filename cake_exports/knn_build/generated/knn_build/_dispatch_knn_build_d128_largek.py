"""Runtime-free tvm-ffi loader for the knn_build_d128_largek dispatch-family .so.

The exported package dispatches through the compiled family ``.so``'s C-ABI —
``route_of(*scalars) -> route_index`` and ``run(*tensors, *workspaces, *scalars)``
— with no ``loom`` import (the tvm-ffi C-ABI is the contract). Workspaces are
sized by the generated ``_dispatch_workspace_shapes`` table and caller-allocated
(``run`` validates numel and never mallocs).
"""

from __future__ import annotations

from functools import lru_cache
from typing import Any

from . import _dispatch_ws_knn_build_d128_largek as _ws
from ._runtime import detect_gpu_arch

FAMILY_SO = 'family.so'
N_SCALARS = 7
# Per-index route metadata (the C ``route_of`` int index -> the route's parent_id
# string / kernel-stage count); the package interface reports these.
ROUTE_IDS = _ws.ROUTE_IDS
ROUTE_LAUNCH_COUNTS = _ws.ROUTE_LAUNCH_COUNTS
ROUTE_INPUT_TENSORS = _ws.ROUTE_INPUT_TENSORS


@lru_cache(maxsize=None)
def _load(arch: str) -> Any:
    import importlib.resources as _res

    import tvm_ffi

    binary = _res.files('knn_build').joinpath(f"_dispatch_so/knn_build_d128_largek/{arch}/{FAMILY_SO}")
    with _res.as_file(binary) as path:
        return tvm_ffi.load_module(str(path))


def _module(arch: str | None = None) -> Any:
    return _load(str(detect_gpu_arch()) if arch is None else str(arch))


def route_of(*scalars: int, arch: str | None = None) -> int:
    return int(_module(arch)["route_of"](*(int(s) for s in scalars)))


def workspace_shapes(*scalars: int, arch: str | None = None) -> dict[str, tuple[int, ...]]:
    index = route_of(*scalars, arch=arch)
    return _ws.workspace_shapes(index, *(int(s) for s in scalars))


def allocate_workspaces(*scalars: int, device: Any, arch: str | None = None) -> list[Any]:
    """torch.empty each family workspace (bound shape or a (1,) placeholder)."""
    import torch

    shapes = workspace_shapes(*scalars, arch=arch)
    return [
        torch.empty(shapes.get(key, (1,)), dtype=getattr(torch, _ws.WORKSPACE_DTYPES[key]), device=device)
        for key in _ws.WORKSPACE_KEYS
    ]


def run(*args: Any, arch: str | None = None) -> None:
    """Dispatch: tensors, then workspaces, then the 7 scalars."""
    import tvm_ffi

    with tvm_ffi.use_torch_stream():
        _module(arch)["run"](*args)
