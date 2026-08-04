"""Runtime-free tvm-ffi loader for the knn_build_d128_largek dispatch-family .so.

The exported package dispatches through the compiled family ``.so``'s C-ABI —
``route_of(*scalars) -> route_index`` and raw
``run(*tensors, *workspaces, *scalars, cuda_stream_ptr)`` — with no ``loom``
import (the tvm-ffi C-ABI is the contract). This loader's public ``run`` obtains
the current PyTorch stream and supplies that final implementation argument.
Workspaces are sized by the generated ``_dispatch_workspace_shapes`` table and
caller-allocated (``run`` validates numel and never mallocs).
"""

from __future__ import annotations

import threading
from collections import OrderedDict
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


_WS_ALIGN = 256
_WS_VIEW_CAP = 512
_WS_ARENAS: dict = {}  # (device_index, stream, arch) -> uint8 arena tensor
_WS_VIEWS: "OrderedDict" = OrderedDict()  # (scalars, device_index, stream, arch) -> [views]
_WS_LOCK = threading.Lock()


_RAW_STREAM_GET: Any = None


def _stream_handle(device_index: int) -> int:
    """Current-stream handle for the raw ``run`` ABI, resolved cheaply.

    ``torch._C._cuda_getCurrentRawStream`` returns the same handle
    ``torch.cuda.current_stream(...).cuda_stream`` does without walking the
    device-resolution helper chain or materializing a Stream object (~5-8us
    per call, measured 4-13% of consumer E2E on the kmeans/knn packages).
    Resolved once per process; a miss pins the public path forever.
    """
    global _RAW_STREAM_GET
    getter = _RAW_STREAM_GET
    if getter is None:
        import torch

        getter = getattr(getattr(torch, "_C", None), "_cuda_getCurrentRawStream", None)
        _RAW_STREAM_GET = getter if getter is not None else False
    if getter:
        return int(getter(device_index))
    import torch

    return int(torch.cuda.current_stream(device_index).cuda_stream)


def _ws_stream_key(device_index: int) -> int:
    return _stream_handle(device_index)


def allocate_workspaces(*scalars: int, device: Any, arch: str | None = None) -> list[Any]:
    """Carve exact-shape workspace views from a per-(device, stream) arena.

    Workspaces are per-call scratch by contract — the previous per-call path
    passed freshly ``torch.empty``-allocated (uninitialized) buffers, so no
    kernel may rely on their contents. The arena is therefore initialized
    once at the running-max byte size and shared by every signature on one
    stream (same-stream launches serialize, which is what makes device-side
    sharing safe); each signature's exact-shape views are cached, so a warm
    call is one dict hit. Growth allocates a larger arena; prior views keep
    the old buffer alive until their cache entries evict (bounded LRU) or
    ``clear_workspaces()`` drops everything.
    """
    import torch

    index = getattr(device, "index", None)
    device_index = int(torch.cuda.current_device() if index is None else index)
    key = (tuple(int(s) for s in scalars), device_index, _ws_stream_key(device_index), arch)
    with _WS_LOCK:
        views = _WS_VIEWS.get(key)
        if views is not None:
            _WS_VIEWS.move_to_end(key)
            return views
    shapes = workspace_shapes(*scalars, arch=arch)
    dtypes, extents, offsets = [], [], []
    total = 0
    for ws_key in _ws.WORKSPACE_KEYS:
        dtype = getattr(torch, _ws.WORKSPACE_DTYPES[ws_key])
        shape = tuple(shapes.get(ws_key, (1,)))
        numel = 1
        for dim in shape:
            numel *= int(dim)
        nbytes = numel * torch.empty(0, dtype=dtype).element_size()
        offsets.append(total)
        extents.append((dtype, shape, nbytes))
        total += (nbytes + _WS_ALIGN - 1) // _WS_ALIGN * _WS_ALIGN
        dtypes.append(dtype)
    arena_key = (device_index, key[2], arch)
    with _WS_LOCK:
        arena = _WS_ARENAS.get(arena_key)
        if arena is None or arena.numel() < total:
            arena = torch.empty(max(total, 1), dtype=torch.uint8, device=device)
            _WS_ARENAS[arena_key] = arena
        views = []
        for offset, (dtype, shape, nbytes) in zip(offsets, extents):
            flat = arena[offset : offset + nbytes].view(dtype)
            views.append(flat.reshape(shape))
        _WS_VIEWS[key] = views
        _WS_VIEWS.move_to_end(key)
        while len(_WS_VIEWS) > _WS_VIEW_CAP:
            _WS_VIEWS.popitem(last=False)
    return views


def clear_workspaces() -> None:
    """Drop the workspace arena and every cached signature view."""
    with _WS_LOCK:
        _WS_VIEWS.clear()
        _WS_ARENAS.clear()


def run(*args: Any, arch: str | None = None) -> None:
    """Dispatch: tensors, then workspaces, then the 7 scalars."""
    module = _module(arch)
    index = args[0].device.index
    if index is None:
        import torch

        index = torch.cuda.current_device()
    module["run"](*args, _stream_handle(index))
