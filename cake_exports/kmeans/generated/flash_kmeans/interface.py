"""Vendored by tools/export-generated-programs (runnable adapter).

Provenance: loom @ 9dd3df73eea8632ec145151e6ae5c5a88dd59969. Flash K-Means assignment; dispatches through the compiled family .so C-ABI.
Runtime-free: this module imports no ``loom`` package.
"""

from __future__ import annotations

import threading
from collections import OrderedDict
from typing import Any

from . import _dispatch_family as _loader

#: Signature scalars, in the order the compiled ``.so`` selector expects them.
SCALAR_NAMES = ('B', 'N', 'D', 'K')
#: Family run-ABI tensor order (inputs, then any norms, then outputs).
TENSOR_KEYS = ('x', 'centroids', 'x_sq', 'c_sq', 'out')
N_SCALARS = len(SCALAR_NAMES)
#: Norm tensor key -> source tensor key; a norm passed as ``None`` to ``run`` is
#: filled by the shipped rowwise squared-norm kernel over its source tensor.
NORM_SOURCES = {'x_sq': 'x', 'c_sq': 'centroids'}
#: Loader module stem of the shipped norm family (None = no internal norm path).
_NORM_LOADER_NAME = '_dispatch_flash_kmeans_rowwise_sqnorm'

_NORM_BUF_CAP = 64
_NORM_BUFS: OrderedDict[tuple, Any] = OrderedDict()
_NORM_LOCK = threading.Lock()


def _split(args: tuple[Any, ...]) -> tuple[tuple[Any, ...], tuple[int, ...]]:
    if len(args) < N_SCALARS:
        raise ValueError(f"expected the {len(TENSOR_KEYS)} tensors then {N_SCALARS} scalars, got {len(args)} args")
    cut = len(args) - N_SCALARS
    return args[:cut], tuple(int(s) for s in args[cut:])


def _norm_loader():
    if _NORM_LOADER_NAME is None:
        raise ValueError("this package ships no internal norm kernel")
    from importlib import import_module

    return import_module(f".{_NORM_LOADER_NAME}", __package__)


def _norm_buffer(src: Any, rows: int) -> Any:
    import torch

    loader = _norm_loader()
    device_index = src.device.index
    if device_index is None:
        device_index = torch.cuda.current_device()
    key = (device_index, loader._ws_stream_key(device_index), rows)
    with _NORM_LOCK:
        buf = _NORM_BUFS.get(key)
        if buf is not None:
            _NORM_BUFS.move_to_end(key)
            return buf
    buf = torch.empty(rows, dtype=torch.float32, device=src.device)
    with _NORM_LOCK:
        _NORM_BUFS[key] = buf
        while len(_NORM_BUFS) > _NORM_BUF_CAP:
            _NORM_BUFS.popitem(last=False)
    return buf


def compute_norms(src: Any, *, out: Any = None, arch: str | None = None) -> Any:
    """Rowwise squared norms of a bf16 ``(..., D)`` tensor via the shipped kernel.

    Returns an fp32 tensor shaped ``src.shape[:-1]``. Without ``out`` the result
    is a view of a per-(device, stream) cached buffer — valid until the next
    ``compute_norms`` call of the same row count on the same stream (same-stream
    launches serialize, so feeding it straight into ``run`` is safe).
    """
    dim = int(src.shape[-1])
    rows = src.numel() // dim
    flat = out.reshape(rows) if out is not None else _norm_buffer(src, rows)
    _norm_loader().run(src, flat, rows, dim, arch=arch)
    return flat.view(*src.shape[:-1])


def clear_norm_buffers() -> None:
    with _NORM_LOCK:
        _NORM_BUFS.clear()


def _fill_norms(tensors: tuple[Any, ...], arch: str | None) -> tuple[Any, ...]:
    filled = list(tensors)
    for index, key in enumerate(TENSOR_KEYS[: len(filled)]):
        if filled[index] is not None:
            continue
        source_key = NORM_SOURCES.get(key)
        if source_key is None:
            raise ValueError(f"tensor {key!r} is None and has no declared norm source")
        source = filled[TENSOR_KEYS.index(source_key)]
        if source is None:
            raise ValueError(f"norm source {source_key!r} for {key!r} is also None")
        filled[index] = compute_norms(source, arch=arch)
    return tuple(filled)


def route_of(*scalars: int, arch: str | None = None) -> int:
    """Ordered first-match route index for a shape, or -1 (no route)."""
    return int(_loader.route_of(*(int(s) for s in scalars), arch=arch))


def covered(*scalars: int, arch: str | None = None) -> bool:
    return route_of(*scalars, arch=arch) >= 0


def route_id(*scalars: int, arch: str | None = None) -> str:
    index = route_of(*scalars, arch=arch)
    if index < 0:
        raise ValueError(f"no dispatch route for scalars {scalars!r}")
    return str(_loader.ROUTE_IDS[index])


def allocate_workspaces(*scalars: int, device: Any, arch: str | None = None) -> list[Any]:
    return _loader.allocate_workspaces(*(int(s) for s in scalars), device=device, arch=arch)


def run(*args: Any, arch: str | None = None) -> None:
    """Route + launch: pass the family tensors then the scalars; workspaces are
    sized and allocated here from the same table the ``.so`` validates against.
    A norm tensor passed as ``None`` is filled by the shipped norm kernel from
    its declared source (see ``NORM_SOURCES``)."""
    tensors, scalars = _split(args)
    if not tensors:
        raise ValueError("run requires at least one tensor argument")
    if any(t is None for t in tensors):
        tensors = _fill_norms(tensors, arch)
    workspaces = _loader.allocate_workspaces(*scalars, device=tensors[0].device, arch=arch)
    _loader.run(*tensors, *workspaces, *scalars, arch=arch)
