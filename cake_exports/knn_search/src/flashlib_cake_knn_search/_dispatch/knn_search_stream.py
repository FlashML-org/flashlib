"""Device/stream ownership helpers for prepared KNN-search workspaces."""
from __future__ import annotations
from json import loads as _json_loads
from .._dispatch_runtime import _capture_cuTensorMapEncodeTiled, _decode_capture, _import_dispatch_module, _ir_proxy
from typing import Any

def current_stream_handle(inputs: dict[str, Any]) -> int:
    """Return the prepared stream handle or the caller's active stream."""
    import torch
    prepared_key = inputs.get('_knn_search_prepared_stream_key')
    if prepared_key is not None:
        return int(prepared_key[1])
    device_index = int(inputs['queries'].device.index or 0)
    return int(torch.cuda.current_stream(device_index).cuda_stream)

def stream_cache_key(inputs: dict[str, Any], *parts: int) -> tuple[int, ...]:
    """Prefix a reusable workspace key with exact device and stream identity."""
    device_index = int(inputs['queries'].device.index or 0)
    return (device_index, current_stream_handle(inputs), *(int(part) for part in parts))
