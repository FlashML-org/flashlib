"""Current-stream ownership helpers for Flash-KMeans production routes."""
from __future__ import annotations
from json import loads as _json_loads
from .._dispatch_runtime import _capture_cuTensorMapEncodeTiled, _decode_capture, _import_dispatch_module, _ir_proxy
from collections.abc import Iterator
from contextlib import contextmanager
from typing import Any

def current_stream(inputs: dict[str, Any]) -> Any:
    """Resolve the caller's current stream on the input tensor device."""
    import torch
    return torch.cuda.current_stream(device=inputs['x'].device)

def stream_cache_key(inputs: dict[str, Any], *parts: int) -> tuple[int, ...]:
    """Prefix a cache key with the exact CUDA device and stream handle."""
    import torch
    prepared_key = inputs.get('_flash_kmeans_assign_prepared_stream_key')
    if prepared_key is not None:
        device_index, stream_handle = prepared_key
    else:
        device = inputs['x'].device
        device_index = getattr(device, 'index', None)
        if device_index is None:
            device_index = torch.cuda.current_device()
        stream_handle = current_stream(inputs).cuda_stream
    return (int(device_index), int(stream_handle), *(int(part) for part in parts))

@contextmanager
def bind_current_stream(inputs: dict[str, Any]) -> Iterator[Any]:
    """Keep a production child pipeline on the caller's entry stream."""
    import torch
    device = inputs['x'].device
    stream = current_stream(inputs)
    device_index = getattr(device, 'index', None)
    current_device = torch.cuda.current_device()
    if device_index is None or int(device_index) == int(current_device):
        yield stream
        return
    with torch.cuda.device(device), torch.cuda.stream(stream):
        yield stream
