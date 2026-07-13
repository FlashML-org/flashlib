"""Fair raw-input runtime adapter around the frozen 07cf Triton assignment."""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass, field
from threading import Condition, RLock
from typing import Any

from flash_kmeans_triton_h200 import euclid_assign_triton_h200

BASELINE_NAME = "triton_h200_07cf_raw_adapter_v1"
BASELINE_COMMIT = "07cf2a27928aacf6790c950a265d8b8dc83c87cf"
PREPROCESS_IMPL = "flashlib_cake_kmeans._row_norm:PreparedBF16PairRowNorm"


def _record_stream(stream: Any, *tensors: Any) -> None:
    """Keep caller and scratch storage allocator-safe after async submission."""

    seen: set[int] = set()
    for tensor in tensors:
        identity = id(tensor)
        if identity in seen:
            continue
        seen.add(identity)
        record_stream = getattr(tensor, "record_stream", None)
        if callable(record_stream):
            record_stream(stream)


@dataclass
class _Slot:
    x_sq: Any
    c_sq: Any
    norm_plan: Any
    lock: RLock = field(default_factory=RLock, repr=False)


class TritonH20007cfRawAdapter:
    """Init-once raw-input adapter around the frozen 07cf assignment kernel.

    The adapter intentionally reuses the candidate's exported pair-row-norm
    implementation. The frozen 07cf assignment consumes both norm fields,
    while a candidate route may elide fields that it does not bind; this lane
    therefore compares complete raw-input operators, not assignment-only
    stages. Scratch and prepared norm launches are isolated by shape and CUDA
    stream. Each public call allocates its output before preprocessing, matching
    ``FlashKMeansAssignRuntime.compute`` ordering and return semantics.
    """

    def __init__(self, *, device_index: int, arch: str) -> None:
        self.device_index = int(device_index)
        self.arch = str(arch)
        self._slots: dict[tuple[Any, ...], _Slot] = {}
        self._cache_lock = RLock()
        self._lifecycle = Condition(RLock())
        self._active_computes = 0
        self._clearing = False
        self._hits = 0
        self._misses = 0

    @contextmanager
    def _compute_lifecycle(self):
        with self._lifecycle:
            while self._clearing:
                self._lifecycle.wait()
            self._active_computes += 1
        try:
            yield
        finally:
            with self._lifecycle:
                self._active_computes -= 1
                if self._active_computes == 0:
                    self._lifecycle.notify_all()

    def cache_info(self) -> dict[str, int]:
        with self._cache_lock:
            return {
                "size": len(self._slots),
                "hits": self._hits,
                "misses": self._misses,
            }

    def clear(self, *, synchronize: bool = True) -> None:
        """Exclusively release scratch after admitted submissions finish."""

        import torch

        with self._lifecycle:
            while self._clearing:
                self._lifecycle.wait()
            try:
                self._clearing = True
                while self._active_computes:
                    self._lifecycle.wait()
                if synchronize:
                    with torch.cuda.device(self.device_index):
                        torch.cuda.synchronize()
                with self._cache_lock:
                    self._slots.clear()
                    self._hits = 0
                    self._misses = 0
            finally:
                self._clearing = False
                self._lifecycle.notify_all()

    def _resolve_stream(self, x: Any, centroids: Any, stream: Any) -> tuple[Any, tuple[Any, ...]]:
        import torch

        if not all(isinstance(item, torch.Tensor) and item.is_cuda for item in (x, centroids)):
            raise TypeError("07cf raw adapter inputs must be CUDA torch.Tensor objects")
        if x.dtype is not torch.bfloat16 or centroids.dtype is not torch.bfloat16:
            raise TypeError("07cf raw adapter inputs must have bfloat16 dtype")
        if x.ndim != 3 or centroids.ndim != 3 or not x.is_contiguous() or not centroids.is_contiguous():
            raise ValueError("07cf raw adapter inputs must be contiguous [B, rows, D] tensors")
        bsz, n_points, dim = map(int, x.shape)
        c_bsz, n_clusters, c_dim = map(int, centroids.shape)
        if (bsz, dim) != (c_bsz, c_dim) or x.device != centroids.device:
            raise ValueError("07cf raw adapter inputs must have matching batch, feature, and device")
        input_device_index = x.device.index
        if input_device_index is None:
            input_device_index = torch.cuda.current_device()
        input_device_index = int(input_device_index)
        if input_device_index != self.device_index:
            raise ValueError(
                f"07cf raw adapter targets CUDA device {self.device_index}, got input device {input_device_index}"
            )
        with torch.cuda.device(self.device_index):
            resolved_stream = torch.cuda.current_stream(self.device_index) if stream is None else stream
        stream_device = getattr(resolved_stream, "device", None)
        stream_device_index = getattr(stream_device, "index", stream_device)
        if stream_device_index is not None and int(stream_device_index) != self.device_index:
            raise ValueError(
                f"07cf raw adapter stream device {stream_device_index} does not match input device {self.device_index}"
            )
        stream_handle = int(resolved_stream.cuda_stream)
        key = (
            self.device_index,
            self.arch,
            bsz,
            n_points,
            n_clusters,
            dim,
            str(x.dtype),
            stream_handle,
        )
        return resolved_stream, key

    def _locked_slot(
        self,
        key: tuple[Any, ...],
        x: Any,
        centroids: Any,
        *,
        stream: Any,
    ) -> tuple[_Slot, bool]:
        import torch
        from flashlib_cake_kmeans._row_norm import prepare_bf16_pair_row_norm

        # Miss preparation is serialized under the cache lock. This is a cold
        # path and prevents two threads from compiling/publishing duplicate
        # launch plans for the same shape/stream key.
        with self._cache_lock:
            slot = self._slots.get(key)
            cache_hit = slot is not None
            if slot is None:
                with torch.cuda.device(self.device_index), torch.cuda.stream(stream):
                    bsz, n_points, _ = map(int, x.shape)
                    n_clusters = int(centroids.shape[1])
                    x_sq = torch.empty((bsz, n_points), dtype=torch.float32, device=x.device)
                    c_sq = torch.empty((bsz, n_clusters), dtype=torch.float32, device=x.device)
                    norm_plan = prepare_bf16_pair_row_norm(
                        x,
                        centroids,
                        x_sq,
                        c_sq,
                        compute_x=True,
                        compute_c=True,
                        arch=self.arch,
                        stream=stream,
                    )
                slot = _Slot(x_sq=x_sq, c_sq=c_sq, norm_plan=norm_plan)
                self._slots[key] = slot
                self._misses += 1
            else:
                self._hits += 1
        slot.lock.acquire()
        return slot, cache_hit

    def compute(
        self,
        x: Any,
        centroids: Any,
        *,
        stream: Any = None,
        return_info: bool = False,
    ):
        import torch

        with self._compute_lifecycle():
            resolved_stream, key = self._resolve_stream(x, centroids, stream)
            with torch.cuda.device(self.device_index), torch.cuda.stream(resolved_stream):
                bsz, n_points, _ = map(int, x.shape)
                out = torch.empty((bsz, n_points), dtype=torch.int32, device=x.device)
            slot, cache_hit = self._locked_slot(
                key,
                x,
                centroids,
                stream=resolved_stream,
            )
            try:
                with torch.cuda.device(self.device_index), torch.cuda.stream(resolved_stream):
                    _record_stream(resolved_stream, x, centroids, slot.x_sq, slot.c_sq, out)
                    try:
                        slot.norm_plan.rebind(
                            x,
                            centroids,
                            slot.x_sq,
                            slot.c_sq,
                            stream=resolved_stream,
                        )
                        slot.norm_plan.launch(stream=resolved_stream)
                    finally:
                        # The prepared driver launch must not keep caller-owned
                        # x/centroids alive after enqueue returns.
                        slot.norm_plan.release_bound_callers(
                            slot.x_sq,
                            stream=resolved_stream,
                        )
                    produced_out, config = euclid_assign_triton_h200(
                        x,
                        centroids,
                        slot.x_sq,
                        slot.c_sq,
                        out=out,
                    )
                    if produced_out is not out:
                        raise RuntimeError("frozen 07cf assignment did not preserve the adapter-owned output")
            finally:
                slot.lock.release()
        if not return_info:
            return out
        return out, {
            "triton_h200_07cf_config": config,
            "runtime_cache_hit": cache_hit,
            "norm_launch_count": 1,
            "norm_compute_fields": ("x_sq", "c_sq"),
            "assignment_launch_count": 1,
            "runtime_launch_count": 2,
            "stream_handle": int(resolved_stream.cuda_stream),
        }
