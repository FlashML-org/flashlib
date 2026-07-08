from __future__ import annotations

import math
from collections import OrderedDict
from contextlib import contextmanager
from dataclasses import dataclass, field
from threading import Condition, Event, RLock
from typing import Any

from ._dispatch import flash_kmeans_assign_dispatcher as _dispatcher
from ._runtime import detect_gpu_arch, launch_context, runtime_activity_snapshot

SEMANTIC_ENTRYPOINT = "loom.examples.weave.flash_kmeans_assign_dispatcher:launch_for_eval"


@dataclass(frozen=True)
class PreparedFlashKMeansAssign:
    """Exact tensors, workspace, output, and direct launch plan for KMeans."""

    inputs: dict[str, Any]
    launch_plan: Any
    stream: Any = None
    timeout_ms: float | None = None

    @property
    def out(self) -> Any:
        return self.inputs["out"]

    @property
    def selected_route(self) -> str:
        return self.launch_plan.selected_route


@dataclass
class _RuntimeSlot:
    inputs: dict[str, Any]
    launch_plan: Any
    norm_plan: Any = None
    internal_x_sq: bool = False
    internal_c_sq: bool = False
    norm_compute_fields: tuple[str, ...] = ()
    record_each_call_fields: tuple[str, ...] = ("x", "centroids")
    norm_rebind_program: tuple[tuple[str, Any], ...] = ()
    lock: RLock = field(default_factory=RLock, repr=False)


@dataclass
class _PendingPreparation:
    event: Event = field(default_factory=Event, repr=False)
    error: BaseException | None = field(default=None, repr=False)


def _normalize_device_index(torch: Any, device: Any) -> int:
    if device is None:
        return int(torch.cuda.current_device())
    if isinstance(device, bool):
        raise TypeError("device must identify a CUDA device")
    if isinstance(device, int):
        if device < 0:
            raise ValueError("CUDA device index must be non-negative")
        return int(device)
    device_type = getattr(device, "type", None)
    device_index = getattr(device, "index", None)
    if device_type is None:
        device_ctor = getattr(torch, "device", None)
        if device_ctor is None:
            raise TypeError("device must identify a CUDA device")
        normalized = device_ctor(device)
        device_type = getattr(normalized, "type", None)
        device_index = getattr(normalized, "index", None)
    if device_type != "cuda":
        raise ValueError(f"Flash-KMeans runtime requires a CUDA device, got {device_type!r}")
    resolved_index = int(torch.cuda.current_device() if device_index is None else device_index)
    if resolved_index < 0:
        raise ValueError("CUDA device index must be non-negative")
    return resolved_index


def _validate_timeout(timeout_ms: float | None) -> float | None:
    if timeout_ms is None:
        return None
    if isinstance(timeout_ms, bool) or not isinstance(timeout_ms, int | float):
        raise TypeError("timeout_ms must be a positive finite number or None")
    value = float(timeout_ms)
    if not math.isfinite(value) or value <= 0:
        raise ValueError("timeout_ms must be a positive finite number")
    return value


def _resolve_stream(torch: Any, device_index: int, stream: Any) -> Any:
    resolved_stream = torch.cuda.current_stream(device_index) if stream is None else stream
    stream_device = getattr(resolved_stream, "device", None)
    stream_device_index = getattr(stream_device, "index", stream_device)
    if stream_device_index is not None and int(stream_device_index) != int(device_index):
        raise ValueError(
            f"Flash-KMeans stream device {stream_device_index} does not match input device {device_index}"
        )
    return resolved_stream


def _same_cuda_stream(left: Any, right: Any) -> bool:
    return left is right or int(left.cuda_stream) == int(right.cuda_stream)


@contextmanager
def _device_stream_context(
    torch: Any,
    device_index: int,
    stream: Any,
    *,
    context_is_active: bool,
):
    """Resolve one stream and enter it only when the caller has not already done so."""

    if context_is_active:
        if stream is None:
            raise RuntimeError("an active Flash-KMeans CUDA context requires an explicit stream")
        yield stream
        return
    with torch.cuda.device(device_index):
        resolved_stream = _resolve_stream(torch, device_index, stream)
        with torch.cuda.stream(resolved_stream):
            yield resolved_stream


def _pointer_alias_topology(*tensors: Any) -> tuple[int, ...]:
    """Canonical pointer-equivalence classes independent of concrete addresses."""

    classes: dict[int, int] = {}
    topology: list[int] = []
    for tensor in tensors:
        if tensor is None:
            topology.append(-1)
            continue
        pointer = int(tensor.data_ptr())
        if pointer not in classes:
            classes[pointer] = len(classes)
        topology.append(classes[pointer])
    return tuple(topology)


class FlashKMeansAssignRuntime:
    """Long-lived KMeans assignment runtime with per-shape, per-stream plans.

    ``compute`` accepts new tensors and new shapes.  The first call for one
    shape/stream resolves and captures the exact route; later calls only
    refresh data-derived norms, rebind caller tensor pointers, and enqueue the
    already captured launch sequence.  Workspace is never shared across CUDA
    streams.
    """

    def __init__(
        self,
        *,
        device: Any = None,
        arch: str | None = None,
        timeout_ms: float | None = None,
        max_cached_shapes: int | None = None,
        compile: str = "lazy",
    ) -> None:
        import torch

        if compile != "lazy":
            raise ValueError("compile must be 'lazy'; use warmup() to eagerly prepare known shapes")
        if max_cached_shapes is not None:
            if isinstance(max_cached_shapes, bool) or not isinstance(max_cached_shapes, int):
                raise TypeError("max_cached_shapes must be a positive integer or None")
            if max_cached_shapes <= 0:
                raise ValueError("max_cached_shapes must be positive")
        self.device_index = _normalize_device_index(torch, device)
        with torch.cuda.device(self.device_index):
            detected_arch = str(detect_gpu_arch())
        self.arch = detected_arch if arch is None else str(arch)
        if self.arch != detected_arch:
            raise ValueError(
                f"Flash-KMeans runtime arch must match its device: requested {self.arch}, detected {detected_arch}"
            )
        self.timeout_ms = _validate_timeout(timeout_ms)
        self.max_cached_shapes = max_cached_shapes
        self._torch = torch
        self._slots: OrderedDict[tuple[Any, ...], _RuntimeSlot] = OrderedDict()
        self._preparing: dict[tuple[Any, ...], _PendingPreparation] = {}
        self._cache_lock = RLock()
        self._lifecycle = Condition(RLock())
        self._active_computes = 0
        self._clearing = False
        self._hits = 0
        self._misses = 0

    @contextmanager
    def _compute_lifecycle(self):
        """Admit concurrent compute calls while making clear() an exclusive barrier."""

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

    def cache_info(self) -> dict[str, int | None]:
        with self._cache_lock:
            return {
                "size": len(self._slots),
                "hits": self._hits,
                "misses": self._misses,
                "max_cached_shapes": self.max_cached_shapes,
            }

    def _cache_key(
        self,
        inputs: dict[str, Any],
        stream_handle: int,
        *,
        caller_supplied_out: bool = True,
    ) -> tuple[Any, ...]:
        alias_topology: tuple[int, ...]
        if not caller_supplied_out and inputs["x_sq"] is None and inputs["c_sq"] is None:
            x_pointer = int(inputs["x"].data_ptr())
            centroids_pointer = int(inputs["centroids"].data_ptr())
            if x_pointer and centroids_pointer:
                # The internally allocated output cannot alias either live
                # input. Preserve the canonical class numbering for both
                # possible x/centroid relationships without reading it.
                alias_topology = (
                    0,
                    0 if x_pointer == centroids_pointer else 1,
                    -1,
                    -1,
                    1 if x_pointer == centroids_pointer else 2,
                )
            else:
                alias_topology = _pointer_alias_topology(
                    inputs["x"],
                    inputs["centroids"],
                    inputs["x_sq"],
                    inputs["c_sq"],
                    inputs["out"],
                )
        else:
            alias_topology = _pointer_alias_topology(
                inputs["x"],
                inputs["centroids"],
                inputs["x_sq"],
                inputs["c_sq"],
                inputs["out"],
            )
        return (
            self.device_index,
            self.arch,
            int(inputs["B"]),
            int(inputs["N"]),
            int(inputs["D"]),
            int(inputs["K"]),
            str(inputs["dtype"]),
            alias_topology,
            int(stream_handle),
        )

    def clear(self, *, synchronize: bool = True) -> None:
        """Exclusively release slots after admitted host calls finish.

        The default waits for device completion. With ``synchronize=False``,
        recorded-stream ownership keeps tensor storage allocator-safe, but the
        caller remains responsible for observing asynchronous completion.
        """
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

    def warmup(self, *args: Any, synchronize: bool = True, **kwargs: Any):
        result = self.compute(*args, **kwargs)
        if synchronize:
            import torch

            with torch.cuda.device(self.device_index):
                torch.cuda.synchronize()
        return result

    def compute(
        self,
        x: Any,
        centroids: Any,
        *,
        out: Any | None = None,
        x_sq: Any | None = None,
        c_sq: Any | None = None,
        stream: Any = None,
        timeout_ms: float | None = None,
        return_info: bool = False,
    ):
        with self._compute_lifecycle():
            torch = self._torch
            if int(torch.cuda.current_device()) == self.device_index:
                if stream is None:
                    resolved_stream = torch.cuda.current_stream(self.device_index)
                    return self._compute(
                        x,
                        centroids,
                        out=out,
                        x_sq=x_sq,
                        c_sq=c_sq,
                        stream=resolved_stream,
                        timeout_ms=timeout_ms,
                        return_info=return_info,
                    )
                resolved_stream = _resolve_stream(torch, self.device_index, stream)
                current_stream = torch.cuda.current_stream(self.device_index)
                if _same_cuda_stream(resolved_stream, current_stream):
                    return self._compute(
                        x,
                        centroids,
                        out=out,
                        x_sq=x_sq,
                        c_sq=c_sq,
                        stream=resolved_stream,
                        timeout_ms=timeout_ms,
                        return_info=return_info,
                    )
                with torch.cuda.stream(resolved_stream):
                    return self._compute(
                        x,
                        centroids,
                        out=out,
                        x_sq=x_sq,
                        c_sq=c_sq,
                        stream=resolved_stream,
                        timeout_ms=timeout_ms,
                        return_info=return_info,
                    )
            with torch.cuda.device(self.device_index):
                resolved_stream = _resolve_stream(torch, self.device_index, stream)
                current_stream = torch.cuda.current_stream(self.device_index)
                if _same_cuda_stream(resolved_stream, current_stream):
                    return self._compute(
                        x,
                        centroids,
                        out=out,
                        x_sq=x_sq,
                        c_sq=c_sq,
                        stream=resolved_stream,
                        timeout_ms=timeout_ms,
                        return_info=return_info,
                    )
                with torch.cuda.stream(resolved_stream):
                    return self._compute(
                        x,
                        centroids,
                        out=out,
                        x_sq=x_sq,
                        c_sq=c_sq,
                        stream=resolved_stream,
                        timeout_ms=timeout_ms,
                        return_info=return_info,
                    )

    def _compute(
        self,
        x: Any,
        centroids: Any,
        *,
        out: Any | None,
        x_sq: Any | None,
        c_sq: Any | None,
        stream: Any,
        timeout_ms: float | None,
        return_info: bool,
    ):
        caller_supplied_out = out is not None
        effective_timeout_ms = self.timeout_ms if timeout_ms is None else _validate_timeout(timeout_ms)
        inputs, resolved_stream, resolved_arch = _prepare_inputs(
            x,
            centroids,
            out=out,
            x_sq=x_sq,
            c_sq=c_sq,
            device_index=self.device_index,
            arch=self.arch,
            stream=stream,
            arch_is_validated=True,
            timeout_ms=effective_timeout_ms,
            defer_missing_norms=True,
            _context_is_active=True,
            _torch=self._torch,
        )
        stream_handle = int(resolved_stream.cuda_stream)
        key = self._cache_key(
            inputs,
            stream_handle,
            caller_supplied_out=caller_supplied_out,
        )
        activity_before = runtime_activity_snapshot() if return_info else None
        slot, cache_hit, owns_slot_lock = self._get_or_create_slot(
            key,
            inputs,
            resolved_stream=resolved_stream,
            resolved_arch=resolved_arch,
            _context_is_active=True,
        )
        activity_after = runtime_activity_snapshot() if return_info else None
        if not owns_slot_lock:
            slot.lock.acquire()
        norm_mode = (
            "fused_bf16_pair_row_norm:" + ",".join(slot.norm_compute_fields)
            if slot.norm_compute_fields
            else "route_elided_internal_norms"
            if slot.internal_x_sq or slot.internal_c_sq
            else "explicit_precomputed"
        )
        try:
            if cache_hit:
                if slot.launch_plan.stream_handle != stream_handle:
                    raise RuntimeError("cached Flash-KMeans slot changed its prepared stream")
                for name in ("x", "centroids", "out"):
                    slot.inputs[name] = inputs[name]
                if not slot.internal_x_sq:
                    slot.inputs["x_sq"] = inputs["x_sq"]
                if not slot.internal_c_sq:
                    slot.inputs["c_sq"] = inputs["c_sq"]
                slot.inputs["_norm_mode"] = norm_mode
            # Protect every new caller-owned tensor before non-retaining
            # rebind. Slot-owned norms and a default output are allocated
            # on this permanently bound stream, so they do not need
            # repeated allocator ownership calls.
            _record_stream_inputs(
                resolved_stream,
                slot.inputs,
                slot.record_each_call_fields,
            )
            if caller_supplied_out:
                _record_stream(resolved_stream, slot.inputs["out"])
            if cache_hit:
                # Slot lookup already proved the public alias topology and
                # stream identity. Rebind only the scrubbed pointer carriers
                # needed by this fixed-stream launch sequence.
                slot.launch_plan.direct_launcher._rebind_stream_bound_scrubbed_inputs(
                    slot.inputs,
                    stream=resolved_stream,
                )
            if slot.norm_plan is not None:
                try:
                    if cache_hit:
                        _rebind_stream_bound_norm_inputs(slot)
                    else:
                        slot.norm_plan.rebind(
                            slot.inputs["x"],
                            slot.inputs["centroids"],
                            slot.inputs["x_sq"],
                            slot.inputs["c_sq"],
                            stream=resolved_stream,
                        )
                    slot.norm_plan.launch(
                        stream=resolved_stream,
                        timeout_ms=effective_timeout_ms,
                    )
                finally:
                    if not cache_hit:
                        _prepare_stream_bound_norm_rebind(slot)
            slot.launch_plan._launch_stream_bound_recorded(
                timeout_ms=effective_timeout_ms,
            )
            output = slot.inputs["out"]
        finally:
            try:
                if not cache_hit:
                    slot.launch_plan.direct_launcher.release_bound_inputs()
            finally:
                for name in ("x", "centroids", "out"):
                    slot.inputs[name] = None
                if not slot.internal_x_sq:
                    slot.inputs["x_sq"] = None
                if not slot.internal_c_sq:
                    slot.inputs["c_sq"] = None
                slot.lock.release()

        if not return_info:
            return output
        cold_activity = {
            "source_read_occurred": bool(activity_after["source_reads"] > activity_before["source_reads"]),
            "nvrtc_compile_occurred": bool(
                activity_after["nvrtc_compiles"] > activity_before["nvrtc_compiles"]
            ),
            "module_load_occurred": bool(activity_after["module_loads"] > activity_before["module_loads"]),
            "scratch_allocation_occurred": not cache_hit,
        }
        return output, {
            "semantic_entrypoint": SEMANTIC_ENTRYPOINT,
            "selected_route": slot.launch_plan.selected_route,
            "launch_entrypoint": slot.launch_plan.launch_entrypoint,
            "exact_launch_plan": True,
            "runtime_cache_hit": cache_hit,
            "assignment_launch_count": slot.launch_plan.launch_count,
            "norm_launch_count": int(slot.norm_plan is not None),
            "norm_compute_fields": list(slot.norm_compute_fields),
            "runtime_launch_count": slot.launch_plan.launch_count + int(slot.norm_plan is not None),
            "norm_mode": norm_mode,
            "arch": slot.launch_plan.arch,
            "device_index": slot.launch_plan.device_index,
            "stream_handle": slot.launch_plan.stream_handle,
            "cold_first_call_activity": cold_activity,
        }

    def _get_or_create_slot(
        self,
        key: tuple[Any, ...],
        inputs: dict[str, Any],
        *,
        resolved_stream: Any,
        resolved_arch: str,
        _context_is_active: bool = False,
    ) -> tuple[_RuntimeSlot, bool, bool]:
        """Return a slot; a newly published slot is returned with its lock held."""

        import torch

        while True:
            with self._cache_lock:
                slot = self._slots.get(key)
                if slot is not None:
                    self._hits += 1
                    return slot, True, False
                pending = self._preparing.get(key)
                if pending is None:
                    if (
                        self.max_cached_shapes is not None
                        and len(self._slots) + len(self._preparing) >= self.max_cached_shapes
                    ):
                        raise RuntimeError(
                            "FlashKMeansAssignRuntime cache is full; call clear() only after in-flight work completes"
                        )
                    pending = _PendingPreparation()
                    self._preparing[key] = pending
                    break
            pending.event.wait()
            if pending.error is not None:
                raise RuntimeError("KMeans slot preparation failed in another thread") from pending.error

        slot: _RuntimeSlot | None = None
        try:
            with _device_stream_context(
                torch,
                self.device_index,
                resolved_stream,
                context_is_active=_context_is_active,
            ):
                internal_x_sq = inputs["x_sq"] is None
                internal_c_sq = inputs["c_sq"] is None
                if internal_x_sq:
                    inputs["x_sq"] = torch.empty(
                        (int(inputs["B"]), int(inputs["N"])),
                        dtype=torch.float32,
                        device=inputs["x"].device,
                    )
                if internal_c_sq:
                    inputs["c_sq"] = torch.empty(
                        (int(inputs["B"]), int(inputs["K"])),
                        dtype=torch.float32,
                        device=inputs["x"].device,
                    )
                launch_plan = _dispatcher.prepare_launch_plan(
                    inputs,
                    arch=resolved_arch,
                    stream=resolved_stream,
                    timeout_ms=None,
                )
                # This slot is permanently bound to resolved_stream. Record
                # route-private scratch and descriptor storage once while the
                # plan owns its original caller tensors; subsequent computes
                # separately record every new public tensor before rebinding.
                launch_plan.direct_launcher.record_stream(resolved_stream)
                bound_input_keys = set(launch_plan.direct_launcher.bound_input_keys)
                compute_x_sq = internal_x_sq and "x_sq" in bound_input_keys
                compute_c_sq = internal_c_sq and "c_sq" in bound_input_keys
                norm_plan = None
                if compute_x_sq or compute_c_sq:
                    from ._row_norm import prepare_bf16_pair_row_norm

                    norm_plan = prepare_bf16_pair_row_norm(
                        inputs["x"],
                        inputs["centroids"],
                        inputs["x_sq"],
                        inputs["c_sq"],
                        compute_x=compute_x_sq,
                        compute_c=compute_c_sq,
                        arch=resolved_arch,
                        stream=resolved_stream,
                    )
                norm_compute_fields = tuple(
                    name for name, enabled in (("x_sq", compute_x_sq), ("c_sq", compute_c_sq)) if enabled
                )
            slot = _RuntimeSlot(
                inputs=inputs,
                launch_plan=launch_plan,
                norm_plan=norm_plan,
                internal_x_sq=internal_x_sq,
                internal_c_sq=internal_c_sq,
                norm_compute_fields=norm_compute_fields,
                record_each_call_fields=(
                    "x",
                    "centroids",
                    *(("x_sq",) if not internal_x_sq else ()),
                    *(("c_sq",) if not internal_c_sq else ()),
                ),
            )
            slot.lock.acquire()
        except BaseException as error:
            if slot is not None:
                try:
                    slot.lock.release()
                except RuntimeError:
                    pass
            with self._cache_lock:
                self._preparing.pop(key, None)
                pending.error = error
                pending.event.set()
            raise
        publication_committed = False
        try:
            with self._cache_lock:
                old_misses = self._misses
                try:
                    self._slots[key] = slot
                    self._misses = old_misses + 1
                    self._preparing.pop(key, None)
                    pending.event.set()
                    publication_committed = True
                except BaseException as error:
                    if self._slots.get(key) is slot:
                        self._slots.pop(key, None)
                    self._misses = old_misses
                    if self._preparing.get(key) is pending:
                        self._preparing.pop(key, None)
                    pending.error = error
                    pending.event.set()
                    raise
        except BaseException as error:
            if not publication_committed:
                with self._cache_lock:
                    if self._slots.get(key) is slot:
                        self._slots.pop(key, None)
                        self._misses -= 1
                    if self._preparing.get(key) is pending:
                        self._preparing.pop(key, None)
                    if pending.error is None:
                        pending.error = error
                    pending.event.set()
            slot.lock.release()
            raise
        return slot, False, True


def init(
    *,
    device: Any = None,
    arch: str | None = None,
    timeout_ms: float | None = None,
    max_cached_shapes: int | None = None,
    compile: str = "lazy",
) -> FlashKMeansAssignRuntime:
    """Create one reusable runtime; shape-specific plans remain lazy."""

    return FlashKMeansAssignRuntime(
        device=device,
        arch=arch,
        timeout_ms=timeout_ms,
        max_cached_shapes=max_cached_shapes,
        compile=compile,
    )


def _require_aux_tensor(
    tensor: Any,
    *,
    name: str,
    shape: tuple[int, ...],
    dtype: Any,
    device: Any,
) -> None:
    if tuple(tensor.shape) != shape or tensor.dtype is not dtype:
        raise ValueError(f"{name} must have dtype {dtype} and shape {shape}")
    if tensor.device != device or not tensor.is_contiguous():
        raise ValueError(f"{name} must be contiguous and on {device}")


def _record_stream(stream: Any, *tensors: Any) -> None:
    seen: set[int] = set()
    for tensor in tensors:
        identity = id(tensor)
        if identity in seen:
            continue
        seen.add(identity)
        record_stream = getattr(tensor, "record_stream", None)
        if callable(record_stream):
            record_stream(stream)


def _record_stream_inputs(
    stream: Any,
    inputs: dict[str, Any],
    fields: tuple[str, ...],
) -> None:
    seen: set[int] = set()
    for name in fields:
        tensor = inputs[name]
        identity = id(tensor)
        if identity in seen:
            continue
        seen.add(identity)
        record_stream = getattr(tensor, "record_stream", None)
        if callable(record_stream):
            record_stream(stream)


def _rebind_stream_bound_norm_inputs(slot: _RuntimeSlot) -> None:
    """Update one validated slot's prepared norm pointer carriers in place."""

    if not slot.norm_rebind_program:
        raise RuntimeError("cached KMeans norm launch has no fixed-stream rebind program")
    pointer_updates = tuple(
        (carrier, int(slot.inputs[name].data_ptr()))
        for name, carrier in slot.norm_rebind_program
    )
    for carrier, pointer in pointer_updates:
        carrier.value = pointer


def _prepare_stream_bound_norm_rebind(slot: _RuntimeSlot) -> None:
    """Scrub cold norm callers and retain only the slot's dynamic carriers."""

    all_bindings = ((0, "x"), (1, "centroids"), (2, "x_sq"), (3, "c_sq"))
    full_program = slot.norm_plan.launch_plan._scrub_stream_bound_pointer_keepalives(
        all_bindings
    )
    dynamic_names = {
        "x",
        "centroids",
        *(("x_sq",) if not slot.internal_x_sq else ()),
        *(("c_sq",) if not slot.internal_c_sq else ()),
    }
    slot.norm_rebind_program = tuple(
        (name, carrier) for name, carrier in full_program if name in dynamic_names
    )


def _prepare_inputs(
    x: Any,
    centroids: Any,
    *,
    out: Any | None,
    x_sq: Any | None,
    c_sq: Any | None,
    device_index: int | None,
    arch: str | None,
    stream: Any,
    arch_is_validated: bool = False,
    timeout_ms: float | None = None,
    defer_missing_norms: bool = False,
    _context_is_active: bool = False,
    _torch: Any = None,
) -> tuple[dict[str, Any], Any, str]:
    if _torch is None:
        import torch
    else:
        torch = _torch

    if not all(isinstance(item, torch.Tensor) and item.is_cuda for item in (x, centroids)):
        raise TypeError("x and centroids must be CUDA torch.Tensor objects")
    if x.dtype is not torch.bfloat16 or centroids.dtype is not torch.bfloat16:
        raise TypeError("x and centroids dtype must be bfloat16")
    if x.ndim != 3 or centroids.ndim != 3 or not x.is_contiguous() or not centroids.is_contiguous():
        raise ValueError("x and centroids must be contiguous [B, rows, D] tensors")
    bsz, n_points, dim = map(int, x.shape)
    c_bsz, n_clusters, c_dim = map(int, centroids.shape)
    if (bsz, dim) != (c_bsz, c_dim) or x.device != centroids.device:
        raise ValueError("x and centroids batch/feature dimensions and device must match")
    input_device_index = x.device.index
    if input_device_index is None:
        input_device_index = torch.cuda.current_device()
    input_device_index = int(input_device_index)
    if device_index is not None and input_device_index != int(device_index):
        raise ValueError(
            f"Flash-KMeans runtime targets CUDA device {device_index}, got input device {input_device_index}"
        )
    with _device_stream_context(
        torch,
        input_device_index,
        stream,
        context_is_active=_context_is_active,
    ) as resolved_stream:
        if arch_is_validated:
            if arch is None:
                raise RuntimeError("validated Flash-KMeans runtime must provide its active arch")
            resolved_arch = str(arch)
        else:
            detected_arch = str(detect_gpu_arch())
            resolved_arch = detected_arch if arch is None else str(arch)
            if resolved_arch != detected_arch:
                raise ValueError(
                    "Flash-KMeans launch arch must match the active device: "
                    f"requested {resolved_arch}, detected {detected_arch}"
                )
        caller_supplied_out = out is not None
        if not caller_supplied_out:
            out = torch.empty((bsz, n_points), dtype=torch.int32, device=x.device)
        else:
            _require_aux_tensor(
                out,
                name="out",
                shape=(bsz, n_points),
                dtype=torch.int32,
                device=x.device,
            )
        compute_x_sq = x_sq is None
        compute_c_sq = c_sq is None
        if not defer_missing_norms:
            if compute_x_sq:
                x_sq = torch.empty((bsz, n_points), dtype=torch.float32, device=x.device)
            if compute_c_sq:
                c_sq = torch.empty((bsz, n_clusters), dtype=torch.float32, device=x.device)
        if x_sq is not None:
            _require_aux_tensor(
                x_sq,
                name="x_sq",
                shape=(bsz, n_points),
                dtype=torch.float32,
                device=x.device,
            )
        if c_sq is not None:
            _require_aux_tensor(
                c_sq,
                name="c_sq",
                shape=(bsz, n_clusters),
                dtype=torch.float32,
                device=x.device,
            )
        if (compute_x_sq or compute_c_sq) and not defer_missing_norms:
            # This single custom launch converts BF16, squares, and reduces
            # both row sets without PyTorch conversion/reduction temporaries.
            from ._row_norm import launch_bf16_pair_row_norm

            _record_stream(resolved_stream, x, centroids, x_sq, c_sq, out)
            launch_bf16_pair_row_norm(
                x,
                centroids,
                x_sq,
                c_sq,
                compute_x=compute_x_sq,
                compute_c=compute_c_sq,
                stream=resolved_stream,
                arch=resolved_arch,
                timeout_ms=timeout_ms,
            )
        norm_mode = (
            "fused_bf16_pair_row_norm"
            if compute_x_sq and compute_c_sq
            else "mixed_fused_bf16_pair_row_norm"
            if compute_x_sq or compute_c_sq
            else "explicit_precomputed"
        )
        inputs = {
            "B": bsz,
            "N": n_points,
            "D": dim,
            "K": n_clusters,
            "dtype": "bfloat16",
            "x": x,
            "centroids": centroids,
            "x_sq": x_sq,
            "c_sq": c_sq,
            "out": out,
            "_norm_mode": norm_mode,
        }
    return inputs, resolved_stream, resolved_arch


def prepare_flash_kmeans_assign(
    x: Any,
    centroids: Any,
    *,
    out: Any | None = None,
    x_sq: Any | None = None,
    c_sq: Any | None = None,
    arch: str | None = None,
    stream: Any = None,
    timeout_ms: float | None = None,
) -> PreparedFlashKMeansAssign:
    """Validate and prepare an allocation-free direct launch for exact tensors."""

    timeout_ms = _validate_timeout(timeout_ms)

    inputs, resolved_stream, resolved_arch = _prepare_inputs(
        x,
        centroids,
        out=out,
        x_sq=x_sq,
        c_sq=c_sq,
        device_index=None,
        arch=arch,
        stream=stream,
        timeout_ms=timeout_ms,
    )
    import torch

    input_device_index = inputs["x"].device.index
    device_index = int(torch.cuda.current_device() if input_device_index is None else input_device_index)
    with torch.cuda.device(device_index), torch.cuda.stream(resolved_stream):
        launch_plan = _dispatcher.prepare_launch_plan(
            inputs,
            arch=resolved_arch,
            stream=resolved_stream,
            timeout_ms=timeout_ms,
        )
    return PreparedFlashKMeansAssign(
        inputs=inputs,
        launch_plan=launch_plan,
        stream=resolved_stream,
        timeout_ms=timeout_ms,
    )


def flash_kmeans_assign_prepared(
    prepared: PreparedFlashKMeansAssign,
    *,
    stream: Any = None,
    timeout_ms: float | None = None,
    return_info: bool = False,
):
    """Submit a prepared plan without route selection, allocation, or packing."""

    if not isinstance(prepared, PreparedFlashKMeansAssign):
        raise TypeError("prepared must be returned by prepare_flash_kmeans_assign")
    effective_timeout_ms = prepared.timeout_ms if timeout_ms is None else _validate_timeout(timeout_ms)
    resolved_stream = prepared.stream if stream is None else stream
    _record_stream(
        resolved_stream,
        *(prepared.inputs[name] for name in ("x", "centroids", "x_sq", "c_sq", "out")),
    )
    result = _dispatcher.launch_prepared(
        prepared.launch_plan,
        stream=resolved_stream,
        timeout_ms=effective_timeout_ms,
        public_inputs_already_recorded=True,
    )
    produced = result.get("cluster_ids", prepared.out) if isinstance(result, dict) else prepared.out
    if produced is not prepared.out:
        prepared.out.copy_(produced)
    if not return_info:
        return prepared.out
    info = {
        "semantic_entrypoint": SEMANTIC_ENTRYPOINT,
        "selected_route": prepared.selected_route,
        "launch_entrypoint": prepared.launch_plan.launch_entrypoint,
        "exact_launch_plan": True,
        "prepared_launch_count": prepared.launch_plan.launch_count,
        "arch": prepared.launch_plan.arch,
        "device_index": prepared.launch_plan.device_index,
        "stream_handle": prepared.launch_plan.stream_handle,
    }
    return prepared.out, info


def flash_kmeans_assign(
    x: Any,
    centroids: Any,
    *,
    out: Any | None = None,
    x_sq: Any | None = None,
    c_sq: Any | None = None,
    arch: str | None = None,
    stream: Any = None,
    timeout_ms: float | None = None,
    return_info: bool = False,
):
    """Assign points to centroids through a cold prepared/direct launch."""
    timeout_ms = _validate_timeout(timeout_ms)
    with launch_context(arch=arch, stream=stream, timeout_ms=timeout_ms):
        prepared = prepare_flash_kmeans_assign(
            x,
            centroids,
            out=out,
            x_sq=x_sq,
            c_sq=c_sq,
            arch=arch,
            stream=stream,
            timeout_ms=timeout_ms,
        )
        return flash_kmeans_assign_prepared(
            prepared,
            stream=prepared.stream,
            return_info=return_info,
        )
