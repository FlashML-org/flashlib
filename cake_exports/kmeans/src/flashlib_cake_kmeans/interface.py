from __future__ import annotations

import math
from collections import OrderedDict
from contextlib import contextmanager
from dataclasses import dataclass, field
from functools import wraps
from threading import Condition, Event, RLock
from typing import Any

from ._dispatch import flash_kmeans_assign_dispatcher as _dispatcher
from ._dispatch_runtime import dispatch_launch_options
from ._launch_plan import (
    GraphCaptureUnsupported,
    build_graph_exec_plan,
    build_launch_plan,
    defer_graph_destroy,
    drain_graph_graveyard,
)
from ._runtime import detect_gpu_arch, runtime_activity_snapshot

SEMANTIC_ENTRYPOINT = "loom.examples.weave.flash_kmeans_assign_dispatcher:launch_for_eval"
_PREPARE_LOCK = RLock()


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


@dataclass(frozen=True)
class _ResolvedRoute:
    """Frozen-cascade decision adapted to the shared plan-constructor contract.

    ``launch`` submits through the root dispatcher, which re-derives exactly
    this decision from the same frozen guard cascade — the cascade remains the
    single source of routing truth and runs once per signature, at plan
    construction (or per call for host-data-dependent routes).
    """

    route_id: str
    launch_entrypoint: str
    exact_contract: bool = True

    def launch(
        self,
        inputs: dict[str, Any],
        *,
        stream: Any = None,
        timeout_ms: float | None = None,
    ) -> Any:
        with dispatch_launch_options(stream=stream, timeout_ms=timeout_ms):
            return _dispatcher.launch_for_eval(inputs)


@dataclass
class _RuntimeSlot:
    """One stream-bound, signature-specialized LaunchPlan plus support state.

    ``plan`` freezes the resolved route's leaf launches — including any
    zero-pad staging kernels the route delegates through — or keeps the cached
    per-call launcher for host-data-dependent routes. ``norm_plan`` is the
    route-required fused BF16 pair row-norm support launch, bound by pointer
    overwrite before the plan; its outputs are the slot-owned ``x_sq``/``c_sq``
    scratch with stable pointers. ``graph`` is the signature's captured CUDA
    graph over the full norm + route kernel chain; when set, a hot call binds
    pointers host-side and replays the graph instead of submitting the
    prepared launches one by one (``graph_capture_error`` records why a frozen
    plan stayed on the prepared path). ``default_outputs`` alternates between
    two plan-owned tensors so consecutive default-output calls never hand the
    caller the allocation the previous call returned. ``default_flip`` is only
    read or written while ``lock`` is held.
    """

    plan: Any
    route: Any
    norm_plan: Any
    x_sq: Any
    c_sq: Any
    internal_x_sq: bool
    internal_c_sq: bool
    norm_compute_fields: tuple[str, ...]
    default_outputs: tuple[Any, Any]
    graph: Any = None
    graph_capture_error: str | None = None
    lock: RLock = field(default_factory=RLock, repr=False)
    default_flip: int = 0


@dataclass
class _PendingPreparation:
    event: Event = field(default_factory=Event, repr=False)
    error: BaseException | None = field(default=None, repr=False)


def _guard_runtime_compute(method: Any) -> Any:
    """Keep clear() from crossing an in-progress lookup/rebind/enqueue call."""

    @wraps(method)
    def guarded(self: Any, *args: Any, **kwargs: Any) -> Any:
        with self._lifecycle:
            while self._clearing:
                self._lifecycle.wait()
            self._active_computes += 1
        try:
            return method(self, *args, **kwargs)
        finally:
            with self._lifecycle:
                self._active_computes -= 1
                if self._active_computes == 0:
                    self._lifecycle.notify_all()

    return guarded


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


@contextmanager
def _device_stream_context(torch: Any, device_index: int, stream: Any):
    """Enter the device, resolve one stream, and enter that stream."""

    with torch.cuda.device(device_index):
        resolved_stream = _resolve_stream(torch, device_index, stream)
        with torch.cuda.stream(resolved_stream):
            yield resolved_stream


def _signature_alias_topology(
    x: Any,
    centroids: Any,
    x_sq: Any,
    c_sq: Any,
    out: Any,
) -> tuple[int, ...]:
    """Canonical pointer-equivalence classes independent of concrete addresses.

    A default output comes from the signature slot's plan-owned pool, which
    can never alias a live caller tensor, so ``out is None`` contributes a
    fresh class without reading any pointer.
    """

    classes: dict[int, int] = {}
    topology: list[int] = []
    for tensor in (x, centroids, x_sq, c_sq):
        if tensor is None:
            topology.append(-1)
            continue
        pointer = int(tensor.data_ptr())
        if pointer not in classes:
            classes[pointer] = len(classes)
        topology.append(classes[pointer])
    if out is None:
        topology.append(len(classes))
    else:
        pointer = int(out.data_ptr())
        if pointer not in classes:
            classes[pointer] = len(classes)
        topology.append(classes[pointer])
    return tuple(topology)


def _require_owned_output(result: Any, inputs: dict[str, Any]) -> None:
    """Reject routes whose hot path would require an uncaptured output copy."""

    produced = result.get("cluster_ids") if isinstance(result, dict) else result
    if produced is None:
        return
    out = inputs["out"]
    if produced is not out and int(produced.data_ptr()) != int(out.data_ptr()):
        raise RuntimeError(
            "prepared Flash-KMeans route must write the caller-owned output through captured launches"
        )


def _validate_compute_tensors(
    torch: Any,
    x: Any,
    centroids: Any,
    *,
    device_index: int,
) -> tuple[int, int, int, int]:
    """Validate the public compute ABI without allocating anything."""

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
    if int(input_device_index) != int(device_index):
        raise ValueError(
            f"Flash-KMeans runtime targets CUDA device {device_index}, got input device {input_device_index}"
        )
    return bsz, n_points, dim, n_clusters


class FlashKMeansAssignRuntime:
    """Long-lived KMeans assignment runtime with per-shape, per-stream plans.

    ``compute`` accepts new tensors and new shapes. The first call for one
    signature runs the frozen guard cascade once, freezes the resolved
    route's exact leaf launches (plus the route-required fused pair row-norm
    support launch) into a per-signature plan, and captures the full kernel
    chain into one CUDA graph. Cache hits overwrite the recorded pointer
    carriers in place and replay the graph on the plan's stream — no dispatch
    re-evaluation, no argument re-marshalling, no per-launch stream query,
    and no per-call default-output allocation (defaults ping-pong between two
    plan-owned tensors, so consecutive default-output calls never alias).
    Workspace is never shared across CUDA streams.
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

    def cache_info(self) -> dict[str, int | None]:
        with self._cache_lock:
            return {
                "size": len(self._slots),
                "hits": self._hits,
                "misses": self._misses,
                "max_cached_shapes": self.max_cached_shapes,
            }

    @_guard_runtime_compute
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
        """Run one KMeans assignment through this signature's launch plan.

        A cache miss runs the frozen guard cascade once and freezes its exact
        leaf launches into a per-signature plan with a captured CUDA graph.
        Cache hits are host-side pointer binding plus one graph replay on the
        plan's stream. A caller-provided ``out=`` is written through; default
        outputs come from the signature's plan-owned pool, so a default-output
        result must be consumed before two further default-output calls of the
        same signature reuse its storage.
        """

        torch = self._torch
        # Graphs dropped by an earlier clear(synchronize=False) destroy here
        # once their completion events fire; an empty graveyard is a single
        # truthiness check.
        drain_graph_graveyard()
        bsz, n_points, dim, n_clusters = _validate_compute_tensors(
            torch,
            x,
            centroids,
            device_index=self.device_index,
        )
        effective_timeout_ms = self.timeout_ms if timeout_ms is None else _validate_timeout(timeout_ms)
        resolved_stream = _resolve_stream(torch, self.device_index, stream)
        stream_handle = int(resolved_stream.cuda_stream)
        if out is not None:
            _require_aux_tensor(
                out,
                name="out",
                shape=(bsz, n_points),
                dtype=torch.int32,
                device=x.device,
            )
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
        key = (
            self.device_index,
            self.arch,
            bsz,
            n_points,
            dim,
            n_clusters,
            "bfloat16",
            x_sq is None,
            c_sq is None,
            _signature_alias_topology(x, centroids, x_sq, c_sq, out),
            stream_handle,
        )
        activity_before = runtime_activity_snapshot() if return_info else None
        with self._cache_lock:
            slot = self._slots.get(key)
            if slot is not None:
                self._hits += 1
        cache_hit = slot is not None
        owns_slot_lock = False
        if slot is None:
            slot, owns_slot_lock = self._create_slot(
                key,
                x=x,
                centroids=centroids,
                out=out,
                x_sq=x_sq,
                c_sq=c_sq,
                shape=(bsz, n_points, dim, n_clusters),
                stream=resolved_stream,
            )
            cache_hit = not owns_slot_lock
        activity_after = runtime_activity_snapshot() if return_info else None
        if not owns_slot_lock:
            slot.lock.acquire()
        try:
            if slot.plan.stream_handle != stream_handle:
                raise RuntimeError("cached Flash-KMeans slot changed its prepared stream")
            if out is None:
                bound_out = slot.default_outputs[slot.default_flip]
            else:
                bound_out = out
            bindings = {
                "x": x,
                "centroids": centroids,
                "x_sq": slot.x_sq if slot.internal_x_sq else x_sq,
                "c_sq": slot.c_sq if slot.internal_c_sq else c_sq,
                "out": bound_out,
            }
            try:
                # Bind first (tensor-map refresh + pointer overwrite are host
                # work), then enqueue the norm support launch, then submit
                # the plan's kernels. Binding after the norm would turn a
                # fresh-pointer tensor-map re-encode into a GPU inter-kernel
                # gap between the norm kernel and the route's first kernel.
                # A captured signature does every bind host-side, then one
                # graph replay covers the whole norm + route kernel chain.
                if slot.graph is not None:
                    slot.plan.bind_hot(bindings)
                    if slot.norm_plan is not None:
                        slot.norm_plan.bind_hot(x, centroids)
                    slot.graph.submit_hot(timeout_ms=effective_timeout_ms)
                else:
                    slot.plan.bind_hot(bindings)
                    if slot.norm_plan is not None:
                        slot.norm_plan.launch_hot(x, centroids)
                    slot.plan.submit_hot(timeout_ms=effective_timeout_ms)
                # The pool slot is consumed only by a successful submission:
                # a call that fails after selecting its default output must
                # not burn the slot, or the next two successful calls would
                # both return the tensor the last successful call already
                # handed its caller.
                if out is None:
                    slot.default_flip ^= 1
            finally:
                # Slot-owned norm scratch and pooled default outputs are
                # allocated on the plan's stream and only released through
                # clear(); caller-provided tensors may live on another
                # stream, so record those for allocator safety even on
                # partial submission.
                _record_stream(resolved_stream, x, centroids, x_sq, c_sq, out)
        finally:
            slot.lock.release()

        if not return_info:
            return bound_out
        norm_mode = (
            "fused_bf16_pair_row_norm:" + ",".join(slot.norm_compute_fields)
            if slot.norm_compute_fields
            else "route_elided_internal_norms"
            if slot.internal_x_sq or slot.internal_c_sq
            else "explicit_precomputed"
        )
        cold_activity = {
            "source_read_occurred": bool(activity_after["source_reads"] > activity_before["source_reads"]),
            "nvrtc_compile_occurred": bool(
                activity_after["nvrtc_compiles"] > activity_before["nvrtc_compiles"]
            ),
            "module_load_occurred": bool(activity_after["module_loads"] > activity_before["module_loads"]),
            "scratch_allocation_occurred": not cache_hit,
        }
        return bound_out, {
            "semantic_entrypoint": SEMANTIC_ENTRYPOINT,
            "selected_route": slot.route.route_id,
            "launch_entrypoint": slot.route.launch_entrypoint,
            "exact_launch_plan": slot.route.exact_contract,
            "runtime_cache_hit": cache_hit,
            "assignment_launch_count": int(slot.plan.launch_count),
            "norm_launch_count": int(slot.norm_plan is not None),
            "norm_compute_fields": list(slot.norm_compute_fields),
            "runtime_launch_count": int(slot.plan.launch_count) + int(slot.norm_plan is not None),
            "norm_mode": norm_mode,
            "hot_launch_path": "cuda_graph" if slot.graph is not None else "prepared_launches",
            "graph_kernel_count": None if slot.graph is None else int(slot.graph.launch_count),
            "graph_capture_error": slot.graph_capture_error,
            "arch": self.arch,
            "device_index": self.device_index,
            "stream_handle": stream_handle,
            "cold_first_call_activity": cold_activity,
        }

    def _create_slot(
        self,
        key: tuple[Any, ...],
        *,
        x: Any,
        centroids: Any,
        out: Any | None,
        x_sq: Any | None,
        c_sq: Any | None,
        shape: tuple[int, int, int, int],
        stream: Any,
    ) -> tuple[_RuntimeSlot, bool]:
        """Construct and publish this signature's LaunchPlan (the slow path).

        Returns ``(slot, owns_slot_lock)``. A newly published slot is returned
        with its lock held so the constructing call launches first; when
        another thread published the slot while this one waited, the existing
        slot is returned unlocked and counted as a cache hit. Preparation
        never holds the runtime cache lock, so resident hot shapes stay
        unblocked while a cold signature captures.
        """

        while True:
            with self._cache_lock:
                slot = self._slots.get(key)
                if slot is not None:
                    self._hits += 1
                    return slot, False
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
            import torch

            from ._row_norm import prepare_bf16_pair_row_norm

            bsz, n_points, dim, n_clusters = shape
            internal_x_sq = x_sq is None
            internal_c_sq = c_sq is None
            with torch.cuda.device(self.device_index), torch.cuda.stream(stream):
                default_outputs = tuple(
                    torch.empty((bsz, n_points), dtype=torch.int32, device=x.device) for _pair in range(2)
                )
                slot_x_sq = (
                    torch.empty((bsz, n_points), dtype=torch.float32, device=x.device)
                    if internal_x_sq
                    else x_sq
                )
                slot_c_sq = (
                    torch.empty((bsz, n_clusters), dtype=torch.float32, device=x.device)
                    if internal_c_sq
                    else c_sq
                )
                capture_out = out if out is not None else default_outputs[0]
                inputs = {
                    "B": bsz,
                    "N": n_points,
                    "D": dim,
                    "K": n_clusters,
                    "dtype": "bfloat16",
                    "x": x,
                    "centroids": centroids,
                    "x_sq": slot_x_sq,
                    "c_sq": slot_c_sq,
                    "out": capture_out,
                }
                with _PREPARE_LOCK:
                    decision = _dispatcher.select_route(inputs)
                    route = _ResolvedRoute(
                        route_id=str(decision.route_id),
                        launch_entrypoint=str(decision.entrypoint),
                    )
                    plan = build_launch_plan(
                        inputs,
                        stream=stream,
                        arch=self.arch,
                        validate_result=_require_owned_output,
                        route=route,
                    )
                bound_keys = getattr(plan, "bound_input_keys", None)
                bound = None if bound_keys is None else set(bound_keys)
                # A frozen plan reports exactly which public inputs its
                # captured launches bind; routes whose kernels never read a
                # norm skip computing it. A per-call plan re-executes the
                # route's host program, so provide every internal norm it
                # could consume.
                compute_x_sq = internal_x_sq and (bound is None or "x_sq" in bound)
                compute_c_sq = internal_c_sq and (bound is None or "c_sq" in bound)
                norm_plan = None
                if compute_x_sq or compute_c_sq:
                    norm_plan = prepare_bf16_pair_row_norm(
                        x,
                        centroids,
                        slot_x_sq,
                        slot_c_sq,
                        compute_x=compute_x_sq,
                        compute_c=compute_c_sq,
                        arch=self.arch,
                        stream=stream,
                    )
                # Capture the signature's stable kernel chain (the fused pair
                # norm first, then the frozen route launches — the exact hot
                # submission order) into one CUDA graph. Per-call routes and
                # launch modes without a validated capture path stay on the
                # prepared-launch hot path, with the reason recorded; any
                # other capture failure is an error, not a fallback.
                graph_plan = None
                graph_capture_error: str | None = None
                try:
                    graph_plan = build_graph_exec_plan(
                        plan,
                        support_launches=() if norm_plan is None else (norm_plan.launch_plan,),
                    )
                except GraphCaptureUnsupported as unsupported:
                    graph_capture_error = str(unsupported)
                # The slot must not pin the creating call's tensors: the norm
                # plan's caller-owned bindings (x, centroids, and any
                # non-computed norm field) release to slot-owned scratch;
                # every hot call rewrites the input carriers before it
                # submits. ``build_launch_plan`` already released the main
                # plan's bound inputs the same way.
                if norm_plan is not None:
                    norm_plan.release_bound_callers(
                        slot_x_sq if compute_x_sq else slot_c_sq,
                        stream=stream,
                    )
            norm_compute_fields = tuple(
                name for name, enabled in (("x_sq", compute_x_sq), ("c_sq", compute_c_sq)) if enabled
            )
            slot = _RuntimeSlot(
                plan=plan,
                route=route,
                norm_plan=norm_plan,
                # Slot-owned norm buffers only: a caller-supplied norm tensor
                # is rebound per call and must not be pinned for the slot's
                # cache lifetime (the hot path never reads these fields for
                # non-internal signatures).
                x_sq=slot_x_sq if internal_x_sq else None,
                c_sq=slot_c_sq if internal_c_sq else None,
                internal_x_sq=internal_x_sq,
                internal_c_sq=internal_c_sq,
                norm_compute_fields=norm_compute_fields,
                default_outputs=default_outputs,
                graph=graph_plan,
                graph_capture_error=graph_capture_error,
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
        return slot, True

    def clear(self, *, synchronize: bool = True) -> None:
        """Exclusively release slots after admitted host calls finish.

        The default waits for device completion and destroys the driver graph
        handles eagerly. With ``synchronize=False``, every plan-held launch
        argument, slot-owned norm buffer, and pooled default output is tied to
        its plan's stream via ``record_stream`` before release, and each
        slot's graph handles go to the shared event-gated graveyard (an
        executing graph must never be destroyed underneath the device) —
        destruction happens on later calls or on a final synchronizing
        clear; the caller remains responsible for observing asynchronous
        completion of the released work itself.
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
                    if synchronize:
                        for slot in self._slots.values():
                            if slot.graph is not None:
                                slot.graph.destroy()
                    else:
                        for slot in self._slots.values():
                            plan_stream = slot.plan.torch_stream
                            slot.plan.record_stream(plan_stream)
                            if slot.norm_plan is not None:
                                slot.norm_plan.record_stream(plan_stream)
                            _record_stream(
                                plan_stream,
                                slot.x_sq,
                                slot.c_sq,
                                *slot.default_outputs,
                            )
                            if slot.graph is not None:
                                defer_graph_destroy(slot.graph)
                    self._slots.clear()
                    self._hits = 0
                    self._misses = 0
                # Either mode also reaps graphs earlier clears deferred:
                # after a synchronizing clear their events have fired.
                drain_graph_graveyard()
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
        if tensor is None:
            continue
        identity = id(tensor)
        if identity in seen:
            continue
        seen.add(identity)
        record_stream = getattr(tensor, "record_stream", None)
        if callable(record_stream):
            record_stream(stream)


def _tensor_device_index(tensor: Any) -> int:
    import torch

    index = tensor.device.index
    return int(torch.cuda.current_device() if index is None else index)


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
    timeout_ms: float | None = None,
) -> tuple[dict[str, Any], Any, str]:
    import torch

    if not all(isinstance(item, torch.Tensor) and item.is_cuda for item in (x, centroids)):
        raise TypeError("x and centroids must be CUDA torch.Tensor objects")
    resolved_device_index = _tensor_device_index(x) if device_index is None else int(device_index)
    bsz, n_points, dim, n_clusters = _validate_compute_tensors(
        torch,
        x,
        centroids,
        device_index=resolved_device_index,
    )
    with _device_stream_context(torch, resolved_device_index, stream) as resolved_stream:
        detected_arch = str(detect_gpu_arch())
        resolved_arch = detected_arch if arch is None else str(arch)
        if resolved_arch != detected_arch:
            raise ValueError(
                "Flash-KMeans launch arch must match the active device: "
                f"requested {resolved_arch}, detected {detected_arch}"
            )
        if out is None:
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
        if compute_x_sq:
            x_sq = torch.empty((bsz, n_points), dtype=torch.float32, device=x.device)
        if compute_c_sq:
            c_sq = torch.empty((bsz, n_clusters), dtype=torch.float32, device=x.device)
        _require_aux_tensor(
            x_sq,
            name="x_sq",
            shape=(bsz, n_points),
            dtype=torch.float32,
            device=x.device,
        )
        _require_aux_tensor(
            c_sq,
            name="c_sq",
            shape=(bsz, n_clusters),
            dtype=torch.float32,
            device=x.device,
        )
        if compute_x_sq or compute_c_sq:
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


_DEFAULT_RUNTIME_LOCK = RLock()
_DEFAULT_RUNTIMES: dict[int, FlashKMeansAssignRuntime] = {}


def _default_runtime(device_index: int) -> FlashKMeansAssignRuntime:
    """Return the process-wide per-device runtime backing ``flash_kmeans_assign``."""

    runtime = _DEFAULT_RUNTIMES.get(device_index)
    if runtime is None:
        with _DEFAULT_RUNTIME_LOCK:
            runtime = _DEFAULT_RUNTIMES.get(device_index)
            if runtime is None:
                runtime = FlashKMeansAssignRuntime(device=device_index)
                _DEFAULT_RUNTIMES[device_index] = runtime
    return runtime


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
    """Assign points to centroids through the per-device signature plan cache.

    The first call for a signature runs the frozen guard cascade once to
    construct its launch plan and CUDA graph; subsequent calls are
    pointer-overwrite graph replays on the plan's stream. A caller-provided
    ``out=`` is written through; default outputs come from the signature's
    plan-owned pool, so a default-output result must be consumed (or cloned)
    before two further default-output calls of the same signature overwrite
    its storage.
    """
    import torch

    if not isinstance(x, torch.Tensor) or not getattr(x, "is_cuda", False):
        raise TypeError("x must be a CUDA torch.Tensor")
    runtime = _default_runtime(_tensor_device_index(x))
    if arch is not None and str(arch) != runtime.arch:
        raise ValueError(
            f"Flash-KMeans launch arch must match the active device: requested {arch}, detected {runtime.arch}"
        )
    return runtime.compute(
        x,
        centroids,
        out=out,
        x_sq=x_sq,
        c_sq=c_sq,
        stream=stream,
        timeout_ms=timeout_ms,
        return_info=return_info,
    )
