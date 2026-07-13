from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass, field
from functools import wraps
from threading import Condition, Event, RLock
from typing import Any

from . import _direct_plan as _direct_plan_runtime
from ._direct_plan import PreparedDirectRoute, prepare_route
from ._launch_plan import (
    GraphCaptureUnsupported,
    build_graph_exec_plan,
    build_launch_plan,
    defer_graph_destroy,
    drain_graph_graveyard,
)

SEMANTIC_ENTRYPOINT = (
    "loom.examples.weave.knn_build_dispatch_q1m524_v10_d320recurrence_consumption_v1:launch_from_contract_inputs"
)
_PREPARE_LOCK = RLock()


@dataclass(frozen=True)
class PreparedKNNBuild:
    """Fixed inputs and a direct route resolved once outside the hot path."""

    inputs: dict[str, Any]
    launch_plan: PreparedDirectRoute
    shape_label: str | None
    stream: Any = None
    timeout_ms: float | None = None

    @property
    def selected_route(self) -> str:
        return self.launch_plan.route_id


@dataclass
class _KNNBuildRuntimeSlot:
    """One stream-bound, signature-specialized LaunchPlan plus support state.

    ``plan`` freezes the resolved route's leaf launches (or keeps the cached
    per-call launcher for host-data-dependent routes). ``norm_launches`` are
    the route-required fused row-norm support launches, submitted by pointer
    overwrite before the plan; their outputs are the slot-owned ``query_sq``/
    ``database_sq`` scratch with stable pointers. ``graph`` is the signature's
    captured CUDA graph over the full norm + route kernel chain; when set,
    a hot call binds pointers host-side and replays the graph instead of
    submitting the prepared launches one by one (``graph_capture_error``
    records why a frozen plan stayed on the prepared path). ``default_outputs``
    alternates between two plan-owned pairs so consecutive default-output
    calls never hand the caller the allocation the previous call returned.
    ``default_flip`` is only read or written while ``lock`` is held.
    """

    plan: Any
    route: Any
    norm_launches: tuple[tuple[str, Any], ...]
    query_sq: Any
    database_sq: Any
    internal_query_norm: bool
    internal_database_norm: bool
    norm_compute_fields: tuple[str, ...]
    default_outputs: tuple[tuple[Any, Any], tuple[Any, Any]]
    graph: Any = None
    graph_capture_error: str | None = None
    lock: Any = field(default_factory=RLock, repr=False)
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
            self._active_calls += 1
        try:
            return method(self, *args, **kwargs)
        finally:
            with self._lifecycle:
                self._active_calls -= 1
                if self._active_calls == 0:
                    self._lifecycle.notify_all()

    return guarded


def _prepare_inputs(
    query: Any,
    database: Any,
    k: int,
    *,
    build: bool,
    shape_label: str | None,
    out: tuple[Any, Any] | None,
    query_sq: Any = None,
    database_sq: Any = None,
    defer_missing_norms: bool = False,
) -> dict[str, Any]:
    """Validate the public ABI and allocate one fixed set of intermediates."""
    import torch

    if not isinstance(query, torch.Tensor) or not query.is_cuda:
        raise TypeError("query must be a CUDA torch.Tensor")
    if not isinstance(database, torch.Tensor) or not database.is_cuda:
        raise TypeError("database must be a CUDA torch.Tensor")
    if query.dtype not in (torch.bfloat16, torch.float16) or database.dtype != query.dtype:
        raise TypeError("query and database must have the same bfloat16 or float16 dtype")
    if query.ndim != 3 or not query.is_contiguous():
        raise ValueError("query must be contiguous with shape [B, Q, D]")
    if database.ndim != 3 or not database.is_contiguous():
        raise ValueError("database must be contiguous with shape [B, N, D]")
    bsz, n_query, dim = map(int, query.shape)
    db_bsz, n_database, db_dim = map(int, database.shape)
    if (db_bsz, db_dim) != (bsz, dim) or query.device != database.device:
        raise ValueError("query and database batch/feature dimensions and device must match")
    if build and (n_query != n_database or query.data_ptr() != database.data_ptr()):
        raise ValueError("build=True requires query to alias database and Q == M")
    k = int(k)
    if not 0 < k <= n_database:
        raise ValueError(f"k must be in [1, {n_database}], got {k}")
    expected = (bsz, n_query, k)
    if out is None:
        out = (
            torch.empty(expected, dtype=torch.float32, device=database.device),
            torch.empty(expected, dtype=torch.int32, device=database.device),
        )
    out_dists, out_indices = out
    if tuple(out_dists.shape) != expected or tuple(out_indices.shape) != expected:
        raise ValueError(f"out tensors must have shape {expected}")
    if out_dists.dtype is not torch.float32 or out_indices.dtype is not torch.int32:
        raise TypeError("out must be (float32 distances, int32 indices)")
    if out_dists.device != database.device or out_indices.device != database.device:
        raise ValueError("out tensors must be on the query/database device")
    if not out_dists.is_contiguous() or not out_indices.is_contiguous():
        raise ValueError("out tensors must be contiguous")
    if build:
        if query_sq is None and database_sq is not None:
            _validate_norm("database_sq", database_sq, (bsz, n_database), database.device)
            query_sq = database_sq
        elif query_sq is None and not defer_missing_norms:
            query_sq = (query.float() ** 2).sum(-1).contiguous()
        elif query_sq is not None:
            _validate_norm("query_sq", query_sq, (bsz, n_query), database.device)
        if database_sq is not None and database_sq is not query_sq:
            raise ValueError("build=True requires query_sq and database_sq to alias when both are provided")
        database_sq = query_sq
    else:
        if query_sq is None and not defer_missing_norms:
            query_sq = (query.float() ** 2).sum(-1).contiguous()
        elif query_sq is not None:
            _validate_norm("query_sq", query_sq, (bsz, n_query), database.device)
        if database_sq is None and not defer_missing_norms:
            database_sq = (database.float() ** 2).sum(-1).contiguous()
        elif database_sq is not None:
            _validate_norm("database_sq", database_sq, (bsz, n_database), database.device)
    return {
        "label": shape_label,
        "B": bsz,
        "Q": n_query,
        "M": n_database,
        "D": dim,
        "K": k,
        "dtype": str(database.dtype).removeprefix("torch."),
        "build": bool(build),
        "query": query,
        "database": database,
        "query_sq": query_sq,
        "database_sq": database_sq,
        "out_dists": out_dists,
        "out_indices": out_indices,
    }


def _validate_norm(name: str, value: Any, expected: tuple[int, int], device: Any) -> None:
    import torch

    if not isinstance(value, torch.Tensor) or not value.is_cuda:
        raise TypeError(f"{name} must be a CUDA torch.Tensor")
    if tuple(value.shape) != expected:
        raise ValueError(f"{name} must have shape {expected}")
    if value.dtype is not torch.float32:
        raise TypeError(f"{name} must have float32 dtype")
    if value.device != device:
        raise ValueError(f"{name} must be on the query/database device")
    if not value.is_contiguous():
        raise ValueError(f"{name} must be contiguous")


def _tensor_device_index(tensor: Any) -> int:
    import torch

    index = tensor.device.index
    return int(torch.cuda.current_device() if index is None else index)


def _runtime_device_index(device: Any) -> int:
    import torch

    if device is None:
        return int(torch.cuda.current_device())
    if isinstance(device, int) and not isinstance(device, bool):
        return int(device)
    resolved = torch.device(device)
    if resolved.type != "cuda":
        raise ValueError(f"KNN-build runtime requires a CUDA device, got {resolved}")
    return int(torch.cuda.current_device() if resolved.index is None else resolved.index)


def _validate_timeout(timeout_ms: float | None) -> float | None:
    if timeout_ms is None:
        return None
    value = float(timeout_ms)
    if value <= 0:
        raise ValueError("timeout_ms must be positive")
    return value


def _record_stream_tensors(tensors: tuple[Any, ...], stream: Any) -> None:
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


def _validate_compute_tensors(
    torch: Any,
    query: Any,
    database: Any,
    k: int,
    *,
    build: bool,
    device_index: int,
) -> tuple[int, int, int, int]:
    """Validate the public compute ABI without allocating anything."""

    if not isinstance(query, torch.Tensor) or not query.is_cuda:
        raise TypeError("query must be a CUDA torch.Tensor")
    if not isinstance(database, torch.Tensor) or not database.is_cuda:
        raise TypeError("database must be a CUDA torch.Tensor")
    if query.dtype not in (torch.bfloat16, torch.float16) or database.dtype != query.dtype:
        raise TypeError("query and database must have the same bfloat16 or float16 dtype")
    if query.ndim != 3 or not query.is_contiguous():
        raise ValueError("query must be contiguous with shape [B, Q, D]")
    if database.ndim != 3 or not database.is_contiguous():
        raise ValueError("database must be contiguous with shape [B, N, D]")
    bsz, n_query, dim = map(int, query.shape)
    db_bsz, n_database, db_dim = map(int, database.shape)
    if (db_bsz, db_dim) != (bsz, dim) or query.device != database.device:
        raise ValueError("query and database batch/feature dimensions and device must match")
    if build and (n_query != n_database or query.data_ptr() != database.data_ptr()):
        raise ValueError("build=True requires query to alias database and Q == M")
    input_device_index = _tensor_device_index(query)
    if input_device_index != device_index:
        raise ValueError(
            f"KNN-build runtime is bound to cuda:{device_index}, but query is on cuda:{input_device_index}"
        )
    k = int(k)
    if not 0 < k <= n_database:
        raise ValueError(f"k must be in [1, {n_database}], got {k}")
    return bsz, n_query, n_database, dim


def _validate_compute_outputs(
    torch: Any,
    out: tuple[Any, Any],
    expected: tuple[int, int, int],
    device: Any,
) -> tuple[Any, Any]:
    if not isinstance(out, (tuple, list)) or len(out) != 2:
        raise TypeError("out must be a (distances, indices) pair")
    out_dists, out_indices = out
    if not all(isinstance(item, torch.Tensor) and item.is_cuda for item in (out_dists, out_indices)):
        raise TypeError("out tensors must be CUDA torch.Tensor objects")
    if tuple(out_dists.shape) != expected or tuple(out_indices.shape) != expected:
        raise ValueError(f"out tensors must have shape {expected}")
    if out_dists.dtype is not torch.float32 or out_indices.dtype is not torch.int32:
        raise TypeError("out must be (float32 distances, int32 indices)")
    if out_dists.device != device or out_indices.device != device:
        raise ValueError("out tensors must be on the query/database device")
    if not out_dists.is_contiguous() or not out_indices.is_contiguous():
        raise ValueError("out tensors must be contiguous")
    return out_dists, out_indices


def _validate_compute_norms(
    query_sq: Any,
    database_sq: Any,
    *,
    build: bool,
    bsz: int,
    n_query: int,
    n_database: int,
    device: Any,
) -> tuple[Any, Any]:
    """Apply the public norm-alias rules without computing missing norms."""

    if build:
        if query_sq is None and database_sq is not None:
            _validate_norm("database_sq", database_sq, (bsz, n_database), device)
            query_sq = database_sq
        elif query_sq is not None:
            _validate_norm("query_sq", query_sq, (bsz, n_query), device)
        if database_sq is not None and database_sq is not query_sq:
            raise ValueError("build=True requires query_sq and database_sq to alias when both are provided")
        database_sq = query_sq
    else:
        if query_sq is not None:
            _validate_norm("query_sq", query_sq, (bsz, n_query), device)
        if database_sq is not None:
            _validate_norm("database_sq", database_sq, (bsz, n_database), device)
    return query_sq, database_sq


def _require_owned_outputs(outputs: Any, inputs: dict[str, Any]) -> None:
    """Reject routes whose hot path would require an uncaptured output copy."""

    owned = (inputs["out_dists"], inputs["out_indices"])
    if outputs is None:
        normalized = owned
    elif isinstance(outputs, (tuple, list)) and len(outputs) == 2:
        normalized = (outputs[0], outputs[1])
    elif isinstance(outputs, dict):
        distances = outputs.get("distances", outputs.get("dists", outputs.get("out_dists")))
        indices = outputs.get("indices", outputs.get("idxs", outputs.get("out_indices")))
        if distances is None or indices is None:
            raise TypeError("knn_build dispatcher output dict must contain distances and indices")
        normalized = (distances, indices)
    else:
        raise TypeError("knn_build dispatcher must return (distances, indices), a matching dict, or write outputs")
    if any(source is not destination for destination, source in zip(owned, normalized)):
        raise RuntimeError("prepared KNN-build route must write caller-owned outputs through captured launches")


class KNNBuildRuntime:
    """One device runtime with reusable launch plans keyed by shape and stream."""

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
        self.device_index = _runtime_device_index(device)
        if max_cached_shapes is not None:
            if isinstance(max_cached_shapes, bool) or not isinstance(max_cached_shapes, int):
                raise TypeError("max_cached_shapes must be a positive integer or None")
            if int(max_cached_shapes) <= 0:
                raise ValueError("max_cached_shapes must be positive")
            max_cached_shapes = int(max_cached_shapes)
        self.timeout_ms = _validate_timeout(timeout_ms)
        self.max_cached_shapes = max_cached_shapes
        with torch.cuda.device(self.device_index):
            detected_arch = str(_direct_plan_runtime.detect_gpu_arch())
        self.arch = detected_arch if arch is None else str(arch)
        if self.arch != detected_arch:
            raise ValueError(
                f"KNN-build runtime arch must match its device: requested {self.arch}, detected {detected_arch}"
            )
        self._cache: OrderedDict[tuple[Any, ...], _KNNBuildRuntimeSlot] = OrderedDict()
        self._preparing: dict[tuple[Any, ...], _PendingPreparation] = {}
        self._cache_lock = RLock()
        self._lifecycle = Condition(RLock())
        self._active_calls = 0
        self._clearing = False
        self._hits = 0
        self._misses = 0

    @_guard_runtime_compute
    def compute(
        self,
        query: Any,
        database: Any,
        k: int,
        *,
        build: bool = False,
        shape_label: str | None = None,
        out: tuple[Any, Any] | None = None,
        query_sq: Any = None,
        database_sq: Any = None,
        stream: Any = None,
        timeout_ms: float | None = None,
        return_info: bool = False,
    ):
        """Run one KNN build/search through this signature's launch plan.

        A cache miss runs the frozen guard cascade once and freezes its exact
        leaf launches (plus the route-required fused row-norm support
        launches) into a per-signature plan. Cache hits overwrite the
        recorded pointer carriers in place and submit the prepared launches
        on the plan's stream — no dispatch re-evaluation, no argument
        re-marshalling, no per-launch stream query, and no per-call
        default-output allocation (defaults ping-pong between two plan-owned
        pairs, so consecutive default-output calls never alias). A
        default-output result must therefore be consumed (or cloned) before
        two further successful default-output calls of the same signature
        reuse its storage.
        """

        import torch

        # Graphs dropped by an earlier clear(synchronize=False) destroy here
        # once their completion events fire; an empty graveyard is a single
        # truthiness check.
        drain_graph_graveyard()
        bsz, n_query, n_database, dim = _validate_compute_tensors(
            torch,
            query,
            database,
            k,
            build=build,
            device_index=self.device_index,
        )
        k = int(k)
        build = bool(build)
        effective_timeout_ms = self.timeout_ms if timeout_ms is None else _validate_timeout(timeout_ms)
        if stream is None:
            resolved_stream = torch.cuda.current_stream(self.device_index)
        else:
            resolved_stream = stream
            stream_device = getattr(resolved_stream, "device", None)
            stream_device_index = getattr(stream_device, "index", stream_device)
            if stream_device_index is not None and int(stream_device_index) != self.device_index:
                raise ValueError(
                    f"KNN-build stream device {stream_device_index} does not match runtime device {self.device_index}"
                )
        stream_handle = int(resolved_stream.cuda_stream)
        query_sq, database_sq = _validate_compute_norms(
            query_sq,
            database_sq,
            build=build,
            bsz=bsz,
            n_query=n_query,
            n_database=n_database,
            device=database.device,
        )
        out_pair = (
            None if out is None else _validate_compute_outputs(torch, out, (bsz, n_query, k), database.device)
        )
        if query_sq is None or database_sq is None:
            norm_alias = bool(build and query_sq is None and database_sq is None)
        else:
            norm_alias = int(query_sq.data_ptr()) == int(database_sq.data_ptr())
        dtype_name = "bfloat16" if query.dtype is torch.bfloat16 else "float16"
        key = (
            self.device_index,
            self.arch,
            bsz,
            n_query,
            n_database,
            dim,
            k,
            dtype_name,
            build,
            int(query.data_ptr()) == int(database.data_ptr()),
            query_sq is None,
            database_sq is None,
            norm_alias,
            stream_handle,
        )
        with self._cache_lock:
            slot = self._cache.get(key)
            if slot is not None:
                self._cache.move_to_end(key)
                self._hits += 1
        cache_hit = slot is not None
        owns_slot_lock = False
        if slot is None:
            slot, owns_slot_lock = self._create_slot(
                key,
                query=query,
                database=database,
                k=k,
                build=build,
                shape_label=shape_label,
                dtype_name=dtype_name,
                shape=(bsz, n_query, n_database, dim),
                out_pair=out_pair,
                query_sq=query_sq,
                database_sq=database_sq,
                stream=resolved_stream,
            )
            cache_hit = not owns_slot_lock
        if not owns_slot_lock:
            slot.lock.acquire()
        try:
            if out_pair is None:
                out_dists, out_indices = slot.default_outputs[slot.default_flip]
            else:
                out_dists, out_indices = out_pair
            bindings = {
                "query": query,
                "database": database,
                "query_sq": slot.query_sq if slot.internal_query_norm else query_sq,
                "database_sq": slot.database_sq if slot.internal_database_norm else database_sq,
                "out_dists": out_dists,
                "out_indices": out_indices,
            }
            try:
                # Bind first (tensor-map refresh + pointer overwrite are host
                # work), then enqueue the norm support launches, then submit
                # the plan's kernels. Binding after the norms would turn a
                # fresh-pointer tensor-map re-encode into a GPU inter-kernel
                # gap between the norm kernel and the route's first kernel.
                # A captured signature does every bind host-side, then one
                # graph replay covers the whole norm + route kernel chain.
                if slot.graph is not None:
                    slot.plan.bind_hot(bindings)
                    for norm_input_key, norm_launch in slot.norm_launches:
                        norm_launch.bind_hot(bindings[norm_input_key])
                    slot.graph.submit_hot(timeout_ms=effective_timeout_ms)
                else:
                    slot.plan.bind_hot(bindings)
                    for norm_input_key, norm_launch in slot.norm_launches:
                        norm_launch.launch_hot(bindings[norm_input_key])
                    slot.plan.submit_hot(timeout_ms=effective_timeout_ms)
                # The pool slot is consumed only by a successful submission:
                # a call that fails after selecting its default pair must not
                # burn the slot, or the next two successful calls would both
                # return the pair the last successful call already handed its
                # caller.
                if out_pair is None:
                    slot.default_flip ^= 1
            finally:
                # Slot-owned scratch and pooled default outputs are allocated
                # on the plan's stream and only released through clear();
                # caller-provided tensors may live on another stream, so
                # record those for allocator safety even on partial
                # submission.
                _record_stream_tensors(
                    (query, database, query_sq, database_sq)
                    + (() if out_pair is None else out_pair),
                    resolved_stream,
                )
        finally:
            slot.lock.release()
        result = (out_dists, out_indices)
        if not return_info:
            return result
        route = slot.route
        info = {
            "semantic_entrypoint": SEMANTIC_ENTRYPOINT,
            "selected_route": route.route_id,
            "launch_entrypoint": route.launch_entrypoint,
            "exact_launch_plan": route.exact_contract,
            "shape_label": getattr(route, "shape_label", None),
            "prepared_launch_count": int(slot.plan.launch_count),
            "runtime_launch_count": int(slot.plan.launch_count) + len(slot.norm_launches),
            "norm_launch_count": len(slot.norm_launches),
            "norm_compute_fields": list(slot.norm_compute_fields),
            "norm_mode": (
                "internal_fused_row_norm:" + ",".join(slot.norm_compute_fields)
                if slot.norm_compute_fields
                else "route_elided_internal_norms"
                if slot.internal_query_norm or slot.internal_database_norm
                else "explicit_precomputed"
            ),
            "hot_launch_path": "cuda_graph" if slot.graph is not None else "prepared_launches",
            "graph_kernel_count": None if slot.graph is None else int(slot.graph.launch_count),
            "graph_capture_error": slot.graph_capture_error,
            "arch": self.arch,
            "device_index": self.device_index,
            "stream_handle": stream_handle,
            "runtime_cache_hit": cache_hit,
        }
        return result, info

    def _create_slot(
        self,
        key: tuple[Any, ...],
        *,
        query: Any,
        database: Any,
        k: int,
        build: bool,
        shape_label: str | None,
        dtype_name: str,
        shape: tuple[int, int, int, int],
        out_pair: tuple[Any, Any] | None,
        query_sq: Any,
        database_sq: Any,
        stream: Any,
    ) -> tuple[_KNNBuildRuntimeSlot, bool]:
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
                slot = self._cache.get(key)
                if slot is not None:
                    self._cache.move_to_end(key)
                    self._hits += 1
                    return slot, False
                pending = self._preparing.get(key)
                if pending is None:
                    if (
                        self.max_cached_shapes is not None
                        and len(self._cache) + len(self._preparing) >= self.max_cached_shapes
                    ):
                        raise RuntimeError(
                            "KNNBuildRuntime cache is full; call clear() only after in-flight work completes"
                        )
                    pending = _PendingPreparation()
                    self._preparing[key] = pending
                    break
            pending.event.wait()
            if pending.error is not None:
                raise RuntimeError("KNN-build slot preparation failed in another thread") from pending.error

        slot: _KNNBuildRuntimeSlot | None = None
        try:
            import torch

            from ._row_norm import prepare_row_squared_norm

            bsz, n_query, n_database, dim = shape
            expected = (bsz, n_query, k)
            internal_query_norm = query_sq is None
            internal_database_norm = database_sq is None
            with torch.cuda.device(self.device_index), torch.cuda.stream(stream):
                default_outputs = tuple(
                    (
                        torch.empty(expected, dtype=torch.float32, device=database.device),
                        torch.empty(expected, dtype=torch.int32, device=database.device),
                    )
                    for _pair in range(2)
                )
                if internal_query_norm:
                    slot_query_sq = torch.empty(
                        (bsz, n_query),
                        dtype=torch.float32,
                        device=query.device,
                    )
                else:
                    slot_query_sq = query_sq
                if build and internal_database_norm:
                    slot_database_sq = slot_query_sq
                elif internal_database_norm:
                    slot_database_sq = torch.empty(
                        (bsz, n_database),
                        dtype=torch.float32,
                        device=database.device,
                    )
                else:
                    slot_database_sq = database_sq
                capture_dists, capture_indices = out_pair if out_pair is not None else default_outputs[0]
                inputs = {
                    "label": shape_label,
                    "B": bsz,
                    "Q": n_query,
                    "M": n_database,
                    "D": dim,
                    "K": k,
                    "dtype": dtype_name,
                    "build": build,
                    "query": query,
                    "database": database,
                    "query_sq": slot_query_sq,
                    "database_sq": slot_database_sq,
                    "out_dists": capture_dists,
                    "out_indices": capture_indices,
                }
                with _PREPARE_LOCK:
                    route = _direct_plan_runtime.resolve_route(inputs)
                    plan = build_launch_plan(
                        inputs,
                        stream=stream,
                        arch=self.arch,
                        validate_result=_require_owned_outputs,
                        route=route,
                    )
                bound_keys = getattr(plan, "bound_input_keys", None)
                bound = None if bound_keys is None else set(bound_keys)
                # A frozen plan reports exactly which public inputs its
                # captured launches bind; routes that compute norms
                # in-kernel skip the support launches entirely. A per-call
                # plan re-executes the route's host program, so provide
                # every internal norm it could consume.
                compute_query_norm = internal_query_norm and (
                    bound is None or "query_sq" in bound or build and "database_sq" in bound
                )
                compute_database_norm = internal_database_norm and not build and (
                    bound is None or "database_sq" in bound
                )
                norm_launches: list[tuple[str, Any]] = []
                if compute_query_norm:
                    norm_launches.append(
                        (
                            "query",
                            prepare_row_squared_norm(query, slot_query_sq, arch=self.arch, stream=stream),
                        )
                    )
                if compute_database_norm:
                    norm_launches.append(
                        (
                            "database",
                            prepare_row_squared_norm(database, slot_database_sq, arch=self.arch, stream=stream),
                        )
                    )
                # Capture the signature's stable kernel chain (norms first,
                # then the frozen route launches — the exact hot submission
                # order) into one CUDA graph. Per-call routes and launch
                # modes without a validated capture path stay on the
                # prepared-launch hot path, with the reason recorded;
                # any other capture failure is an error, not a fallback.
                graph_plan = None
                graph_capture_error: str | None = None
                try:
                    graph_plan = build_graph_exec_plan(
                        plan,
                        support_launches=tuple(
                            norm_launch.launch_plan for _key, norm_launch in norm_launches
                        ),
                    )
                except GraphCaptureUnsupported as unsupported:
                    graph_capture_error = str(unsupported)
                # The slot must not pin the creating call's tensors: each
                # norm plan's caller-owned input releases to its slot-owned
                # output scratch; every hot call rewrites the input carrier
                # before it submits. ``build_launch_plan`` already released
                # the main plan's bound inputs the same way.
                norm_outputs = {"query": slot_query_sq, "database": slot_database_sq}
                for input_key, norm_launch in norm_launches:
                    norm_launch.release_bound_input(norm_outputs[input_key], stream=stream)
            norm_compute_fields = tuple(
                name
                for name, enabled in (
                    ("query_sq", compute_query_norm),
                    ("database_sq", compute_database_norm),
                )
                if enabled
            )
            slot = _KNNBuildRuntimeSlot(
                plan=plan,
                route=route,
                norm_launches=tuple(norm_launches),
                # Slot-owned norm buffers only: a caller-supplied norm tensor
                # is rebound per call (see the compute() bindings) and must
                # not be pinned for the slot's cache lifetime.
                query_sq=slot_query_sq if internal_query_norm else None,
                database_sq=slot_database_sq if internal_database_norm else None,
                internal_query_norm=internal_query_norm,
                internal_database_norm=internal_database_norm,
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
                    self._cache[key] = slot
                    self._misses = old_misses + 1
                    self._preparing.pop(key, None)
                    pending.event.set()
                    publication_committed = True
                except BaseException as error:
                    if self._cache.get(key) is slot:
                        self._cache.pop(key, None)
                    self._misses = old_misses
                    if self._preparing.get(key) is pending:
                        self._preparing.pop(key, None)
                    pending.error = error
                    pending.event.set()
                    raise
        except BaseException as error:
            if not publication_committed:
                with self._cache_lock:
                    if self._cache.get(key) is slot:
                        self._cache.pop(key, None)
                        self._misses -= 1
                    if self._preparing.get(key) is pending:
                        self._preparing.pop(key, None)
                    if pending.error is None:
                        pending.error = error
                    pending.event.set()
            slot.lock.release()
            raise
        return slot, True

    def cache_info(self) -> dict[str, int | None]:
        with self._cache_lock:
            return {
                "hits": self._hits,
                "misses": self._misses,
                "size": len(self._cache),
                "max_cached_shapes": self.max_cached_shapes,
            }

    def clear(self, *, synchronize: bool = True) -> None:
        """Drop cached plans after host calls finish.

        The default waits for device completion and destroys the driver
        graph handles eagerly. With ``synchronize=False``, every plan-held
        launch argument, slot-owned norm buffer, and pooled default output
        is tied to its plan's stream via ``record_stream`` before release,
        and each slot's graph handles go to the shared event-gated graveyard
        (an executing graph must never be destroyed underneath the device) —
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
                while self._active_calls:
                    self._lifecycle.wait()
                if synchronize:
                    with torch.cuda.device(self.device_index):
                        torch.cuda.synchronize()
                with self._cache_lock:
                    if synchronize:
                        for slot in self._cache.values():
                            if slot.graph is not None:
                                slot.graph.destroy()
                    else:
                        for slot in self._cache.values():
                            plan_stream = slot.plan.torch_stream
                            slot.plan.record_stream(plan_stream)
                            for _input_key, norm_launch in slot.norm_launches:
                                norm_launch.record_stream(plan_stream)
                            _record_stream_tensors(
                                (slot.query_sq, slot.database_sq)
                                + tuple(tensor for pair in slot.default_outputs for tensor in pair),
                                plan_stream,
                            )
                            if slot.graph is not None:
                                defer_graph_destroy(slot.graph)
                    self._cache.clear()
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
    device: Any = None,
    arch: str | None = None,
    timeout_ms: float | None = None,
    max_cached_shapes: int | None = None,
    compile: str = "lazy",
) -> KNNBuildRuntime:
    """Initialize one reusable KNN-build runtime without binding input tensors."""

    return KNNBuildRuntime(
        device=device,
        arch=arch,
        timeout_ms=timeout_ms,
        max_cached_shapes=max_cached_shapes,
        compile=compile,
    )


def prepare_knn_build(
    query: Any,
    database: Any,
    k: int,
    *,
    build: bool = False,
    shape_label: str | None = None,
    out: tuple[Any, Any] | None = None,
    arch: str | None = None,
    stream: Any = None,
    timeout_ms: float | None = None,
) -> PreparedKNNBuild:
    """Prepare norms, outputs, scratch, and a fully marshalled direct leaf."""

    import torch

    if not isinstance(query, torch.Tensor) or not query.is_cuda:
        raise TypeError("query must be a CUDA torch.Tensor")
    device_index = query.device.index
    if device_index is None:
        device_index = torch.cuda.current_device()
    with torch.cuda.device(device_index):
        resolved_stream = torch.cuda.current_stream(device_index) if stream is None else stream
        stream_device = getattr(resolved_stream, "device", None)
        stream_device_index = getattr(stream_device, "index", stream_device)
        if stream_device_index is not None and int(stream_device_index) != int(device_index):
            raise ValueError(
                f"KNN-build stream device {stream_device_index} does not match input device {device_index}"
            )
        with torch.cuda.stream(resolved_stream):
            inputs = _prepare_inputs(query, database, k, build=build, shape_label=shape_label, out=out)
            launch_plan = prepare_route(inputs, arch=arch, stream=resolved_stream)
    return PreparedKNNBuild(
        inputs=inputs,
        launch_plan=launch_plan,
        shape_label=launch_plan.shape_label,
        stream=resolved_stream,
        timeout_ms=timeout_ms,
    )


def knn_build_prepared(
    prepared: PreparedKNNBuild,
    *,
    arch: str | None = None,
    stream: Any = None,
    timeout_ms: float | None = None,
    return_info: bool = False,
):
    """Launch a fixed route without re-entering the parent dispatcher or importer."""
    if not isinstance(prepared, PreparedKNNBuild):
        raise TypeError("prepared must be returned by prepare_knn_build")
    plan = prepared.launch_plan
    if arch is not None and str(arch) != plan.arch:
        raise ValueError(f"prepared KNN-build route targets {plan.arch}, requested incompatible arch {arch}")
    plan.launch(
        prepared.inputs,
        stream=stream,
        timeout_ms=prepared.timeout_ms if timeout_ms is None else timeout_ms,
    )
    out = (prepared.inputs["out_dists"], prepared.inputs["out_indices"])
    if not return_info:
        return out
    info = {
        "semantic_entrypoint": SEMANTIC_ENTRYPOINT,
        "selected_route": prepared.selected_route,
        "launch_entrypoint": plan.launch_entrypoint,
        "exact_launch_plan": plan.exact_contract,
        "shape_label": prepared.shape_label,
        "prepared_launch_count": plan.launch_count,
        "arch": plan.arch,
        "device_index": plan.device_index,
        "stream_handle": plan.stream_handle,
    }
    return out, info


_DEFAULT_RUNTIME_LOCK = RLock()
_DEFAULT_RUNTIMES: dict[int, KNNBuildRuntime] = {}


def _default_runtime(device_index: int) -> KNNBuildRuntime:
    """Return the process-wide per-device runtime backing ``knn_build``."""

    runtime = _DEFAULT_RUNTIMES.get(device_index)
    if runtime is None:
        with _DEFAULT_RUNTIME_LOCK:
            runtime = _DEFAULT_RUNTIMES.get(device_index)
            if runtime is None:
                runtime = KNNBuildRuntime(device=device_index)
                _DEFAULT_RUNTIMES[device_index] = runtime
    return runtime


def knn_build(
    query: Any,
    database: Any,
    k: int,
    *,
    build: bool = False,
    shape_label: str | None = None,
    out: tuple[Any, Any] | None = None,
    arch: str | None = None,
    stream: Any = None,
    timeout_ms: float | None = None,
    return_info: bool = False,
):
    """Run one KNN build/search through the per-device signature plan cache.

    The first call for a signature runs the frozen guard cascade once to
    construct its launch plan; subsequent calls are pointer-overwrite
    launches on the plan's stream. A caller-provided ``out=`` pair is
    written through; default outputs come from the signature's plan-owned
    pool, so a default-output result must be consumed (or cloned) before
    two further default-output calls of the same signature overwrite its
    storage.
    """
    import torch

    if not isinstance(query, torch.Tensor) or not getattr(query, "is_cuda", False):
        raise TypeError("query must be a CUDA torch.Tensor")
    runtime = _default_runtime(_tensor_device_index(query))
    if arch is not None and str(arch) != runtime.arch:
        raise ValueError(
            f"KNN-build launch arch must match the active device: requested {arch}, detected {runtime.arch}"
        )
    return runtime.compute(
        query,
        database,
        k,
        build=build,
        shape_label=shape_label,
        out=out,
        stream=stream,
        timeout_ms=timeout_ms,
        return_info=return_info,
    )
