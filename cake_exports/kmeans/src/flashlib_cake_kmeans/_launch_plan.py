from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from functools import lru_cache
from typing import Any

from ._dispatch import flash_kmeans_assign_dispatcher as _root
from ._dispatch_runtime import _import_dispatch_module, dispatch_launch_options

_WEAVE_PREFIX = 'loom.examples.weave.'
_ROOT_MODULE = 'flash_kmeans_assign_dispatcher'
_ROOT_CALLABLE = 'launch_for_eval'
_EXACT_LAUNCH_SPECS = {}


@dataclass(frozen=True)
class RouteDecision:
    """Resolved semantic route with a launcher that can be reused directly."""

    route_id: str
    launch_entrypoint: str
    launcher: Callable[[dict[str, Any]], Any] = field(repr=False, compare=False)
    exact_contract: bool = False

    def launch(
        self,
        inputs: dict[str, Any],
        *,
        stream: Any = None,
        timeout_ms: float | None = None,
    ) -> Any:
        with dispatch_launch_options(stream=stream, timeout_ms=timeout_ms):
            return self.launcher(inputs)


def _route_key(inputs: dict[str, Any]) -> tuple[int, int, int, int, int, str, bool, bool]:
    dtype = str(inputs.get("dtype", "bfloat16"))
    if dtype.startswith("torch."):
        dtype = dtype[6:]
    return (
        *(int(inputs[name]) for name in ("B", "Q", "M", "D", "K")),
        dtype,
        bool(inputs.get("self_search", False)),
        bool(inputs.get("force_fallback", False)),
    )


@lru_cache(maxsize=None)
def _load_launcher(module_name: str, callable_name: str) -> Callable[[dict[str, Any]], Any]:
    module = _import_dispatch_module(module_name)
    launcher = getattr(module, callable_name, None)
    if not callable(launcher):
        raise RuntimeError(f"resolved dispatcher launcher is not callable: {module_name}:{callable_name}")
    return launcher


@lru_cache(maxsize=None)
def _make_decision(
    route_id: str,
    module_name: str,
    callable_name: str,
    exact_contract: bool,
) -> RouteDecision:
    return RouteDecision(
        route_id=route_id,
        launch_entrypoint=f"{_WEAVE_PREFIX}{module_name}:{callable_name}",
        launcher=_load_launcher(module_name, callable_name),
        exact_contract=exact_contract,
    )


def _entrypoint_spec(entrypoint: object) -> tuple[str, str] | None:
    if not isinstance(entrypoint, str):
        return None
    module_name, separator, callable_name = entrypoint.partition(":")
    if not separator or not module_name.startswith(_WEAVE_PREFIX) or not callable_name.isidentifier():
        return None
    return module_name.removeprefix(_WEAVE_PREFIX), callable_name


def _generic_decision(inputs: dict[str, Any]) -> RouteDecision:
    info_fn = getattr(_root, "route_info", None)
    info = dict(info_fn(inputs)) if callable(info_fn) else {}
    route_id = info.get("selected_route", info.get("route"))
    if route_id is None:
        select_route = getattr(_root, "selected_route", None)
        route_id = select_route(inputs) if callable(select_route) else _ROOT_CALLABLE
    # ``resolved_launch_entrypoint`` is an explicit launch contract and may
    # bypass the root. ``selected_entrypoint`` is seed provenance: it can name
    # a narrower module whose own guards would silently miss for this
    # signature (the K11 prefix route reports its K64 seed there, and
    # launching the seed with K=11 inputs falls through to a slower exact
    # parent). Signatures without a launch contract go through the root
    # dispatcher, which reproduces the frozen guard cascade exactly.
    spec = _entrypoint_spec(info.get("resolved_launch_entrypoint"))
    if spec is None:
        spec = (_ROOT_MODULE, _ROOT_CALLABLE)
    return _make_decision(str(route_id), *spec, False)


def resolve_route(inputs: dict[str, Any]) -> RouteDecision:
    """Resolve once; exact exported shapes never re-enter the root dispatcher."""

    spec = _EXACT_LAUNCH_SPECS.get(_route_key(inputs))
    if spec is None:
        return _generic_decision(inputs)
    return _make_decision(*spec, True)


class LaunchPlan:
    """Per-signature resolved execution state for the exported hot path.

    Migration step 2 of ``PLAN_BASED_EXPORT_RUNTIME.md``: the guard cascade
    (``resolve_route`` plus one captured dispatcher traversal) runs exactly
    once, at construction. A hot call overwrites the recorded 8-byte pointer
    carriers in place, refreshes device tensor-map descriptors only when a
    bound pointer actually changed, and submits the already-marshalled
    launches on their construction-time stream — no re-marshalling, no
    per-launch stream query, no dispatcher re-entry.
    """

    __slots__ = (
        "route",
        "sequence",
        "torch_stream",
        "stream_handle",
        "_launches",
        "_pointer_writers",
        "_tma_bindings",
    )

    def __init__(self, route: RouteDecision, sequence: Any, *, torch_stream: Any, stream_handle: int):
        launches = tuple(sequence._launches)
        input_bindings = tuple(sequence._input_bindings)
        if input_bindings and len(input_bindings) != len(launches):
            raise RuntimeError("launch plan capture has corrupt input bindings")
        writers: dict[str, list[Any]] = {}
        for launch, bindings in zip(launches, input_bindings):
            carriers = launch._packed._prevent_gc
            for index, key in bindings:
                writers.setdefault(key, []).append(carriers[index])
        self.route = route
        self.sequence = sequence
        self.torch_stream = torch_stream
        self.stream_handle = int(stream_handle)
        self._launches = launches
        self._pointer_writers = tuple((key, tuple(items)) for key, items in writers.items())
        self._tma_bindings = tuple(sequence._tensor_map_bindings)

    @property
    def launch_count(self) -> int:
        return len(self._launches)

    @property
    def bound_input_keys(self) -> tuple[str, ...]:
        direct = {key for key, _carriers in self._pointer_writers}
        derived = {binding.input_key for binding in self._tma_bindings}
        return tuple(sorted(direct | derived))

    def bind_hot(self, bindings: dict[str, Any]) -> None:
        """Refresh tensor maps and overwrite bound pointer carriers in place.

        This is the host-side half of a hot call. Callers that enqueue their
        own support launches between the plan's pointer binding and its
        kernel submission (the KNN-build fused row norms) must call this
        BEFORE those launches: a fresh-pointer tensor-map re-encode costs
        host time, and paying it after any kernel is already enqueued turns
        that host time into a GPU inter-kernel gap.

        Tensor maps go through the per-pointer variant bank: a pointer the
        bank has seen keeps its device-resident descriptor, so re-activating
        it is a handful of carrier writes with no ``cuTensorMapEncodeTiled``,
        no pinned-staging refresh, and no H2D copy. The plan's signature slot
        is stream-keyed and alias-keyed by the caller, which is the safety
        contract ``rebind_stream_bound`` requires.
        """

        for binding in self._tma_bindings:
            binding.rebind_stream_bound(bindings[binding.input_key], stream=self.torch_stream)
        for key, carriers in self._pointer_writers:
            pointer = bindings[key].data_ptr()
            for carrier in carriers:
                carrier.value = pointer

    def submit_hot(self, *, timeout_ms: float | None = None) -> None:
        """Submit every prepared launch on the plan's construction stream."""

        launches = self._launches
        last_index = len(launches) - 1
        for index, launch in enumerate(launches):
            launch.launch(stream=None, timeout_ms=timeout_ms if index == last_index else None)

    def launch_hot(self, bindings: dict[str, Any], *, timeout_ms: float | None = None) -> None:
        """Patch bound pointer carriers from ``bindings`` and submit every launch."""

        self.bind_hot(bindings)
        self.submit_hot(timeout_ms=timeout_ms)

    def record_stream(self, stream: Any) -> None:
        """Tie every plan-held launch argument to ``stream`` before release."""

        self.sequence.record_stream(stream)


class PerCallRoutePlan:
    """Per-signature plan for routes whose host logic reads device results.

    Capture observed the route reading GPU memory (for example an
    ``overflow_flag.item()`` certification) while its kernels were only being
    recorded, so a frozen launch list cannot reproduce the route's per-call
    branch decisions — freezing would bake whichever branch the capture-time
    garbage selected. These signatures keep the generic per-call launcher:
    route resolution stays cached and the guard cascade still ran exactly
    once, but every hot call re-executes the resolved route's host program.
    Device-side repair (Migration step 3) makes such routes replayable.
    """

    __slots__ = (
        "route",
        "torch_stream",
        "stream_handle",
        "launch_count",
        "host_data_reads",
        "_static_inputs",
        "_pending_bindings",
    )

    def __init__(
        self,
        route: RouteDecision,
        *,
        torch_stream: Any,
        stream_handle: int,
        static_inputs: dict[str, Any],
        launch_count: int,
        host_data_reads: int,
    ):
        self.route = route
        self.torch_stream = torch_stream
        self.stream_handle = int(stream_handle)
        self.launch_count = int(launch_count)
        self.host_data_reads = int(host_data_reads)
        self._static_inputs = dict(static_inputs)
        self._pending_bindings = None

    def bind_hot(self, bindings: dict[str, Any]) -> None:
        """Stage this call's tensor bindings for ``submit_hot``.

        Mirrors ``LaunchPlan``'s two-phase hot call so callers with support
        launches use one code path. The caller's per-signature slot lock
        serializes bind/submit pairs on a plan instance.
        """

        self._pending_bindings = dict(bindings)

    def submit_hot(self, *, timeout_ms: float | None = None) -> None:
        """Re-execute the resolved route with the staged tensor bindings."""

        bindings = self._pending_bindings
        if bindings is None:
            raise RuntimeError("PerCallRoutePlan.submit_hot requires a preceding bind_hot")
        self._pending_bindings = None
        self.launch_hot(bindings, timeout_ms=timeout_ms)

    def launch_hot(self, bindings: dict[str, Any], *, timeout_ms: float | None = None) -> None:
        """Re-execute the resolved route with this call's tensor bindings.

        The torch stream context keeps the route's tensor operations (scratch
        fills, the certification read-back) ordered with its kernel launches
        on the plan's stream, exactly as the live evaluation path runs it.
        """

        import torch

        inputs = dict(self._static_inputs)
        inputs.update(bindings)
        with torch.cuda.stream(self.torch_stream):
            self.route.launch(inputs, stream=self.torch_stream, timeout_ms=timeout_ms)

    def record_stream(self, stream: Any) -> None:
        """Per-call plans hold no launch arguments; nothing to record."""


def build_launch_plan(
    inputs: dict[str, Any],
    *,
    stream: Any,
    arch: str,
    validate_result: Callable[[Any, dict[str, Any]], None] | None = None,
    route: Any = None,
) -> LaunchPlan | PerCallRoutePlan:
    """Run the guard cascade once and freeze its launches into a LaunchPlan.

    This is the per-signature slow path; ``resolve_route`` remains the single
    source of routing truth. ``validate_result(result, inputs)`` must raise
    when the resolved route's outputs cannot be retargeted by pointer
    rebinding (for example outputs that are not caller-owned tensors).
    Routes whose host logic read device memory during capture resolve to a
    ``PerCallRoutePlan`` instead of a frozen launch list.

    ``route`` accepts an already-resolved decision from a sibling routing
    layer with the same contract (``route_id``/``launch_entrypoint``/
    ``exact_contract`` plus ``launch(inputs, stream=..., timeout_ms=...)``),
    for workloads whose exact-contract table lives outside this module (the
    KNN-build direct-manifest resolver). It must come from that workload's
    frozen routing surface, never from re-guessing the cascade.
    """

    from ._dispatch_runtime import capture_kernel_launches
    from ._runtime import launch_context

    if stream is None:
        raise ValueError("build_launch_plan requires a resolved torch CUDA stream, not None")
    if route is None:
        route = resolve_route(inputs)
    with capture_kernel_launches(stream=stream, arch=arch, inputs=inputs) as captured:
        with launch_context(arch=arch, stream=stream, timeout_ms=None):
            result = route.launch(inputs, stream=stream, timeout_ms=None)
    if validate_result is not None:
        validate_result(result, inputs)
    if captured.host_data_dependent:
        static_inputs = {
            key: value for key, value in inputs.items() if not callable(getattr(value, "data_ptr", None))
        }
        return PerCallRoutePlan(
            route,
            torch_stream=stream,
            stream_handle=int(stream.cuda_stream),
            static_inputs=static_inputs,
            launch_count=len(captured._launches),
            host_data_reads=captured.host_data_reads,
        )
    sequence = captured.bind(result)
    return LaunchPlan(
        route,
        sequence,
        torch_stream=stream,
        stream_handle=int(stream.cuda_stream),
    )


class GraphCaptureUnsupported(RuntimeError):
    """A plan's launches have no validated CUDA-graph capture path."""


def _check_cu(err: Any, message: str) -> None:
    code = err[0] if isinstance(err, tuple) else err
    if int(code) != 0:
        raise RuntimeError(f"{message}: CUresult={int(code)}")


class GraphExecPlan:
    """One per-signature CUDA graph over a LaunchPlan plus support launches.

    Migration step 3 of ``PLAN_BASED_EXPORT_RUNTIME.md``: the signature's
    stable kernel chain (support launches such as fused row norms, then the
    frozen route launches) is stream-captured once at plan construction. A
    hot call is host-only binding (the caller's ``plan.bind_hot`` plus
    support pointer writes into the same persistent packed argument buffers),
    then ``submit_hot``: every kernel node's packed buffer is pushed through
    ``cuGraphExecKernelNodeSetParams`` and the chain replays with one
    ``cuGraphLaunch`` on the plan's construction-time stream. Kernel-node
    launch attributes recorded at capture (cluster dimensions, scheduling
    preference) persist across exec-node parameter updates.
    """

    __slots__ = (
        "plan",
        "_launches",
        "_graph",
        "_graph_exec",
        "_node_params",
        "_cu_stream",
        "_set_params",
        "_graph_launch",
        "_cu_success",
        "_destroyed",
    )

    def __init__(self, plan: LaunchPlan, launches, graph, graph_exec, node_params, cu_stream):
        from cuda.bindings import driver

        self.plan = plan
        self._launches = tuple(launches)
        self._graph = graph
        self._graph_exec = graph_exec
        self._node_params = tuple(node_params)
        self._cu_stream = cu_stream
        self._set_params = driver.cuGraphExecKernelNodeSetParams
        self._graph_launch = driver.cuGraphLaunch
        self._cu_success = driver.CUresult.CUDA_SUCCESS
        self._destroyed = False

    @property
    def launch_count(self) -> int:
        return len(self._launches)

    def submit_hot(self, *, timeout_ms: float | None = None) -> None:
        """Push the persistent packed argument buffers and replay the graph.

        The caller must have completed every pointer/tensor-map bind for this
        call (``plan.bind_hot`` plus support-launch binds) first; parameter
        values are copied out of the packed buffers here.
        """

        if self._destroyed:
            raise RuntimeError("graph plan was destroyed by a runtime clear()")
        set_params = self._set_params
        graph_exec = self._graph_exec
        success = self._cu_success
        for node, params in self._node_params:
            (err,) = set_params(graph_exec, node, params)
            if err != success:
                _check_cu(err, "cuGraphExecKernelNodeSetParams failed")
        (err,) = self._graph_launch(graph_exec, self._cu_stream)
        if err != success:
            _check_cu(err, "cuGraphLaunch failed")
        if timeout_ms is not None:
            self._launches[-1]._kernel._wait_with_timeout(self._cu_stream, timeout_ms)

    def destroy(self) -> None:
        """Release the driver graph handles. Device work must be complete."""

        if self._destroyed:
            return
        from cuda.bindings import driver

        self._destroyed = True
        driver.cuGraphExecDestroy(self._graph_exec)
        driver.cuGraphDestroy(self._graph)


def build_graph_exec_plan(plan: Any, *, support_launches: tuple = ()) -> GraphExecPlan:
    """Capture ``support_launches`` then ``plan``'s launches into one graph.

    The per-signature slow path, run once at plan construction. Launches are
    replayed onto a dedicated capture stream (graph construction only — no
    kernel executes), each launch is mapped to its kernel node through the
    capture stream's leaf-dependency query, and the captured topology is
    hard-checked to contain exactly the expected kernel nodes so foreign
    work (for example watchdog event records) can never silently ride along.

    Raises ``GraphCaptureUnsupported`` for plans that cannot replay from a
    frozen kernel chain (per-call routes) and for launch modes without a
    validated capture path (cooperative). Any other failure propagates —
    a capture that should work but does not is an error, not a fallback.
    """

    import ctypes
    import sys
    from contextlib import nullcontext

    import torch
    from cuda.bindings import driver

    if not isinstance(plan, LaunchPlan):
        raise GraphCaptureUnsupported(
            "only frozen LaunchPlans are graph-capturable; per-call routes re-execute host logic"
        )
    launches = tuple(support_launches) + tuple(plan._launches)
    for launch in launches:
        if launch._mode not in ("regular", "cluster"):
            raise GraphCaptureUnsupported(
                f"launch mode {launch._mode!r} has no validated graph-capture path"
            )

    # Captured launches build graph nodes and do not execute, so loom's CUDA
    # watchdog (present only when the in-repo runtime shares this process)
    # must not record completion events for them: the event record would be
    # captured as a foreign node and the poller would query a captured event.
    watchdog = sys.modules.get("loom.runtime.cuda_watchdog")
    suspend = getattr(watchdog, "suspend_tracking", None)
    suspension = suspend() if callable(suspend) else nullcontext()

    capture_stream = torch.cuda.Stream(device=plan.torch_stream.device)
    cu_capture = driver.CUstream(capture_stream.cuda_stream)
    nodes = []
    with suspension:
        (err,) = driver.cuStreamBeginCapture(
            cu_capture, driver.CUstreamCaptureMode.CU_STREAM_CAPTURE_MODE_THREAD_LOCAL
        )
        _check_cu(err, "cuStreamBeginCapture failed")
        try:
            for launch in launches:
                launch.launch(stream=capture_stream, timeout_ms=None)
                info = driver.cuStreamGetCaptureInfo(cu_capture)
                _check_cu(info[0], "cuStreamGetCaptureInfo failed")
                status, leaves = info[1], info[4]
                if (
                    status != driver.CUstreamCaptureStatus.CU_STREAM_CAPTURE_STATUS_ACTIVE
                    or len(leaves) != 1
                ):
                    raise RuntimeError(
                        "graph capture did not add exactly one leaf node for a prepared launch"
                    )
                nodes.append(leaves[0])
        except BaseException:
            driver.cuStreamEndCapture(cu_capture)  # abandon the partial capture
            raise
        err, graph = driver.cuStreamEndCapture(cu_capture)
        _check_cu(err, "cuStreamEndCapture failed")

    try:
        err, _probe, total_nodes = driver.cuGraphGetNodes(graph, 0)
        _check_cu(err, "cuGraphGetNodes failed")
        if int(total_nodes) != len(launches):
            raise RuntimeError(
                f"captured graph has {int(total_nodes)} nodes, expected {len(launches)}; "
                "foreign work was injected into the capture"
            )
        for node in nodes:
            err, node_type = driver.cuGraphNodeGetType(node)
            _check_cu(err, "cuGraphNodeGetType failed")
            if node_type != driver.CUgraphNodeType.CU_GRAPH_NODE_TYPE_KERNEL:
                raise RuntimeError("captured graph node is not a kernel node")
        err, graph_exec = driver.cuGraphInstantiate(graph, 0)
        _check_cu(err, "cuGraphInstantiate failed")
    except BaseException:
        driver.cuGraphDestroy(graph)
        raise

    node_params = []
    for launch, node in zip(launches, nodes):
        params = driver.CUDA_KERNEL_NODE_PARAMS()
        params.func = launch._kernel._func
        params.gridDimX, params.gridDimY, params.gridDimZ = launch._grid
        params.blockDimX, params.blockDimY, params.blockDimZ = launch._block
        params.sharedMemBytes = launch._shared_mem
        params.kernelParams = ctypes.addressof(launch._packed)
        params.extra = 0
        node_params.append((node, params))
    return GraphExecPlan(
        plan,
        launches,
        graph,
        graph_exec,
        node_params,
        driver.CUstream(plan.stream_handle),
    )
