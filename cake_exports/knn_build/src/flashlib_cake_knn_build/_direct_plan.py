from __future__ import annotations

import json
from collections.abc import Callable
from dataclasses import dataclass, field
from functools import cache
from importlib import resources
from threading import RLock
from typing import Any

from ._dispatch import knn_build_dispatch_q1m524_v10_d320recurrence_consumption_v1 as _root
from ._dispatch_runtime import (
    _import_dispatch_module,
    capture_kernel_launches,
    detect_gpu_arch,
    dispatch_launch_options,
)

_WEAVE_PREFIX = "loom.examples.weave."
_ROOT_MODULE = "knn_build_dispatch_q1m524_v10_d320recurrence_consumption_v1"
_ROOT_CALLABLE = "launch_from_contract_inputs"
_CONTRACT_FIELDS = ("B", "Q", "M", "D", "K")
_PREPARE_LOCK = RLock()


def _load_json(name: str) -> Any:
    package = __package__ or __name__.rpartition(".")[0]
    return json.loads(resources.files(package).joinpath(name).read_text(encoding="utf-8"))


def _normalize_dtype(value: Any) -> str:
    dtype = str(value)
    return dtype[6:] if dtype.startswith("torch.") else dtype


def _contract_key(inputs: dict[str, Any]) -> tuple[int, int, int, int, int, str, bool]:
    return (
        *(int(inputs[name]) for name in _CONTRACT_FIELDS),
        _normalize_dtype(inputs.get("dtype", "bfloat16")),
        bool(inputs.get("build", False)),
    )


def _load_exact_specs() -> dict[tuple[int, int, int, int, int, str, bool], tuple[str, str, str]]:
    shapes = _load_json("_shape_records.json")
    routes = {row["shape"]: row["selected_route"] for row in _load_json("_routes.json")}
    entrypoints = _load_json("_entrypoints.json")
    specs: dict[tuple[int, int, int, int, int, str, bool], tuple[str, str, str]] = {}
    for row in shapes:
        label = str(row["label"])
        params = dict(row["params"])
        if label not in routes or label not in entrypoints:
            raise RuntimeError(f"missing direct KNN-build route metadata for {label!r}")
        key = (
            *(int(params[name]) for name in _CONTRACT_FIELDS),
            _normalize_dtype(params.get("dtype", "bfloat16")),
            bool(params.get("build", False)),
        )
        spec = (label, str(routes[label]), str(entrypoints[label]))
        prior = specs.setdefault(key, spec)
        if prior != spec:
            raise RuntimeError(f"ambiguous direct KNN-build contract {key!r}: {prior!r} versus {spec!r}")
    return specs


_EXACT_SPECS = _load_exact_specs()


@dataclass(frozen=True)
class DirectRouteDecision:
    """One resolved KNN-build route and its imported direct callable."""

    route_id: str
    launch_entrypoint: str
    launcher: Callable[[dict[str, Any]], Any] = field(repr=False, compare=False)
    shape_label: str | None = None
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


@dataclass(frozen=True)
class PreparedDirectRoute:
    """One exact leaf frozen into a device/stream-bound CUDA launch sequence."""

    decision: DirectRouteDecision
    direct_launcher: Callable[..., Any] = field(repr=False, compare=False)
    inputs: dict[str, Any] = field(repr=False, compare=False)
    arch: str
    device_index: int
    stream: Any = field(repr=False, compare=False)
    stream_handle: int
    launch_count: int

    @property
    def route_id(self) -> str:
        return self.decision.route_id

    @property
    def launch_entrypoint(self) -> str:
        return self.decision.launch_entrypoint

    @property
    def shape_label(self) -> str | None:
        return self.decision.shape_label

    @property
    def exact_contract(self) -> bool:
        return self.decision.exact_contract

    def launch(
        self,
        inputs: dict[str, Any],
        *,
        stream: Any = None,
        timeout_ms: float | None = None,
    ) -> Any:
        """Submit the frozen sequence; another stream requires another plan."""

        import torch

        if inputs is not self.inputs:
            raise ValueError("prepared KNN-build route is bound to its original input/output tensors")
        with torch.cuda.device(self.device_index):
            requested_stream = self.stream if stream is None else stream
            requested_handle = int(requested_stream.cuda_stream)
            if requested_handle != self.stream_handle:
                raise RuntimeError(
                    "prepared KNN-build route is stream-bound: "
                    f"prepared on stream 0x{self.stream_handle:x}, requested 0x{requested_handle:x}; "
                    "prepare a separate plan inside the target torch.cuda.stream(...) context"
                )
            try:
                result = self.direct_launcher(inputs, stream=None, timeout_ms=timeout_ms)
            finally:
                # A later launch can fail after an earlier launch was already
                # enqueued. Keep every captured argument allocator-safe even
                # on that partial-submission path.
                self.direct_launcher.record_stream(requested_stream)
                _record_input_streams(inputs, requested_stream)
            return result

    def rebind_inputs(
        self,
        inputs: dict[str, Any],
        *,
        stream: Any = None,
    ) -> None:
        """Rebind public tensor arguments without resolving the route again."""

        import torch

        if inputs is not self.inputs:
            raise ValueError("prepared KNN-build route is bound to its original input dictionary")
        with torch.cuda.device(self.device_index):
            requested_stream = self.stream if stream is None else stream
            requested_handle = int(requested_stream.cuda_stream)
            if requested_handle != self.stream_handle:
                raise RuntimeError(
                    "prepared KNN-build route is stream-bound: "
                    f"prepared on stream 0x{self.stream_handle:x}, requested 0x{requested_handle:x}; "
                    "prepare a separate plan inside the target torch.cuda.stream(...) context"
                )
            self.direct_launcher.rebind_inputs(inputs, stream=requested_stream)


@cache
def _load_launcher(entrypoint: str) -> Callable[[dict[str, Any]], Any]:
    module_name, separator, callable_name = entrypoint.partition(":")
    if not separator or not module_name.startswith(_WEAVE_PREFIX) or not callable_name.isidentifier():
        raise RuntimeError(f"invalid direct KNN-build entrypoint: {entrypoint!r}")
    module = _import_dispatch_module(module_name.removeprefix(_WEAVE_PREFIX))
    launcher = getattr(module, callable_name, None)
    if not callable(launcher):
        raise RuntimeError(f"direct KNN-build entrypoint is not callable: {entrypoint!r}")
    return launcher


def _record_input_streams(inputs: dict[str, Any], stream: Any) -> None:
    seen: set[int] = set()
    for value in inputs.values():
        identity = id(value)
        record_stream = getattr(value, "record_stream", None)
        if identity not in seen and callable(record_stream):
            seen.add(identity)
            record_stream(stream)


@cache
def _make_decision(
    shape_label: str | None,
    route_id: str,
    launch_entrypoint: str,
    exact_contract: bool,
) -> DirectRouteDecision:
    return DirectRouteDecision(
        route_id=route_id,
        launch_entrypoint=launch_entrypoint,
        launcher=_load_launcher(launch_entrypoint),
        shape_label=shape_label,
        exact_contract=exact_contract,
    )


def resolve_route(inputs: dict[str, Any]) -> DirectRouteDecision:
    """Resolve and import a direct launcher once for one fixed input contract."""

    spec = _EXACT_SPECS.get(_contract_key(inputs))
    if spec is not None:
        shape_label, route_id, launch_entrypoint = spec
        # Several historical leaf guards use the canonical label as a second
        # exact-contract check.  This dictionary is private to the public API,
        # so canonicalizing it cannot mutate caller state.
        inputs["label"] = shape_label
        return _make_decision(shape_label, route_id, launch_entrypoint, True)
    route_id = str(_root.route_for_contract_inputs(inputs))
    launch_entrypoint = f"{_WEAVE_PREFIX}{_ROOT_MODULE}:{_ROOT_CALLABLE}"
    return _make_decision(None, route_id, launch_entrypoint, False)


def prepare_route(
    inputs: dict[str, Any],
    *,
    arch: str | None = None,
    stream: Any = None,
) -> PreparedDirectRoute:
    """Resolve one leaf and capture all host setup and launches exactly once."""

    import torch

    input_device = inputs["database"].device
    device_index = input_device.index
    if device_index is None:
        device_index = torch.cuda.current_device()
    device_index = int(device_index)
    with torch.cuda.device(device_index):
        resolved_stream = torch.cuda.current_stream(device_index) if stream is None else stream
        stream_device = getattr(resolved_stream, "device", None)
        stream_device_index = getattr(stream_device, "index", stream_device)
        if stream_device_index is not None and int(stream_device_index) != device_index:
            raise ValueError(
                f"KNN-build stream device {stream_device_index} does not match input device {device_index}"
            )
        stream_handle = int(resolved_stream.cuda_stream)
        with torch.cuda.stream(resolved_stream), _PREPARE_LOCK:
            detected_arch = detect_gpu_arch()
            resolved_arch = detected_arch if arch is None else str(arch)
            if resolved_arch != detected_arch:
                raise ValueError(
                    f"KNN-build launch arch must match the active device: "
                    f"requested {resolved_arch}, detected {detected_arch}"
                )
            decision = resolve_route(inputs)
            inputs["_knn_build_prepared_stream_key"] = (device_index, stream_handle)
            with capture_kernel_launches(
                stream=resolved_stream,
                arch=resolved_arch,
                inputs=inputs,
            ) as captured:
                with dispatch_launch_options(stream=resolved_stream, timeout_ms=None):
                    prepared_result = decision.launcher(inputs)
            direct_launcher = captured.bind(prepared_result)
    return PreparedDirectRoute(
        decision=decision,
        direct_launcher=direct_launcher,
        inputs=inputs,
        arch=resolved_arch,
        device_index=device_index,
        stream=resolved_stream,
        stream_handle=stream_handle,
        launch_count=direct_launcher.launch_count,
    )
