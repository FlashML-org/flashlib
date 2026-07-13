"""Optional Apache TVM FFI adapters for the exported kernel package."""

from __future__ import annotations

import contextlib
import importlib
import json
from pathlib import Path
from typing import Any

from .kernels import get_kernel


_PACKAGE = 'flashlib_cake_kmeans'
_KERNEL_ALIASES = ['dispatch_kernel_0000', 'dispatch_kernel_0001', 'dispatch_kernel_0002', 'dispatch_kernel_0003', 'dispatch_kernel_0004', 'dispatch_kernel_0005', 'dispatch_kernel_0006', 'dispatch_kernel_0007', 'dispatch_kernel_0008', 'dispatch_kernel_0009', 'dispatch_kernel_0010', 'dispatch_kernel_0011', 'dispatch_kernel_0012', 'dispatch_kernel_0013', 'dispatch_kernel_0014', 'dispatch_kernel_0015', 'dispatch_kernel_0016', 'dispatch_kernel_0017', 'dispatch_kernel_0018', 'dispatch_kernel_0019', 'dispatch_kernel_0020', 'dispatch_kernel_0021', 'dispatch_kernel_0022', 'dispatch_kernel_0023', 'dispatch_kernel_0024', 'dispatch_kernel_0025', 'dispatch_kernel_0026', 'dispatch_kernel_0027', 'dispatch_kernel_0028', 'dispatch_kernel_0029', 'dispatch_kernel_0030', 'dispatch_kernel_0031', 'dispatch_kernel_0032', 'dispatch_kernel_0033', 'dispatch_kernel_0034', 'dispatch_kernel_0035', 'dispatch_kernel_0036', 'dispatch_kernel_0037']
_REGISTERED: dict[str, tuple[str, ...]] = {}


class _RawCUDAStream:
    def __init__(self, handle: int):
        self.cuda_stream = int(handle)


def _manifest() -> dict[str, Any]:
    return json.loads(Path(__file__).with_name("manifest.json").read_text(encoding="utf-8"))


def _planned_public_exports() -> dict[str, str] | None:
    export_plan = _manifest().get("export_plan", {})
    if "tvm_ffi_exports" in export_plan:
        return dict(export_plan["tvm_ffi_exports"])
    if "package_exports" in export_plan:
        return dict(export_plan["package_exports"])
    return None


def _public_export_names() -> tuple[str, ...]:
    planned = _planned_public_exports()
    if planned is not None:
        return tuple(planned)
    package = importlib.import_module(__package__)
    excluded = {
        "KERNELS",
        "ExportedKernel",
        "get_kernel",
        "register_tvm_ffi",
        "tvm_ffi_function_names",
        *(_KERNEL_ALIASES),
        *(f"launch_{name}" for name in _KERNEL_ALIASES),
    }
    return tuple(
        name
        for name in getattr(package, "__all__", ())
        if name not in excluded and callable(getattr(package, name, None))
    )


def tvm_ffi_function_names(namespace: str | None = None) -> tuple[str, ...]:
    """Return the deterministic TVM FFI global names this package registers."""

    prefix = namespace or _PACKAGE
    public_names = [f"{prefix}.{name}" for name in _public_export_names()]
    kernel_names = [f"{prefix}.launch_{name}" for name in _KERNEL_ALIASES]
    return tuple(public_names + kernel_names)


def _tensor_stream(arg: Any, tvm_ffi: Any) -> tuple[int, int] | None:
    if not isinstance(arg, tvm_ffi.Tensor):
        return None
    device = arg.device
    device_type = getattr(device, "type", None) or str(device).split(":", 1)[0]
    if str(device_type) != "cuda":
        return None
    device_id = int(getattr(device, "index", 0))
    return device_id, int(tvm_ffi.get_raw_stream(device))


def _convert_arg(arg: Any, tvm_ffi: Any) -> Any:
    if not isinstance(arg, tvm_ffi.Tensor):
        return arg
    import torch

    return torch.from_dlpack(arg)


def _torch_stream_context(stream: tuple[int, int] | None):
    if stream is None:
        return contextlib.nullcontext()
    import torch

    device_id, handle = stream
    return torch.cuda.stream(torch.cuda.ExternalStream(handle, device=device_id))


def _semantic_target(public_name: str):
    planned = _planned_public_exports()
    target = None if planned is None else planned.get(public_name)
    if target is None:
        return getattr(importlib.import_module(__package__), public_name)
    module_name, separator, attr = target.partition(":")
    if not separator:
        raise ValueError(f"invalid package export target: {target!r}")
    module = importlib.import_module(module_name, package=__package__)
    return getattr(module, attr)


def _semantic_wrapper(public_name: str, tvm_ffi: Any):
    function = _semantic_target(public_name)

    def call(*args):
        stream = next((item for arg in args if (item := _tensor_stream(arg, tvm_ffi)) is not None), None)
        converted = tuple(_convert_arg(arg, tvm_ffi) for arg in args)
        with _torch_stream_context(stream):
            return function(*converted)

    return call


def _kernel_wrapper(alias: str, tvm_ffi: Any):
    kernel = get_kernel(alias)
    parameter_count = len(kernel.parameters)

    def call(*args):
        expected = parameter_count + 7
        if len(args) != expected:
            raise TypeError(
                f"{alias} TVM FFI launch expects {expected} positional arguments "
                f"({parameter_count} kernel + grid xyz + block xyz + shared_mem), got {len(args)}"
            )
        kernel_args = args[:parameter_count]
        config = tuple(int(value) for value in args[parameter_count:])
        grid = config[:3]
        block = config[3:6]
        shared_mem = config[6]
        stream = next(
            (item for arg in kernel_args if (item := _tensor_stream(arg, tvm_ffi)) is not None),
            None,
        )
        converted = tuple(_convert_arg(arg, tvm_ffi) for arg in kernel_args)
        return kernel.launch(
            *converted,
            grid=grid,
            block=block,
            shared_mem=shared_mem,
            stream=_RawCUDAStream(stream[1]) if stream is not None else None,
        )

    return call


def register_tvm_ffi(namespace: str | None = None, *, override: bool = False) -> tuple[str, ...]:
    """Register semantic and low-level launch functions in Apache TVM FFI."""

    try:
        import tvm_ffi
    except ImportError as exc:
        raise ImportError(
            'TVM FFI support requires `python -m pip install -e ".[tvm-ffi]"`'
        ) from exc

    prefix = namespace or _PACKAGE
    if prefix in _REGISTERED and not override:
        return _REGISTERED[prefix]

    registered: list[str] = []
    for public_name in _public_export_names():
        name = f"{prefix}.{public_name}"
        tvm_ffi.register_global_func(
            name, _semantic_wrapper(public_name, tvm_ffi), override=override
        )
        registered.append(name)
    for alias in _KERNEL_ALIASES:
        name = f"{prefix}.launch_{alias}"
        tvm_ffi.register_global_func(name, _kernel_wrapper(alias, tvm_ffi), override=override)
        registered.append(name)
    result = tuple(registered)
    _REGISTERED[prefix] = result
    return result

