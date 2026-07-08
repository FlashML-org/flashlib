"""Optional Apache TVM FFI adapters for the exported kernel package."""

from __future__ import annotations

import contextlib
import importlib
import json
from pathlib import Path
from typing import Any

from .kernels import get_kernel


_PACKAGE = 'flashlib_cake_knn_build'
_KERNEL_ALIASES = ['dispatch_kernel_0000', 'dispatch_kernel_0001', 'dispatch_kernel_0002', 'dispatch_kernel_0003', 'dispatch_kernel_0004', 'dispatch_kernel_0005', 'dispatch_kernel_0006', 'dispatch_kernel_0007', 'dispatch_kernel_0008', 'dispatch_kernel_0009', 'dispatch_kernel_0010', 'dispatch_kernel_0011', 'dispatch_kernel_0012', 'dispatch_kernel_0013', 'dispatch_kernel_0014', 'dispatch_kernel_0015', 'dispatch_kernel_0016', 'dispatch_kernel_0017', 'dispatch_kernel_0018', 'dispatch_kernel_0019', 'dispatch_kernel_0020', 'dispatch_kernel_0021', 'dispatch_kernel_0022', 'dispatch_kernel_0023', 'dispatch_kernel_0024', 'dispatch_kernel_0025', 'dispatch_kernel_0026', 'dispatch_kernel_0027', 'dispatch_kernel_0028', 'dispatch_kernel_0029', 'dispatch_kernel_0030', 'dispatch_kernel_0031', 'dispatch_kernel_0032', 'dispatch_kernel_0033', 'dispatch_kernel_0034', 'dispatch_kernel_0035', 'dispatch_kernel_0036', 'dispatch_kernel_0037', 'dispatch_kernel_0038', 'dispatch_kernel_0039', 'dispatch_kernel_0040', 'dispatch_kernel_0041', 'dispatch_kernel_0042', 'dispatch_kernel_0043', 'dispatch_kernel_0044', 'dispatch_kernel_0045', 'dispatch_kernel_0046', 'dispatch_kernel_0047', 'dispatch_kernel_0048', 'dispatch_kernel_0049', 'dispatch_kernel_0050', 'dispatch_kernel_0051', 'dispatch_kernel_0052', 'dispatch_kernel_0053', 'dispatch_kernel_0054', 'dispatch_kernel_0055', 'dispatch_kernel_0056', 'dispatch_kernel_0057', 'dispatch_kernel_0058', 'dispatch_kernel_0059', 'dispatch_kernel_0060', 'dispatch_kernel_0061', 'dispatch_kernel_0062', 'dispatch_kernel_0063', 'dispatch_kernel_0064', 'dispatch_kernel_0065', 'dispatch_kernel_0066', 'dispatch_kernel_0067', 'dispatch_kernel_0068', 'dispatch_kernel_0069', 'dispatch_kernel_0070', 'dispatch_kernel_0071', 'dispatch_kernel_0072', 'dispatch_kernel_0073', 'dispatch_kernel_0074', 'dispatch_kernel_0075', 'dispatch_kernel_0076', 'dispatch_kernel_0077', 'dispatch_kernel_0078', 'dispatch_kernel_0079', 'dispatch_kernel_0080', 'dispatch_kernel_0081', 'dispatch_kernel_0082', 'dispatch_kernel_0083', 'dispatch_kernel_0084', 'dispatch_kernel_0085', 'dispatch_kernel_0086', 'dispatch_kernel_0087', 'dispatch_kernel_0088', 'dispatch_kernel_0089', 'dispatch_kernel_0090', 'dispatch_kernel_0091', 'dispatch_kernel_0092', 'dispatch_kernel_0093', 'dispatch_kernel_0094', 'dispatch_kernel_0095', 'dispatch_kernel_0096', 'dispatch_kernel_0097', 'dispatch_kernel_0098', 'dispatch_kernel_0099', 'dispatch_kernel_0100', 'dispatch_kernel_0101', 'dispatch_kernel_0102', 'dispatch_kernel_0103', 'dispatch_kernel_0104', 'dispatch_kernel_0105', 'dispatch_kernel_0106', 'dispatch_kernel_0107', 'dispatch_kernel_0108', 'dispatch_kernel_0109', 'dispatch_kernel_0110', 'dispatch_kernel_0111', 'dispatch_kernel_0112', 'dispatch_kernel_0113', 'dispatch_kernel_0114', 'dispatch_kernel_0115', 'dispatch_kernel_0116', 'dispatch_kernel_0117', 'dispatch_kernel_0118', 'dispatch_kernel_0119', 'dispatch_kernel_0120', 'dispatch_kernel_0121', 'dispatch_kernel_0122', 'dispatch_kernel_0123', 'dispatch_kernel_0124', 'dispatch_kernel_0125', 'dispatch_kernel_0126', 'dispatch_kernel_0127', 'dispatch_kernel_0128', 'dispatch_kernel_0129', 'dispatch_kernel_0130', 'dispatch_kernel_0131', 'dispatch_kernel_0132', 'dispatch_kernel_0133', 'dispatch_kernel_0134', 'dispatch_kernel_0135', 'dispatch_kernel_0136', 'dispatch_kernel_0137', 'dispatch_kernel_0138', 'dispatch_kernel_0139', 'dispatch_kernel_0140', 'dispatch_kernel_0141', 'dispatch_kernel_0142', 'dispatch_kernel_0143', 'dispatch_kernel_0144', 'dispatch_kernel_0145', 'dispatch_kernel_0146', 'dispatch_kernel_0147', 'dispatch_kernel_0148', 'dispatch_kernel_0149', 'dispatch_kernel_0150', 'dispatch_kernel_0151', 'dispatch_kernel_0152', 'dispatch_kernel_0153', 'dispatch_kernel_0154', 'dispatch_kernel_0155', 'dispatch_kernel_0156', 'dispatch_kernel_0157', 'dispatch_kernel_0158', 'dispatch_kernel_0159', 'dispatch_kernel_0160', 'dispatch_kernel_0161', 'dispatch_kernel_0162', 'dispatch_kernel_0163', 'dispatch_kernel_0164', 'dispatch_kernel_0165', 'dispatch_kernel_0166', 'dispatch_kernel_0167', 'dispatch_kernel_0168', 'dispatch_kernel_0169', 'dispatch_kernel_0170', 'dispatch_kernel_0171', 'dispatch_kernel_0172', 'dispatch_kernel_0173', 'dispatch_kernel_0174', 'dispatch_kernel_0175', 'dispatch_kernel_0176', 'dispatch_kernel_0177', 'dispatch_kernel_0178', 'dispatch_kernel_0179', 'dispatch_kernel_0180', 'dispatch_kernel_0181', 'dispatch_kernel_0182', 'dispatch_kernel_0183', 'dispatch_kernel_0184', 'dispatch_kernel_0185', 'dispatch_kernel_0186', 'dispatch_kernel_0187', 'dispatch_kernel_0188', 'dispatch_kernel_0189', 'dispatch_kernel_0190', 'dispatch_kernel_0191', 'dispatch_kernel_0192', 'dispatch_kernel_0193', 'dispatch_kernel_0194', 'dispatch_kernel_0195', 'dispatch_kernel_0196', 'dispatch_kernel_0197', 'dispatch_kernel_0198', 'dispatch_kernel_0199', 'dispatch_kernel_0200', 'dispatch_kernel_0201', 'dispatch_kernel_0202', 'dispatch_kernel_0203', 'dispatch_kernel_0204', 'dispatch_kernel_0205', 'dispatch_kernel_0206', 'dispatch_kernel_0207', 'dispatch_kernel_0208', 'dispatch_kernel_0209', 'dispatch_kernel_0210', 'dispatch_kernel_0211', 'dispatch_kernel_0212', 'dispatch_kernel_0213', 'dispatch_kernel_0214', 'dispatch_kernel_0215', 'dispatch_kernel_0216', 'dispatch_kernel_0217', 'dispatch_kernel_0218', 'dispatch_kernel_0219', 'dispatch_kernel_0220', 'dispatch_kernel_0221', 'dispatch_kernel_0222', 'dispatch_kernel_0223', 'dispatch_kernel_0224', 'dispatch_kernel_0225', 'dispatch_kernel_0226', 'dispatch_kernel_0227', 'dispatch_kernel_0228', 'dispatch_kernel_0229', 'dispatch_kernel_0230', 'dispatch_kernel_0231', 'dispatch_kernel_0232']
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

