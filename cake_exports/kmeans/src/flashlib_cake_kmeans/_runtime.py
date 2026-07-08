from __future__ import annotations

import contextlib
import contextvars
import ctypes
import os
import shutil
import threading
import time
from collections.abc import Iterator, Sequence
from typing import Any

import torch
from cuda.bindings import driver, nvrtc


_LAUNCH_DEFAULTS: contextvars.ContextVar[tuple[str | None, Any, float | None] | None] = (
    contextvars.ContextVar("loom_export_launch_defaults", default=None)
)
_COMPILATION_CACHE_LOCK = threading.RLock()
_CUBIN_CACHE: dict[tuple[str, str, tuple[str, ...]], bytes] = {}
_MODULE_CACHE: dict[tuple[str, str, tuple[str, ...], int], "_LoadedModule"] = {}
_KERNEL_CACHE: dict[tuple[str, str, tuple[str, ...], int, str], "CUDAKernel"] = {}
_COMPILATION_CACHE_GENERATION = 0
_RUNTIME_ACTIVITY_COUNTS = {
    "source_reads": 0,
    "nvrtc_compiles": 0,
    "module_loads": 0,
}


def _record_runtime_activity(name: str) -> None:
    with _COMPILATION_CACHE_LOCK:
        _RUNTIME_ACTIVITY_COUNTS[name] += 1


def record_source_read() -> None:
    _record_runtime_activity("source_reads")


def runtime_activity_snapshot() -> dict[str, int]:
    with _COMPILATION_CACHE_LOCK:
        return dict(_RUNTIME_ACTIVITY_COUNTS)


class NVRTCError(RuntimeError):
    def __init__(self, message: str, log: str):
        self.log = log
        super().__init__(f"{message}\n--- NVRTC log ---\n{log}")


def _check(err: int, msg: str) -> None:
    if err != 0:
        raise RuntimeError(f"{msg}: CUresult={err}")


def _nvrtc_check(err: int, msg: str) -> None:
    if err != 0:
        raise RuntimeError(f"{msg}: nvrtcResult={err}")


def _arch_flag_for_cc(major: int, minor: int) -> str:
    sm = int(major) * 10 + int(minor)
    return f"sm_{sm}a" if sm >= 90 else f"sm_{sm}"


def detect_gpu_arch() -> str:
    forced = os.environ.get("LOOM_EXPORTED_FORCE_ARCH") or os.environ.get("LOOM_FORCE_ARCH")
    if forced:
        return forced
    try:
        (err,) = driver.cuInit(0)
        if err != 0:
            return "sm_100a"
        err, dev = driver.cuDeviceGet(0)
        if err != 0:
            return "sm_100a"
        err, major = driver.cuDeviceGetAttribute(
            driver.CUdevice_attribute.CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR, dev
        )
        err2, minor = driver.cuDeviceGetAttribute(
            driver.CUdevice_attribute.CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR, dev
        )
        if err != 0 or err2 != 0:
            return "sm_100a"
        return _arch_flag_for_cc(int(major), int(minor))
    except Exception:
        return "sm_100a"


def resolve_gpu_arch(arch: str | None) -> str:
    return detect_gpu_arch() if arch is None else str(arch)


def current_cuda_device_index() -> int:
    return int(torch.cuda.current_device())


def _is_torch_stream(stream: Any) -> bool:
    return isinstance(stream, torch.cuda.Stream)


def launch_stream_context(stream: Any):
    return torch.cuda.stream(stream) if _is_torch_stream(stream) else contextlib.nullcontext()


@contextlib.contextmanager
def launch_context(
    *,
    arch: str | None = None,
    stream: Any = None,
    timeout_ms: float | None = None,
) -> Iterator[None]:
    '''Apply semantic-call launch defaults to every frozen kernel stage.

    The current PyTorch stream is captured lazily once before the first frozen
    launch when no explicit stream is supplied.  Passing a PyTorch stream also
    makes dispatcher-side PyTorch tensor operations use that stream, not only
    the generated driver launches. Context variables keep concurrent threads
    and nested semantic calls isolated.
    '''
    resolved_stream = stream
    token = _LAUNCH_DEFAULTS.set((arch, resolved_stream, timeout_ms))
    stream_context = launch_stream_context(resolved_stream)
    try:
        with stream_context:
            yield
    finally:
        _LAUNCH_DEFAULTS.reset(token)


def resolve_launch_defaults(
    *,
    arch: str | None,
    stream: Any,
    timeout_ms: float | None,
) -> tuple[str | None, Any, float | None]:
    defaults = _LAUNCH_DEFAULTS.get()
    if defaults is None:
        return arch, stream, timeout_ms
    default_arch, default_stream, default_timeout_ms = defaults
    if stream is None and default_stream is None:
        default_stream = torch.cuda.current_stream()
        _LAUNCH_DEFAULTS.set((default_arch, default_stream, default_timeout_ms))
    return (
        default_arch if arch is None else arch,
        default_stream if stream is None else stream,
        default_timeout_ms if timeout_ms is None else timeout_ms,
    )


def compilation_cache_generation() -> int:
    return _COMPILATION_CACHE_GENERATION


def _cuda_include_dirs() -> list[str]:
    candidates = ["/usr/local/cuda/include"]
    nvcc = shutil.which("nvcc")
    if nvcc:
        candidates.insert(0, os.path.join(os.path.dirname(os.path.dirname(nvcc)), "include"))
    return [d for d in candidates if os.path.isdir(d)]


def _get_compile_log(prog: nvrtc.nvrtcProgram) -> str:
    err, size = nvrtc.nvrtcGetProgramLogSize(prog)
    if err != 0 or size <= 1:
        return ""
    log = b"\x00" * size
    nvrtc.nvrtcGetProgramLog(prog, log)
    return log.decode(errors="replace").rstrip("\x00")


def compile_cuda(
    source: str,
    *,
    arch: str | None = None,
    name: str = "kernel.cu",
    options: list[str] | None = None,
) -> bytes:
    _record_runtime_activity("nvrtc_compiles")
    if arch is None:
        arch = detect_gpu_arch()
    opts = [
        f"--gpu-architecture={arch}",
        "-std=c++17",
        "-default-device",
    ]
    for include_dir in _cuda_include_dirs():
        opts.append(f"-I{include_dir}")
        cccl = os.path.join(include_dir, "cccl")
        if os.path.exists(os.path.join(cccl, "cuda", "std")):
            opts.append(f"-I{cccl}")
    if options:
        opts.extend(options)

    err, prog = nvrtc.nvrtcCreateProgram(source.encode(), name.encode(), 0, [], [])
    _nvrtc_check(err, "nvrtcCreateProgram failed")
    try:
        encoded_opts = [opt.encode() for opt in opts]
        (err,) = nvrtc.nvrtcCompileProgram(prog, len(encoded_opts), encoded_opts)
        if err != 0:
            raise NVRTCError(f"Compilation failed for {name!r}", _get_compile_log(prog))
        err, size = nvrtc.nvrtcGetCUBINSize(prog)
        _nvrtc_check(err, "nvrtcGetCUBINSize failed")
        cubin = b"\x00" * size
        (err,) = nvrtc.nvrtcGetCUBIN(prog, cubin)
        _nvrtc_check(err, "nvrtcGetCUBIN failed")
        return cubin
    finally:
        nvrtc.nvrtcDestroyProgram(prog)


def _is_tensor(arg: Any) -> bool:
    return isinstance(arg, torch.Tensor)


def _marshal_arg(arg: Any, ctype: str) -> ctypes._SimpleCData:
    normalized = " ".join(ctype.replace("__restrict__", "").split())
    if "*" in normalized:
        if _is_tensor(arg):
            return ctypes.c_void_p(arg.data_ptr())
        if isinstance(arg, ctypes.c_void_p):
            return arg
        if isinstance(arg, int) and not isinstance(arg, bool):
            return ctypes.c_void_p(arg)
        raise TypeError(f"Pointer argument {ctype!r} requires a CUDA tensor or integer device pointer, got {type(arg)}")
    if normalized in {"bool", "_Bool"}:
        if not isinstance(arg, bool):
            raise TypeError(f"Scalar argument {ctype!r} requires bool, got {type(arg)}")
        return ctypes.c_bool(arg)
    integer_abis = {
        "int8_t": (ctypes.c_int8, -(1 << 7), (1 << 7) - 1),
        "signed char": (ctypes.c_int8, -(1 << 7), (1 << 7) - 1),
        "uint8_t": (ctypes.c_uint8, 0, (1 << 8) - 1),
        "unsigned char": (ctypes.c_uint8, 0, (1 << 8) - 1),
        "int16_t": (ctypes.c_int16, -(1 << 15), (1 << 15) - 1),
        "short": (ctypes.c_int16, -(1 << 15), (1 << 15) - 1),
        "short int": (ctypes.c_int16, -(1 << 15), (1 << 15) - 1),
        "uint16_t": (ctypes.c_uint16, 0, (1 << 16) - 1),
        "unsigned short": (ctypes.c_uint16, 0, (1 << 16) - 1),
        "unsigned short int": (ctypes.c_uint16, 0, (1 << 16) - 1),
        "int32_t": (ctypes.c_int32, -(1 << 31), (1 << 31) - 1),
        "int": (ctypes.c_int32, -(1 << 31), (1 << 31) - 1),
        "signed int": (ctypes.c_int32, -(1 << 31), (1 << 31) - 1),
        "uint32_t": (ctypes.c_uint32, 0, (1 << 32) - 1),
        "unsigned int": (ctypes.c_uint32, 0, (1 << 32) - 1),
        "int64_t": (ctypes.c_int64, -(1 << 63), (1 << 63) - 1),
        "long long": (ctypes.c_int64, -(1 << 63), (1 << 63) - 1),
        "uint64_t": (ctypes.c_uint64, 0, (1 << 64) - 1),
        "unsigned long long": (ctypes.c_uint64, 0, (1 << 64) - 1),
        "size_t": (ctypes.c_uint64, 0, (1 << 64) - 1),
    }
    integer_abi = integer_abis.get(normalized)
    if integer_abi is not None:
        if not isinstance(arg, int) or isinstance(arg, bool):
            raise TypeError(f"Scalar argument {ctype!r} requires int, got {type(arg)}")
        scalar_ctype, minimum, maximum = integer_abi
        if arg < minimum or arg > maximum:
            raise OverflowError(f"Scalar argument {ctype!r} is out of range: {arg}")
        return scalar_ctype(arg)
    if normalized in {"float", "double"}:
        if not isinstance(arg, (int, float)) or isinstance(arg, bool):
            raise TypeError(f"Scalar argument {ctype!r} requires int or float, got {type(arg)}")
        return ctypes.c_float(arg) if normalized == "float" else ctypes.c_double(arg)
    raise TypeError(f"Unsupported scalar ABI type {ctype!r}")


def _pack_args(args: Sequence[Any], arg_types: Sequence[str]) -> ctypes.Array:
    if len(args) != len(arg_types):
        raise ValueError(f"Argument count mismatch: got {len(args)}, expected {len(arg_types)}")
    c_args = [_marshal_arg(arg, ctype) for arg, ctype in zip(args, arg_types, strict=True)]
    ptrs = (ctypes.c_void_p * len(c_args))(*(ctypes.cast(ctypes.pointer(arg), ctypes.c_void_p) for arg in c_args))
    ptrs._prevent_gc = c_args  # type: ignore[attr-defined]
    return ptrs


class PreparedCUDAKernelLaunch:
    # A fully marshalled generated-runtime launch reusable on the hot path.

    def __init__(
        self,
        kernel,
        *,
        mode,
        grid,
        block,
        arg_types,
        packed,
        keepalive,
        shared_mem,
        cu_stream,
        cluster_dims=None,
        config=None,
    ):
        self._kernel = kernel
        self._mode = mode
        self._grid = tuple(grid)
        self._block = tuple(block)
        self._arg_types = tuple(arg_types)
        self._packed = packed
        self._keepalive = keepalive
        self._shared_mem = shared_mem
        self._cu_stream = cu_stream
        self._cluster_dims = None if cluster_dims is None else tuple(cluster_dims)
        self._config = config

    def rebind(
        self,
        kernel,
        *,
        mode,
        grid,
        block,
        args,
        arg_types,
        shared_mem,
        stream=None,
        cluster_dims=None,
    ):
        # Reuse the existing void** and scalar carriers while replacing values.

        candidate_grid = tuple(grid)
        candidate_block = tuple(block)
        candidate_arg_types = tuple(arg_types)
        candidate_cluster_dims = None if cluster_dims is None else tuple(cluster_dims)
        mismatches = []
        if kernel is not self._kernel:
            mismatches.append("kernel")
        if mode != self._mode:
            mismatches.append(f"mode ({self._mode!r} != {mode!r})")
        if candidate_grid != self._grid:
            mismatches.append(f"grid ({self._grid!r} != {candidate_grid!r})")
        if candidate_block != self._block:
            mismatches.append(f"block ({self._block!r} != {candidate_block!r})")
        if int(shared_mem) != self._shared_mem:
            mismatches.append(f"shared_mem ({self._shared_mem!r} != {int(shared_mem)!r})")
        if candidate_cluster_dims != self._cluster_dims:
            mismatches.append(
                f"cluster_dims ({self._cluster_dims!r} != {candidate_cluster_dims!r})"
            )
        if candidate_arg_types != self._arg_types:
            mismatches.append("arg_types")
        if mismatches:
            raise RuntimeError(
                "prepared CUDA launch topology mismatch: " + ", ".join(mismatches)
            )

        if len(args) != len(self._arg_types):
            raise RuntimeError(
                f"prepared CUDA launch argument count mismatch: "
                f"expected {len(self._arg_types)}, got {len(args)}"
            )
        return self.rebind_arguments(dict(enumerate(args)), stream=stream)

    def rebind_arguments(self, replacements, *, stream=None):
        # Update selected ABI carriers without rebuilding or traversing the launch.

        old_c_args = self._packed._prevent_gc
        if len(old_c_args) != len(self._arg_types):
            raise RuntimeError("prepared CUDA launch has a corrupt packed argument array")
        rebound = []
        for index, arg in replacements.items():
            if type(index) is not int or index < 0 or index >= len(self._arg_types):
                raise IndexError(f"prepared CUDA launch argument index is out of range: {index!r}")
            new_arg = _marshal_arg(arg, self._arg_types[index])
            old_arg = old_c_args[index]
            if type(old_arg) is not type(new_arg):
                raise RuntimeError(
                    f"prepared CUDA launch ABI mismatch at argument {index}: "
                    f"{type(old_arg).__name__} != {type(new_arg).__name__}"
                )
            rebound.append((index, arg, old_arg, new_arg))

        cu_stream = self._kernel._cu_stream(stream)
        keepalive = list(self._keepalive)
        for index, arg, old_arg, new_arg in rebound:
            old_arg.value = new_arg.value
            keepalive[index] = arg
        self._keepalive = tuple(keepalive)
        self._cu_stream = cu_stream
        return self

    def rebind_tensor_arguments(
        self,
        bindings,
        inputs,
        *,
        stream=None,
        preserve_stream=False,
        retain_inputs=True,
        pointer_values=None,
        inputs_already_scrubbed=False,
    ):
        # Captured public bindings are pointer-only. Update their existing
        # c_void_p carriers directly instead of rematerializing ctypes objects.

        if not isinstance(preserve_stream, bool):
            raise TypeError("preserve_stream must be a bool")
        if not isinstance(retain_inputs, bool):
            raise TypeError("retain_inputs must be a bool")
        if not isinstance(inputs_already_scrubbed, bool):
            raise TypeError("inputs_already_scrubbed must be a bool")
        if inputs_already_scrubbed and retain_inputs:
            raise ValueError("inputs_already_scrubbed requires retain_inputs=False")
        if pointer_values is not None and retain_inputs:
            raise ValueError("pointer_values requires retain_inputs=False")
        old_c_args = self._packed._prevent_gc
        if len(old_c_args) != len(self._arg_types):
            raise RuntimeError("prepared CUDA launch has a corrupt packed argument array")
        replacements = []
        for index, key in bindings:
            if type(index) is not int or index < 0 or index >= len(self._arg_types):
                raise IndexError(f"prepared CUDA launch argument index is out of range: {index!r}")
            arg = None
            if pointer_values is None:
                try:
                    arg = inputs[key]
                except KeyError:
                    raise KeyError(f"missing prepared CUDA tensor binding: {key!r}") from None
                data_ptr = getattr(arg, "data_ptr", None)
                if not callable(data_ptr):
                    raise TypeError(f"prepared CUDA tensor binding {key!r} is not tensor-like")
                pointer = int(data_ptr())
            else:
                try:
                    pointer = pointer_values[key]
                except KeyError:
                    raise KeyError(f"missing prepared CUDA pointer binding: {key!r}") from None
                if type(pointer) is not int:
                    raise TypeError(f"prepared CUDA pointer binding {key!r} is not an int")
            old_arg = old_c_args[index]
            if type(old_arg) is not ctypes.c_void_p:
                raise RuntimeError(
                    f"prepared CUDA tensor binding {key!r} does not target a pointer carrier"
                )
            if inputs_already_scrubbed and type(self._keepalive[index]) is not int:
                raise RuntimeError(
                    f"prepared CUDA tensor binding {key!r} retained an unexpected caller reference"
                )
            replacements.append((index, key, arg, pointer, old_arg))

        keepalive = None if inputs_already_scrubbed else list(self._keepalive)
        for index, key, arg, pointer, old_arg in replacements:
            old_arg.value = pointer
            if keepalive is not None:
                if retain_inputs and arg is None:
                    arg = inputs[key]
                keepalive[index] = arg if retain_inputs else pointer
        if keepalive is not None:
            self._keepalive = tuple(keepalive)
        if not preserve_stream:
            self._cu_stream = self._kernel._cu_stream(stream)
        return self

    def _scrub_stream_bound_pointer_keepalives(self, bindings):
        '''Drop selected tensor references without changing carriers or stream.'''

        old_c_args = self._packed._prevent_gc
        if len(old_c_args) != len(self._arg_types):
            raise RuntimeError("prepared CUDA launch has a corrupt packed argument array")
        scrubbed = list(self._keepalive)
        program = []
        for index, key in bindings:
            if type(index) is not int or index < 0 or index >= len(self._arg_types):
                raise IndexError(f"prepared CUDA launch argument index is out of range: {index!r}")
            carrier = old_c_args[index]
            if type(carrier) is not ctypes.c_void_p:
                raise RuntimeError(
                    f"prepared CUDA tensor binding {key!r} does not target a pointer carrier"
                )
            value = scrubbed[index]
            data_ptr = getattr(value, "data_ptr", None)
            if not callable(data_ptr):
                raise RuntimeError(
                    f"prepared CUDA tensor binding {key!r} retained an unexpected value"
                )
            pointer = int(data_ptr())
            if carrier.value != pointer:
                raise RuntimeError(
                    f"prepared CUDA tensor binding {key!r} carrier disagrees with its keepalive"
                )
            scrubbed[index] = pointer
            program.append((key, carrier))
        self._keepalive = tuple(scrubbed)
        return tuple(program)

    def launch(self, *, stream=None, timeout_ms=None):
        kernel = self._kernel
        if kernel._closed:
            raise RuntimeError("Kernel has been unloaded")
        cu_stream = self._cu_stream if stream is None else kernel._cu_stream(stream)
        if self._mode == "cluster":
            self._config.hStream = cu_stream
            (err,) = driver.cuLaunchKernelEx(self._config, kernel._func, self._packed, 0)
            api = "cuLaunchKernelEx"
        elif self._mode == "cooperative":
            (err,) = driver.cuLaunchCooperativeKernel(
                kernel._func,
                *self._grid,
                *self._block,
                self._shared_mem,
                cu_stream,
                self._packed,
            )
            api = "cuLaunchCooperativeKernel"
        else:
            (err,) = driver.cuLaunchKernel(
                kernel._func,
                *self._grid,
                *self._block,
                self._shared_mem,
                cu_stream,
                self._packed,
                0,
            )
            api = "cuLaunchKernel"
        _check(err, f"{api} failed for {kernel._func_name!r}")
        if timeout_ms is not None:
            kernel._wait_with_timeout(cu_stream, timeout_ms)


class _LoadedModule:
    def __init__(self, cubin: bytes):
        _record_runtime_activity("module_loads")
        torch.empty(0, device="cuda")
        err, self._module = driver.cuModuleLoadData(cubin)
        _check(err, "cuModuleLoadData failed")
        self._functions: dict[str, Any] = {}
        self._closed = False

    @property
    def handle(self):
        return self._module

    @property
    def closed(self) -> bool:
        return self._closed

    def function(self, func_name: str):
        if self._closed:
            raise RuntimeError("CUDA module has been unloaded")
        function = self._functions.get(func_name)
        if function is None:
            err, function = driver.cuModuleGetFunction(self._module, func_name.encode())
            _check(err, f"cuModuleGetFunction failed for {func_name!r}")
            self._functions[func_name] = function
        return function

    def close(self) -> None:
        if not self._closed:
            driver.cuModuleUnload(self._module)
            self._closed = True


class CUDAKernel:
    def __init__(self, cubin: bytes, func_name: str):
        self._module_owner = _LoadedModule(cubin)
        self._owns_module = True
        self._initialize_from_module(func_name)

    @classmethod
    def from_loaded_module(cls, module: _LoadedModule, func_name: str) -> "CUDAKernel":
        kernel = cls.__new__(cls)
        kernel._module_owner = module
        kernel._owns_module = False
        kernel._initialize_from_module(func_name)
        return kernel

    def _initialize_from_module(self, func_name: str) -> None:
        self._closed = True
        self._func_name = func_name
        self._module = self._module_owner.handle
        self._func = self._module_owner.function(func_name)
        self._dynamic_smem_opt_in_bytes = 0
        self._closed = False

    @property
    def closed(self) -> bool:
        return self._closed

    def _ensure_dynamic_smem_opt_in(self, shared_mem: int) -> None:
        if shared_mem <= 48 * 1024 or shared_mem <= self._dynamic_smem_opt_in_bytes:
            return
        (err,) = driver.cuFuncSetAttribute(
            self._func,
            driver.CUfunction_attribute.CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES,
            shared_mem,
        )
        _check(err, "cuFuncSetAttribute MAX_DYNAMIC_SHARED_SIZE_BYTES failed")
        self._dynamic_smem_opt_in_bytes = max(self._dynamic_smem_opt_in_bytes, shared_mem)

    def _cu_stream(self, stream=None):
        if stream is None:
            stream = torch.cuda.current_stream()
        handle = getattr(stream, "cuda_stream", stream)
        return driver.CUstream(int(handle))

    def _wait_with_timeout(self, cu_stream: driver.CUstream, timeout_ms: float) -> None:
        deadline = time.monotonic() + timeout_ms / 1000.0
        while True:
            (err,) = driver.cuStreamQuery(cu_stream)
            if err == 0:
                return
            if err != 600:
                _check(err, f"cuStreamQuery failed for {self._func_name!r}")
            if time.monotonic() >= deadline:
                raise TimeoutError(f"Kernel {self._func_name!r} did not complete within {timeout_ms:.0f} ms")
            time.sleep(0.001)

    def launch(
        self,
        *,
        grid: tuple[int, int, int],
        block: tuple[int, int, int],
        args: Sequence[Any],
        arg_types: Sequence[str],
        shared_mem: int = 0,
        stream=None,
        timeout_ms: float | None = None,
    ) -> None:
        self.prepare_launch(
            grid=grid,
            block=block,
            args=args,
            arg_types=arg_types,
            shared_mem=shared_mem,
            stream=stream,
        ).launch(timeout_ms=timeout_ms)

    def prepare_launch(
        self,
        *,
        grid,
        block,
        args,
        arg_types,
        shared_mem=0,
        stream=None,
    ):
        if self._closed:
            raise RuntimeError("Kernel has been unloaded")
        self._ensure_dynamic_smem_opt_in(shared_mem)
        return PreparedCUDAKernelLaunch(
            self,
            mode="regular",
            grid=grid,
            block=block,
            arg_types=arg_types,
            packed=_pack_args(args, arg_types),
            keepalive=tuple(args),
            shared_mem=shared_mem,
            cu_stream=self._cu_stream(stream),
        )

    def rebind_launch(
        self,
        prepared,
        *,
        grid,
        block,
        args,
        arg_types,
        shared_mem=0,
        stream=None,
    ):
        if not isinstance(prepared, PreparedCUDAKernelLaunch):
            raise TypeError("prepared must be a PreparedCUDAKernelLaunch")
        return prepared.rebind(
            self,
            mode="regular",
            grid=grid,
            block=block,
            args=args,
            arg_types=arg_types,
            shared_mem=shared_mem,
            stream=stream,
        )

    def launch_cluster(
        self,
        *,
        grid: tuple[int, int, int],
        block: tuple[int, int, int],
        args: Sequence[Any],
        arg_types: Sequence[str],
        cluster_dims: tuple[int, int, int],
        shared_mem: int = 0,
        stream=None,
        timeout_ms: float | None = None,
    ) -> None:
        self.prepare_launch_cluster(
            grid=grid,
            block=block,
            args=args,
            arg_types=arg_types,
            cluster_dims=cluster_dims,
            shared_mem=shared_mem,
            stream=stream,
        ).launch(timeout_ms=timeout_ms)

    def prepare_launch_cluster(
        self,
        *,
        grid,
        block,
        args,
        arg_types,
        cluster_dims,
        shared_mem=0,
        stream=None,
    ):
        if self._closed:
            raise RuntimeError("Kernel has been unloaded")
        self._ensure_dynamic_smem_opt_in(shared_mem)
        packed = _pack_args(args, arg_types)
        cu_stream = self._cu_stream(stream)

        attr_cluster = driver.CUlaunchAttribute()
        attr_cluster.id = driver.CUlaunchAttributeID.CU_LAUNCH_ATTRIBUTE_CLUSTER_DIMENSION
        attr_cluster.value.clusterDim.x = cluster_dims[0]
        attr_cluster.value.clusterDim.y = cluster_dims[1]
        attr_cluster.value.clusterDim.z = cluster_dims[2]

        attr_sched = driver.CUlaunchAttribute()
        attr_sched.id = driver.CUlaunchAttributeID.CU_LAUNCH_ATTRIBUTE_CLUSTER_SCHEDULING_POLICY_PREFERENCE
        attr_sched.value.clusterSchedulingPolicyPreference = (
            driver.CUclusterSchedulingPolicy.CU_CLUSTER_SCHEDULING_POLICY_SPREAD
        )

        config = driver.CUlaunchConfig()
        config.gridDimX = grid[0]
        config.gridDimY = grid[1]
        config.gridDimZ = grid[2]
        config.blockDimX = block[0]
        config.blockDimY = block[1]
        config.blockDimZ = block[2]
        config.sharedMemBytes = shared_mem
        config.hStream = cu_stream
        config.attrs = [attr_cluster, attr_sched]
        config.numAttrs = 2
        return PreparedCUDAKernelLaunch(
            self,
            mode="cluster",
            grid=grid,
            block=block,
            arg_types=arg_types,
            packed=packed,
            keepalive=tuple(args),
            shared_mem=shared_mem,
            cu_stream=cu_stream,
            cluster_dims=cluster_dims,
            config=config,
        )

    def rebind_launch_cluster(
        self,
        prepared,
        *,
        grid,
        block,
        args,
        arg_types,
        cluster_dims,
        shared_mem=0,
        stream=None,
    ):
        if not isinstance(prepared, PreparedCUDAKernelLaunch):
            raise TypeError("prepared must be a PreparedCUDAKernelLaunch")
        return prepared.rebind(
            self,
            mode="cluster",
            grid=grid,
            block=block,
            args=args,
            arg_types=arg_types,
            cluster_dims=cluster_dims,
            shared_mem=shared_mem,
            stream=stream,
        )

    def launch_cooperative(
        self,
        *,
        grid: tuple[int, int, int],
        block: tuple[int, int, int],
        args: Sequence[Any],
        arg_types: Sequence[str],
        shared_mem: int = 0,
        stream=None,
        timeout_ms: float | None = None,
    ) -> None:
        self.prepare_launch_cooperative(
            grid=grid,
            block=block,
            args=args,
            arg_types=arg_types,
            shared_mem=shared_mem,
            stream=stream,
        ).launch(timeout_ms=timeout_ms)

    def prepare_launch_cooperative(
        self,
        *,
        grid,
        block,
        args,
        arg_types,
        shared_mem=0,
        stream=None,
    ):
        if self._closed:
            raise RuntimeError("Kernel has been unloaded")
        self._ensure_dynamic_smem_opt_in(shared_mem)
        return PreparedCUDAKernelLaunch(
            self,
            mode="cooperative",
            grid=grid,
            block=block,
            arg_types=arg_types,
            packed=_pack_args(args, arg_types),
            keepalive=tuple(args),
            shared_mem=shared_mem,
            cu_stream=self._cu_stream(stream),
        )

    def rebind_launch_cooperative(
        self,
        prepared,
        *,
        grid,
        block,
        args,
        arg_types,
        shared_mem=0,
        stream=None,
    ):
        if not isinstance(prepared, PreparedCUDAKernelLaunch):
            raise TypeError("prepared must be a PreparedCUDAKernelLaunch")
        return prepared.rebind(
            self,
            mode="cooperative",
            grid=grid,
            block=block,
            args=args,
            arg_types=arg_types,
            shared_mem=shared_mem,
            stream=stream,
        )

    def close(self) -> None:
        if not self._closed:
            if self._owns_module:
                self._module_owner.close()
            self._closed = True

    def __del__(self):
        try:
            self.close()
        except Exception:
            pass


def load_cached_kernel(
    source: str,
    *,
    source_digest: str,
    func_name: str,
    arch: str,
    device_index: int,
    name: str,
    options: Sequence[str] = (),
) -> CUDAKernel:
    '''Compile/load one content-addressed kernel shared by equivalent aliases.

    Identical source, architecture, and options share NVRTC output.  The active
    device is added for module loading, and the function symbol only for the
    final wrapper, so a multi-entrypoint translation unit loads one module per
    device without conflating its functions.
    '''
    option_tuple = tuple(options)
    cubin_key = (source_digest, arch, option_tuple)
    module_key = (*cubin_key, int(device_index))
    kernel_key = (*module_key, func_name)
    with _COMPILATION_CACHE_LOCK:
        kernel = _KERNEL_CACHE.get(kernel_key)
        if kernel is not None and not kernel.closed:
            return kernel
        cubin = _CUBIN_CACHE.get(cubin_key)
        if cubin is None:
            cubin = compile_cuda(source, arch=arch, name=name, options=list(option_tuple))
            _CUBIN_CACHE[cubin_key] = cubin
        module = _MODULE_CACHE.get(module_key)
        if module is None or module.closed:
            module = _LoadedModule(cubin)
            _MODULE_CACHE[module_key] = module
        kernel = CUDAKernel.from_loaded_module(module, func_name)
        _KERNEL_CACHE[kernel_key] = kernel
        return kernel


def clear_compilation_cache() -> None:
    '''Clear process-local cubin/module caches (primarily for diagnostics).'''
    global _COMPILATION_CACHE_GENERATION
    with _COMPILATION_CACHE_LOCK:
        for kernel in _KERNEL_CACHE.values():
            kernel._closed = True
        for module in _MODULE_CACHE.values():
            module.close()
        _KERNEL_CACHE.clear()
        _MODULE_CACHE.clear()
        _CUBIN_CACHE.clear()
        _COMPILATION_CACHE_GENERATION += 1
