from __future__ import annotations

import ctypes
import os
import shutil
import time
from typing import Any

import torch
from cuda.bindings import driver, nvrtc


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
        if isinstance(arg, int):
            return ctypes.c_void_p(arg)
        raise TypeError(f"Pointer argument {ctype!r} requires a CUDA tensor or integer device pointer, got {type(arg)}")
    if normalized in {"float"}:
        return ctypes.c_float(float(arg))
    if normalized in {"double"}:
        return ctypes.c_double(float(arg))
    if normalized in {"uint64_t", "unsigned long long", "size_t"}:
        return ctypes.c_uint64(int(arg))
    if normalized in {"int64_t", "long long"}:
        return ctypes.c_int64(int(arg))
    if normalized in {"uint32_t", "unsigned int"}:
        return ctypes.c_uint32(int(arg))
    if normalized in {"int32_t", "int", "signed int"}:
        return ctypes.c_int32(int(arg))
    if isinstance(arg, float):
        return ctypes.c_float(arg)
    if isinstance(arg, int):
        return ctypes.c_int(arg)
    raise TypeError(f"Unsupported scalar argument {ctype!r}: {type(arg)}")


def _pack_args(args: list[Any], arg_types: list[str]) -> ctypes.Array:
    if len(args) != len(arg_types):
        raise ValueError(f"Argument count mismatch: got {len(args)}, expected {len(arg_types)}")
    c_args = [_marshal_arg(arg, ctype) for arg, ctype in zip(args, arg_types, strict=True)]
    ptrs = (ctypes.c_void_p * len(c_args))(*(ctypes.cast(ctypes.pointer(arg), ctypes.c_void_p) for arg in c_args))
    ptrs._prevent_gc = c_args  # type: ignore[attr-defined]
    return ptrs


class CUDAKernel:
    def __init__(self, cubin: bytes, func_name: str):
        self._closed = True
        self._func_name = func_name
        torch.empty(0, device="cuda")
        err, self._module = driver.cuModuleLoadData(cubin)
        _check(err, f"cuModuleLoadData failed for {func_name!r}")
        err, self._func = driver.cuModuleGetFunction(self._module, func_name.encode())
        _check(err, f"cuModuleGetFunction failed for {func_name!r}")
        self._dynamic_smem_opt_in_bytes = 0
        self._closed = False

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
        return driver.CUstream(stream.cuda_stream)

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
        args: list[Any],
        arg_types: list[str],
        shared_mem: int = 0,
        stream=None,
        timeout_ms: float | None = None,
    ) -> None:
        if self._closed:
            raise RuntimeError("Kernel has been unloaded")
        self._ensure_dynamic_smem_opt_in(shared_mem)
        packed = _pack_args(args, arg_types)
        cu_stream = self._cu_stream(stream)
        (err,) = driver.cuLaunchKernel(
            self._func,
            grid[0],
            grid[1],
            grid[2],
            block[0],
            block[1],
            block[2],
            shared_mem,
            cu_stream,
            packed,
            0,
        )
        _check(err, f"cuLaunchKernel failed for {self._func_name!r}")
        if timeout_ms is not None:
            self._wait_with_timeout(cu_stream, timeout_ms)

    def launch_cluster(
        self,
        *,
        grid: tuple[int, int, int],
        block: tuple[int, int, int],
        args: list[Any],
        arg_types: list[str],
        cluster_dims: tuple[int, int, int],
        shared_mem: int = 0,
        stream=None,
        timeout_ms: float | None = None,
    ) -> None:
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
        (err,) = driver.cuLaunchKernelEx(config, self._func, packed, 0)
        _check(err, f"cuLaunchKernelEx failed for {self._func_name!r}")
        if timeout_ms is not None:
            self._wait_with_timeout(cu_stream, timeout_ms)

    def launch_cooperative(
        self,
        *,
        grid: tuple[int, int, int],
        block: tuple[int, int, int],
        args: list[Any],
        arg_types: list[str],
        shared_mem: int = 0,
        stream=None,
        timeout_ms: float | None = None,
    ) -> None:
        if self._closed:
            raise RuntimeError("Kernel has been unloaded")
        self._ensure_dynamic_smem_opt_in(shared_mem)
        packed = _pack_args(args, arg_types)
        cu_stream = self._cu_stream(stream)
        (err,) = driver.cuLaunchCooperativeKernel(
            self._func,
            grid[0],
            grid[1],
            grid[2],
            block[0],
            block[1],
            block[2],
            shared_mem,
            cu_stream,
            packed,
        )
        _check(err, f"cuLaunchCooperativeKernel failed for {self._func_name!r}")
        if timeout_ms is not None:
            self._wait_with_timeout(cu_stream, timeout_ms)

    def close(self) -> None:
        if not self._closed:
            driver.cuModuleUnload(self._module)
            self._closed = True

    def __del__(self):
        try:
            self.close()
        except Exception:
            pass
