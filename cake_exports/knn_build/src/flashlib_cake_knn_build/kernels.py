from __future__ import annotations

import json
from dataclasses import dataclass
from importlib import resources
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from ._runtime import CUDAKernel


@dataclass(frozen=True)
class KernelSpec:
    name: str
    symbol: str
    source: str
    threads: int
    shared_mem_bytes: int
    cluster_dims: tuple[int, int, int]
    launch_mode: str
    parameters: tuple[dict[str, str], ...]
    specializations: dict[str, int | str]
    compile_options: tuple[str, ...]

    @staticmethod
    def from_manifest(entry: dict[str, Any]) -> "KernelSpec":
        return KernelSpec(
            name=entry["name"],
            symbol=entry["symbol"],
            source=entry["source"],
            threads=int(entry["threads"]),
            shared_mem_bytes=int(entry["shared_mem_bytes"]),
            cluster_dims=tuple(int(v) for v in entry["cluster_dims"]),
            launch_mode=entry["launch_mode"],
            parameters=tuple(entry["parameters"]),
            specializations=dict(entry.get("specializations", {})),
            compile_options=tuple(str(option) for option in entry.get("compile_options", ())),
        )


class ExportedKernel:
    def __init__(self, spec: KernelSpec):
        self.spec = spec
        self._compiled: dict[tuple[str, tuple[str, ...], int, int], "CUDAKernel"] = {}
        self._arg_types = tuple(parameter["ctype"] for parameter in spec.parameters)
        self._default_block = (spec.threads, 1, 1)
        self._default_shared_mem = spec.shared_mem_bytes
        self._source_cache: str | None = None
        self._source_digest: str | None = None

    @property
    def parameters(self) -> tuple[dict[str, str], ...]:
        return self.spec.parameters

    @property
    def arg_types(self) -> tuple[str, ...]:
        return self._arg_types

    def source_text(self) -> str:
        if self._source_cache is None:
            from ._runtime import record_source_read

            package = __package__ or __name__.rpartition(".")[0]
            self._source_cache = resources.files(package).joinpath(self.spec.source).read_text(encoding="utf-8")
            record_source_read()
        return self._source_cache

    def compile(self, *, arch: str | None = None, options: list[str] | None = None) -> "CUDAKernel":
        effective_options = tuple(dict.fromkeys((*self.spec.compile_options, *(options or ()))))
        from ._runtime import (
            compilation_cache_generation,
            current_cuda_device_index,
            load_cached_kernel,
            resolve_gpu_arch,
        )

        resolved_arch = resolve_gpu_arch(arch)
        device_index = current_cuda_device_index()
        generation = compilation_cache_generation()
        key = (resolved_arch, effective_options, device_index, generation)
        kernel = self._compiled.get(key)
        if kernel is None or kernel.closed:
            source = self.source_text()
            if self._source_digest is None:
                import hashlib

                self._source_digest = hashlib.sha256(source.encode("utf-8")).hexdigest()
            kernel = load_cached_kernel(
                source,
                source_digest=self._source_digest,
                func_name=self.spec.symbol,
                arch=resolved_arch,
                device_index=device_index,
                name=f"{self.spec.name}.cu",
                options=effective_options,
            )
            self._compiled[key] = kernel
        return kernel

    def launch(
        self,
        *args,
        grid: tuple[int, int, int],
        block: tuple[int, int, int] | None = None,
        shared_mem: int | None = None,
        stream=None,
        timeout_ms: float | None = None,
        arch: str | None = None,
        options: list[str] | None = None,
    ) -> None:
        from ._runtime import resolve_launch_defaults

        arch, stream, timeout_ms = resolve_launch_defaults(
            arch=arch,
            stream=stream,
            timeout_ms=timeout_ms,
        )
        prepared = self.prepare_launch(
            *args,
            grid=grid,
            block=block,
            shared_mem=shared_mem,
            stream=stream,
            arch=arch,
            options=options,
        )
        prepared.launch(timeout_ms=timeout_ms)

    def prepare_launch(
        self,
        *args,
        grid: tuple[int, int, int],
        block: tuple[int, int, int] | None = None,
        shared_mem: int | None = None,
        stream=None,
        arch: str | None = None,
        options: list[str] | None = None,
    ):
        # Compile and fully marshal one launch without submitting GPU work.

        if len(args) != len(self.spec.parameters):
            expected = ", ".join(p["name"] for p in self.spec.parameters)
            raise TypeError(f"{self.spec.name} expects {len(self.spec.parameters)} args ({expected}), got {len(args)}")
        from ._runtime import launch_stream_context, resolve_launch_defaults

        arch, stream, _ = resolve_launch_defaults(
            arch=arch,
            stream=stream,
            timeout_ms=None,
        )
        if block is None:
            block = self._default_block
        if shared_mem is None:
            shared_mem = self._default_shared_mem
        with launch_stream_context(stream):
            kernel = self.compile(arch=arch, options=options)
            if self.spec.launch_mode == "cluster":
                return kernel.prepare_launch_cluster(
                    grid=grid,
                    block=block,
                    args=args,
                    arg_types=self._arg_types,
                    cluster_dims=self.spec.cluster_dims,
                    shared_mem=shared_mem,
                    stream=stream,
                )
            if self.spec.launch_mode == "cooperative":
                return kernel.prepare_launch_cooperative(
                    grid=grid,
                    block=block,
                    args=args,
                    arg_types=self._arg_types,
                    shared_mem=shared_mem,
                    stream=stream,
                )
            return kernel.prepare_launch(
                grid=grid,
                block=block,
                args=args,
                arg_types=self._arg_types,
                shared_mem=shared_mem,
                stream=stream,
            )


def _load_manifest() -> dict[str, Any]:
    package = __package__ or __name__.rpartition(".")[0]
    return json.loads(resources.files(package).joinpath("manifest.json").read_text(encoding="utf-8"))


_MANIFEST = _load_manifest()
KERNELS = {entry["name"]: ExportedKernel(KernelSpec.from_manifest(entry)) for entry in _MANIFEST["kernels"]}


def get_kernel(name: str) -> ExportedKernel:
    try:
        return KERNELS[name]
    except KeyError as exc:
        available = ", ".join(sorted(KERNELS))
        raise KeyError(f"Unknown exported kernel {name!r}. Available: {available}") from exc


dispatch_kernel_0000 = get_kernel('dispatch_kernel_0000')


def launch_dispatch_kernel_0000(
    *args,
    grid: tuple[int, int, int],
    block: tuple[int, int, int] | None = None,
    shared_mem: int | None = None,
    stream=None,
    timeout_ms: float | None = None,
    arch: str | None = None,
    options: list[str] | None = None,
):
    return dispatch_kernel_0000.launch(
        *args,
        grid=grid,
        block=block,
        shared_mem=shared_mem,
        stream=stream,
        timeout_ms=timeout_ms,
        arch=arch,
        options=options,
    )


dispatch_kernel_0001 = get_kernel('dispatch_kernel_0001')


def launch_dispatch_kernel_0001(
    *args,
    grid: tuple[int, int, int],
    block: tuple[int, int, int] | None = None,
    shared_mem: int | None = None,
    stream=None,
    timeout_ms: float | None = None,
    arch: str | None = None,
    options: list[str] | None = None,
):
    return dispatch_kernel_0001.launch(
        *args,
        grid=grid,
        block=block,
        shared_mem=shared_mem,
        stream=stream,
        timeout_ms=timeout_ms,
        arch=arch,
        options=options,
    )


dispatch_kernel_0002 = get_kernel('dispatch_kernel_0002')


def launch_dispatch_kernel_0002(
    *args,
    grid: tuple[int, int, int],
    block: tuple[int, int, int] | None = None,
    shared_mem: int | None = None,
    stream=None,
    timeout_ms: float | None = None,
    arch: str | None = None,
    options: list[str] | None = None,
):
    return dispatch_kernel_0002.launch(
        *args,
        grid=grid,
        block=block,
        shared_mem=shared_mem,
        stream=stream,
        timeout_ms=timeout_ms,
        arch=arch,
        options=options,
    )


dispatch_kernel_0003 = get_kernel('dispatch_kernel_0003')


def launch_dispatch_kernel_0003(
    *args,
    grid: tuple[int, int, int],
    block: tuple[int, int, int] | None = None,
    shared_mem: int | None = None,
    stream=None,
    timeout_ms: float | None = None,
    arch: str | None = None,
    options: list[str] | None = None,
):
    return dispatch_kernel_0003.launch(
        *args,
        grid=grid,
        block=block,
        shared_mem=shared_mem,
        stream=stream,
        timeout_ms=timeout_ms,
        arch=arch,
        options=options,
    )


dispatch_kernel_0004 = get_kernel('dispatch_kernel_0004')


def launch_dispatch_kernel_0004(
    *args,
    grid: tuple[int, int, int],
    block: tuple[int, int, int] | None = None,
    shared_mem: int | None = None,
    stream=None,
    timeout_ms: float | None = None,
    arch: str | None = None,
    options: list[str] | None = None,
):
    return dispatch_kernel_0004.launch(
        *args,
        grid=grid,
        block=block,
        shared_mem=shared_mem,
        stream=stream,
        timeout_ms=timeout_ms,
        arch=arch,
        options=options,
    )


dispatch_kernel_0005 = get_kernel('dispatch_kernel_0005')


def launch_dispatch_kernel_0005(
    *args,
    grid: tuple[int, int, int],
    block: tuple[int, int, int] | None = None,
    shared_mem: int | None = None,
    stream=None,
    timeout_ms: float | None = None,
    arch: str | None = None,
    options: list[str] | None = None,
):
    return dispatch_kernel_0005.launch(
        *args,
        grid=grid,
        block=block,
        shared_mem=shared_mem,
        stream=stream,
        timeout_ms=timeout_ms,
        arch=arch,
        options=options,
    )


dispatch_kernel_0006 = get_kernel('dispatch_kernel_0006')


def launch_dispatch_kernel_0006(
    *args,
    grid: tuple[int, int, int],
    block: tuple[int, int, int] | None = None,
    shared_mem: int | None = None,
    stream=None,
    timeout_ms: float | None = None,
    arch: str | None = None,
    options: list[str] | None = None,
):
    return dispatch_kernel_0006.launch(
        *args,
        grid=grid,
        block=block,
        shared_mem=shared_mem,
        stream=stream,
        timeout_ms=timeout_ms,
        arch=arch,
        options=options,
    )


dispatch_kernel_0007 = get_kernel('dispatch_kernel_0007')


def launch_dispatch_kernel_0007(
    *args,
    grid: tuple[int, int, int],
    block: tuple[int, int, int] | None = None,
    shared_mem: int | None = None,
    stream=None,
    timeout_ms: float | None = None,
    arch: str | None = None,
    options: list[str] | None = None,
):
    return dispatch_kernel_0007.launch(
        *args,
        grid=grid,
        block=block,
        shared_mem=shared_mem,
        stream=stream,
        timeout_ms=timeout_ms,
        arch=arch,
        options=options,
    )


dispatch_kernel_0008 = get_kernel('dispatch_kernel_0008')


def launch_dispatch_kernel_0008(
    *args,
    grid: tuple[int, int, int],
    block: tuple[int, int, int] | None = None,
    shared_mem: int | None = None,
    stream=None,
    timeout_ms: float | None = None,
    arch: str | None = None,
    options: list[str] | None = None,
):
    return dispatch_kernel_0008.launch(
        *args,
        grid=grid,
        block=block,
        shared_mem=shared_mem,
        stream=stream,
        timeout_ms=timeout_ms,
        arch=arch,
        options=options,
    )


dispatch_kernel_0009 = get_kernel('dispatch_kernel_0009')


def launch_dispatch_kernel_0009(
    *args,
    grid: tuple[int, int, int],
    block: tuple[int, int, int] | None = None,
    shared_mem: int | None = None,
    stream=None,
    timeout_ms: float | None = None,
    arch: str | None = None,
    options: list[str] | None = None,
):
    return dispatch_kernel_0009.launch(
        *args,
        grid=grid,
        block=block,
        shared_mem=shared_mem,
        stream=stream,
        timeout_ms=timeout_ms,
        arch=arch,
        options=options,
    )


dispatch_kernel_0010 = get_kernel('dispatch_kernel_0010')


def launch_dispatch_kernel_0010(
    *args,
    grid: tuple[int, int, int],
    block: tuple[int, int, int] | None = None,
    shared_mem: int | None = None,
    stream=None,
    timeout_ms: float | None = None,
    arch: str | None = None,
    options: list[str] | None = None,
):
    return dispatch_kernel_0010.launch(
        *args,
        grid=grid,
        block=block,
        shared_mem=shared_mem,
        stream=stream,
        timeout_ms=timeout_ms,
        arch=arch,
        options=options,
    )


dispatch_kernel_0011 = get_kernel('dispatch_kernel_0011')


def launch_dispatch_kernel_0011(
    *args,
    grid: tuple[int, int, int],
    block: tuple[int, int, int] | None = None,
    shared_mem: int | None = None,
    stream=None,
    timeout_ms: float | None = None,
    arch: str | None = None,
    options: list[str] | None = None,
):
    return dispatch_kernel_0011.launch(
        *args,
        grid=grid,
        block=block,
        shared_mem=shared_mem,
        stream=stream,
        timeout_ms=timeout_ms,
        arch=arch,
        options=options,
    )


dispatch_kernel_0012 = get_kernel('dispatch_kernel_0012')


def launch_dispatch_kernel_0012(
    *args,
    grid: tuple[int, int, int],
    block: tuple[int, int, int] | None = None,
    shared_mem: int | None = None,
    stream=None,
    timeout_ms: float | None = None,
    arch: str | None = None,
    options: list[str] | None = None,
):
    return dispatch_kernel_0012.launch(
        *args,
        grid=grid,
        block=block,
        shared_mem=shared_mem,
        stream=stream,
        timeout_ms=timeout_ms,
        arch=arch,
        options=options,
    )


dispatch_kernel_0013 = get_kernel('dispatch_kernel_0013')


def launch_dispatch_kernel_0013(
    *args,
    grid: tuple[int, int, int],
    block: tuple[int, int, int] | None = None,
    shared_mem: int | None = None,
    stream=None,
    timeout_ms: float | None = None,
    arch: str | None = None,
    options: list[str] | None = None,
):
    return dispatch_kernel_0013.launch(
        *args,
        grid=grid,
        block=block,
        shared_mem=shared_mem,
        stream=stream,
        timeout_ms=timeout_ms,
        arch=arch,
        options=options,
    )


dispatch_kernel_0014 = get_kernel('dispatch_kernel_0014')


def launch_dispatch_kernel_0014(
    *args,
    grid: tuple[int, int, int],
    block: tuple[int, int, int] | None = None,
    shared_mem: int | None = None,
    stream=None,
    timeout_ms: float | None = None,
    arch: str | None = None,
    options: list[str] | None = None,
):
    return dispatch_kernel_0014.launch(
        *args,
        grid=grid,
        block=block,
        shared_mem=shared_mem,
        stream=stream,
        timeout_ms=timeout_ms,
        arch=arch,
        options=options,
    )


dispatch_kernel_0015 = get_kernel('dispatch_kernel_0015')


def launch_dispatch_kernel_0015(
    *args,
    grid: tuple[int, int, int],
    block: tuple[int, int, int] | None = None,
    shared_mem: int | None = None,
    stream=None,
    timeout_ms: float | None = None,
    arch: str | None = None,
    options: list[str] | None = None,
):
    return dispatch_kernel_0015.launch(
        *args,
        grid=grid,
        block=block,
        shared_mem=shared_mem,
        stream=stream,
        timeout_ms=timeout_ms,
        arch=arch,
        options=options,
    )


dispatch_kernel_0016 = get_kernel('dispatch_kernel_0016')


def launch_dispatch_kernel_0016(
    *args,
    grid: tuple[int, int, int],
    block: tuple[int, int, int] | None = None,
    shared_mem: int | None = None,
    stream=None,
    timeout_ms: float | None = None,
    arch: str | None = None,
    options: list[str] | None = None,
):
    return dispatch_kernel_0016.launch(
        *args,
        grid=grid,
        block=block,
        shared_mem=shared_mem,
        stream=stream,
        timeout_ms=timeout_ms,
        arch=arch,
        options=options,
    )


dispatch_kernel_0017 = get_kernel('dispatch_kernel_0017')


def launch_dispatch_kernel_0017(
    *args,
    grid: tuple[int, int, int],
    block: tuple[int, int, int] | None = None,
    shared_mem: int | None = None,
    stream=None,
    timeout_ms: float | None = None,
    arch: str | None = None,
    options: list[str] | None = None,
):
    return dispatch_kernel_0017.launch(
        *args,
        grid=grid,
        block=block,
        shared_mem=shared_mem,
        stream=stream,
        timeout_ms=timeout_ms,
        arch=arch,
        options=options,
    )


dispatch_kernel_0018 = get_kernel('dispatch_kernel_0018')


def launch_dispatch_kernel_0018(
    *args,
    grid: tuple[int, int, int],
    block: tuple[int, int, int] | None = None,
    shared_mem: int | None = None,
    stream=None,
    timeout_ms: float | None = None,
    arch: str | None = None,
    options: list[str] | None = None,
):
    return dispatch_kernel_0018.launch(
        *args,
        grid=grid,
        block=block,
        shared_mem=shared_mem,
        stream=stream,
        timeout_ms=timeout_ms,
        arch=arch,
        options=options,
    )


dispatch_kernel_0019 = get_kernel('dispatch_kernel_0019')


def launch_dispatch_kernel_0019(
    *args,
    grid: tuple[int, int, int],
    block: tuple[int, int, int] | None = None,
    shared_mem: int | None = None,
    stream=None,
    timeout_ms: float | None = None,
    arch: str | None = None,
    options: list[str] | None = None,
):
    return dispatch_kernel_0019.launch(
        *args,
        grid=grid,
        block=block,
        shared_mem=shared_mem,
        stream=stream,
        timeout_ms=timeout_ms,
        arch=arch,
        options=options,
    )


dispatch_kernel_0020 = get_kernel('dispatch_kernel_0020')


def launch_dispatch_kernel_0020(
    *args,
    grid: tuple[int, int, int],
    block: tuple[int, int, int] | None = None,
    shared_mem: int | None = None,
    stream=None,
    timeout_ms: float | None = None,
    arch: str | None = None,
    options: list[str] | None = None,
):
    return dispatch_kernel_0020.launch(
        *args,
        grid=grid,
        block=block,
        shared_mem=shared_mem,
        stream=stream,
        timeout_ms=timeout_ms,
        arch=arch,
        options=options,
    )


dispatch_kernel_0021 = get_kernel('dispatch_kernel_0021')


def launch_dispatch_kernel_0021(
    *args,
    grid: tuple[int, int, int],
    block: tuple[int, int, int] | None = None,
    shared_mem: int | None = None,
    stream=None,
    timeout_ms: float | None = None,
    arch: str | None = None,
    options: list[str] | None = None,
):
    return dispatch_kernel_0021.launch(
        *args,
        grid=grid,
        block=block,
        shared_mem=shared_mem,
        stream=stream,
        timeout_ms=timeout_ms,
        arch=arch,
        options=options,
    )


dispatch_kernel_0022 = get_kernel('dispatch_kernel_0022')


def launch_dispatch_kernel_0022(
    *args,
    grid: tuple[int, int, int],
    block: tuple[int, int, int] | None = None,
    shared_mem: int | None = None,
    stream=None,
    timeout_ms: float | None = None,
    arch: str | None = None,
    options: list[str] | None = None,
):
    return dispatch_kernel_0022.launch(
        *args,
        grid=grid,
        block=block,
        shared_mem=shared_mem,
        stream=stream,
        timeout_ms=timeout_ms,
        arch=arch,
        options=options,
    )


dispatch_kernel_0023 = get_kernel('dispatch_kernel_0023')


def launch_dispatch_kernel_0023(
    *args,
    grid: tuple[int, int, int],
    block: tuple[int, int, int] | None = None,
    shared_mem: int | None = None,
    stream=None,
    timeout_ms: float | None = None,
    arch: str | None = None,
    options: list[str] | None = None,
):
    return dispatch_kernel_0023.launch(
        *args,
        grid=grid,
        block=block,
        shared_mem=shared_mem,
        stream=stream,
        timeout_ms=timeout_ms,
        arch=arch,
        options=options,
    )


dispatch_kernel_0024 = get_kernel('dispatch_kernel_0024')


def launch_dispatch_kernel_0024(
    *args,
    grid: tuple[int, int, int],
    block: tuple[int, int, int] | None = None,
    shared_mem: int | None = None,
    stream=None,
    timeout_ms: float | None = None,
    arch: str | None = None,
    options: list[str] | None = None,
):
    return dispatch_kernel_0024.launch(
        *args,
        grid=grid,
        block=block,
        shared_mem=shared_mem,
        stream=stream,
        timeout_ms=timeout_ms,
        arch=arch,
        options=options,
    )


dispatch_kernel_0025 = get_kernel('dispatch_kernel_0025')


def launch_dispatch_kernel_0025(
    *args,
    grid: tuple[int, int, int],
    block: tuple[int, int, int] | None = None,
    shared_mem: int | None = None,
    stream=None,
    timeout_ms: float | None = None,
    arch: str | None = None,
    options: list[str] | None = None,
):
    return dispatch_kernel_0025.launch(
        *args,
        grid=grid,
        block=block,
        shared_mem=shared_mem,
        stream=stream,
        timeout_ms=timeout_ms,
        arch=arch,
        options=options,
    )


dispatch_kernel_0026 = get_kernel('dispatch_kernel_0026')


def launch_dispatch_kernel_0026(
    *args,
    grid: tuple[int, int, int],
    block: tuple[int, int, int] | None = None,
    shared_mem: int | None = None,
    stream=None,
    timeout_ms: float | None = None,
    arch: str | None = None,
    options: list[str] | None = None,
):
    return dispatch_kernel_0026.launch(
        *args,
        grid=grid,
        block=block,
        shared_mem=shared_mem,
        stream=stream,
        timeout_ms=timeout_ms,
        arch=arch,
        options=options,
    )


dispatch_kernel_0027 = get_kernel('dispatch_kernel_0027')


def launch_dispatch_kernel_0027(
    *args,
    grid: tuple[int, int, int],
    block: tuple[int, int, int] | None = None,
    shared_mem: int | None = None,
    stream=None,
    timeout_ms: float | None = None,
    arch: str | None = None,
    options: list[str] | None = None,
):
    return dispatch_kernel_0027.launch(
        *args,
        grid=grid,
        block=block,
        shared_mem=shared_mem,
        stream=stream,
        timeout_ms=timeout_ms,
        arch=arch,
        options=options,
    )


dispatch_kernel_0028 = get_kernel('dispatch_kernel_0028')


def launch_dispatch_kernel_0028(
    *args,
    grid: tuple[int, int, int],
    block: tuple[int, int, int] | None = None,
    shared_mem: int | None = None,
    stream=None,
    timeout_ms: float | None = None,
    arch: str | None = None,
    options: list[str] | None = None,
):
    return dispatch_kernel_0028.launch(
        *args,
        grid=grid,
        block=block,
        shared_mem=shared_mem,
        stream=stream,
        timeout_ms=timeout_ms,
        arch=arch,
        options=options,
    )


dispatch_kernel_0029 = get_kernel('dispatch_kernel_0029')


def launch_dispatch_kernel_0029(
    *args,
    grid: tuple[int, int, int],
    block: tuple[int, int, int] | None = None,
    shared_mem: int | None = None,
    stream=None,
    timeout_ms: float | None = None,
    arch: str | None = None,
    options: list[str] | None = None,
):
    return dispatch_kernel_0029.launch(
        *args,
        grid=grid,
        block=block,
        shared_mem=shared_mem,
        stream=stream,
        timeout_ms=timeout_ms,
        arch=arch,
        options=options,
    )


dispatch_kernel_0030 = get_kernel('dispatch_kernel_0030')


def launch_dispatch_kernel_0030(
    *args,
    grid: tuple[int, int, int],
    block: tuple[int, int, int] | None = None,
    shared_mem: int | None = None,
    stream=None,
    timeout_ms: float | None = None,
    arch: str | None = None,
    options: list[str] | None = None,
):
    return dispatch_kernel_0030.launch(
        *args,
        grid=grid,
        block=block,
        shared_mem=shared_mem,
        stream=stream,
        timeout_ms=timeout_ms,
        arch=arch,
        options=options,
    )


dispatch_kernel_0031 = get_kernel('dispatch_kernel_0031')


def launch_dispatch_kernel_0031(
    *args,
    grid: tuple[int, int, int],
    block: tuple[int, int, int] | None = None,
    shared_mem: int | None = None,
    stream=None,
    timeout_ms: float | None = None,
    arch: str | None = None,
    options: list[str] | None = None,
):
    return dispatch_kernel_0031.launch(
        *args,
        grid=grid,
        block=block,
        shared_mem=shared_mem,
        stream=stream,
        timeout_ms=timeout_ms,
        arch=arch,
        options=options,
    )


dispatch_kernel_0032 = get_kernel('dispatch_kernel_0032')


def launch_dispatch_kernel_0032(
    *args,
    grid: tuple[int, int, int],
    block: tuple[int, int, int] | None = None,
    shared_mem: int | None = None,
    stream=None,
    timeout_ms: float | None = None,
    arch: str | None = None,
    options: list[str] | None = None,
):
    return dispatch_kernel_0032.launch(
        *args,
        grid=grid,
        block=block,
        shared_mem=shared_mem,
        stream=stream,
        timeout_ms=timeout_ms,
        arch=arch,
        options=options,
    )


dispatch_kernel_0033 = get_kernel('dispatch_kernel_0033')


def launch_dispatch_kernel_0033(
    *args,
    grid: tuple[int, int, int],
    block: tuple[int, int, int] | None = None,
    shared_mem: int | None = None,
    stream=None,
    timeout_ms: float | None = None,
    arch: str | None = None,
    options: list[str] | None = None,
):
    return dispatch_kernel_0033.launch(
        *args,
        grid=grid,
        block=block,
        shared_mem=shared_mem,
        stream=stream,
        timeout_ms=timeout_ms,
        arch=arch,
        options=options,
    )


dispatch_kernel_0034 = get_kernel('dispatch_kernel_0034')


def launch_dispatch_kernel_0034(
    *args,
    grid: tuple[int, int, int],
    block: tuple[int, int, int] | None = None,
    shared_mem: int | None = None,
    stream=None,
    timeout_ms: float | None = None,
    arch: str | None = None,
    options: list[str] | None = None,
):
    return dispatch_kernel_0034.launch(
        *args,
        grid=grid,
        block=block,
        shared_mem=shared_mem,
        stream=stream,
        timeout_ms=timeout_ms,
        arch=arch,
        options=options,
    )


dispatch_kernel_0035 = get_kernel('dispatch_kernel_0035')


def launch_dispatch_kernel_0035(
    *args,
    grid: tuple[int, int, int],
    block: tuple[int, int, int] | None = None,
    shared_mem: int | None = None,
    stream=None,
    timeout_ms: float | None = None,
    arch: str | None = None,
    options: list[str] | None = None,
):
    return dispatch_kernel_0035.launch(
        *args,
        grid=grid,
        block=block,
        shared_mem=shared_mem,
        stream=stream,
        timeout_ms=timeout_ms,
        arch=arch,
        options=options,
    )


dispatch_kernel_0036 = get_kernel('dispatch_kernel_0036')


def launch_dispatch_kernel_0036(
    *args,
    grid: tuple[int, int, int],
    block: tuple[int, int, int] | None = None,
    shared_mem: int | None = None,
    stream=None,
    timeout_ms: float | None = None,
    arch: str | None = None,
    options: list[str] | None = None,
):
    return dispatch_kernel_0036.launch(
        *args,
        grid=grid,
        block=block,
        shared_mem=shared_mem,
        stream=stream,
        timeout_ms=timeout_ms,
        arch=arch,
        options=options,
    )


dispatch_kernel_0037 = get_kernel('dispatch_kernel_0037')


def launch_dispatch_kernel_0037(
    *args,
    grid: tuple[int, int, int],
    block: tuple[int, int, int] | None = None,
    shared_mem: int | None = None,
    stream=None,
    timeout_ms: float | None = None,
    arch: str | None = None,
    options: list[str] | None = None,
):
    return dispatch_kernel_0037.launch(
        *args,
        grid=grid,
        block=block,
        shared_mem=shared_mem,
        stream=stream,
        timeout_ms=timeout_ms,
        arch=arch,
        options=options,
    )


dispatch_kernel_0038 = get_kernel('dispatch_kernel_0038')


def launch_dispatch_kernel_0038(
    *args,
    grid: tuple[int, int, int],
    block: tuple[int, int, int] | None = None,
    shared_mem: int | None = None,
    stream=None,
    timeout_ms: float | None = None,
    arch: str | None = None,
    options: list[str] | None = None,
):
    return dispatch_kernel_0038.launch(
        *args,
        grid=grid,
        block=block,
        shared_mem=shared_mem,
        stream=stream,
        timeout_ms=timeout_ms,
        arch=arch,
        options=options,
    )


dispatch_kernel_0039 = get_kernel('dispatch_kernel_0039')


def launch_dispatch_kernel_0039(
    *args,
    grid: tuple[int, int, int],
    block: tuple[int, int, int] | None = None,
    shared_mem: int | None = None,
    stream=None,
    timeout_ms: float | None = None,
    arch: str | None = None,
    options: list[str] | None = None,
):
    return dispatch_kernel_0039.launch(
        *args,
        grid=grid,
        block=block,
        shared_mem=shared_mem,
        stream=stream,
        timeout_ms=timeout_ms,
        arch=arch,
        options=options,
    )


dispatch_kernel_0040 = get_kernel('dispatch_kernel_0040')


def launch_dispatch_kernel_0040(
    *args,
    grid: tuple[int, int, int],
    block: tuple[int, int, int] | None = None,
    shared_mem: int | None = None,
    stream=None,
    timeout_ms: float | None = None,
    arch: str | None = None,
    options: list[str] | None = None,
):
    return dispatch_kernel_0040.launch(
        *args,
        grid=grid,
        block=block,
        shared_mem=shared_mem,
        stream=stream,
        timeout_ms=timeout_ms,
        arch=arch,
        options=options,
    )


dispatch_kernel_0041 = get_kernel('dispatch_kernel_0041')


def launch_dispatch_kernel_0041(
    *args,
    grid: tuple[int, int, int],
    block: tuple[int, int, int] | None = None,
    shared_mem: int | None = None,
    stream=None,
    timeout_ms: float | None = None,
    arch: str | None = None,
    options: list[str] | None = None,
):
    return dispatch_kernel_0041.launch(
        *args,
        grid=grid,
        block=block,
        shared_mem=shared_mem,
        stream=stream,
        timeout_ms=timeout_ms,
        arch=arch,
        options=options,
    )


dispatch_kernel_0042 = get_kernel('dispatch_kernel_0042')


def launch_dispatch_kernel_0042(
    *args,
    grid: tuple[int, int, int],
    block: tuple[int, int, int] | None = None,
    shared_mem: int | None = None,
    stream=None,
    timeout_ms: float | None = None,
    arch: str | None = None,
    options: list[str] | None = None,
):
    return dispatch_kernel_0042.launch(
        *args,
        grid=grid,
        block=block,
        shared_mem=shared_mem,
        stream=stream,
        timeout_ms=timeout_ms,
        arch=arch,
        options=options,
    )


dispatch_kernel_0043 = get_kernel('dispatch_kernel_0043')


def launch_dispatch_kernel_0043(
    *args,
    grid: tuple[int, int, int],
    block: tuple[int, int, int] | None = None,
    shared_mem: int | None = None,
    stream=None,
    timeout_ms: float | None = None,
    arch: str | None = None,
    options: list[str] | None = None,
):
    return dispatch_kernel_0043.launch(
        *args,
        grid=grid,
        block=block,
        shared_mem=shared_mem,
        stream=stream,
        timeout_ms=timeout_ms,
        arch=arch,
        options=options,
    )


dispatch_kernel_0044 = get_kernel('dispatch_kernel_0044')


def launch_dispatch_kernel_0044(
    *args,
    grid: tuple[int, int, int],
    block: tuple[int, int, int] | None = None,
    shared_mem: int | None = None,
    stream=None,
    timeout_ms: float | None = None,
    arch: str | None = None,
    options: list[str] | None = None,
):
    return dispatch_kernel_0044.launch(
        *args,
        grid=grid,
        block=block,
        shared_mem=shared_mem,
        stream=stream,
        timeout_ms=timeout_ms,
        arch=arch,
        options=options,
    )


dispatch_kernel_0045 = get_kernel('dispatch_kernel_0045')


def launch_dispatch_kernel_0045(
    *args,
    grid: tuple[int, int, int],
    block: tuple[int, int, int] | None = None,
    shared_mem: int | None = None,
    stream=None,
    timeout_ms: float | None = None,
    arch: str | None = None,
    options: list[str] | None = None,
):
    return dispatch_kernel_0045.launch(
        *args,
        grid=grid,
        block=block,
        shared_mem=shared_mem,
        stream=stream,
        timeout_ms=timeout_ms,
        arch=arch,
        options=options,
    )


dispatch_kernel_0046 = get_kernel('dispatch_kernel_0046')


def launch_dispatch_kernel_0046(
    *args,
    grid: tuple[int, int, int],
    block: tuple[int, int, int] | None = None,
    shared_mem: int | None = None,
    stream=None,
    timeout_ms: float | None = None,
    arch: str | None = None,
    options: list[str] | None = None,
):
    return dispatch_kernel_0046.launch(
        *args,
        grid=grid,
        block=block,
        shared_mem=shared_mem,
        stream=stream,
        timeout_ms=timeout_ms,
        arch=arch,
        options=options,
    )


dispatch_kernel_0047 = get_kernel('dispatch_kernel_0047')


def launch_dispatch_kernel_0047(
    *args,
    grid: tuple[int, int, int],
    block: tuple[int, int, int] | None = None,
    shared_mem: int | None = None,
    stream=None,
    timeout_ms: float | None = None,
    arch: str | None = None,
    options: list[str] | None = None,
):
    return dispatch_kernel_0047.launch(
        *args,
        grid=grid,
        block=block,
        shared_mem=shared_mem,
        stream=stream,
        timeout_ms=timeout_ms,
        arch=arch,
        options=options,
    )


dispatch_kernel_0048 = get_kernel('dispatch_kernel_0048')


def launch_dispatch_kernel_0048(
    *args,
    grid: tuple[int, int, int],
    block: tuple[int, int, int] | None = None,
    shared_mem: int | None = None,
    stream=None,
    timeout_ms: float | None = None,
    arch: str | None = None,
    options: list[str] | None = None,
):
    return dispatch_kernel_0048.launch(
        *args,
        grid=grid,
        block=block,
        shared_mem=shared_mem,
        stream=stream,
        timeout_ms=timeout_ms,
        arch=arch,
        options=options,
    )


dispatch_kernel_0049 = get_kernel('dispatch_kernel_0049')


def launch_dispatch_kernel_0049(
    *args,
    grid: tuple[int, int, int],
    block: tuple[int, int, int] | None = None,
    shared_mem: int | None = None,
    stream=None,
    timeout_ms: float | None = None,
    arch: str | None = None,
    options: list[str] | None = None,
):
    return dispatch_kernel_0049.launch(
        *args,
        grid=grid,
        block=block,
        shared_mem=shared_mem,
        stream=stream,
        timeout_ms=timeout_ms,
        arch=arch,
        options=options,
    )


dispatch_kernel_0050 = get_kernel('dispatch_kernel_0050')


def launch_dispatch_kernel_0050(
    *args,
    grid: tuple[int, int, int],
    block: tuple[int, int, int] | None = None,
    shared_mem: int | None = None,
    stream=None,
    timeout_ms: float | None = None,
    arch: str | None = None,
    options: list[str] | None = None,
):
    return dispatch_kernel_0050.launch(
        *args,
        grid=grid,
        block=block,
        shared_mem=shared_mem,
        stream=stream,
        timeout_ms=timeout_ms,
        arch=arch,
        options=options,
    )


dispatch_kernel_0051 = get_kernel('dispatch_kernel_0051')


def launch_dispatch_kernel_0051(
    *args,
    grid: tuple[int, int, int],
    block: tuple[int, int, int] | None = None,
    shared_mem: int | None = None,
    stream=None,
    timeout_ms: float | None = None,
    arch: str | None = None,
    options: list[str] | None = None,
):
    return dispatch_kernel_0051.launch(
        *args,
        grid=grid,
        block=block,
        shared_mem=shared_mem,
        stream=stream,
        timeout_ms=timeout_ms,
        arch=arch,
        options=options,
    )


dispatch_kernel_0052 = get_kernel('dispatch_kernel_0052')


def launch_dispatch_kernel_0052(
    *args,
    grid: tuple[int, int, int],
    block: tuple[int, int, int] | None = None,
    shared_mem: int | None = None,
    stream=None,
    timeout_ms: float | None = None,
    arch: str | None = None,
    options: list[str] | None = None,
):
    return dispatch_kernel_0052.launch(
        *args,
        grid=grid,
        block=block,
        shared_mem=shared_mem,
        stream=stream,
        timeout_ms=timeout_ms,
        arch=arch,
        options=options,
    )


dispatch_kernel_0053 = get_kernel('dispatch_kernel_0053')


def launch_dispatch_kernel_0053(
    *args,
    grid: tuple[int, int, int],
    block: tuple[int, int, int] | None = None,
    shared_mem: int | None = None,
    stream=None,
    timeout_ms: float | None = None,
    arch: str | None = None,
    options: list[str] | None = None,
):
    return dispatch_kernel_0053.launch(
        *args,
        grid=grid,
        block=block,
        shared_mem=shared_mem,
        stream=stream,
        timeout_ms=timeout_ms,
        arch=arch,
        options=options,
    )


dispatch_kernel_0054 = get_kernel('dispatch_kernel_0054')


def launch_dispatch_kernel_0054(
    *args,
    grid: tuple[int, int, int],
    block: tuple[int, int, int] | None = None,
    shared_mem: int | None = None,
    stream=None,
    timeout_ms: float | None = None,
    arch: str | None = None,
    options: list[str] | None = None,
):
    return dispatch_kernel_0054.launch(
        *args,
        grid=grid,
        block=block,
        shared_mem=shared_mem,
        stream=stream,
        timeout_ms=timeout_ms,
        arch=arch,
        options=options,
    )


dispatch_kernel_0055 = get_kernel('dispatch_kernel_0055')


def launch_dispatch_kernel_0055(
    *args,
    grid: tuple[int, int, int],
    block: tuple[int, int, int] | None = None,
    shared_mem: int | None = None,
    stream=None,
    timeout_ms: float | None = None,
    arch: str | None = None,
    options: list[str] | None = None,
):
    return dispatch_kernel_0055.launch(
        *args,
        grid=grid,
        block=block,
        shared_mem=shared_mem,
        stream=stream,
        timeout_ms=timeout_ms,
        arch=arch,
        options=options,
    )


dispatch_kernel_0056 = get_kernel('dispatch_kernel_0056')


def launch_dispatch_kernel_0056(
    *args,
    grid: tuple[int, int, int],
    block: tuple[int, int, int] | None = None,
    shared_mem: int | None = None,
    stream=None,
    timeout_ms: float | None = None,
    arch: str | None = None,
    options: list[str] | None = None,
):
    return dispatch_kernel_0056.launch(
        *args,
        grid=grid,
        block=block,
        shared_mem=shared_mem,
        stream=stream,
        timeout_ms=timeout_ms,
        arch=arch,
        options=options,
    )


dispatch_kernel_0057 = get_kernel('dispatch_kernel_0057')


def launch_dispatch_kernel_0057(
    *args,
    grid: tuple[int, int, int],
    block: tuple[int, int, int] | None = None,
    shared_mem: int | None = None,
    stream=None,
    timeout_ms: float | None = None,
    arch: str | None = None,
    options: list[str] | None = None,
):
    return dispatch_kernel_0057.launch(
        *args,
        grid=grid,
        block=block,
        shared_mem=shared_mem,
        stream=stream,
        timeout_ms=timeout_ms,
        arch=arch,
        options=options,
    )


dispatch_kernel_0058 = get_kernel('dispatch_kernel_0058')


def launch_dispatch_kernel_0058(
    *args,
    grid: tuple[int, int, int],
    block: tuple[int, int, int] | None = None,
    shared_mem: int | None = None,
    stream=None,
    timeout_ms: float | None = None,
    arch: str | None = None,
    options: list[str] | None = None,
):
    return dispatch_kernel_0058.launch(
        *args,
        grid=grid,
        block=block,
        shared_mem=shared_mem,
        stream=stream,
        timeout_ms=timeout_ms,
        arch=arch,
        options=options,
    )


dispatch_kernel_0059 = get_kernel('dispatch_kernel_0059')


def launch_dispatch_kernel_0059(
    *args,
    grid: tuple[int, int, int],
    block: tuple[int, int, int] | None = None,
    shared_mem: int | None = None,
    stream=None,
    timeout_ms: float | None = None,
    arch: str | None = None,
    options: list[str] | None = None,
):
    return dispatch_kernel_0059.launch(
        *args,
        grid=grid,
        block=block,
        shared_mem=shared_mem,
        stream=stream,
        timeout_ms=timeout_ms,
        arch=arch,
        options=options,
    )


dispatch_kernel_0060 = get_kernel('dispatch_kernel_0060')


def launch_dispatch_kernel_0060(
    *args,
    grid: tuple[int, int, int],
    block: tuple[int, int, int] | None = None,
    shared_mem: int | None = None,
    stream=None,
    timeout_ms: float | None = None,
    arch: str | None = None,
    options: list[str] | None = None,
):
    return dispatch_kernel_0060.launch(
        *args,
        grid=grid,
        block=block,
        shared_mem=shared_mem,
        stream=stream,
        timeout_ms=timeout_ms,
        arch=arch,
        options=options,
    )


dispatch_kernel_0061 = get_kernel('dispatch_kernel_0061')


def launch_dispatch_kernel_0061(
    *args,
    grid: tuple[int, int, int],
    block: tuple[int, int, int] | None = None,
    shared_mem: int | None = None,
    stream=None,
    timeout_ms: float | None = None,
    arch: str | None = None,
    options: list[str] | None = None,
):
    return dispatch_kernel_0061.launch(
        *args,
        grid=grid,
        block=block,
        shared_mem=shared_mem,
        stream=stream,
        timeout_ms=timeout_ms,
        arch=arch,
        options=options,
    )


dispatch_kernel_0062 = get_kernel('dispatch_kernel_0062')


def launch_dispatch_kernel_0062(
    *args,
    grid: tuple[int, int, int],
    block: tuple[int, int, int] | None = None,
    shared_mem: int | None = None,
    stream=None,
    timeout_ms: float | None = None,
    arch: str | None = None,
    options: list[str] | None = None,
):
    return dispatch_kernel_0062.launch(
        *args,
        grid=grid,
        block=block,
        shared_mem=shared_mem,
        stream=stream,
        timeout_ms=timeout_ms,
        arch=arch,
        options=options,
    )


dispatch_kernel_0063 = get_kernel('dispatch_kernel_0063')


def launch_dispatch_kernel_0063(
    *args,
    grid: tuple[int, int, int],
    block: tuple[int, int, int] | None = None,
    shared_mem: int | None = None,
    stream=None,
    timeout_ms: float | None = None,
    arch: str | None = None,
    options: list[str] | None = None,
):
    return dispatch_kernel_0063.launch(
        *args,
        grid=grid,
        block=block,
        shared_mem=shared_mem,
        stream=stream,
        timeout_ms=timeout_ms,
        arch=arch,
        options=options,
    )


dispatch_kernel_0064 = get_kernel('dispatch_kernel_0064')


def launch_dispatch_kernel_0064(
    *args,
    grid: tuple[int, int, int],
    block: tuple[int, int, int] | None = None,
    shared_mem: int | None = None,
    stream=None,
    timeout_ms: float | None = None,
    arch: str | None = None,
    options: list[str] | None = None,
):
    return dispatch_kernel_0064.launch(
        *args,
        grid=grid,
        block=block,
        shared_mem=shared_mem,
        stream=stream,
        timeout_ms=timeout_ms,
        arch=arch,
        options=options,
    )


dispatch_kernel_0065 = get_kernel('dispatch_kernel_0065')


def launch_dispatch_kernel_0065(
    *args,
    grid: tuple[int, int, int],
    block: tuple[int, int, int] | None = None,
    shared_mem: int | None = None,
    stream=None,
    timeout_ms: float | None = None,
    arch: str | None = None,
    options: list[str] | None = None,
):
    return dispatch_kernel_0065.launch(
        *args,
        grid=grid,
        block=block,
        shared_mem=shared_mem,
        stream=stream,
        timeout_ms=timeout_ms,
        arch=arch,
        options=options,
    )


dispatch_kernel_0066 = get_kernel('dispatch_kernel_0066')


def launch_dispatch_kernel_0066(
    *args,
    grid: tuple[int, int, int],
    block: tuple[int, int, int] | None = None,
    shared_mem: int | None = None,
    stream=None,
    timeout_ms: float | None = None,
    arch: str | None = None,
    options: list[str] | None = None,
):
    return dispatch_kernel_0066.launch(
        *args,
        grid=grid,
        block=block,
        shared_mem=shared_mem,
        stream=stream,
        timeout_ms=timeout_ms,
        arch=arch,
        options=options,
    )


dispatch_kernel_0067 = get_kernel('dispatch_kernel_0067')


def launch_dispatch_kernel_0067(
    *args,
    grid: tuple[int, int, int],
    block: tuple[int, int, int] | None = None,
    shared_mem: int | None = None,
    stream=None,
    timeout_ms: float | None = None,
    arch: str | None = None,
    options: list[str] | None = None,
):
    return dispatch_kernel_0067.launch(
        *args,
        grid=grid,
        block=block,
        shared_mem=shared_mem,
        stream=stream,
        timeout_ms=timeout_ms,
        arch=arch,
        options=options,
    )


dispatch_kernel_0068 = get_kernel('dispatch_kernel_0068')


def launch_dispatch_kernel_0068(
    *args,
    grid: tuple[int, int, int],
    block: tuple[int, int, int] | None = None,
    shared_mem: int | None = None,
    stream=None,
    timeout_ms: float | None = None,
    arch: str | None = None,
    options: list[str] | None = None,
):
    return dispatch_kernel_0068.launch(
        *args,
        grid=grid,
        block=block,
        shared_mem=shared_mem,
        stream=stream,
        timeout_ms=timeout_ms,
        arch=arch,
        options=options,
    )


dispatch_kernel_0069 = get_kernel('dispatch_kernel_0069')


def launch_dispatch_kernel_0069(
    *args,
    grid: tuple[int, int, int],
    block: tuple[int, int, int] | None = None,
    shared_mem: int | None = None,
    stream=None,
    timeout_ms: float | None = None,
    arch: str | None = None,
    options: list[str] | None = None,
):
    return dispatch_kernel_0069.launch(
        *args,
        grid=grid,
        block=block,
        shared_mem=shared_mem,
        stream=stream,
        timeout_ms=timeout_ms,
        arch=arch,
        options=options,
    )


dispatch_kernel_0070 = get_kernel('dispatch_kernel_0070')


def launch_dispatch_kernel_0070(
    *args,
    grid: tuple[int, int, int],
    block: tuple[int, int, int] | None = None,
    shared_mem: int | None = None,
    stream=None,
    timeout_ms: float | None = None,
    arch: str | None = None,
    options: list[str] | None = None,
):
    return dispatch_kernel_0070.launch(
        *args,
        grid=grid,
        block=block,
        shared_mem=shared_mem,
        stream=stream,
        timeout_ms=timeout_ms,
        arch=arch,
        options=options,
    )


dispatch_kernel_0071 = get_kernel('dispatch_kernel_0071')


def launch_dispatch_kernel_0071(
    *args,
    grid: tuple[int, int, int],
    block: tuple[int, int, int] | None = None,
    shared_mem: int | None = None,
    stream=None,
    timeout_ms: float | None = None,
    arch: str | None = None,
    options: list[str] | None = None,
):
    return dispatch_kernel_0071.launch(
        *args,
        grid=grid,
        block=block,
        shared_mem=shared_mem,
        stream=stream,
        timeout_ms=timeout_ms,
        arch=arch,
        options=options,
    )


dispatch_kernel_0072 = get_kernel('dispatch_kernel_0072')


def launch_dispatch_kernel_0072(
    *args,
    grid: tuple[int, int, int],
    block: tuple[int, int, int] | None = None,
    shared_mem: int | None = None,
    stream=None,
    timeout_ms: float | None = None,
    arch: str | None = None,
    options: list[str] | None = None,
):
    return dispatch_kernel_0072.launch(
        *args,
        grid=grid,
        block=block,
        shared_mem=shared_mem,
        stream=stream,
        timeout_ms=timeout_ms,
        arch=arch,
        options=options,
    )


dispatch_kernel_0073 = get_kernel('dispatch_kernel_0073')


def launch_dispatch_kernel_0073(
    *args,
    grid: tuple[int, int, int],
    block: tuple[int, int, int] | None = None,
    shared_mem: int | None = None,
    stream=None,
    timeout_ms: float | None = None,
    arch: str | None = None,
    options: list[str] | None = None,
):
    return dispatch_kernel_0073.launch(
        *args,
        grid=grid,
        block=block,
        shared_mem=shared_mem,
        stream=stream,
        timeout_ms=timeout_ms,
        arch=arch,
        options=options,
    )


dispatch_kernel_0074 = get_kernel('dispatch_kernel_0074')


def launch_dispatch_kernel_0074(
    *args,
    grid: tuple[int, int, int],
    block: tuple[int, int, int] | None = None,
    shared_mem: int | None = None,
    stream=None,
    timeout_ms: float | None = None,
    arch: str | None = None,
    options: list[str] | None = None,
):
    return dispatch_kernel_0074.launch(
        *args,
        grid=grid,
        block=block,
        shared_mem=shared_mem,
        stream=stream,
        timeout_ms=timeout_ms,
        arch=arch,
        options=options,
    )


dispatch_kernel_0075 = get_kernel('dispatch_kernel_0075')


def launch_dispatch_kernel_0075(
    *args,
    grid: tuple[int, int, int],
    block: tuple[int, int, int] | None = None,
    shared_mem: int | None = None,
    stream=None,
    timeout_ms: float | None = None,
    arch: str | None = None,
    options: list[str] | None = None,
):
    return dispatch_kernel_0075.launch(
        *args,
        grid=grid,
        block=block,
        shared_mem=shared_mem,
        stream=stream,
        timeout_ms=timeout_ms,
        arch=arch,
        options=options,
    )


dispatch_kernel_0076 = get_kernel('dispatch_kernel_0076')


def launch_dispatch_kernel_0076(
    *args,
    grid: tuple[int, int, int],
    block: tuple[int, int, int] | None = None,
    shared_mem: int | None = None,
    stream=None,
    timeout_ms: float | None = None,
    arch: str | None = None,
    options: list[str] | None = None,
):
    return dispatch_kernel_0076.launch(
        *args,
        grid=grid,
        block=block,
        shared_mem=shared_mem,
        stream=stream,
        timeout_ms=timeout_ms,
        arch=arch,
        options=options,
    )


dispatch_kernel_0077 = get_kernel('dispatch_kernel_0077')


def launch_dispatch_kernel_0077(
    *args,
    grid: tuple[int, int, int],
    block: tuple[int, int, int] | None = None,
    shared_mem: int | None = None,
    stream=None,
    timeout_ms: float | None = None,
    arch: str | None = None,
    options: list[str] | None = None,
):
    return dispatch_kernel_0077.launch(
        *args,
        grid=grid,
        block=block,
        shared_mem=shared_mem,
        stream=stream,
        timeout_ms=timeout_ms,
        arch=arch,
        options=options,
    )


dispatch_kernel_0078 = get_kernel('dispatch_kernel_0078')


def launch_dispatch_kernel_0078(
    *args,
    grid: tuple[int, int, int],
    block: tuple[int, int, int] | None = None,
    shared_mem: int | None = None,
    stream=None,
    timeout_ms: float | None = None,
    arch: str | None = None,
    options: list[str] | None = None,
):
    return dispatch_kernel_0078.launch(
        *args,
        grid=grid,
        block=block,
        shared_mem=shared_mem,
        stream=stream,
        timeout_ms=timeout_ms,
        arch=arch,
        options=options,
    )


dispatch_kernel_0079 = get_kernel('dispatch_kernel_0079')


def launch_dispatch_kernel_0079(
    *args,
    grid: tuple[int, int, int],
    block: tuple[int, int, int] | None = None,
    shared_mem: int | None = None,
    stream=None,
    timeout_ms: float | None = None,
    arch: str | None = None,
    options: list[str] | None = None,
):
    return dispatch_kernel_0079.launch(
        *args,
        grid=grid,
        block=block,
        shared_mem=shared_mem,
        stream=stream,
        timeout_ms=timeout_ms,
        arch=arch,
        options=options,
    )


dispatch_kernel_0080 = get_kernel('dispatch_kernel_0080')


def launch_dispatch_kernel_0080(
    *args,
    grid: tuple[int, int, int],
    block: tuple[int, int, int] | None = None,
    shared_mem: int | None = None,
    stream=None,
    timeout_ms: float | None = None,
    arch: str | None = None,
    options: list[str] | None = None,
):
    return dispatch_kernel_0080.launch(
        *args,
        grid=grid,
        block=block,
        shared_mem=shared_mem,
        stream=stream,
        timeout_ms=timeout_ms,
        arch=arch,
        options=options,
    )


dispatch_kernel_0081 = get_kernel('dispatch_kernel_0081')


def launch_dispatch_kernel_0081(
    *args,
    grid: tuple[int, int, int],
    block: tuple[int, int, int] | None = None,
    shared_mem: int | None = None,
    stream=None,
    timeout_ms: float | None = None,
    arch: str | None = None,
    options: list[str] | None = None,
):
    return dispatch_kernel_0081.launch(
        *args,
        grid=grid,
        block=block,
        shared_mem=shared_mem,
        stream=stream,
        timeout_ms=timeout_ms,
        arch=arch,
        options=options,
    )


dispatch_kernel_0082 = get_kernel('dispatch_kernel_0082')


def launch_dispatch_kernel_0082(
    *args,
    grid: tuple[int, int, int],
    block: tuple[int, int, int] | None = None,
    shared_mem: int | None = None,
    stream=None,
    timeout_ms: float | None = None,
    arch: str | None = None,
    options: list[str] | None = None,
):
    return dispatch_kernel_0082.launch(
        *args,
        grid=grid,
        block=block,
        shared_mem=shared_mem,
        stream=stream,
        timeout_ms=timeout_ms,
        arch=arch,
        options=options,
    )


dispatch_kernel_0083 = get_kernel('dispatch_kernel_0083')


def launch_dispatch_kernel_0083(
    *args,
    grid: tuple[int, int, int],
    block: tuple[int, int, int] | None = None,
    shared_mem: int | None = None,
    stream=None,
    timeout_ms: float | None = None,
    arch: str | None = None,
    options: list[str] | None = None,
):
    return dispatch_kernel_0083.launch(
        *args,
        grid=grid,
        block=block,
        shared_mem=shared_mem,
        stream=stream,
        timeout_ms=timeout_ms,
        arch=arch,
        options=options,
    )


dispatch_kernel_0084 = get_kernel('dispatch_kernel_0084')


def launch_dispatch_kernel_0084(
    *args,
    grid: tuple[int, int, int],
    block: tuple[int, int, int] | None = None,
    shared_mem: int | None = None,
    stream=None,
    timeout_ms: float | None = None,
    arch: str | None = None,
    options: list[str] | None = None,
):
    return dispatch_kernel_0084.launch(
        *args,
        grid=grid,
        block=block,
        shared_mem=shared_mem,
        stream=stream,
        timeout_ms=timeout_ms,
        arch=arch,
        options=options,
    )


dispatch_kernel_0085 = get_kernel('dispatch_kernel_0085')


def launch_dispatch_kernel_0085(
    *args,
    grid: tuple[int, int, int],
    block: tuple[int, int, int] | None = None,
    shared_mem: int | None = None,
    stream=None,
    timeout_ms: float | None = None,
    arch: str | None = None,
    options: list[str] | None = None,
):
    return dispatch_kernel_0085.launch(
        *args,
        grid=grid,
        block=block,
        shared_mem=shared_mem,
        stream=stream,
        timeout_ms=timeout_ms,
        arch=arch,
        options=options,
    )


dispatch_kernel_0086 = get_kernel('dispatch_kernel_0086')


def launch_dispatch_kernel_0086(
    *args,
    grid: tuple[int, int, int],
    block: tuple[int, int, int] | None = None,
    shared_mem: int | None = None,
    stream=None,
    timeout_ms: float | None = None,
    arch: str | None = None,
    options: list[str] | None = None,
):
    return dispatch_kernel_0086.launch(
        *args,
        grid=grid,
        block=block,
        shared_mem=shared_mem,
        stream=stream,
        timeout_ms=timeout_ms,
        arch=arch,
        options=options,
    )


dispatch_kernel_0087 = get_kernel('dispatch_kernel_0087')


def launch_dispatch_kernel_0087(
    *args,
    grid: tuple[int, int, int],
    block: tuple[int, int, int] | None = None,
    shared_mem: int | None = None,
    stream=None,
    timeout_ms: float | None = None,
    arch: str | None = None,
    options: list[str] | None = None,
):
    return dispatch_kernel_0087.launch(
        *args,
        grid=grid,
        block=block,
        shared_mem=shared_mem,
        stream=stream,
        timeout_ms=timeout_ms,
        arch=arch,
        options=options,
    )


dispatch_kernel_0088 = get_kernel('dispatch_kernel_0088')


def launch_dispatch_kernel_0088(
    *args,
    grid: tuple[int, int, int],
    block: tuple[int, int, int] | None = None,
    shared_mem: int | None = None,
    stream=None,
    timeout_ms: float | None = None,
    arch: str | None = None,
    options: list[str] | None = None,
):
    return dispatch_kernel_0088.launch(
        *args,
        grid=grid,
        block=block,
        shared_mem=shared_mem,
        stream=stream,
        timeout_ms=timeout_ms,
        arch=arch,
        options=options,
    )


dispatch_kernel_0089 = get_kernel('dispatch_kernel_0089')


def launch_dispatch_kernel_0089(
    *args,
    grid: tuple[int, int, int],
    block: tuple[int, int, int] | None = None,
    shared_mem: int | None = None,
    stream=None,
    timeout_ms: float | None = None,
    arch: str | None = None,
    options: list[str] | None = None,
):
    return dispatch_kernel_0089.launch(
        *args,
        grid=grid,
        block=block,
        shared_mem=shared_mem,
        stream=stream,
        timeout_ms=timeout_ms,
        arch=arch,
        options=options,
    )


dispatch_kernel_0090 = get_kernel('dispatch_kernel_0090')


def launch_dispatch_kernel_0090(
    *args,
    grid: tuple[int, int, int],
    block: tuple[int, int, int] | None = None,
    shared_mem: int | None = None,
    stream=None,
    timeout_ms: float | None = None,
    arch: str | None = None,
    options: list[str] | None = None,
):
    return dispatch_kernel_0090.launch(
        *args,
        grid=grid,
        block=block,
        shared_mem=shared_mem,
        stream=stream,
        timeout_ms=timeout_ms,
        arch=arch,
        options=options,
    )


dispatch_kernel_0091 = get_kernel('dispatch_kernel_0091')


def launch_dispatch_kernel_0091(
    *args,
    grid: tuple[int, int, int],
    block: tuple[int, int, int] | None = None,
    shared_mem: int | None = None,
    stream=None,
    timeout_ms: float | None = None,
    arch: str | None = None,
    options: list[str] | None = None,
):
    return dispatch_kernel_0091.launch(
        *args,
        grid=grid,
        block=block,
        shared_mem=shared_mem,
        stream=stream,
        timeout_ms=timeout_ms,
        arch=arch,
        options=options,
    )


dispatch_kernel_0092 = get_kernel('dispatch_kernel_0092')


def launch_dispatch_kernel_0092(
    *args,
    grid: tuple[int, int, int],
    block: tuple[int, int, int] | None = None,
    shared_mem: int | None = None,
    stream=None,
    timeout_ms: float | None = None,
    arch: str | None = None,
    options: list[str] | None = None,
):
    return dispatch_kernel_0092.launch(
        *args,
        grid=grid,
        block=block,
        shared_mem=shared_mem,
        stream=stream,
        timeout_ms=timeout_ms,
        arch=arch,
        options=options,
    )


dispatch_kernel_0093 = get_kernel('dispatch_kernel_0093')


def launch_dispatch_kernel_0093(
    *args,
    grid: tuple[int, int, int],
    block: tuple[int, int, int] | None = None,
    shared_mem: int | None = None,
    stream=None,
    timeout_ms: float | None = None,
    arch: str | None = None,
    options: list[str] | None = None,
):
    return dispatch_kernel_0093.launch(
        *args,
        grid=grid,
        block=block,
        shared_mem=shared_mem,
        stream=stream,
        timeout_ms=timeout_ms,
        arch=arch,
        options=options,
    )


dispatch_kernel_0094 = get_kernel('dispatch_kernel_0094')


def launch_dispatch_kernel_0094(
    *args,
    grid: tuple[int, int, int],
    block: tuple[int, int, int] | None = None,
    shared_mem: int | None = None,
    stream=None,
    timeout_ms: float | None = None,
    arch: str | None = None,
    options: list[str] | None = None,
):
    return dispatch_kernel_0094.launch(
        *args,
        grid=grid,
        block=block,
        shared_mem=shared_mem,
        stream=stream,
        timeout_ms=timeout_ms,
        arch=arch,
        options=options,
    )


dispatch_kernel_0095 = get_kernel('dispatch_kernel_0095')


def launch_dispatch_kernel_0095(
    *args,
    grid: tuple[int, int, int],
    block: tuple[int, int, int] | None = None,
    shared_mem: int | None = None,
    stream=None,
    timeout_ms: float | None = None,
    arch: str | None = None,
    options: list[str] | None = None,
):
    return dispatch_kernel_0095.launch(
        *args,
        grid=grid,
        block=block,
        shared_mem=shared_mem,
        stream=stream,
        timeout_ms=timeout_ms,
        arch=arch,
        options=options,
    )


dispatch_kernel_0096 = get_kernel('dispatch_kernel_0096')


def launch_dispatch_kernel_0096(
    *args,
    grid: tuple[int, int, int],
    block: tuple[int, int, int] | None = None,
    shared_mem: int | None = None,
    stream=None,
    timeout_ms: float | None = None,
    arch: str | None = None,
    options: list[str] | None = None,
):
    return dispatch_kernel_0096.launch(
        *args,
        grid=grid,
        block=block,
        shared_mem=shared_mem,
        stream=stream,
        timeout_ms=timeout_ms,
        arch=arch,
        options=options,
    )


dispatch_kernel_0097 = get_kernel('dispatch_kernel_0097')


def launch_dispatch_kernel_0097(
    *args,
    grid: tuple[int, int, int],
    block: tuple[int, int, int] | None = None,
    shared_mem: int | None = None,
    stream=None,
    timeout_ms: float | None = None,
    arch: str | None = None,
    options: list[str] | None = None,
):
    return dispatch_kernel_0097.launch(
        *args,
        grid=grid,
        block=block,
        shared_mem=shared_mem,
        stream=stream,
        timeout_ms=timeout_ms,
        arch=arch,
        options=options,
    )


dispatch_kernel_0098 = get_kernel('dispatch_kernel_0098')


def launch_dispatch_kernel_0098(
    *args,
    grid: tuple[int, int, int],
    block: tuple[int, int, int] | None = None,
    shared_mem: int | None = None,
    stream=None,
    timeout_ms: float | None = None,
    arch: str | None = None,
    options: list[str] | None = None,
):
    return dispatch_kernel_0098.launch(
        *args,
        grid=grid,
        block=block,
        shared_mem=shared_mem,
        stream=stream,
        timeout_ms=timeout_ms,
        arch=arch,
        options=options,
    )


dispatch_kernel_0099 = get_kernel('dispatch_kernel_0099')


def launch_dispatch_kernel_0099(
    *args,
    grid: tuple[int, int, int],
    block: tuple[int, int, int] | None = None,
    shared_mem: int | None = None,
    stream=None,
    timeout_ms: float | None = None,
    arch: str | None = None,
    options: list[str] | None = None,
):
    return dispatch_kernel_0099.launch(
        *args,
        grid=grid,
        block=block,
        shared_mem=shared_mem,
        stream=stream,
        timeout_ms=timeout_ms,
        arch=arch,
        options=options,
    )


dispatch_kernel_0100 = get_kernel('dispatch_kernel_0100')


def launch_dispatch_kernel_0100(
    *args,
    grid: tuple[int, int, int],
    block: tuple[int, int, int] | None = None,
    shared_mem: int | None = None,
    stream=None,
    timeout_ms: float | None = None,
    arch: str | None = None,
    options: list[str] | None = None,
):
    return dispatch_kernel_0100.launch(
        *args,
        grid=grid,
        block=block,
        shared_mem=shared_mem,
        stream=stream,
        timeout_ms=timeout_ms,
        arch=arch,
        options=options,
    )


dispatch_kernel_0101 = get_kernel('dispatch_kernel_0101')


def launch_dispatch_kernel_0101(
    *args,
    grid: tuple[int, int, int],
    block: tuple[int, int, int] | None = None,
    shared_mem: int | None = None,
    stream=None,
    timeout_ms: float | None = None,
    arch: str | None = None,
    options: list[str] | None = None,
):
    return dispatch_kernel_0101.launch(
        *args,
        grid=grid,
        block=block,
        shared_mem=shared_mem,
        stream=stream,
        timeout_ms=timeout_ms,
        arch=arch,
        options=options,
    )


dispatch_kernel_0102 = get_kernel('dispatch_kernel_0102')


def launch_dispatch_kernel_0102(
    *args,
    grid: tuple[int, int, int],
    block: tuple[int, int, int] | None = None,
    shared_mem: int | None = None,
    stream=None,
    timeout_ms: float | None = None,
    arch: str | None = None,
    options: list[str] | None = None,
):
    return dispatch_kernel_0102.launch(
        *args,
        grid=grid,
        block=block,
        shared_mem=shared_mem,
        stream=stream,
        timeout_ms=timeout_ms,
        arch=arch,
        options=options,
    )


dispatch_kernel_0103 = get_kernel('dispatch_kernel_0103')


def launch_dispatch_kernel_0103(
    *args,
    grid: tuple[int, int, int],
    block: tuple[int, int, int] | None = None,
    shared_mem: int | None = None,
    stream=None,
    timeout_ms: float | None = None,
    arch: str | None = None,
    options: list[str] | None = None,
):
    return dispatch_kernel_0103.launch(
        *args,
        grid=grid,
        block=block,
        shared_mem=shared_mem,
        stream=stream,
        timeout_ms=timeout_ms,
        arch=arch,
        options=options,
    )


dispatch_kernel_0104 = get_kernel('dispatch_kernel_0104')


def launch_dispatch_kernel_0104(
    *args,
    grid: tuple[int, int, int],
    block: tuple[int, int, int] | None = None,
    shared_mem: int | None = None,
    stream=None,
    timeout_ms: float | None = None,
    arch: str | None = None,
    options: list[str] | None = None,
):
    return dispatch_kernel_0104.launch(
        *args,
        grid=grid,
        block=block,
        shared_mem=shared_mem,
        stream=stream,
        timeout_ms=timeout_ms,
        arch=arch,
        options=options,
    )


dispatch_kernel_0105 = get_kernel('dispatch_kernel_0105')


def launch_dispatch_kernel_0105(
    *args,
    grid: tuple[int, int, int],
    block: tuple[int, int, int] | None = None,
    shared_mem: int | None = None,
    stream=None,
    timeout_ms: float | None = None,
    arch: str | None = None,
    options: list[str] | None = None,
):
    return dispatch_kernel_0105.launch(
        *args,
        grid=grid,
        block=block,
        shared_mem=shared_mem,
        stream=stream,
        timeout_ms=timeout_ms,
        arch=arch,
        options=options,
    )


dispatch_kernel_0106 = get_kernel('dispatch_kernel_0106')


def launch_dispatch_kernel_0106(
    *args,
    grid: tuple[int, int, int],
    block: tuple[int, int, int] | None = None,
    shared_mem: int | None = None,
    stream=None,
    timeout_ms: float | None = None,
    arch: str | None = None,
    options: list[str] | None = None,
):
    return dispatch_kernel_0106.launch(
        *args,
        grid=grid,
        block=block,
        shared_mem=shared_mem,
        stream=stream,
        timeout_ms=timeout_ms,
        arch=arch,
        options=options,
    )


dispatch_kernel_0107 = get_kernel('dispatch_kernel_0107')


def launch_dispatch_kernel_0107(
    *args,
    grid: tuple[int, int, int],
    block: tuple[int, int, int] | None = None,
    shared_mem: int | None = None,
    stream=None,
    timeout_ms: float | None = None,
    arch: str | None = None,
    options: list[str] | None = None,
):
    return dispatch_kernel_0107.launch(
        *args,
        grid=grid,
        block=block,
        shared_mem=shared_mem,
        stream=stream,
        timeout_ms=timeout_ms,
        arch=arch,
        options=options,
    )


dispatch_kernel_0108 = get_kernel('dispatch_kernel_0108')


def launch_dispatch_kernel_0108(
    *args,
    grid: tuple[int, int, int],
    block: tuple[int, int, int] | None = None,
    shared_mem: int | None = None,
    stream=None,
    timeout_ms: float | None = None,
    arch: str | None = None,
    options: list[str] | None = None,
):
    return dispatch_kernel_0108.launch(
        *args,
        grid=grid,
        block=block,
        shared_mem=shared_mem,
        stream=stream,
        timeout_ms=timeout_ms,
        arch=arch,
        options=options,
    )


dispatch_kernel_0109 = get_kernel('dispatch_kernel_0109')


def launch_dispatch_kernel_0109(
    *args,
    grid: tuple[int, int, int],
    block: tuple[int, int, int] | None = None,
    shared_mem: int | None = None,
    stream=None,
    timeout_ms: float | None = None,
    arch: str | None = None,
    options: list[str] | None = None,
):
    return dispatch_kernel_0109.launch(
        *args,
        grid=grid,
        block=block,
        shared_mem=shared_mem,
        stream=stream,
        timeout_ms=timeout_ms,
        arch=arch,
        options=options,
    )


dispatch_kernel_0110 = get_kernel('dispatch_kernel_0110')


def launch_dispatch_kernel_0110(
    *args,
    grid: tuple[int, int, int],
    block: tuple[int, int, int] | None = None,
    shared_mem: int | None = None,
    stream=None,
    timeout_ms: float | None = None,
    arch: str | None = None,
    options: list[str] | None = None,
):
    return dispatch_kernel_0110.launch(
        *args,
        grid=grid,
        block=block,
        shared_mem=shared_mem,
        stream=stream,
        timeout_ms=timeout_ms,
        arch=arch,
        options=options,
    )


dispatch_kernel_0111 = get_kernel('dispatch_kernel_0111')


def launch_dispatch_kernel_0111(
    *args,
    grid: tuple[int, int, int],
    block: tuple[int, int, int] | None = None,
    shared_mem: int | None = None,
    stream=None,
    timeout_ms: float | None = None,
    arch: str | None = None,
    options: list[str] | None = None,
):
    return dispatch_kernel_0111.launch(
        *args,
        grid=grid,
        block=block,
        shared_mem=shared_mem,
        stream=stream,
        timeout_ms=timeout_ms,
        arch=arch,
        options=options,
    )


dispatch_kernel_0112 = get_kernel('dispatch_kernel_0112')


def launch_dispatch_kernel_0112(
    *args,
    grid: tuple[int, int, int],
    block: tuple[int, int, int] | None = None,
    shared_mem: int | None = None,
    stream=None,
    timeout_ms: float | None = None,
    arch: str | None = None,
    options: list[str] | None = None,
):
    return dispatch_kernel_0112.launch(
        *args,
        grid=grid,
        block=block,
        shared_mem=shared_mem,
        stream=stream,
        timeout_ms=timeout_ms,
        arch=arch,
        options=options,
    )


dispatch_kernel_0113 = get_kernel('dispatch_kernel_0113')


def launch_dispatch_kernel_0113(
    *args,
    grid: tuple[int, int, int],
    block: tuple[int, int, int] | None = None,
    shared_mem: int | None = None,
    stream=None,
    timeout_ms: float | None = None,
    arch: str | None = None,
    options: list[str] | None = None,
):
    return dispatch_kernel_0113.launch(
        *args,
        grid=grid,
        block=block,
        shared_mem=shared_mem,
        stream=stream,
        timeout_ms=timeout_ms,
        arch=arch,
        options=options,
    )


dispatch_kernel_0114 = get_kernel('dispatch_kernel_0114')


def launch_dispatch_kernel_0114(
    *args,
    grid: tuple[int, int, int],
    block: tuple[int, int, int] | None = None,
    shared_mem: int | None = None,
    stream=None,
    timeout_ms: float | None = None,
    arch: str | None = None,
    options: list[str] | None = None,
):
    return dispatch_kernel_0114.launch(
        *args,
        grid=grid,
        block=block,
        shared_mem=shared_mem,
        stream=stream,
        timeout_ms=timeout_ms,
        arch=arch,
        options=options,
    )


dispatch_kernel_0115 = get_kernel('dispatch_kernel_0115')


def launch_dispatch_kernel_0115(
    *args,
    grid: tuple[int, int, int],
    block: tuple[int, int, int] | None = None,
    shared_mem: int | None = None,
    stream=None,
    timeout_ms: float | None = None,
    arch: str | None = None,
    options: list[str] | None = None,
):
    return dispatch_kernel_0115.launch(
        *args,
        grid=grid,
        block=block,
        shared_mem=shared_mem,
        stream=stream,
        timeout_ms=timeout_ms,
        arch=arch,
        options=options,
    )


dispatch_kernel_0116 = get_kernel('dispatch_kernel_0116')


def launch_dispatch_kernel_0116(
    *args,
    grid: tuple[int, int, int],
    block: tuple[int, int, int] | None = None,
    shared_mem: int | None = None,
    stream=None,
    timeout_ms: float | None = None,
    arch: str | None = None,
    options: list[str] | None = None,
):
    return dispatch_kernel_0116.launch(
        *args,
        grid=grid,
        block=block,
        shared_mem=shared_mem,
        stream=stream,
        timeout_ms=timeout_ms,
        arch=arch,
        options=options,
    )


dispatch_kernel_0117 = get_kernel('dispatch_kernel_0117')


def launch_dispatch_kernel_0117(
    *args,
    grid: tuple[int, int, int],
    block: tuple[int, int, int] | None = None,
    shared_mem: int | None = None,
    stream=None,
    timeout_ms: float | None = None,
    arch: str | None = None,
    options: list[str] | None = None,
):
    return dispatch_kernel_0117.launch(
        *args,
        grid=grid,
        block=block,
        shared_mem=shared_mem,
        stream=stream,
        timeout_ms=timeout_ms,
        arch=arch,
        options=options,
    )


dispatch_kernel_0118 = get_kernel('dispatch_kernel_0118')


def launch_dispatch_kernel_0118(
    *args,
    grid: tuple[int, int, int],
    block: tuple[int, int, int] | None = None,
    shared_mem: int | None = None,
    stream=None,
    timeout_ms: float | None = None,
    arch: str | None = None,
    options: list[str] | None = None,
):
    return dispatch_kernel_0118.launch(
        *args,
        grid=grid,
        block=block,
        shared_mem=shared_mem,
        stream=stream,
        timeout_ms=timeout_ms,
        arch=arch,
        options=options,
    )


dispatch_kernel_0119 = get_kernel('dispatch_kernel_0119')


def launch_dispatch_kernel_0119(
    *args,
    grid: tuple[int, int, int],
    block: tuple[int, int, int] | None = None,
    shared_mem: int | None = None,
    stream=None,
    timeout_ms: float | None = None,
    arch: str | None = None,
    options: list[str] | None = None,
):
    return dispatch_kernel_0119.launch(
        *args,
        grid=grid,
        block=block,
        shared_mem=shared_mem,
        stream=stream,
        timeout_ms=timeout_ms,
        arch=arch,
        options=options,
    )


dispatch_kernel_0120 = get_kernel('dispatch_kernel_0120')


def launch_dispatch_kernel_0120(
    *args,
    grid: tuple[int, int, int],
    block: tuple[int, int, int] | None = None,
    shared_mem: int | None = None,
    stream=None,
    timeout_ms: float | None = None,
    arch: str | None = None,
    options: list[str] | None = None,
):
    return dispatch_kernel_0120.launch(
        *args,
        grid=grid,
        block=block,
        shared_mem=shared_mem,
        stream=stream,
        timeout_ms=timeout_ms,
        arch=arch,
        options=options,
    )


dispatch_kernel_0121 = get_kernel('dispatch_kernel_0121')


def launch_dispatch_kernel_0121(
    *args,
    grid: tuple[int, int, int],
    block: tuple[int, int, int] | None = None,
    shared_mem: int | None = None,
    stream=None,
    timeout_ms: float | None = None,
    arch: str | None = None,
    options: list[str] | None = None,
):
    return dispatch_kernel_0121.launch(
        *args,
        grid=grid,
        block=block,
        shared_mem=shared_mem,
        stream=stream,
        timeout_ms=timeout_ms,
        arch=arch,
        options=options,
    )


dispatch_kernel_0122 = get_kernel('dispatch_kernel_0122')


def launch_dispatch_kernel_0122(
    *args,
    grid: tuple[int, int, int],
    block: tuple[int, int, int] | None = None,
    shared_mem: int | None = None,
    stream=None,
    timeout_ms: float | None = None,
    arch: str | None = None,
    options: list[str] | None = None,
):
    return dispatch_kernel_0122.launch(
        *args,
        grid=grid,
        block=block,
        shared_mem=shared_mem,
        stream=stream,
        timeout_ms=timeout_ms,
        arch=arch,
        options=options,
    )


dispatch_kernel_0123 = get_kernel('dispatch_kernel_0123')


def launch_dispatch_kernel_0123(
    *args,
    grid: tuple[int, int, int],
    block: tuple[int, int, int] | None = None,
    shared_mem: int | None = None,
    stream=None,
    timeout_ms: float | None = None,
    arch: str | None = None,
    options: list[str] | None = None,
):
    return dispatch_kernel_0123.launch(
        *args,
        grid=grid,
        block=block,
        shared_mem=shared_mem,
        stream=stream,
        timeout_ms=timeout_ms,
        arch=arch,
        options=options,
    )


dispatch_kernel_0124 = get_kernel('dispatch_kernel_0124')


def launch_dispatch_kernel_0124(
    *args,
    grid: tuple[int, int, int],
    block: tuple[int, int, int] | None = None,
    shared_mem: int | None = None,
    stream=None,
    timeout_ms: float | None = None,
    arch: str | None = None,
    options: list[str] | None = None,
):
    return dispatch_kernel_0124.launch(
        *args,
        grid=grid,
        block=block,
        shared_mem=shared_mem,
        stream=stream,
        timeout_ms=timeout_ms,
        arch=arch,
        options=options,
    )


dispatch_kernel_0125 = get_kernel('dispatch_kernel_0125')


def launch_dispatch_kernel_0125(
    *args,
    grid: tuple[int, int, int],
    block: tuple[int, int, int] | None = None,
    shared_mem: int | None = None,
    stream=None,
    timeout_ms: float | None = None,
    arch: str | None = None,
    options: list[str] | None = None,
):
    return dispatch_kernel_0125.launch(
        *args,
        grid=grid,
        block=block,
        shared_mem=shared_mem,
        stream=stream,
        timeout_ms=timeout_ms,
        arch=arch,
        options=options,
    )


dispatch_kernel_0126 = get_kernel('dispatch_kernel_0126')


def launch_dispatch_kernel_0126(
    *args,
    grid: tuple[int, int, int],
    block: tuple[int, int, int] | None = None,
    shared_mem: int | None = None,
    stream=None,
    timeout_ms: float | None = None,
    arch: str | None = None,
    options: list[str] | None = None,
):
    return dispatch_kernel_0126.launch(
        *args,
        grid=grid,
        block=block,
        shared_mem=shared_mem,
        stream=stream,
        timeout_ms=timeout_ms,
        arch=arch,
        options=options,
    )


dispatch_kernel_0127 = get_kernel('dispatch_kernel_0127')


def launch_dispatch_kernel_0127(
    *args,
    grid: tuple[int, int, int],
    block: tuple[int, int, int] | None = None,
    shared_mem: int | None = None,
    stream=None,
    timeout_ms: float | None = None,
    arch: str | None = None,
    options: list[str] | None = None,
):
    return dispatch_kernel_0127.launch(
        *args,
        grid=grid,
        block=block,
        shared_mem=shared_mem,
        stream=stream,
        timeout_ms=timeout_ms,
        arch=arch,
        options=options,
    )


dispatch_kernel_0128 = get_kernel('dispatch_kernel_0128')


def launch_dispatch_kernel_0128(
    *args,
    grid: tuple[int, int, int],
    block: tuple[int, int, int] | None = None,
    shared_mem: int | None = None,
    stream=None,
    timeout_ms: float | None = None,
    arch: str | None = None,
    options: list[str] | None = None,
):
    return dispatch_kernel_0128.launch(
        *args,
        grid=grid,
        block=block,
        shared_mem=shared_mem,
        stream=stream,
        timeout_ms=timeout_ms,
        arch=arch,
        options=options,
    )


dispatch_kernel_0129 = get_kernel('dispatch_kernel_0129')


def launch_dispatch_kernel_0129(
    *args,
    grid: tuple[int, int, int],
    block: tuple[int, int, int] | None = None,
    shared_mem: int | None = None,
    stream=None,
    timeout_ms: float | None = None,
    arch: str | None = None,
    options: list[str] | None = None,
):
    return dispatch_kernel_0129.launch(
        *args,
        grid=grid,
        block=block,
        shared_mem=shared_mem,
        stream=stream,
        timeout_ms=timeout_ms,
        arch=arch,
        options=options,
    )


dispatch_kernel_0130 = get_kernel('dispatch_kernel_0130')


def launch_dispatch_kernel_0130(
    *args,
    grid: tuple[int, int, int],
    block: tuple[int, int, int] | None = None,
    shared_mem: int | None = None,
    stream=None,
    timeout_ms: float | None = None,
    arch: str | None = None,
    options: list[str] | None = None,
):
    return dispatch_kernel_0130.launch(
        *args,
        grid=grid,
        block=block,
        shared_mem=shared_mem,
        stream=stream,
        timeout_ms=timeout_ms,
        arch=arch,
        options=options,
    )


dispatch_kernel_0131 = get_kernel('dispatch_kernel_0131')


def launch_dispatch_kernel_0131(
    *args,
    grid: tuple[int, int, int],
    block: tuple[int, int, int] | None = None,
    shared_mem: int | None = None,
    stream=None,
    timeout_ms: float | None = None,
    arch: str | None = None,
    options: list[str] | None = None,
):
    return dispatch_kernel_0131.launch(
        *args,
        grid=grid,
        block=block,
        shared_mem=shared_mem,
        stream=stream,
        timeout_ms=timeout_ms,
        arch=arch,
        options=options,
    )


dispatch_kernel_0132 = get_kernel('dispatch_kernel_0132')


def launch_dispatch_kernel_0132(
    *args,
    grid: tuple[int, int, int],
    block: tuple[int, int, int] | None = None,
    shared_mem: int | None = None,
    stream=None,
    timeout_ms: float | None = None,
    arch: str | None = None,
    options: list[str] | None = None,
):
    return dispatch_kernel_0132.launch(
        *args,
        grid=grid,
        block=block,
        shared_mem=shared_mem,
        stream=stream,
        timeout_ms=timeout_ms,
        arch=arch,
        options=options,
    )


dispatch_kernel_0133 = get_kernel('dispatch_kernel_0133')


def launch_dispatch_kernel_0133(
    *args,
    grid: tuple[int, int, int],
    block: tuple[int, int, int] | None = None,
    shared_mem: int | None = None,
    stream=None,
    timeout_ms: float | None = None,
    arch: str | None = None,
    options: list[str] | None = None,
):
    return dispatch_kernel_0133.launch(
        *args,
        grid=grid,
        block=block,
        shared_mem=shared_mem,
        stream=stream,
        timeout_ms=timeout_ms,
        arch=arch,
        options=options,
    )


dispatch_kernel_0134 = get_kernel('dispatch_kernel_0134')


def launch_dispatch_kernel_0134(
    *args,
    grid: tuple[int, int, int],
    block: tuple[int, int, int] | None = None,
    shared_mem: int | None = None,
    stream=None,
    timeout_ms: float | None = None,
    arch: str | None = None,
    options: list[str] | None = None,
):
    return dispatch_kernel_0134.launch(
        *args,
        grid=grid,
        block=block,
        shared_mem=shared_mem,
        stream=stream,
        timeout_ms=timeout_ms,
        arch=arch,
        options=options,
    )


dispatch_kernel_0135 = get_kernel('dispatch_kernel_0135')


def launch_dispatch_kernel_0135(
    *args,
    grid: tuple[int, int, int],
    block: tuple[int, int, int] | None = None,
    shared_mem: int | None = None,
    stream=None,
    timeout_ms: float | None = None,
    arch: str | None = None,
    options: list[str] | None = None,
):
    return dispatch_kernel_0135.launch(
        *args,
        grid=grid,
        block=block,
        shared_mem=shared_mem,
        stream=stream,
        timeout_ms=timeout_ms,
        arch=arch,
        options=options,
    )


dispatch_kernel_0136 = get_kernel('dispatch_kernel_0136')


def launch_dispatch_kernel_0136(
    *args,
    grid: tuple[int, int, int],
    block: tuple[int, int, int] | None = None,
    shared_mem: int | None = None,
    stream=None,
    timeout_ms: float | None = None,
    arch: str | None = None,
    options: list[str] | None = None,
):
    return dispatch_kernel_0136.launch(
        *args,
        grid=grid,
        block=block,
        shared_mem=shared_mem,
        stream=stream,
        timeout_ms=timeout_ms,
        arch=arch,
        options=options,
    )


dispatch_kernel_0137 = get_kernel('dispatch_kernel_0137')


def launch_dispatch_kernel_0137(
    *args,
    grid: tuple[int, int, int],
    block: tuple[int, int, int] | None = None,
    shared_mem: int | None = None,
    stream=None,
    timeout_ms: float | None = None,
    arch: str | None = None,
    options: list[str] | None = None,
):
    return dispatch_kernel_0137.launch(
        *args,
        grid=grid,
        block=block,
        shared_mem=shared_mem,
        stream=stream,
        timeout_ms=timeout_ms,
        arch=arch,
        options=options,
    )


dispatch_kernel_0138 = get_kernel('dispatch_kernel_0138')


def launch_dispatch_kernel_0138(
    *args,
    grid: tuple[int, int, int],
    block: tuple[int, int, int] | None = None,
    shared_mem: int | None = None,
    stream=None,
    timeout_ms: float | None = None,
    arch: str | None = None,
    options: list[str] | None = None,
):
    return dispatch_kernel_0138.launch(
        *args,
        grid=grid,
        block=block,
        shared_mem=shared_mem,
        stream=stream,
        timeout_ms=timeout_ms,
        arch=arch,
        options=options,
    )


dispatch_kernel_0139 = get_kernel('dispatch_kernel_0139')


def launch_dispatch_kernel_0139(
    *args,
    grid: tuple[int, int, int],
    block: tuple[int, int, int] | None = None,
    shared_mem: int | None = None,
    stream=None,
    timeout_ms: float | None = None,
    arch: str | None = None,
    options: list[str] | None = None,
):
    return dispatch_kernel_0139.launch(
        *args,
        grid=grid,
        block=block,
        shared_mem=shared_mem,
        stream=stream,
        timeout_ms=timeout_ms,
        arch=arch,
        options=options,
    )


dispatch_kernel_0140 = get_kernel('dispatch_kernel_0140')


def launch_dispatch_kernel_0140(
    *args,
    grid: tuple[int, int, int],
    block: tuple[int, int, int] | None = None,
    shared_mem: int | None = None,
    stream=None,
    timeout_ms: float | None = None,
    arch: str | None = None,
    options: list[str] | None = None,
):
    return dispatch_kernel_0140.launch(
        *args,
        grid=grid,
        block=block,
        shared_mem=shared_mem,
        stream=stream,
        timeout_ms=timeout_ms,
        arch=arch,
        options=options,
    )


dispatch_kernel_0141 = get_kernel('dispatch_kernel_0141')


def launch_dispatch_kernel_0141(
    *args,
    grid: tuple[int, int, int],
    block: tuple[int, int, int] | None = None,
    shared_mem: int | None = None,
    stream=None,
    timeout_ms: float | None = None,
    arch: str | None = None,
    options: list[str] | None = None,
):
    return dispatch_kernel_0141.launch(
        *args,
        grid=grid,
        block=block,
        shared_mem=shared_mem,
        stream=stream,
        timeout_ms=timeout_ms,
        arch=arch,
        options=options,
    )


dispatch_kernel_0142 = get_kernel('dispatch_kernel_0142')


def launch_dispatch_kernel_0142(
    *args,
    grid: tuple[int, int, int],
    block: tuple[int, int, int] | None = None,
    shared_mem: int | None = None,
    stream=None,
    timeout_ms: float | None = None,
    arch: str | None = None,
    options: list[str] | None = None,
):
    return dispatch_kernel_0142.launch(
        *args,
        grid=grid,
        block=block,
        shared_mem=shared_mem,
        stream=stream,
        timeout_ms=timeout_ms,
        arch=arch,
        options=options,
    )


dispatch_kernel_0143 = get_kernel('dispatch_kernel_0143')


def launch_dispatch_kernel_0143(
    *args,
    grid: tuple[int, int, int],
    block: tuple[int, int, int] | None = None,
    shared_mem: int | None = None,
    stream=None,
    timeout_ms: float | None = None,
    arch: str | None = None,
    options: list[str] | None = None,
):
    return dispatch_kernel_0143.launch(
        *args,
        grid=grid,
        block=block,
        shared_mem=shared_mem,
        stream=stream,
        timeout_ms=timeout_ms,
        arch=arch,
        options=options,
    )


dispatch_kernel_0144 = get_kernel('dispatch_kernel_0144')


def launch_dispatch_kernel_0144(
    *args,
    grid: tuple[int, int, int],
    block: tuple[int, int, int] | None = None,
    shared_mem: int | None = None,
    stream=None,
    timeout_ms: float | None = None,
    arch: str | None = None,
    options: list[str] | None = None,
):
    return dispatch_kernel_0144.launch(
        *args,
        grid=grid,
        block=block,
        shared_mem=shared_mem,
        stream=stream,
        timeout_ms=timeout_ms,
        arch=arch,
        options=options,
    )


dispatch_kernel_0145 = get_kernel('dispatch_kernel_0145')


def launch_dispatch_kernel_0145(
    *args,
    grid: tuple[int, int, int],
    block: tuple[int, int, int] | None = None,
    shared_mem: int | None = None,
    stream=None,
    timeout_ms: float | None = None,
    arch: str | None = None,
    options: list[str] | None = None,
):
    return dispatch_kernel_0145.launch(
        *args,
        grid=grid,
        block=block,
        shared_mem=shared_mem,
        stream=stream,
        timeout_ms=timeout_ms,
        arch=arch,
        options=options,
    )


dispatch_kernel_0146 = get_kernel('dispatch_kernel_0146')


def launch_dispatch_kernel_0146(
    *args,
    grid: tuple[int, int, int],
    block: tuple[int, int, int] | None = None,
    shared_mem: int | None = None,
    stream=None,
    timeout_ms: float | None = None,
    arch: str | None = None,
    options: list[str] | None = None,
):
    return dispatch_kernel_0146.launch(
        *args,
        grid=grid,
        block=block,
        shared_mem=shared_mem,
        stream=stream,
        timeout_ms=timeout_ms,
        arch=arch,
        options=options,
    )


dispatch_kernel_0147 = get_kernel('dispatch_kernel_0147')


def launch_dispatch_kernel_0147(
    *args,
    grid: tuple[int, int, int],
    block: tuple[int, int, int] | None = None,
    shared_mem: int | None = None,
    stream=None,
    timeout_ms: float | None = None,
    arch: str | None = None,
    options: list[str] | None = None,
):
    return dispatch_kernel_0147.launch(
        *args,
        grid=grid,
        block=block,
        shared_mem=shared_mem,
        stream=stream,
        timeout_ms=timeout_ms,
        arch=arch,
        options=options,
    )


dispatch_kernel_0148 = get_kernel('dispatch_kernel_0148')


def launch_dispatch_kernel_0148(
    *args,
    grid: tuple[int, int, int],
    block: tuple[int, int, int] | None = None,
    shared_mem: int | None = None,
    stream=None,
    timeout_ms: float | None = None,
    arch: str | None = None,
    options: list[str] | None = None,
):
    return dispatch_kernel_0148.launch(
        *args,
        grid=grid,
        block=block,
        shared_mem=shared_mem,
        stream=stream,
        timeout_ms=timeout_ms,
        arch=arch,
        options=options,
    )


dispatch_kernel_0149 = get_kernel('dispatch_kernel_0149')


def launch_dispatch_kernel_0149(
    *args,
    grid: tuple[int, int, int],
    block: tuple[int, int, int] | None = None,
    shared_mem: int | None = None,
    stream=None,
    timeout_ms: float | None = None,
    arch: str | None = None,
    options: list[str] | None = None,
):
    return dispatch_kernel_0149.launch(
        *args,
        grid=grid,
        block=block,
        shared_mem=shared_mem,
        stream=stream,
        timeout_ms=timeout_ms,
        arch=arch,
        options=options,
    )


dispatch_kernel_0150 = get_kernel('dispatch_kernel_0150')


def launch_dispatch_kernel_0150(
    *args,
    grid: tuple[int, int, int],
    block: tuple[int, int, int] | None = None,
    shared_mem: int | None = None,
    stream=None,
    timeout_ms: float | None = None,
    arch: str | None = None,
    options: list[str] | None = None,
):
    return dispatch_kernel_0150.launch(
        *args,
        grid=grid,
        block=block,
        shared_mem=shared_mem,
        stream=stream,
        timeout_ms=timeout_ms,
        arch=arch,
        options=options,
    )


dispatch_kernel_0151 = get_kernel('dispatch_kernel_0151')


def launch_dispatch_kernel_0151(
    *args,
    grid: tuple[int, int, int],
    block: tuple[int, int, int] | None = None,
    shared_mem: int | None = None,
    stream=None,
    timeout_ms: float | None = None,
    arch: str | None = None,
    options: list[str] | None = None,
):
    return dispatch_kernel_0151.launch(
        *args,
        grid=grid,
        block=block,
        shared_mem=shared_mem,
        stream=stream,
        timeout_ms=timeout_ms,
        arch=arch,
        options=options,
    )


dispatch_kernel_0152 = get_kernel('dispatch_kernel_0152')


def launch_dispatch_kernel_0152(
    *args,
    grid: tuple[int, int, int],
    block: tuple[int, int, int] | None = None,
    shared_mem: int | None = None,
    stream=None,
    timeout_ms: float | None = None,
    arch: str | None = None,
    options: list[str] | None = None,
):
    return dispatch_kernel_0152.launch(
        *args,
        grid=grid,
        block=block,
        shared_mem=shared_mem,
        stream=stream,
        timeout_ms=timeout_ms,
        arch=arch,
        options=options,
    )


dispatch_kernel_0153 = get_kernel('dispatch_kernel_0153')


def launch_dispatch_kernel_0153(
    *args,
    grid: tuple[int, int, int],
    block: tuple[int, int, int] | None = None,
    shared_mem: int | None = None,
    stream=None,
    timeout_ms: float | None = None,
    arch: str | None = None,
    options: list[str] | None = None,
):
    return dispatch_kernel_0153.launch(
        *args,
        grid=grid,
        block=block,
        shared_mem=shared_mem,
        stream=stream,
        timeout_ms=timeout_ms,
        arch=arch,
        options=options,
    )


dispatch_kernel_0154 = get_kernel('dispatch_kernel_0154')


def launch_dispatch_kernel_0154(
    *args,
    grid: tuple[int, int, int],
    block: tuple[int, int, int] | None = None,
    shared_mem: int | None = None,
    stream=None,
    timeout_ms: float | None = None,
    arch: str | None = None,
    options: list[str] | None = None,
):
    return dispatch_kernel_0154.launch(
        *args,
        grid=grid,
        block=block,
        shared_mem=shared_mem,
        stream=stream,
        timeout_ms=timeout_ms,
        arch=arch,
        options=options,
    )


dispatch_kernel_0155 = get_kernel('dispatch_kernel_0155')


def launch_dispatch_kernel_0155(
    *args,
    grid: tuple[int, int, int],
    block: tuple[int, int, int] | None = None,
    shared_mem: int | None = None,
    stream=None,
    timeout_ms: float | None = None,
    arch: str | None = None,
    options: list[str] | None = None,
):
    return dispatch_kernel_0155.launch(
        *args,
        grid=grid,
        block=block,
        shared_mem=shared_mem,
        stream=stream,
        timeout_ms=timeout_ms,
        arch=arch,
        options=options,
    )


dispatch_kernel_0156 = get_kernel('dispatch_kernel_0156')


def launch_dispatch_kernel_0156(
    *args,
    grid: tuple[int, int, int],
    block: tuple[int, int, int] | None = None,
    shared_mem: int | None = None,
    stream=None,
    timeout_ms: float | None = None,
    arch: str | None = None,
    options: list[str] | None = None,
):
    return dispatch_kernel_0156.launch(
        *args,
        grid=grid,
        block=block,
        shared_mem=shared_mem,
        stream=stream,
        timeout_ms=timeout_ms,
        arch=arch,
        options=options,
    )


dispatch_kernel_0157 = get_kernel('dispatch_kernel_0157')


def launch_dispatch_kernel_0157(
    *args,
    grid: tuple[int, int, int],
    block: tuple[int, int, int] | None = None,
    shared_mem: int | None = None,
    stream=None,
    timeout_ms: float | None = None,
    arch: str | None = None,
    options: list[str] | None = None,
):
    return dispatch_kernel_0157.launch(
        *args,
        grid=grid,
        block=block,
        shared_mem=shared_mem,
        stream=stream,
        timeout_ms=timeout_ms,
        arch=arch,
        options=options,
    )


dispatch_kernel_0158 = get_kernel('dispatch_kernel_0158')


def launch_dispatch_kernel_0158(
    *args,
    grid: tuple[int, int, int],
    block: tuple[int, int, int] | None = None,
    shared_mem: int | None = None,
    stream=None,
    timeout_ms: float | None = None,
    arch: str | None = None,
    options: list[str] | None = None,
):
    return dispatch_kernel_0158.launch(
        *args,
        grid=grid,
        block=block,
        shared_mem=shared_mem,
        stream=stream,
        timeout_ms=timeout_ms,
        arch=arch,
        options=options,
    )


dispatch_kernel_0159 = get_kernel('dispatch_kernel_0159')


def launch_dispatch_kernel_0159(
    *args,
    grid: tuple[int, int, int],
    block: tuple[int, int, int] | None = None,
    shared_mem: int | None = None,
    stream=None,
    timeout_ms: float | None = None,
    arch: str | None = None,
    options: list[str] | None = None,
):
    return dispatch_kernel_0159.launch(
        *args,
        grid=grid,
        block=block,
        shared_mem=shared_mem,
        stream=stream,
        timeout_ms=timeout_ms,
        arch=arch,
        options=options,
    )


dispatch_kernel_0160 = get_kernel('dispatch_kernel_0160')


def launch_dispatch_kernel_0160(
    *args,
    grid: tuple[int, int, int],
    block: tuple[int, int, int] | None = None,
    shared_mem: int | None = None,
    stream=None,
    timeout_ms: float | None = None,
    arch: str | None = None,
    options: list[str] | None = None,
):
    return dispatch_kernel_0160.launch(
        *args,
        grid=grid,
        block=block,
        shared_mem=shared_mem,
        stream=stream,
        timeout_ms=timeout_ms,
        arch=arch,
        options=options,
    )


dispatch_kernel_0161 = get_kernel('dispatch_kernel_0161')


def launch_dispatch_kernel_0161(
    *args,
    grid: tuple[int, int, int],
    block: tuple[int, int, int] | None = None,
    shared_mem: int | None = None,
    stream=None,
    timeout_ms: float | None = None,
    arch: str | None = None,
    options: list[str] | None = None,
):
    return dispatch_kernel_0161.launch(
        *args,
        grid=grid,
        block=block,
        shared_mem=shared_mem,
        stream=stream,
        timeout_ms=timeout_ms,
        arch=arch,
        options=options,
    )


dispatch_kernel_0162 = get_kernel('dispatch_kernel_0162')


def launch_dispatch_kernel_0162(
    *args,
    grid: tuple[int, int, int],
    block: tuple[int, int, int] | None = None,
    shared_mem: int | None = None,
    stream=None,
    timeout_ms: float | None = None,
    arch: str | None = None,
    options: list[str] | None = None,
):
    return dispatch_kernel_0162.launch(
        *args,
        grid=grid,
        block=block,
        shared_mem=shared_mem,
        stream=stream,
        timeout_ms=timeout_ms,
        arch=arch,
        options=options,
    )


dispatch_kernel_0163 = get_kernel('dispatch_kernel_0163')


def launch_dispatch_kernel_0163(
    *args,
    grid: tuple[int, int, int],
    block: tuple[int, int, int] | None = None,
    shared_mem: int | None = None,
    stream=None,
    timeout_ms: float | None = None,
    arch: str | None = None,
    options: list[str] | None = None,
):
    return dispatch_kernel_0163.launch(
        *args,
        grid=grid,
        block=block,
        shared_mem=shared_mem,
        stream=stream,
        timeout_ms=timeout_ms,
        arch=arch,
        options=options,
    )


dispatch_kernel_0164 = get_kernel('dispatch_kernel_0164')


def launch_dispatch_kernel_0164(
    *args,
    grid: tuple[int, int, int],
    block: tuple[int, int, int] | None = None,
    shared_mem: int | None = None,
    stream=None,
    timeout_ms: float | None = None,
    arch: str | None = None,
    options: list[str] | None = None,
):
    return dispatch_kernel_0164.launch(
        *args,
        grid=grid,
        block=block,
        shared_mem=shared_mem,
        stream=stream,
        timeout_ms=timeout_ms,
        arch=arch,
        options=options,
    )


dispatch_kernel_0165 = get_kernel('dispatch_kernel_0165')


def launch_dispatch_kernel_0165(
    *args,
    grid: tuple[int, int, int],
    block: tuple[int, int, int] | None = None,
    shared_mem: int | None = None,
    stream=None,
    timeout_ms: float | None = None,
    arch: str | None = None,
    options: list[str] | None = None,
):
    return dispatch_kernel_0165.launch(
        *args,
        grid=grid,
        block=block,
        shared_mem=shared_mem,
        stream=stream,
        timeout_ms=timeout_ms,
        arch=arch,
        options=options,
    )


dispatch_kernel_0166 = get_kernel('dispatch_kernel_0166')


def launch_dispatch_kernel_0166(
    *args,
    grid: tuple[int, int, int],
    block: tuple[int, int, int] | None = None,
    shared_mem: int | None = None,
    stream=None,
    timeout_ms: float | None = None,
    arch: str | None = None,
    options: list[str] | None = None,
):
    return dispatch_kernel_0166.launch(
        *args,
        grid=grid,
        block=block,
        shared_mem=shared_mem,
        stream=stream,
        timeout_ms=timeout_ms,
        arch=arch,
        options=options,
    )


dispatch_kernel_0167 = get_kernel('dispatch_kernel_0167')


def launch_dispatch_kernel_0167(
    *args,
    grid: tuple[int, int, int],
    block: tuple[int, int, int] | None = None,
    shared_mem: int | None = None,
    stream=None,
    timeout_ms: float | None = None,
    arch: str | None = None,
    options: list[str] | None = None,
):
    return dispatch_kernel_0167.launch(
        *args,
        grid=grid,
        block=block,
        shared_mem=shared_mem,
        stream=stream,
        timeout_ms=timeout_ms,
        arch=arch,
        options=options,
    )


dispatch_kernel_0168 = get_kernel('dispatch_kernel_0168')


def launch_dispatch_kernel_0168(
    *args,
    grid: tuple[int, int, int],
    block: tuple[int, int, int] | None = None,
    shared_mem: int | None = None,
    stream=None,
    timeout_ms: float | None = None,
    arch: str | None = None,
    options: list[str] | None = None,
):
    return dispatch_kernel_0168.launch(
        *args,
        grid=grid,
        block=block,
        shared_mem=shared_mem,
        stream=stream,
        timeout_ms=timeout_ms,
        arch=arch,
        options=options,
    )


dispatch_kernel_0169 = get_kernel('dispatch_kernel_0169')


def launch_dispatch_kernel_0169(
    *args,
    grid: tuple[int, int, int],
    block: tuple[int, int, int] | None = None,
    shared_mem: int | None = None,
    stream=None,
    timeout_ms: float | None = None,
    arch: str | None = None,
    options: list[str] | None = None,
):
    return dispatch_kernel_0169.launch(
        *args,
        grid=grid,
        block=block,
        shared_mem=shared_mem,
        stream=stream,
        timeout_ms=timeout_ms,
        arch=arch,
        options=options,
    )


dispatch_kernel_0170 = get_kernel('dispatch_kernel_0170')


def launch_dispatch_kernel_0170(
    *args,
    grid: tuple[int, int, int],
    block: tuple[int, int, int] | None = None,
    shared_mem: int | None = None,
    stream=None,
    timeout_ms: float | None = None,
    arch: str | None = None,
    options: list[str] | None = None,
):
    return dispatch_kernel_0170.launch(
        *args,
        grid=grid,
        block=block,
        shared_mem=shared_mem,
        stream=stream,
        timeout_ms=timeout_ms,
        arch=arch,
        options=options,
    )


dispatch_kernel_0171 = get_kernel('dispatch_kernel_0171')


def launch_dispatch_kernel_0171(
    *args,
    grid: tuple[int, int, int],
    block: tuple[int, int, int] | None = None,
    shared_mem: int | None = None,
    stream=None,
    timeout_ms: float | None = None,
    arch: str | None = None,
    options: list[str] | None = None,
):
    return dispatch_kernel_0171.launch(
        *args,
        grid=grid,
        block=block,
        shared_mem=shared_mem,
        stream=stream,
        timeout_ms=timeout_ms,
        arch=arch,
        options=options,
    )


dispatch_kernel_0172 = get_kernel('dispatch_kernel_0172')


def launch_dispatch_kernel_0172(
    *args,
    grid: tuple[int, int, int],
    block: tuple[int, int, int] | None = None,
    shared_mem: int | None = None,
    stream=None,
    timeout_ms: float | None = None,
    arch: str | None = None,
    options: list[str] | None = None,
):
    return dispatch_kernel_0172.launch(
        *args,
        grid=grid,
        block=block,
        shared_mem=shared_mem,
        stream=stream,
        timeout_ms=timeout_ms,
        arch=arch,
        options=options,
    )


dispatch_kernel_0173 = get_kernel('dispatch_kernel_0173')


def launch_dispatch_kernel_0173(
    *args,
    grid: tuple[int, int, int],
    block: tuple[int, int, int] | None = None,
    shared_mem: int | None = None,
    stream=None,
    timeout_ms: float | None = None,
    arch: str | None = None,
    options: list[str] | None = None,
):
    return dispatch_kernel_0173.launch(
        *args,
        grid=grid,
        block=block,
        shared_mem=shared_mem,
        stream=stream,
        timeout_ms=timeout_ms,
        arch=arch,
        options=options,
    )


dispatch_kernel_0174 = get_kernel('dispatch_kernel_0174')


def launch_dispatch_kernel_0174(
    *args,
    grid: tuple[int, int, int],
    block: tuple[int, int, int] | None = None,
    shared_mem: int | None = None,
    stream=None,
    timeout_ms: float | None = None,
    arch: str | None = None,
    options: list[str] | None = None,
):
    return dispatch_kernel_0174.launch(
        *args,
        grid=grid,
        block=block,
        shared_mem=shared_mem,
        stream=stream,
        timeout_ms=timeout_ms,
        arch=arch,
        options=options,
    )


dispatch_kernel_0175 = get_kernel('dispatch_kernel_0175')


def launch_dispatch_kernel_0175(
    *args,
    grid: tuple[int, int, int],
    block: tuple[int, int, int] | None = None,
    shared_mem: int | None = None,
    stream=None,
    timeout_ms: float | None = None,
    arch: str | None = None,
    options: list[str] | None = None,
):
    return dispatch_kernel_0175.launch(
        *args,
        grid=grid,
        block=block,
        shared_mem=shared_mem,
        stream=stream,
        timeout_ms=timeout_ms,
        arch=arch,
        options=options,
    )


dispatch_kernel_0176 = get_kernel('dispatch_kernel_0176')


def launch_dispatch_kernel_0176(
    *args,
    grid: tuple[int, int, int],
    block: tuple[int, int, int] | None = None,
    shared_mem: int | None = None,
    stream=None,
    timeout_ms: float | None = None,
    arch: str | None = None,
    options: list[str] | None = None,
):
    return dispatch_kernel_0176.launch(
        *args,
        grid=grid,
        block=block,
        shared_mem=shared_mem,
        stream=stream,
        timeout_ms=timeout_ms,
        arch=arch,
        options=options,
    )


dispatch_kernel_0177 = get_kernel('dispatch_kernel_0177')


def launch_dispatch_kernel_0177(
    *args,
    grid: tuple[int, int, int],
    block: tuple[int, int, int] | None = None,
    shared_mem: int | None = None,
    stream=None,
    timeout_ms: float | None = None,
    arch: str | None = None,
    options: list[str] | None = None,
):
    return dispatch_kernel_0177.launch(
        *args,
        grid=grid,
        block=block,
        shared_mem=shared_mem,
        stream=stream,
        timeout_ms=timeout_ms,
        arch=arch,
        options=options,
    )


dispatch_kernel_0178 = get_kernel('dispatch_kernel_0178')


def launch_dispatch_kernel_0178(
    *args,
    grid: tuple[int, int, int],
    block: tuple[int, int, int] | None = None,
    shared_mem: int | None = None,
    stream=None,
    timeout_ms: float | None = None,
    arch: str | None = None,
    options: list[str] | None = None,
):
    return dispatch_kernel_0178.launch(
        *args,
        grid=grid,
        block=block,
        shared_mem=shared_mem,
        stream=stream,
        timeout_ms=timeout_ms,
        arch=arch,
        options=options,
    )


dispatch_kernel_0179 = get_kernel('dispatch_kernel_0179')


def launch_dispatch_kernel_0179(
    *args,
    grid: tuple[int, int, int],
    block: tuple[int, int, int] | None = None,
    shared_mem: int | None = None,
    stream=None,
    timeout_ms: float | None = None,
    arch: str | None = None,
    options: list[str] | None = None,
):
    return dispatch_kernel_0179.launch(
        *args,
        grid=grid,
        block=block,
        shared_mem=shared_mem,
        stream=stream,
        timeout_ms=timeout_ms,
        arch=arch,
        options=options,
    )


dispatch_kernel_0180 = get_kernel('dispatch_kernel_0180')


def launch_dispatch_kernel_0180(
    *args,
    grid: tuple[int, int, int],
    block: tuple[int, int, int] | None = None,
    shared_mem: int | None = None,
    stream=None,
    timeout_ms: float | None = None,
    arch: str | None = None,
    options: list[str] | None = None,
):
    return dispatch_kernel_0180.launch(
        *args,
        grid=grid,
        block=block,
        shared_mem=shared_mem,
        stream=stream,
        timeout_ms=timeout_ms,
        arch=arch,
        options=options,
    )


dispatch_kernel_0181 = get_kernel('dispatch_kernel_0181')


def launch_dispatch_kernel_0181(
    *args,
    grid: tuple[int, int, int],
    block: tuple[int, int, int] | None = None,
    shared_mem: int | None = None,
    stream=None,
    timeout_ms: float | None = None,
    arch: str | None = None,
    options: list[str] | None = None,
):
    return dispatch_kernel_0181.launch(
        *args,
        grid=grid,
        block=block,
        shared_mem=shared_mem,
        stream=stream,
        timeout_ms=timeout_ms,
        arch=arch,
        options=options,
    )


dispatch_kernel_0182 = get_kernel('dispatch_kernel_0182')


def launch_dispatch_kernel_0182(
    *args,
    grid: tuple[int, int, int],
    block: tuple[int, int, int] | None = None,
    shared_mem: int | None = None,
    stream=None,
    timeout_ms: float | None = None,
    arch: str | None = None,
    options: list[str] | None = None,
):
    return dispatch_kernel_0182.launch(
        *args,
        grid=grid,
        block=block,
        shared_mem=shared_mem,
        stream=stream,
        timeout_ms=timeout_ms,
        arch=arch,
        options=options,
    )


dispatch_kernel_0183 = get_kernel('dispatch_kernel_0183')


def launch_dispatch_kernel_0183(
    *args,
    grid: tuple[int, int, int],
    block: tuple[int, int, int] | None = None,
    shared_mem: int | None = None,
    stream=None,
    timeout_ms: float | None = None,
    arch: str | None = None,
    options: list[str] | None = None,
):
    return dispatch_kernel_0183.launch(
        *args,
        grid=grid,
        block=block,
        shared_mem=shared_mem,
        stream=stream,
        timeout_ms=timeout_ms,
        arch=arch,
        options=options,
    )


dispatch_kernel_0184 = get_kernel('dispatch_kernel_0184')


def launch_dispatch_kernel_0184(
    *args,
    grid: tuple[int, int, int],
    block: tuple[int, int, int] | None = None,
    shared_mem: int | None = None,
    stream=None,
    timeout_ms: float | None = None,
    arch: str | None = None,
    options: list[str] | None = None,
):
    return dispatch_kernel_0184.launch(
        *args,
        grid=grid,
        block=block,
        shared_mem=shared_mem,
        stream=stream,
        timeout_ms=timeout_ms,
        arch=arch,
        options=options,
    )


dispatch_kernel_0185 = get_kernel('dispatch_kernel_0185')


def launch_dispatch_kernel_0185(
    *args,
    grid: tuple[int, int, int],
    block: tuple[int, int, int] | None = None,
    shared_mem: int | None = None,
    stream=None,
    timeout_ms: float | None = None,
    arch: str | None = None,
    options: list[str] | None = None,
):
    return dispatch_kernel_0185.launch(
        *args,
        grid=grid,
        block=block,
        shared_mem=shared_mem,
        stream=stream,
        timeout_ms=timeout_ms,
        arch=arch,
        options=options,
    )


dispatch_kernel_0186 = get_kernel('dispatch_kernel_0186')


def launch_dispatch_kernel_0186(
    *args,
    grid: tuple[int, int, int],
    block: tuple[int, int, int] | None = None,
    shared_mem: int | None = None,
    stream=None,
    timeout_ms: float | None = None,
    arch: str | None = None,
    options: list[str] | None = None,
):
    return dispatch_kernel_0186.launch(
        *args,
        grid=grid,
        block=block,
        shared_mem=shared_mem,
        stream=stream,
        timeout_ms=timeout_ms,
        arch=arch,
        options=options,
    )


dispatch_kernel_0187 = get_kernel('dispatch_kernel_0187')


def launch_dispatch_kernel_0187(
    *args,
    grid: tuple[int, int, int],
    block: tuple[int, int, int] | None = None,
    shared_mem: int | None = None,
    stream=None,
    timeout_ms: float | None = None,
    arch: str | None = None,
    options: list[str] | None = None,
):
    return dispatch_kernel_0187.launch(
        *args,
        grid=grid,
        block=block,
        shared_mem=shared_mem,
        stream=stream,
        timeout_ms=timeout_ms,
        arch=arch,
        options=options,
    )


dispatch_kernel_0188 = get_kernel('dispatch_kernel_0188')


def launch_dispatch_kernel_0188(
    *args,
    grid: tuple[int, int, int],
    block: tuple[int, int, int] | None = None,
    shared_mem: int | None = None,
    stream=None,
    timeout_ms: float | None = None,
    arch: str | None = None,
    options: list[str] | None = None,
):
    return dispatch_kernel_0188.launch(
        *args,
        grid=grid,
        block=block,
        shared_mem=shared_mem,
        stream=stream,
        timeout_ms=timeout_ms,
        arch=arch,
        options=options,
    )


dispatch_kernel_0189 = get_kernel('dispatch_kernel_0189')


def launch_dispatch_kernel_0189(
    *args,
    grid: tuple[int, int, int],
    block: tuple[int, int, int] | None = None,
    shared_mem: int | None = None,
    stream=None,
    timeout_ms: float | None = None,
    arch: str | None = None,
    options: list[str] | None = None,
):
    return dispatch_kernel_0189.launch(
        *args,
        grid=grid,
        block=block,
        shared_mem=shared_mem,
        stream=stream,
        timeout_ms=timeout_ms,
        arch=arch,
        options=options,
    )


dispatch_kernel_0190 = get_kernel('dispatch_kernel_0190')


def launch_dispatch_kernel_0190(
    *args,
    grid: tuple[int, int, int],
    block: tuple[int, int, int] | None = None,
    shared_mem: int | None = None,
    stream=None,
    timeout_ms: float | None = None,
    arch: str | None = None,
    options: list[str] | None = None,
):
    return dispatch_kernel_0190.launch(
        *args,
        grid=grid,
        block=block,
        shared_mem=shared_mem,
        stream=stream,
        timeout_ms=timeout_ms,
        arch=arch,
        options=options,
    )


dispatch_kernel_0191 = get_kernel('dispatch_kernel_0191')


def launch_dispatch_kernel_0191(
    *args,
    grid: tuple[int, int, int],
    block: tuple[int, int, int] | None = None,
    shared_mem: int | None = None,
    stream=None,
    timeout_ms: float | None = None,
    arch: str | None = None,
    options: list[str] | None = None,
):
    return dispatch_kernel_0191.launch(
        *args,
        grid=grid,
        block=block,
        shared_mem=shared_mem,
        stream=stream,
        timeout_ms=timeout_ms,
        arch=arch,
        options=options,
    )


dispatch_kernel_0192 = get_kernel('dispatch_kernel_0192')


def launch_dispatch_kernel_0192(
    *args,
    grid: tuple[int, int, int],
    block: tuple[int, int, int] | None = None,
    shared_mem: int | None = None,
    stream=None,
    timeout_ms: float | None = None,
    arch: str | None = None,
    options: list[str] | None = None,
):
    return dispatch_kernel_0192.launch(
        *args,
        grid=grid,
        block=block,
        shared_mem=shared_mem,
        stream=stream,
        timeout_ms=timeout_ms,
        arch=arch,
        options=options,
    )


dispatch_kernel_0193 = get_kernel('dispatch_kernel_0193')


def launch_dispatch_kernel_0193(
    *args,
    grid: tuple[int, int, int],
    block: tuple[int, int, int] | None = None,
    shared_mem: int | None = None,
    stream=None,
    timeout_ms: float | None = None,
    arch: str | None = None,
    options: list[str] | None = None,
):
    return dispatch_kernel_0193.launch(
        *args,
        grid=grid,
        block=block,
        shared_mem=shared_mem,
        stream=stream,
        timeout_ms=timeout_ms,
        arch=arch,
        options=options,
    )


dispatch_kernel_0194 = get_kernel('dispatch_kernel_0194')


def launch_dispatch_kernel_0194(
    *args,
    grid: tuple[int, int, int],
    block: tuple[int, int, int] | None = None,
    shared_mem: int | None = None,
    stream=None,
    timeout_ms: float | None = None,
    arch: str | None = None,
    options: list[str] | None = None,
):
    return dispatch_kernel_0194.launch(
        *args,
        grid=grid,
        block=block,
        shared_mem=shared_mem,
        stream=stream,
        timeout_ms=timeout_ms,
        arch=arch,
        options=options,
    )


dispatch_kernel_0195 = get_kernel('dispatch_kernel_0195')


def launch_dispatch_kernel_0195(
    *args,
    grid: tuple[int, int, int],
    block: tuple[int, int, int] | None = None,
    shared_mem: int | None = None,
    stream=None,
    timeout_ms: float | None = None,
    arch: str | None = None,
    options: list[str] | None = None,
):
    return dispatch_kernel_0195.launch(
        *args,
        grid=grid,
        block=block,
        shared_mem=shared_mem,
        stream=stream,
        timeout_ms=timeout_ms,
        arch=arch,
        options=options,
    )


dispatch_kernel_0196 = get_kernel('dispatch_kernel_0196')


def launch_dispatch_kernel_0196(
    *args,
    grid: tuple[int, int, int],
    block: tuple[int, int, int] | None = None,
    shared_mem: int | None = None,
    stream=None,
    timeout_ms: float | None = None,
    arch: str | None = None,
    options: list[str] | None = None,
):
    return dispatch_kernel_0196.launch(
        *args,
        grid=grid,
        block=block,
        shared_mem=shared_mem,
        stream=stream,
        timeout_ms=timeout_ms,
        arch=arch,
        options=options,
    )


dispatch_kernel_0197 = get_kernel('dispatch_kernel_0197')


def launch_dispatch_kernel_0197(
    *args,
    grid: tuple[int, int, int],
    block: tuple[int, int, int] | None = None,
    shared_mem: int | None = None,
    stream=None,
    timeout_ms: float | None = None,
    arch: str | None = None,
    options: list[str] | None = None,
):
    return dispatch_kernel_0197.launch(
        *args,
        grid=grid,
        block=block,
        shared_mem=shared_mem,
        stream=stream,
        timeout_ms=timeout_ms,
        arch=arch,
        options=options,
    )


dispatch_kernel_0198 = get_kernel('dispatch_kernel_0198')


def launch_dispatch_kernel_0198(
    *args,
    grid: tuple[int, int, int],
    block: tuple[int, int, int] | None = None,
    shared_mem: int | None = None,
    stream=None,
    timeout_ms: float | None = None,
    arch: str | None = None,
    options: list[str] | None = None,
):
    return dispatch_kernel_0198.launch(
        *args,
        grid=grid,
        block=block,
        shared_mem=shared_mem,
        stream=stream,
        timeout_ms=timeout_ms,
        arch=arch,
        options=options,
    )


dispatch_kernel_0199 = get_kernel('dispatch_kernel_0199')


def launch_dispatch_kernel_0199(
    *args,
    grid: tuple[int, int, int],
    block: tuple[int, int, int] | None = None,
    shared_mem: int | None = None,
    stream=None,
    timeout_ms: float | None = None,
    arch: str | None = None,
    options: list[str] | None = None,
):
    return dispatch_kernel_0199.launch(
        *args,
        grid=grid,
        block=block,
        shared_mem=shared_mem,
        stream=stream,
        timeout_ms=timeout_ms,
        arch=arch,
        options=options,
    )


dispatch_kernel_0200 = get_kernel('dispatch_kernel_0200')


def launch_dispatch_kernel_0200(
    *args,
    grid: tuple[int, int, int],
    block: tuple[int, int, int] | None = None,
    shared_mem: int | None = None,
    stream=None,
    timeout_ms: float | None = None,
    arch: str | None = None,
    options: list[str] | None = None,
):
    return dispatch_kernel_0200.launch(
        *args,
        grid=grid,
        block=block,
        shared_mem=shared_mem,
        stream=stream,
        timeout_ms=timeout_ms,
        arch=arch,
        options=options,
    )


dispatch_kernel_0201 = get_kernel('dispatch_kernel_0201')


def launch_dispatch_kernel_0201(
    *args,
    grid: tuple[int, int, int],
    block: tuple[int, int, int] | None = None,
    shared_mem: int | None = None,
    stream=None,
    timeout_ms: float | None = None,
    arch: str | None = None,
    options: list[str] | None = None,
):
    return dispatch_kernel_0201.launch(
        *args,
        grid=grid,
        block=block,
        shared_mem=shared_mem,
        stream=stream,
        timeout_ms=timeout_ms,
        arch=arch,
        options=options,
    )


dispatch_kernel_0202 = get_kernel('dispatch_kernel_0202')


def launch_dispatch_kernel_0202(
    *args,
    grid: tuple[int, int, int],
    block: tuple[int, int, int] | None = None,
    shared_mem: int | None = None,
    stream=None,
    timeout_ms: float | None = None,
    arch: str | None = None,
    options: list[str] | None = None,
):
    return dispatch_kernel_0202.launch(
        *args,
        grid=grid,
        block=block,
        shared_mem=shared_mem,
        stream=stream,
        timeout_ms=timeout_ms,
        arch=arch,
        options=options,
    )


dispatch_kernel_0203 = get_kernel('dispatch_kernel_0203')


def launch_dispatch_kernel_0203(
    *args,
    grid: tuple[int, int, int],
    block: tuple[int, int, int] | None = None,
    shared_mem: int | None = None,
    stream=None,
    timeout_ms: float | None = None,
    arch: str | None = None,
    options: list[str] | None = None,
):
    return dispatch_kernel_0203.launch(
        *args,
        grid=grid,
        block=block,
        shared_mem=shared_mem,
        stream=stream,
        timeout_ms=timeout_ms,
        arch=arch,
        options=options,
    )


dispatch_kernel_0204 = get_kernel('dispatch_kernel_0204')


def launch_dispatch_kernel_0204(
    *args,
    grid: tuple[int, int, int],
    block: tuple[int, int, int] | None = None,
    shared_mem: int | None = None,
    stream=None,
    timeout_ms: float | None = None,
    arch: str | None = None,
    options: list[str] | None = None,
):
    return dispatch_kernel_0204.launch(
        *args,
        grid=grid,
        block=block,
        shared_mem=shared_mem,
        stream=stream,
        timeout_ms=timeout_ms,
        arch=arch,
        options=options,
    )


dispatch_kernel_0205 = get_kernel('dispatch_kernel_0205')


def launch_dispatch_kernel_0205(
    *args,
    grid: tuple[int, int, int],
    block: tuple[int, int, int] | None = None,
    shared_mem: int | None = None,
    stream=None,
    timeout_ms: float | None = None,
    arch: str | None = None,
    options: list[str] | None = None,
):
    return dispatch_kernel_0205.launch(
        *args,
        grid=grid,
        block=block,
        shared_mem=shared_mem,
        stream=stream,
        timeout_ms=timeout_ms,
        arch=arch,
        options=options,
    )


dispatch_kernel_0206 = get_kernel('dispatch_kernel_0206')


def launch_dispatch_kernel_0206(
    *args,
    grid: tuple[int, int, int],
    block: tuple[int, int, int] | None = None,
    shared_mem: int | None = None,
    stream=None,
    timeout_ms: float | None = None,
    arch: str | None = None,
    options: list[str] | None = None,
):
    return dispatch_kernel_0206.launch(
        *args,
        grid=grid,
        block=block,
        shared_mem=shared_mem,
        stream=stream,
        timeout_ms=timeout_ms,
        arch=arch,
        options=options,
    )


dispatch_kernel_0207 = get_kernel('dispatch_kernel_0207')


def launch_dispatch_kernel_0207(
    *args,
    grid: tuple[int, int, int],
    block: tuple[int, int, int] | None = None,
    shared_mem: int | None = None,
    stream=None,
    timeout_ms: float | None = None,
    arch: str | None = None,
    options: list[str] | None = None,
):
    return dispatch_kernel_0207.launch(
        *args,
        grid=grid,
        block=block,
        shared_mem=shared_mem,
        stream=stream,
        timeout_ms=timeout_ms,
        arch=arch,
        options=options,
    )


dispatch_kernel_0208 = get_kernel('dispatch_kernel_0208')


def launch_dispatch_kernel_0208(
    *args,
    grid: tuple[int, int, int],
    block: tuple[int, int, int] | None = None,
    shared_mem: int | None = None,
    stream=None,
    timeout_ms: float | None = None,
    arch: str | None = None,
    options: list[str] | None = None,
):
    return dispatch_kernel_0208.launch(
        *args,
        grid=grid,
        block=block,
        shared_mem=shared_mem,
        stream=stream,
        timeout_ms=timeout_ms,
        arch=arch,
        options=options,
    )


dispatch_kernel_0209 = get_kernel('dispatch_kernel_0209')


def launch_dispatch_kernel_0209(
    *args,
    grid: tuple[int, int, int],
    block: tuple[int, int, int] | None = None,
    shared_mem: int | None = None,
    stream=None,
    timeout_ms: float | None = None,
    arch: str | None = None,
    options: list[str] | None = None,
):
    return dispatch_kernel_0209.launch(
        *args,
        grid=grid,
        block=block,
        shared_mem=shared_mem,
        stream=stream,
        timeout_ms=timeout_ms,
        arch=arch,
        options=options,
    )


dispatch_kernel_0210 = get_kernel('dispatch_kernel_0210')


def launch_dispatch_kernel_0210(
    *args,
    grid: tuple[int, int, int],
    block: tuple[int, int, int] | None = None,
    shared_mem: int | None = None,
    stream=None,
    timeout_ms: float | None = None,
    arch: str | None = None,
    options: list[str] | None = None,
):
    return dispatch_kernel_0210.launch(
        *args,
        grid=grid,
        block=block,
        shared_mem=shared_mem,
        stream=stream,
        timeout_ms=timeout_ms,
        arch=arch,
        options=options,
    )


dispatch_kernel_0211 = get_kernel('dispatch_kernel_0211')


def launch_dispatch_kernel_0211(
    *args,
    grid: tuple[int, int, int],
    block: tuple[int, int, int] | None = None,
    shared_mem: int | None = None,
    stream=None,
    timeout_ms: float | None = None,
    arch: str | None = None,
    options: list[str] | None = None,
):
    return dispatch_kernel_0211.launch(
        *args,
        grid=grid,
        block=block,
        shared_mem=shared_mem,
        stream=stream,
        timeout_ms=timeout_ms,
        arch=arch,
        options=options,
    )


dispatch_kernel_0212 = get_kernel('dispatch_kernel_0212')


def launch_dispatch_kernel_0212(
    *args,
    grid: tuple[int, int, int],
    block: tuple[int, int, int] | None = None,
    shared_mem: int | None = None,
    stream=None,
    timeout_ms: float | None = None,
    arch: str | None = None,
    options: list[str] | None = None,
):
    return dispatch_kernel_0212.launch(
        *args,
        grid=grid,
        block=block,
        shared_mem=shared_mem,
        stream=stream,
        timeout_ms=timeout_ms,
        arch=arch,
        options=options,
    )


dispatch_kernel_0213 = get_kernel('dispatch_kernel_0213')


def launch_dispatch_kernel_0213(
    *args,
    grid: tuple[int, int, int],
    block: tuple[int, int, int] | None = None,
    shared_mem: int | None = None,
    stream=None,
    timeout_ms: float | None = None,
    arch: str | None = None,
    options: list[str] | None = None,
):
    return dispatch_kernel_0213.launch(
        *args,
        grid=grid,
        block=block,
        shared_mem=shared_mem,
        stream=stream,
        timeout_ms=timeout_ms,
        arch=arch,
        options=options,
    )


dispatch_kernel_0214 = get_kernel('dispatch_kernel_0214')


def launch_dispatch_kernel_0214(
    *args,
    grid: tuple[int, int, int],
    block: tuple[int, int, int] | None = None,
    shared_mem: int | None = None,
    stream=None,
    timeout_ms: float | None = None,
    arch: str | None = None,
    options: list[str] | None = None,
):
    return dispatch_kernel_0214.launch(
        *args,
        grid=grid,
        block=block,
        shared_mem=shared_mem,
        stream=stream,
        timeout_ms=timeout_ms,
        arch=arch,
        options=options,
    )


dispatch_kernel_0215 = get_kernel('dispatch_kernel_0215')


def launch_dispatch_kernel_0215(
    *args,
    grid: tuple[int, int, int],
    block: tuple[int, int, int] | None = None,
    shared_mem: int | None = None,
    stream=None,
    timeout_ms: float | None = None,
    arch: str | None = None,
    options: list[str] | None = None,
):
    return dispatch_kernel_0215.launch(
        *args,
        grid=grid,
        block=block,
        shared_mem=shared_mem,
        stream=stream,
        timeout_ms=timeout_ms,
        arch=arch,
        options=options,
    )


dispatch_kernel_0216 = get_kernel('dispatch_kernel_0216')


def launch_dispatch_kernel_0216(
    *args,
    grid: tuple[int, int, int],
    block: tuple[int, int, int] | None = None,
    shared_mem: int | None = None,
    stream=None,
    timeout_ms: float | None = None,
    arch: str | None = None,
    options: list[str] | None = None,
):
    return dispatch_kernel_0216.launch(
        *args,
        grid=grid,
        block=block,
        shared_mem=shared_mem,
        stream=stream,
        timeout_ms=timeout_ms,
        arch=arch,
        options=options,
    )


dispatch_kernel_0217 = get_kernel('dispatch_kernel_0217')


def launch_dispatch_kernel_0217(
    *args,
    grid: tuple[int, int, int],
    block: tuple[int, int, int] | None = None,
    shared_mem: int | None = None,
    stream=None,
    timeout_ms: float | None = None,
    arch: str | None = None,
    options: list[str] | None = None,
):
    return dispatch_kernel_0217.launch(
        *args,
        grid=grid,
        block=block,
        shared_mem=shared_mem,
        stream=stream,
        timeout_ms=timeout_ms,
        arch=arch,
        options=options,
    )


dispatch_kernel_0218 = get_kernel('dispatch_kernel_0218')


def launch_dispatch_kernel_0218(
    *args,
    grid: tuple[int, int, int],
    block: tuple[int, int, int] | None = None,
    shared_mem: int | None = None,
    stream=None,
    timeout_ms: float | None = None,
    arch: str | None = None,
    options: list[str] | None = None,
):
    return dispatch_kernel_0218.launch(
        *args,
        grid=grid,
        block=block,
        shared_mem=shared_mem,
        stream=stream,
        timeout_ms=timeout_ms,
        arch=arch,
        options=options,
    )


dispatch_kernel_0219 = get_kernel('dispatch_kernel_0219')


def launch_dispatch_kernel_0219(
    *args,
    grid: tuple[int, int, int],
    block: tuple[int, int, int] | None = None,
    shared_mem: int | None = None,
    stream=None,
    timeout_ms: float | None = None,
    arch: str | None = None,
    options: list[str] | None = None,
):
    return dispatch_kernel_0219.launch(
        *args,
        grid=grid,
        block=block,
        shared_mem=shared_mem,
        stream=stream,
        timeout_ms=timeout_ms,
        arch=arch,
        options=options,
    )


dispatch_kernel_0220 = get_kernel('dispatch_kernel_0220')


def launch_dispatch_kernel_0220(
    *args,
    grid: tuple[int, int, int],
    block: tuple[int, int, int] | None = None,
    shared_mem: int | None = None,
    stream=None,
    timeout_ms: float | None = None,
    arch: str | None = None,
    options: list[str] | None = None,
):
    return dispatch_kernel_0220.launch(
        *args,
        grid=grid,
        block=block,
        shared_mem=shared_mem,
        stream=stream,
        timeout_ms=timeout_ms,
        arch=arch,
        options=options,
    )


dispatch_kernel_0221 = get_kernel('dispatch_kernel_0221')


def launch_dispatch_kernel_0221(
    *args,
    grid: tuple[int, int, int],
    block: tuple[int, int, int] | None = None,
    shared_mem: int | None = None,
    stream=None,
    timeout_ms: float | None = None,
    arch: str | None = None,
    options: list[str] | None = None,
):
    return dispatch_kernel_0221.launch(
        *args,
        grid=grid,
        block=block,
        shared_mem=shared_mem,
        stream=stream,
        timeout_ms=timeout_ms,
        arch=arch,
        options=options,
    )


dispatch_kernel_0222 = get_kernel('dispatch_kernel_0222')


def launch_dispatch_kernel_0222(
    *args,
    grid: tuple[int, int, int],
    block: tuple[int, int, int] | None = None,
    shared_mem: int | None = None,
    stream=None,
    timeout_ms: float | None = None,
    arch: str | None = None,
    options: list[str] | None = None,
):
    return dispatch_kernel_0222.launch(
        *args,
        grid=grid,
        block=block,
        shared_mem=shared_mem,
        stream=stream,
        timeout_ms=timeout_ms,
        arch=arch,
        options=options,
    )


dispatch_kernel_0223 = get_kernel('dispatch_kernel_0223')


def launch_dispatch_kernel_0223(
    *args,
    grid: tuple[int, int, int],
    block: tuple[int, int, int] | None = None,
    shared_mem: int | None = None,
    stream=None,
    timeout_ms: float | None = None,
    arch: str | None = None,
    options: list[str] | None = None,
):
    return dispatch_kernel_0223.launch(
        *args,
        grid=grid,
        block=block,
        shared_mem=shared_mem,
        stream=stream,
        timeout_ms=timeout_ms,
        arch=arch,
        options=options,
    )


dispatch_kernel_0224 = get_kernel('dispatch_kernel_0224')


def launch_dispatch_kernel_0224(
    *args,
    grid: tuple[int, int, int],
    block: tuple[int, int, int] | None = None,
    shared_mem: int | None = None,
    stream=None,
    timeout_ms: float | None = None,
    arch: str | None = None,
    options: list[str] | None = None,
):
    return dispatch_kernel_0224.launch(
        *args,
        grid=grid,
        block=block,
        shared_mem=shared_mem,
        stream=stream,
        timeout_ms=timeout_ms,
        arch=arch,
        options=options,
    )


dispatch_kernel_0225 = get_kernel('dispatch_kernel_0225')


def launch_dispatch_kernel_0225(
    *args,
    grid: tuple[int, int, int],
    block: tuple[int, int, int] | None = None,
    shared_mem: int | None = None,
    stream=None,
    timeout_ms: float | None = None,
    arch: str | None = None,
    options: list[str] | None = None,
):
    return dispatch_kernel_0225.launch(
        *args,
        grid=grid,
        block=block,
        shared_mem=shared_mem,
        stream=stream,
        timeout_ms=timeout_ms,
        arch=arch,
        options=options,
    )


dispatch_kernel_0226 = get_kernel('dispatch_kernel_0226')


def launch_dispatch_kernel_0226(
    *args,
    grid: tuple[int, int, int],
    block: tuple[int, int, int] | None = None,
    shared_mem: int | None = None,
    stream=None,
    timeout_ms: float | None = None,
    arch: str | None = None,
    options: list[str] | None = None,
):
    return dispatch_kernel_0226.launch(
        *args,
        grid=grid,
        block=block,
        shared_mem=shared_mem,
        stream=stream,
        timeout_ms=timeout_ms,
        arch=arch,
        options=options,
    )


dispatch_kernel_0227 = get_kernel('dispatch_kernel_0227')


def launch_dispatch_kernel_0227(
    *args,
    grid: tuple[int, int, int],
    block: tuple[int, int, int] | None = None,
    shared_mem: int | None = None,
    stream=None,
    timeout_ms: float | None = None,
    arch: str | None = None,
    options: list[str] | None = None,
):
    return dispatch_kernel_0227.launch(
        *args,
        grid=grid,
        block=block,
        shared_mem=shared_mem,
        stream=stream,
        timeout_ms=timeout_ms,
        arch=arch,
        options=options,
    )


dispatch_kernel_0228 = get_kernel('dispatch_kernel_0228')


def launch_dispatch_kernel_0228(
    *args,
    grid: tuple[int, int, int],
    block: tuple[int, int, int] | None = None,
    shared_mem: int | None = None,
    stream=None,
    timeout_ms: float | None = None,
    arch: str | None = None,
    options: list[str] | None = None,
):
    return dispatch_kernel_0228.launch(
        *args,
        grid=grid,
        block=block,
        shared_mem=shared_mem,
        stream=stream,
        timeout_ms=timeout_ms,
        arch=arch,
        options=options,
    )


dispatch_kernel_0229 = get_kernel('dispatch_kernel_0229')


def launch_dispatch_kernel_0229(
    *args,
    grid: tuple[int, int, int],
    block: tuple[int, int, int] | None = None,
    shared_mem: int | None = None,
    stream=None,
    timeout_ms: float | None = None,
    arch: str | None = None,
    options: list[str] | None = None,
):
    return dispatch_kernel_0229.launch(
        *args,
        grid=grid,
        block=block,
        shared_mem=shared_mem,
        stream=stream,
        timeout_ms=timeout_ms,
        arch=arch,
        options=options,
    )


dispatch_kernel_0230 = get_kernel('dispatch_kernel_0230')


def launch_dispatch_kernel_0230(
    *args,
    grid: tuple[int, int, int],
    block: tuple[int, int, int] | None = None,
    shared_mem: int | None = None,
    stream=None,
    timeout_ms: float | None = None,
    arch: str | None = None,
    options: list[str] | None = None,
):
    return dispatch_kernel_0230.launch(
        *args,
        grid=grid,
        block=block,
        shared_mem=shared_mem,
        stream=stream,
        timeout_ms=timeout_ms,
        arch=arch,
        options=options,
    )


dispatch_kernel_0231 = get_kernel('dispatch_kernel_0231')


def launch_dispatch_kernel_0231(
    *args,
    grid: tuple[int, int, int],
    block: tuple[int, int, int] | None = None,
    shared_mem: int | None = None,
    stream=None,
    timeout_ms: float | None = None,
    arch: str | None = None,
    options: list[str] | None = None,
):
    return dispatch_kernel_0231.launch(
        *args,
        grid=grid,
        block=block,
        shared_mem=shared_mem,
        stream=stream,
        timeout_ms=timeout_ms,
        arch=arch,
        options=options,
    )


dispatch_kernel_0232 = get_kernel('dispatch_kernel_0232')


def launch_dispatch_kernel_0232(
    *args,
    grid: tuple[int, int, int],
    block: tuple[int, int, int] | None = None,
    shared_mem: int | None = None,
    stream=None,
    timeout_ms: float | None = None,
    arch: str | None = None,
    options: list[str] | None = None,
):
    return dispatch_kernel_0232.launch(
        *args,
        grid=grid,
        block=block,
        shared_mem=shared_mem,
        stream=stream,
        timeout_ms=timeout_ms,
        arch=arch,
        options=options,
    )
