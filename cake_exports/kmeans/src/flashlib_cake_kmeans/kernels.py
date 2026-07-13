from __future__ import annotations

import json
import os
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
            load_prebuilt_kernel,
            resolve_gpu_arch,
        )

        resolved_arch = resolve_gpu_arch(arch)
        device_index = current_cuda_device_index()
        generation = compilation_cache_generation()
        # The disable escape hatch is part of the memo key: toggling it
        # mid-process must not keep serving the previously loaded binary.
        key = (resolved_arch, effective_options, device_index, generation, _prebuilt_disabled())
        kernel = self._compiled.get(key)
        if kernel is not None and not kernel.closed:
            return kernel
        prebuilt = _prebuilt_entry(self.spec.source, resolved_arch, effective_options)
        if prebuilt is not None:
            package = __package__ or __name__.rpartition(".")[0]
            binary = resources.files(package).joinpath(prebuilt["path"])
            kernel = load_prebuilt_kernel(
                binary.read_bytes,
                binary_digest=prebuilt["sha256"],
                func_name=self.spec.symbol,
                arch=resolved_arch,
                device_index=device_index,
                name=prebuilt["path"],
                options=effective_options,
            )
        else:
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
        param_names = tuple(parameter["name"] for parameter in self.spec.parameters)
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
                    param_names=param_names,
                )
            if self.spec.launch_mode == "cooperative":
                return kernel.prepare_launch_cooperative(
                    grid=grid,
                    block=block,
                    args=args,
                    arg_types=self._arg_types,
                    shared_mem=shared_mem,
                    stream=stream,
                    param_names=param_names,
                )
            return kernel.prepare_launch(
                grid=grid,
                block=block,
                args=args,
                arg_types=self._arg_types,
                shared_mem=shared_mem,
                stream=stream,
                param_names=param_names,
            )


def _load_manifest() -> dict[str, Any]:
    package = __package__ or __name__.rpartition(".")[0]
    return json.loads(resources.files(package).joinpath("manifest.json").read_text(encoding="utf-8"))


def _prebuilt_manifest_index(
    manifest: dict[str, Any],
) -> dict[tuple[str, str, tuple[str, ...]], dict[str, Any]]:
    section = manifest.get("prebuilt_binaries")
    if not isinstance(section, dict):
        return {}
    return {
        (entry["source"], entry["arch"], tuple(entry["compile_options"])): entry
        for entry in section.get("binaries", ())
    }


def _prebuilt_disabled() -> bool:
    value = os.environ.get("LOOM_EXPORTED_DISABLE_PREBUILT", "")
    return value.strip().lower() not in ("", "0", "false", "no", "off")


def _prebuilt_entry(source: str, arch: str, options: tuple[str, ...]) -> dict[str, Any] | None:
    if _prebuilt_disabled():
        return None
    return _PREBUILT_INDEX.get((source, arch, options))


_MANIFEST = _load_manifest()
_PREBUILT_INDEX = _prebuilt_manifest_index(_MANIFEST)
KERNELS = {entry["name"]: ExportedKernel(KernelSpec.from_manifest(entry)) for entry in _MANIFEST["kernels"]}


def get_kernel(name: str) -> ExportedKernel:
    try:
        return KERNELS[name]
    except KeyError as exc:
        available = ", ".join(sorted(KERNELS))
        raise KeyError(f"Unknown exported kernel {name!r}. Available: {available}") from exc


def _make_launcher(name: str):
    kernel = get_kernel(name)

    def launch(
        *args,
        grid: tuple[int, int, int],
        block: tuple[int, int, int] | None = None,
        shared_mem: int | None = None,
        stream=None,
        timeout_ms: float | None = None,
        arch: str | None = None,
        options: list[str] | None = None,
    ):
        return kernel.launch(
            *args,
            grid=grid,
            block=block,
            shared_mem=shared_mem,
            stream=stream,
            timeout_ms=timeout_ms,
            arch=arch,
            options=options,
        )

    launch.__name__ = f"launch_{name}"
    launch.__qualname__ = launch.__name__
    return launch


def _install_kernel_exports() -> None:
    for name in KERNELS:
        globals()[name] = get_kernel(name)
        globals()[f"launch_{name}"] = _make_launcher(name)


_install_kernel_exports()
