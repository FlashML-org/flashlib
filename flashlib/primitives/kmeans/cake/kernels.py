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
        )


class ExportedKernel:
    def __init__(self, spec: KernelSpec):
        self.spec = spec
        self._compiled: dict[tuple[str | None, tuple[str, ...]], "CUDAKernel"] = {}

    @property
    def parameters(self) -> tuple[dict[str, str], ...]:
        return self.spec.parameters

    def source_text(self) -> str:
        package = __package__ or __name__.rpartition(".")[0]
        return resources.files(package).joinpath(self.spec.source).read_text(encoding="utf-8")

    def compile(self, *, arch: str | None = None, options: list[str] | None = None) -> "CUDAKernel":
        key = (arch, tuple(options or ()))
        kernel = self._compiled.get(key)
        if kernel is None:
            from ._runtime import CUDAKernel, compile_cuda

            cubin = compile_cuda(self.source_text(), arch=arch, name=f"{self.spec.name}.cu", options=options)
            kernel = CUDAKernel(cubin, self.spec.symbol)
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
        if len(args) != len(self.spec.parameters):
            expected = ", ".join(p["name"] for p in self.spec.parameters)
            raise TypeError(f"{self.spec.name} expects {len(self.spec.parameters)} args ({expected}), got {len(args)}")
        if block is None:
            block = (self.spec.threads, 1, 1)
        if shared_mem is None:
            shared_mem = self.spec.shared_mem_bytes
        arg_types = [p["ctype"] for p in self.spec.parameters]
        kernel = self.compile(arch=arch, options=options)
        if self.spec.launch_mode == "cluster":
            kernel.launch_cluster(
                grid=grid,
                block=block,
                args=list(args),
                arg_types=arg_types,
                cluster_dims=self.spec.cluster_dims,
                shared_mem=shared_mem,
                stream=stream,
                timeout_ms=timeout_ms,
            )
        elif self.spec.launch_mode == "cooperative":
            kernel.launch_cooperative(
                grid=grid,
                block=block,
                args=list(args),
                arg_types=arg_types,
                shared_mem=shared_mem,
                stream=stream,
                timeout_ms=timeout_ms,
            )
        else:
            kernel.launch(
                grid=grid,
                block=block,
                args=list(args),
                arg_types=arg_types,
                shared_mem=shared_mem,
                stream=stream,
                timeout_ms=timeout_ms,
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


def _make_launcher(kernel: ExportedKernel):
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

    launch.__name__ = f"launch_{kernel.spec.name}"
    launch.__qualname__ = launch.__name__
    return launch


for _name, _kernel in KERNELS.items():
    globals()[_name] = _kernel
    globals()[f"launch_{_name}"] = _make_launcher(_kernel)


__all__ = (
    ["KERNELS", "KernelSpec", "ExportedKernel", "get_kernel"]
    + sorted(KERNELS)
    + [f"launch_{name}" for name in sorted(KERNELS)]
)
