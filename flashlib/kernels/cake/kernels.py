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


stage1_k32_unordered_ir = get_kernel('stage1_k32_unordered_ir')


def launch_stage1_k32_unordered_ir(
    *args,
    grid: tuple[int, int, int],
    block: tuple[int, int, int] | None = None,
    shared_mem: int | None = None,
    stream=None,
    timeout_ms: float | None = None,
    arch: str | None = None,
    options: list[str] | None = None,
):
    return stage1_k32_unordered_ir.launch(
        *args,
        grid=grid,
        block=block,
        shared_mem=shared_mem,
        stream=stream,
        timeout_ms=timeout_ms,
        arch=arch,
        options=options,
    )


stage1_k12_ir = get_kernel('stage1_k12_ir')


def launch_stage1_k12_ir(
    *args,
    grid: tuple[int, int, int],
    block: tuple[int, int, int] | None = None,
    shared_mem: int | None = None,
    stream=None,
    timeout_ms: float | None = None,
    arch: str | None = None,
    options: list[str] | None = None,
):
    return stage1_k12_ir.launch(
        *args,
        grid=grid,
        block=block,
        shared_mem=shared_mem,
        stream=stream,
        timeout_ms=timeout_ms,
        arch=arch,
        options=options,
    )


stage1_k16_ir = get_kernel('stage1_k16_ir')


def launch_stage1_k16_ir(
    *args,
    grid: tuple[int, int, int],
    block: tuple[int, int, int] | None = None,
    shared_mem: int | None = None,
    stream=None,
    timeout_ms: float | None = None,
    arch: str | None = None,
    options: list[str] | None = None,
):
    return stage1_k16_ir.launch(
        *args,
        grid=grid,
        block=block,
        shared_mem=shared_mem,
        stream=stream,
        timeout_ms=timeout_ms,
        arch=arch,
        options=options,
    )


stage1_k20_ir = get_kernel('stage1_k20_ir')


def launch_stage1_k20_ir(
    *args,
    grid: tuple[int, int, int],
    block: tuple[int, int, int] | None = None,
    shared_mem: int | None = None,
    stream=None,
    timeout_ms: float | None = None,
    arch: str | None = None,
    options: list[str] | None = None,
):
    return stage1_k20_ir.launch(
        *args,
        grid=grid,
        block=block,
        shared_mem=shared_mem,
        stream=stream,
        timeout_ms=timeout_ms,
        arch=arch,
        options=options,
    )


stage1_k25_ir = get_kernel('stage1_k25_ir')


def launch_stage1_k25_ir(
    *args,
    grid: tuple[int, int, int],
    block: tuple[int, int, int] | None = None,
    shared_mem: int | None = None,
    stream=None,
    timeout_ms: float | None = None,
    arch: str | None = None,
    options: list[str] | None = None,
):
    return stage1_k25_ir.launch(
        *args,
        grid=grid,
        block=block,
        shared_mem=shared_mem,
        stream=stream,
        timeout_ms=timeout_ms,
        arch=arch,
        options=options,
    )


stage1_k30_ir = get_kernel('stage1_k30_ir')


def launch_stage1_k30_ir(
    *args,
    grid: tuple[int, int, int],
    block: tuple[int, int, int] | None = None,
    shared_mem: int | None = None,
    stream=None,
    timeout_ms: float | None = None,
    arch: str | None = None,
    options: list[str] | None = None,
):
    return stage1_k30_ir.launch(
        *args,
        grid=grid,
        block=block,
        shared_mem=shared_mem,
        stream=stream,
        timeout_ms=timeout_ms,
        arch=arch,
        options=options,
    )


stage1_ir = get_kernel('stage1_ir')


def launch_stage1_ir(
    *args,
    grid: tuple[int, int, int],
    block: tuple[int, int, int] | None = None,
    shared_mem: int | None = None,
    stream=None,
    timeout_ms: float | None = None,
    arch: str | None = None,
    options: list[str] | None = None,
):
    return stage1_ir.launch(
        *args,
        grid=grid,
        block=block,
        shared_mem=shared_mem,
        stream=stream,
        timeout_ms=timeout_ms,
        arch=arch,
        options=options,
    )


stage1_k20_unordered_ir = get_kernel('stage1_k20_unordered_ir')


def launch_stage1_k20_unordered_ir(
    *args,
    grid: tuple[int, int, int],
    block: tuple[int, int, int] | None = None,
    shared_mem: int | None = None,
    stream=None,
    timeout_ms: float | None = None,
    arch: str | None = None,
    options: list[str] | None = None,
):
    return stage1_k20_unordered_ir.launch(
        *args,
        grid=grid,
        block=block,
        shared_mem=shared_mem,
        stream=stream,
        timeout_ms=timeout_ms,
        arch=arch,
        options=options,
    )


stage1_k30_unordered_ir = get_kernel('stage1_k30_unordered_ir')


def launch_stage1_k30_unordered_ir(
    *args,
    grid: tuple[int, int, int],
    block: tuple[int, int, int] | None = None,
    shared_mem: int | None = None,
    stream=None,
    timeout_ms: float | None = None,
    arch: str | None = None,
    options: list[str] | None = None,
):
    return stage1_k30_unordered_ir.launch(
        *args,
        grid=grid,
        block=block,
        shared_mem=shared_mem,
        stream=stream,
        timeout_ms=timeout_ms,
        arch=arch,
        options=options,
    )


knn_build_evolve_7bfc_k32_merge_s4_rowbase_cache = get_kernel('knn_build_evolve_7bfc_k32_merge_s4_rowbase_cache')


def launch_knn_build_evolve_7bfc_k32_merge_s4_rowbase_cache(
    *args,
    grid: tuple[int, int, int],
    block: tuple[int, int, int] | None = None,
    shared_mem: int | None = None,
    stream=None,
    timeout_ms: float | None = None,
    arch: str | None = None,
    options: list[str] | None = None,
):
    return knn_build_evolve_7bfc_k32_merge_s4_rowbase_cache.launch(
        *args,
        grid=grid,
        block=block,
        shared_mem=shared_mem,
        stream=stream,
        timeout_ms=timeout_ms,
        arch=arch,
        options=options,
    )


merge_k12_ir = get_kernel('merge_k12_ir')


def launch_merge_k12_ir(
    *args,
    grid: tuple[int, int, int],
    block: tuple[int, int, int] | None = None,
    shared_mem: int | None = None,
    stream=None,
    timeout_ms: float | None = None,
    arch: str | None = None,
    options: list[str] | None = None,
):
    return merge_k12_ir.launch(
        *args,
        grid=grid,
        block=block,
        shared_mem=shared_mem,
        stream=stream,
        timeout_ms=timeout_ms,
        arch=arch,
        options=options,
    )


merge_k16_ir = get_kernel('merge_k16_ir')


def launch_merge_k16_ir(
    *args,
    grid: tuple[int, int, int],
    block: tuple[int, int, int] | None = None,
    shared_mem: int | None = None,
    stream=None,
    timeout_ms: float | None = None,
    arch: str | None = None,
    options: list[str] | None = None,
):
    return merge_k16_ir.launch(
        *args,
        grid=grid,
        block=block,
        shared_mem=shared_mem,
        stream=stream,
        timeout_ms=timeout_ms,
        arch=arch,
        options=options,
    )


merge_k20_ir = get_kernel('merge_k20_ir')


def launch_merge_k20_ir(
    *args,
    grid: tuple[int, int, int],
    block: tuple[int, int, int] | None = None,
    shared_mem: int | None = None,
    stream=None,
    timeout_ms: float | None = None,
    arch: str | None = None,
    options: list[str] | None = None,
):
    return merge_k20_ir.launch(
        *args,
        grid=grid,
        block=block,
        shared_mem=shared_mem,
        stream=stream,
        timeout_ms=timeout_ms,
        arch=arch,
        options=options,
    )


merge_k25_ir = get_kernel('merge_k25_ir')


def launch_merge_k25_ir(
    *args,
    grid: tuple[int, int, int],
    block: tuple[int, int, int] | None = None,
    shared_mem: int | None = None,
    stream=None,
    timeout_ms: float | None = None,
    arch: str | None = None,
    options: list[str] | None = None,
):
    return merge_k25_ir.launch(
        *args,
        grid=grid,
        block=block,
        shared_mem=shared_mem,
        stream=stream,
        timeout_ms=timeout_ms,
        arch=arch,
        options=options,
    )


merge_k30_ir = get_kernel('merge_k30_ir')


def launch_merge_k30_ir(
    *args,
    grid: tuple[int, int, int],
    block: tuple[int, int, int] | None = None,
    shared_mem: int | None = None,
    stream=None,
    timeout_ms: float | None = None,
    arch: str | None = None,
    options: list[str] | None = None,
):
    return merge_k30_ir.launch(
        *args,
        grid=grid,
        block=block,
        shared_mem=shared_mem,
        stream=stream,
        timeout_ms=timeout_ms,
        arch=arch,
        options=options,
    )


merge_ir = get_kernel('merge_ir')


def launch_merge_ir(
    *args,
    grid: tuple[int, int, int],
    block: tuple[int, int, int] | None = None,
    shared_mem: int | None = None,
    stream=None,
    timeout_ms: float | None = None,
    arch: str | None = None,
    options: list[str] | None = None,
):
    return merge_ir.launch(
        *args,
        grid=grid,
        block=block,
        shared_mem=shared_mem,
        stream=stream,
        timeout_ms=timeout_ms,
        arch=arch,
        options=options,
    )


merge_k32_unordered_ir = get_kernel('merge_k32_unordered_ir')


def launch_merge_k32_unordered_ir(
    *args,
    grid: tuple[int, int, int],
    block: tuple[int, int, int] | None = None,
    shared_mem: int | None = None,
    stream=None,
    timeout_ms: float | None = None,
    arch: str | None = None,
    options: list[str] | None = None,
):
    return merge_k32_unordered_ir.launch(
        *args,
        grid=grid,
        block=block,
        shared_mem=shared_mem,
        stream=stream,
        timeout_ms=timeout_ms,
        arch=arch,
        options=options,
    )


merge_k20_unordered_ir = get_kernel('merge_k20_unordered_ir')


def launch_merge_k20_unordered_ir(
    *args,
    grid: tuple[int, int, int],
    block: tuple[int, int, int] | None = None,
    shared_mem: int | None = None,
    stream=None,
    timeout_ms: float | None = None,
    arch: str | None = None,
    options: list[str] | None = None,
):
    return merge_k20_unordered_ir.launch(
        *args,
        grid=grid,
        block=block,
        shared_mem=shared_mem,
        stream=stream,
        timeout_ms=timeout_ms,
        arch=arch,
        options=options,
    )


merge_k30_unordered_ir = get_kernel('merge_k30_unordered_ir')


def launch_merge_k30_unordered_ir(
    *args,
    grid: tuple[int, int, int],
    block: tuple[int, int, int] | None = None,
    shared_mem: int | None = None,
    stream=None,
    timeout_ms: float | None = None,
    arch: str | None = None,
    options: list[str] | None = None,
):
    return merge_k30_unordered_ir.launch(
        *args,
        grid=grid,
        block=block,
        shared_mem=shared_mem,
        stream=stream,
        timeout_ms=timeout_ms,
        arch=arch,
        options=options,
    )


merge_k30_s8_ir = get_kernel('merge_k30_s8_ir')


def launch_merge_k30_s8_ir(
    *args,
    grid: tuple[int, int, int],
    block: tuple[int, int, int] | None = None,
    shared_mem: int | None = None,
    stream=None,
    timeout_ms: float | None = None,
    arch: str | None = None,
    options: list[str] | None = None,
):
    return merge_k30_s8_ir.launch(
        *args,
        grid=grid,
        block=block,
        shared_mem=shared_mem,
        stream=stream,
        timeout_ms=timeout_ms,
        arch=arch,
        options=options,
    )


merge_k12_s8_ir = get_kernel('merge_k12_s8_ir')


def launch_merge_k12_s8_ir(
    *args,
    grid: tuple[int, int, int],
    block: tuple[int, int, int] | None = None,
    shared_mem: int | None = None,
    stream=None,
    timeout_ms: float | None = None,
    arch: str | None = None,
    options: list[str] | None = None,
):
    return merge_k12_s8_ir.launch(
        *args,
        grid=grid,
        block=block,
        shared_mem=shared_mem,
        stream=stream,
        timeout_ms=timeout_ms,
        arch=arch,
        options=options,
    )


k5t64_stage1_ir = get_kernel('k5t64_stage1_ir')


def launch_k5t64_stage1_ir(
    *args,
    grid: tuple[int, int, int],
    block: tuple[int, int, int] | None = None,
    shared_mem: int | None = None,
    stream=None,
    timeout_ms: float | None = None,
    arch: str | None = None,
    options: list[str] | None = None,
):
    return k5t64_stage1_ir.launch(
        *args,
        grid=grid,
        block=block,
        shared_mem=shared_mem,
        stream=stream,
        timeout_ms=timeout_ms,
        arch=arch,
        options=options,
    )


k5t64_merge_ir = get_kernel('k5t64_merge_ir')


def launch_k5t64_merge_ir(
    *args,
    grid: tuple[int, int, int],
    block: tuple[int, int, int] | None = None,
    shared_mem: int | None = None,
    stream=None,
    timeout_ms: float | None = None,
    arch: str | None = None,
    options: list[str] | None = None,
):
    return k5t64_merge_ir.launch(
        *args,
        grid=grid,
        block=block,
        shared_mem=shared_mem,
        stream=stream,
        timeout_ms=timeout_ms,
        arch=arch,
        options=options,
    )


k5t64_merge_k10_s4_cache_ir = get_kernel('k5t64_merge_k10_s4_cache_ir')


def launch_k5t64_merge_k10_s4_cache_ir(
    *args,
    grid: tuple[int, int, int],
    block: tuple[int, int, int] | None = None,
    shared_mem: int | None = None,
    stream=None,
    timeout_ms: float | None = None,
    arch: str | None = None,
    options: list[str] | None = None,
):
    return k5t64_merge_k10_s4_cache_ir.launch(
        *args,
        grid=grid,
        block=block,
        shared_mem=shared_mem,
        stream=stream,
        timeout_ms=timeout_ms,
        arch=arch,
        options=options,
    )


k5t64_merge_k10_s7_cache_ir = get_kernel('k5t64_merge_k10_s7_cache_ir')


def launch_k5t64_merge_k10_s7_cache_ir(
    *args,
    grid: tuple[int, int, int],
    block: tuple[int, int, int] | None = None,
    shared_mem: int | None = None,
    stream=None,
    timeout_ms: float | None = None,
    arch: str | None = None,
    options: list[str] | None = None,
):
    return k5t64_merge_k10_s7_cache_ir.launch(
        *args,
        grid=grid,
        block=block,
        shared_mem=shared_mem,
        stream=stream,
        timeout_ms=timeout_ms,
        arch=arch,
        options=options,
    )


k10t32_stage1_ir = get_kernel('k10t32_stage1_ir')


def launch_k10t32_stage1_ir(
    *args,
    grid: tuple[int, int, int],
    block: tuple[int, int, int] | None = None,
    shared_mem: int | None = None,
    stream=None,
    timeout_ms: float | None = None,
    arch: str | None = None,
    options: list[str] | None = None,
):
    return k10t32_stage1_ir.launch(
        *args,
        grid=grid,
        block=block,
        shared_mem=shared_mem,
        stream=stream,
        timeout_ms=timeout_ms,
        arch=arch,
        options=options,
    )


k10t32_merge_ir = get_kernel('k10t32_merge_ir')


def launch_k10t32_merge_ir(
    *args,
    grid: tuple[int, int, int],
    block: tuple[int, int, int] | None = None,
    shared_mem: int | None = None,
    stream=None,
    timeout_ms: float | None = None,
    arch: str | None = None,
    options: list[str] | None = None,
):
    return k10t32_merge_ir.launch(
        *args,
        grid=grid,
        block=block,
        shared_mem=shared_mem,
        stream=stream,
        timeout_ms=timeout_ms,
        arch=arch,
        options=options,
    )


k10t32_merge_k10_s4_cache_ir = get_kernel('k10t32_merge_k10_s4_cache_ir')


def launch_k10t32_merge_k10_s4_cache_ir(
    *args,
    grid: tuple[int, int, int],
    block: tuple[int, int, int] | None = None,
    shared_mem: int | None = None,
    stream=None,
    timeout_ms: float | None = None,
    arch: str | None = None,
    options: list[str] | None = None,
):
    return k10t32_merge_k10_s4_cache_ir.launch(
        *args,
        grid=grid,
        block=block,
        shared_mem=shared_mem,
        stream=stream,
        timeout_ms=timeout_ms,
        arch=arch,
        options=options,
    )


k10t32_merge_k10_s7_cache_ir = get_kernel('k10t32_merge_k10_s7_cache_ir')


def launch_k10t32_merge_k10_s7_cache_ir(
    *args,
    grid: tuple[int, int, int],
    block: tuple[int, int, int] | None = None,
    shared_mem: int | None = None,
    stream=None,
    timeout_ms: float | None = None,
    arch: str | None = None,
    options: list[str] | None = None,
):
    return k10t32_merge_k10_s7_cache_ir.launch(
        *args,
        grid=grid,
        block=block,
        shared_mem=shared_mem,
        stream=stream,
        timeout_ms=timeout_ms,
        arch=arch,
        options=options,
    )


k10root_stage1 = get_kernel('k10root_stage1')


def launch_k10root_stage1(
    *args,
    grid: tuple[int, int, int],
    block: tuple[int, int, int] | None = None,
    shared_mem: int | None = None,
    stream=None,
    timeout_ms: float | None = None,
    arch: str | None = None,
    options: list[str] | None = None,
):
    return k10root_stage1.launch(
        *args,
        grid=grid,
        block=block,
        shared_mem=shared_mem,
        stream=stream,
        timeout_ms=timeout_ms,
        arch=arch,
        options=options,
    )


search_knn_search_warp_direct_v1 = get_kernel('search_knn_search_warp_direct_v1')


def launch_search_knn_search_warp_direct_v1(
    *args,
    grid: tuple[int, int, int],
    block: tuple[int, int, int] | None = None,
    shared_mem: int | None = None,
    stream=None,
    timeout_ms: float | None = None,
    arch: str | None = None,
    options: list[str] | None = None,
):
    return search_knn_search_warp_direct_v1.launch(
        *args,
        grid=grid,
        block=block,
        shared_mem=shared_mem,
        stream=stream,
        timeout_ms=timeout_ms,
        arch=arch,
        options=options,
    )


search_knn_search_warp_split_partial_v1 = get_kernel('search_knn_search_warp_split_partial_v1')


def launch_search_knn_search_warp_split_partial_v1(
    *args,
    grid: tuple[int, int, int],
    block: tuple[int, int, int] | None = None,
    shared_mem: int | None = None,
    stream=None,
    timeout_ms: float | None = None,
    arch: str | None = None,
    options: list[str] | None = None,
):
    return search_knn_search_warp_split_partial_v1.launch(
        *args,
        grid=grid,
        block=block,
        shared_mem=shared_mem,
        stream=stream,
        timeout_ms=timeout_ms,
        arch=arch,
        options=options,
    )


search_knn_search_warp_split_merge_v1 = get_kernel('search_knn_search_warp_split_merge_v1')


def launch_search_knn_search_warp_split_merge_v1(
    *args,
    grid: tuple[int, int, int],
    block: tuple[int, int, int] | None = None,
    shared_mem: int | None = None,
    stream=None,
    timeout_ms: float | None = None,
    arch: str | None = None,
    options: list[str] | None = None,
):
    return search_knn_search_warp_split_merge_v1.launch(
        *args,
        grid=grid,
        block=block,
        shared_mem=shared_mem,
        stream=stream,
        timeout_ms=timeout_ms,
        arch=arch,
        options=options,
    )


search_q1_knn_search_q1_tile_reduce_partial_v1 = get_kernel('search_q1_knn_search_q1_tile_reduce_partial_v1')


def launch_search_q1_knn_search_q1_tile_reduce_partial_v1(
    *args,
    grid: tuple[int, int, int],
    block: tuple[int, int, int] | None = None,
    shared_mem: int | None = None,
    stream=None,
    timeout_ms: float | None = None,
    arch: str | None = None,
    options: list[str] | None = None,
):
    return search_q1_knn_search_q1_tile_reduce_partial_v1.launch(
        *args,
        grid=grid,
        block=block,
        shared_mem=shared_mem,
        stream=stream,
        timeout_ms=timeout_ms,
        arch=arch,
        options=options,
    )


search_q1_knn_search_q1_tile_reduce_merge_v1 = get_kernel('search_q1_knn_search_q1_tile_reduce_merge_v1')


def launch_search_q1_knn_search_q1_tile_reduce_merge_v1(
    *args,
    grid: tuple[int, int, int],
    block: tuple[int, int, int] | None = None,
    shared_mem: int | None = None,
    stream=None,
    timeout_ms: float | None = None,
    arch: str | None = None,
    options: list[str] | None = None,
):
    return search_q1_knn_search_q1_tile_reduce_merge_v1.launch(
        *args,
        grid=grid,
        block=block,
        shared_mem=shared_mem,
        stream=stream,
        timeout_ms=timeout_ms,
        arch=arch,
        options=options,
    )


kmeans_v10 = get_kernel('kmeans_v10')


def launch_kmeans_v10(
    *args,
    grid: tuple[int, int, int],
    block: tuple[int, int, int] | None = None,
    shared_mem: int | None = None,
    stream=None,
    timeout_ms: float | None = None,
    arch: str | None = None,
    options: list[str] | None = None,
):
    return kmeans_v10.launch(
        *args,
        grid=grid,
        block=block,
        shared_mem=shared_mem,
        stream=stream,
        timeout_ms=timeout_ms,
        arch=arch,
        options=options,
    )


kmeans_v15 = get_kernel('kmeans_v15')


def launch_kmeans_v15(
    *args,
    grid: tuple[int, int, int],
    block: tuple[int, int, int] | None = None,
    shared_mem: int | None = None,
    stream=None,
    timeout_ms: float | None = None,
    arch: str | None = None,
    options: list[str] | None = None,
):
    return kmeans_v15.launch(
        *args,
        grid=grid,
        block=block,
        shared_mem=shared_mem,
        stream=stream,
        timeout_ms=timeout_ms,
        arch=arch,
        options=options,
    )


kmeans_d64_direct = get_kernel('kmeans_d64_direct')


def launch_kmeans_d64_direct(
    *args,
    grid: tuple[int, int, int],
    block: tuple[int, int, int] | None = None,
    shared_mem: int | None = None,
    stream=None,
    timeout_ms: float | None = None,
    arch: str | None = None,
    options: list[str] | None = None,
):
    return kmeans_d64_direct.launch(
        *args,
        grid=grid,
        block=block,
        shared_mem=shared_mem,
        stream=stream,
        timeout_ms=timeout_ms,
        arch=arch,
        options=options,
    )


kmeans_microdim_direct = get_kernel('kmeans_microdim_direct')


def launch_kmeans_microdim_direct(
    *args,
    grid: tuple[int, int, int],
    block: tuple[int, int, int] | None = None,
    shared_mem: int | None = None,
    stream=None,
    timeout_ms: float | None = None,
    arch: str | None = None,
    options: list[str] | None = None,
):
    return kmeans_microdim_direct.launch(
        *args,
        grid=grid,
        block=block,
        shared_mem=shared_mem,
        stream=stream,
        timeout_ms=timeout_ms,
        arch=arch,
        options=options,
    )


kmeans_microdim_pack = get_kernel('kmeans_microdim_pack')


def launch_kmeans_microdim_pack(
    *args,
    grid: tuple[int, int, int],
    block: tuple[int, int, int] | None = None,
    shared_mem: int | None = None,
    stream=None,
    timeout_ms: float | None = None,
    arch: str | None = None,
    options: list[str] | None = None,
):
    return kmeans_microdim_pack.launch(
        *args,
        grid=grid,
        block=block,
        shared_mem=shared_mem,
        stream=stream,
        timeout_ms=timeout_ms,
        arch=arch,
        options=options,
    )


kmeans_microdim_main = get_kernel('kmeans_microdim_main')


def launch_kmeans_microdim_main(
    *args,
    grid: tuple[int, int, int],
    block: tuple[int, int, int] | None = None,
    shared_mem: int | None = None,
    stream=None,
    timeout_ms: float | None = None,
    arch: str | None = None,
    options: list[str] | None = None,
):
    return kmeans_microdim_main.launch(
        *args,
        grid=grid,
        block=block,
        shared_mem=shared_mem,
        stream=stream,
        timeout_ms=timeout_ms,
        arch=arch,
        options=options,
    )


kmeans_lowdim_pack = get_kernel('kmeans_lowdim_pack')


def launch_kmeans_lowdim_pack(
    *args,
    grid: tuple[int, int, int],
    block: tuple[int, int, int] | None = None,
    shared_mem: int | None = None,
    stream=None,
    timeout_ms: float | None = None,
    arch: str | None = None,
    options: list[str] | None = None,
):
    return kmeans_lowdim_pack.launch(
        *args,
        grid=grid,
        block=block,
        shared_mem=shared_mem,
        stream=stream,
        timeout_ms=timeout_ms,
        arch=arch,
        options=options,
    )


kmeans_lowdim_main = get_kernel('kmeans_lowdim_main')


def launch_kmeans_lowdim_main(
    *args,
    grid: tuple[int, int, int],
    block: tuple[int, int, int] | None = None,
    shared_mem: int | None = None,
    stream=None,
    timeout_ms: float | None = None,
    arch: str | None = None,
    options: list[str] | None = None,
):
    return kmeans_lowdim_main.launch(
        *args,
        grid=grid,
        block=block,
        shared_mem=shared_mem,
        stream=stream,
        timeout_ms=timeout_ms,
        arch=arch,
        options=options,
    )


kmeans_gap_pad_pack = get_kernel('kmeans_gap_pad_pack')


def launch_kmeans_gap_pad_pack(
    *args,
    grid: tuple[int, int, int],
    block: tuple[int, int, int] | None = None,
    shared_mem: int | None = None,
    stream=None,
    timeout_ms: float | None = None,
    arch: str | None = None,
    options: list[str] | None = None,
):
    return kmeans_gap_pad_pack.launch(
        *args,
        grid=grid,
        block=block,
        shared_mem=shared_mem,
        stream=stream,
        timeout_ms=timeout_ms,
        arch=arch,
        options=options,
    )


kmeans_pad192_pack = get_kernel('kmeans_pad192_pack')


def launch_kmeans_pad192_pack(
    *args,
    grid: tuple[int, int, int],
    block: tuple[int, int, int] | None = None,
    shared_mem: int | None = None,
    stream=None,
    timeout_ms: float | None = None,
    arch: str | None = None,
    options: list[str] | None = None,
):
    return kmeans_pad192_pack.launch(
        *args,
        grid=grid,
        block=block,
        shared_mem=shared_mem,
        stream=stream,
        timeout_ms=timeout_ms,
        arch=arch,
        options=options,
    )


kmeans_d160_padded_pack = get_kernel('kmeans_d160_padded_pack')


def launch_kmeans_d160_padded_pack(
    *args,
    grid: tuple[int, int, int],
    block: tuple[int, int, int] | None = None,
    shared_mem: int | None = None,
    stream=None,
    timeout_ms: float | None = None,
    arch: str | None = None,
    options: list[str] | None = None,
):
    return kmeans_d160_padded_pack.launch(
        *args,
        grid=grid,
        block=block,
        shared_mem=shared_mem,
        stream=stream,
        timeout_ms=timeout_ms,
        arch=arch,
        options=options,
    )


kmeans_d160_splitd = get_kernel('kmeans_d160_splitd')


def launch_kmeans_d160_splitd(
    *args,
    grid: tuple[int, int, int],
    block: tuple[int, int, int] | None = None,
    shared_mem: int | None = None,
    stream=None,
    timeout_ms: float | None = None,
    arch: str | None = None,
    options: list[str] | None = None,
):
    return kmeans_d160_splitd.launch(
        *args,
        grid=grid,
        block=block,
        shared_mem=shared_mem,
        stream=stream,
        timeout_ms=timeout_ms,
        arch=arch,
        options=options,
    )


kmeans_d192_single = get_kernel('kmeans_d192_single')


def launch_kmeans_d192_single(
    *args,
    grid: tuple[int, int, int],
    block: tuple[int, int, int] | None = None,
    shared_mem: int | None = None,
    stream=None,
    timeout_ms: float | None = None,
    arch: str | None = None,
    options: list[str] | None = None,
):
    return kmeans_d192_single.launch(
        *args,
        grid=grid,
        block=block,
        shared_mem=shared_mem,
        stream=stream,
        timeout_ms=timeout_ms,
        arch=arch,
        options=options,
    )


kmeans_d192_splitd = get_kernel('kmeans_d192_splitd')


def launch_kmeans_d192_splitd(
    *args,
    grid: tuple[int, int, int],
    block: tuple[int, int, int] | None = None,
    shared_mem: int | None = None,
    stream=None,
    timeout_ms: float | None = None,
    arch: str | None = None,
    options: list[str] | None = None,
):
    return kmeans_d192_splitd.launch(
        *args,
        grid=grid,
        block=block,
        shared_mem=shared_mem,
        stream=stream,
        timeout_ms=timeout_ms,
        arch=arch,
        options=options,
    )


kmeans_d256_single = get_kernel('kmeans_d256_single')


def launch_kmeans_d256_single(
    *args,
    grid: tuple[int, int, int],
    block: tuple[int, int, int] | None = None,
    shared_mem: int | None = None,
    stream=None,
    timeout_ms: float | None = None,
    arch: str | None = None,
    options: list[str] | None = None,
):
    return kmeans_d256_single.launch(
        *args,
        grid=grid,
        block=block,
        shared_mem=shared_mem,
        stream=stream,
        timeout_ms=timeout_ms,
        arch=arch,
        options=options,
    )


kmeans_d256_splitd = get_kernel('kmeans_d256_splitd')


def launch_kmeans_d256_splitd(
    *args,
    grid: tuple[int, int, int],
    block: tuple[int, int, int] | None = None,
    shared_mem: int | None = None,
    stream=None,
    timeout_ms: float | None = None,
    arch: str | None = None,
    options: list[str] | None = None,
):
    return kmeans_d256_splitd.launch(
        *args,
        grid=grid,
        block=block,
        shared_mem=shared_mem,
        stream=stream,
        timeout_ms=timeout_ms,
        arch=arch,
        options=options,
    )


kmeans_highd_splitd = get_kernel('kmeans_highd_splitd')


def launch_kmeans_highd_splitd(
    *args,
    grid: tuple[int, int, int],
    block: tuple[int, int, int] | None = None,
    shared_mem: int | None = None,
    stream=None,
    timeout_ms: float | None = None,
    arch: str | None = None,
    options: list[str] | None = None,
):
    return kmeans_highd_splitd.launch(
        *args,
        grid=grid,
        block=block,
        shared_mem=shared_mem,
        stream=stream,
        timeout_ms=timeout_ms,
        arch=arch,
        options=options,
    )


kmeans_highd_splitk_partial = get_kernel('kmeans_highd_splitk_partial')


def launch_kmeans_highd_splitk_partial(
    *args,
    grid: tuple[int, int, int],
    block: tuple[int, int, int] | None = None,
    shared_mem: int | None = None,
    stream=None,
    timeout_ms: float | None = None,
    arch: str | None = None,
    options: list[str] | None = None,
):
    return kmeans_highd_splitk_partial.launch(
        *args,
        grid=grid,
        block=block,
        shared_mem=shared_mem,
        stream=stream,
        timeout_ms=timeout_ms,
        arch=arch,
        options=options,
    )


kmeans_highd_splitk_reduce = get_kernel('kmeans_highd_splitk_reduce')


def launch_kmeans_highd_splitk_reduce(
    *args,
    grid: tuple[int, int, int],
    block: tuple[int, int, int] | None = None,
    shared_mem: int | None = None,
    stream=None,
    timeout_ms: float | None = None,
    arch: str | None = None,
    options: list[str] | None = None,
):
    return kmeans_highd_splitk_reduce.launch(
        *args,
        grid=grid,
        block=block,
        shared_mem=shared_mem,
        stream=stream,
        timeout_ms=timeout_ms,
        arch=arch,
        options=options,
    )


kmeans_highd_splitk_blockn64_g2r4_partial = get_kernel('kmeans_highd_splitk_blockn64_g2r4_partial')


def launch_kmeans_highd_splitk_blockn64_g2r4_partial(
    *args,
    grid: tuple[int, int, int],
    block: tuple[int, int, int] | None = None,
    shared_mem: int | None = None,
    stream=None,
    timeout_ms: float | None = None,
    arch: str | None = None,
    options: list[str] | None = None,
):
    return kmeans_highd_splitk_blockn64_g2r4_partial.launch(
        *args,
        grid=grid,
        block=block,
        shared_mem=shared_mem,
        stream=stream,
        timeout_ms=timeout_ms,
        arch=arch,
        options=options,
    )


kmeans_highd_splitk_blockn64_g2r4_reduce = get_kernel('kmeans_highd_splitk_blockn64_g2r4_reduce')


def launch_kmeans_highd_splitk_blockn64_g2r4_reduce(
    *args,
    grid: tuple[int, int, int],
    block: tuple[int, int, int] | None = None,
    shared_mem: int | None = None,
    stream=None,
    timeout_ms: float | None = None,
    arch: str | None = None,
    options: list[str] | None = None,
):
    return kmeans_highd_splitk_blockn64_g2r4_reduce.launch(
        *args,
        grid=grid,
        block=block,
        shared_mem=shared_mem,
        stream=stream,
        timeout_ms=timeout_ms,
        arch=arch,
        options=options,
    )


kmeans_highd_splitk_blockn64_g1r4_partial = get_kernel('kmeans_highd_splitk_blockn64_g1r4_partial')


def launch_kmeans_highd_splitk_blockn64_g1r4_partial(
    *args,
    grid: tuple[int, int, int],
    block: tuple[int, int, int] | None = None,
    shared_mem: int | None = None,
    stream=None,
    timeout_ms: float | None = None,
    arch: str | None = None,
    options: list[str] | None = None,
):
    return kmeans_highd_splitk_blockn64_g1r4_partial.launch(
        *args,
        grid=grid,
        block=block,
        shared_mem=shared_mem,
        stream=stream,
        timeout_ms=timeout_ms,
        arch=arch,
        options=options,
    )


kmeans_highd_splitk_blockn64_g1r4_reduce = get_kernel('kmeans_highd_splitk_blockn64_g1r4_reduce')


def launch_kmeans_highd_splitk_blockn64_g1r4_reduce(
    *args,
    grid: tuple[int, int, int],
    block: tuple[int, int, int] | None = None,
    shared_mem: int | None = None,
    stream=None,
    timeout_ms: float | None = None,
    arch: str | None = None,
    options: list[str] | None = None,
):
    return kmeans_highd_splitk_blockn64_g1r4_reduce.launch(
        *args,
        grid=grid,
        block=block,
        shared_mem=shared_mem,
        stream=stream,
        timeout_ms=timeout_ms,
        arch=arch,
        options=options,
    )


kmeans_highd_paired_xreuse_r47_partial = get_kernel('kmeans_highd_paired_xreuse_r47_partial')


def launch_kmeans_highd_paired_xreuse_r47_partial(
    *args,
    grid: tuple[int, int, int],
    block: tuple[int, int, int] | None = None,
    shared_mem: int | None = None,
    stream=None,
    timeout_ms: float | None = None,
    arch: str | None = None,
    options: list[str] | None = None,
):
    return kmeans_highd_paired_xreuse_r47_partial.launch(
        *args,
        grid=grid,
        block=block,
        shared_mem=shared_mem,
        stream=stream,
        timeout_ms=timeout_ms,
        arch=arch,
        options=options,
    )


kmeans_highd_paired_ownerreduce_r39_reduce1 = get_kernel('kmeans_highd_paired_ownerreduce_r39_reduce1')


def launch_kmeans_highd_paired_ownerreduce_r39_reduce1(
    *args,
    grid: tuple[int, int, int],
    block: tuple[int, int, int] | None = None,
    shared_mem: int | None = None,
    stream=None,
    timeout_ms: float | None = None,
    arch: str | None = None,
    options: list[str] | None = None,
):
    return kmeans_highd_paired_ownerreduce_r39_reduce1.launch(
        *args,
        grid=grid,
        block=block,
        shared_mem=shared_mem,
        stream=stream,
        timeout_ms=timeout_ms,
        arch=arch,
        options=options,
    )


kmeans_highd_paired_packedpartial_r2_partial = get_kernel('kmeans_highd_paired_packedpartial_r2_partial')


def launch_kmeans_highd_paired_packedpartial_r2_partial(
    *args,
    grid: tuple[int, int, int],
    block: tuple[int, int, int] | None = None,
    shared_mem: int | None = None,
    stream=None,
    timeout_ms: float | None = None,
    arch: str | None = None,
    options: list[str] | None = None,
):
    return kmeans_highd_paired_packedpartial_r2_partial.launch(
        *args,
        grid=grid,
        block=block,
        shared_mem=shared_mem,
        stream=stream,
        timeout_ms=timeout_ms,
        arch=arch,
        options=options,
    )


kmeans_highd_paired_packedpartial_r2_reduce = get_kernel('kmeans_highd_paired_packedpartial_r2_reduce')


def launch_kmeans_highd_paired_packedpartial_r2_reduce(
    *args,
    grid: tuple[int, int, int],
    block: tuple[int, int, int] | None = None,
    shared_mem: int | None = None,
    stream=None,
    timeout_ms: float | None = None,
    arch: str | None = None,
    options: list[str] | None = None,
):
    return kmeans_highd_paired_packedpartial_r2_reduce.launch(
        *args,
        grid=grid,
        block=block,
        shared_mem=shared_mem,
        stream=stream,
        timeout_ms=timeout_ms,
        arch=arch,
        options=options,
    )


