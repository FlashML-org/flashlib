"""Standalone fused BF16 pair row norms; minimum architecture: sm_80."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from .kernels import ExportedKernel, KernelSpec

_PAIR_ROW_NORM_KERNEL = ExportedKernel(
    KernelSpec(
        name="flashlib_cake_bf16_pair_row_norm",
        symbol="flashlib_cake_bf16_pair_row_norm",
        source="_row_norm.cu",
        threads=256,
        shared_mem_bytes=0,
        cluster_dims=(1, 1, 1),
        launch_mode="standard",
        parameters=(
            {"name": "x", "ctype": "const __nv_bfloat16 *"},
            {"name": "centroids", "ctype": "const __nv_bfloat16 *"},
            {"name": "x_sq", "ctype": "float *"},
            {"name": "c_sq", "ctype": "float *"},
            {"name": "x_rows", "ctype": "int64_t"},
            {"name": "c_rows", "ctype": "int64_t"},
            {"name": "dim", "ctype": "int32_t"},
            {"name": "compute_x", "ctype": "int32_t"},
            {"name": "compute_c", "ctype": "int32_t"},
        ),
        specializations={},
        compile_options=("--std=c++17",),
    )
)


def _validate(
    x: Any,
    centroids: Any,
    x_sq: Any,
    c_sq: Any,
) -> tuple[int, int, int]:
    import torch

    if not all(isinstance(item, torch.Tensor) and item.is_cuda for item in (x, centroids)):
        raise TypeError("pair-row-norm inputs must be CUDA torch.Tensor objects")
    if x.dtype is not torch.bfloat16 or centroids.dtype is not torch.bfloat16:
        raise TypeError("pair-row-norm inputs must have bfloat16 dtype")
    if x.ndim != 3 or centroids.ndim != 3 or not x.is_contiguous() or not centroids.is_contiguous():
        raise ValueError("pair-row-norm inputs must be contiguous [B, rows, D] tensors")
    bsz, x_rows, dim = map(int, x.shape)
    c_bsz, c_rows, c_dim = map(int, centroids.shape)
    if (bsz, dim) != (c_bsz, c_dim) or x.device != centroids.device:
        raise ValueError("pair-row-norm inputs must have matching batch, feature, and device")
    for name, output, shape in (
        ("x_sq", x_sq, (bsz, x_rows)),
        ("c_sq", c_sq, (bsz, c_rows)),
    ):
        if not isinstance(output, torch.Tensor) or not output.is_cuda:
            raise TypeError(f"{name} must be a CUDA torch.Tensor")
        if tuple(output.shape) != shape or output.dtype is not torch.float32:
            raise ValueError(f"{name} must have dtype float32 and shape {shape}")
        if output.device != x.device or not output.is_contiguous():
            raise ValueError(f"{name} must be contiguous and on {x.device}")
    return bsz * x_rows, bsz * c_rows, dim


def _threads_for_dim(dim: int) -> int:
    return min(256, max(32, 1 << (int(dim) - 1).bit_length()))


@dataclass
class PreparedBF16PairRowNorm:
    """One pointer-rebindable launch for either or both KMeans row norms."""

    launch_plan: Any
    x_rows: int
    c_rows: int
    dim: int
    compute_x: bool
    compute_c: bool

    def rebind(
        self,
        x: Any,
        centroids: Any,
        x_sq: Any,
        c_sq: Any,
        *,
        stream: Any = None,
    ) -> None:
        x_rows, c_rows, dim = _validate(x, centroids, x_sq, c_sq)
        if (x_rows, c_rows, dim) != (self.x_rows, self.c_rows, self.dim):
            raise RuntimeError("pair-row-norm prepared launch topology changed")
        self.launch_plan.rebind_arguments(
            {0: x, 1: centroids, 2: x_sq, 3: c_sq},
            stream=stream,
        )

    def launch(self, *, stream: Any = None, timeout_ms: float | None = None) -> None:
        self.launch_plan.launch(stream=stream, timeout_ms=timeout_ms)

    def release_bound_callers(self, keepalive: Any, *, stream: Any = None) -> None:
        """Drop caller tensors while retaining one slot-owned scratch tensor."""

        self.launch_plan.rebind_arguments(
            {0: keepalive, 1: keepalive, 2: keepalive, 3: keepalive},
            stream=stream,
        )


def prepare_bf16_pair_row_norm(
    x: Any,
    centroids: Any,
    x_sq: Any,
    c_sq: Any,
    *,
    compute_x: bool,
    compute_c: bool,
    arch: str | None = None,
    stream: Any = None,
) -> PreparedBF16PairRowNorm:
    if not compute_x and not compute_c:
        raise ValueError("pair-row-norm preparation requires at least one output")
    x_rows, c_rows, dim = _validate(x, centroids, x_sq, c_sq)
    launch_plan = _PAIR_ROW_NORM_KERNEL.prepare_launch(
        x,
        centroids,
        x_sq,
        c_sq,
        x_rows,
        c_rows,
        dim,
        int(compute_x),
        int(compute_c),
        grid=(x_rows + c_rows, 1, 1),
        block=(_threads_for_dim(dim), 1, 1),
        stream=stream,
        arch=arch,
    )
    return PreparedBF16PairRowNorm(
        launch_plan=launch_plan,
        x_rows=x_rows,
        c_rows=c_rows,
        dim=dim,
        compute_x=bool(compute_x),
        compute_c=bool(compute_c),
    )


def launch_bf16_pair_row_norm(
    x: Any,
    centroids: Any,
    x_sq: Any,
    c_sq: Any,
    *,
    compute_x: bool,
    compute_c: bool,
    stream: Any,
    arch: str,
    timeout_ms: float | None,
) -> None:
    """Compute either or both BF16 row-squared norms in one CUDA activity."""

    prepared = prepare_bf16_pair_row_norm(
        x,
        centroids,
        x_sq,
        c_sq,
        compute_x=compute_x,
        compute_c=compute_c,
        arch=arch,
        stream=stream,
    )
    try:
        prepared.launch(stream=stream, timeout_ms=timeout_ms)
    finally:
        prepared.release_bound_callers(
            x_sq if compute_x else c_sq,
            stream=stream,
        )
