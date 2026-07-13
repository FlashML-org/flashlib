"""Standalone fused row-squared norms; minimum architecture: sm_80."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from .kernels import ExportedKernel, KernelSpec

_ROW_NORM_KERNEL = ExportedKernel(
    KernelSpec(
        name="cake_row_squared_norm",
        symbol="cake_row_squared_norm",
        source="_row_norm.cu",
        threads=256,
        shared_mem_bytes=0,
        cluster_dims=(1, 1, 1),
        launch_mode="standard",
        parameters=(
            {"name": "input", "ctype": "const void *"},
            {"name": "output", "ctype": "float *"},
            {"name": "rows", "ctype": "int64_t"},
            {"name": "dim", "ctype": "int32_t"},
            {"name": "dtype_code", "ctype": "int32_t"},
        ),
        specializations={},
        compile_options=("--std=c++17",),
    )
)


def _validate(input_tensor: Any, output: Any) -> tuple[int, int, int]:
    import torch

    if not isinstance(input_tensor, torch.Tensor) or not input_tensor.is_cuda:
        raise TypeError("row-norm input must be a CUDA torch.Tensor")
    if input_tensor.dtype not in (torch.bfloat16, torch.float16):
        raise TypeError("row-norm input must have bfloat16 or float16 dtype")
    if input_tensor.ndim != 3 or not input_tensor.is_contiguous():
        raise ValueError("row-norm input must be contiguous with shape [B, rows, D]")
    if not isinstance(output, torch.Tensor) or not output.is_cuda:
        raise TypeError("row-norm output must be a CUDA torch.Tensor")
    if output.dtype is not torch.float32:
        raise TypeError("row-norm output must have float32 dtype")
    if tuple(output.shape) != tuple(input_tensor.shape[:-1]):
        raise ValueError("row-norm output must have shape input.shape[:-1]")
    if output.device != input_tensor.device or not output.is_contiguous():
        raise ValueError("row-norm output must be contiguous and on the input device")
    bsz, rows, dim = map(int, input_tensor.shape)
    return bsz * rows, dim, 0 if input_tensor.dtype is torch.bfloat16 else 1


_GRID_CAP = 4096


def _launch_geometry(rows: int, dim: int) -> tuple[int, int]:
    """Throughput-hint grid/block for ``cake_row_squared_norm``.

    Mirrors the kernel's (rows, dim)-keyed path selection: block-per-row for
    few wide rows, otherwise one power-of-two lane group per row with eight
    warps per block. The kernel grid-strides its row space, so this geometry
    only affects throughput — any grid is correct, which keeps prepared
    launches safe under pointer rebinds.
    """

    if dim % 8 == 0:
        chunks = dim // 8
        if chunks >= 33 and rows <= 512:
            return max(1, min(rows, _GRID_CAP)), 256
        lanes = 1
        while lanes < chunks and lanes < 32:
            lanes <<= 1
    else:
        lanes = 32
    rows_per_block = 8 * (32 // lanes)
    blocks = (rows + rows_per_block - 1) // rows_per_block
    return max(1, min(blocks, _GRID_CAP)), 256


@dataclass
class PreparedRowSquaredNorm:
    """One pointer-rebindable fused row-norm launch."""

    launch_plan: Any
    rows: int
    dim: int
    dtype_code: int
    _input_carrier: Any = field(default=None, repr=False)

    def rebind(self, input_tensor: Any, output: Any, *, stream: Any = None) -> None:
        rows, dim, dtype_code = _validate(input_tensor, output)
        if (rows, dim, dtype_code) != (self.rows, self.dim, self.dtype_code):
            raise RuntimeError("row-norm prepared launch topology changed")
        self.launch_plan.rebind_arguments({0: input_tensor, 1: output}, stream=stream)

    def launch(self, *, stream: Any = None, timeout_ms: float | None = None) -> None:
        self.launch_plan.launch(stream=stream, timeout_ms=timeout_ms)

    def bind_hot(self, input_tensor: Any) -> None:
        """Overwrite the input pointer carrier in place without submitting.

        Graph-captured runtimes bind here and replay the captured kernel
        chain themselves; the carrier targets the same persistent packed
        argument buffer the captured node's parameter update reads.
        """

        carrier = self._input_carrier
        if carrier is None:
            carrier = self.launch_plan.pointer_carrier("input")
            self._input_carrier = carrier
        carrier.value = input_tensor.data_ptr()

    def launch_hot(self, input_tensor: Any) -> None:
        """Overwrite the input pointer carrier in place and submit the launch.

        The prepared output buffer stays bound (plan-owned scratch with a
        stable pointer) and the launch goes to its preparation-time stream —
        no re-marshal, no per-launch stream query. The caller's signature
        cache must guarantee the tensor topology matches the preparation.
        """

        self.bind_hot(input_tensor)
        self.launch_plan.launch(stream=None, timeout_ms=None)

    def record_stream(self, stream: Any) -> None:
        """Tie every prepared launch argument to ``stream`` before release."""

        for value in self.launch_plan._keepalive:
            record_stream = getattr(value, "record_stream", None)
            if callable(record_stream):
                record_stream(stream)

    def release_bound_input(self, output: Any, *, stream: Any = None) -> None:
        """Replace the caller-owned input keepalive with slot-owned scratch."""

        self.launch_plan.rebind_arguments({0: output}, stream=stream)


def prepare_row_squared_norm(
    input_tensor: Any,
    output: Any,
    *,
    arch: str | None = None,
    stream: Any = None,
) -> PreparedRowSquaredNorm:
    rows, dim, dtype_code = _validate(input_tensor, output)
    grid_x, threads = _launch_geometry(rows, dim)
    launch_plan = _ROW_NORM_KERNEL.prepare_launch(
        input_tensor,
        output,
        rows,
        dim,
        dtype_code,
        grid=(grid_x, 1, 1),
        block=(threads, 1, 1),
        stream=stream,
        arch=arch,
    )
    return PreparedRowSquaredNorm(
        launch_plan=launch_plan,
        rows=rows,
        dim=dim,
        dtype_code=dtype_code,
    )


def row_squared_norm(
    input_tensor: Any,
    *,
    output: Any = None,
    arch: str | None = None,
    stream: Any = None,
    timeout_ms: float | None = None,
):
    """Compute FP32 squared L2 norms with one CUDA activity."""

    import torch

    if output is None:
        output = torch.empty(
            tuple(input_tensor.shape[:-1]),
            dtype=torch.float32,
            device=input_tensor.device,
        )
    prepared = prepare_row_squared_norm(input_tensor, output, arch=arch, stream=stream)
    prepared.launch(stream=stream, timeout_ms=timeout_ms)
    return output
