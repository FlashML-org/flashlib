from __future__ import annotations

import ctypes
from dataclasses import dataclass
from typing import Any

from . import kernels as _kernels


BLOCK_N = 128
BLOCK_N64 = 64
BLOCK_K = 256
FEAT_CHUNK = 64
PACK_THREADS = 256
PACK_GRID_CAP = 4096
PAIRED_GRID_CAP = 2516
SPLITK_GRID_CAP = 4096
SPLITK_MIN_K_TILES = 16
SPLITK_GROUP_K_TILES_G1 = 1
SPLITK_GROUP_K_TILES_G2 = 2
PAIRED_K_SLICES = 8
PACKED_KEY_PRODUCER_GRID_CAP = 160
_COMPILE_OPTIONS = ["--use_fast_math"]
_CU_TENSOR_MAP_SIZE = 128

_GAP_PAD: dict[int, int] = {
    48: 64,
    112: 128,
    224: 256,
    288: 320,
    352: 384,
    416: 448,
    480: 512,
}
SUPPORTED_DIMS = (
    16,
    32,
    48,
    64,
    80,
    96,
    112,
    128,
    144,
    160,
    176,
    192,
    224,
    256,
    288,
    320,
    352,
    384,
    416,
    448,
    480,
    512,
)

_TMAP_CACHE: dict[tuple[int, int, int, int, int, int], Any] = {}
_SCRATCH_CACHE: dict[tuple[int, int, int, int, int, int, int, int], tuple[Any, Any]] = {}
_PARTIAL_CACHE: dict[tuple[int, int, int, int, int, int, int], tuple[Any, Any]] = {}
_PARTIAL64_CACHE: dict[tuple[int, int, int, int, int, int, int, int], tuple[Any, Any]] = {}
_PARTIAL_KEY_CACHE: dict[tuple[int, int, int, int, int, int, int, int], Any] = {}
_LIVE_LAUNCH_TENSORS: list[tuple[Any, ...]] = []

_NO_PADDING_HIGHD_SHAPES = {
    (1, 512, 8192, 320),
    (1, 1024, 8192, 320),
    (1, 2048, 8192, 320),
    (1, 512, 8192, 384),
    (1, 768, 8192, 384),
    (1, 1024, 8192, 384),
    (1, 512, 4096, 448),
    (1, 1024, 4096, 448),
    (1, 512, 8192, 448),
    (1, 1024, 8192, 448),
    (1, 2048, 4096, 448),
    (1, 512, 4096, 512),
    (1, 1024, 4096, 512),
    (1, 512, 8192, 512),
    (1, 1024, 8192, 512),
    (1, 2048, 4096, 512),
}


@dataclass(frozen=True)
class RouteDecision:
    route_id: str
    kernel: str | None = None
    child_route: str | None = None
    padded_dim: int | None = None


def _torch():
    import torch

    return torch


def _driver():
    from cuda.bindings import driver

    return driver


def _device_index(device: Any) -> int:
    torch = _torch()
    index = device.index
    if index is None and device.type == "cuda":
        index = torch.cuda.current_device()
    return int(index or 0)


def _tmap_to_device(tmap: Any, *, device_index: int):
    torch = _torch()
    host_ptr = tmap.getPtr()
    tmap_bytes = bytes((ctypes.c_ubyte * _CU_TENSOR_MAP_SIZE).from_address(host_ptr))
    tmap_host = torch.frombuffer(bytearray(tmap_bytes), dtype=torch.uint8)
    with torch.cuda.device(device_index):
        tmap_device = torch.empty(_CU_TENSOR_MAP_SIZE, dtype=torch.uint8, device="cuda")
        tmap_device.copy_(tmap_host)
    return tmap_device


def _create_tensor_map_3d(
    data_ptr: int,
    global_height: int,
    shared_height: int,
    width: int,
    block_width: int,
    *,
    device_index: int,
):
    if width % 64 != 0 or block_width % 64 != 0:
        raise ValueError(f"3D BF16 tensor maps require width and block_width multiples of 64, got {width}, {block_width}")
    driver = _driver()
    key = (device_index, int(data_ptr), int(global_height), int(shared_height), int(width), int(block_width))
    cached = _TMAP_CACHE.get(key)
    if cached is not None:
        return cached

    err, tmap = driver.cuTensorMapEncodeTiled(
        driver.CUtensorMapDataType.CU_TENSOR_MAP_DATA_TYPE_BFLOAT16,
        3,
        data_ptr,
        [
            driver.cuuint64_t(64),
            driver.cuuint64_t(global_height),
            driver.cuuint64_t(width // 64),
        ],
        [
            driver.cuuint64_t(width * 2),
            driver.cuuint64_t(128),
        ],
        [
            driver.cuuint32_t(64),
            driver.cuuint32_t(shared_height),
            driver.cuuint32_t(block_width // 64),
        ],
        [
            driver.cuuint32_t(1),
            driver.cuuint32_t(1),
            driver.cuuint32_t(1),
        ],
        driver.CUtensorMapInterleave.CU_TENSOR_MAP_INTERLEAVE_NONE,
        driver.CUtensorMapSwizzle.CU_TENSOR_MAP_SWIZZLE_128B,
        driver.CUtensorMapL2promotion.CU_TENSOR_MAP_L2_PROMOTION_NONE,
        driver.CUtensorMapFloatOOBfill.CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE,
    )
    if err != 0:
        raise RuntimeError(f"cuTensorMapEncodeTiled (3D BF16) failed: CUresult={err}")
    cached = _tmap_to_device(tmap, device_index=device_index)
    _TMAP_CACHE[key] = cached
    return cached


def _retain_launch_tensors(*tensors: Any) -> None:
    _LIVE_LAUNCH_TENSORS.append(tuple(tensors))
    del _LIVE_LAUNCH_TENSORS[:-16]


def _launch(
    kernel: _kernels.ExportedKernel,
    *args: Any,
    grid: tuple[int, int, int],
    block: tuple[int, int, int] | None = None,
    arch: str | None,
    stream: Any,
    timeout_ms: float | None,
) -> None:
    kernel.launch(
        *args,
        grid=grid,
        block=block,
        shared_mem=kernel.spec.shared_mem_bytes,
        stream=stream,
        timeout_ms=timeout_ms,
        arch=arch,
        options=_COMPILE_OPTIONS,
    )


def _validate_tensor(name: str, tensor: Any) -> None:
    torch = _torch()
    if not isinstance(tensor, torch.Tensor):
        raise TypeError(f"{name} must be a torch.Tensor")
    if not tensor.is_cuda:
        raise ValueError(f"{name} must be a CUDA tensor")
    if tensor.dtype is not torch.bfloat16:
        raise ValueError(f"{name} must have dtype torch.bfloat16")
    if tensor.ndim != 3:
        raise ValueError(f"{name} must have shape [B, N, D]")
    if not tensor.is_contiguous():
        raise ValueError(f"{name} must be contiguous")


def _validate_pair(x: Any, centroids: Any) -> tuple[int, int, int, int]:
    _validate_tensor("x", x)
    _validate_tensor("centroids", centroids)
    if x.device != centroids.device:
        raise ValueError("x and centroids must be on the same CUDA device")
    if int(x.shape[0]) != int(centroids.shape[0]):
        raise ValueError("x and centroids must have the same batch dimension")
    if int(x.shape[2]) != int(centroids.shape[2]):
        raise ValueError("x and centroids must have the same feature dimension")
    bsz, n_points, dim = (int(v) for v in x.shape)
    n_clusters = int(centroids.shape[1])
    _validate_supported_shape(B=bsz, N=n_points, D=dim, K=n_clusters, dtype="bfloat16")
    return bsz, n_points, dim, n_clusters


def _allocate_output(x: Any, out: Any | None):
    torch = _torch()
    expected = (int(x.shape[0]), int(x.shape[1]))
    if out is None:
        return torch.empty(expected, dtype=torch.int32, device=x.device)
    if not isinstance(out, torch.Tensor):
        raise TypeError("out must be a torch.Tensor")
    if tuple(out.shape) != expected:
        raise ValueError(f"out must have shape {expected}")
    if out.dtype is not torch.int32:
        raise ValueError("out must have dtype torch.int32")
    if out.device != x.device:
        raise ValueError("out must be on the x device")
    if not out.is_contiguous():
        raise ValueError("out must be contiguous")
    return out


def _validate_sq_tensor(name: str, tensor: Any, expected: tuple[int, int], device: Any):
    torch = _torch()
    if not isinstance(tensor, torch.Tensor):
        raise TypeError(f"{name} must be a torch.Tensor")
    if tuple(tensor.shape) != expected:
        raise ValueError(f"{name} must have shape {expected}")
    if tensor.dtype is not torch.float32:
        raise ValueError(f"{name} must have dtype torch.float32")
    if tensor.device != device:
        raise ValueError(f"{name} must be on the input device")
    if not tensor.is_contiguous():
        raise ValueError(f"{name} must be contiguous")
    return tensor


def _validate_supported_shape(*, B: int, N: int, D: int, K: int, dtype: Any) -> None:
    dtype_name = str(dtype).replace("torch.", "")
    if dtype_name not in {"bfloat16", "bf16"}:
        raise ValueError(f"flash_kmeans_assign requires bfloat16 input, got {dtype}")
    if D not in SUPPORTED_DIMS:
        raise ValueError(f"flash_kmeans_assign has no exported route for D={D}; supported D values are {SUPPORTED_DIMS}")
    if N % BLOCK_N != 0:
        raise ValueError(f"N must be divisible by {BLOCK_N}, got {N}")
    if K % BLOCK_K != 0:
        raise ValueError(f"K must be divisible by {BLOCK_K}, got {K}")
    if B <= 0 or N <= 0 or K <= 0:
        raise ValueError(f"B, N, and K must be positive, got B={B}, N={N}, K={K}")


def _use_small_grid(n_points: int, n_clusters: int) -> bool:
    return (n_points // BLOCK_N) <= 8 and (n_clusters // BLOCK_K) <= 2


def _use_highd_splitk(*, dim: int, num_n_tiles: int, k_tiles: int) -> bool:
    if num_n_tiles > 16:
        return False
    return k_tiles >= 32 or (k_tiles >= SPLITK_MIN_K_TILES and dim >= 448)


def _use_no_padding_portfolio_r63(*, B: int, N: int, D: int, K: int) -> bool:
    return (B, N, K, D) in _NO_PADDING_HIGHD_SHAPES


def select_flash_kmeans_route(*, B: int, N: int, D: int, K: int, dtype: Any = "bfloat16") -> RouteDecision:
    _validate_supported_shape(B=B, N=N, D=D, K=K, dtype=dtype)
    num_n_tiles = N // BLOCK_N
    k_tiles = K // BLOCK_K

    if D == 64:
        return RouteDecision("d64_direct_single64_1p2gap_9f2a_v1", kernel="kmeans_d64_direct")
    if D in {16, 32}:
        child = "microdim_direct_9c0d_v1" if K == 512 and N <= 2432 else "microdim_pack_6cd2_v1"
        return RouteDecision("microdim_hybrid_9c0d_v1", child_route=child)
    if D in _GAP_PAD:
        return RouteDecision("gap_pad_to_supported_seed_v1", padded_dim=_GAP_PAD[D])
    if D in {80, 96}:
        return RouteDecision("lowdim_e50c_v1", kernel="kmeans_lowdim_main", padded_dim=128)
    if D in {144, 176}:
        route = "d192_single_repeated_mma_v1" if (num_n_tiles % 2 != 0) or _use_small_grid(N, K) else "d192_paired_repeated_mma_v1"
        return RouteDecision("d144_d160_d176_pad192_tail_repair_f9b2_v1", child_route=route, padded_dim=192)
    if D == 160:
        return RouteDecision("d160_padded_single_repeated_mma_v2", kernel="kmeans_d192_single", padded_dim=192)
    if D == 192:
        if (num_n_tiles % 2 != 0) or _use_small_grid(N, K):
            return RouteDecision("d192_single_repeated_mma_v1", kernel="kmeans_d192_single")
        return RouteDecision("d192_paired_repeated_mma_v1", kernel="kmeans_d192_splitd")
    if D == 256:
        return RouteDecision("d256_single_repeated_mma_v1", kernel="kmeans_d256_single")
    if _use_no_padding_portfolio_r63(B=B, N=N, D=D, K=K):
        return RouteDecision("flash_kmeans_assign_no_padding_portfolio_r63_g1_d512n512_v1")
    if D in {320, 384, 448, 512}:
        if _use_highd_splitk(dim=D, num_n_tiles=num_n_tiles, k_tiles=k_tiles):
            return RouteDecision("highd_splitk_8de8_v1")
        return RouteDecision("highd_splitd_single_tile_6fcf_v1", kernel="kmeans_highd_splitd")
    if B == 8 and N == 8192 and K == 256:
        return RouteDecision("d128_even_near_floor_v10_repair", kernel="kmeans_v10")
    if _use_small_grid(N, K):
        return RouteDecision("small_grid_single_tile_v10", kernel="kmeans_v10")
    if num_n_tiles % 2 == 0:
        return RouteDecision("paired_large_v15", kernel="kmeans_v15")
    return RouteDecision("aligned_v10_fallback", kernel="kmeans_v10")


def _scratch_buffers(inputs: dict[str, Any], d_pad: int) -> tuple[Any, Any]:
    torch = _torch()
    device_index = _device_index(inputs["x"].device)
    bsz = int(inputs["B"])
    n_points = int(inputs["N"])
    n_clusters = int(inputs["K"])
    dim = int(inputs["D"])
    key = (
        device_index,
        int(inputs["x"].data_ptr()),
        int(inputs["centroids"].data_ptr()),
        bsz,
        n_points,
        n_clusters,
        dim,
        d_pad,
    )
    cached = _SCRATCH_CACHE.get(key)
    if cached is not None:
        return cached
    x_pad = torch.empty((bsz, n_points, d_pad), dtype=inputs["x"].dtype, device=inputs["x"].device)
    c_pad = torch.empty((bsz, n_clusters, d_pad), dtype=inputs["centroids"].dtype, device=inputs["centroids"].device)
    _SCRATCH_CACHE[key] = (x_pad, c_pad)
    return x_pad, c_pad


def _partial_buffers(inputs: dict[str, Any], total_work: int) -> tuple[Any, Any]:
    torch = _torch()
    device_index = _device_index(inputs["x"].device)
    key = (
        device_index,
        int(inputs["x"].data_ptr()),
        int(inputs["centroids"].data_ptr()),
        int(inputs["B"]),
        int(inputs["N"]),
        int(inputs["K"]),
        int(total_work),
    )
    cached = _PARTIAL_CACHE.get(key)
    if cached is not None:
        return cached
    scores = torch.empty((total_work, BLOCK_N), dtype=torch.float32, device=inputs["x"].device)
    indices = torch.empty((total_work, BLOCK_N), dtype=torch.int32, device=inputs["x"].device)
    _PARTIAL_CACHE[key] = (scores, indices)
    return scores, indices


def _partial64_buffers(inputs: dict[str, Any], total_work: int, group_k_tiles: int) -> tuple[Any, Any]:
    torch = _torch()
    device_index = _device_index(inputs["x"].device)
    key = (
        device_index,
        int(inputs["x"].data_ptr()),
        int(inputs["centroids"].data_ptr()),
        int(inputs["B"]),
        int(inputs["N"]),
        int(inputs["K"]),
        int(total_work),
        int(group_k_tiles),
    )
    cached = _PARTIAL64_CACHE.get(key)
    if cached is not None:
        return cached
    scores = torch.empty((total_work, BLOCK_N64), dtype=torch.float32, device=inputs["x"].device)
    indices = torch.empty((total_work, BLOCK_N64), dtype=torch.int32, device=inputs["x"].device)
    _PARTIAL64_CACHE[key] = (scores, indices)
    return scores, indices


def _partial_key_buffer(inputs: dict[str, Any], total_work: int, group_k_tiles: int):
    torch = _torch()
    device_index = _device_index(inputs["x"].device)
    key = (
        device_index,
        int(inputs["x"].data_ptr()),
        int(inputs["centroids"].data_ptr()),
        int(inputs["B"]),
        int(inputs["N"]),
        int(inputs["K"]),
        int(total_work),
        int(group_k_tiles),
    )
    cached = _PARTIAL_KEY_CACHE.get(key)
    if cached is not None:
        return cached
    keys = torch.empty((total_work, BLOCK_N64), dtype=torch.int64, device=inputs["x"].device)
    _PARTIAL_KEY_CACHE[key] = keys
    return keys


def _make_tmaps(inputs: dict[str, Any], *, tma_width: int, block_width: int) -> tuple[Any, Any]:
    device_index = _device_index(inputs["x"].device)
    bsz = int(inputs["B"])
    n_points = int(inputs["N"])
    n_clusters = int(inputs["K"])
    tmap_x = _create_tensor_map_3d(
        inputs["x"].data_ptr(),
        bsz * n_points,
        BLOCK_N,
        tma_width,
        block_width,
        device_index=device_index,
    )
    tmap_c = _create_tensor_map_3d(
        inputs["centroids"].data_ptr(),
        bsz * n_clusters,
        BLOCK_K,
        tma_width,
        block_width,
        device_index=device_index,
    )
    return tmap_x, tmap_c


def _make_tmaps_blockn64(inputs: dict[str, Any]) -> tuple[Any, Any]:
    device_index = _device_index(inputs["x"].device)
    bsz = int(inputs["B"])
    n_points = int(inputs["N"])
    n_clusters = int(inputs["K"])
    dim = int(inputs["D"])
    tmap_x = _create_tensor_map_3d(
        inputs["x"].data_ptr(),
        bsz * n_points,
        BLOCK_N64,
        dim,
        FEAT_CHUNK,
        device_index=device_index,
    )
    tmap_c = _create_tensor_map_3d(
        inputs["centroids"].data_ptr(),
        bsz * n_clusters,
        BLOCK_K,
        dim,
        FEAT_CHUNK,
        device_index=device_index,
    )
    return tmap_x, tmap_c


def _launch_pack(
    kernel_name: str,
    inputs: dict[str, Any],
    x_pad: Any,
    c_pad: Any,
    d_pad: int,
    *,
    include_d_pad_arg: bool,
    arch: str | None,
    stream: Any,
    timeout_ms: float | None,
) -> None:
    bsz = int(inputs["B"])
    n_points = int(inputs["N"])
    dim = int(inputs["D"])
    n_clusters = int(inputs["K"])
    total_x_pad = bsz * n_points * d_pad
    total_c_pad = bsz * n_clusters * d_pad
    work_items = max(total_x_pad, total_c_pad)
    grid_x = min((work_items + PACK_THREADS - 1) // PACK_THREADS, PACK_GRID_CAP)
    args: list[Any] = [
        inputs["x"],
        inputs["centroids"],
        x_pad,
        c_pad,
        bsz,
        n_points,
        dim,
        n_clusters,
    ]
    if include_d_pad_arg:
        args.append(d_pad)
    args.extend([total_x_pad, total_c_pad])
    _launch(
        _kernels.get_kernel(kernel_name),
        *args,
        grid=(grid_x, 1, 1),
        block=(PACK_THREADS, 1, 1),
        arch=arch,
        stream=stream,
        timeout_ms=timeout_ms,
    )
    _retain_launch_tensors(x_pad, c_pad)


def _launch_tma_assign(
    kernel_name: str,
    inputs: dict[str, Any],
    *,
    logical_d: int,
    tma_width: int,
    block_width: int,
    paired_grid: bool = False,
    capped_grid: bool = False,
    arch: str | None,
    stream: Any,
    timeout_ms: float | None,
) -> None:
    bsz = int(inputs["B"])
    n_points = int(inputs["N"])
    n_clusters = int(inputs["K"])
    num_n_tiles = n_points // BLOCK_N
    k_tiles = n_clusters // BLOCK_K
    if paired_grid:
        total_tiles = bsz * (num_n_tiles // 2)
    else:
        total_tiles = bsz * num_n_tiles
    grid_x = min(total_tiles, PAIRED_GRID_CAP) if capped_grid or paired_grid else total_tiles
    tmap_x, tmap_c = _make_tmaps(inputs, tma_width=tma_width, block_width=block_width)
    _launch(
        _kernels.get_kernel(kernel_name),
        inputs["x_sq"],
        inputs["c_sq"],
        inputs["out"],
        tmap_x,
        tmap_c,
        bsz,
        n_points,
        logical_d,
        n_clusters,
        num_n_tiles,
        k_tiles,
        grid=(grid_x, 1, 1),
        block=(int(_kernels.get_kernel(kernel_name).spec.threads), 1, 1),
        arch=arch,
        stream=stream,
        timeout_ms=timeout_ms,
    )
    _retain_launch_tensors(tmap_x, tmap_c)


def _launch_microdim(
    inputs: dict[str, Any],
    *,
    arch: str | None,
    stream: Any,
    timeout_ms: float | None,
) -> str:
    bsz = int(inputs["B"])
    n_points = int(inputs["N"])
    dim = int(inputs["D"])
    n_clusters = int(inputs["K"])
    num_n_tiles = n_points // BLOCK_N
    k_tiles = n_clusters // BLOCK_K
    if n_clusters == 512 and n_points <= 2432:
        _launch(
            _kernels.get_kernel("kmeans_microdim_direct"),
            inputs["x"],
            inputs["centroids"],
            inputs["x_sq"],
            inputs["c_sq"],
            inputs["out"],
            bsz,
            n_points,
            dim,
            n_clusters,
            num_n_tiles,
            k_tiles,
            grid=(bsz * num_n_tiles, 1, 1),
            block=(int(_kernels.get_kernel("kmeans_microdim_direct").spec.threads), 1, 1),
            arch=arch,
            stream=stream,
            timeout_ms=timeout_ms,
        )
        return "microdim_direct_9c0d_v1"

    x_pad, c_pad = _scratch_buffers(inputs, 64)
    _launch_pack(
        "kmeans_microdim_pack",
        inputs,
        x_pad,
        c_pad,
        64,
        include_d_pad_arg=False,
        arch=arch,
        stream=stream,
        timeout_ms=timeout_ms,
    )
    route_inputs = {**inputs, "x": x_pad, "centroids": c_pad}
    _launch_tma_assign(
        "kmeans_microdim_main",
        route_inputs,
        logical_d=dim,
        tma_width=64,
        block_width=64,
        arch=arch,
        stream=stream,
        timeout_ms=timeout_ms,
    )
    return "microdim_pack_6cd2_v1"


def _launch_lowdim(
    inputs: dict[str, Any],
    *,
    arch: str | None,
    stream: Any,
    timeout_ms: float | None,
) -> None:
    dim = int(inputs["D"])
    x_pad, c_pad = _scratch_buffers(inputs, 128)
    _launch_pack(
        "kmeans_lowdim_pack",
        inputs,
        x_pad,
        c_pad,
        128,
        include_d_pad_arg=False,
        arch=arch,
        stream=stream,
        timeout_ms=timeout_ms,
    )
    route_inputs = {**inputs, "x": x_pad, "centroids": c_pad}
    _launch_tma_assign(
        "kmeans_lowdim_main",
        route_inputs,
        logical_d=dim,
        tma_width=128,
        block_width=128,
        arch=arch,
        stream=stream,
        timeout_ms=timeout_ms,
    )


def _launch_gap_pad(
    inputs: dict[str, Any],
    decision: RouteDecision,
    *,
    arch: str | None,
    stream: Any,
    timeout_ms: float | None,
) -> str:
    d_pad = int(decision.padded_dim or _GAP_PAD[int(inputs["D"])])
    x_pad, c_pad = _scratch_buffers(inputs, d_pad)
    _launch_pack(
        "kmeans_gap_pad_pack",
        inputs,
        x_pad,
        c_pad,
        d_pad,
        include_d_pad_arg=True,
        arch=arch,
        stream=stream,
        timeout_ms=timeout_ms,
    )
    route_inputs = {**inputs, "x": x_pad, "centroids": c_pad, "D": d_pad}
    if d_pad == 64:
        _launch_tma_assign(
            "kmeans_d64_direct",
            route_inputs,
            logical_d=64,
            tma_width=64,
            block_width=64,
            arch=arch,
            stream=stream,
            timeout_ms=timeout_ms,
        )
        return "d64_direct_single64_1p2gap_9f2a_v1"
    if d_pad == 128:
        if _use_small_grid(int(inputs["N"]), int(inputs["K"])):
            _launch_tma_assign("kmeans_v10", route_inputs, logical_d=128, tma_width=128, block_width=128, arch=arch, stream=stream, timeout_ms=timeout_ms)
            return "small_grid_single_tile_v10"
        if (int(inputs["N"]) // BLOCK_N) % 2 == 0:
            _launch_tma_assign("kmeans_v15", route_inputs, logical_d=128, tma_width=128, block_width=128, paired_grid=True, arch=arch, stream=stream, timeout_ms=timeout_ms)
            return "paired_large_v15"
        _launch_tma_assign("kmeans_v10", route_inputs, logical_d=128, tma_width=128, block_width=128, arch=arch, stream=stream, timeout_ms=timeout_ms)
        return "aligned_v10_fallback"
    if d_pad == 256:
        _launch_tma_assign("kmeans_d256_single", route_inputs, logical_d=256, tma_width=256, block_width=256, arch=arch, stream=stream, timeout_ms=timeout_ms)
        return "d256_single_repeated_mma_v1"
    if d_pad in {320, 384, 448, 512}:
        num_n_tiles = int(inputs["N"]) // BLOCK_N
        k_tiles = int(inputs["K"]) // BLOCK_K
        if _use_highd_splitk(dim=d_pad, num_n_tiles=num_n_tiles, k_tiles=k_tiles):
            _launch_highd_splitk(route_inputs, arch=arch, stream=stream, timeout_ms=timeout_ms)
            return "highd_splitk_8de8_v1"
        _launch_tma_assign(
            "kmeans_highd_splitd",
            route_inputs,
            logical_d=d_pad,
            tma_width=d_pad,
            block_width=FEAT_CHUNK,
            capped_grid=True,
            arch=arch,
            stream=stream,
            timeout_ms=timeout_ms,
        )
        return "highd_splitd_single_tile_6fcf_v1"
    raise ValueError(f"no padded route for D_PAD={d_pad}")


def _launch_pad192(
    inputs: dict[str, Any],
    *,
    arch: str | None,
    stream: Any,
    timeout_ms: float | None,
) -> str:
    x_pad, c_pad = _scratch_buffers(inputs, 192)
    _launch_pack(
        "kmeans_pad192_pack",
        inputs,
        x_pad,
        c_pad,
        192,
        include_d_pad_arg=False,
        arch=arch,
        stream=stream,
        timeout_ms=timeout_ms,
    )
    route_inputs = {**inputs, "x": x_pad, "centroids": c_pad, "D": 192}
    if (int(inputs["N"]) // BLOCK_N) % 2 != 0 or _use_small_grid(int(inputs["N"]), int(inputs["K"])):
        _launch_tma_assign("kmeans_d192_single", route_inputs, logical_d=192, tma_width=192, block_width=192, arch=arch, stream=stream, timeout_ms=timeout_ms)
        return "d192_single_repeated_mma_v1"
    _launch_tma_assign("kmeans_d192_splitd", route_inputs, logical_d=192, tma_width=192, block_width=192, paired_grid=True, arch=arch, stream=stream, timeout_ms=timeout_ms)
    return "d192_paired_repeated_mma_v1"


def _launch_d160_padded(
    inputs: dict[str, Any],
    *,
    arch: str | None,
    stream: Any,
    timeout_ms: float | None,
) -> None:
    x_pad, c_pad = _scratch_buffers(inputs, 192)
    _launch_pack(
        "kmeans_d160_padded_pack",
        inputs,
        x_pad,
        c_pad,
        192,
        include_d_pad_arg=False,
        arch=arch,
        stream=stream,
        timeout_ms=timeout_ms,
    )
    route_inputs = {**inputs, "x": x_pad, "centroids": c_pad}
    _launch_tma_assign(
        "kmeans_d192_single",
        route_inputs,
        logical_d=160,
        tma_width=192,
        block_width=192,
        arch=arch,
        stream=stream,
        timeout_ms=timeout_ms,
    )


def _launch_highd_splitk(
    inputs: dict[str, Any],
    *,
    arch: str | None,
    stream: Any,
    timeout_ms: float | None,
) -> None:
    bsz = int(inputs["B"])
    n_points = int(inputs["N"])
    dim = int(inputs["D"])
    n_clusters = int(inputs["K"])
    num_n_tiles = n_points // BLOCK_N
    k_tiles = n_clusters // BLOCK_K
    total_work = bsz * num_n_tiles * k_tiles
    partial_scores, partial_indices = _partial_buffers(inputs, total_work)
    tmap_x, tmap_c = _make_tmaps(inputs, tma_width=dim, block_width=FEAT_CHUNK)
    _launch(
        _kernels.get_kernel("kmeans_highd_splitk_partial"),
        inputs["c_sq"],
        partial_scores,
        partial_indices,
        tmap_x,
        tmap_c,
        bsz,
        n_points,
        dim,
        n_clusters,
        num_n_tiles,
        k_tiles,
        grid=(min(total_work, SPLITK_GRID_CAP), 1, 1),
        block=(int(_kernels.get_kernel("kmeans_highd_splitk_partial").spec.threads), 1, 1),
        arch=arch,
        stream=stream,
        timeout_ms=timeout_ms,
    )
    _launch(
        _kernels.get_kernel("kmeans_highd_splitk_reduce"),
        partial_scores,
        partial_indices,
        inputs["out"],
        bsz,
        n_points,
        n_clusters,
        num_n_tiles,
        k_tiles,
        grid=(min(bsz * num_n_tiles, SPLITK_GRID_CAP), 1, 1),
        block=(int(_kernels.get_kernel("kmeans_highd_splitk_reduce").spec.threads), 1, 1),
        arch=arch,
        stream=stream,
        timeout_ms=timeout_ms,
    )
    _retain_launch_tensors(partial_scores, partial_indices, tmap_x, tmap_c)


def _launch_blockn64_splitk(
    inputs: dict[str, Any],
    *,
    partial_kernel_name: str,
    reduce_kernel_name: str,
    group_k_tiles: int,
    arch: str | None,
    stream: Any,
    timeout_ms: float | None,
) -> None:
    bsz = int(inputs["B"])
    n_points = int(inputs["N"])
    dim = int(inputs["D"])
    n_clusters = int(inputs["K"])
    num_n_tiles = n_points // BLOCK_N64
    k_tiles = n_clusters // BLOCK_K
    if k_tiles % group_k_tiles != 0:
        raise ValueError(f"K tiles must be divisible by grouped K tiles {group_k_tiles}, got {k_tiles}")
    k_slices = k_tiles // group_k_tiles
    total_work = bsz * num_n_tiles * k_slices
    partial_scores, partial_indices = _partial64_buffers(inputs, total_work, group_k_tiles)
    tmap_x, tmap_c = _make_tmaps_blockn64(inputs)
    partial_kernel = _kernels.get_kernel(partial_kernel_name)
    reduce_kernel = _kernels.get_kernel(reduce_kernel_name)
    _launch(
        partial_kernel,
        inputs["c_sq"],
        partial_scores,
        partial_indices,
        tmap_x,
        tmap_c,
        bsz,
        n_points,
        dim,
        n_clusters,
        num_n_tiles,
        k_tiles,
        k_slices,
        grid=(min(total_work, SPLITK_GRID_CAP), 1, 1),
        block=(int(partial_kernel.spec.threads), 1, 1),
        arch=arch,
        stream=stream,
        timeout_ms=timeout_ms,
    )
    _launch(
        reduce_kernel,
        partial_scores,
        partial_indices,
        inputs["out"],
        bsz,
        n_points,
        n_clusters,
        num_n_tiles,
        k_slices,
        grid=(min(bsz * num_n_tiles, SPLITK_GRID_CAP), 1, 1),
        block=(int(reduce_kernel.spec.threads), 1, 1),
        arch=arch,
        stream=stream,
        timeout_ms=timeout_ms,
    )
    _retain_launch_tensors(partial_scores, partial_indices, tmap_x, tmap_c)


def _launch_packed_key_pair(
    inputs: dict[str, Any],
    *,
    partial_kernel_name: str,
    reduce_kernel_name: str,
    group_k_tiles: int,
    producer_grid_cap: int,
    arch: str | None,
    stream: Any,
    timeout_ms: float | None,
) -> None:
    bsz = int(inputs["B"])
    n_points = int(inputs["N"])
    dim = int(inputs["D"])
    n_clusters = int(inputs["K"])
    num_n_tiles = n_points // BLOCK_N64
    k_tiles = n_clusters // BLOCK_K
    if k_tiles % group_k_tiles != 0:
        raise ValueError(f"K tiles must be divisible by grouped K tiles {group_k_tiles}, got {k_tiles}")
    k_slices = k_tiles // group_k_tiles
    total_work = bsz * num_n_tiles * k_slices
    partial_keys = _partial_key_buffer(inputs, total_work, group_k_tiles)
    tmap_x, tmap_c = _make_tmaps_blockn64(inputs)
    partial_kernel = _kernels.get_kernel(partial_kernel_name)
    reduce_kernel = _kernels.get_kernel(reduce_kernel_name)
    _launch(
        partial_kernel,
        inputs["c_sq"],
        partial_keys,
        tmap_x,
        tmap_c,
        bsz,
        n_points,
        dim,
        n_clusters,
        num_n_tiles,
        k_tiles,
        k_slices,
        grid=(min(total_work, producer_grid_cap), 1, 1),
        block=(int(partial_kernel.spec.threads), 1, 1),
        arch=arch,
        stream=stream,
        timeout_ms=timeout_ms,
    )
    _launch(
        reduce_kernel,
        partial_keys,
        inputs["out"],
        bsz,
        n_points,
        n_clusters,
        num_n_tiles,
        k_slices,
        grid=(min(bsz * num_n_tiles, SPLITK_GRID_CAP), 1, 1),
        block=(int(reduce_kernel.spec.threads), 1, 1),
        arch=arch,
        stream=stream,
        timeout_ms=timeout_ms,
    )
    _retain_launch_tensors(partial_keys, tmap_x, tmap_c)


def _launch_no_padding_portfolio_r63(
    inputs: dict[str, Any],
    *,
    arch: str | None,
    stream: Any,
    timeout_ms: float | None,
) -> str:
    bsz = int(inputs["B"])
    n_points = int(inputs["N"])
    dim = int(inputs["D"])
    n_clusters = int(inputs["K"])
    if not _use_no_padding_portfolio_r63(B=bsz, N=n_points, D=dim, K=n_clusters):
        raise ValueError(
            "R63 no-padding portfolio only supports exact high-D shapes, "
            f"got B={bsz}, N={n_points}, K={n_clusters}, D={dim}"
        )
    if bsz == 1 and n_points == 512 and dim == 512 and n_clusters == 4096:
        _launch_blockn64_splitk(
            inputs,
            partial_kernel_name="kmeans_highd_splitk_blockn64_g1r4_partial",
            reduce_kernel_name="kmeans_highd_splitk_blockn64_g1r4_reduce",
            group_k_tiles=SPLITK_GROUP_K_TILES_G1,
            arch=arch,
            stream=stream,
            timeout_ms=timeout_ms,
        )
        return "highd_splitk_blockn64_g1r4_streamdep_r63_v1"
    if bsz == 1 and n_points == 2048 and dim == 448 and n_clusters == 4096:
        _launch_packed_key_pair(
            inputs,
            partial_kernel_name="kmeans_highd_paired_xreuse_r47_partial",
            reduce_kernel_name="kmeans_highd_paired_ownerreduce_r39_reduce1",
            group_k_tiles=SPLITK_GROUP_K_TILES_G2,
            producer_grid_cap=PACKED_KEY_PRODUCER_GRID_CAP,
            arch=arch,
            stream=stream,
            timeout_ms=timeout_ms,
        )
        return "r47_dualtmem_xreuse_gridcap160_plus_r39_reduce1"
    if bsz == 1 and n_points == 2048 and dim == 512 and n_clusters == 4096:
        _launch_packed_key_pair(
            inputs,
            partial_kernel_name="kmeans_highd_paired_packedpartial_r2_partial",
            reduce_kernel_name="kmeans_highd_paired_packedpartial_r2_reduce",
            group_k_tiles=SPLITK_GROUP_K_TILES_G2,
            producer_grid_cap=PACKED_KEY_PRODUCER_GRID_CAP,
            arch=arch,
            stream=stream,
            timeout_ms=timeout_ms,
        )
        return "highd_paired_packedpartial_gridcap160_0194_v1"
    _launch_blockn64_splitk(
        inputs,
        partial_kernel_name="kmeans_highd_splitk_blockn64_g2r4_partial",
        reduce_kernel_name="kmeans_highd_splitk_blockn64_g2r4_reduce",
        group_k_tiles=SPLITK_GROUP_K_TILES_G2,
        arch=arch,
        stream=stream,
        timeout_ms=timeout_ms,
    )
    return "highd_splitk_blockn64_g2r4_streamdep_4f2c_v1"


def _launch_route(
    inputs: dict[str, Any],
    decision: RouteDecision,
    *,
    arch: str | None,
    stream: Any,
    timeout_ms: float | None,
) -> dict[str, Any]:
    child_route = decision.child_route
    dim = int(inputs["D"])
    n_points = int(inputs["N"])
    n_clusters = int(inputs["K"])

    if decision.route_id == "microdim_hybrid_9c0d_v1":
        child_route = _launch_microdim(inputs, arch=arch, stream=stream, timeout_ms=timeout_ms)
    elif decision.route_id == "lowdim_e50c_v1":
        _launch_lowdim(inputs, arch=arch, stream=stream, timeout_ms=timeout_ms)
    elif decision.route_id == "gap_pad_to_supported_seed_v1":
        child_route = _launch_gap_pad(inputs, decision, arch=arch, stream=stream, timeout_ms=timeout_ms)
    elif decision.route_id == "d144_d160_d176_pad192_tail_repair_f9b2_v1":
        child_route = _launch_pad192(inputs, arch=arch, stream=stream, timeout_ms=timeout_ms)
    elif decision.route_id == "d160_padded_single_repeated_mma_v2":
        _launch_d160_padded(inputs, arch=arch, stream=stream, timeout_ms=timeout_ms)
    elif decision.route_id == "highd_splitk_8de8_v1":
        _launch_highd_splitk(inputs, arch=arch, stream=stream, timeout_ms=timeout_ms)
    elif decision.route_id == "flash_kmeans_assign_no_padding_portfolio_r63_g1_d512n512_v1":
        child_route = _launch_no_padding_portfolio_r63(inputs, arch=arch, stream=stream, timeout_ms=timeout_ms)
    else:
        kernel_name = decision.kernel
        if kernel_name is None:
            raise ValueError(f"route {decision.route_id!r} does not name a kernel")
        if kernel_name == "kmeans_v15":
            _launch_tma_assign(kernel_name, inputs, logical_d=dim, tma_width=dim, block_width=dim, paired_grid=True, arch=arch, stream=stream, timeout_ms=timeout_ms)
        elif kernel_name == "kmeans_d192_splitd":
            _launch_tma_assign(kernel_name, inputs, logical_d=dim, tma_width=dim, block_width=dim, paired_grid=True, arch=arch, stream=stream, timeout_ms=timeout_ms)
        elif kernel_name == "kmeans_highd_splitd":
            _launch_tma_assign(kernel_name, inputs, logical_d=dim, tma_width=dim, block_width=FEAT_CHUNK, capped_grid=True, arch=arch, stream=stream, timeout_ms=timeout_ms)
        else:
            _launch_tma_assign(kernel_name, inputs, logical_d=dim, tma_width=dim, block_width=dim, arch=arch, stream=stream, timeout_ms=timeout_ms)

    return {
        "selected_route": decision.route_id,
        "child_route": child_route,
        "B": int(inputs["B"]),
        "N": n_points,
        "D": dim,
        "K": n_clusters,
    }


def flash_kmeans_assign(
    x: Any,
    centroids: Any,
    *,
    out: Any | None = None,
    x_sq: Any | None = None,
    c_sq: Any | None = None,
    arch: str | None = None,
    stream: Any = None,
    timeout_ms: float | None = None,
    return_info: bool = False,
):
    """Assign each point in ``x`` to the nearest centroid.

    Args:
        x: Contiguous CUDA BF16 tensor with shape ``[B, N, D]``.
        centroids: Contiguous CUDA BF16 tensor with shape ``[B, K, D]``.
        out: Optional contiguous CUDA int32 output tensor with shape ``[B, N]``.
        x_sq: Optional precomputed ``(x.float() ** 2).sum(-1)`` tensor.
        c_sq: Optional precomputed ``(centroids.float() ** 2).sum(-1)`` tensor.
        arch: Optional NVRTC architecture, for example ``"sm_100a"``.
        stream: Optional PyTorch CUDA stream.
        timeout_ms: Optional driver-side completion timeout per launched kernel.
        return_info: When true, return ``(cluster_ids, info)``.

    Returns:
        A CUDA int32 tensor of shape ``[B, N]`` containing cluster indices, or
        ``(cluster_ids, info)`` when ``return_info=True``.
    """
    bsz, n_points, dim, n_clusters = _validate_pair(x, centroids)
    output = _allocate_output(x, out)
    if x_sq is None:
        x_sq = (x.float() ** 2).sum(-1).contiguous()
    else:
        x_sq = _validate_sq_tensor("x_sq", x_sq, (bsz, n_points), x.device)
    if c_sq is None:
        c_sq = (centroids.float() ** 2).sum(-1).contiguous()
    else:
        c_sq = _validate_sq_tensor("c_sq", c_sq, (bsz, n_clusters), x.device)
    inputs = {
        "x": x,
        "centroids": centroids,
        "x_sq": x_sq,
        "c_sq": c_sq,
        "out": output,
        "B": bsz,
        "N": n_points,
        "D": dim,
        "K": n_clusters,
        "dtype": "bfloat16",
    }
    decision = select_flash_kmeans_route(B=bsz, N=n_points, D=dim, K=n_clusters)
    info = _launch_route(inputs, decision, arch=arch, stream=stream, timeout_ms=timeout_ms)
    _retain_launch_tensors(x_sq, c_sq, output)
    if return_info:
        return output, info
    return output
