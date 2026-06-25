from __future__ import annotations

import ctypes
import math
from typing import Any

from . import kernels as _kernels


BLOCK_Q = 128
BLOCK_M = 64
FEAT_D = 128
CTA_GROUP = 2
GRID_DIM_DEFAULT = 2048
MEDIUM_SPLITS = 4
RAG_SPLITS = 7
K12_MID_SPLITS = 8
K30_SMALL_SPLITS = 8
K30_SMALL_SHAPE_MAX = 512
SMALL_SHAPE_MAX = 512
K32_BUCKETS = (12, 16, 20, 25, 30, 32)
SEARCH_K_MAX = 10
SEARCH_THREADS = 256
SEARCH_NUM_WARPS = SEARCH_THREADS // 32
SEARCH_BLOCK_M = 512
SEARCH_PARTIAL_ELEMS_PER_TILE = SEARCH_NUM_WARPS * SEARCH_K_MAX
SEARCH_Q1_BLOCK_M = 256
SEARCH_Q1_MERGE_TILES_PER_GROUP = 64
SEARCH_Q1_MERGE_WARPS = SEARCH_THREADS // 32
_COMPILE_OPTIONS = ["--use_fast_math"]
_CU_TENSOR_MAP_SIZE = 128
_PARTIAL_CACHE: dict[tuple[str, int, int, int, int, int], tuple[Any, Any]] = {}
_SEARCH_PARTIAL_CACHE: dict[tuple[str, int, int, int, int, int, int], tuple[Any, Any]] = {}
_SEARCH_Q1_PARTIAL_CACHE: dict[tuple[str, int, int, int, int, int], tuple[Any, Any]] = {}
_TMAP_CACHE: dict[tuple[int, int, int, int, int, int], Any] = {}
_LIVE_LAUNCH_TENSORS: list[tuple[Any, ...]] = []


def _torch():
    import torch

    return torch


def _driver():
    from cuda.bindings import driver

    return driver


def _tmap_to_device(tmap: Any, *, device_index: int):
    torch = _torch()
    host_ptr = tmap.getPtr()
    tmap_bytes = bytes((ctypes.c_ubyte * _CU_TENSOR_MAP_SIZE).from_address(host_ptr))
    tmap_host = torch.frombuffer(bytearray(tmap_bytes), dtype=torch.uint8)
    with torch.cuda.device(device_index):
        tmap_device = torch.empty(_CU_TENSOR_MAP_SIZE, dtype=torch.uint8, device="cuda")
        tmap_device.copy_(tmap_host)
    return tmap_device


def _create_tensor_map_3d_oob_zero(
    data_ptr: int,
    global_height: int,
    shared_height: int,
    width: int,
    block_width: int,
    *,
    device_index: int,
):
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
        driver.CUtensorMapFloatOOBfill.CU_TENSOR_MAP_FLOAT_OOB_FILL_NAN_REQUEST_ZERO_FMA,
    )
    if err != 0:
        raise RuntimeError(f"cuTensorMapEncodeTiled (3D, OOB zero) failed: CUresult={err}")
    cached = _tmap_to_device(tmap, device_index=device_index)
    _TMAP_CACHE[key] = cached
    return cached


def _device_index(device: Any) -> int:
    torch = _torch()
    index = device.index
    if index is None and device.type == "cuda":
        index = torch.cuda.current_device()
    return int(index or 0)


def _partial_buffers(*, split_count: int, bsz: int, n_query: int, top_k: int, device: Any) -> tuple[Any, Any]:
    torch = _torch()
    index = _device_index(device)
    key = (device.type, index, int(split_count), int(bsz), int(n_query), int(top_k))
    cached = _PARTIAL_CACHE.get(key)
    if cached is None:
        cached = (
            torch.empty((split_count, bsz, n_query, top_k), dtype=torch.float32, device=device),
            torch.empty((split_count, bsz, n_query, top_k), dtype=torch.int32, device=device),
        )
        _PARTIAL_CACHE[key] = cached
    return cached


def _search_partial_buffers(*, bsz: int, n_query: int, num_m_tiles: int, top_k: int, device: Any) -> tuple[Any, Any]:
    torch = _torch()
    index = _device_index(device)
    key = (device.type, index, int(bsz), int(n_query), int(num_m_tiles), int(top_k), SEARCH_K_MAX)
    cached = _SEARCH_PARTIAL_CACHE.get(key)
    if cached is None:
        shape = (bsz, n_query, num_m_tiles, SEARCH_NUM_WARPS, SEARCH_K_MAX)
        cached = (
            torch.empty(shape, dtype=torch.float32, device=device),
            torch.empty(shape, dtype=torch.int32, device=device),
        )
        _SEARCH_PARTIAL_CACHE[key] = cached
    return cached


def _search_q1_partial_buffers(*, bsz: int, num_m_tiles: int, top_k: int, device: Any) -> tuple[Any, Any]:
    torch = _torch()
    index = _device_index(device)
    key = (device.type, index, int(bsz), int(num_m_tiles), int(top_k), SEARCH_K_MAX)
    cached = _SEARCH_Q1_PARTIAL_CACHE.get(key)
    if cached is None:
        shape = (bsz, num_m_tiles, SEARCH_K_MAX)
        cached = (
            torch.empty(shape, dtype=torch.float32, device=device),
            torch.empty(shape, dtype=torch.int32, device=device),
        )
        _SEARCH_Q1_PARTIAL_CACHE[key] = cached
    return cached


def _retain_launch_tensors(*tensors: Any) -> None:
    _LIVE_LAUNCH_TENSORS.append(tuple(tensors))
    del _LIVE_LAUNCH_TENSORS[:-16]


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
    if int(tensor.shape[2]) != FEAT_D:
        raise ValueError(
            f"{name} must have D={FEAT_D}, got {int(tensor.shape[2])}. "
            "The exported FlashKNN kernels are specialized for feature dimension 128."
        )
    if not tensor.is_contiguous():
        raise ValueError(f"{name} must be contiguous")


def _validate_pair(query: Any, database: Any, k: int) -> tuple[int, int, int, int]:
    _validate_tensor("query", query)
    _validate_tensor("database", database)
    if query.device != database.device:
        raise ValueError("query and database must be on the same CUDA device")
    if int(query.shape[0]) != int(database.shape[0]):
        raise ValueError("query and database must have the same batch dimension")
    if k <= 0:
        raise ValueError(f"k must be positive, got {k}")
    if k > int(database.shape[1]):
        raise ValueError(f"k must be <= database length, got k={k}, M={int(database.shape[1])}")
    return int(query.shape[0]), int(query.shape[1]), int(database.shape[1]), int(query.shape[2])


def _allocate_outputs(query: Any, k: int, out: tuple[Any, Any] | None) -> tuple[Any, Any]:
    torch = _torch()
    bsz, n_query = int(query.shape[0]), int(query.shape[1])
    expected_dists = (bsz, n_query, k)
    if out is None:
        return (
            torch.empty(expected_dists, dtype=torch.float32, device=query.device),
            torch.empty(expected_dists, dtype=torch.int32, device=query.device),
        )
    if not isinstance(out, tuple) or len(out) != 2:
        raise TypeError("out must be a (distances, indices) tuple")
    out_dists, out_indices = out
    if out_dists.shape != expected_dists or out_indices.shape != expected_dists:
        raise ValueError(f"out tensors must have shape {expected_dists}")
    if out_dists.dtype is not torch.float32:
        raise ValueError("out distances must have dtype torch.float32")
    if out_indices.dtype is not torch.int32:
        raise ValueError("out indices must have dtype torch.int32")
    if out_dists.device != query.device or out_indices.device != query.device:
        raise ValueError("out tensors must be on the query device")
    if not out_dists.is_contiguous() or not out_indices.is_contiguous():
        raise ValueError("out tensors must be contiguous")
    return out_dists, out_indices


def _make_build_inputs(database: Any, k: int, out: tuple[Any, Any] | None) -> dict[str, Any]:
    bsz, n_query, n_database, dim = _validate_pair(database, database, k)
    if n_query != n_database:
        raise ValueError("knn_build requires database shape [B, N, D]")
    out_dists, out_indices = _allocate_outputs(database, k, out)
    sq = (database.float() ** 2).sum(-1).contiguous()
    return {
        "query": database,
        "database": database,
        "query_sq": sq,
        "database_sq": sq,
        "out_dists": out_dists,
        "out_indices": out_indices,
        "B": bsz,
        "Q": n_query,
        "M": n_database,
        "D": dim,
        "K": int(k),
        "build": True,
    }


def _launch(
    kernel: _kernels.ExportedKernel,
    *args: Any,
    grid: tuple[int, int, int],
    block: tuple[int, int, int] | None = None,
    arch: str | None = None,
    stream: Any = None,
    timeout_ms: float | None = None,
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


def _launch_stage1_merge(
    inputs: dict[str, Any],
    *,
    split_count: int,
    stage1_kernel: _kernels.ExportedKernel,
    merge_kernel: _kernels.ExportedKernel,
    merge_with_k_arg: bool,
    arch: str | None,
    stream: Any,
    timeout_ms: float | None,
) -> None:
    query = inputs["query"]
    database = inputs["database"]
    bsz = int(inputs["B"])
    n_query = int(inputs["Q"])
    n_database = int(inputs["M"])
    dim = int(inputs["D"])
    top_k = int(inputs["K"])

    num_q_tiles = (n_query + BLOCK_Q - 1) // BLOCK_Q
    num_q_tile_pairs = (num_q_tiles + CTA_GROUP - 1) // CTA_GROUP
    num_db_tiles = (n_database + BLOCK_M - 1) // BLOCK_M
    db_tiles_per_split = (num_db_tiles + split_count - 1) // split_count
    total_work = bsz * num_q_tile_pairs * split_count
    stage1_grid = min(total_work * CTA_GROUP, GRID_DIM_DEFAULT)
    merge_threads = int(merge_kernel.spec.threads)
    merge_grid = min((bsz * n_query + merge_threads - 1) // merge_threads, GRID_DIM_DEFAULT)
    partial_dists, partial_indices = _partial_buffers(
        split_count=split_count,
        bsz=bsz,
        n_query=n_query,
        top_k=top_k,
        device=query.device,
    )

    device_index = _device_index(query.device)
    tmap_query = _create_tensor_map_3d_oob_zero(
        query.data_ptr(),
        bsz * n_query,
        BLOCK_Q,
        dim,
        dim,
        device_index=device_index,
    )
    tmap_database = _create_tensor_map_3d_oob_zero(
        database.data_ptr(),
        bsz * n_database,
        BLOCK_M,
        dim,
        dim,
        device_index=device_index,
    )

    _launch(
        stage1_kernel,
        inputs["query_sq"],
        inputs["database_sq"],
        partial_dists,
        partial_indices,
        tmap_query,
        tmap_database,
        bsz,
        n_query,
        n_database,
        top_k,
        num_q_tile_pairs,
        db_tiles_per_split,
        split_count,
        total_work,
        grid=(stage1_grid, 1, 1),
        block=(int(stage1_kernel.spec.threads), 1, 1),
        arch=arch,
        stream=stream,
        timeout_ms=timeout_ms,
    )

    merge_args: list[Any] = [
        partial_dists,
        partial_indices,
        inputs["out_dists"],
        inputs["out_indices"],
    ]
    if merge_with_k_arg:
        merge_args.append(top_k)
    merge_args.append(bsz * n_query)
    _launch(
        merge_kernel,
        *merge_args,
        grid=(merge_grid, 1, 1),
        block=(merge_threads, 1, 1),
        arch=arch,
        stream=stream,
        timeout_ms=timeout_ms,
    )
    _retain_launch_tensors(inputs["query_sq"], inputs["database_sq"], tmap_query, tmap_database)


def _k10_split_count(n_query: int) -> int:
    if n_query <= SMALL_SHAPE_MAX:
        return MEDIUM_SPLITS
    if n_query < 4096:
        return RAG_SPLITS
    num_q_tiles = (n_query + BLOCK_Q - 1) // BLOCK_Q
    num_q_tile_pairs = (num_q_tiles + CTA_GROUP - 1) // CTA_GROUP
    if 38 <= num_q_tile_pairs <= 42:
        return RAG_SPLITS
    return MEDIUM_SPLITS


def _k32_split_count(k: int, n_query: int, n_database: int) -> int:
    if k not in K32_BUCKETS:
        raise NotImplementedError(f"knn_build K>10 currently supports exact k in {K32_BUCKETS}, got k={k}")
    if n_query != n_database or n_query < 512 or n_query > 4096:
        raise NotImplementedError("K>10 build path requires Q == M and 512 <= N <= 4096")
    if k == 12 and 1024 <= n_query <= 2048:
        return K12_MID_SPLITS
    if k == 30 and n_query <= K30_SMALL_SHAPE_MAX:
        return K30_SMALL_SPLITS
    return MEDIUM_SPLITS


def _k32_stage1_merge(k: int, split_count: int, n_query: int) -> tuple[_kernels.ExportedKernel, _kernels.ExportedKernel, bool]:
    use_unordered = k in (20, 30, 32) and split_count == MEDIUM_SPLITS and n_query >= 4096
    if use_unordered:
        stage1_name = {20: "stage1_k20_unordered_ir", 30: "stage1_k30_unordered_ir", 32: "stage1_k32_unordered_ir"}[k]
        merge_name = {20: "merge_k20_unordered_ir", 30: "merge_k30_unordered_ir", 32: "merge_k32_unordered_ir"}[k]
        return _kernels.get_kernel(stage1_name), _kernels.get_kernel(merge_name), False
    if split_count == K12_MID_SPLITS and k == 12:
        return _kernels.get_kernel("stage1_k12_ir"), _kernels.get_kernel("merge_k12_s8_ir"), False
    if split_count == K30_SMALL_SPLITS and k == 30:
        return _kernels.get_kernel("stage1_k30_ir"), _kernels.get_kernel("merge_k30_s8_ir"), False
    stage1_name = "stage1_ir" if k == 32 else f"stage1_k{k}_ir"
    merge_name = "merge_ir" if k == 32 else f"merge_k{k}_ir"
    return _kernels.get_kernel(stage1_name), _kernels.get_kernel(merge_name), True


def _launch_search_generic(
    query: Any,
    database: Any,
    out_dists: Any,
    out_indices: Any,
    k: int,
    *,
    arch: str | None,
    stream: Any,
    timeout_ms: float | None,
) -> None:
    bsz = int(query.shape[0])
    n_query = int(query.shape[1])
    n_database = int(database.shape[1])
    num_m_tiles = math.ceil(n_database / SEARCH_BLOCK_M)
    if num_m_tiles == 1:
        _launch(
            _kernels.get_kernel("search_knn_search_warp_direct_v1"),
            query,
            database,
            out_dists,
            out_indices,
            bsz,
            n_query,
            n_database,
            k,
            grid=(bsz * n_query, 1, 1),
            block=(SEARCH_THREADS, 1, 1),
            arch=arch,
            stream=stream,
            timeout_ms=timeout_ms,
        )
        return

    partial_dists, partial_indices = _search_partial_buffers(
        bsz=bsz,
        n_query=n_query,
        num_m_tiles=num_m_tiles,
        top_k=k,
        device=query.device,
    )
    _launch(
        _kernels.get_kernel("search_knn_search_warp_split_partial_v1"),
        query,
        database,
        partial_dists,
        partial_indices,
        bsz,
        n_query,
        n_database,
        k,
        num_m_tiles,
        grid=(bsz * n_query * num_m_tiles, 1, 1),
        block=(SEARCH_THREADS, 1, 1),
        arch=arch,
        stream=stream,
        timeout_ms=timeout_ms,
    )
    _launch(
        _kernels.get_kernel("search_knn_search_warp_split_merge_v1"),
        partial_dists,
        partial_indices,
        out_dists,
        out_indices,
        bsz,
        n_query,
        k,
        num_m_tiles,
        grid=(bsz * n_query, 1, 1),
        block=(SEARCH_THREADS, 1, 1),
        arch=arch,
        stream=stream,
        timeout_ms=timeout_ms,
    )


def _launch_search_q1(
    query: Any,
    database: Any,
    out_dists: Any,
    out_indices: Any,
    k: int,
    *,
    arch: str | None,
    stream: Any,
    timeout_ms: float | None,
) -> None:
    bsz = int(query.shape[0])
    n_query = int(query.shape[1])
    n_database = int(database.shape[1])
    num_m_tiles = math.ceil(n_database / SEARCH_Q1_BLOCK_M)
    tiles_per_group = max(SEARCH_Q1_MERGE_TILES_PER_GROUP, math.ceil(num_m_tiles / SEARCH_Q1_MERGE_WARPS))
    num_groups = math.ceil(num_m_tiles / tiles_per_group)
    partial_dists, partial_indices = _search_q1_partial_buffers(
        bsz=bsz,
        num_m_tiles=num_m_tiles,
        top_k=k,
        device=query.device,
    )
    _launch(
        _kernels.get_kernel("search_q1_knn_search_q1_tile_reduce_partial_v1"),
        query,
        database,
        partial_dists,
        partial_indices,
        bsz,
        n_query,
        n_database,
        k,
        num_m_tiles,
        grid=(bsz * num_m_tiles, 1, 1),
        block=(SEARCH_THREADS, 1, 1),
        arch=arch,
        stream=stream,
        timeout_ms=timeout_ms,
    )
    _launch(
        _kernels.get_kernel("search_q1_knn_search_q1_tile_reduce_merge_v1"),
        partial_dists,
        partial_indices,
        out_dists,
        out_indices,
        bsz,
        n_query,
        k,
        num_m_tiles,
        num_groups,
        tiles_per_group,
        grid=(bsz, 1, 1),
        block=(SEARCH_THREADS, 1, 1),
        arch=arch,
        stream=stream,
        timeout_ms=timeout_ms,
    )


def knn_build(
    database: Any,
    k: int,
    *,
    out: tuple[Any, Any] | None = None,
    arch: str | None = None,
    stream: Any = None,
    timeout_ms: float | None = None,
) -> tuple[Any, Any]:
    """Build an exact self-kNN graph for a BF16 [B, N, 128] CUDA tensor."""

    k = int(k)
    inputs = _make_build_inputs(database, k, out)
    n_query = int(inputs["Q"])
    n_database = int(inputs["M"])

    if k == 5:
        _launch_stage1_merge(
            inputs,
            split_count=MEDIUM_SPLITS,
            stage1_kernel=_kernels.get_kernel("k5t64_stage1_ir"),
            merge_kernel=_kernels.get_kernel("k5t64_merge_ir"),
            merge_with_k_arg=False,
            arch=arch,
            stream=stream,
            timeout_ms=timeout_ms,
        )
    elif k == 10:
        split_count = _k10_split_count(n_query)
        merge_name = "k5t64_merge_k10_s4_cache_ir" if split_count == MEDIUM_SPLITS else "k10t32_merge_k10_s7_cache_ir"
        _launch_stage1_merge(
            inputs,
            split_count=split_count,
            stage1_kernel=_kernels.get_kernel("k10root_stage1"),
            merge_kernel=_kernels.get_kernel(merge_name),
            merge_with_k_arg=False,
            arch=arch,
            stream=stream,
            timeout_ms=timeout_ms,
        )
    elif 10 < k <= 32:
        split_count = _k32_split_count(k, n_query, n_database)
        stage1_kernel, merge_kernel, merge_with_k_arg = _k32_stage1_merge(k, split_count, n_query)
        _launch_stage1_merge(
            inputs,
            split_count=split_count,
            stage1_kernel=stage1_kernel,
            merge_kernel=merge_kernel,
            merge_with_k_arg=merge_with_k_arg,
            arch=arch,
            stream=stream,
            timeout_ms=timeout_ms,
        )
    else:
        raise NotImplementedError("knn_build currently supports k=5, k=10, and exact K32 buckets 12/16/20/25/30/32")

    return inputs["out_dists"], inputs["out_indices"]


def knn_search(
    query: Any,
    database: Any,
    k: int,
    *,
    out: tuple[Any, Any] | None = None,
    arch: str | None = None,
    stream: Any = None,
    timeout_ms: float | None = None,
) -> tuple[Any, Any]:
    """Search exact BF16 squared-L2 top-k neighbors for [B, Q, 128] x [B, M, 128]."""

    k = int(k)
    _validate_pair(query, database, k)
    if k > SEARCH_K_MAX:
        raise NotImplementedError(f"knn_search currently supports k <= {SEARCH_K_MAX}, got k={k}")
    out_dists, out_indices = _allocate_outputs(query, k, out)
    if int(query.shape[1]) == 1:
        _launch_search_q1(
            query,
            database,
            out_dists,
            out_indices,
            k,
            arch=arch,
            stream=stream,
            timeout_ms=timeout_ms,
        )
    else:
        _launch_search_generic(
            query,
            database,
            out_dists,
            out_indices,
            k,
            arch=arch,
            stream=stream,
            timeout_ms=timeout_ms,
        )
    return out_dists, out_indices
