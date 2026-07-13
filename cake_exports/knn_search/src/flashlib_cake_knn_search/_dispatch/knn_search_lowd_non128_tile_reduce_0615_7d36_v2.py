"""Round-21 exact low-D non-D128 BF16 squared-L2 kNN seed.

Minimum target architecture: sm_80. This additive shape seed handles the
``B=1,Q=128,M=65536,K<=10,D in {64,96,192,320}`` blind-spot bucket with a
CUDA-core subwarp tile-reduce path. Each producer CTA owns one
``(batch, query, M tile)``, computes a tile-local sorted top-10 list, and a
second reducer CTA per query merges those sorted tile lists into the contract
``distances`` and ``indices`` outputs.
"""
from __future__ import annotations
from json import loads as _json_loads
from .._dispatch_runtime import _capture_cuTensorMapEncodeTiled, _decode_capture, _import_dispatch_module, _ir_proxy
import math
from typing import Any
from .._dispatch_runtime import select_named_shapes
from .knn_search_stream import current_stream_handle
THREADS = 256
NUM_WARPS = THREADS // 32
MERGE_THREADS = 256
MERGE_WARPS = MERGE_THREADS // 32
K_MAX = 10
MAX_TILE_LISTS = 128
MAX_Q_CACHE_ELEMS = 40
MERGE_TILES_PER_GROUP = 64
TILE_DIST_BYTES = MAX_TILE_LISTS * K_MAX * 4
TILE_IDX_BYTES = MAX_TILE_LISTS * K_MAX * 4
TILE_SMEM_BYTES = TILE_DIST_BYTES + TILE_IDX_BYTES
MERGE_GROUP_DIST_BYTES = MERGE_WARPS * K_MAX * 4
MERGE_GROUP_IDX_BYTES = MERGE_WARPS * K_MAX * 4
MERGE_SMEM_BYTES = MERGE_GROUP_DIST_BYTES + MERGE_GROUP_IDX_BYTES
LOWD_NON128_LABELS: tuple[str, ...] = ('blind_d64_q128_m65536_k10', 'blind_d96_q128_m65536_k10', 'blind_d192_q128_m65536_k10', 'blind_d320_q128_m65536_k10')
LOWD_NON128_SHAPES = _decode_capture(_json_loads('[{"__dict_items__": [["label", "blind_d64_q128_m65536_k10"], ["params", {"__dict_items__": [["B", 1], ["Q", 128], ["M", 65536], ["D", 64], ["K", 10], ["dtype", "bfloat16"], ["seed", 610508], ["self_search", false], ["min_recall", 0.999]]}]]}, {"__dict_items__": [["label", "blind_d96_q128_m65536_k10"], ["params", {"__dict_items__": [["B", 1], ["Q", 128], ["M", 65536], ["D", 96], ["K", 10], ["dtype", "bfloat16"], ["seed", 610520], ["self_search", false], ["min_recall", 0.999]]}]]}, {"__dict_items__": [["label", "blind_d192_q128_m65536_k10"], ["params", {"__dict_items__": [["B", 1], ["Q", 128], ["M", 65536], ["D", 192], ["K", 10], ["dtype", "bfloat16"], ["seed", 610521], ["self_search", false], ["min_recall", 0.999]]}]]}, {"__dict_items__": [["label", "blind_d320_q128_m65536_k10"], ["params", {"__dict_items__": [["B", 1], ["Q", 128], ["M", 65536], ["D", 320], ["K", 10], ["dtype", "bfloat16"], ["seed", 610509], ["self_search", false], ["min_recall", 0.999]]}]]}]'))
_KERNELS: dict[int, dict[str, Any]] = {}
_SCRATCH: dict[tuple[int, int, int, int, int, int, str, int], tuple[Any, Any]] = {}
D96_BLOCK_M = 640
D96_SUBWARP_WIDTH = 4
D96_SUBWARPS_PER_WARP = 8
D96_NUM_ROW_WORKERS = NUM_WARPS * D96_SUBWARPS_PER_WARP
D96_ROWS_PER_WORKER = _decode_capture(_json_loads('10'))
D192_BLOCK_M = 320
D192_SUBWARP_WIDTH = 8
D192_SUBWARPS_PER_WARP = 4
D192_NUM_ROW_WORKERS = NUM_WARPS * D192_SUBWARPS_PER_WARP
D192_ROWS_PER_WORKER = _decode_capture(_json_loads('10'))
D320_BLOCK_M = 320
D320_SUBWARP_WIDTH = 8
D320_SUBWARPS_PER_WARP = 4
D320_NUM_ROW_WORKERS = NUM_WARPS * D320_SUBWARPS_PER_WARP
D320_ROWS_PER_WORKER = _decode_capture(_json_loads('10'))
knn_search_lowd_non128_tile_reduce_partial_0615_7d36_v2 = _decode_capture(_json_loads('{"__ir__": "knn_search_lowd_non128_tile_reduce_partial_0615_7d36_v2", "arg_keys": ["queries", "database", "partial_distances", "partial_indices", "B", "Q", "M", "K", "num_m_tiles"], "cluster_dims": [1, 1, 1], "computed_smem_bytes": 10240, "constants": [["D_", 64], ["BLOCK_M_", 1280], ["SUBWARP_WIDTH_", 2], ["SUBWARPS_PER_WARP_", 16], ["NUM_ROW_WORKERS_", 128], ["ROWS_PER_WORKER_", 10], ["VECS_PER_LANE_", 4], ["K_MAX_", 10]], "cta_group": 1, "threads": 256}'))
knn_search_lowd_d96_tile_reduce_partial_0615_7d36_v2 = _decode_capture(_json_loads('{"__ir__": "knn_search_lowd_d96_tile_reduce_partial_0615_7d36_v2", "arg_keys": ["queries", "database", "partial_distances", "partial_indices", "B", "Q", "M", "K", "num_m_tiles"], "cluster_dims": [1, 1, 1], "computed_smem_bytes": 10240, "constants": [], "cta_group": 1, "threads": 256}'))
knn_search_lowd_d192_tile_reduce_partial_0615_7d36_v2 = _decode_capture(_json_loads('{"__ir__": "knn_search_lowd_d192_tile_reduce_partial_0615_7d36_v2", "arg_keys": ["queries", "database", "partial_distances", "partial_indices", "B", "Q", "M", "K", "num_m_tiles"], "cluster_dims": [1, 1, 1], "computed_smem_bytes": 10240, "constants": [], "cta_group": 1, "threads": 256}'))
knn_search_lowd_d320_tile_reduce_partial_0615_7d36_v2 = _decode_capture(_json_loads('{"__ir__": "knn_search_lowd_d320_tile_reduce_partial_0615_7d36_v2", "arg_keys": ["queries", "database", "partial_distances", "partial_indices", "B", "Q", "M", "K", "num_m_tiles"], "cluster_dims": [1, 1, 1], "computed_smem_bytes": 10240, "constants": [], "cta_group": 1, "threads": 256}'))
knn_search_lowd_non128_tile_reduce_merge_0615_7d36_v2 = _decode_capture(_json_loads('{"__ir__": "knn_search_lowd_non128_tile_reduce_merge_0615_7d36_v2", "arg_keys": ["partial_distances", "partial_indices", "out_distances", "out_indices", "B", "Q", "K", "num_m_tiles", "num_groups", "tiles_per_group"], "cluster_dims": [1, 1, 1], "computed_smem_bytes": 640, "constants": [["K_MAX_", 10]], "cta_group": 1, "threads": 256}'))
ir = _decode_capture(_json_loads('{"__ir__": "knn_search_lowd_non128_tile_reduce_partial_0615_7d36_v2", "arg_keys": ["queries", "database", "partial_distances", "partial_indices", "B", "Q", "M", "K", "num_m_tiles"], "cluster_dims": [1, 1, 1], "computed_smem_bytes": 10240, "constants": [["D_", 64], ["BLOCK_M_", 1280], ["SUBWARP_WIDTH_", 2], ["SUBWARPS_PER_WARP_", 16], ["NUM_ROW_WORKERS_", 128], ["ROWS_PER_WORKER_", 10], ["VECS_PER_LANE_", 4], ["K_MAX_", 10]], "cta_group": 1, "threads": 256}'))
partial_ir = _decode_capture(_json_loads('{"__ir__": "knn_search_lowd_non128_tile_reduce_partial_0615_7d36_v2", "arg_keys": ["queries", "database", "partial_distances", "partial_indices", "B", "Q", "M", "K", "num_m_tiles"], "cluster_dims": [1, 1, 1], "computed_smem_bytes": 10240, "constants": [["D_", 64], ["BLOCK_M_", 1280], ["SUBWARP_WIDTH_", 2], ["SUBWARPS_PER_WARP_", 16], ["NUM_ROW_WORKERS_", 128], ["ROWS_PER_WORKER_", 10], ["VECS_PER_LANE_", 4], ["K_MAX_", 10]], "cta_group": 1, "threads": 256}'))
merge_ir = _decode_capture(_json_loads('{"__ir__": "knn_search_lowd_non128_tile_reduce_merge_0615_7d36_v2", "arg_keys": ["partial_distances", "partial_indices", "out_distances", "out_indices", "B", "Q", "K", "num_m_tiles", "num_groups", "tiles_per_group"], "cluster_dims": [1, 1, 1], "computed_smem_bytes": 640, "constants": [["K_MAX_", 10]], "cta_group": 1, "threads": 256}'))

def _config_for_d(d: int) -> dict[str, int]:
    if d == 64:
        subwarp_width = 2
        block_m = 1280
    elif d == 96:
        subwarp_width = 4
        block_m = D96_BLOCK_M
    elif d == 192:
        subwarp_width = 8
        block_m = D192_BLOCK_M
    elif d == 320:
        subwarp_width = 8
        block_m = 320
    else:
        raise ValueError(''.join(['knn_search_lowd_non128_tile_reduce_0615_7d36_v2 unsupported D=', format(d, '')]))
    subwarps_per_warp = 32 // subwarp_width
    num_row_workers = NUM_WARPS * subwarps_per_warp
    rows_per_worker = math.ceil(block_m / num_row_workers)
    vecs_per_lane = math.ceil(d / (subwarp_width * 8))
    return {'D_': d, 'BLOCK_M_': block_m, 'SUBWARP_WIDTH_': subwarp_width, 'SUBWARPS_PER_WARP_': subwarps_per_warp, 'NUM_ROW_WORKERS_': num_row_workers, 'ROWS_PER_WORKER_': rows_per_worker, 'VECS_PER_LANE_': vecs_per_lane}

def _compile_kernels(d: int) -> dict[str, Any]:
    from .._dispatch_runtime import generate_kernel
    from .._dispatch_runtime import _cuda_include_dirs
    from .._dispatch_runtime import compile_cuda, detect_gpu_arch
    from .._dispatch_runtime import CUDAKernel
    arch = detect_gpu_arch()
    include_dirs = _cuda_include_dirs()
    if d == 96:
        partial_ir_for_d = knn_search_lowd_d96_tile_reduce_partial_0615_7d36_v2
        partial_source = generate_kernel(partial_ir_for_d, validate=False, smem_bytes=TILE_SMEM_BYTES)
    elif d == 192:
        partial_ir_for_d = knn_search_lowd_d192_tile_reduce_partial_0615_7d36_v2
        partial_source = generate_kernel(partial_ir_for_d, validate=False, smem_bytes=TILE_SMEM_BYTES)
    elif d == 320:
        partial_ir_for_d = knn_search_lowd_d320_tile_reduce_partial_0615_7d36_v2
        partial_source = generate_kernel(partial_ir_for_d, validate=False, smem_bytes=TILE_SMEM_BYTES)
    else:
        partial_ir_for_d = knn_search_lowd_non128_tile_reduce_partial_0615_7d36_v2
        cfg = _config_for_d(d)
        partial_source = generate_kernel(partial_ir_for_d, validate=False, smem_bytes=TILE_SMEM_BYTES, **cfg)
    merge_source = generate_kernel(knn_search_lowd_non128_tile_reduce_merge_0615_7d36_v2, validate=False, smem_bytes=MERGE_SMEM_BYTES)
    partial_cubin = compile_cuda(partial_source, arch=arch, options=['--use_fast_math'], include_dirs=include_dirs)
    merge_cubin = compile_cuda(merge_source, arch=arch, options=['--use_fast_math'], include_dirs=include_dirs)
    return {'partial': CUDAKernel(partial_cubin, ''.join(['kernel_', format(partial_ir_for_d.symbol, '')])), 'merge': CUDAKernel(merge_cubin, ''.join(['kernel_', format(knn_search_lowd_non128_tile_reduce_merge_0615_7d36_v2.symbol, '')]))}

def _scratch(inputs: dict[str, Any], num_m_tiles: int) -> tuple[Any, Any]:
    import torch
    key = (int(inputs['B']), int(inputs['Q']), int(inputs['M']), int(inputs['D']), int(num_m_tiles), int(inputs['queries'].device.index or 0), str(inputs['queries'].dtype), current_stream_handle(inputs))
    cached = _SCRATCH.get(key)
    if cached is None:
        shape = (int(inputs['B']), int(inputs['Q']), int(num_m_tiles), K_MAX)
        cached = (torch.empty(shape, dtype=torch.float32, device=inputs['queries'].device), torch.empty(shape, dtype=torch.int32, device=inputs['queries'].device))
        _SCRATCH[key] = cached
    return cached

def _shape_guard(inputs: dict[str, Any]) -> bool:
    return int(inputs['B']) == 1 and int(inputs['Q']) == 128 and (int(inputs['M']) == 65536) and (int(inputs['K']) <= K_MAX) and (int(inputs['D']) in {64, 96, 192, 320})

def launch_for_eval(inputs: dict[str, Any]) -> dict[str, Any]:
    if not _shape_guard(inputs):
        raise ValueError('knn_search_lowd_non128_tile_reduce_0615_7d36_v2 supports only B=1,Q=128,M=65536,K<=10,D in {64,96,192,320}')
    d = int(inputs['D'])
    kernels = _KERNELS.get(d)
    if kernels is None:
        kernels = _compile_kernels(d)
        _KERNELS[d] = kernels
    bsz = int(inputs['B'])
    q_rows = int(inputs['Q'])
    m_rows = int(inputs['M'])
    k = int(inputs['K'])
    cfg = _config_for_d(d)
    block_m = int(cfg['BLOCK_M_'])
    num_m_tiles = math.ceil(m_rows / block_m)
    tiles_per_group = max(MERGE_TILES_PER_GROUP, math.ceil(num_m_tiles / MERGE_WARPS))
    num_groups = math.ceil(num_m_tiles / tiles_per_group)
    partial_dist, partial_idx = _scratch(inputs, num_m_tiles)
    kernels['partial'].launch(grid=(bsz * q_rows * num_m_tiles, 1, 1), block=(THREADS, 1, 1), args=[inputs['queries'], inputs['database'], partial_dist, partial_idx, bsz, q_rows, m_rows, k, num_m_tiles], shared_mem=TILE_SMEM_BYTES)
    kernels['merge'].launch(grid=(bsz * q_rows, 1, 1), block=(MERGE_THREADS, 1, 1), args=[partial_dist, partial_idx, inputs['out_distances'], inputs['out_indices'], bsz, q_rows, k, num_m_tiles, num_groups, tiles_per_group], shared_mem=MERGE_SMEM_BYTES)
    return {'distances': inputs['out_distances'], 'indices': inputs['out_indices']}
SHAPE_DISPATCH_REGISTRY = [{'shape_key': label, 'guard': ''.join(['B == 1 and Q == 128 and M == 65536 and D == ', format(d, ''), ' and K <= 10']), 'candidate_entrypoint': 'loom.examples.weave.knn_search_lowd_non128_tile_reduce_0615_7d36_v2:launch_for_eval'} for label, d in (('blind_d64_q128_m65536_k10', 64), ('blind_d96_q128_m65536_k10', 96), ('blind_d192_q128_m65536_k10', 192), ('blind_d320_q128_m65536_k10', 320))]

def knn_search_compile_and_launch_lowd_non128_tile_reduce(*, benchmark: bool=True, shapes: list[dict[str, Any]] | None=None) -> dict[str, Any]:
    from .._dispatch_runtime import evaluate
    result = evaluate(launch_for_eval, shapes=LOWD_NON128_SHAPES if shapes is None else shapes, benchmark=benchmark)
    result['passed'] = bool(result.get('summary', {}).get('all_correct'))
    print(result)
    return result
