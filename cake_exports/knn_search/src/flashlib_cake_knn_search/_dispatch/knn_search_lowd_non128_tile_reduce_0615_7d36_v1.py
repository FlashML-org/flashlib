"""Round-20/7d36 low-D non-D128 exact BF16 squared-L2 kNN seed.

Minimum target architecture: sm_80. This additive bucket kernel targets the
auto-tuning blind low-D rows ``B=1,Q=128,M=65536,K=10,D in {64,96,192,320}``
with a CUDA-core tile-reduction path. It is source-clean relative to FlashLib:
the candidate streams database tiles, keeps row-worker top-K lists in shared
memory, emits one tile-local top-K list, and performs a second exact merge into
the contract-visible ``distances`` and ``indices`` outputs.
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
LOCAL_LIST_CAP = 10
MAX_ROW_WORKERS = 128
TILE_DIST_BYTES = MAX_ROW_WORKERS * K_MAX * 4
TILE_IDX_BYTES = MAX_ROW_WORKERS * K_MAX * 4
TILE_SMEM_BYTES = TILE_DIST_BYTES + TILE_IDX_BYTES
MERGE_GROUP_DIST_BYTES = MERGE_WARPS * K_MAX * 4
MERGE_GROUP_IDX_BYTES = MERGE_WARPS * K_MAX * 4
MERGE_SMEM_BYTES = MERGE_GROUP_DIST_BYTES + MERGE_GROUP_IDX_BYTES
MERGE_TILES_PER_GROUP = 64
SUPPORTED_D = {64, 96, 192, 320}
LOWD_NON128_LABELS: tuple[str, ...] = ('blind_d64_q128_m65536_k10', 'blind_d96_q128_m65536_k10', 'blind_d192_q128_m65536_k10', 'blind_d320_q128_m65536_k10')
LOWD_NON128_SHAPES = _decode_capture(_json_loads('[{"__dict_items__": [["label", "blind_d64_q128_m65536_k10"], ["params", {"__dict_items__": [["B", 1], ["Q", 128], ["M", 65536], ["D", 64], ["K", 10], ["dtype", "bfloat16"], ["seed", 610508], ["self_search", false], ["min_recall", 0.999]]}]]}, {"__dict_items__": [["label", "blind_d96_q128_m65536_k10"], ["params", {"__dict_items__": [["B", 1], ["Q", 128], ["M", 65536], ["D", 96], ["K", 10], ["dtype", "bfloat16"], ["seed", 610520], ["self_search", false], ["min_recall", 0.999]]}]]}, {"__dict_items__": [["label", "blind_d192_q128_m65536_k10"], ["params", {"__dict_items__": [["B", 1], ["Q", 128], ["M", 65536], ["D", 192], ["K", 10], ["dtype", "bfloat16"], ["seed", 610521], ["self_search", false], ["min_recall", 0.999]]}]]}, {"__dict_items__": [["label", "blind_d320_q128_m65536_k10"], ["params", {"__dict_items__": [["B", 1], ["Q", 128], ["M", 65536], ["D", 320], ["K", 10], ["dtype", "bfloat16"], ["seed", 610509], ["self_search", false], ["min_recall", 0.999]]}]]}]'))
ROUTE_LOWD_NON128_TILE_REDUCE = 'round20_7d36_lowd_non128_tile_reduce'
SHAPE_DISPATCH_REGISTRY: tuple[dict[str, str], ...] = ({'shape_key': 'round20_7d36_lowd_non128_q128_m65536_k10', 'guard': 'B == 1 and Q == 128 and M == 65536 and D in {64,96,192,320} and K == 10', 'route': ROUTE_LOWD_NON128_TILE_REDUCE},)
ir = _decode_capture(_json_loads('{"__ir__": "knn_search_lowd_non128_tile_reduce_partial_0615_7d36_v1", "arg_keys": ["queries", "database", "partial_distances", "partial_indices", "B", "Q", "M", "K", "num_m_tiles"], "cluster_dims": [1, 1, 1], "computed_smem_bytes": 10240, "constants": [["D_", 64], ["K_MAX_", 10], ["BLOCK_M_", 1280], ["NUM_ROW_WORKERS_", 128], ["SUBWARP_WIDTH_", 2], ["SUBWARPS_PER_WARP_", 16], ["CHUNKS_", 4], ["LOCAL_LIST_CAP_", 10]], "cta_group": 1, "threads": 256}'))
partial_ir = _decode_capture(_json_loads('{"__ir__": "knn_search_lowd_non128_tile_reduce_partial_0615_7d36_v1", "arg_keys": ["queries", "database", "partial_distances", "partial_indices", "B", "Q", "M", "K", "num_m_tiles"], "cluster_dims": [1, 1, 1], "computed_smem_bytes": 10240, "constants": [["D_", 64], ["K_MAX_", 10], ["BLOCK_M_", 1280], ["NUM_ROW_WORKERS_", 128], ["SUBWARP_WIDTH_", 2], ["SUBWARPS_PER_WARP_", 16], ["CHUNKS_", 4], ["LOCAL_LIST_CAP_", 10]], "cta_group": 1, "threads": 256}'))
merge_ir = _decode_capture(_json_loads('{"__ir__": "knn_search_lowd_non128_tile_reduce_merge_0615_7d36_v1", "arg_keys": ["partial_distances", "partial_indices", "out_distances", "out_indices", "B", "Q", "K", "num_m_tiles", "num_groups", "tiles_per_group"], "cluster_dims": [1, 1, 1], "computed_smem_bytes": 640, "constants": [["K_MAX_", 10]], "cta_group": 1, "threads": 256}'))
_LOWD_KERNELS: dict[int, dict[str, Any]] = {}
_LOWD_SCRATCH: dict[tuple[int, int, int, int, int, int, str, int], tuple[Any, Any]] = {}

def _config_for_d(d: int) -> dict[str, int]:
    if d == 64:
        subwarp_width = 2
    elif d == 96:
        subwarp_width = 4
    elif d in {192, 320}:
        subwarp_width = 8
    else:
        raise ValueError(''.join(['unsupported low-D non-D128 dimension: ', format(d, '')]))
    subwarps_per_warp = 32 // subwarp_width
    num_row_workers = NUM_WARPS * subwarps_per_warp
    return {'D': d, 'SUBWARP_WIDTH': subwarp_width, 'SUBWARPS_PER_WARP': subwarps_per_warp, 'NUM_ROW_WORKERS': num_row_workers, 'BLOCK_M': num_row_workers * LOCAL_LIST_CAP, 'CHUNKS': math.ceil(d / (subwarp_width * 8))}
knn_search_lowd_non128_tile_reduce_partial_0615_7d36_v1 = _decode_capture(_json_loads('{"__ir__": "knn_search_lowd_non128_tile_reduce_partial_0615_7d36_v1", "arg_keys": ["queries", "database", "partial_distances", "partial_indices", "B", "Q", "M", "K", "num_m_tiles"], "cluster_dims": [1, 1, 1], "computed_smem_bytes": 10240, "constants": [["D_", 64], ["K_MAX_", 10], ["BLOCK_M_", 1280], ["NUM_ROW_WORKERS_", 128], ["SUBWARP_WIDTH_", 2], ["SUBWARPS_PER_WARP_", 16], ["CHUNKS_", 4], ["LOCAL_LIST_CAP_", 10]], "cta_group": 1, "threads": 256}'))
knn_search_lowd_non128_tile_reduce_merge_0615_7d36_v1 = _decode_capture(_json_loads('{"__ir__": "knn_search_lowd_non128_tile_reduce_merge_0615_7d36_v1", "arg_keys": ["partial_distances", "partial_indices", "out_distances", "out_indices", "B", "Q", "K", "num_m_tiles", "num_groups", "tiles_per_group"], "cluster_dims": [1, 1, 1], "computed_smem_bytes": 640, "constants": [["K_MAX_", 10]], "cta_group": 1, "threads": 256}'))
ir = _decode_capture(_json_loads('{"__ir__": "knn_search_lowd_non128_tile_reduce_partial_0615_7d36_v1", "arg_keys": ["queries", "database", "partial_distances", "partial_indices", "B", "Q", "M", "K", "num_m_tiles"], "cluster_dims": [1, 1, 1], "computed_smem_bytes": 10240, "constants": [["D_", 64], ["K_MAX_", 10], ["BLOCK_M_", 1280], ["NUM_ROW_WORKERS_", 128], ["SUBWARP_WIDTH_", 2], ["SUBWARPS_PER_WARP_", 16], ["CHUNKS_", 4], ["LOCAL_LIST_CAP_", 10]], "cta_group": 1, "threads": 256}'))
partial_ir = _decode_capture(_json_loads('{"__ir__": "knn_search_lowd_non128_tile_reduce_partial_0615_7d36_v1", "arg_keys": ["queries", "database", "partial_distances", "partial_indices", "B", "Q", "M", "K", "num_m_tiles"], "cluster_dims": [1, 1, 1], "computed_smem_bytes": 10240, "constants": [["D_", 64], ["K_MAX_", 10], ["BLOCK_M_", 1280], ["NUM_ROW_WORKERS_", 128], ["SUBWARP_WIDTH_", 2], ["SUBWARPS_PER_WARP_", 16], ["CHUNKS_", 4], ["LOCAL_LIST_CAP_", 10]], "cta_group": 1, "threads": 256}'))
merge_ir = _decode_capture(_json_loads('{"__ir__": "knn_search_lowd_non128_tile_reduce_merge_0615_7d36_v1", "arg_keys": ["partial_distances", "partial_indices", "out_distances", "out_indices", "B", "Q", "K", "num_m_tiles", "num_groups", "tiles_per_group"], "cluster_dims": [1, 1, 1], "computed_smem_bytes": 640, "constants": [["K_MAX_", 10]], "cta_group": 1, "threads": 256}'))

def _compile_kernels(d: int) -> dict[str, Any]:
    from .._dispatch_runtime import generate_kernel
    from .._dispatch_runtime import _cuda_include_dirs
    from .._dispatch_runtime import compile_cuda, detect_gpu_arch
    from .._dispatch_runtime import CUDAKernel
    cfg = _config_for_d(d)
    arch = detect_gpu_arch()
    include_dirs = _cuda_include_dirs()
    partial_source = generate_kernel(knn_search_lowd_non128_tile_reduce_partial_0615_7d36_v1, validate=False, smem_bytes=TILE_SMEM_BYTES, D_=cfg['D'], BLOCK_M_=cfg['BLOCK_M'], NUM_ROW_WORKERS_=cfg['NUM_ROW_WORKERS'], SUBWARP_WIDTH_=cfg['SUBWARP_WIDTH'], SUBWARPS_PER_WARP_=cfg['SUBWARPS_PER_WARP'], CHUNKS_=cfg['CHUNKS'])
    merge_source = generate_kernel(knn_search_lowd_non128_tile_reduce_merge_0615_7d36_v1, validate=False, smem_bytes=MERGE_SMEM_BYTES)
    partial_cubin = compile_cuda(partial_source, arch=arch, options=['--use_fast_math'], include_dirs=include_dirs)
    merge_cubin = compile_cuda(merge_source, arch=arch, options=['--use_fast_math'], include_dirs=include_dirs)
    return {'partial': CUDAKernel(partial_cubin, ''.join(['kernel_', format(knn_search_lowd_non128_tile_reduce_partial_0615_7d36_v1.symbol, '')])), 'merge': CUDAKernel(merge_cubin, ''.join(['kernel_', format(knn_search_lowd_non128_tile_reduce_merge_0615_7d36_v1.symbol, '')]))}

def _scratch(inputs: dict[str, Any], num_m_tiles: int) -> tuple[Any, Any]:
    import torch
    key = (int(inputs['B']), int(inputs['Q']), int(inputs['D']), int(num_m_tiles), int(inputs['queries'].device.index or 0), int(inputs['K']), str(inputs['queries'].dtype), current_stream_handle(inputs))
    cached = _LOWD_SCRATCH.get(key)
    if cached is None:
        shape = (int(inputs['B']), int(inputs['Q']), int(num_m_tiles), K_MAX)
        cached = (torch.empty(shape, dtype=torch.float32, device=inputs['queries'].device), torch.empty(shape, dtype=torch.int32, device=inputs['queries'].device))
        _LOWD_SCRATCH[key] = cached
    return cached

def _use_lowd_non128_tile_reduce(inputs: dict[str, Any]) -> bool:
    return int(inputs['B']) == 1 and int(inputs['Q']) == 128 and (int(inputs['M']) == 65536) and (int(inputs['D']) in SUPPORTED_D) and (int(inputs['K']) == K_MAX)

def selected_route(inputs: dict[str, Any]) -> str:
    if _use_lowd_non128_tile_reduce(inputs):
        return ROUTE_LOWD_NON128_TILE_REDUCE
    return 'unsupported_guard_miss'

def route_info(inputs: dict[str, Any]) -> dict[str, Any]:
    route = selected_route(inputs)
    return {'route': route, 'selected_route': route, 'selected_entrypoint': 'loom.examples.weave.knn_search_lowd_non128_tile_reduce_0615_7d36_v1:launch_for_eval', 'route_kind': 'specialized' if route == ROUTE_LOWD_NON128_TILE_REDUCE else 'unsupported', 'coverage_class': 'performance_route_lowd_non128_tile_reduce', 'coverage_only': False, 'production_policy': 'weave_only', 'external_fallback': None, 'selected_guard': SHAPE_DISPATCH_REGISTRY[0]['guard']}

def route_trace_entry(label: str, inputs: dict[str, Any]) -> dict[str, Any]:
    return {'label': label, 'B': int(inputs['B']), 'Q': int(inputs['Q']), 'M': int(inputs['M']), 'D': int(inputs['D']), 'K': int(inputs['K']), **route_info(inputs)}

def launch_for_eval(inputs: dict[str, Any]) -> dict[str, Any]:
    if not _use_lowd_non128_tile_reduce(inputs):
        raise ValueError('knn_search_lowd_non128_tile_reduce_0615_7d36_v1 supports only B=1,Q=128,M=65536,K=10,D in {64,96,192,320}')
    d = int(inputs['D'])
    kernels = _LOWD_KERNELS.get(d)
    if kernels is None:
        kernels = _compile_kernels(d)
        _LOWD_KERNELS[d] = kernels
    bsz = int(inputs['B'])
    q_rows = int(inputs['Q'])
    m_rows = int(inputs['M'])
    k = int(inputs['K'])
    cfg = _config_for_d(d)
    num_m_tiles = math.ceil(m_rows / cfg['BLOCK_M'])
    tiles_per_group = max(MERGE_TILES_PER_GROUP, math.ceil(num_m_tiles / MERGE_WARPS))
    num_groups = math.ceil(num_m_tiles / tiles_per_group)
    partial_dist, partial_idx = _scratch(inputs, num_m_tiles)
    kernels['partial'].launch(grid=(bsz * q_rows * num_m_tiles, 1, 1), block=(THREADS, 1, 1), args=[inputs['queries'], inputs['database'], partial_dist, partial_idx, bsz, q_rows, m_rows, k, num_m_tiles], shared_mem=TILE_SMEM_BYTES)
    kernels['merge'].launch(grid=(bsz * q_rows, 1, 1), block=(MERGE_THREADS, 1, 1), args=[partial_dist, partial_idx, inputs['out_distances'], inputs['out_indices'], bsz, q_rows, k, num_m_tiles, num_groups, tiles_per_group], shared_mem=MERGE_SMEM_BYTES)
    return {'distances': inputs['out_distances'], 'indices': inputs['out_indices']}

def knn_search_compile_and_launch_lowd_non128_tile_reduce(*, benchmark: bool=True, shapes: list[dict[str, Any]] | None=None) -> dict[str, Any]:
    from .._dispatch_runtime import evaluate
    result = evaluate(launch_for_eval, shapes=LOWD_NON128_SHAPES if shapes is None else shapes, benchmark=benchmark)
    result['passed'] = bool(result.get('summary', {}).get('all_correct'))
    print(result)
    return result
