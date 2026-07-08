"""Dynamic D3 tile-reduce seed for exact BF16 squared-L2 kNN.

Minimum target architecture: sm_80. This additive bucket kernel targets the
generalize-auto-tuning dynamic-D blind spot
``B=1,Q=128,M=65536,K=10,D=3`` without host padding. This file is a D3-only
replay of the b5b2 tile-reduce seed; D7/D63 are intentionally left to the
353b tiny-D seed in the production dispatcher.
"""
from __future__ import annotations
from json import loads as _json_loads
from .._dispatch_runtime import _capture_cuTensorMapEncodeTiled, _decode_capture, _import_dispatch_module, _ir_proxy
import math
from typing import Any
from .._dispatch_runtime import select_named_shapes
THREADS = 128
NUM_WARPS = THREADS // 32
MERGE_THREADS = 128
MERGE_WARPS = MERGE_THREADS // 32
K_MAX = 10
MAX_D = 64
MAX_ROW_WORKERS = THREADS
MERGE_TILES_PER_GROUP = 32
TILE_DIST_BYTES = MAX_ROW_WORKERS * K_MAX * 4
TILE_IDX_BYTES = MAX_ROW_WORKERS * K_MAX * 4
TILE_SMEM_BYTES = TILE_DIST_BYTES + TILE_IDX_BYTES
MERGE_GROUP_DIST_BYTES = MERGE_WARPS * K_MAX * 4
MERGE_GROUP_IDX_BYTES = MERGE_WARPS * K_MAX * 4
MERGE_SMEM_BYTES = MERGE_GROUP_DIST_BYTES + MERGE_GROUP_IDX_BYTES
SUPPORTED_D = {3}
TINYD_LABELS: tuple[str, ...] = ('blind_dyn_d3_q128_m65536_k10',)
TINYD_SHAPES = _decode_capture(_json_loads('[{"__dict_items__": [["label", "blind_dyn_d3_q128_m65536_k10"], ["params", {"__dict_items__": [["B", 1], ["Q", 128], ["M", 65536], ["D", 3], ["K", 10], ["dtype", "bfloat16"], ["seed", 610801], ["self_search", false], ["min_recall", 0.999]]}]]}]'))
ROUTE_TINYD_D3_TILE_REDUCE = 'b5b2_dynamic_d3_tinyd_tile_reduce'
ir = _decode_capture(_json_loads('{"__ir__": "knn_search_dynamic_tinyd_tile_reduce_partial_0618_c8b9_v1", "arg_keys": ["queries", "database", "partial_distances", "partial_indices", "B", "Q", "M", "K", "num_m_tiles"], "cluster_dims": [1, 1, 1], "computed_smem_bytes": 10240, "constants": [["D_", 3], ["BLOCK_M_", 4096], ["ROWS_PER_WORKER_", 32], ["K_MAX_", 10]], "cta_group": 1, "threads": 128}'))
partial_ir = _decode_capture(_json_loads('{"__ir__": "knn_search_dynamic_tinyd_tile_reduce_partial_0618_c8b9_v1", "arg_keys": ["queries", "database", "partial_distances", "partial_indices", "B", "Q", "M", "K", "num_m_tiles"], "cluster_dims": [1, 1, 1], "computed_smem_bytes": 10240, "constants": [["D_", 3], ["BLOCK_M_", 4096], ["ROWS_PER_WORKER_", 32], ["K_MAX_", 10]], "cta_group": 1, "threads": 128}'))
merge_ir = _decode_capture(_json_loads('{"__ir__": "knn_search_dynamic_tinyd_tile_reduce_merge_0618_c8b9_v1", "arg_keys": ["partial_distances", "partial_indices", "out_distances", "out_indices", "B", "Q", "K", "num_m_tiles", "num_groups", "tiles_per_group"], "cluster_dims": [1, 1, 1], "computed_smem_bytes": 384, "constants": [["K_MAX_", 10]], "cta_group": 1, "threads": 128}'))
_KERNELS: dict[tuple[int, int], dict[str, Any]] = {}
_SCRATCH: dict[tuple[int, int, int, int, int, int, str], tuple[Any, Any]] = {}
knn_search_dynamic_tinyd_tile_reduce_partial_0618_c8b9_v1 = _decode_capture(_json_loads('{"__ir__": "knn_search_dynamic_tinyd_tile_reduce_partial_0618_c8b9_v1", "arg_keys": ["queries", "database", "partial_distances", "partial_indices", "B", "Q", "M", "K", "num_m_tiles"], "cluster_dims": [1, 1, 1], "computed_smem_bytes": 10240, "constants": [["D_", 3], ["BLOCK_M_", 4096], ["ROWS_PER_WORKER_", 32], ["K_MAX_", 10]], "cta_group": 1, "threads": 128}'))
knn_search_dynamic_d63_tile_reduce_partial_0618_c8b9_v1 = _decode_capture(_json_loads('{"__ir__": "knn_search_dynamic_d63_tile_reduce_partial_0618_c8b9_v1", "arg_keys": ["queries", "database", "partial_distances", "partial_indices", "B", "Q", "M", "K", "num_m_tiles"], "cluster_dims": [1, 1, 1], "computed_smem_bytes": 10240, "constants": [["K_MAX_", 10]], "cta_group": 1, "threads": 128}'))
knn_search_dynamic_tinyd_tile_reduce_merge_0618_c8b9_v1 = _decode_capture(_json_loads('{"__ir__": "knn_search_dynamic_tinyd_tile_reduce_merge_0618_c8b9_v1", "arg_keys": ["partial_distances", "partial_indices", "out_distances", "out_indices", "B", "Q", "K", "num_m_tiles", "num_groups", "tiles_per_group"], "cluster_dims": [1, 1, 1], "computed_smem_bytes": 384, "constants": [["K_MAX_", 10]], "cta_group": 1, "threads": 128}'))
ir = _decode_capture(_json_loads('{"__ir__": "knn_search_dynamic_tinyd_tile_reduce_partial_0618_c8b9_v1", "arg_keys": ["queries", "database", "partial_distances", "partial_indices", "B", "Q", "M", "K", "num_m_tiles"], "cluster_dims": [1, 1, 1], "computed_smem_bytes": 10240, "constants": [["D_", 3], ["BLOCK_M_", 4096], ["ROWS_PER_WORKER_", 32], ["K_MAX_", 10]], "cta_group": 1, "threads": 128}'))
partial_ir = _decode_capture(_json_loads('{"__ir__": "knn_search_dynamic_tinyd_tile_reduce_partial_0618_c8b9_v1", "arg_keys": ["queries", "database", "partial_distances", "partial_indices", "B", "Q", "M", "K", "num_m_tiles"], "cluster_dims": [1, 1, 1], "computed_smem_bytes": 10240, "constants": [["D_", 3], ["BLOCK_M_", 4096], ["ROWS_PER_WORKER_", 32], ["K_MAX_", 10]], "cta_group": 1, "threads": 128}'))
merge_ir = _decode_capture(_json_loads('{"__ir__": "knn_search_dynamic_tinyd_tile_reduce_merge_0618_c8b9_v1", "arg_keys": ["partial_distances", "partial_indices", "out_distances", "out_indices", "B", "Q", "K", "num_m_tiles", "num_groups", "tiles_per_group"], "cluster_dims": [1, 1, 1], "computed_smem_bytes": 384, "constants": [["K_MAX_", 10]], "cta_group": 1, "threads": 128}'))
d63_partial_ir = _decode_capture(_json_loads('{"__ir__": "knn_search_dynamic_d63_tile_reduce_partial_0618_c8b9_v1", "arg_keys": ["queries", "database", "partial_distances", "partial_indices", "B", "Q", "M", "K", "num_m_tiles"], "cluster_dims": [1, 1, 1], "computed_smem_bytes": 10240, "constants": [["K_MAX_", 10]], "cta_group": 1, "threads": 128}'))

def _config_for_d(d: int) -> dict[str, int]:
    if d != 3:
        raise ValueError(''.join(['knn_search_dynamic_tinyd_d3_tile_reduce_0618_b5b2_v1 unsupported D=', format(d, '')]))
    block_m = 4096
    return {'D_': d, 'BLOCK_M_': block_m, 'ROWS_PER_WORKER_': math.ceil(block_m / THREADS)}

def _compile_kernels(d: int, block_m: int) -> dict[str, Any]:
    from .._dispatch_runtime import generate_kernel
    from .._dispatch_runtime import _cuda_include_dirs
    from .._dispatch_runtime import compile_cuda, detect_gpu_arch
    from .._dispatch_runtime import CUDAKernel
    arch = detect_gpu_arch()
    include_dirs = _cuda_include_dirs()
    if int(d) == 63:
        partial_ir_for_d = knn_search_dynamic_d63_tile_reduce_partial_0618_c8b9_v1
        partial_source = generate_kernel(partial_ir_for_d, validate=False, smem_bytes=TILE_SMEM_BYTES)
    else:
        partial_ir_for_d = knn_search_dynamic_tinyd_tile_reduce_partial_0618_c8b9_v1
        partial_source = generate_kernel(partial_ir_for_d, validate=False, smem_bytes=TILE_SMEM_BYTES, D_=int(d), BLOCK_M_=int(block_m), ROWS_PER_WORKER_=math.ceil(int(block_m) / THREADS))
    merge_source = generate_kernel(knn_search_dynamic_tinyd_tile_reduce_merge_0618_c8b9_v1, validate=False, smem_bytes=MERGE_SMEM_BYTES)
    partial_cubin = compile_cuda(partial_source, arch=arch, options=['--use_fast_math'], include_dirs=include_dirs)
    merge_cubin = compile_cuda(merge_source, arch=arch, options=['--use_fast_math'], include_dirs=include_dirs)
    return {'partial': CUDAKernel(partial_cubin, ''.join(['kernel_', format(partial_ir_for_d.symbol, '')])), 'merge': CUDAKernel(merge_cubin, ''.join(['kernel_', format(knn_search_dynamic_tinyd_tile_reduce_merge_0618_c8b9_v1.symbol, '')]))}

def _scratch(inputs: dict[str, Any], num_m_tiles: int) -> tuple[Any, Any]:
    import torch
    key = (int(inputs['B']), int(inputs['Q']), int(inputs['M']), int(inputs['D']), int(num_m_tiles), int(inputs['queries'].device.index or 0), str(inputs['queries'].dtype))
    cached = _SCRATCH.get(key)
    if cached is None:
        shape = (int(inputs['B']), int(inputs['Q']), int(num_m_tiles), K_MAX)
        cached = (torch.empty(shape, dtype=torch.float32, device=inputs['queries'].device), torch.empty(shape, dtype=torch.int32, device=inputs['queries'].device))
        _SCRATCH[key] = cached
    return cached

def _shape_guard(inputs: dict[str, Any]) -> bool:
    return int(inputs['B']) == 1 and int(inputs['Q']) == 128 and (int(inputs['M']) == 65536) and (int(inputs['K']) == K_MAX) and (int(inputs['D']) in SUPPORTED_D) and (not bool(inputs.get('self_search', False))) and (not bool(inputs.get('force_fallback', False)))

def launch_for_eval(inputs: dict[str, Any]) -> dict[str, Any]:
    import torch
    if not _shape_guard(inputs):
        raise ValueError('knn_search_dynamic_tinyd_d3_tile_reduce_0618_b5b2_v1 supports only B=1,Q=128,M=65536,K=10,D=3, self_search=false')
    d = int(inputs['D'])
    cfg = _config_for_d(d)
    block_m = int(cfg['BLOCK_M_'])
    key = (d, block_m)
    kernels = _KERNELS.get(key)
    if kernels is None:
        kernels = _compile_kernels(d, block_m)
        _KERNELS[key] = kernels
    bsz = int(inputs['B'])
    q_rows = int(inputs['Q'])
    m_rows = int(inputs['M'])
    k = int(inputs['K'])
    num_m_tiles = math.ceil(m_rows / block_m)
    tiles_per_group = max(MERGE_TILES_PER_GROUP, math.ceil(num_m_tiles / MERGE_WARPS))
    num_groups = math.ceil(num_m_tiles / tiles_per_group)
    partial_dist, partial_idx = _scratch(inputs, num_m_tiles)
    kernels['partial'].launch(grid=(bsz * q_rows * num_m_tiles, 1, 1), block=(THREADS, 1, 1), args=[inputs['queries'], inputs['database'], partial_dist, partial_idx, bsz, q_rows, m_rows, k, num_m_tiles], shared_mem=TILE_SMEM_BYTES)
    kernels['merge'].launch(grid=(bsz * q_rows, 1, 1), block=(MERGE_THREADS, 1, 1), args=[partial_dist, partial_idx, inputs['out_distances'], inputs['out_indices'], bsz, q_rows, k, num_m_tiles, num_groups, tiles_per_group], shared_mem=MERGE_SMEM_BYTES)
    return {'distances': inputs['out_distances'], 'indices': inputs['out_indices']}
SHAPE_DISPATCH_REGISTRY = [{'shape_key': 'dynamic_d_tiny_d3_q128_m65536_k10', 'guard': 'B == 1 and Q == 128 and M == 65536 and D == 3 and K == 10 and not self_search', 'candidate_entrypoint': 'loom.examples.weave.knn_search_dynamic_tinyd_d3_tile_reduce_0618_b5b2_v1:launch_for_eval', 'route': ROUTE_TINYD_D3_TILE_REDUCE, 'source_task': 'weave-evolve-knn-search-b5b2', 'replay_scope': 'D3-only'}]

def knn_search_compile_and_launch_dynamic_tinyd_tile_reduce(*, benchmark: bool=True, shapes: list[dict[str, Any]] | None=None) -> dict[str, Any]:
    from .._dispatch_runtime import evaluate
    result = evaluate(launch_for_eval, shapes=TINYD_SHAPES if shapes is None else shapes, benchmark=benchmark)
    result['passed'] = bool(result.get('summary', {}).get('all_correct'))
    print(result)
    return result
