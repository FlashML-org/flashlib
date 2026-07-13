"""Exact tiny-D D1/D5 tile-reduce seed for extended kNN blind spots.

Minimum target architecture: sm_80. This additive bucket seed targets
``B=1,Q=128,M=65536,K=10,D in {1,5}`` from the 0618 extended dynamic-D
blind-spot suite. It reuses the existing CUDA-core tiny-D tile-reduce producer
with ``D_`` compiled to the runtime feature dimension, preserving a stable
distance/index top-k tie policy for the D1 near-tie-heavy contract row.
"""
from __future__ import annotations
from json import loads as _json_loads
from .._dispatch_runtime import _capture_cuTensorMapEncodeTiled, _decode_capture, _import_dispatch_module, _ir_proxy
import math
from typing import Any
from .._dispatch_runtime import select_named_shapes
from . import knn_search_dynamic_tinyd_d3_tile_reduce_0618_b5b2_v1 as base
THREADS = base.THREADS
MERGE_THREADS = base.MERGE_THREADS
MERGE_WARPS = base.MERGE_WARPS
K_MAX = base.K_MAX
TILE_SMEM_BYTES = base.TILE_SMEM_BYTES
MERGE_SMEM_BYTES = base.MERGE_SMEM_BYTES
MERGE_TILES_PER_GROUP = base.MERGE_TILES_PER_GROUP
SUPPORTED_D = {1, 5}
EXT_TINYD_LABELS: tuple[str, ...] = ('blind_ext_dyn_d1_q128_m65536_k10', 'blind_ext_dyn_d5_q128_m65536_k10')
EXT_TINYD_SHAPES = _decode_capture(_json_loads('[{"__dict_items__": [["label", "blind_ext_dyn_d1_q128_m65536_k10"], ["params", {"__dict_items__": [["B", 1], ["Q", 128], ["M", 65536], ["D", 1], ["K", 10], ["dtype", "bfloat16"], ["seed", 610913], ["self_search", false], ["min_recall", 0.999]]}]]}, {"__dict_items__": [["label", "blind_ext_dyn_d5_q128_m65536_k10"], ["params", {"__dict_items__": [["B", 1], ["Q", 128], ["M", 65536], ["D", 5], ["K", 10], ["dtype", "bfloat16"], ["seed", 610914], ["self_search", false], ["min_recall", 0.999]]}]]}]'))
ROUTE_EXT_TINYD_D1D5_TILE_REDUCE = '9286_ext_dynamic_d1d5_q128_tile_reduce'
CONSUMED_SEED = 'weave-evolve-knn-search-9286-d1d5-tile-reduce'
ir = _decode_capture(_json_loads('{"__ir__": "knn_search_dynamic_tinyd_tile_reduce_partial_0618_c8b9_v1", "arg_keys": ["queries", "database", "partial_distances", "partial_indices", "B", "Q", "M", "K", "num_m_tiles"], "cluster_dims": [1, 1, 1], "computed_smem_bytes": 10240, "constants": [["D_", 3], ["BLOCK_M_", 4096], ["ROWS_PER_WORKER_", 32], ["K_MAX_", 10]], "cta_group": 1, "threads": 128}'))
partial_ir = _decode_capture(_json_loads('{"__ir__": "knn_search_dynamic_tinyd_tile_reduce_partial_0618_c8b9_v1", "arg_keys": ["queries", "database", "partial_distances", "partial_indices", "B", "Q", "M", "K", "num_m_tiles"], "cluster_dims": [1, 1, 1], "computed_smem_bytes": 10240, "constants": [["D_", 3], ["BLOCK_M_", 4096], ["ROWS_PER_WORKER_", 32], ["K_MAX_", 10]], "cta_group": 1, "threads": 128}'))
merge_ir = _decode_capture(_json_loads('{"__ir__": "knn_search_dynamic_tinyd_tile_reduce_merge_0618_c8b9_v1", "arg_keys": ["partial_distances", "partial_indices", "out_distances", "out_indices", "B", "Q", "K", "num_m_tiles", "num_groups", "tiles_per_group"], "cluster_dims": [1, 1, 1], "computed_smem_bytes": 384, "constants": [["K_MAX_", 10]], "cta_group": 1, "threads": 128}'))
_KERNELS: dict[tuple[int, int], dict[str, Any]] = {}
_SCRATCH: dict[tuple[int, int, int, int, int, int, str], tuple[Any, Any]] = {}

def _config_for_d(d: int) -> dict[str, int]:
    if d not in SUPPORTED_D:
        raise ValueError(''.join(['knn_search_dynamic_d1d5_q128_m65536_tile_reduce_0618_9286_v1 supports only D in ', format(sorted(SUPPORTED_D), ''), ', got D=', format(d, '')]))
    block_m = 4096
    return {'D_': int(d), 'BLOCK_M_': block_m, 'ROWS_PER_WORKER_': math.ceil(block_m / THREADS)}

def _compile_kernels(d: int, block_m: int) -> dict[str, Any]:
    return base._compile_kernels(int(d), int(block_m))

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

def selected_route(inputs: dict[str, Any]) -> str:
    if _shape_guard(inputs):
        return ROUTE_EXT_TINYD_D1D5_TILE_REDUCE
    return 'unsupported_shape'

def selected_route_name(inputs: dict[str, Any]) -> str:
    return selected_route(inputs)

def route_info(inputs: dict[str, Any]) -> dict[str, Any]:
    route = selected_route(inputs)
    if route == ROUTE_EXT_TINYD_D1D5_TILE_REDUCE:
        return {'route': route, 'selected_route': route, 'selected_entrypoint': 'loom.examples.weave.knn_search_dynamic_d1d5_q128_m65536_tile_reduce_0618_9286_v1:launch_for_eval', 'route_kind': 'specialized', 'route_source': 'bucket-specific-seed', 'coverage_class': 'bucket_seed_ext_dynamic_d1d5_q128_m65536_k10', 'classification': 'seed-produced', 'coverage_only': False, 'production_policy': 'weave_only', 'external_fallback': None, 'guard_id': '9286_ext_dynamic_d1d5_q128_m65536_k10', 'forced_fallback': False, 'selected_guard': SHAPE_DISPATCH_REGISTRY[0]['guard'], 'fallback': None, 'missing_weave_route': False, 'selected_seed': CONSUMED_SEED}
    return {'route': route, 'selected_route': route, 'route_kind': 'unsupported', 'route_source': None, 'coverage_only': False, 'production_policy': 'weave_only', 'external_fallback': None, 'forced_fallback': bool(inputs.get('force_fallback', False)), 'missing_weave_route': True}

def launch_for_eval(inputs: dict[str, Any]) -> dict[str, Any]:
    import torch
    if not _shape_guard(inputs):
        raise ValueError('knn_search_dynamic_d1d5_q128_m65536_tile_reduce_0618_9286_v1 supports only B=1,Q=128,M=65536,K=10,D in {1,5}, self_search=false')
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
SHAPE_DISPATCH_REGISTRY = [{'shape_key': 'ext_dynamic_d1d5_q128_m65536_k10', 'guard': 'B == 1 and Q == 128 and M == 65536 and D in {1,5} and K == 10 and not self_search and not forced_fallback', 'candidate_entrypoint': 'loom.examples.weave.knn_search_dynamic_d1d5_q128_m65536_tile_reduce_0618_9286_v1:launch_for_eval', 'route': ROUTE_EXT_TINYD_D1D5_TILE_REDUCE, 'selected_seed': CONSUMED_SEED, 'source_task': 'weave-evolve-knn-search-9286-d1d5', 'workflow_mode': 'generalize_auto_tuning', 'auto_tuning_stage': 'bucket-kernel'}]

def knn_search_compile_and_launch_dynamic_d1d5_tile_reduce(*, benchmark: bool=True, shapes: list[dict[str, Any]] | None=None) -> dict[str, Any]:
    from .._dispatch_runtime import evaluate
    result = evaluate(launch_for_eval, shapes=EXT_TINYD_SHAPES if shapes is None else shapes, benchmark=benchmark)
    result['passed'] = bool(result.get('summary', {}).get('all_correct'))
    print(result)
    return result
