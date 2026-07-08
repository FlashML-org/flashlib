"""Round-122/5132 exact Q4096/M32768/K64 prefix8 KNN seed.

Minimum target architecture: sm_100a for the tcgen05 producer path. This
additive bucket seed targets only ``B=1,Q=4096,M=32768,D=128,K=64``. It keeps
the validated 50cc two-tile producer structure for the exact M32768 row, but
stores only the local top-8 entries for each partial list before the final
merge consumes those compact prefixes.
"""
from __future__ import annotations
from json import loads as _json_loads
from .._dispatch_runtime import _capture_cuTensorMapEncodeTiled, _decode_capture, _import_dispatch_module, _ir_proxy
from typing import Any
from .._dispatch_runtime import select_named_shapes
from . import knn_search_blind_k64_q4096_merge16_0614_1968_v1 as merge16_parent
from . import knn_search_blind_k64_twotile_portfolio_0614_50cc_v1 as parent
from . import knn_search_k64_q4096split79_twotile_oddevensort_fastmerge_0612_r34_11c1_v1 as q4096_active
from .knn_search_stream import current_stream_handle
K64_MAX = parent.K64_MAX
LOCAL_PREFIX_K = 8
THREADS = parent.THREADS
BLOCK_Q = parent.BLOCK_Q
BLOCK_M = parent.BLOCK_M
D_STATIC = parent.D_STATIC
MERGE_THREADS = parent.MERGE_THREADS
MMA_POST_MMA_COL_COHORTS = parent.MMA_POST_MMA_COL_COHORTS
MMA_SMEM_BYTES = parent.MMA_SMEM_BYTES
MERGE_SMEM_BYTES = parent.MERGE_SMEM_BYTES
MMA_SMEM_POOL_BYTES = parent.MMA_SMEM_POOL_BYTES
MMA_SMEM_B0_OFFSET = parent.MMA_SMEM_B0_OFFSET
MMA_SMEM_B1_OFFSET = parent.MMA_SMEM_B1_OFFSET
MMA_SMEM_Q_NORM_PART_OFFSET = parent.MMA_SMEM_Q_NORM_PART_OFFSET
MMA_SMEM_DB_NORM_PART0_OFFSET = parent.MMA_SMEM_DB_NORM_PART0_OFFSET
MMA_SMEM_DB_NORM_PART1_OFFSET = parent.MMA_SMEM_DB_NORM_PART1_OFFSET
MMA_SMEM_DB_NORM0_OFFSET = parent.MMA_SMEM_DB_NORM0_OFFSET
MMA_SMEM_DB_NORM1_OFFSET = parent.MMA_SMEM_DB_NORM1_OFFSET
MMA_Q_NORM_PARTS = parent.MMA_Q_NORM_PARTS
Q4096_ROWS = parent.Q4096_ROWS
Q4096_M32768_ROWS = parent.Q4096_M32768_ROWS
Q4096_M32768_SPLIT_M = parent.Q4096_M32768_SPLIT_M
Q4096_M32768_PARTIAL_LISTS = parent.Q4096_M32768_PARTIAL_LISTS
Q4096_M32768_PREFIX_SPLITS_PER_LANE = Q4096_M32768_PARTIAL_LISTS // MERGE_THREADS
ROUTE_Q4096_K64_PREFIX8 = 'round122_5132_q4096_m32768_k64_prefix8'
ROUTE_PARENT = merge16_parent.ROUTE_Q4096_K64_MERGE16
TARGET_LABELS: tuple[str, ...] = ('blind_k64_q4096_m32768_d128_k64',)
TARGET_SHAPES = _decode_capture(_json_loads('[{"__dict_items__": [["label", "blind_k64_q4096_m32768_d128_k64"], ["params", {"__dict_items__": [["B", 1], ["Q", 4096], ["M", 32768], ["D", 128], ["K", 64], ["dtype", "bfloat16"], ["seed", 610507], ["self_search", false], ["min_recall", 0.999]]}]]}]'))
K64_Q4096_M32768_PREFIX8_SHAPES = TARGET_SHAPES
_KNN_SEARCH_Q4096_M32768_PREFIX8_KERNELS: dict[str, Any] = {}
_KNN_SEARCH_Q4096_M32768_PREFIX8_SCRATCH: dict[tuple[int, int, int, int, int, int, int, str, int], tuple[Any, Any]] = {}
knn_search_blind_k64_q4096_m32768_prefix8_partial_5132_v1 = _decode_capture(_json_loads('{"__ir__": "knn_search_blind_k64_q4096_m32768_prefix8_partial_5132_v1", "arg_keys": ["queries", "database", "partial_distances", "partial_indices"], "cluster_dims": [1, 1, 1], "computed_smem_bytes": 165120, "constants": [["K_MAX_", 64], ["K_PREFIX_", 8]], "cta_group": 1, "threads": 512}'))
knn_search_blind_k64_q4096_m32768_prefix8_merge_5132_v1 = _decode_capture(_json_loads('{"__ir__": "knn_search_blind_k64_q4096_m32768_prefix8_merge_5132_v1", "arg_keys": ["partial_distances", "partial_indices", "out_distances", "out_indices"], "cluster_dims": [1, 1, 1], "computed_smem_bytes": 0, "constants": [["K_MAX_", 64], ["K_PREFIX_", 8]], "cta_group": 1, "threads": 32}'))
partial_ir = _decode_capture(_json_loads('{"__ir__": "knn_search_blind_k64_q4096_m32768_prefix8_partial_5132_v1", "arg_keys": ["queries", "database", "partial_distances", "partial_indices"], "cluster_dims": [1, 1, 1], "computed_smem_bytes": 165120, "constants": [["K_MAX_", 64], ["K_PREFIX_", 8]], "cta_group": 1, "threads": 512}'))
merge_ir = _decode_capture(_json_loads('{"__ir__": "knn_search_blind_k64_q4096_m32768_prefix8_merge_5132_v1", "arg_keys": ["partial_distances", "partial_indices", "out_distances", "out_indices"], "cluster_dims": [1, 1, 1], "computed_smem_bytes": 0, "constants": [["K_MAX_", 64], ["K_PREFIX_", 8]], "cta_group": 1, "threads": 32}'))
ir = _decode_capture(_json_loads('{"__ir__": "knn_search_blind_k64_q4096_m32768_prefix8_partial_5132_v1", "arg_keys": ["queries", "database", "partial_distances", "partial_indices"], "cluster_dims": [1, 1, 1], "computed_smem_bytes": 165120, "constants": [["K_MAX_", 64], ["K_PREFIX_", 8]], "cta_group": 1, "threads": 512}'))
SHAPE_DISPATCH_REGISTRY: tuple[dict[str, str], ...] = ({'shape_key': 'round122_5132_q4096_m32768_d128_k64_prefix8', 'guard': 'B == 1 and Q == 4096 and M == 32768 and D == 128 and K == 64 and not self_search and not forced_fallback and tcgen05_capable_arch', 'route': ROUTE_Q4096_K64_PREFIX8, 'entrypoint': 'loom.examples.weave.knn_search_blind_k64_q4096_m32768_prefix8_5132_v1:launch_for_eval', 'source_task': 'weave-evolve-knn-search-5132', 'selected_seed': 'weave-evolve-knn-search-5132-prefix8'}, *merge16_parent.SHAPE_DISPATCH_REGISTRY)

def _compile_prefix8_kernels() -> dict[str, Any]:
    return _decode_capture(_json_loads('{"merge": {"__kernel__": "dispatch_kernel_0436"}, "partial": {"__kernel__": "dispatch_kernel_0435"}}'))

def _ensure_prefix8_kernels() -> dict[str, Any]:
    if not _KNN_SEARCH_Q4096_M32768_PREFIX8_KERNELS:
        _KNN_SEARCH_Q4096_M32768_PREFIX8_KERNELS.update(_compile_prefix8_kernels())
    return _KNN_SEARCH_Q4096_M32768_PREFIX8_KERNELS

def _scratch_prefix8(inputs: dict[str, Any]) -> tuple[Any, Any]:
    import torch
    num_q_tiles = int(inputs['Q']) // BLOCK_Q
    key = (int(inputs['B']), int(inputs['Q']), int(inputs['M']), Q4096_M32768_PARTIAL_LISTS, num_q_tiles, LOCAL_PREFIX_K, int(inputs['queries'].device.index or 0), str(inputs['queries'].dtype), current_stream_handle(inputs))
    cached = _KNN_SEARCH_Q4096_M32768_PREFIX8_SCRATCH.get(key)
    if cached is None:
        shape = (int(inputs['B']), num_q_tiles, Q4096_M32768_PARTIAL_LISTS, BLOCK_Q, LOCAL_PREFIX_K)
        cached = (torch.empty(shape, dtype=torch.float32, device=inputs['queries'].device), torch.empty(shape, dtype=torch.int32, device=inputs['queries'].device))
        _KNN_SEARCH_Q4096_M32768_PREFIX8_SCRATCH[key] = cached
    return cached

def _use_q4096_m32768_k64_prefix8(inputs: dict[str, Any]) -> bool:
    return int(inputs['B']) == 1 and int(inputs['Q']) == Q4096_ROWS and (int(inputs['M']) == Q4096_M32768_ROWS) and (int(inputs['D']) == D_STATIC) and (int(inputs['K']) == K64_MAX) and (not bool(inputs.get('self_search', False))) and (not bool(inputs.get('force_fallback', False))) and parent.base._tcgen05_capable_arch()

def _launch_q4096_m32768_k64_prefix8(inputs: dict[str, Any]) -> dict[str, Any]:
    kernels = _ensure_prefix8_kernels()
    partial_dist, partial_idx = _scratch_prefix8(inputs)
    kernels['partial'].launch(grid=(Q4096_ROWS // BLOCK_Q * Q4096_M32768_SPLIT_M, 1, 1), block=(THREADS, 1, 1), args=[inputs['queries'], inputs['database'], partial_dist, partial_idx], shared_mem=MMA_SMEM_BYTES)
    kernels['merge'].launch(grid=(Q4096_ROWS, 1, 1), block=(MERGE_THREADS, 1, 1), args=[partial_dist, partial_idx, inputs['out_distances'], inputs['out_indices']], shared_mem=MERGE_SMEM_BYTES)
    return {'distances': inputs['out_distances'], 'indices': inputs['out_indices']}

def selected_route(inputs: dict[str, Any]) -> str:
    if _use_q4096_m32768_k64_prefix8(inputs):
        return ROUTE_Q4096_K64_PREFIX8
    return merge16_parent.selected_route_name(inputs)

def selected_route_name(inputs: dict[str, Any]) -> str:
    return selected_route(inputs)

def route_info(inputs: dict[str, Any]) -> dict[str, Any]:
    route = selected_route(inputs)
    if route == ROUTE_Q4096_K64_PREFIX8:
        entry = SHAPE_DISPATCH_REGISTRY[0]
        return {'route': route, 'selected_route': route, 'selected_entrypoint': entry['entrypoint'], 'route_kind': 'specialized', 'route_source': 'shape-specific-seed', 'coverage_class': 'k64_q4096_m32768_prefix8', 'coverage_only': False, 'production_policy': 'weave_only', 'external_fallback': None, 'guard_id': entry['shape_key'], 'selected_guard': entry['guard'], 'guard_order': [item['shape_key'] for item in SHAPE_DISPATCH_REGISTRY], 'selected_seed': entry['selected_seed'], 'selected_seed_task': entry['source_task'], 'parent_route': merge16_parent.selected_route_name(inputs), 'replaced_route': merge16_parent.selected_route_name(inputs)}
    info = {'route': route, 'selected_route': route}
    if hasattr(merge16_parent, 'route_info'):
        info.update(merge16_parent.route_info(inputs))
    info.setdefault('production_policy', 'weave_only')
    info.setdefault('external_fallback', None)
    info['guard_order'] = [item['shape_key'] for item in SHAPE_DISPATCH_REGISTRY]
    return info

def route_trace_entry(label: str, inputs: dict[str, Any]) -> dict[str, Any]:
    return {'label': label, 'B': int(inputs['B']), 'Q': int(inputs['Q']), 'M': int(inputs['M']), 'D': int(inputs['D']), 'K': int(inputs['K']), **route_info(inputs)}

def launch_for_eval(inputs: dict[str, Any]) -> dict[str, Any]:
    if _use_q4096_m32768_k64_prefix8(inputs):
        return _launch_q4096_m32768_k64_prefix8(inputs)
    return merge16_parent.launch_for_eval(inputs)

def knn_search_compile_and_launch_q4096_m32768_prefix8(*, benchmark: bool=True, shapes: list[dict[str, Any]] | None=None) -> dict[str, Any]:
    from .._dispatch_runtime import evaluate
    result = evaluate(launch_for_eval, shapes=K64_Q4096_M32768_PREFIX8_SHAPES if shapes is None else shapes, benchmark=benchmark)
    result['passed'] = bool(result.get('summary', {}).get('all_correct'))
    print(result)
    return result
