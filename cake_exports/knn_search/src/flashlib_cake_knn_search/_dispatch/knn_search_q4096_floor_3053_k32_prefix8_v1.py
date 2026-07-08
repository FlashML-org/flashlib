"""Round-124/3053 Q4096 floor probe with a K32 prefix8 seed.

Minimum target architecture: sm_100a for the tcgen05 producer path. This
additive wrapper targets the Q4096 floor-repair bucket from round 123. It keeps
the current bbab+9286dynamic dispatcher for all guard misses, but replaces the
exact ``B=1,Q=4096,M=32768,D=128,K=32`` row with a prefix8 partial-list seed.

The K32 seed reuses the validated Q4096/M32768/K64 prefix8 tcgen05 producer
because the scan shape is identical; only the final merge emits 32 ranks into
the contract output.
"""
from __future__ import annotations
from json import loads as _json_loads
from .._dispatch_runtime import _capture_cuTensorMapEncodeTiled, _decode_capture, _import_dispatch_module, _ir_proxy
from typing import Any
from .._dispatch_runtime import select_named_shapes
from . import knn_search_blind_k64_q4096_m32768_prefix8_5132_v1 as prefix8_parent
from . import knn_search_dispatch0618_6bea_plus_31af_bbab_9286dynamic_v1 as parent
K32_TARGET = 32
LOCAL_PREFIX_K = prefix8_parent.LOCAL_PREFIX_K
THREADS = prefix8_parent.THREADS
BLOCK_Q = prefix8_parent.BLOCK_Q
BLOCK_M = prefix8_parent.BLOCK_M
D_STATIC = prefix8_parent.D_STATIC
MERGE_THREADS = prefix8_parent.MERGE_THREADS
MMA_SMEM_BYTES = prefix8_parent.MMA_SMEM_BYTES
MERGE_SMEM_BYTES = prefix8_parent.MERGE_SMEM_BYTES
Q4096_ROWS = prefix8_parent.Q4096_ROWS
Q4096_M32768_ROWS = prefix8_parent.Q4096_M32768_ROWS
Q4096_M32768_SPLIT_M = prefix8_parent.Q4096_M32768_SPLIT_M
Q4096_M32768_PARTIAL_LISTS = prefix8_parent.Q4096_M32768_PARTIAL_LISTS
Q4096_M32768_PREFIX_SPLITS_PER_LANE = prefix8_parent.Q4096_M32768_PREFIX_SPLITS_PER_LANE
ROUTE_Q4096_K32_PREFIX8 = 'round124_3053_q4096_m32768_k32_prefix8'
CONSUMED_K32_PREFIX8_SEED = 'weave-evolve-knn-search-3053-k32-prefix8'
Q4096_FLOOR_LABELS: tuple[str, ...] = ('ksweep_q4096_m20000_d128_k1', 'blind_k32_q4096_m32768_d128_k32', 'blind_k64_q4096_m32768_d128_k64')
TARGET_LABELS: tuple[str, ...] = Q4096_FLOOR_LABELS
TARGET_SHAPES = _decode_capture(_json_loads('[{"__dict_items__": [["label", "ksweep_q4096_m20000_d128_k1"], ["params", {"__dict_items__": [["B", 1], ["Q", 4096], ["M", 20000], ["D", 128], ["K", 1], ["dtype", "bfloat16"], ["seed", 610310], ["self_search", false], ["min_recall", 1.0]]}]]}, {"__dict_items__": [["label", "blind_k32_q4096_m32768_d128_k32"], ["params", {"__dict_items__": [["B", 1], ["Q", 4096], ["M", 32768], ["D", 128], ["K", 32], ["dtype", "bfloat16"], ["seed", 610610], ["self_search", false], ["min_recall", 0.999]]}]]}, {"__dict_items__": [["label", "blind_k64_q4096_m32768_d128_k64"], ["params", {"__dict_items__": [["B", 1], ["Q", 4096], ["M", 32768], ["D", 128], ["K", 64], ["dtype", "bfloat16"], ["seed", 610507], ["self_search", false], ["min_recall", 0.999]]}]]}]'))
K32_PREFIX8_SHAPES = _decode_capture(_json_loads('[{"__dict_items__": [["label", "blind_k32_q4096_m32768_d128_k32"], ["params", {"__dict_items__": [["B", 1], ["Q", 4096], ["M", 32768], ["D", 128], ["K", 32], ["dtype", "bfloat16"], ["seed", 610610], ["self_search", false], ["min_recall", 0.999]]}]]}]'))
_KNN_SEARCH_Q4096_M32768_K32_PREFIX8_KERNELS: dict[str, Any] = {}
knn_search_q4096_m32768_k32_prefix8_merge_3053_v1 = _decode_capture(_json_loads('{"__ir__": "knn_search_q4096_m32768_k32_prefix8_merge_3053_v1", "arg_keys": ["partial_distances", "partial_indices", "out_distances", "out_indices"], "cluster_dims": [1, 1, 1], "computed_smem_bytes": 0, "constants": [["K_OUT_", 32], ["K_PREFIX_", 8]], "cta_group": 1, "threads": 32}'))
partial_ir = _decode_capture(_json_loads('{"__ir__": "knn_search_blind_k64_q4096_m32768_prefix8_partial_5132_v1", "arg_keys": ["queries", "database", "partial_distances", "partial_indices"], "cluster_dims": [1, 1, 1], "computed_smem_bytes": 165120, "constants": [["K_MAX_", 64], ["K_PREFIX_", 8]], "cta_group": 1, "threads": 512}'))
merge_ir = _decode_capture(_json_loads('{"__ir__": "knn_search_q4096_m32768_k32_prefix8_merge_3053_v1", "arg_keys": ["partial_distances", "partial_indices", "out_distances", "out_indices"], "cluster_dims": [1, 1, 1], "computed_smem_bytes": 0, "constants": [["K_OUT_", 32], ["K_PREFIX_", 8]], "cta_group": 1, "threads": 32}'))
ir = _decode_capture(_json_loads('{"__ir__": "knn_search_q4096_m32768_k32_prefix8_merge_3053_v1", "arg_keys": ["partial_distances", "partial_indices", "out_distances", "out_indices"], "cluster_dims": [1, 1, 1], "computed_smem_bytes": 0, "constants": [["K_OUT_", 32], ["K_PREFIX_", 8]], "cta_group": 1, "threads": 32}'))
parent_ir = _decode_capture(_json_loads('{"__ir__": "knn_search_mma_split_partial_v1", "arg_keys": ["queries", "database", "partial_distances", "partial_indices", "B", "Q", "M", "split_m", "num_q_tiles", "total_m_tiles", "tiles_per_split"], "cluster_dims": [1, 1, 1], "computed_smem_bytes": 108800, "constants": [["K_MAX_", 10], ["EXPOSE_COL_COHORTS", 0], ["FULL_M_TILES", 0]], "cta_group": 1, "threads": 640}'))
SHAPE_DISPATCH_REGISTRY: tuple[dict[str, Any], ...] = ({'shape_key': 'round124_3053_q4096_m32768_d128_k32_prefix8', 'label': 'blind_k32_q4096_m32768_d128_k32', 'guard': 'B == 1 and Q == 4096 and M == 32768 and D == 128 and K == 32 and not self_search and not forced_fallback and tcgen05_capable_arch', 'route': ROUTE_Q4096_K32_PREFIX8, 'entrypoint': 'loom.examples.weave.knn_search_q4096_floor_3053_k32_prefix8_v1:launch_for_eval', 'selected_seed': CONSUMED_K32_PREFIX8_SEED, 'source_task': 'weave-evolve-knn-search-3053', 'coverage_class': 'performance_probe_q4096_m32768_d128_k32_prefix8', 'route_source': 'shape-specific-seed'}, *parent.SHAPE_DISPATCH_REGISTRY)

def _compile_k32_prefix8_merge_kernel() -> dict[str, Any]:
    return _decode_capture(_json_loads('{"merge": {"__kernel__": "dispatch_kernel_0434"}}'))

def _ensure_k32_prefix8_kernel() -> dict[str, Any]:
    if not _KNN_SEARCH_Q4096_M32768_K32_PREFIX8_KERNELS:
        _KNN_SEARCH_Q4096_M32768_K32_PREFIX8_KERNELS.update(_compile_k32_prefix8_merge_kernel())
    return _KNN_SEARCH_Q4096_M32768_K32_PREFIX8_KERNELS

def _tcgen05_capable_arch() -> bool:
    return bool(prefix8_parent.parent.base._tcgen05_capable_arch())

def _use_q4096_m32768_k32_prefix8(inputs: dict[str, Any]) -> bool:
    return int(inputs['B']) == 1 and int(inputs['Q']) == Q4096_ROWS and (int(inputs['M']) == Q4096_M32768_ROWS) and (int(inputs['D']) == D_STATIC) and (int(inputs['K']) == K32_TARGET) and (not bool(inputs.get('self_search', False))) and (not bool(inputs.get('force_fallback', False))) and _tcgen05_capable_arch()

def _launch_q4096_m32768_k32_prefix8(inputs: dict[str, Any]) -> dict[str, Any]:
    import torch
    partial_kernels = prefix8_parent._ensure_prefix8_kernels()
    kernels = _ensure_k32_prefix8_kernel()
    partial_dist, partial_idx = prefix8_parent._scratch_prefix8(inputs)
    partial_kernels['partial'].launch(grid=(Q4096_ROWS // BLOCK_Q * Q4096_M32768_SPLIT_M, 1, 1), block=(THREADS, 1, 1), args=[inputs['queries'], inputs['database'], partial_dist, partial_idx], shared_mem=MMA_SMEM_BYTES)
    kernels['merge'].launch(grid=(Q4096_ROWS, 1, 1), block=(MERGE_THREADS, 1, 1), args=[partial_dist, partial_idx, inputs['out_distances'], inputs['out_indices']], shared_mem=MERGE_SMEM_BYTES)
    return {'distances': inputs['out_distances'], 'indices': inputs['out_indices']}

def _guard_order() -> list[str]:
    return [entry['shape_key'] for entry in SHAPE_DISPATCH_REGISTRY]

def selected_route(inputs: dict[str, Any]) -> str:
    if _use_q4096_m32768_k32_prefix8(inputs):
        return ROUTE_Q4096_K32_PREFIX8
    return parent.selected_route(inputs)

def selected_route_name(inputs: dict[str, Any]) -> str:
    return selected_route(inputs)

def route_info(inputs: dict[str, Any]) -> dict[str, Any]:
    if _use_q4096_m32768_k32_prefix8(inputs):
        entry = SHAPE_DISPATCH_REGISTRY[0]
        parent_info = dict(parent.route_info(inputs))
        parent_route = str(parent_info.get('selected_route') or parent_info.get('route') or parent.selected_route(inputs))
        return {'route': ROUTE_Q4096_K32_PREFIX8, 'selected_route': ROUTE_Q4096_K32_PREFIX8, 'selected_entrypoint': entry['entrypoint'], 'parent_route': parent_route, 'replaced_route': parent_route, 'route_kind': 'specialized', 'route_source': entry['route_source'], 'coverage_class': entry['coverage_class'], 'classification': 'seed-produced', 'coverage_only': False, 'production_policy': 'weave_only', 'external_fallback': None, 'guard_order': _guard_order(), 'guard_id': entry['shape_key'], 'forced_fallback': False, 'selected_guard': entry['guard'], 'fallback': parent_route, 'missing_weave_route': False, 'source_task': entry['source_task'], 'selected_seed': entry['selected_seed'], 'selected_seed_task': entry['source_task'], 'expected_seed': entry['selected_seed'], 'producer_seed': 'weave-evolve-knn-search-5132-prefix8', 'prefix_ranks_per_partial': LOCAL_PREFIX_K, 'contract_k': K32_TARGET}
    info = dict(parent.route_info(inputs))
    info['guard_order'] = _guard_order()
    info.setdefault('production_policy', 'weave_only')
    info.setdefault('external_fallback', None)
    return info

def route_trace_entry(label: str, inputs: dict[str, Any]) -> dict[str, Any]:
    return {'label': label, 'B': int(inputs['B']), 'Q': int(inputs['Q']), 'M': int(inputs['M']), 'D': int(inputs['D']), 'K': int(inputs['K']), **route_info(inputs)}

def launch_for_eval(inputs: dict[str, Any]) -> dict[str, Any]:
    if _use_q4096_m32768_k32_prefix8(inputs):
        return _launch_q4096_m32768_k32_prefix8(inputs)
    return parent.launch_for_eval(inputs)

def knn_search_compile_and_launch_q4096_floor_3053(*, benchmark: bool=True, shapes: list[dict[str, Any]] | None=None) -> dict[str, Any]:
    from .._dispatch_runtime import evaluate
    result = evaluate(launch_for_eval, shapes=TARGET_SHAPES if shapes is None else shapes, benchmark=benchmark)
    result['passed'] = bool(result.get('summary', {}).get('all_correct'))
    print(result)
    return result
