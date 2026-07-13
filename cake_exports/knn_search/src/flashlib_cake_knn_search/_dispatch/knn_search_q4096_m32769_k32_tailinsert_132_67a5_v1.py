"""Round-132/67a5 Q4096/M32769/K32 split main-tail seed.

Minimum target architecture: sm_100a for the tcgen05 producer path. This
additive bucket seed targets only ``B=1,Q=4096,M=32769,D=128,K=32``. It reuses
the validated round-124/3053 M32768 prefix8 producer/merge for the hot main
scan, then computes the one-row M tail and inserts it into the contract-visible
top-32 output.
"""
from __future__ import annotations
from json import loads as _json_loads
from .._dispatch_runtime import _capture_cuTensorMapEncodeTiled, _decode_capture, _import_dispatch_module, _ir_proxy
from typing import Any
from . import knn_search_dispatch0618_6bea_plus_31af_bbab_9286dynamic_4832_ffc4residual_598a_k1_v1 as parent
from . import knn_search_q4096_floor_3053_k32_prefix8_v1 as main_seed
K32_TARGET = main_seed.K32_TARGET
LOCAL_PREFIX_K = main_seed.LOCAL_PREFIX_K
THREADS = main_seed.THREADS
BLOCK_Q = main_seed.BLOCK_Q
BLOCK_M = main_seed.BLOCK_M
D_STATIC = main_seed.D_STATIC
MERGE_THREADS = main_seed.MERGE_THREADS
MMA_SMEM_BYTES = main_seed.MMA_SMEM_BYTES
MERGE_SMEM_BYTES = main_seed.MERGE_SMEM_BYTES
Q4096_ROWS = main_seed.Q4096_ROWS
Q4096_M32768_ROWS = main_seed.Q4096_M32768_ROWS
Q4096_M32769_ROWS = Q4096_M32768_ROWS + 1
TAIL_INSERT_THREADS = 32
ROUTE_Q4096_M32769_K32_TAILINSERT = 'round132_67a5_q4096_m32769_k32_prefix8_tailinsert'
CONSUMED_MAIN_SEED = main_seed.CONSUMED_K32_PREFIX8_SEED
CONSUMED_TAILINSERT_SEED = 'weave-evolve-knn-search-132-67a5-k32-tailinsert'
TARGET_LABELS: tuple[str, ...] = ('exp_tail_q4096_m32769_d128_k32',)
TARGET_SHAPES: list[dict[str, Any]] = [{'label': 'exp_tail_q4096_m32769_d128_k32', 'params': {'B': 1, 'Q': Q4096_ROWS, 'M': Q4096_M32769_ROWS, 'D': D_STATIC, 'K': K32_TARGET, 'dtype': 'bfloat16', 'seed': 610610, 'self_search': False, 'min_recall': 0.999}}]
partial_ir = _decode_capture(_json_loads('{"__ir__": "knn_search_blind_k64_q4096_m32768_prefix8_partial_5132_v1", "arg_keys": ["queries", "database", "partial_distances", "partial_indices"], "cluster_dims": [1, 1, 1], "computed_smem_bytes": 165120, "constants": [["K_MAX_", 64], ["K_PREFIX_", 8]], "cta_group": 1, "threads": 512}'))
merge_ir = _decode_capture(_json_loads('{"__ir__": "knn_search_q4096_m32768_k32_prefix8_merge_3053_v1", "arg_keys": ["partial_distances", "partial_indices", "out_distances", "out_indices"], "cluster_dims": [1, 1, 1], "computed_smem_bytes": 0, "constants": [["K_OUT_", 32], ["K_PREFIX_", 8]], "cta_group": 1, "threads": 32}'))
parent_ir = _decode_capture(_json_loads('{"__ir__": "knn_search_mma_split_partial_v1", "arg_keys": ["queries", "database", "partial_distances", "partial_indices", "B", "Q", "M", "split_m", "num_q_tiles", "total_m_tiles", "tiles_per_split"], "cluster_dims": [1, 1, 1], "computed_smem_bytes": 108800, "constants": [["K_MAX_", 10], ["EXPOSE_COL_COHORTS", 0], ["FULL_M_TILES", 0]], "cta_group": 1, "threads": 640}'))
_KNN_SEARCH_Q4096_M32769_K32_TAILINSERT_KERNELS: dict[str, Any] = {}
knn_search_q4096_m32769_k32_tailinsert_132_67a5_v1 = _decode_capture(_json_loads('{"__ir__": "knn_search_q4096_m32769_k32_tailinsert_132_67a5_v1", "arg_keys": ["queries", "database", "out_distances", "out_indices"], "cluster_dims": [1, 1, 1], "computed_smem_bytes": 0, "constants": [["K_OUT_", 32], ["D_", 128], ["M_MAIN_", 32768]], "cta_group": 1, "threads": 32}'))
ir = _decode_capture(_json_loads('{"__ir__": "knn_search_q4096_m32769_k32_tailinsert_132_67a5_v1", "arg_keys": ["queries", "database", "out_distances", "out_indices"], "cluster_dims": [1, 1, 1], "computed_smem_bytes": 0, "constants": [["K_OUT_", 32], ["D_", 128], ["M_MAIN_", 32768]], "cta_group": 1, "threads": 32}'))
SHAPE_DISPATCH_REGISTRY: tuple[dict[str, Any], ...] = ({'shape_key': 'round132_67a5_q4096_m32769_d128_k32_prefix8_tailinsert', 'label': 'exp_tail_q4096_m32769_d128_k32', 'guard': 'B == 1 and Q == 4096 and M == 32769 and D == 128 and K == 32 and not self_search and not forced_fallback and tcgen05_capable_arch', 'route': ROUTE_Q4096_M32769_K32_TAILINSERT, 'entrypoint': 'loom.examples.weave.knn_search_q4096_m32769_k32_tailinsert_132_67a5_v1:launch_for_eval', 'selected_seed': CONSUMED_TAILINSERT_SEED, 'producer_seed': CONSUMED_MAIN_SEED, 'source_task': 'weave-evolve-knn-search-132-67a5', 'coverage_class': 'performance_route_q4096_m32769_d128_k32_prefix8_tailinsert', 'route_source': 'shape-specific-seed'}, *parent.SHAPE_DISPATCH_REGISTRY)

def _compile_tailinsert_kernel() -> dict[str, Any]:
    return _decode_capture(_json_loads('{"tailinsert": {"__kernel__": "dispatch_kernel_0260"}}'))

def _ensure_tailinsert_kernel() -> dict[str, Any]:
    if not _KNN_SEARCH_Q4096_M32769_K32_TAILINSERT_KERNELS:
        _KNN_SEARCH_Q4096_M32769_K32_TAILINSERT_KERNELS.update(_compile_tailinsert_kernel())
    return _KNN_SEARCH_Q4096_M32769_K32_TAILINSERT_KERNELS

def _tcgen05_capable_arch() -> bool:
    return bool(main_seed._tcgen05_capable_arch())

def _use_q4096_m32769_k32_tailinsert(inputs: dict[str, Any]) -> bool:
    return int(inputs['B']) == 1 and int(inputs['Q']) == Q4096_ROWS and (int(inputs['M']) == Q4096_M32769_ROWS) and (int(inputs['D']) == D_STATIC) and (int(inputs['K']) == K32_TARGET) and (not bool(inputs.get('self_search', False))) and (not bool(inputs.get('force_fallback', False))) and _tcgen05_capable_arch()

def _launch_q4096_m32769_k32_tailinsert(inputs: dict[str, Any]) -> dict[str, Any]:
    import torch
    partial_kernels = main_seed.prefix8_parent._ensure_prefix8_kernels()
    merge_kernels = main_seed._ensure_k32_prefix8_kernel()
    tail_kernels = _ensure_tailinsert_kernel()
    partial_dist, partial_idx = main_seed.prefix8_parent._scratch_prefix8(inputs)
    partial_kernels['partial'].launch(grid=(Q4096_ROWS // BLOCK_Q * main_seed.Q4096_M32768_SPLIT_M, 1, 1), block=(THREADS, 1, 1), args=[inputs['queries'], inputs['database'], partial_dist, partial_idx], shared_mem=MMA_SMEM_BYTES)
    merge_kernels['merge'].launch(grid=(Q4096_ROWS, 1, 1), block=(MERGE_THREADS, 1, 1), args=[partial_dist, partial_idx, inputs['out_distances'], inputs['out_indices']], shared_mem=MERGE_SMEM_BYTES)
    tail_kernels['tailinsert'].launch(grid=(Q4096_ROWS, 1, 1), block=(TAIL_INSERT_THREADS, 1, 1), args=[inputs['queries'], inputs['database'], inputs['out_distances'], inputs['out_indices']], shared_mem=0)
    return {'distances': inputs['out_distances'], 'indices': inputs['out_indices']}

def _guard_order() -> list[str]:
    return [entry['shape_key'] for entry in SHAPE_DISPATCH_REGISTRY]

def selected_route(inputs: dict[str, Any]) -> str:
    if _use_q4096_m32769_k32_tailinsert(inputs):
        return ROUTE_Q4096_M32769_K32_TAILINSERT
    return parent.selected_route(inputs)

def selected_route_name(inputs: dict[str, Any]) -> str:
    return selected_route(inputs)

def route_info(inputs: dict[str, Any]) -> dict[str, Any]:
    if _use_q4096_m32769_k32_tailinsert(inputs):
        entry = SHAPE_DISPATCH_REGISTRY[0]
        parent_info = dict(parent.route_info(inputs))
        parent_route = str(parent_info.get('selected_route') or parent_info.get('route') or parent.selected_route(inputs))
        return {'route': ROUTE_Q4096_M32769_K32_TAILINSERT, 'selected_route': ROUTE_Q4096_M32769_K32_TAILINSERT, 'selected_entrypoint': entry['entrypoint'], 'parent_route': parent_route, 'replaced_route': parent_route, 'route_kind': 'specialized', 'route_source': entry['route_source'], 'coverage_class': entry['coverage_class'], 'classification': 'seed-produced', 'coverage_only': False, 'production_policy': 'weave_only', 'external_fallback': None, 'guard_order': _guard_order(), 'guard_id': entry['shape_key'], 'forced_fallback': False, 'selected_guard': entry['guard'], 'fallback': parent_route, 'missing_weave_route': False, 'source_task': entry['source_task'], 'selected_seed': entry['selected_seed'], 'selected_seed_task': entry['source_task'], 'expected_seed': entry['selected_seed'], 'producer_seed': entry['producer_seed'], 'prefix_ranks_per_partial': LOCAL_PREFIX_K, 'main_rows': Q4096_M32768_ROWS, 'tail_rows': Q4096_M32769_ROWS - Q4096_M32768_ROWS, 'contract_k': K32_TARGET}
    info = dict(parent.route_info(inputs))
    info['guard_order'] = _guard_order()
    info.setdefault('production_policy', 'weave_only')
    info.setdefault('external_fallback', None)
    return info

def route_trace_entry(label: str, inputs: dict[str, Any]) -> dict[str, Any]:
    return {'label': label, 'B': int(inputs['B']), 'Q': int(inputs['Q']), 'M': int(inputs['M']), 'D': int(inputs['D']), 'K': int(inputs['K']), **route_info(inputs)}

def launch_for_eval(inputs: dict[str, Any]) -> dict[str, Any]:
    if _use_q4096_m32769_k32_tailinsert(inputs):
        return _launch_q4096_m32769_k32_tailinsert(inputs)
    return parent.launch_for_eval(inputs)

def knn_search_compile_and_launch_q4096_m32769_k32_tailinsert(*, benchmark: bool=True, shapes: list[dict[str, Any]] | None=None) -> dict[str, Any]:
    from .._dispatch_runtime import evaluate
    result = evaluate(launch_for_eval, shapes=TARGET_SHAPES if shapes is None else shapes, benchmark=benchmark)
    result['passed'] = bool(result.get('summary', {}).get('all_correct'))
    print(result)
    return result
