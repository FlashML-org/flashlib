"""Round-19/50cc blind-K64 two-tile portfolio seed.

Minimum target architecture: sm_100a for the tcgen05 producer path. This
additive seed repairs blind K64 dispatcher holes without changing the
production dispatcher: exact Q64/M131072/K64 uses a guarded two-tile producer
plus the proven Q128 hierarchical K64 merge, exact Q512/M65536/K64 and
Q4096/M32768/K64 use the same guarded producer plus a dynamic 32-head K64
merge, and all other shapes delegate to the round-27 K64 parent.
"""
from __future__ import annotations
from json import loads as _json_loads
from .._dispatch_runtime import _capture_cuTensorMapEncodeTiled, _decode_capture, _import_dispatch_module, _ir_proxy
import math
from typing import Any
from . import knn_search_k32_mma_capacity_0611_r12_v1 as base
from . import knn_search_k64_q128m65536_twotileproducer_kexact_0614_r27_k64thin_v1 as parent
from . import knn_search_k64_q128split512_hiermerge32_0613_r43_11c1_v1 as q128_parent
from . import knn_search_k64_q128split512_hiermerge32_kexact_0614_r25_k64thin_v1 as q128_kexact
from . import knn_search_k64_q4096split79_twotile_oddevensort_fastmerge_0612_r34_11c1_v1 as q4096_active
from . import knn_search_k64_q4096split80_twotile_distanceonly_stagingunrolled_0612_r31_11c1_v1 as q4096_producer
K64_MAX = parent.K64_MAX
THREADS = parent.THREADS
BLOCK_Q = parent.BLOCK_Q
BLOCK_M = parent.BLOCK_M
D_STATIC = parent.D_STATIC
MERGE_THREADS = parent.MERGE_THREADS
MMA_POST_MMA_COL_COHORTS = parent.MMA_POST_MMA_COL_COHORTS
MMA_SMEM_BYTES = parent.MMA_SMEM_BYTES
MERGE_SMEM_BYTES = parent.MERGE_SMEM_BYTES
MMA_SMEM_POOL_BYTES = base.MMA_SMEM_POOL_BYTES
MMA_SMEM_B0_OFFSET = base.MMA_SMEM_B0_OFFSET
MMA_SMEM_B1_OFFSET = base.MMA_SMEM_B1_OFFSET
MMA_SMEM_Q_NORM_PART_OFFSET = base.MMA_SMEM_Q_NORM_PART_OFFSET
MMA_SMEM_DB_NORM_PART0_OFFSET = base.MMA_SMEM_DB_NORM_PART0_OFFSET
MMA_SMEM_DB_NORM_PART1_OFFSET = base.MMA_SMEM_DB_NORM_PART1_OFFSET
MMA_SMEM_DB_NORM0_OFFSET = base.MMA_SMEM_DB_NORM0_OFFSET
MMA_SMEM_DB_NORM1_OFFSET = base.MMA_SMEM_DB_NORM1_OFFSET
MMA_Q_NORM_PARTS = base.MMA_Q_NORM_PARTS
MMA_STAGE_VEC_ELEMS = base.MMA_STAGE_VEC_ELEMS
MMA_STAGE_PACK_WORDS = base.MMA_STAGE_PACK_WORDS
MMA_DB_NORM_PARTS = base.MMA_DB_NORM_PARTS
MMA_DB_NORM_CHUNK = base.MMA_DB_NORM_CHUNK
Q64_ROWS = 64
Q64_M_ROWS = 131072
Q64_SPLIT_M = 512
Q64_PARTIAL_LISTS = Q64_SPLIT_M * MMA_POST_MMA_COL_COHORTS
Q512_ROWS = 512
Q512_M_ROWS = 65536
Q512_SPLIT_M = 256
Q512_PARTIAL_LISTS = Q512_SPLIT_M * MMA_POST_MMA_COL_COHORTS
Q4096_ROWS = 4096
Q4096_M32768_ROWS = 32768
Q4096_M32768_SPLIT_M = 128
Q4096_M32768_PARTIAL_LISTS = Q4096_M32768_SPLIT_M * MMA_POST_MMA_COL_COHORTS
HIGHQ_MERGE_SPLITS_PER_LANE_MAX = Q512_PARTIAL_LISTS // MERGE_THREADS
_KNN_SEARCH_BLIND_K64_KERNELS: dict[str, Any] = {}
_KNN_SEARCH_BLIND_K64_SCRATCH: dict[tuple[int, int, int, int, int, int, str], tuple[Any, Any]] = {}
_knn_stage_query_vec_guarded_k64 = _ir_proxy('loom.examples.weave.knn_search_blind_k64_twotile_portfolio_0614_50cc_v1:_knn_stage_query_vec_guarded_k64', 256)
_knn_stage_database_tile_guarded_k64 = _ir_proxy('loom.examples.weave.knn_search_blind_k64_twotile_portfolio_0614_50cc_v1:_knn_stage_database_tile_guarded_k64', 256)
knn_search_blind_k64_twotile_partial_0614_50cc_v1 = _decode_capture(_json_loads('{"__ir__": "knn_search_blind_k64_twotile_partial_0614_50cc_v1", "arg_keys": ["queries", "database", "partial_distances", "partial_indices", "B", "Q", "M", "split_m", "num_q_tiles", "total_m_tiles", "tiles_per_split", "partial_list_count"], "cluster_dims": [1, 1, 1], "computed_smem_bytes": 165120, "constants": [["K_MAX_", 64]], "cta_group": 1, "threads": 512}'))
knn_search_blind_k64_highq_merge32_0614_50cc_v1 = _decode_capture(_json_loads('{"__ir__": "knn_search_blind_k64_highq_merge32_0614_50cc_v1", "arg_keys": ["partial_distances", "partial_indices", "out_distances", "out_indices", "B", "Q", "K", "partial_list_count", "num_q_tiles"], "cluster_dims": [1, 1, 1], "computed_smem_bytes": 0, "constants": [["K_MAX_", 64]], "cta_group": 1, "threads": 32}'))
partial_ir = _decode_capture(_json_loads('{"__ir__": "knn_search_blind_k64_twotile_partial_0614_50cc_v1", "arg_keys": ["queries", "database", "partial_distances", "partial_indices", "B", "Q", "M", "split_m", "num_q_tiles", "total_m_tiles", "tiles_per_split", "partial_list_count"], "cluster_dims": [1, 1, 1], "computed_smem_bytes": 165120, "constants": [["K_MAX_", 64]], "cta_group": 1, "threads": 512}'))
merge_ir = _decode_capture(_json_loads('{"__ir__": "knn_search_blind_k64_highq_merge32_0614_50cc_v1", "arg_keys": ["partial_distances", "partial_indices", "out_distances", "out_indices", "B", "Q", "K", "partial_list_count", "num_q_tiles"], "cluster_dims": [1, 1, 1], "computed_smem_bytes": 0, "constants": [["K_MAX_", 64]], "cta_group": 1, "threads": 32}'))
q64_group_merge_ir = _decode_capture(_json_loads('{"__ir__": "knn_search_k64_q128split512_groupmerge64_kexact_0614_r25_k64thin_v1", "arg_keys": ["partial_distances", "partial_indices", "group_distances", "group_indices", "B", "Q", "K", "num_q_tiles"], "cluster_dims": [1, 1, 1], "computed_smem_bytes": 0, "constants": [["K_MAX_", 64]], "cta_group": 1, "threads": 32}'))
q64_final_merge_ir = _decode_capture(_json_loads('{"__ir__": "knn_search_k64_q128split512_finalmerge32_kexact_0614_r25_k64thin_v1", "arg_keys": ["group_distances", "group_indices", "out_distances", "out_indices", "B", "Q", "K"], "cluster_dims": [1, 1, 1], "computed_smem_bytes": 0, "constants": [["K_MAX_", 64]], "cta_group": 1, "threads": 32}'))
ir = _decode_capture(_json_loads('{"__ir__": "knn_search_blind_k64_twotile_partial_0614_50cc_v1", "arg_keys": ["queries", "database", "partial_distances", "partial_indices", "B", "Q", "M", "split_m", "num_q_tiles", "total_m_tiles", "tiles_per_split", "partial_list_count"], "cluster_dims": [1, 1, 1], "computed_smem_bytes": 165120, "constants": [["K_MAX_", 64]], "cta_group": 1, "threads": 512}'))
ROUTE_Q64_K64 = 'round19_50cc_q64_m131072_k64_twotile_hiermerge32'
ROUTE_Q512_K64 = 'round19_50cc_q512_m65536_k64_twotile_merge32'
ROUTE_Q4096_K64 = 'round19_50cc_q4096_m32768_k64_twotile_merge32'
ROUTE_PARENT = 'round27_k64_parent'
SHAPE_DISPATCH_REGISTRY = _decode_capture(_json_loads('{"__tuple__": [{"__dict_items__": [["shape_key", "B1_Q64_M131072_D128_K64"], ["guard", "B == 1 and Q == 64 and M == 131072 and D == 128 and K == 64 and tcgen05_capable_arch"], ["route", "round19_50cc_q64_m131072_k64_twotile_hiermerge32"]]}, {"__dict_items__": [["shape_key", "B1_Q512_M65536_D128_K64"], ["guard", "B == 1 and Q == 512 and M == 65536 and D == 128 and K == 64 and tcgen05_capable_arch"], ["route", "round19_50cc_q512_m65536_k64_twotile_merge32"]]}, {"__dict_items__": [["shape_key", "B1_Q4096_M32768_D128_K64"], ["guard", "B == 1 and Q == 4096 and M == 32768 and D == 128 and K == 64 and tcgen05_capable_arch"], ["route", "round19_50cc_q4096_m32768_k64_twotile_merge32"]]}]}'))

def _compile_blind_k64_kernels() -> dict[str, Any]:
    return _decode_capture(_json_loads('{"merge": {"__kernel__": "dispatch_kernel_0040"}, "partial": {"__kernel__": "dispatch_kernel_0039"}, "q64_final_merge": {"__kernel__": "dispatch_kernel_0042"}, "q64_group_merge": {"__kernel__": "dispatch_kernel_0041"}}'))

def _ensure_kernels() -> dict[str, Any]:
    if not _KNN_SEARCH_BLIND_K64_KERNELS:
        _KNN_SEARCH_BLIND_K64_KERNELS.update(_compile_blind_k64_kernels())
    return _KNN_SEARCH_BLIND_K64_KERNELS

def _partial_scratch(inputs: dict[str, Any], partial_list_count: int, num_q_tiles: int) -> tuple[Any, Any]:
    import torch
    key = (int(inputs['B']), int(inputs['Q']), int(inputs['M']), int(partial_list_count), int(num_q_tiles), int(inputs['queries'].device.index or 0), str(inputs['queries'].dtype))
    cached = _KNN_SEARCH_BLIND_K64_SCRATCH.get(key)
    if cached is None:
        shape = (int(inputs['B']), int(num_q_tiles), int(partial_list_count), BLOCK_Q, K64_MAX)
        cached = (torch.empty(shape, dtype=torch.float32, device=inputs['queries'].device), torch.empty(shape, dtype=torch.int32, device=inputs['queries'].device))
        _KNN_SEARCH_BLIND_K64_SCRATCH[key] = cached
    return cached

def _tcgen05_shape(inputs: dict[str, Any]) -> bool:
    return int(inputs['B']) == 1 and int(inputs['D']) == D_STATIC and (int(inputs['K']) == K64_MAX) and base._tcgen05_capable_arch()

def _use_q64_k64(inputs: dict[str, Any]) -> bool:
    return _tcgen05_shape(inputs) and int(inputs['Q']) == Q64_ROWS and (int(inputs['M']) == Q64_M_ROWS)

def _use_q512_k64(inputs: dict[str, Any]) -> bool:
    return _tcgen05_shape(inputs) and int(inputs['Q']) == Q512_ROWS and (int(inputs['M']) == Q512_M_ROWS)

def _use_q4096_m32768_k64(inputs: dict[str, Any]) -> bool:
    return _tcgen05_shape(inputs) and int(inputs['Q']) == Q4096_ROWS and (int(inputs['M']) == Q4096_M32768_ROWS)

def _launch_partial(inputs: dict[str, Any], *, split_m: int, partial_list_count: int) -> tuple[Any, Any, int, int, int, int]:
    kernels = _ensure_kernels()
    bsz = int(inputs['B'])
    q_rows = int(inputs['Q'])
    m_rows = int(inputs['M'])
    num_q_tiles = math.ceil(q_rows / BLOCK_Q)
    total_m_tiles = math.ceil(m_rows / BLOCK_M)
    tiles_per_split = math.ceil(total_m_tiles / split_m)
    partial_dist, partial_idx = _partial_scratch(inputs, partial_list_count, num_q_tiles)
    kernels['partial'].launch(grid=(bsz * num_q_tiles * split_m, 1, 1), block=(THREADS, 1, 1), args=[inputs['queries'], inputs['database'], partial_dist, partial_idx, bsz, q_rows, m_rows, split_m, num_q_tiles, total_m_tiles, tiles_per_split, partial_list_count], shared_mem=MMA_SMEM_BYTES)
    return (partial_dist, partial_idx, bsz, q_rows, int(inputs['K']), num_q_tiles)

def _launch_q64_k64(inputs: dict[str, Any]) -> dict[str, Any]:
    import torch
    kernels = _ensure_kernels()
    partial_dist, partial_idx, bsz, q_rows, k, num_q_tiles = _launch_partial(inputs, split_m=Q64_SPLIT_M, partial_list_count=Q64_PARTIAL_LISTS)
    group_dist, group_idx = q128_parent._group_scratch(inputs)
    kernels['q64_group_merge'].launch(grid=(bsz * q_rows * q128_parent.HIERMERGE_GROUPS, 1, 1), block=(MERGE_THREADS, 1, 1), args=[partial_dist, partial_idx, group_dist, group_idx, bsz, q_rows, k, num_q_tiles], shared_mem=MERGE_SMEM_BYTES)
    kernels['q64_final_merge'].launch(grid=(bsz * q_rows, 1, 1), block=(MERGE_THREADS, 1, 1), args=[group_dist, group_idx, inputs['out_distances'], inputs['out_indices'], bsz, q_rows, k], shared_mem=MERGE_SMEM_BYTES)
    return {'distances': inputs['out_distances'], 'indices': inputs['out_indices']}

def _launch_highq_k64(inputs: dict[str, Any], *, split_m: int, partial_list_count: int) -> dict[str, Any]:
    import torch
    kernels = _ensure_kernels()
    partial_dist, partial_idx, bsz, q_rows, k, num_q_tiles = _launch_partial(inputs, split_m=split_m, partial_list_count=partial_list_count)
    kernels['merge'].launch(grid=(bsz * q_rows, 1, 1), block=(MERGE_THREADS, 1, 1), args=[partial_dist, partial_idx, inputs['out_distances'], inputs['out_indices'], bsz, q_rows, k, partial_list_count, num_q_tiles], shared_mem=MERGE_SMEM_BYTES)
    return {'distances': inputs['out_distances'], 'indices': inputs['out_indices']}

def selected_route_name(inputs: dict[str, Any]) -> str:
    if _use_q64_k64(inputs):
        return ROUTE_Q64_K64
    if _use_q512_k64(inputs):
        return ROUTE_Q512_K64
    if _use_q4096_m32768_k64(inputs):
        return ROUTE_Q4096_K64
    return parent.selected_route_name(inputs)

def route_info(inputs: dict[str, Any]) -> dict[str, Any]:
    route = selected_route_name(inputs)
    if route in {ROUTE_Q64_K64, ROUTE_Q512_K64, ROUTE_Q4096_K64}:
        registry = next((entry for entry in SHAPE_DISPATCH_REGISTRY if entry['route'] == route))
        return {'route': route, 'selected_route': route, 'selected_entrypoint': 'loom.examples.weave.knn_search_blind_k64_twotile_portfolio_0614_50cc_v1:launch_for_eval', 'route_kind': 'specialized', 'coverage_only': False, 'production_policy': 'weave_only', 'external_fallback': None, 'selected_guard': registry['guard'], 'guard_order': [entry['shape_key'] for entry in SHAPE_DISPATCH_REGISTRY], 'parent_route': parent.selected_route_name(inputs)}
    info = {'route': route}
    if hasattr(parent, 'route_info'):
        info.update(parent.route_info(inputs))
    info.setdefault('production_policy', 'weave_only')
    info.setdefault('external_fallback', None)
    info['guard_order'] = [entry['shape_key'] for entry in SHAPE_DISPATCH_REGISTRY]
    return info

def route_trace_entry(label: str, inputs: dict[str, Any]) -> dict[str, Any]:
    return {'label': label, 'B': int(inputs['B']), 'Q': int(inputs['Q']), 'M': int(inputs['M']), 'D': int(inputs['D']), 'K': int(inputs['K']), **route_info(inputs)}

def launch_for_eval(inputs: dict[str, Any]) -> dict[str, Any]:
    if _use_q64_k64(inputs):
        return _launch_q64_k64(inputs)
    if _use_q512_k64(inputs):
        return _launch_highq_k64(inputs, split_m=Q512_SPLIT_M, partial_list_count=Q512_PARTIAL_LISTS)
    if _use_q4096_m32768_k64(inputs):
        return _launch_highq_k64(inputs, split_m=Q4096_M32768_SPLIT_M, partial_list_count=Q4096_M32768_PARTIAL_LISTS)
    return parent.launch_for_eval(inputs)

def _select_contract_shapes(shape_labels: str | tuple[str, ...] | list[str] | None):
    if shape_labels is None:
        return None
    from .._dispatch_runtime import select_named_shapes
    return select_named_shapes(shape_labels)

def knn_search_compile_and_launch_blind_k64_50cc(*, benchmark: bool=True, shape_labels: str | tuple[str, ...] | list[str] | None=None) -> dict[str, Any]:
    from .._dispatch_runtime import evaluate
    labels = ('blind_k64_q64_m131072_d128_k64', 'blind_k64_q128_m65536_d128_k64', 'blind_k64_q512_m65536_d128_k64', 'blind_k64_q4096_m32768_d128_k64')
    selected = _select_contract_shapes(labels if shape_labels is None else shape_labels)
    result = evaluate(launch_for_eval, shapes=selected, benchmark=benchmark)
    result['passed'] = bool(result.get('summary', {}).get('all_correct'))
    print(result)
    return result
