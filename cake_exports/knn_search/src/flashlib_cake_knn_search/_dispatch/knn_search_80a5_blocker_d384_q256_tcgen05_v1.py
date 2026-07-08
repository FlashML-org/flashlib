"""Round-80a5 D384/Q256 tcgen05 seed for exact BF16 squared-L2 kNN.

Minimum target architecture: sm_100a for the D384 tcgen05 route. This additive
shape seed targets only ``B=1,Q=256,M=32768,D=384,K=10,self_search=False``.
The producer evaluates each database tile as three 128-wide BF16 MMA passes
into the same accumulator, then reuses the existing split-148 merge consumer to
produce contract distances and indices. Guard misses delegate to the current
default afe6 dispatcher unchanged.
"""
from __future__ import annotations
from json import loads as _json_loads
from .._dispatch_runtime import _capture_cuTensorMapEncodeTiled, _decode_capture, _import_dispatch_module, _ir_proxy
import math
from typing import Any
from .._dispatch_runtime import select_named_shapes
from . import knn_search_dispatch0617_default_afe6_v1 as parent
from . import knn_search_mma_split_v1 as mma
THREADS = mma.THREADS
BLOCK_Q = mma.BLOCK_Q
BLOCK_M = mma.BLOCK_M
D_TOTAL = 384
D_STAGE = mma.D_STATIC
K_MAX = mma.K_MAX
TARGET_Q = 256
TARGET_M = 32768
MMA_POST_MMA_COL_COHORTS = mma.MMA_POST_MMA_COL_COHORTS
MMA_STAGE_VEC_ELEMS = mma.MMA_STAGE_VEC_ELEMS
MMA_STAGE_PACK_WORDS = mma.MMA_STAGE_PACK_WORDS
MMA_Q_NORM_PARTS = D_TOTAL // MMA_STAGE_VEC_ELEMS
MMA_Q_NORM_VECS = BLOCK_Q * D_TOTAL // MMA_STAGE_VEC_ELEMS
MMA_DB_NORM_PARTS_PER_HALF = mma.MMA_DB_NORM_PARTS
MMA_DB_NORM_CHUNK = D_STAGE // MMA_DB_NORM_PARTS_PER_HALF
MMA_DB_NORM_PART_VECS = MMA_DB_NORM_CHUNK // MMA_STAGE_VEC_ELEMS
MMA_SMEM_A_BYTES = BLOCK_Q * D_STAGE * 2
MMA_SMEM_B_BYTES = BLOCK_M * D_STAGE * 2
MMA_SMEM_Q_NORM_PART_BYTES = BLOCK_Q * MMA_Q_NORM_PARTS * 4
MMA_SMEM_DB_NORM_PART_BYTES = BLOCK_M * MMA_DB_NORM_PARTS_PER_HALF * 4
MMA_SMEM_DB_NORM_BYTES = BLOCK_M * 4
MMA_COHORT_TOPK_BYTES = BLOCK_Q * K_MAX * 4
MMA_SMEM_B_OFFSET = MMA_SMEM_A_BYTES
MMA_SMEM_Q_NORM_PART_OFFSET = MMA_SMEM_B_OFFSET + MMA_SMEM_B_BYTES
MMA_SMEM_DB_NORM_PART0_OFFSET = MMA_SMEM_Q_NORM_PART_OFFSET + MMA_SMEM_Q_NORM_PART_BYTES
MMA_SMEM_DB_NORM_PART1_OFFSET = MMA_SMEM_DB_NORM_PART0_OFFSET + MMA_SMEM_DB_NORM_PART_BYTES
MMA_SMEM_DB_NORM_PART2_OFFSET = MMA_SMEM_DB_NORM_PART1_OFFSET + MMA_SMEM_DB_NORM_PART_BYTES
MMA_SMEM_DB_NORM_OFFSET = MMA_SMEM_DB_NORM_PART2_OFFSET + MMA_SMEM_DB_NORM_PART_BYTES
MMA_COHORT_TOPK_D_OFFSET = MMA_SMEM_DB_NORM_OFFSET + MMA_SMEM_DB_NORM_BYTES
MMA_COHORT_TOPK_I_OFFSET = MMA_COHORT_TOPK_D_OFFSET + MMA_POST_MMA_COL_COHORTS * MMA_COHORT_TOPK_BYTES
MMA_SMEM_POOL_BYTES = MMA_COHORT_TOPK_I_OFFSET + MMA_POST_MMA_COL_COHORTS * MMA_COHORT_TOPK_BYTES + 256
MMA_SMEM_BYTES = MMA_SMEM_POOL_BYTES + mma.WEAVE_SMEM_SYSTEM_BYTES
MERGE_THREADS = mma.MERGE_THREADS
MERGE_SMEM_BYTES = mma.MERGE_SMEM_BYTES
D384_SPLIT_M = mma.Q128_SPLIT_M
ROUTE_D384_Q256_TCGEN05 = 'round80a5_d384_q256_m32768_k10_tcgen05_split148'
ROUTE_PARENT_DEFAULT_AFE6 = parent.PROFILE_ALL
partial_ir = _decode_capture(_json_loads('{"__ir__": "knn_search_d384_mma_split_partial_0612_r34_v1", "arg_keys": ["queries", "database", "partial_distances", "partial_indices", "B", "Q", "M", "split_m", "num_q_tiles", "total_m_tiles", "tiles_per_split"], "cluster_dims": [1, 1, 1], "computed_smem_bytes": 126720, "constants": [["K_MAX_", 10]], "cta_group": 1, "threads": 640}'))
merge_ir = _decode_capture(_json_loads('{"__ir__": "knn_search_mma_split_merge_q128_const148_v1", "arg_keys": ["partial_distances", "partial_indices", "out_distances", "out_indices", "B", "Q", "K", "split_m", "num_q_tiles"], "cluster_dims": [1, 1, 1], "computed_smem_bytes": 0, "constants": [["K_MAX_", 10]], "cta_group": 1, "threads": 32}'))
ir = _decode_capture(_json_loads('{"__ir__": "knn_search_d384_mma_split_partial_0612_r34_v1", "arg_keys": ["queries", "database", "partial_distances", "partial_indices", "B", "Q", "M", "split_m", "num_q_tiles", "total_m_tiles", "tiles_per_split"], "cluster_dims": [1, 1, 1], "computed_smem_bytes": 126720, "constants": [["K_MAX_", 10]], "cta_group": 1, "threads": 640}'))
_D384_MMA_KERNELS: dict[str, Any] = {}
_D384_MMA_SCRATCH: dict[tuple[int, int, int, int, int, int, int, str], tuple[Any, Any]] = {}
D384_Q256_LABELS: tuple[str, ...] = ('blind_post6912_d384_q256_m32768_k10',)
D384_Q256_SHAPES = _decode_capture(_json_loads('[{"__dict_items__": [["label", "blind_post6912_d384_q256_m32768_k10"], ["params", {"__dict_items__": [["B", 1], ["Q", 256], ["M", 32768], ["D", 384], ["K", 10], ["dtype", "bfloat16"], ["seed", 610707], ["self_search", false], ["min_recall", 0.999]]}]]}]'))
_D384_Q256_ENTRY: dict[str, str] = {'shape_key': 'round80a5_d384_q256_m32768_k10_tcgen05_split148', 'guard': 'B == 1 and Q == 256 and M == 32768 and D == 384 and K == 10 and not self_search and not forced_fallback and tcgen05_capable_arch', 'route': ROUTE_D384_Q256_TCGEN05, 'entrypoint': 'loom.examples.weave.knn_search_80a5_blocker_d384_q256_tcgen05_v1:launch_for_eval', 'source_task': 'generalize-auto-tuning-knn-search-80a5', 'source_round_doc': 'design_doc/active/weave_evolve_knn_search_round_116_80a5_d384_q256_reissue_request.md', 'selected_seed': 'd384_q256_tcgen05_split148_80a5'}
SHAPE_DISPATCH_REGISTRY: tuple[dict[str, str], ...] = (_D384_Q256_ENTRY, *parent.SHAPE_DISPATCH_REGISTRY)
_knn_accumulate_q_norm_d384 = _ir_proxy('loom.examples.weave.knn_search_80a5_blocker_d384_q256_tcgen05_v1:_knn_accumulate_q_norm_d384', 256)
_knn_stage_q_half_d384 = _ir_proxy('loom.examples.weave.knn_search_80a5_blocker_d384_q256_tcgen05_v1:_knn_stage_q_half_d384', 256)
_knn_stage_database_half_d384 = _ir_proxy('loom.examples.weave.knn_search_80a5_blocker_d384_q256_tcgen05_v1:_knn_stage_database_half_d384', 256)
_knn_combine_db_norm_d384 = _ir_proxy('loom.examples.weave.knn_search_80a5_blocker_d384_q256_tcgen05_v1:_knn_combine_db_norm_d384', 256)
_knn_capture_d384_distance_tile = _ir_proxy('loom.examples.weave.knn_search_80a5_blocker_d384_q256_tcgen05_v1:_knn_capture_d384_distance_tile', 256)
knn_search_d384_mma_split_partial_0612_r34_v1 = _decode_capture(_json_loads('{"__ir__": "knn_search_d384_mma_split_partial_0612_r34_v1", "arg_keys": ["queries", "database", "partial_distances", "partial_indices", "B", "Q", "M", "split_m", "num_q_tiles", "total_m_tiles", "tiles_per_split"], "cluster_dims": [1, 1, 1], "computed_smem_bytes": 126720, "constants": [["K_MAX_", 10]], "cta_group": 1, "threads": 640}'))
partial_ir = _decode_capture(_json_loads('{"__ir__": "knn_search_d384_mma_split_partial_0612_r34_v1", "arg_keys": ["queries", "database", "partial_distances", "partial_indices", "B", "Q", "M", "split_m", "num_q_tiles", "total_m_tiles", "tiles_per_split"], "cluster_dims": [1, 1, 1], "computed_smem_bytes": 126720, "constants": [["K_MAX_", 10]], "cta_group": 1, "threads": 640}'))
ir = _decode_capture(_json_loads('{"__ir__": "knn_search_d384_mma_split_partial_0612_r34_v1", "arg_keys": ["queries", "database", "partial_distances", "partial_indices", "B", "Q", "M", "split_m", "num_q_tiles", "total_m_tiles", "tiles_per_split"], "cluster_dims": [1, 1, 1], "computed_smem_bytes": 126720, "constants": [["K_MAX_", 10]], "cta_group": 1, "threads": 640}'))

def _compile_d384_mma_kernels() -> dict[str, Any]:
    return _decode_capture(_json_loads('{"merge": {"__kernel__": "dispatch_kernel_0420"}, "partial": {"__kernel__": "dispatch_kernel_0419"}}'))

def _d384_mma_scratch(inputs: dict[str, Any], split_m: int, num_q_tiles: int) -> tuple[Any, Any]:
    import torch
    key = (int(inputs['B']), int(inputs['Q']), int(inputs['M']), int(inputs['D']), int(split_m), int(num_q_tiles), int(inputs['queries'].device.index or 0), str(inputs['queries'].dtype))
    cached = _D384_MMA_SCRATCH.get(key)
    if cached is None:
        shape = (int(inputs['B']), int(num_q_tiles), int(split_m), BLOCK_Q, K_MAX)
        cached = (torch.empty(shape, dtype=torch.float32, device=inputs['queries'].device), torch.empty(shape, dtype=torch.int32, device=inputs['queries'].device))
        _D384_MMA_SCRATCH[key] = cached
    return cached

def _use_d384_q256_tcgen05(inputs: dict[str, Any]) -> bool:
    return int(inputs.get('B', 1)) == 1 and int(inputs['Q']) == TARGET_Q and (int(inputs['M']) == TARGET_M) and (int(inputs['D']) == D_TOTAL) and (int(inputs['K']) == K_MAX) and (not bool(inputs.get('self_search', False))) and (not bool(inputs.get('force_fallback', False))) and mma._tcgen05_capable_arch()

def selected_route(inputs: dict[str, Any]) -> str:
    if _use_d384_q256_tcgen05(inputs):
        return ROUTE_D384_Q256_TCGEN05
    return parent.selected_route(inputs)

def selected_route_name(inputs: dict[str, Any]) -> str:
    return selected_route(inputs)

def _launch_d384_q256_tcgen05(inputs: dict[str, Any]) -> dict[str, Any]:
    import torch
    if not _D384_MMA_KERNELS:
        _D384_MMA_KERNELS.update(_compile_d384_mma_kernels())
    bsz = int(inputs['B'])
    q_rows = int(inputs['Q'])
    m_rows = int(inputs['M'])
    k = int(inputs['K'])
    split_m = D384_SPLIT_M
    num_q_tiles = math.ceil(q_rows / BLOCK_Q)
    total_m_tiles = math.ceil(m_rows / BLOCK_M)
    tiles_per_split = math.ceil(total_m_tiles / split_m)
    partial_dist, partial_idx = _d384_mma_scratch(inputs, split_m, num_q_tiles)
    _D384_MMA_KERNELS['partial'].launch(grid=(bsz * num_q_tiles * split_m, 1, 1), block=(THREADS, 1, 1), args=[inputs['queries'], inputs['database'], partial_dist, partial_idx, bsz, q_rows, m_rows, split_m, num_q_tiles, total_m_tiles, tiles_per_split], shared_mem=MMA_SMEM_BYTES)
    _D384_MMA_KERNELS['merge'].launch(grid=(bsz * q_rows, 1, 1), block=(MERGE_THREADS, 1, 1), args=[partial_dist, partial_idx, inputs['out_distances'], inputs['out_indices'], bsz, q_rows, k, split_m, num_q_tiles], shared_mem=MERGE_SMEM_BYTES)
    return {'distances': inputs['out_distances'], 'indices': inputs['out_indices']}

def launch_for_eval(inputs: dict[str, Any]) -> dict[str, Any]:
    if _use_d384_q256_tcgen05(inputs):
        return _launch_d384_q256_tcgen05(inputs)
    return parent.launch_for_eval(inputs)

def route_info(inputs: dict[str, Any]) -> dict[str, Any]:
    route = selected_route(inputs)
    if route == ROUTE_D384_Q256_TCGEN05:
        parent_info = dict(parent.route_info(inputs))
        parent_route = parent_info.get('route') or parent_info.get('selected_route')
        return {'route': route, 'selected_route': route, 'selected_entrypoint': _D384_Q256_ENTRY['entrypoint'], 'parent_route': parent_route, 'replaced_route': parent_route, 'route_kind': 'specialized', 'route_source': 'shape-specific-seed', 'coverage_class': 'performance_route_d384_q256_tcgen05', 'coverage_only': False, 'production_policy': 'weave_only', 'external_fallback': None, 'guard_order': [entry['shape_key'] for entry in SHAPE_DISPATCH_REGISTRY], 'forced_fallback': False, 'selected_guard': _D384_Q256_ENTRY['guard'], 'fallback': ROUTE_PARENT_DEFAULT_AFE6, 'split_m': D384_SPLIT_M, 'partial_key': 'partial', 'merge_key': 'merge_q128_const148', 'source_task': _D384_Q256_ENTRY['source_task'], 'source_round_doc': _D384_Q256_ENTRY['source_round_doc'], 'selected_seed': _D384_Q256_ENTRY['selected_seed'], 'selected_seed_task': _D384_Q256_ENTRY['source_task']}
    info = dict(parent.route_info(inputs))
    info['route'] = route
    info['selected_route'] = route
    info['guard_order'] = [entry['shape_key'] for entry in SHAPE_DISPATCH_REGISTRY]
    return info

def route_trace_entry(label: str, inputs: dict[str, Any]) -> dict[str, Any]:
    return {'label': label, 'shape_key': label, 'B': int(inputs['B']), 'Q': int(inputs['Q']), 'M': int(inputs['M']), 'D': int(inputs['D']), 'K': int(inputs['K']), 'self_search': bool(inputs.get('self_search', False)), **route_info(inputs)}

def launch_parent_for_eval(inputs: dict[str, Any]) -> dict[str, Any]:
    return parent.launch_for_eval(inputs)

def _select_contract_shapes(shape_labels: str | tuple[str, ...] | list[str] | None):
    if shape_labels is None:
        return D384_Q256_SHAPES
    return select_named_shapes(shape_labels)

def knn_search_compile_and_launch_mma_split(*, benchmark: bool=True, shape_labels: str | tuple[str, ...] | list[str] | None=None) -> dict[str, Any]:
    from .._dispatch_runtime import evaluate
    result = evaluate(launch_for_eval, shapes=_select_contract_shapes(shape_labels), benchmark=benchmark)
    result['passed'] = bool(result.get('summary', {}).get('all_correct'))
    print(result)
    return result

def knn_search_compile_and_launch_d384_q256_tcgen05(*, benchmark: bool=True, shapes: list[dict[str, Any]] | None=None) -> dict[str, Any]:
    from .._dispatch_runtime import evaluate
    result = evaluate(launch_for_eval, shapes=D384_Q256_SHAPES if shapes is None else shapes, benchmark=benchmark)
    result['passed'] = bool(result.get('summary', {}).get('all_correct'))
    print(result)
    return result
