"""Exact D384/Q32/M131072 dynamic-D kNN seed.

Minimum target architecture: sm_100a. This additive bucket seed owns only
``B=1,Q=32,M=131072,D=384,K=10``. It keeps the three-pass tcgen05 producer
from the exact D384 lineage, removes the full database packing step used by
the inherited dynamic-D wrapper, and retains Q32 bounds only where needed to
avoid out-of-bounds query loads. Guard misses delegate to the current ccef v2
dynamic-D seed registry.
"""
from __future__ import annotations
from json import loads as _json_loads
from .._dispatch_runtime import _capture_cuTensorMapEncodeTiled, _decode_capture, _import_dispatch_module, _ir_proxy
from typing import Any
from .._dispatch_runtime import select_named_shapes
from . import knn_search_blind_d384_tcgen05_dispatch0610_r2_f94e_v1 as exact384
from . import knn_search_dynamic_d_remaining_seeds_0618_ccef_v2 as parent
from . import knn_search_mma_split_v1 as mma
THREADS = exact384.THREADS
BLOCK_Q = exact384.BLOCK_Q
BLOCK_M = exact384.BLOCK_M
D_STAGE = exact384.D_STAGE
D_TOTAL = exact384.D_TOTAL
K_MAX = exact384.K_MAX
MMA_POST_MMA_COL_COHORTS = exact384.MMA_POST_MMA_COL_COHORTS
MMA_STAGE_VEC_ELEMS = exact384.MMA_STAGE_VEC_ELEMS
MMA_STAGE_PACK_WORDS = exact384.MMA_STAGE_PACK_WORDS
MMA_Q_STAGE_VECS = exact384.MMA_Q_STAGE_VECS
MMA_Q_NORM_PARTS = exact384.MMA_Q_NORM_PARTS
MMA_Q_NORM_PARTS_MAX = exact384.MMA_Q_NORM_PARTS_MAX
MMA_SMEM_A_BYTES = exact384.MMA_SMEM_A_BYTES
MMA_SMEM_B_BYTES = exact384.MMA_SMEM_B_BYTES
MMA_SMEM_Q_NORM_PART_BYTES = exact384.MMA_SMEM_Q_NORM_PART_BYTES
MMA_SMEM_DB_NORM_PART_BYTES = exact384.MMA_SMEM_DB_NORM_PART_BYTES
MMA_SMEM_DB_NORM_BYTES = exact384.MMA_SMEM_DB_NORM_BYTES
MMA_COHORT_TOPK_BYTES = exact384.MMA_COHORT_TOPK_BYTES
MMA_SMEM_B_OFFSET = exact384.MMA_SMEM_B_OFFSET
MMA_SMEM_Q_NORM_PART_OFFSET = exact384.MMA_SMEM_Q_NORM_PART_OFFSET
MMA_SMEM_DB_NORM_PART_OFFSET = exact384.MMA_SMEM_DB_NORM_PART_OFFSET
MMA_SMEM_DB_NORM_OFFSET = exact384.MMA_SMEM_DB_NORM_OFFSET
MMA_COHORT_TOPK_D_OFFSET = exact384.MMA_COHORT_TOPK_D_OFFSET
MMA_COHORT_TOPK_I_OFFSET = exact384.MMA_COHORT_TOPK_I_OFFSET
MMA_SMEM_POOL_BYTES = exact384.MMA_SMEM_POOL_BYTES
MMA_SMEM_BYTES = exact384.MMA_SMEM_BYTES
MERGE_THREADS = exact384.MERGE_THREADS
MERGE_SMEM_BYTES = exact384.MERGE_SMEM_BYTES
D384_Q32_SPLIT_M = exact384.D384_SPLIT_M
D384_Q32_NUM_Q_TILES = 1
D384_Q32_TOTAL_M_TILES = 1024
ROUTE_D384_Q32_EXACT_TCGEN05 = '5847_dynamic_d384_q32_m131072_exact_tcgen05'
CONSUMED_PARENT_SEED = 'weave-evolve-knn-search-ccef-v2'
CONSUMED_EXACT_D384_SOURCE = 'weave-evolve-knn-search-dispatch0610-r2-f94e-d384-exact'
CONSUMED_SEEDS: tuple[str, ...] = (*parent.CONSUMED_SEEDS, CONSUMED_EXACT_D384_SOURCE)
D384_Q32_LABELS: tuple[str, ...] = ('blind_dyn_d384_q32_m131072_k10',)
D384_Q32_SHAPES = _decode_capture(_json_loads('[{"__dict_items__": [["label", "blind_dyn_d384_q32_m131072_k10"], ["params", {"__dict_items__": [["B", 1], ["Q", 32], ["M", 131072], ["D", 384], ["K", 10], ["dtype", "bfloat16"], ["seed", 610807], ["self_search", false], ["min_recall", 0.999]]}]]}]'))
_D384_Q32_KERNELS: dict[str, Any] | None = None
_D384_Q32_SCRATCH: dict[tuple[int, int, int, int, int, str], tuple[Any, Any]] = {}
_knn_accumulate_q_norm_d384_q32 = _ir_proxy('loom.examples.weave.knn_search_dynamic_d_d384_q32_m131072_exact_tcgen05_0618_5847_v1:_knn_accumulate_q_norm_d384_q32', 256)
_knn_stage_q_pass_d384_q32 = _ir_proxy('loom.examples.weave.knn_search_dynamic_d_d384_q32_m131072_exact_tcgen05_0618_5847_v1:_knn_stage_q_pass_d384_q32', 256)
knn_search_dynamic_d_d384_q32_m131072_exact_tcgen05_partial_0618_5847_v1 = _decode_capture(_json_loads('{"__ir__": "knn_search_dynamic_d_d384_q32_m131072_exact_tcgen05_partial_0618_5847_v1", "arg_keys": ["queries", "database", "partial_distances", "partial_indices", "B", "Q", "M", "split_m", "num_q_tiles", "total_m_tiles", "tiles_per_split"], "cluster_dims": [1, 1, 1], "computed_smem_bytes": 122624, "constants": [["K_MAX_", 10]], "cta_group": 1, "threads": 640}'))
partial_ir = _decode_capture(_json_loads('{"__ir__": "knn_search_dynamic_d_d384_q32_m131072_exact_tcgen05_partial_0618_5847_v1", "arg_keys": ["queries", "database", "partial_distances", "partial_indices", "B", "Q", "M", "split_m", "num_q_tiles", "total_m_tiles", "tiles_per_split"], "cluster_dims": [1, 1, 1], "computed_smem_bytes": 122624, "constants": [["K_MAX_", 10]], "cta_group": 1, "threads": 640}'))
merge_ir = _decode_capture(_json_loads('{"__ir__": "knn_search_mma_split_merge_q128_const148_v1", "arg_keys": ["partial_distances", "partial_indices", "out_distances", "out_indices", "B", "Q", "K", "split_m", "num_q_tiles"], "cluster_dims": [1, 1, 1], "computed_smem_bytes": 0, "constants": [["K_MAX_", 10]], "cta_group": 1, "threads": 32}'))
ir = _decode_capture(_json_loads('{"__ir__": "knn_search_dynamic_d_d384_q32_m131072_exact_tcgen05_partial_0618_5847_v1", "arg_keys": ["queries", "database", "partial_distances", "partial_indices", "B", "Q", "M", "split_m", "num_q_tiles", "total_m_tiles", "tiles_per_split"], "cluster_dims": [1, 1, 1], "computed_smem_bytes": 122624, "constants": [["K_MAX_", 10]], "cta_group": 1, "threads": 640}'))
SHAPE_DISPATCH_REGISTRY: tuple[dict[str, Any], ...] = ({'overlay': '5847_dynamic_d384_q32_m131072_exact_tcgen05', 'shape_key': 'B=1,Q=32,M=131072,D=384,K=10', 'labels': D384_Q32_LABELS, 'guard': 'B == 1 and Q == 32 and M == 131072 and D == 384 and K == 10 and not self_search and not forced_fallback', 'route': ROUTE_D384_Q32_EXACT_TCGEN05, 'entrypoint': 'loom.examples.weave.knn_search_dynamic_d_d384_q32_m131072_exact_tcgen05_0618_5847_v1:launch_for_eval', 'selected_seed': 'weave-evolve-knn-search-5847-d384-q32-exact', 'source_round_doc': 'design_doc/active/weave_evolve_knn_search_round_112_5847_d384_q32_exact.md', 'coverage_class': 'bucket_seed_dynamic_d_d384_q32_m131072_k10', 'workflow_mode': 'generalize_auto_tuning', 'auto_tuning_stage': 'bucket-kernel'},)

def _use_d384_q32_exact_tcgen05(inputs: dict[str, Any]) -> bool:
    return int(inputs['B']) == 1 and int(inputs['Q']) == 32 and (int(inputs['M']) == 131072) and (int(inputs['D']) == D_TOTAL) and (int(inputs['K']) == K_MAX) and (not bool(inputs.get('self_search', False))) and (not bool(inputs.get('force_fallback', False))) and mma._tcgen05_capable_arch()

def selected_route(inputs: dict[str, Any]) -> str:
    if _use_d384_q32_exact_tcgen05(inputs):
        return ROUTE_D384_Q32_EXACT_TCGEN05
    return parent.selected_route(inputs)

def selected_route_name(inputs: dict[str, Any]) -> str:
    return selected_route(inputs)

def route_info(inputs: dict[str, Any]) -> dict[str, Any]:
    if _use_d384_q32_exact_tcgen05(inputs):
        entry = SHAPE_DISPATCH_REGISTRY[0]
        return {'route': ROUTE_D384_Q32_EXACT_TCGEN05, 'selected_route': ROUTE_D384_Q32_EXACT_TCGEN05, 'selected_entrypoint': entry['entrypoint'], 'route_kind': 'specialized', 'route_source': 'bucket-specific-seed', 'coverage_class': entry['coverage_class'], 'classification': 'seed-produced', 'coverage_only': False, 'production_policy': 'weave_only', 'external_fallback': None, 'guard_id': entry['shape_key'], 'selected_guard': entry['guard'], 'fallback': parent.ROUTE_SCALAR_CAPACITY, 'missing_weave_route': False, 'forced_fallback': False, 'selected_seed': entry['selected_seed'], 'source_round_doc': entry['source_round_doc']}
    return parent.route_info(inputs)

def route_trace_entry(label: str, inputs: dict[str, Any]) -> dict[str, Any]:
    return {'label': label, 'shape_key': label, 'B': int(inputs['B']), 'Q': int(inputs['Q']), 'M': int(inputs['M']), 'D': int(inputs['D']), 'K': int(inputs['K']), 'self_search': bool(inputs.get('self_search', False)), **route_info(inputs)}

def _compile_d384_q32_exact_kernels() -> dict[str, Any]:
    return _decode_capture(_json_loads('{"merge": {"__kernel__": "dispatch_kernel_0307"}, "partial": {"__kernel__": "dispatch_kernel_0306"}}'))

def _d384_q32_scratch(inputs: dict[str, Any]) -> tuple[Any, Any]:
    import torch
    key = (int(inputs['B']), int(inputs['Q']), int(inputs['M']), int(inputs['D']), int(inputs['queries'].device.index or 0), str(inputs['queries'].dtype))
    cached = _D384_Q32_SCRATCH.get(key)
    if cached is None:
        shape = (int(inputs['B']), D384_Q32_NUM_Q_TILES, D384_Q32_SPLIT_M, BLOCK_Q, K_MAX)
        cached = (torch.empty(shape, dtype=torch.float32, device=inputs['queries'].device), torch.empty(shape, dtype=torch.int32, device=inputs['queries'].device))
        _D384_Q32_SCRATCH[key] = cached
    return cached

def _launch_d384_q32_exact_tcgen05(inputs: dict[str, Any]) -> dict[str, Any]:
    import torch
    global _D384_Q32_KERNELS
    if _D384_Q32_KERNELS is None:
        _D384_Q32_KERNELS = _compile_d384_q32_exact_kernels()
    bsz = int(inputs['B'])
    q_rows = int(inputs['Q'])
    m_rows = int(inputs['M'])
    k = int(inputs['K'])
    partial_dist, partial_idx = _d384_q32_scratch(inputs)
    _D384_Q32_KERNELS['partial'].launch(grid=(D384_Q32_SPLIT_M, 1, 1), block=(THREADS, 1, 1), args=[inputs['queries'], inputs['database'], partial_dist, partial_idx, bsz, q_rows, m_rows, D384_Q32_SPLIT_M, D384_Q32_NUM_Q_TILES, D384_Q32_TOTAL_M_TILES, 7], shared_mem=MMA_SMEM_BYTES)
    _D384_Q32_KERNELS['merge'].launch(grid=(bsz * q_rows, 1, 1), block=(MERGE_THREADS, 1, 1), args=[partial_dist, partial_idx, inputs['out_distances'], inputs['out_indices'], bsz, q_rows, k, D384_Q32_SPLIT_M, D384_Q32_NUM_Q_TILES], shared_mem=MERGE_SMEM_BYTES)
    return {'distances': inputs['out_distances'], 'indices': inputs['out_indices']}

def launch_for_eval(inputs: dict[str, Any]) -> dict[str, Any]:
    if _use_d384_q32_exact_tcgen05(inputs):
        return _launch_d384_q32_exact_tcgen05(inputs)
    return parent.launch_for_eval(inputs)

def knn_search_compile_and_launch_dynamic_d_d384_q32_exact_tcgen05(*, benchmark: bool=True, shapes: list[dict[str, Any]] | None=None) -> dict[str, Any]:
    from .._dispatch_runtime import evaluate
    selected = D384_Q32_SHAPES if shapes is None else shapes
    result = evaluate(launch_for_eval, shapes=selected, benchmark=benchmark)
    result['passed'] = bool(result.get('summary', {}).get('all_correct'))
    print(result)
    return result
