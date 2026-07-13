"""D512/Q32/K64 distance-only merge probe over the r121 merge-fast seed.

Minimum target architecture: sm_100a. This additive bucket-kernel candidate
targets ``blind_ext_dyn_d512_k64_q32_m32768`` without editing the production
dispatcher. It reuses the verified r121 merge-fast tcgen05 producer and changes
only the final exact-shape merge consumer: comparisons are distance-only, so
the hot merge loop skips lower-index tie branches.
"""
from __future__ import annotations
from json import loads as _json_loads
from .._dispatch_runtime import _capture_cuTensorMapEncodeTiled, _decode_capture, _import_dispatch_module, _ir_proxy
from typing import Any
from .._dispatch_runtime import select_named_shapes
from . import knn_search_dynamic_d512_k64_q32_6bea_r121_83da_mergefast_v1 as parent
THREADS = parent.THREADS
K64_MAX = parent.K64_MAX
Q_ROWS = parent.Q_ROWS
M_ROWS = parent.M_ROWS
D_ORIGINAL = parent.D_ORIGINAL
Q32_SPLIT_M = parent.Q32_SPLIT_M
MMA_POST_MMA_COL_COHORTS = parent.MMA_POST_MMA_COL_COHORTS
Q32_PARTIAL_LISTS = parent.Q32_PARTIAL_LISTS
PARTIAL_STRIDE_Q = parent.PARTIAL_STRIDE_Q
MERGE_THREADS = parent.MERGE_THREADS
MERGE_SPLITS_PER_LANE_MAX = parent.MERGE_SPLITS_PER_LANE_MAX
MMA_SMEM_BYTES = parent.MMA_SMEM_BYTES
MERGE_SMEM_BYTES = parent.MERGE_SMEM_BYTES
ROUTE_D512_Q32_K64_DISTONLYMERGE = '177e_r121_d512_q32_k64_distonlymerge_tcgen05'
CONSUMED_SEED = 'weave-evolve-knn-search-177e-r121-d512-q32-k64-distonlymerge'
REPLACED_SEED = parent.CONSUMED_SEED
TARGET_LABELS: tuple[str, ...] = ('blind_ext_dyn_d512_k64_q32_m32768',)
TARGET_SHAPES = _decode_capture(_json_loads('[{"__dict_items__": [["label", "blind_ext_dyn_d512_k64_q32_m32768"], ["params", {"__dict_items__": [["B", 1], ["Q", 32], ["M", 32768], ["D", 512], ["K", 64], ["dtype", "bfloat16"], ["seed", 610930], ["self_search", false], ["min_recall", 0.999]]}]]}]'))
_KERNELS: dict[str, Any] = {}
knn_search_d512_q32_k64_distonlymerge_merge_177e_r121_v1 = _decode_capture(_json_loads('{"__ir__": "knn_search_d512_q32_k64_distonlymerge_merge_177e_r121_v1", "arg_keys": ["partial_distances", "partial_indices", "out_distances", "out_indices"], "cluster_dims": [1, 1, 1], "computed_smem_bytes": 0, "constants": [["K_MAX_", 64]], "cta_group": 1, "threads": 32}'))
partial_ir = _decode_capture(_json_loads('{"__ir__": "knn_search_d512_q32_k64_mergefast_partial_83da_r121_v1", "arg_keys": ["queries", "database", "partial_distances", "partial_indices"], "cluster_dims": [1, 1, 1], "computed_smem_bytes": 88576, "constants": [["K_MAX_", 64]], "cta_group": 1, "threads": 256}'))
merge_ir = _decode_capture(_json_loads('{"__ir__": "knn_search_d512_q32_k64_distonlymerge_merge_177e_r121_v1", "arg_keys": ["partial_distances", "partial_indices", "out_distances", "out_indices"], "cluster_dims": [1, 1, 1], "computed_smem_bytes": 0, "constants": [["K_MAX_", 64]], "cta_group": 1, "threads": 32}'))
ir = _decode_capture(_json_loads('{"__ir__": "knn_search_d512_q32_k64_distonlymerge_merge_177e_r121_v1", "arg_keys": ["partial_distances", "partial_indices", "out_distances", "out_indices"], "cluster_dims": [1, 1, 1], "computed_smem_bytes": 0, "constants": [["K_MAX_", 64]], "cta_group": 1, "threads": 32}'))
SHAPE_DISPATCH_REGISTRY: tuple[dict[str, Any], ...] = ({'overlay': '177e_r121_d512_q32_k64_distonlymerge_tcgen05', 'shape_key': 'blind_ext_dyn_d512_k64_q32_m32768', 'labels': TARGET_LABELS, 'guard': 'B == 1 and Q == 32 and M == 32768 and D == 512 and K == 64 and not self_search and not forced_fallback and tcgen05_capable_arch', 'route': ROUTE_D512_Q32_K64_DISTONLYMERGE, 'entrypoint': 'loom.examples.weave.knn_search_dynamic_d512_k64_q32_6bea_r121_177e_distonlymerge_v1:launch_for_eval', 'selected_seed': CONSUMED_SEED, 'source_task': CONSUMED_SEED, 'coverage_class': 'bucket_seed_dynamic_d512_q32_k64_distonlymerge', 'workflow_mode': 'generalize_auto_tuning', 'auto_tuning_stage': 'bucket-kernel'},)

def _tcgen05_capable_arch() -> bool:
    return bool(parent._tcgen05_capable_arch())

def _use_d512_q32_k64_distonlymerge(inputs: dict[str, Any]) -> bool:
    return int(inputs['B']) == 1 and int(inputs['Q']) == Q_ROWS and (int(inputs['M']) == M_ROWS) and (int(inputs['D']) == D_ORIGINAL) and (int(inputs['K']) == K64_MAX) and (not bool(inputs.get('self_search', False))) and (not bool(inputs.get('force_fallback', False))) and _tcgen05_capable_arch()

def _compile_kernels() -> dict[str, Any]:
    return _decode_capture(_json_loads('{"merge": {"__kernel__": "dispatch_kernel_0107"}, "partial": {"__kernel__": "dispatch_kernel_0106"}}'))

def _ensure_kernels() -> dict[str, Any]:
    if not _KERNELS:
        _KERNELS.update(_compile_kernels())
    return _KERNELS

def selected_route(inputs: dict[str, Any]) -> str:
    if _use_d512_q32_k64_distonlymerge(inputs):
        return ROUTE_D512_Q32_K64_DISTONLYMERGE
    return parent.selected_route(inputs)

def selected_route_name(inputs: dict[str, Any]) -> str:
    return selected_route(inputs)

def route_info(inputs: dict[str, Any]) -> dict[str, Any]:
    if _use_d512_q32_k64_distonlymerge(inputs):
        parent_info = dict(parent.route_info(inputs))
        return {'route': ROUTE_D512_Q32_K64_DISTONLYMERGE, 'selected_route': ROUTE_D512_Q32_K64_DISTONLYMERGE, 'selected_entrypoint': 'loom.examples.weave.knn_search_dynamic_d512_k64_q32_6bea_r121_177e_distonlymerge_v1:launch_for_eval', 'parent_route': parent_info.get('selected_route', parent.selected_route(inputs)), 'replaced_route': parent_info.get('selected_route', parent.selected_route(inputs)), 'route_kind': 'specialized', 'route_source': 'bucket-specific-seed', 'coverage_class': 'bucket_seed_dynamic_d512_q32_k64_distonlymerge', 'classification': 'seed-produced', 'coverage_only': False, 'production_policy': 'weave_only', 'external_fallback': None, 'guard_id': '177e_r121_d512_q32_k64_distonlymerge', 'forced_fallback': False, 'selected_guard': SHAPE_DISPATCH_REGISTRY[0]['guard'], 'fallback': parent_info.get('selected_route', parent.selected_route(inputs)), 'missing_weave_route': False, 'selected_seed': CONSUMED_SEED, 'replaced_seed': REPLACED_SEED}
    return parent.route_info(inputs)

def route_trace_entry(label: str, inputs: dict[str, Any]) -> dict[str, Any]:
    return {'label': label, 'shape_key': label, 'B': int(inputs['B']), 'Q': int(inputs['Q']), 'M': int(inputs['M']), 'D': int(inputs['D']), 'K': int(inputs['K']), 'self_search': bool(inputs.get('self_search', False)), **route_info(inputs)}

def _launch_d512_q32_k64_distonlymerge(inputs: dict[str, Any]) -> dict[str, Any]:
    kernels = _ensure_kernels()
    bsz = 1
    q_rows = Q_ROWS
    num_q_tiles = 1
    split_m = Q32_SPLIT_M
    partial_list_count = split_m * MMA_POST_MMA_COL_COHORTS
    partial_dist, partial_idx = parent._partial_scratch(inputs, partial_list_count, num_q_tiles)
    kernels['partial'].launch(grid=(bsz * num_q_tiles * split_m, 1, 1), block=(THREADS, 1, 1), args=[inputs['queries'], inputs['database'], partial_dist, partial_idx], shared_mem=MMA_SMEM_BYTES)
    kernels['merge'].launch(grid=(bsz * q_rows, 1, 1), block=(MERGE_THREADS, 1, 1), args=[partial_dist, partial_idx, inputs['out_distances'], inputs['out_indices']], shared_mem=MERGE_SMEM_BYTES)
    return {'distances': inputs['out_distances'], 'indices': inputs['out_indices']}

def launch_for_eval(inputs: dict[str, Any]) -> dict[str, Any]:
    if _use_d512_q32_k64_distonlymerge(inputs):
        return _launch_d512_q32_k64_distonlymerge(inputs)
    return parent.launch_for_eval(inputs)

def knn_search_compile_and_launch_dynamic_d512_k64_q32_6bea_r121_177e_distonlymerge(*, benchmark: bool=True, shapes: list[dict[str, Any]] | None=None) -> dict[str, Any]:
    from .._dispatch_runtime import evaluate
    selected = TARGET_SHAPES if shapes is None else shapes
    result = evaluate(launch_for_eval, shapes=selected, benchmark=benchmark)
    result['passed'] = bool(result.get('summary', {}).get('all_correct'))
    print(result)
    return result
