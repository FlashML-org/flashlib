"""D1024/Q32/M32768/K10 exact target-D kNN bucket seed.

Minimum target architecture: sm_100a. This additive generalize-auto-tuning
bucket seed owns ``B=1,Q=32,M=32768,D=1024,K=10`` and keeps production runtime
Weave-only. It reuses the verified D768/D1024 direct-stride tcgen05 producer
and Q128 split-M merge path from the 9286 high-D K10 seed, with a new exact
guard for the expanded target-D guard-overlap row.
"""
from __future__ import annotations
from json import loads as _json_loads
from .._dispatch_runtime import _capture_cuTensorMapEncodeTiled, _decode_capture, _import_dispatch_module, _ir_proxy
from typing import Any
from .._dispatch_runtime import evaluate
from . import knn_search_dispatch0623_c0f1_plus6da2_41b3_v1 as parent
from . import knn_search_dynamic_d768d1024_q32q16_tcgen05_0618_9286_v1 as highd_seed
THREADS = highd_seed.THREADS
MERGE_THREADS = highd_seed.MERGE_THREADS
BLOCK_Q = highd_seed.BLOCK_Q
BLOCK_M = highd_seed.BLOCK_M
D_ORIG = 1024
K_MAX = highd_seed.K_MAX
SPLIT_M = highd_seed.HIGH_DYNAMIC_D_SPLIT_M
SMEM_BYTES = highd_seed.HIGH_MMA_SMEM_BYTES
ROUTE_D1024_Q32_M32768_K10_TCGEN05 = '28cc_d1024_q32_m32768_k10_directstride_tcgen05'
CONSUMED_SEED = 'weave-evolve-knn-search-28cc-d1024-q32-m32768-k10'
TARGET_LABEL = 'exp_targetd_guard_overlap_d1024_q32_m32768_k10'
TARGET_SHAPES: list[dict[str, Any]] = [{'label': TARGET_LABEL, 'params': {'B': 1, 'Q': 32, 'M': 32768, 'D': D_ORIG, 'K': K_MAX, 'dtype': 'bfloat16', 'seed': 620202, 'self_search': False, 'min_recall': 0.999}}]
partial_ir = _decode_capture(_json_loads('{"__ir__": "knn_search_dynamic_d768d1024_q32q16_tcgen05_partial_0618_9286_v1", "arg_keys": ["queries", "database", "partial_distances", "partial_indices", "B", "Q", "M", "split_m", "num_q_tiles", "total_m_tiles", "tiles_per_split"], "cluster_dims": [1, 1, 1], "computed_smem_bytes": 143104, "constants": [["K_MAX_", 10], ["D_ORIG_", 1024], ["NUM_D_PASSES_", 8], ["Q_NORM_PARTS_", 64]], "cta_group": 1, "threads": 640}'))
merge_ir = _decode_capture(_json_loads('{"__ir__": "knn_search_mma_split_merge_q128_const148_v1", "arg_keys": ["partial_distances", "partial_indices", "out_distances", "out_indices", "B", "Q", "K", "split_m", "num_q_tiles"], "cluster_dims": [1, 1, 1], "computed_smem_bytes": 0, "constants": [["K_MAX_", 10]], "cta_group": 1, "threads": 32}'))
ir = _decode_capture(_json_loads('{"__ir__": "knn_search_dynamic_d768d1024_q32q16_tcgen05_partial_0618_9286_v1", "arg_keys": ["queries", "database", "partial_distances", "partial_indices", "B", "Q", "M", "split_m", "num_q_tiles", "total_m_tiles", "tiles_per_split"], "cluster_dims": [1, 1, 1], "computed_smem_bytes": 143104, "constants": [["K_MAX_", 10], ["D_ORIG_", 1024], ["NUM_D_PASSES_", 8], ["Q_NORM_PARTS_", 64]], "cta_group": 1, "threads": 640}'))
SHAPE_DISPATCH_REGISTRY: tuple[dict[str, Any], ...] = ({'overlay': 'd1024_q32_m32768_k10_directstride_28cc', 'shape_key': TARGET_LABEL, 'labels': (TARGET_LABEL,), 'guard': 'B == 1 and Q == 32 and M == 32768 and D == 1024 and K == 10 and not self_search and not forced_fallback and arch in {sm_100a,sm_103a}', 'route': ROUTE_D1024_Q32_M32768_K10_TCGEN05, 'entrypoint': 'loom.examples.weave.knn_search_d1024_q32_m32768_k10_0623_28cc_v1:launch_for_eval', 'selected_seed': CONSUMED_SEED, 'source_task': 'weave-evolve-knn-search-28cc-d1024-q32-m32768-k10', 'coverage_class': 'expanded_targetd_d1024_q32_m32768_k10', 'workflow_mode': 'generalize_auto_tuning', 'auto_tuning_stage': 'bucket-kernel'},)

def _tcgen05_capable_arch() -> bool:
    return bool(highd_seed.mma._tcgen05_capable_arch())

def _use_d1024_q32_m32768_k10_tcgen05(inputs: dict[str, Any]) -> bool:
    return int(inputs['B']) == 1 and int(inputs['Q']) == 32 and (int(inputs['M']) == 32768) and (int(inputs['D']) == D_ORIG) and (int(inputs['K']) == K_MAX) and (not bool(inputs.get('self_search', False))) and (not bool(inputs.get('force_fallback', False))) and _tcgen05_capable_arch()

def selected_route(inputs: dict[str, Any]) -> str:
    if _use_d1024_q32_m32768_k10_tcgen05(inputs):
        return ROUTE_D1024_Q32_M32768_K10_TCGEN05
    return parent.selected_route(inputs)

def selected_route_name(inputs: dict[str, Any]) -> str:
    return selected_route(inputs)

def route_info(inputs: dict[str, Any]) -> dict[str, Any]:
    if _use_d1024_q32_m32768_k10_tcgen05(inputs):
        parent_info = dict(parent.route_info(inputs))
        parent_route = parent_info.get('selected_route', parent.selected_route(inputs))
        return {'route': ROUTE_D1024_Q32_M32768_K10_TCGEN05, 'selected_route': ROUTE_D1024_Q32_M32768_K10_TCGEN05, 'selected_entrypoint': 'loom.examples.weave.knn_search_d1024_q32_m32768_k10_0623_28cc_v1:launch_for_eval', 'parent_route': parent_route, 'replaced_route': parent_route, 'route_kind': 'specialized', 'route_source': 'bucket-specific-seed', 'coverage_class': 'expanded_targetd_d1024_q32_m32768_k10', 'classification': 'seed-produced', 'coverage_only': False, 'production_policy': 'weave_only', 'external_fallback': None, 'guard_order': [TARGET_LABEL, *parent._guard_order(parent.PROFILE_ALL)], 'guard_id': TARGET_LABEL, 'forced_fallback': False, 'selected_guard': SHAPE_DISPATCH_REGISTRY[0]['guard'], 'fallback': parent_route, 'missing_weave_route': False, 'selected_seed': CONSUMED_SEED}
    return parent.route_info(inputs)

def route_trace_entry(label: str, inputs: dict[str, Any]) -> dict[str, Any]:
    return {'label': label, 'shape_key': label, 'B': int(inputs['B']), 'Q': int(inputs['Q']), 'M': int(inputs['M']), 'D': int(inputs['D']), 'K': int(inputs['K']), 'self_search': bool(inputs.get('self_search', False)), **route_info(inputs)}

def launch_for_eval(inputs: dict[str, Any]) -> dict[str, Any]:
    if _use_d1024_q32_m32768_k10_tcgen05(inputs):
        return highd_seed._launch_high_dynamic_d_tcgen05(inputs)
    return parent.launch_for_eval(inputs)

def knn_search_compile_and_launch_d1024_q32_m32768_k10_28cc(*, benchmark: bool=True) -> dict[str, Any]:
    result = evaluate(launch_for_eval, shapes=TARGET_SHAPES, benchmark=benchmark)
    result['passed'] = bool(result.get('summary', {}).get('all_correct'))
    print(result)
    return result
