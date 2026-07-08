"""Exact D1024/Q64/M32768/K10 direct-stride tcgen05 kNN seed.

Minimum target architecture: sm_100a.  This additive auto-tuning bucket seed
keeps the proven direct-stride tcgen05 partial-topk producer and its on-device
split-M merge, while exposing the Q64 target-D frontier row to the contract.
"""
from __future__ import annotations
from json import loads as _json_loads
from .._dispatch_runtime import _capture_cuTensorMapEncodeTiled, _decode_capture, _import_dispatch_module, _ir_proxy
from typing import Any
from .._dispatch_runtime import evaluate
from . import knn_search_dynamic_d768d1024_q32q16_tcgen05_0618_9286_v1 as producer
D_ORIG = 1024
Q_ROWS = 64
M_ROWS = 32768
K_MAX = 10
TARGET_LABEL = 'target0627_d1024_q64_m32768_k10'
ROUTE = '974f_target0629_d1024_q64_m32768_k10_directstride_tcgen05'
CONSUMED_SEED = 'weave-evolve-knn-search-974f-d1024-q64-m32768-k10'
ENTRYPOINT = 'loom.examples.weave.knn_search_target0629_d1024_q64_m32768_k10_974f_v1:launch_for_eval'
TARGET_SHAPES: list[dict[str, Any]] = [{'label': TARGET_LABEL, 'params': {'B': 1, 'Q': Q_ROWS, 'M': M_ROWS, 'D': D_ORIG, 'K': K_MAX, 'dtype': 'bfloat16', 'seed': 612111, 'self_search': False, 'min_recall': 0.999}}]
partial_ir = _decode_capture(_json_loads('{"__ir__": "knn_search_dynamic_d768d1024_q32q16_tcgen05_partial_0618_9286_v1", "arg_keys": ["queries", "database", "partial_distances", "partial_indices", "B", "Q", "M", "split_m", "num_q_tiles", "total_m_tiles", "tiles_per_split"], "cluster_dims": [1, 1, 1], "computed_smem_bytes": 143104, "constants": [["K_MAX_", 10], ["D_ORIG_", 1024], ["NUM_D_PASSES_", 8], ["Q_NORM_PARTS_", 64]], "cta_group": 1, "threads": 640}'))
merge_ir = _decode_capture(_json_loads('{"__ir__": "knn_search_mma_split_merge_q128_const148_v1", "arg_keys": ["partial_distances", "partial_indices", "out_distances", "out_indices", "B", "Q", "K", "split_m", "num_q_tiles"], "cluster_dims": [1, 1, 1], "computed_smem_bytes": 0, "constants": [["K_MAX_", 10]], "cta_group": 1, "threads": 32}'))
ir = _decode_capture(_json_loads('{"__ir__": "knn_search_dynamic_d768d1024_q32q16_tcgen05_partial_0618_9286_v1", "arg_keys": ["queries", "database", "partial_distances", "partial_indices", "B", "Q", "M", "split_m", "num_q_tiles", "total_m_tiles", "tiles_per_split"], "cluster_dims": [1, 1, 1], "computed_smem_bytes": 143104, "constants": [["K_MAX_", 10], ["D_ORIG_", 1024], ["NUM_D_PASSES_", 8], ["Q_NORM_PARTS_", 64]], "cta_group": 1, "threads": 640}'))

def _matches(inputs: dict[str, Any]) -> bool:
    return int(inputs['B']) == 1 and int(inputs['Q']) == Q_ROWS and (int(inputs['M']) == M_ROWS) and (int(inputs['D']) == D_ORIG) and (int(inputs['K']) == K_MAX) and (not bool(inputs.get('self_search', False))) and (not bool(inputs.get('force_fallback', False))) and producer.mma._tcgen05_capable_arch()

def selected_route(inputs: dict[str, Any]) -> str:
    return ROUTE if _matches(inputs) else 'unsupported_shape'

def selected_route_name(inputs: dict[str, Any]) -> str:
    return selected_route(inputs)

def route_info(inputs: dict[str, Any]) -> dict[str, Any]:
    if not _matches(inputs):
        return {'route': 'unsupported_shape', 'selected_route': 'unsupported_shape', 'route_kind': 'unsupported'}
    return {'route': ROUTE, 'selected_route': ROUTE, 'selected_entrypoint': 'loom.examples.weave.knn_search_target0629_d1024_q64_m32768_k10_974f_v1:launch_for_eval', 'route_kind': 'specialized', 'route_source': 'shape-specific-seed', 'coverage_class': TARGET_LABEL, 'classification': 'seed-produced', 'coverage_only': False, 'production_policy': 'weave_only', 'external_fallback': None, 'guard_id': TARGET_LABEL, 'selected_guard': 'exact_B1_Q64_M32768_D1024_K10_not_self_not_forced_fallback', 'forced_fallback': False, 'missing_weave_route': False, 'selected_seed': CONSUMED_SEED, 'padding_tag': 'none', 'uses_materialized_padding': False, 'uses_kernel_padding': False, 'padding_overhead_timed': False, 'padded_D': D_ORIG, 'padding_ratio': 1.0, 'workspace_reuse': 'producer_partial_scratch_cache'}

def launch_for_eval(inputs: dict[str, Any]) -> dict[str, Any]:
    if not _matches(inputs):
        raise ValueError('974f seed supports only B=1,Q=64,M=32768,D=1024,K=10, non-self search')
    return producer._launch_high_dynamic_d_tcgen05(inputs)

def compile_and_launch(*, benchmark: bool=True) -> dict[str, Any]:
    return evaluate(launch_for_eval, shapes=TARGET_SHAPES, benchmark=benchmark)
