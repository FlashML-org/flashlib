"""Exact Q8/D1024/M65536/K10 Weave seed with an explicit forced-route guard.

Minimum target architecture: sm_100a.  This additive seed retains the validated
tcgen05 scan/top-10 implementation and makes the dispatch ABI explicit: a
``force_fallback`` request never selects this ordinary specialized route.
"""
from __future__ import annotations
from json import loads as _json_loads
from .._dispatch_runtime import _capture_cuTensorMapEncodeTiled, _decode_capture, _import_dispatch_module, _ir_proxy
from typing import Any
from . import knn_search_q8_blockm256_8d4fe4ead6cd_v2 as _parent
ENTRYPOINT = 'loom.examples.weave.knn_search_q8_d1024_repair_q8_d1024_seed_v1:launch_for_eval'
TARGET_LABEL = _parent.TARGET_LABEL
TARGET_ROUTE = 'q8_d1024_exact_tcgen05_seed_repair_v1'
partial_ir = _decode_capture(_json_loads('{"__ir__": "knn_search_q8_blockm256_two_stripe_tmem_8d4fe4ead6cd_v1", "arg_keys": ["queries", "database", "partial_distances", "partial_indices", "B", "Q", "M", "split_m", "num_q_tiles", "total_m_tiles", "active_split_m"], "cluster_dims": [1, 1, 1], "computed_smem_bytes": 178432, "constants": [["K_MAX_", 10], ["D_ORIG_", 1024], ["NUM_D_PASSES_", 8], ["Q_NORM_PARTS_", 64]], "cta_group": 1, "threads": 640}'))
merge_ir = _decode_capture(_json_loads('{"__ir__": "knn_search_mma_split_merge_q128_const148_v1", "arg_keys": ["partial_distances", "partial_indices", "out_distances", "out_indices", "B", "Q", "K", "split_m", "num_q_tiles"], "cluster_dims": [1, 1, 1], "computed_smem_bytes": 0, "constants": [["K_MAX_", 10]], "cta_group": 1, "threads": 32}'))
ir = _decode_capture(_json_loads('{"__ir__": "knn_search_q8_blockm256_two_stripe_tmem_8d4fe4ead6cd_v1", "arg_keys": ["queries", "database", "partial_distances", "partial_indices", "B", "Q", "M", "split_m", "num_q_tiles", "total_m_tiles", "active_split_m"], "cluster_dims": [1, 1, 1], "computed_smem_bytes": 178432, "constants": [["K_MAX_", 10], ["D_ORIG_", 1024], ["NUM_D_PASSES_", 8], ["Q_NORM_PARTS_", 64]], "cta_group": 1, "threads": 640}'))

def _matches(inputs: dict[str, Any]) -> bool:
    return int(inputs['B']) == 1 and int(inputs['Q']) == _parent.TARGET_Q and (int(inputs['M']) == _parent.TARGET_M) and (int(inputs['D']) == _parent.HIGH_D_MAX) and (int(inputs['K']) == _parent.K_MAX) and (not bool(inputs.get('self_search', False))) and (not bool(inputs.get('force_fallback', False))) and _parent.mma._tcgen05_capable_arch()

def selected_route(inputs: dict[str, Any]) -> str:
    return TARGET_ROUTE if _matches(inputs) else 'unsupported_shape'

def selected_route_name(inputs: dict[str, Any]) -> str:
    return selected_route(inputs)

def route_info(inputs: dict[str, Any]) -> dict[str, Any]:
    if not _matches(inputs):
        return {'route': 'unsupported_shape', 'selected_route': 'unsupported_shape', 'selected_entrypoint': None, 'route_kind': 'unsupported', 'route_source': None, 'coverage_only': False, 'production_policy': 'weave_only', 'external_fallback': None, 'forced_fallback': bool(inputs.get('force_fallback', False))}
    info = dict(_parent.route_info(inputs))
    info.update({'route': TARGET_ROUTE, 'selected_route': TARGET_ROUTE, 'selected_entrypoint': ENTRYPOINT, 'route_source': 'q8-d1024-exact-seed-repair', 'classification': 'seed-produced', 'selected_seed': 'repair-q8-d1024-seed', 'selected_guard': 'B == 1 and Q == 8 and M == 65536 and D == 1024 and K == 10 and not self_search and not force_fallback and arch in {sm_100a,sm_103a}'})
    return info

def launch_for_eval(inputs: dict[str, Any]) -> dict[str, Any]:
    if not _matches(inputs):
        raise ValueError('q8/d1024 repair seed supports only non-forced B=1,Q=8,M=65536,D=1024,K=10, non-self search')
    return _parent.launch_for_eval(inputs)
