"""Exact D1024/Q8/M65536/K10 N=256 two-stripe TMEM retune candidate.

Minimum target architecture: sm_100a.  This additive tile-search candidate
selects the BLOCK_M=256 two-stripe tcgen05 producer whose database staging
assigns adjacent lanes to D32 chunks of one row.  The producer's TMEM top-10
feeds the fixed148 merge and the contract-visible distances and indices.
"""
from __future__ import annotations
from json import loads as _json_loads
from .._dispatch_runtime import _capture_cuTensorMapEncodeTiled, _decode_capture, _import_dispatch_module, _ir_proxy
from typing import Any
from . import knn_search_d1024_q8_stagingownership_1c61_v1 as parent
ENTRYPOINT = 'loom.examples.weave.knn_search_d1024q8_blockm256_retune_shared_q8_v1:launch_for_eval'
BLOCK_M = 256
TARGET_LABELS = ('target0627_d1024_q8_m65536_k10', 'coverage_request_d1024_q8_m65536_k10')
ROUTE = 'd1024_q8_n256_two_stripe_tmem_retune_shared_q8_v1'
ir = _decode_capture(_json_loads('{"__ir__": "knn_search_q8_blockm256_two_stripe_tmem_8d4fe4ead6cd_v1", "arg_keys": ["queries", "database", "partial_distances", "partial_indices", "B", "Q", "M", "split_m", "num_q_tiles", "total_m_tiles", "active_split_m"], "cluster_dims": [1, 1, 1], "computed_smem_bytes": 178432, "constants": [["K_MAX_", 10], ["D_ORIG_", 1024], ["NUM_D_PASSES_", 8], ["Q_NORM_PARTS_", 64]], "cta_group": 1, "threads": 640}'))

def _matches(inputs: dict[str, Any]) -> bool:
    return not bool(inputs.get('force_fallback', False)) and parent.selected_route(inputs) == parent.TARGET_ROUTE

def selected_route(inputs: dict[str, Any]) -> str:
    return ROUTE if _matches(inputs) else 'unsupported_shape'

def selected_route_name(inputs: dict[str, Any]) -> str:
    return selected_route(inputs)

def route_info(inputs: dict[str, Any]) -> dict[str, Any]:
    if not _matches(inputs):
        return {'route': 'unsupported_shape', 'selected_route': 'unsupported_shape', 'route_kind': 'unsupported'}
    info = dict(parent.route_info(inputs))
    info.update({'route': ROUTE, 'selected_route': ROUTE, 'selected_entrypoint': ENTRYPOINT, 'route_source': 'retune-shared-q8-blockm256-coalesced-db-staging', 'block_m': BLOCK_M, 'tile_shape': 'BLOCK_M=256', 'target_labels': TARGET_LABELS, 'selected_seed': 'weave-evolve-retune-shared-q8', 'forced_fallback': False})
    return info

def launch_for_eval(inputs: dict[str, Any]) -> dict[str, Any]:
    if not _matches(inputs):
        raise ValueError('retuned seed supports only the non-forced B=1,Q=8,M=65536,D=1024,K=10 route')
    return parent.launch_for_eval(inputs)
