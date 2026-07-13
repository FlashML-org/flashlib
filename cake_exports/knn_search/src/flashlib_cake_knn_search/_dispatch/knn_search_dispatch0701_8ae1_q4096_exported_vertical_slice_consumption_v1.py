"""Additive exported dispatcher consuming the independent 8ae1 Q4096 slice.

Minimum target architecture: sm_100a.  The exact RAG Q4096/M20000/D128/K10
default and forced paths stay inside the independently validated 8ae1 wrapper;
all other shapes retain the existing exported Weave-only dispatcher.
"""
from __future__ import annotations
from json import loads as _json_loads
from .._dispatch_runtime import _capture_cuTensorMapEncodeTiled, _decode_capture, _import_dispatch_module, _ir_proxy
from importlib import import_module
from typing import Any
from . import knn_search_r278_q4096_exported_vertical_slice_b4ae378a3287_v1 as seed
ENTRYPOINT = 'loom.examples.weave.knn_search_dispatch0701_8ae1_q4096_exported_vertical_slice_consumption_v1:launch_for_eval'
CONSUMED_SEED = 'weave-evolve-knn-search-8ae1'
GUARD_ID = 'r279_q4096_m20000_d128_k10_8ae1_independent_vertical_slice'
GUARD = 'B == 1 and Q == 4096 and M == 20000 and D == 128 and K == 10 and not self_search and arch in {sm_100a,sm_103a}'
ir = _decode_capture(_json_loads('{"__ir__": "knn_search_mma_split_partial_v1", "arg_keys": ["queries", "database", "partial_distances", "partial_indices", "B", "Q", "M", "split_m", "num_q_tiles", "total_m_tiles", "tiles_per_split"], "cluster_dims": [1, 1, 1], "computed_smem_bytes": 108800, "constants": [["K_MAX_", 10], ["EXPOSE_COL_COHORTS", 0], ["FULL_M_TILES", 0]], "cta_group": 1, "threads": 640}'))
_ENTRY = {'shape_key': 'rag_q4096_m20000_d128_k10', 'route': 'r279_8ae1_independent_real72_pairlocal72_exported', 'entrypoint': seed.ENTRYPOINT, 'selected_seed': CONSUMED_SEED, 'guard': GUARD}
SHAPE_DISPATCH_REGISTRY = (_ENTRY,)
_SEED_BLOCK_M = 128

def _parent() -> Any:
    return _import_dispatch_module('knn_search_dispatch0630_exported_q8_rag_synthesis_6675_v1')

def __getattr__(name: str) -> Any:
    return getattr(_parent(), name)

def _is_target(inputs: dict[str, Any]) -> bool:
    return seed.selected_route(inputs) in {seed.d07b.TARGET_ROUTE, seed.d07b.FORCED_FALLBACK_ROUTE}

def selected_route(inputs: dict[str, Any]) -> str:
    return _ENTRY['route'] if _is_target(inputs) else _parent().selected_route(inputs)

def selected_route_name(inputs: dict[str, Any]) -> str:
    return selected_route(inputs)

def route_info(inputs: dict[str, Any]) -> dict[str, Any]:
    if not _is_target(inputs):
        info = dict(_parent().route_info(inputs))
        info.update({'dispatcher_entrypoint': ENTRYPOINT, 'guard_order': [GUARD_ID]})
        return info
    info = dict(seed.route_info(inputs))
    forced = bool(inputs.get('force_fallback', False))
    info.update({'route': _ENTRY['route'], 'selected_route': _ENTRY['route'], 'selected_entrypoint': seed.ENTRYPOINT, 'dispatcher_entrypoint': ENTRYPOINT, 'selected_seed': CONSUMED_SEED, 'expected_seed': CONSUMED_SEED, 'guard_id': ''.join([format(GUARD_ID, ''), '_forced']) if forced else GUARD_ID, 'selected_guard': GUARD, 'guard_condition': GUARD, 'route_kind': 'forced-fallback' if forced else 'specialized', 'route_source': 'shape-specific-seed', 'classification': 'seed-consumed', 'coverage_only': False, 'production_policy': 'weave_only', 'external_fallback': None, 'guard_order': [GUARD_ID], 'consumed_wrapper': seed.ENTRYPOINT})
    return info

def route_trace_entry(label: str, inputs: dict[str, Any]) -> dict[str, Any]:
    return {'label': label, 'shape_key': label, **route_info(inputs)}

def _launch_owned_seed(inputs: dict[str, Any]) -> dict[str, Any]:
    mma = seed.d07b.mma
    previous_block_m = mma.BLOCK_M
    mma.BLOCK_M = _SEED_BLOCK_M
    try:
        return seed.launch_for_eval(inputs)
    finally:
        mma.BLOCK_M = previous_block_m

def launch_for_eval(inputs: dict[str, Any]) -> dict[str, Any]:
    return _launch_owned_seed(inputs) if _is_target(inputs) else _parent().launch_for_eval(inputs)
