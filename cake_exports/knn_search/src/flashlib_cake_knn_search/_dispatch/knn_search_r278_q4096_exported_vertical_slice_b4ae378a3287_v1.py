"""R278 exported Q4096 repair: one D07B-owned tcgen05 vertical slice.

Minimum target architecture: sm_100a.  For the exact exported Q4096 contract
row, both default and forced calls keep their original input/output tensors and
their real 72-list scratch inside the D07B tcgen05 producer-to-pairlocal72
consumer path.  This prevents the inherited exported fallback from replacing
the physical scratch ABI before exported distances and indices are written.
"""
from __future__ import annotations
from json import loads as _json_loads
from .._dispatch_runtime import _capture_cuTensorMapEncodeTiled, _decode_capture, _import_dispatch_module, _ir_proxy
from importlib import import_module
from typing import Any
from . import knn_search_rag_q4096_m20000_d128_k10_repair_q4096_seed3_real_scratch_v1 as d07b
ENTRYPOINT = 'loom.examples.weave.knn_search_r278_q4096_exported_vertical_slice_b4ae378a3287_v1:launch_for_eval'
CONSUMED_SEED = 'weave-evolve-knn-search-d07b'
GUARD_ID = 'r278_q4096_d07b_owned_real72_vertical_slice'
GUARD = 'B == 1 and Q == 4096 and M == 20000 and D == 128 and K == 10 and not self_search and arch in {sm_100a,sm_103a}'
ir = _decode_capture(_json_loads('{"__ir__": "knn_search_mma_split_partial_v1", "arg_keys": ["queries", "database", "partial_distances", "partial_indices", "B", "Q", "M", "split_m", "num_q_tiles", "total_m_tiles", "tiles_per_split"], "cluster_dims": [1, 1, 1], "computed_smem_bytes": 108800, "constants": [["K_MAX_", 10], ["EXPOSE_COL_COHORTS", 0], ["FULL_M_TILES", 0]], "cta_group": 1, "threads": 640}'))

def _parent() -> Any:
    return _import_dispatch_module('knn_search_dispatch0630_exported_q8_rag_synthesis_6675_v1')

def _is_target(inputs: dict[str, Any]) -> bool:
    return d07b.selected_route(inputs) in {d07b.TARGET_ROUTE, d07b.FORCED_FALLBACK_ROUTE}

def selected_route(inputs: dict[str, Any]) -> str:
    return d07b.selected_route(inputs) if _is_target(inputs) else _parent().selected_route(inputs)

def selected_route_name(inputs: dict[str, Any]) -> str:
    return selected_route(inputs)

def route_info(inputs: dict[str, Any]) -> dict[str, Any]:
    if not _is_target(inputs):
        info = dict(_parent().route_info(inputs))
        info.update({'dispatcher_entrypoint': ENTRYPOINT, 'guard_order': [GUARD_ID]})
        return info
    info = dict(d07b.route_info(inputs))
    forced = bool(inputs.get('force_fallback', False))
    info.update({'selected_entrypoint': 'loom.examples.weave.knn_search_rag_q4096_m20000_d128_k10_repair_q4096_seed3_real_scratch_v1:launch_for_eval', 'dispatcher_entrypoint': ENTRYPOINT, 'selected_seed': CONSUMED_SEED, 'expected_seed': CONSUMED_SEED, 'guard_id': GUARD_ID, 'selected_guard': GUARD, 'guard_condition': GUARD, 'guard_order': [GUARD_ID], 'route_kind': 'forced-fallback' if forced else 'specialized', 'route_source': 'shape-specific-seed', 'classification': 'seed-consumed', 'coverage_only': False, 'production_policy': 'weave_only', 'external_fallback': None, 'forced_fallback': forced, 'workspace_owner': 'd07b_exact_seed', 'exported_input_owner': 'd07b_exact_seed', 'output_owner': 'd07b_pairlocal72_merge'})
    return info

def route_trace_entry(label: str, inputs: dict[str, Any]) -> dict[str, Any]:
    return {'label': label, 'shape_key': label, **route_info(inputs)}

def launch_for_eval(inputs: dict[str, Any]) -> dict[str, Any]:
    """Preserve the caller's exported tensors on the complete exact D07B slice."""
    return d07b.launch_for_eval(inputs) if _is_target(inputs) else _parent().launch_for_eval(inputs)

def trace_real_scratch(inputs: dict[str, Any]) -> dict[str, Any]:
    """Diagnostic only: trace all Q tiles across the same owned real-72 scratch."""
    if not _is_target(inputs):
        raise ValueError('R278 vertical-slice trace only accepts rag_q4096_m20000_d128_k10')
    return d07b.trace_real_scratch(inputs)
