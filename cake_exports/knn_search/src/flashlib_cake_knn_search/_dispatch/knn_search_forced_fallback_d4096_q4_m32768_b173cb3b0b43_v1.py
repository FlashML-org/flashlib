"""Forced-fallback Weave seed for B1/Q4/M32768/D4096/K10 kNN.

Minimum target architecture: sm_100a.  This is an explicit *fallback* route:
it is selected only when the public request carries ``force_fallback=True``.
It reuses the validated d15e tcgen05 producer, its TMEM-to-register top-10
selection, and its contract-visible merge, rather than falling through to the
scalar-capacity implementation.  The copied input only clears the dispatch
control bit at the seed boundary; all tensor arguments and output buffers are
the public contract objects.
"""
from __future__ import annotations
from json import loads as _json_loads
from .._dispatch_runtime import _capture_cuTensorMapEncodeTiled, _decode_capture, _import_dispatch_module, _ir_proxy
from typing import Any
from . import knn_search_dispatch0630_d15e_af19_portfolio_floor14_7b75_v1 as parent
from . import knn_search_target0630_d4096_q4_m32768_k10_profiled_weave_evolve_v1 as d15e
ENTRYPOINT = 'loom.examples.weave.knn_search_forced_fallback_d4096_q4_m32768_b173cb3b0b43_v1:launch_for_eval'
ROUTE = 'forced_fallback_d4096_q4_m32768_qreuse_tcgen05'
CONSUMED_SEED = 'weave-evolve-knn-search-d15e'
TARGET_SHAPE_KEY = 'coverage_forced_fallback_d4096_q4_m32768_k10'
ir = _decode_capture(_json_loads('{"__ir__": "knn_search_target0630_d4096_q4_m32768_k10_partial_qreuse_v1", "arg_keys": ["queries", "database", "partial_distances", "partial_indices", "B", "Q", "M", "split_m", "num_q_tiles", "total_m_tiles"], "cluster_dims": [1, 1, 1], "computed_smem_bytes": 192768, "constants": [["K_MAX_", 10]], "cta_group": 1, "threads": 512}'))
partial_ir = _decode_capture(_json_loads('{"__ir__": "knn_search_target0630_d4096_q4_m32768_k10_partial_qreuse_v1", "arg_keys": ["queries", "database", "partial_distances", "partial_indices", "B", "Q", "M", "split_m", "num_q_tiles", "total_m_tiles"], "cluster_dims": [1, 1, 1], "computed_smem_bytes": 192768, "constants": [["K_MAX_", 10]], "cta_group": 1, "threads": 512}'))
merge_ir = _decode_capture(_json_loads('{"__ir__": "knn_search_target0627_d4096_q4_m32768_k10_merge128_r221_v1", "arg_keys": ["partial_distances", "partial_indices", "out_distances", "out_indices", "B", "Q", "K", "split_m", "num_q_tiles"], "cluster_dims": [1, 1, 1], "computed_smem_bytes": 0, "constants": [["K_MAX_", 10]], "cta_group": 1, "threads": 32}'))
_ENTRY = {'shape_key': TARGET_SHAPE_KEY, 'route': ROUTE, 'entrypoint': ENTRYPOINT, 'selected_seed': CONSUMED_SEED, 'guard': 'force_fallback and B == 1 and Q == 4 and M == 32768 and D == 4096 and K == 10 and not self_search and arch in {sm_100a,sm_103a}', 'source_round_doc': 'design_doc/active/weave_evolve_knn_search_round_231_profiled_weave_evolve.md'}

def _active(inputs: dict[str, Any]) -> bool:
    return bool(inputs.get('force_fallback', False)) and (not bool(inputs.get('self_search', False))) and ((int(inputs.get('B', 1)), int(inputs['Q']), int(inputs['M']), int(inputs['D']), int(inputs['K'])) == (1, 4, 32768, 4096, 10)) and d15e.seed._tcgen05_capable_arch()

def selected_route(inputs: dict[str, Any]) -> str:
    return ROUTE if _active(inputs) else parent.selected_route(inputs)

def selected_route_name(inputs: dict[str, Any]) -> str:
    return selected_route(inputs)

def route_info(inputs: dict[str, Any]) -> dict[str, Any]:
    if not _active(inputs):
        return dict(parent.route_info(inputs))
    previous = dict(parent.route_info(inputs))
    return {'route': ROUTE, 'selected_route': ROUTE, 'selected_entrypoint': ENTRYPOINT, 'source_entrypoint': d15e.ENTRYPOINT, 'parent_route': previous.get('selected_route'), 'replaced_route': previous.get('selected_route'), 'fallback': previous.get('selected_route'), 'route_kind': 'fallback', 'route_source': 'shape-specific-seed', 'classification': 'seed-consumed', 'coverage_only': False, 'forced_fallback': True, 'production_policy': 'weave_only', 'external_fallback': None, 'missing_weave_route': False, 'guard_id': TARGET_SHAPE_KEY, 'selected_guard': _ENTRY['guard'], 'guard_condition': _ENTRY['guard'], 'selected_seed': CONSUMED_SEED, 'expected_seed': CONSUMED_SEED, 'source_round_doc': _ENTRY['source_round_doc'], 'arch_requirement': 'sm_100a', 'padding_tag': 'none', 'uses_materialized_padding': False, 'uses_kernel_padding': False, 'padding_overhead_timed': False, 'padded_D': 4096, 'workspace_reuse': True}

def route_trace_entry(label: str, inputs: dict[str, Any]) -> dict[str, Any]:
    return {'label': label, 'shape_key': label, **route_info(inputs)}

def launch_for_eval(inputs: dict[str, Any]) -> dict[str, Any]:
    if not _active(inputs):
        return parent.launch_for_eval(inputs)
    seed_inputs = dict(inputs)
    seed_inputs['force_fallback'] = False
    return d15e.launch_for_eval(seed_inputs)
