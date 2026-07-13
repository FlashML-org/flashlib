"""Low-Q policy dispatcher for guarded BF16 squared-L2 kNN.

Minimum target architecture: sm_100a for the inherited tcgen05 row16 routes.
This wrapper composes existing measured Weave seeds only: the round-57 low-Q
tail row16 route repairs below-131072 M-boundary rows, and the 4aa6 Block-M640
over-r55 policy handles the large-M low-Q bucket. It does not retune seed
schedules.
"""
from __future__ import annotations
from json import loads as _json_loads
from .._dispatch_runtime import _capture_cuTensorMapEncodeTiled, _decode_capture, _import_dispatch_module, _ir_proxy
from typing import Any
from . import knn_search_lowq_q2q4_blockm640_r55_dispatch_0614_4aa6_v1 as large_m
from . import knn_search_lowq_tail_row16_0614_r57_ee96_v1 as tail
THREADS = large_m.THREADS
MERGE_THREADS = large_m.MERGE_THREADS
BLOCK_Q = large_m.BLOCK_Q
BLOCK_M = large_m.BLOCK_M
D_STATIC = large_m.D_STATIC
K_MAX = large_m.K_MAX
SPLIT_M = large_m.SPLIT_M
ir = _decode_capture(_json_loads('{"__ir__": "knn_search_lowq_row16_mma_partial_dispatch0610_r18_d14a_vec16_bguard_v1", "arg_keys": ["queries", "database", "partial_distances", "partial_indices", "B", "Q", "M", "split_m", "num_q_tiles", "total_m_tiles"], "cluster_dims": [1, 1, 1], "computed_smem_bytes": 96000, "constants": [["K_MAX_", 10]], "cta_group": 1, "threads": 512}'))
partial_ir = _decode_capture(_json_loads('{"__ir__": "knn_search_lowq_row16_mma_partial_dispatch0610_r18_d14a_vec16_bguard_v1", "arg_keys": ["queries", "database", "partial_distances", "partial_indices", "B", "Q", "M", "split_m", "num_q_tiles", "total_m_tiles"], "cluster_dims": [1, 1, 1], "computed_smem_bytes": 96000, "constants": [["K_MAX_", 10]], "cta_group": 1, "threads": 512}'))
merge_ir = _decode_capture(_json_loads('{"__ir__": "knn_search_mma_split_merge_q128_const148_v1", "arg_keys": ["partial_distances", "partial_indices", "out_distances", "out_indices", "B", "Q", "K", "split_m", "num_q_tiles"], "cluster_dims": [1, 1, 1], "computed_smem_bytes": 0, "constants": [["K_MAX_", 10]], "cta_group": 1, "threads": 32}'))
tail_ir = _decode_capture(_json_loads('{"__ir__": "knn_search_lowq_row16_mma_partial_dispatch0610_r18_d14a_vec16_bguard_v1", "arg_keys": ["queries", "database", "partial_distances", "partial_indices", "B", "Q", "M", "split_m", "num_q_tiles", "total_m_tiles"], "cluster_dims": [1, 1, 1], "computed_smem_bytes": 96000, "constants": [["K_MAX_", 10]], "cta_group": 1, "threads": 512}'))
tail_partial_ir = _decode_capture(_json_loads('{"__ir__": "knn_search_lowq_row16_mma_partial_dispatch0610_r18_d14a_vec16_bguard_v1", "arg_keys": ["queries", "database", "partial_distances", "partial_indices", "B", "Q", "M", "split_m", "num_q_tiles", "total_m_tiles"], "cluster_dims": [1, 1, 1], "computed_smem_bytes": 96000, "constants": [["K_MAX_", 10]], "cta_group": 1, "threads": 512}'))
tail_merge_ir = _decode_capture(_json_loads('{"__ir__": "knn_search_mma_split_merge_q128_const148_v1", "arg_keys": ["partial_distances", "partial_indices", "out_distances", "out_indices", "B", "Q", "K", "split_m", "num_q_tiles"], "cluster_dims": [1, 1, 1], "computed_smem_bytes": 0, "constants": [["K_MAX_", 10]], "cta_group": 1, "threads": 32}'))
parent_ir = _decode_capture(_json_loads('{"__ir__": "knn_search_lowq_row16_mma_partial_dispatch0610_r18_d14a_vec16_bguard_v1", "arg_keys": ["queries", "database", "partial_distances", "partial_indices", "B", "Q", "M", "split_m", "num_q_tiles", "total_m_tiles"], "cluster_dims": [1, 1, 1], "computed_smem_bytes": 96000, "constants": [["K_MAX_", 10]], "cta_group": 1, "threads": 512}'))
ROUTE_LOWQ_TAIL_ROW16 = tail.ROUTE_LOWQ_TAIL_ROW16
ROUTE_LOWQ_Q2Q4_BLOCKM640 = large_m.ROUTE_LOWQ_Q2Q4_BLOCKM640
ROUTE_OUT_OF_CONTRACT = 'out_of_contract_lowq_policy_9dbc'
LOWQ_Q_VALUES: tuple[int, ...] = (2, 4, 8, 16, 32, 64)
LOWQ_LARGE_M_LABELS = large_m.LOWQ_FULL_LARGE_M_LABELS
LOWQ_LARGE_M_SHAPES = large_m.LOWQ_FULL_LARGE_M_SHAPES

def _shape(label: str, *, q: int, m: int, seed: int, d: int=128, k: int=10, b: int=1, force_fallback: bool=False) -> dict[str, Any]:
    params: dict[str, Any] = {'B': b, 'Q': q, 'M': m, 'D': d, 'K': k, 'dtype': 'bfloat16', 'seed': seed, 'self_search': False, 'min_recall': 0.999 if q > 1 else 1.0}
    if force_fallback:
        params['force_fallback'] = True
    return {'label': label, 'params': params}
LOWQ_HELDOUT_M262144_SHAPES = large_m.LOWQ_HELDOUT_M262144_SHAPES
LOWQ_GUARD_MISS_POLICY_SHAPES = _decode_capture(_json_loads('[{"__dict_items__": [["label", "guardmiss_lowq_q2_m131071_d128_k10"], ["params", {"__dict_items__": [["B", 1], ["Q", 2], ["M", 131071], ["D", 128], ["K", 10], ["dtype", "bfloat16"], ["seed", 775002], ["self_search", false], ["min_recall", 0.999]]}]]}, {"__dict_items__": [["label", "guardmiss_lowq_q4_m262145_d128_k10"], ["params", {"__dict_items__": [["B", 1], ["Q", 4], ["M", 262145], ["D", 128], ["K", 10], ["dtype", "bfloat16"], ["seed", 775004], ["self_search", false], ["min_recall", 0.999]]}]]}, {"__dict_items__": [["label", "guardmiss_lowq_q4_m131072_d128_k8"], ["params", {"__dict_items__": [["B", 1], ["Q", 4], ["M", 131072], ["D", 128], ["K", 8], ["dtype", "bfloat16"], ["seed", 775014], ["self_search", false], ["min_recall", 0.999]]}]]}, {"__dict_items__": [["label", "guardmiss_lowq_q4_m131072_d256_k10"], ["params", {"__dict_items__": [["B", 1], ["Q", 4], ["M", 131072], ["D", 256], ["K", 10], ["dtype", "bfloat16"], ["seed", 775024], ["self_search", false], ["min_recall", 0.999]]}]]}, {"__dict_items__": [["label", "guardmiss_lowq_b2_q4_m131072_d128_k10"], ["params", {"__dict_items__": [["B", 2], ["Q", 4], ["M", 131072], ["D", 128], ["K", 10], ["dtype", "bfloat16"], ["seed", 775034], ["self_search", false], ["min_recall", 0.999]]}]]}]'))
LOWQ_FORCED_FALLBACK_SHAPES = large_m.LOWQ_FORCED_FALLBACK_SHAPES
_TAIL_REGISTRY_ENTRY: dict[str, Any] = {'shape_key': 'lowq_tail_row16_ee96_repair', 'guard': 'B == 1 and Q in {2,4,8,16,32,64} and 65536 <= M < 131072 and D == 128 and K == 10 and tcgen05', 'route': ROUTE_LOWQ_TAIL_ROW16, 'entrypoint': 'loom.examples.weave.knn_search_lowq_tail_row16_0614_r57_ee96_v1:launch_for_eval', 'source_kernel': 'loom.examples.weave.knn_search_lowq_tail_row16_0614_r57_ee96_v1', 'source_round_doc': 'design_doc/active/weave_evolve_knn_search_round_92_eb96_tail_guard_miss_row16.md', 'workflow_mode': 'generalize_auto_tuning'}
_OUT_OF_CONTRACT_REGISTRY_ENTRY: dict[str, Any] = {'shape_key': 'lowq_policy_explicit_out_of_contract', 'guard': 'low-Q policy excludes D != 128 and M > 262144; no Weave production route is claimed in this bucket', 'route': ROUTE_OUT_OF_CONTRACT, 'entrypoint': None, 'source_kernel': None, 'source_round_doc': 'design_doc/active/generalize_auto_tuning_knn_search_round_18_196e.md', 'workflow_mode': 'generalize_auto_tuning'}
SHAPE_DISPATCH_REGISTRY: tuple[dict[str, Any], ...] = (_TAIL_REGISTRY_ENTRY, *large_m.SHAPE_DISPATCH_REGISTRY, _OUT_OF_CONTRACT_REGISTRY_ENTRY)

def _forced_fallback(inputs: dict[str, Any]) -> bool:
    return large_m._forced_fallback(inputs)

def _lowq_policy_shape(inputs: dict[str, Any]) -> bool:
    return int(inputs.get('B', 1)) == 1 and int(inputs['Q']) in LOWQ_Q_VALUES and (int(inputs['K']) == K_MAX)

def _use_lowq_tail_row16(inputs: dict[str, Any]) -> bool:
    q_rows = int(inputs['Q'])
    m_rows = int(inputs['M'])
    return not _forced_fallback(inputs) and int(inputs.get('B', 1)) == 1 and (q_rows in LOWQ_Q_VALUES) and (65536 <= m_rows < 131072) and (int(inputs['D']) == D_STATIC) and (int(inputs['K']) == K_MAX) and tail.row16._tcgen05_capable_arch()

def _out_of_lowq_contract(inputs: dict[str, Any]) -> bool:
    if _forced_fallback(inputs):
        return False
    if not _lowq_policy_shape(inputs):
        return False
    d_rows = int(inputs['D'])
    m_rows = int(inputs['M'])
    return d_rows != D_STATIC or m_rows > 262144

def selected_route(inputs: dict[str, Any]) -> str:
    if _out_of_lowq_contract(inputs):
        return ROUTE_OUT_OF_CONTRACT
    if _use_lowq_tail_row16(inputs):
        return ROUTE_LOWQ_TAIL_ROW16
    return large_m.selected_route(inputs)

def selected_route_name(inputs: dict[str, Any]) -> str:
    return selected_route(inputs)

def _classification(inputs: dict[str, Any], route: str) -> str:
    if route == ROUTE_OUT_OF_CONTRACT:
        return 'out_of_contract'
    if route == ROUTE_LOWQ_TAIL_ROW16:
        return 'repaired'
    if _forced_fallback(inputs):
        return 'pass_forced_fallback'
    return 'pass_inherited_or_large_m_policy'

def _selected_guard(inputs: dict[str, Any], route: str, inherited_info: dict[str, Any] | None) -> str:
    if route == ROUTE_OUT_OF_CONTRACT:
        return _OUT_OF_CONTRACT_REGISTRY_ENTRY['guard']
    if route == ROUTE_LOWQ_TAIL_ROW16:
        return _TAIL_REGISTRY_ENTRY['guard']
    if inherited_info is not None:
        return str(inherited_info.get('selected_guard', 'delegate to 4aa6/r55 low-Q policy'))
    return 'delegate to 4aa6/r55 low-Q policy'

def route_info(inputs: dict[str, Any]) -> dict[str, Any]:
    route = selected_route(inputs)
    inherited_info = None if route == ROUTE_OUT_OF_CONTRACT else dict(large_m.route_info(inputs))
    if route == ROUTE_OUT_OF_CONTRACT:
        return {'route': route, 'selected_route': route, 'selected_entrypoint': None, 'parent_route': None, 'replaced_route': None, 'route_kind': 'none', 'coverage_class': 'explicitly_excluded_lowq_policy_shape', 'classification': _classification(inputs, route), 'coverage_only': False, 'production_policy': 'weave_only', 'external_fallback': None, 'guard_order': [entry['shape_key'] for entry in SHAPE_DISPATCH_REGISTRY], 'forced_fallback': False, 'selected_guard': _selected_guard(inputs, route, None), 'fallback': None, 'missing_weave_route': True}
    if route == ROUTE_LOWQ_TAIL_ROW16:
        return {'route': route, 'selected_route': route, 'selected_entrypoint': _TAIL_REGISTRY_ENTRY['entrypoint'], 'parent_route': inherited_info.get('route') if inherited_info is not None else None, 'replaced_route': inherited_info.get('route') if inherited_info is not None else None, 'route_kind': 'specialized', 'coverage_class': 'repaired_guard_miss_lowq_tail_row16', 'classification': _classification(inputs, route), 'coverage_only': False, 'production_policy': 'weave_only', 'external_fallback': None, 'guard_order': [entry['shape_key'] for entry in SHAPE_DISPATCH_REGISTRY], 'forced_fallback': False, 'selected_guard': _selected_guard(inputs, route, inherited_info), 'fallback': None, 'missing_weave_route': False}
    assert inherited_info is not None
    inherited_info['guard_order'] = [entry['shape_key'] for entry in SHAPE_DISPATCH_REGISTRY]
    inherited_info['selected_guard'] = _selected_guard(inputs, route, inherited_info)
    inherited_info['classification'] = _classification(inputs, route)
    inherited_info['missing_weave_route'] = False
    inherited_info.setdefault('external_fallback', None)
    inherited_info.setdefault('production_policy', 'weave_only')
    return inherited_info

def route_trace_entry(label: str, inputs: dict[str, Any]) -> dict[str, Any]:
    return {'label': label, 'B': int(inputs['B']), 'Q': int(inputs['Q']), 'M': int(inputs['M']), 'D': int(inputs['D']), 'K': int(inputs['K']), **route_info(inputs)}

def coverage_route_trace() -> list[dict[str, Any]]:
    entries: list[dict[str, Any]] = []
    for category, shapes in {'representative': LOWQ_LARGE_M_SHAPES, 'heldout': LOWQ_HELDOUT_M262144_SHAPES, 'guard_miss_fallback': LOWQ_GUARD_MISS_POLICY_SHAPES, 'forced_fallback': LOWQ_FORCED_FALLBACK_SHAPES}.items():
        for shape in shapes:
            entry = route_trace_entry(str(shape['label']), dict(shape['params']))
            entry['category'] = category
            entries.append(entry)
    return entries

def launch_for_eval(inputs: dict[str, Any]) -> dict[str, Any]:
    route = selected_route(inputs)
    if route == ROUTE_OUT_OF_CONTRACT:
        info = route_info(inputs)
        raise ValueError(''.join(['low-Q 9dbc policy excludes shape: ', format(info['selected_guard'], '')]))
    if route == ROUTE_LOWQ_TAIL_ROW16:
        return tail.launch_for_eval(inputs)
    return large_m.launch_for_eval(inputs)

def knn_search_compile_and_launch_lowq_policy_dispatch(*, benchmark: bool=True, shapes: list[dict[str, Any]] | None=None) -> dict[str, Any]:
    from .._dispatch_runtime import evaluate
    selected = list(LOWQ_LARGE_M_SHAPES) if shapes is None else shapes
    result = evaluate(launch_for_eval, shapes=selected, benchmark=benchmark)
    result['passed'] = bool(result.get('summary', {}).get('all_correct'))
    print(result)
    return result
