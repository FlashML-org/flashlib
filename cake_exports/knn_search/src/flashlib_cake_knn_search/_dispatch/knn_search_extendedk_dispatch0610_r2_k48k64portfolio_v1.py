"""Dispatch-Slurm round-2 extended-K portfolio seed for exact BF16 kNN.

Minimum target architecture: sm_100a for the tcgen05 seed routes. This
additive wrapper targets the extended-K dispatch backlog without changing the
production dispatcher: exact K48 blind-spot rows route to the ddbc K48 seeds,
exact D128/K64 rows route to measured K64 seeds, and all misses delegate to the
source-clean Q128 extended-K dispatcher.
"""
from __future__ import annotations
from json import loads as _json_loads
from .._dispatch_runtime import _capture_cuTensorMapEncodeTiled, _decode_capture, _import_dispatch_module, _ir_proxy
from typing import Any
from .._dispatch_runtime import select_named_shapes
from . import knn_search_blind_k64_m262144_portfolio_0614_r19_6389_v1 as k64_6389
from . import knn_search_blind_k64_q4096_merge16_0614_1968_v1 as k64_1968
from . import knn_search_k48_q128split512_stridedfinal_0614_ddbc_v1 as q128_k48
from . import knn_search_k48_q4096split128_m32768_k48scratch_0614_ddbc_q4096k48_v2 as q4096_k48
from . import knn_search_k64_q128split512_hiermerge32_kexact_0614_r25_k64thin_v1 as q128_k64
from . import knn_search_k64_q4096split79_localprefix6_certfallback_0615_245d_v1 as q4096_k64
from . import knn_search_q128_extendedk_dispatch_0613_r39_48e9_v1 as base
THREADS = base.THREADS
MERGE_THREADS = q128_k64.MERGE_THREADS
BLOCK_Q = base.BLOCK_Q
BLOCK_M = base.BLOCK_M
D_STATIC = base.D_STATIC
K_MAX = q128_k64.K64_MAX
ir = _decode_capture(_json_loads('{"__ir__": "knn_search_k48_q4096split128_m32768_k48scratch_partial_0614_ddbc_q4096k48_v2", "arg_keys": ["queries", "database", "partial_distances", "partial_indices", "B", "Q", "M", "split_m", "num_q_tiles", "total_m_tiles", "tiles_per_split"], "cluster_dims": [1, 1, 1], "computed_smem_bytes": 165120, "constants": [["K_MAX_", 64]], "cta_group": 1, "threads": 512}'))
base_ir = _decode_capture(_json_loads('{"__ir__": "knn_search_mma_split_partial_v1", "arg_keys": ["queries", "database", "partial_distances", "partial_indices", "B", "Q", "M", "split_m", "num_q_tiles", "total_m_tiles", "tiles_per_split"], "cluster_dims": [1, 1, 1], "computed_smem_bytes": 165120, "constants": [["K_MAX_", 20], ["EXPOSE_COL_COHORTS", 0]], "cta_group": 1, "threads": 512}'))
q128_k48_final_ir = _decode_capture(_json_loads('{"__ir__": "knn_search_k48_q128split512_finalmerge32_strided_0614_ddbc_v1", "arg_keys": ["group_distances", "group_indices", "out_distances", "out_indices", "B", "Q", "K"], "cluster_dims": [1, 1, 1], "computed_smem_bytes": 0, "constants": [["K_MAX_", 64]], "cta_group": 1, "threads": 32}'))
q4096_k48_partial_ir = _decode_capture(_json_loads('{"__ir__": "knn_search_k48_q4096split128_m32768_k48scratch_partial_0614_ddbc_q4096k48_v2", "arg_keys": ["queries", "database", "partial_distances", "partial_indices", "B", "Q", "M", "split_m", "num_q_tiles", "total_m_tiles", "tiles_per_split"], "cluster_dims": [1, 1, 1], "computed_smem_bytes": 165120, "constants": [["K_MAX_", 64]], "cta_group": 1, "threads": 512}'))
q4096_k48_merge_ir = _decode_capture(_json_loads('{"__ir__": "knn_search_k48_q4096split128_m32768_k48scratch_merge16_0614_ddbc_q4096k48_v2", "arg_keys": ["partial_distances", "partial_indices", "out_distances", "out_indices", "B", "Q", "K", "partial_list_count", "num_q_tiles"], "cluster_dims": [1, 1, 1], "computed_smem_bytes": 0, "constants": [["K_MAX_", 48]], "cta_group": 1, "threads": 32}'))
q128_k64_partial_ir = _decode_capture(_json_loads('{"__ir__": "knn_search_mma_split_partial_v1", "arg_keys": ["queries", "database", "partial_distances", "partial_indices", "B", "Q", "M", "split_m", "num_q_tiles", "total_m_tiles", "tiles_per_split"], "cluster_dims": [1, 1, 1], "computed_smem_bytes": 165120, "constants": [["K_MAX_", 64], ["EXPOSE_COL_COHORTS", 1]], "cta_group": 1, "threads": 512}'))
q128_k64_group_merge_ir = _decode_capture(_json_loads('{"__ir__": "knn_search_k64_q128split512_groupmerge64_kexact_0614_r25_k64thin_v1", "arg_keys": ["partial_distances", "partial_indices", "group_distances", "group_indices", "B", "Q", "K", "num_q_tiles"], "cluster_dims": [1, 1, 1], "computed_smem_bytes": 0, "constants": [["K_MAX_", 64]], "cta_group": 1, "threads": 32}'))
q128_k64_final_merge_ir = _decode_capture(_json_loads('{"__ir__": "knn_search_k64_q128split512_finalmerge32_kexact_0614_r25_k64thin_v1", "arg_keys": ["group_distances", "group_indices", "out_distances", "out_indices", "B", "Q", "K"], "cluster_dims": [1, 1, 1], "computed_smem_bytes": 0, "constants": [["K_MAX_", 64]], "cta_group": 1, "threads": 32}'))
q4096_k64_partial_ir = _decode_capture(_json_loads('{"__ir__": "knn_search_k64_q4096split79_localprefix7_partial_0615_245d_v1", "arg_keys": ["queries", "database", "partial_distances", "partial_indices", "B", "Q", "M", "split_m", "num_q_tiles", "total_m_tiles", "tiles_per_split"], "cluster_dims": [1, 1, 1], "computed_smem_bytes": 165120, "constants": [["K_MAX_", 64], ["K_PREFIX_", 7]], "cta_group": 1, "threads": 512}'))
q4096_k64_merge_ir = _decode_capture(_json_loads('{"__ir__": "knn_search_k64_q4096split79_localprefix6_certmerge_0615_245d_v1", "arg_keys": ["partial_distances", "partial_indices", "out_distances", "out_indices", "B", "Q", "K", "partial_list_count", "num_q_tiles"], "cluster_dims": [1, 1, 1], "computed_smem_bytes": 0, "constants": [["K_MAX_", 64], ["K_PREFIX_READ_", 6], ["K_STRIDE_", 7]], "cta_group": 1, "threads": 32}'))
q4096_k64_certflag_init_ir = _decode_capture(_json_loads('{"__ir__": "knn_search_k64_q4096split79_localprefix6_certflag_init_0615_245d_v1", "arg_keys": ["overflow_flag"], "cluster_dims": [1, 1, 1], "computed_smem_bytes": 0, "constants": [], "cta_group": 1, "threads": 32}'))
q4096_k64_cert_ir = _decode_capture(_json_loads('{"__ir__": "knn_search_k64_q4096split79_localprefix6_cert_0615_245d_v1", "arg_keys": ["partial_distances", "partial_indices", "out_distances", "out_indices", "overflow_flag", "B", "Q", "K", "partial_list_count", "num_q_tiles"], "cluster_dims": [1, 1, 1], "computed_smem_bytes": 0, "constants": [["K_MAX_", 64], ["K_PREFIX_READ_", 6], ["K_STRIDE_", 7]], "cta_group": 1, "threads": 32}'))
blind_k64_1968_merge_ir = _decode_capture(_json_loads('{"__ir__": "knn_search_blind_k64_q4096_m32768_merge16_0614_1968_v1", "arg_keys": ["partial_distances", "partial_indices", "out_distances", "out_indices"], "cluster_dims": [1, 1, 1], "computed_smem_bytes": 0, "constants": [["K_MAX_", 64], ["PARTIAL_LISTS_", 512], ["SPLITS_PER_LANE_", 16]], "cta_group": 1, "threads": 32}'))
blind_k64_6389_ir = _decode_capture(_json_loads('{"__ir__": "knn_search_blind_k64_twotile_partial_0614_50cc_v1", "arg_keys": ["queries", "database", "partial_distances", "partial_indices", "B", "Q", "M", "split_m", "num_q_tiles", "total_m_tiles", "tiles_per_split", "partial_list_count"], "cluster_dims": [1, 1, 1], "computed_smem_bytes": 165120, "constants": [["K_MAX_", 64]], "cta_group": 1, "threads": 512}'))
blind_k64_6389_group_merge_ir = _decode_capture(_json_loads('{"__ir__": "knn_search_blind_k64_q128m262144_groupmerge64_0614_r19_6389_v1", "arg_keys": ["partial_distances", "partial_indices", "group_distances", "group_indices", "B", "Q", "K", "num_q_tiles"], "cluster_dims": [1, 1, 1], "computed_smem_bytes": 0, "constants": [["K_MAX_", 64]], "cta_group": 1, "threads": 32}'))
ROUTE_BASE_EXTENDEDK = 'round39_q128_extendedk_dispatch'
ROUTE_Q128_K48 = 'round2_ddbc_q128_k48_split512_strided_final'
ROUTE_Q4096_K48 = 'round2_28dd_q4096_k48_split128_k48scratch'
ROUTE_Q128_K64 = 'round2_r25_q128_k64_split512_hiermerge32_kexact'
ROUTE_Q4096_M20000_K64 = 'round2_245d_q4096_m20000_k64_prefix6cert'
ROUTE_Q4096_M32768_K64 = 'round2_1968_q4096_m32768_k64_merge16'
ROUTE_K64_6389 = 'round2_6389_blind_k64_portfolio'
SHAPE_DISPATCH_REGISTRY: tuple[dict[str, str], ...] = ({'shape_key': 'dispatch0610_r2_exact_q128_m131072_k48', 'guard': 'B == 1 and Q == 128 and M == 131072 and D == 128 and K == 48 and tcgen05', 'route': ROUTE_Q128_K48}, {'shape_key': 'dispatch0610_r2_exact_q4096_m32768_k48', 'guard': 'B == 1 and Q == 4096 and M == 32768 and D == 128 and K == 48 and tcgen05', 'route': ROUTE_Q4096_K48}, {'shape_key': 'dispatch0610_r2_exact_q128_m131072_k64', 'guard': 'B == 1 and Q == 128 and M == 131072 and D == 128 and K == 64 and tcgen05', 'route': ROUTE_Q128_K64}, {'shape_key': 'dispatch0610_r2_exact_q4096_m20000_k64', 'guard': 'B == 1 and Q == 4096 and M == 20000 and D == 128 and K == 64 and tcgen05', 'route': ROUTE_Q4096_M20000_K64}, {'shape_key': 'dispatch0610_r2_exact_q4096_m32768_k64', 'guard': 'B == 1 and Q == 4096 and M == 32768 and D == 128 and K == 64 and tcgen05', 'route': ROUTE_Q4096_M32768_K64}, {'shape_key': 'dispatch0610_r2_blind_k64_6389_portfolio', 'guard': 'B == 1 and D == 128 and K == 64 and 6389 selects a specialized blind-K64 route', 'route': ROUTE_K64_6389}, *base.SHAPE_DISPATCH_REGISTRY)
_K64_6389_SPECIALIZED_ROUTES = {k64_6389.ROUTE_Q128_M262144_K64, k64_6389.parent.ROUTE_Q64_K64, 'q128_m65536_d128_k64_split256_twotileproducer_hiermerge16_kexact', k64_6389.parent.ROUTE_Q512_K64}
EXTENDEDK_R2_LABELS: tuple[str, ...] = ('ksweep_q128_m131072_d128_k10', 'ksweep_q128_m131072_d128_k11', 'ksweep_q128_m131072_d128_k12', 'blind_k16_q128_m131072_d128_k16', 'ksweep_q128_m131072_d128_k20', 'ksweep_q128_m131072_d128_k30', 'blind_k48_q128_m131072_d128_k48', 'blind_k48_q4096_m32768_d128_k48', 'ksweep_q128_m131072_d128_k64', 'ksweep_q4096_m20000_d128_k64', 'blind_k64_q64_m131072_d128_k64', 'blind_k64_q128_m65536_d128_k64', 'blind_k64_q128_m262144_d128_k64', 'blind_k64_q512_m65536_d128_k64', 'blind_k64_q4096_m32768_d128_k64')
EXTENDEDK_R2_EVAL_SHAPES = _decode_capture(_json_loads('[{"__dict_items__": [["label", "ksweep_q128_m131072_d128_k10"], ["params", {"__dict_items__": [["B", 1], ["Q", 128], ["M", 131072], ["D", 128], ["K", 10], ["dtype", "bfloat16"], ["seed", 610305], ["self_search", false], ["min_recall", 0.999]]}]]}, {"__dict_items__": [["label", "ksweep_q128_m131072_d128_k11"], ["params", {"__dict_items__": [["B", 1], ["Q", 128], ["M", 131072], ["D", 128], ["K", 11], ["dtype", "bfloat16"], ["seed", 610306], ["self_search", false], ["min_recall", 0.999]]}]]}, {"__dict_items__": [["label", "ksweep_q128_m131072_d128_k12"], ["params", {"__dict_items__": [["B", 1], ["Q", 128], ["M", 131072], ["D", 128], ["K", 12], ["dtype", "bfloat16"], ["seed", 610307], ["self_search", false], ["min_recall", 0.999]]}]]}, {"__dict_items__": [["label", "blind_k16_q128_m131072_d128_k16"], ["params", {"__dict_items__": [["B", 1], ["Q", 128], ["M", 131072], ["D", 128], ["K", 16], ["dtype", "bfloat16"], ["seed", 610517], ["self_search", false], ["min_recall", 0.999]]}]]}, {"__dict_items__": [["label", "ksweep_q128_m131072_d128_k20"], ["params", {"__dict_items__": [["B", 1], ["Q", 128], ["M", 131072], ["D", 128], ["K", 20], ["dtype", "bfloat16"], ["seed", 610308], ["self_search", false], ["min_recall", 0.999]]}]]}, {"__dict_items__": [["label", "ksweep_q128_m131072_d128_k30"], ["params", {"__dict_items__": [["B", 1], ["Q", 128], ["M", 131072], ["D", 128], ["K", 30], ["dtype", "bfloat16"], ["seed", 610309], ["self_search", false], ["min_recall", 0.999]]}]]}, {"__dict_items__": [["label", "blind_k48_q128_m131072_d128_k48"], ["params", {"__dict_items__": [["B", 1], ["Q", 128], ["M", 131072], ["D", 128], ["K", 48], ["dtype", "bfloat16"], ["seed", 610518], ["self_search", false], ["min_recall", 0.999]]}]]}, {"__dict_items__": [["label", "blind_k48_q4096_m32768_d128_k48"], ["params", {"__dict_items__": [["B", 1], ["Q", 4096], ["M", 32768], ["D", 128], ["K", 48], ["dtype", "bfloat16"], ["seed", 610519], ["self_search", false], ["min_recall", 0.999]]}]]}, {"__dict_items__": [["label", "ksweep_q128_m131072_d128_k64"], ["params", {"__dict_items__": [["B", 1], ["Q", 128], ["M", 131072], ["D", 128], ["K", 64], ["dtype", "bfloat16"], ["seed", 610312], ["self_search", false], ["min_recall", 0.999]]}]]}, {"__dict_items__": [["label", "ksweep_q4096_m20000_d128_k64"], ["params", {"__dict_items__": [["B", 1], ["Q", 4096], ["M", 20000], ["D", 128], ["K", 64], ["dtype", "bfloat16"], ["seed", 610313], ["self_search", false], ["min_recall", 0.999]]}]]}, {"__dict_items__": [["label", "blind_k64_q64_m131072_d128_k64"], ["params", {"__dict_items__": [["B", 1], ["Q", 64], ["M", 131072], ["D", 128], ["K", 64], ["dtype", "bfloat16"], ["seed", 610503], ["self_search", false], ["min_recall", 0.999]]}]]}, {"__dict_items__": [["label", "blind_k64_q128_m65536_d128_k64"], ["params", {"__dict_items__": [["B", 1], ["Q", 128], ["M", 65536], ["D", 128], ["K", 64], ["dtype", "bfloat16"], ["seed", 610504], ["self_search", false], ["min_recall", 0.999]]}]]}, {"__dict_items__": [["label", "blind_k64_q128_m262144_d128_k64"], ["params", {"__dict_items__": [["B", 1], ["Q", 128], ["M", 262144], ["D", 128], ["K", 64], ["dtype", "bfloat16"], ["seed", 610505], ["self_search", false], ["min_recall", 0.999]]}]]}, {"__dict_items__": [["label", "blind_k64_q512_m65536_d128_k64"], ["params", {"__dict_items__": [["B", 1], ["Q", 512], ["M", 65536], ["D", 128], ["K", 64], ["dtype", "bfloat16"], ["seed", 610506], ["self_search", false], ["min_recall", 0.999]]}]]}, {"__dict_items__": [["label", "blind_k64_q4096_m32768_d128_k64"], ["params", {"__dict_items__": [["B", 1], ["Q", 4096], ["M", 32768], ["D", 128], ["K", 64], ["dtype", "bfloat16"], ["seed", 610507], ["self_search", false], ["min_recall", 0.999]]}]]}]'))

def _k64_6389_route(inputs: dict[str, Any]) -> str:
    return str(k64_6389.selected_route_name(inputs))

def _use_k64_6389_specialized(inputs: dict[str, Any]) -> bool:
    return _k64_6389_route(inputs) in _K64_6389_SPECIALIZED_ROUTES

def selected_route(inputs: dict[str, Any]) -> str:
    if q128_k48._use_q128_k48_split512_stridedfinal(inputs):
        return ROUTE_Q128_K48
    if q4096_k48._use_q4096_k48_split128_k48scratch(inputs):
        return ROUTE_Q4096_K48
    if q128_k64._use_q128_k64_split512_indexfast(inputs):
        return ROUTE_Q128_K64
    if q4096_k64._use_q4096_k64_prefix6cert(inputs):
        return ROUTE_Q4096_M20000_K64
    if k64_1968._use_q4096_m32768_k64(inputs):
        return ROUTE_Q4096_M32768_K64
    if _use_k64_6389_specialized(inputs):
        return ROUTE_K64_6389
    return base.selected_route(inputs)

def selected_route_name(inputs: dict[str, Any]) -> str:
    return selected_route(inputs)

def _selected_entrypoint(route: str) -> str:
    if route == ROUTE_Q128_K48:
        return 'loom.examples.weave.knn_search_k48_q128split512_stridedfinal_0614_ddbc_v1:launch_for_eval'
    if route == ROUTE_Q4096_K48:
        return 'loom.examples.weave.knn_search_k48_q4096split128_m32768_k48scratch_0614_ddbc_q4096k48_v2:launch_for_eval'
    if route == ROUTE_Q128_K64:
        return 'loom.examples.weave.knn_search_k64_q128split512_hiermerge32_kexact_0614_r25_k64thin_v1:launch_for_eval'
    if route == ROUTE_Q4096_M20000_K64:
        return 'loom.examples.weave.knn_search_k64_q4096split79_localprefix6_certfallback_0615_245d_v1:launch_for_eval'
    if route == ROUTE_Q4096_M32768_K64:
        return 'loom.examples.weave.knn_search_blind_k64_q4096_merge16_0614_1968_v1:launch_for_eval'
    if route == ROUTE_K64_6389:
        return 'loom.examples.weave.knn_search_blind_k64_m262144_portfolio_0614_r19_6389_v1:launch_for_eval'
    return 'loom.examples.weave.knn_search_q128_extendedk_dispatch_0613_r39_48e9_v1:launch_for_eval'

def _selected_guard(route: str) -> str:
    for entry in SHAPE_DISPATCH_REGISTRY:
        if entry.get('route') == route:
            return str(entry['guard'])
    return 'dispatch0610_r2 guard miss; delegate to round39 Q128 extended-K dispatcher'

def route_info(inputs: dict[str, Any]) -> dict[str, Any]:
    route = selected_route(inputs)
    if route == ROUTE_K64_6389:
        parent_info = dict(k64_6389.route_info(inputs))
        return {**parent_info, 'route': route, 'selected_route': route, 'selected_entrypoint': _selected_entrypoint(route), 'replaced_route': parent_info.get('route') or parent_info.get('selected_route'), 'route_kind': 'specialized', 'coverage_class': 'performance_route_6389_blind_k64_portfolio', 'coverage_only': False, 'production_policy': 'weave_only', 'external_fallback': None, 'selected_guard': _selected_guard(route), 'guard_order': [entry['shape_key'] for entry in SHAPE_DISPATCH_REGISTRY]}
    if route in {ROUTE_Q128_K48, ROUTE_Q4096_K48, ROUTE_Q128_K64, ROUTE_Q4096_M20000_K64, ROUTE_Q4096_M32768_K64}:
        info: dict[str, Any] = {'route': route, 'selected_route': route, 'selected_entrypoint': _selected_entrypoint(route), 'route_kind': 'specialized', 'coverage_class': 'performance_route_extended_k', 'coverage_only': False, 'production_policy': 'weave_only', 'external_fallback': None, 'selected_guard': _selected_guard(route), 'guard_order': [entry['shape_key'] for entry in SHAPE_DISPATCH_REGISTRY], 'fallback': None}
        if route == ROUTE_Q4096_M20000_K64:
            info['certification_policy'] = 'prefix6_plus_sentinel_with_weave_full_k64_fallback_on_overflow'
            info['fallback_entrypoint'] = 'loom.examples.weave.knn_search_k64_q4096split79_twotile_oddevensort_fastmerge_0612_r34_11c1_v1:launch_for_eval'
        return info
    info = {'route': route}
    if hasattr(base, 'route_info'):
        info.update(base.route_info(inputs))
    info['route'] = route
    info['selected_route'] = route
    info.setdefault('selected_entrypoint', _selected_entrypoint(route))
    info.setdefault('coverage_class', 'inherited_round39_q128_extendedk')
    info.setdefault('coverage_only', False)
    info.setdefault('production_policy', 'weave_only')
    info.setdefault('external_fallback', None)
    info['selected_guard'] = _selected_guard(route)
    info['guard_order'] = [entry['shape_key'] for entry in SHAPE_DISPATCH_REGISTRY]
    return info

def route_trace_entry(label: str, inputs: dict[str, Any]) -> dict[str, Any]:
    return {'label': label, 'B': int(inputs['B']), 'Q': int(inputs['Q']), 'M': int(inputs['M']), 'D': int(inputs['D']), 'K': int(inputs['K']), **route_info(inputs)}

def launch_for_eval(inputs: dict[str, Any]) -> dict[str, Any]:
    route = selected_route(inputs)
    if route == ROUTE_Q128_K48:
        return q128_k48.launch_for_eval(inputs)
    if route == ROUTE_Q4096_K48:
        return q4096_k48.launch_for_eval(inputs)
    if route == ROUTE_Q128_K64:
        return q128_k64.launch_for_eval(inputs)
    if route == ROUTE_Q4096_M20000_K64:
        return q4096_k64.launch_for_eval(inputs)
    if route == ROUTE_Q4096_M32768_K64:
        return k64_1968.launch_for_eval(inputs)
    if route == ROUTE_K64_6389:
        return k64_6389.launch_for_eval(inputs)
    return base.launch_for_eval(inputs)

def knn_search_compile_and_launch_extendedk_r2_portfolio(*, benchmark: bool=True, shapes: list[dict[str, Any]] | None=None) -> dict[str, Any]:
    from .._dispatch_runtime import evaluate
    result = evaluate(launch_for_eval, shapes=EXTENDEDK_R2_EVAL_SHAPES if shapes is None else shapes, benchmark=benchmark)
    result['passed'] = bool(result.get('summary', {}).get('all_correct'))
    print(result)
    return result
