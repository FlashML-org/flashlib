"""Dispatch-Slurm 176c round-2 extended-K sweep overlay for exact BF16 kNN.

Minimum target architecture: sm_100a for the K>10 tcgen05 capacity routes.
This additive wrapper restores the 176c-style extended-K overlay and expands
the eval denominator with low-K preservation rows plus a synthetic exact K32
row. It does not modify the production dispatcher.
"""
from __future__ import annotations
from json import loads as _json_loads
from .._dispatch_runtime import _capture_cuTensorMapEncodeTiled, _decode_capture, _import_dispatch_module, _ir_proxy
from typing import Any
from .._dispatch_runtime import select_named_shapes
from . import knn_search_extendedk_dispatch0610_r2_k48k64portfolio_v1 as extk_portfolio
from . import knn_search_k48_q128split512_truek48_kexact_0615_2d9eee_v1 as truek48
from . import knn_search_mma_split_v1 as incumbent
THREADS = extk_portfolio.THREADS
MERGE_THREADS = extk_portfolio.MERGE_THREADS
BLOCK_Q = extk_portfolio.BLOCK_Q
BLOCK_M = extk_portfolio.BLOCK_M
D_STATIC = extk_portfolio.D_STATIC
K10_MAX = incumbent.K_MAX
K_MAX = extk_portfolio.K_MAX
ir = _decode_capture(_json_loads('{"__ir__": "knn_search_k48_q4096split128_m32768_k48scratch_partial_0614_ddbc_q4096k48_v2", "arg_keys": ["queries", "database", "partial_distances", "partial_indices", "B", "Q", "M", "split_m", "num_q_tiles", "total_m_tiles", "tiles_per_split"], "cluster_dims": [1, 1, 1], "computed_smem_bytes": 165120, "constants": [["K_MAX_", 64]], "cta_group": 1, "threads": 512}'))
incumbent_ir = _decode_capture(_json_loads('{"__ir__": "knn_search_mma_split_partial_v1", "arg_keys": ["queries", "database", "partial_distances", "partial_indices", "B", "Q", "M", "split_m", "num_q_tiles", "total_m_tiles", "tiles_per_split"], "cluster_dims": [1, 1, 1], "computed_smem_bytes": 108800, "constants": [["K_MAX_", 10], ["EXPOSE_COL_COHORTS", 0], ["FULL_M_TILES", 0]], "cta_group": 1, "threads": 640}'))
extk_portfolio_ir = _decode_capture(_json_loads('{"__ir__": "knn_search_k48_q4096split128_m32768_k48scratch_partial_0614_ddbc_q4096k48_v2", "arg_keys": ["queries", "database", "partial_distances", "partial_indices", "B", "Q", "M", "split_m", "num_q_tiles", "total_m_tiles", "tiles_per_split"], "cluster_dims": [1, 1, 1], "computed_smem_bytes": 165120, "constants": [["K_MAX_", 64]], "cta_group": 1, "threads": 512}'))
truek48_ir = _decode_capture(_json_loads('{"__ir__": "knn_search_mma_split_partial_v1", "arg_keys": ["queries", "database", "partial_distances", "partial_indices", "B", "Q", "M", "split_m", "num_q_tiles", "total_m_tiles", "tiles_per_split"], "cluster_dims": [1, 1, 1], "computed_smem_bytes": 165120, "constants": [["K_MAX_", 48], ["EXPOSE_COL_COHORTS", 1]], "cta_group": 1, "threads": 512}'))
ROUTE_INCUMBENT_K10 = 'round176c_r2_incumbent_k_le_10'
ROUTE_TRUEK48 = truek48.ROUTE_Q128_K48_TRUEK48
K32_SWEEP_SHAPE: dict[str, Any] = {'label': 'ksweep_q128_m131072_d128_k32', 'params': {'B': 1, 'Q': 128, 'M': 131072, 'D': 128, 'K': 32, 'dtype': 'bfloat16', 'seed': 610315, 'self_search': False, 'min_recall': 0.999}}
EXTK_SWEEP_Q128_LOWK_LABELS: tuple[str, ...] = ('ksweep_q128_m131072_d128_k1', 'ksweep_q128_m131072_d128_k2', 'ksweep_q128_m131072_d128_k5', 'ksweep_q128_m131072_d128_k8', 'ksweep_q128_m131072_d128_k10')
EXTK_SWEEP_Q4096_LOWK_LABELS: tuple[str, ...] = ('ksweep_q4096_m20000_d128_k1', 'ksweep_q4096_m20000_d128_k2')
EXTK_SWEEP_CAPACITY_LABELS: tuple[str, ...] = ('ksweep_q128_m131072_d128_k11', 'ksweep_q128_m131072_d128_k12', 'blind_k16_q128_m131072_d128_k16', 'ksweep_q128_m131072_d128_k20', 'ksweep_q128_m131072_d128_k30', 'blind_k48_q128_m131072_d128_k48', 'blind_k48_q4096_m32768_d128_k48', 'ksweep_q128_m131072_d128_k64', 'ksweep_q4096_m20000_d128_k64')
EXTK_SWEEP_BLIND_K64_LABELS: tuple[str, ...] = ('blind_k64_q64_m131072_d128_k64', 'blind_k64_q128_m65536_d128_k64', 'blind_k64_q128_m262144_d128_k64', 'blind_k64_q512_m65536_d128_k64', 'blind_k64_q4096_m32768_d128_k64')
EXTK_SWEEP_EVAL_SHAPES = _decode_capture(_json_loads('[{"__dict_items__": [["label", "ksweep_q128_m131072_d128_k1"], ["params", {"__dict_items__": [["B", 1], ["Q", 128], ["M", 131072], ["D", 128], ["K", 1], ["dtype", "bfloat16"], ["seed", 610301], ["self_search", false], ["min_recall", 1.0]]}]]}, {"__dict_items__": [["label", "ksweep_q128_m131072_d128_k2"], ["params", {"__dict_items__": [["B", 1], ["Q", 128], ["M", 131072], ["D", 128], ["K", 2], ["dtype", "bfloat16"], ["seed", 610302], ["self_search", false], ["min_recall", 1.0]]}]]}, {"__dict_items__": [["label", "ksweep_q128_m131072_d128_k5"], ["params", {"__dict_items__": [["B", 1], ["Q", 128], ["M", 131072], ["D", 128], ["K", 5], ["dtype", "bfloat16"], ["seed", 610303], ["self_search", false], ["min_recall", 0.999]]}]]}, {"__dict_items__": [["label", "ksweep_q128_m131072_d128_k8"], ["params", {"__dict_items__": [["B", 1], ["Q", 128], ["M", 131072], ["D", 128], ["K", 8], ["dtype", "bfloat16"], ["seed", 610304], ["self_search", false], ["min_recall", 0.999]]}]]}, {"__dict_items__": [["label", "ksweep_q128_m131072_d128_k10"], ["params", {"__dict_items__": [["B", 1], ["Q", 128], ["M", 131072], ["D", 128], ["K", 10], ["dtype", "bfloat16"], ["seed", 610305], ["self_search", false], ["min_recall", 0.999]]}]]}, {"__dict_items__": [["label", "ksweep_q4096_m20000_d128_k1"], ["params", {"__dict_items__": [["B", 1], ["Q", 4096], ["M", 20000], ["D", 128], ["K", 1], ["dtype", "bfloat16"], ["seed", 610310], ["self_search", false], ["min_recall", 1.0]]}]]}, {"__dict_items__": [["label", "ksweep_q4096_m20000_d128_k2"], ["params", {"__dict_items__": [["B", 1], ["Q", 4096], ["M", 20000], ["D", 128], ["K", 2], ["dtype", "bfloat16"], ["seed", 610311], ["self_search", false], ["min_recall", 1.0]]}]]}, {"__dict_items__": [["label", "ksweep_q128_m131072_d128_k11"], ["params", {"__dict_items__": [["B", 1], ["Q", 128], ["M", 131072], ["D", 128], ["K", 11], ["dtype", "bfloat16"], ["seed", 610306], ["self_search", false], ["min_recall", 0.999]]}]]}, {"__dict_items__": [["label", "ksweep_q128_m131072_d128_k12"], ["params", {"__dict_items__": [["B", 1], ["Q", 128], ["M", 131072], ["D", 128], ["K", 12], ["dtype", "bfloat16"], ["seed", 610307], ["self_search", false], ["min_recall", 0.999]]}]]}, {"__dict_items__": [["label", "blind_k16_q128_m131072_d128_k16"], ["params", {"__dict_items__": [["B", 1], ["Q", 128], ["M", 131072], ["D", 128], ["K", 16], ["dtype", "bfloat16"], ["seed", 610517], ["self_search", false], ["min_recall", 0.999]]}]]}, {"__dict_items__": [["label", "ksweep_q128_m131072_d128_k20"], ["params", {"__dict_items__": [["B", 1], ["Q", 128], ["M", 131072], ["D", 128], ["K", 20], ["dtype", "bfloat16"], ["seed", 610308], ["self_search", false], ["min_recall", 0.999]]}]]}, {"__dict_items__": [["label", "ksweep_q128_m131072_d128_k30"], ["params", {"__dict_items__": [["B", 1], ["Q", 128], ["M", 131072], ["D", 128], ["K", 30], ["dtype", "bfloat16"], ["seed", 610309], ["self_search", false], ["min_recall", 0.999]]}]]}, {"__dict_items__": [["label", "ksweep_q128_m131072_d128_k32"], ["params", {"__dict_items__": [["B", 1], ["Q", 128], ["M", 131072], ["D", 128], ["K", 32], ["dtype", "bfloat16"], ["seed", 610315], ["self_search", false], ["min_recall", 0.999]]}]]}, {"__dict_items__": [["label", "blind_k48_q128_m131072_d128_k48"], ["params", {"__dict_items__": [["B", 1], ["Q", 128], ["M", 131072], ["D", 128], ["K", 48], ["dtype", "bfloat16"], ["seed", 610518], ["self_search", false], ["min_recall", 0.999]]}]]}, {"__dict_items__": [["label", "blind_k48_q4096_m32768_d128_k48"], ["params", {"__dict_items__": [["B", 1], ["Q", 4096], ["M", 32768], ["D", 128], ["K", 48], ["dtype", "bfloat16"], ["seed", 610519], ["self_search", false], ["min_recall", 0.999]]}]]}, {"__dict_items__": [["label", "ksweep_q128_m131072_d128_k64"], ["params", {"__dict_items__": [["B", 1], ["Q", 128], ["M", 131072], ["D", 128], ["K", 64], ["dtype", "bfloat16"], ["seed", 610312], ["self_search", false], ["min_recall", 0.999]]}]]}, {"__dict_items__": [["label", "ksweep_q4096_m20000_d128_k64"], ["params", {"__dict_items__": [["B", 1], ["Q", 4096], ["M", 20000], ["D", 128], ["K", 64], ["dtype", "bfloat16"], ["seed", 610313], ["self_search", false], ["min_recall", 0.999]]}]]}, {"__dict_items__": [["label", "blind_k64_q64_m131072_d128_k64"], ["params", {"__dict_items__": [["B", 1], ["Q", 64], ["M", 131072], ["D", 128], ["K", 64], ["dtype", "bfloat16"], ["seed", 610503], ["self_search", false], ["min_recall", 0.999]]}]]}, {"__dict_items__": [["label", "blind_k64_q128_m65536_d128_k64"], ["params", {"__dict_items__": [["B", 1], ["Q", 128], ["M", 65536], ["D", 128], ["K", 64], ["dtype", "bfloat16"], ["seed", 610504], ["self_search", false], ["min_recall", 0.999]]}]]}, {"__dict_items__": [["label", "blind_k64_q128_m262144_d128_k64"], ["params", {"__dict_items__": [["B", 1], ["Q", 128], ["M", 262144], ["D", 128], ["K", 64], ["dtype", "bfloat16"], ["seed", 610505], ["self_search", false], ["min_recall", 0.999]]}]]}, {"__dict_items__": [["label", "blind_k64_q512_m65536_d128_k64"], ["params", {"__dict_items__": [["B", 1], ["Q", 512], ["M", 65536], ["D", 128], ["K", 64], ["dtype", "bfloat16"], ["seed", 610506], ["self_search", false], ["min_recall", 0.999]]}]]}, {"__dict_items__": [["label", "blind_k64_q4096_m32768_d128_k64"], ["params", {"__dict_items__": [["B", 1], ["Q", 4096], ["M", 32768], ["D", 128], ["K", 64], ["dtype", "bfloat16"], ["seed", 610507], ["self_search", false], ["min_recall", 0.999]]}]]}]'))
SHAPE_DISPATCH_REGISTRY: tuple[dict[str, str], ...] = ({'shape_key': '176c_r2_k_le_10_incumbent_preservation', 'guard': 'B == 1 and D == 128 and K <= 10; preserves current incumbent route', 'route': ROUTE_INCUMBENT_K10}, {'shape_key': '176c_r2_exact_q128_m131072_d128_k48_truek48', 'guard': 'B == 1 and Q == 128 and M == 131072 and D == 128 and K == 48 and tcgen05', 'route': ROUTE_TRUEK48}, *extk_portfolio.SHAPE_DISPATCH_REGISTRY)

def _use_incumbent_k10(inputs: dict[str, Any]) -> bool:
    return int(inputs['K']) <= K10_MAX

def _use_truek48(inputs: dict[str, Any]) -> bool:
    return bool(truek48._use_q128_k48_truek48(inputs))

def selected_route(inputs: dict[str, Any]) -> str:
    if _use_incumbent_k10(inputs):
        return ROUTE_INCUMBENT_K10
    if _use_truek48(inputs):
        return ROUTE_TRUEK48
    return extk_portfolio.selected_route(inputs)

def selected_route_name(inputs: dict[str, Any]) -> str:
    return selected_route(inputs)

def _selected_entrypoint(route: str) -> str:
    if route == ROUTE_INCUMBENT_K10:
        return 'loom.examples.weave.knn_search_mma_split_v1:launch_for_eval'
    if route == ROUTE_TRUEK48:
        return 'loom.examples.weave.knn_search_k48_q128split512_truek48_kexact_0615_2d9eee_v1:launch_for_eval'
    if hasattr(extk_portfolio, '_selected_entrypoint'):
        return extk_portfolio._selected_entrypoint(route)
    return 'loom.examples.weave.knn_search_extendedk_dispatch0610_r2_k48k64portfolio_v1:launch_for_eval'

def _selected_guard(route: str) -> str:
    for entry in SHAPE_DISPATCH_REGISTRY:
        if entry.get('route') == route:
            return str(entry['guard'])
    return '176c round-2 sweep guard miss; delegate to extended-K r2 portfolio'

def route_info(inputs: dict[str, Any]) -> dict[str, Any]:
    route = selected_route(inputs)
    if route == ROUTE_INCUMBENT_K10:
        return {'route': route, 'selected_route': route, 'selected_entrypoint': _selected_entrypoint(route), 'route_kind': 'preserved_incumbent', 'coverage_class': 'k_le_10_no_regression_guard', 'coverage_only': False, 'production_policy': 'weave_only', 'external_fallback': None, 'selected_guard': _selected_guard(route), 'guard_order': [entry['shape_key'] for entry in SHAPE_DISPATCH_REGISTRY], 'fallback': None}
    if route == ROUTE_TRUEK48:
        info = dict(truek48.route_info(inputs))
    else:
        info = dict(extk_portfolio.route_info(inputs))
    info['route'] = route
    info['selected_route'] = route
    info.setdefault('selected_entrypoint', _selected_entrypoint(route))
    info.setdefault('route_kind', 'specialized')
    info.setdefault('coverage_class', 'performance_route_extended_k')
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
    if route == ROUTE_INCUMBENT_K10:
        return incumbent.launch_for_eval(inputs)
    if route == ROUTE_TRUEK48:
        return truek48.launch_for_eval(inputs)
    return extk_portfolio.launch_for_eval(inputs)

def knn_search_compile_and_launch_extk_sweep_176c_r2(*, benchmark: bool=True, shapes: list[dict[str, Any]] | None=None) -> dict[str, Any]:
    from .._dispatch_runtime import evaluate
    result = evaluate(launch_for_eval, shapes=EXTK_SWEEP_EVAL_SHAPES if shapes is None else shapes, benchmark=benchmark)
    result['passed'] = bool(result.get('summary', {}).get('all_correct'))
    print(result)
    return result
