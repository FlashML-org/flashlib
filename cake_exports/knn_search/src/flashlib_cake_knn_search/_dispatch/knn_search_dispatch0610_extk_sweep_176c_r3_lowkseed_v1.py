"""Dispatch-0610 extended-K sweep overlay with Q4096 low-K seed overrides.

Minimum target architecture: sm_100a for the tcgen05-backed routes. This
additive candidate preserves the round-2 extended-K sweep structure and replaces
only the exact ``B=1,Q=4096,M=20000,D=128,K in {1,2}`` rows with dedicated
Weave-only low-K seeds.
"""
from __future__ import annotations
from json import loads as _json_loads
from .._dispatch_runtime import _capture_cuTensorMapEncodeTiled, _decode_capture, _import_dispatch_module, _ir_proxy
from typing import Any
from .._dispatch_runtime import select_named_shapes
from . import knn_search_dispatch0610_extk_overlay_176c_v1 as parent176c
from . import knn_search_extendedk_dispatch0610_r2_k48k64portfolio_v1 as extendedk
from . import knn_search_k1_post_merge8_0614_qfull_merge8_cd66_v1 as lowk_k1
from . import knn_search_k48_q128split512_truek48_kexact_0615_2d9eee_v1 as truek48
from . import knn_search_mma_split_v1 as incumbent
from . import knn_search_q128_extendedk_dispatch_0613_r39_48e9_v1 as q128_ext
from . import knn_search_q4096_lowk_k2partial_split9_0613_r46_48e9_v1 as lowk_k2
THREADS = incumbent.THREADS
MERGE_THREADS = truek48.MERGE_THREADS
BLOCK_Q = incumbent.BLOCK_Q
BLOCK_M = incumbent.BLOCK_M
D_STATIC = incumbent.D_STATIC
K10_MAX = incumbent.K_MAX
K_MAX = extendedk.K_MAX
ir = _decode_capture(_json_loads('{"__ir__": "knn_search_k1_top1_margin_qfull_partial_0614_r93_v1", "arg_keys": ["queries", "database", "partial_distances", "partial_indices", "B", "Q", "M", "split_m", "num_q_tiles", "total_m_tiles", "tiles_per_split"], "cluster_dims": [1, 1, 1], "computed_smem_bytes": 108800, "constants": [], "cta_group": 1, "threads": 640}'))
incumbent_ir = _decode_capture(_json_loads('{"__ir__": "knn_search_mma_split_partial_v1", "arg_keys": ["queries", "database", "partial_distances", "partial_indices", "B", "Q", "M", "split_m", "num_q_tiles", "total_m_tiles", "tiles_per_split"], "cluster_dims": [1, 1, 1], "computed_smem_bytes": 108800, "constants": [["K_MAX_", 10], ["EXPOSE_COL_COHORTS", 0], ["FULL_M_TILES", 0]], "cta_group": 1, "threads": 640}'))
parent176c_ir = _decode_capture(_json_loads('{"__ir__": "knn_search_mma_split_partial_v1", "arg_keys": ["queries", "database", "partial_distances", "partial_indices", "B", "Q", "M", "split_m", "num_q_tiles", "total_m_tiles", "tiles_per_split"], "cluster_dims": [1, 1, 1], "computed_smem_bytes": 165120, "constants": [["K_MAX_", 48], ["EXPOSE_COL_COHORTS", 1]], "cta_group": 1, "threads": 512}'))
extendedk_ir = _decode_capture(_json_loads('{"__ir__": "knn_search_k48_q4096split128_m32768_k48scratch_partial_0614_ddbc_q4096k48_v2", "arg_keys": ["queries", "database", "partial_distances", "partial_indices", "B", "Q", "M", "split_m", "num_q_tiles", "total_m_tiles", "tiles_per_split"], "cluster_dims": [1, 1, 1], "computed_smem_bytes": 165120, "constants": [["K_MAX_", 64]], "cta_group": 1, "threads": 512}'))
truek48_ir = _decode_capture(_json_loads('{"__ir__": "knn_search_mma_split_partial_v1", "arg_keys": ["queries", "database", "partial_distances", "partial_indices", "B", "Q", "M", "split_m", "num_q_tiles", "total_m_tiles", "tiles_per_split"], "cluster_dims": [1, 1, 1], "computed_smem_bytes": 165120, "constants": [["K_MAX_", 48], ["EXPOSE_COL_COHORTS", 1]], "cta_group": 1, "threads": 512}'))
lowk_k1_partial_ir = _decode_capture(_json_loads('{"__ir__": "knn_search_k1_top1_margin_qfull_partial_0614_r93_v1", "arg_keys": ["queries", "database", "partial_distances", "partial_indices", "B", "Q", "M", "split_m", "num_q_tiles", "total_m_tiles", "tiles_per_split"], "cluster_dims": [1, 1, 1], "computed_smem_bytes": 108800, "constants": [], "cta_group": 1, "threads": 640}'))
lowk_k1_merge_ir = _decode_capture(_json_loads('{"__ir__": "knn_search_k1_top1_margin_0614_r93_merge8_v1", "arg_keys": ["partial_distances", "partial_indices", "out_distances", "out_indices", "B", "Q", "K", "split_m", "num_q_tiles"], "cluster_dims": [1, 1, 1], "computed_smem_bytes": 0, "constants": [], "cta_group": 1, "threads": 128}'))
lowk_k2_partial_ir = _decode_capture(_json_loads('{"__ir__": "knn_search_q4096_lowk_k2partial_0613_r45_48e9_v1", "arg_keys": ["queries", "database", "partial_distances", "partial_indices", "B", "Q", "M", "split_m", "num_q_tiles", "total_m_tiles", "tiles_per_split"], "cluster_dims": [1, 1, 1], "computed_smem_bytes": 108800, "constants": [], "cta_group": 1, "threads": 640}'))
lowk_k2_merge_ir = _decode_capture(_json_loads('{"__ir__": "knn_search_q4096_lowk_k2partial_split9_merge_0613_r46_48e9_v1", "arg_keys": ["partial_distances", "partial_indices", "out_distances", "out_indices", "B", "Q", "K", "split_m", "num_q_tiles"], "cluster_dims": [1, 1, 1], "computed_smem_bytes": 0, "constants": [["K_STRIDE_", 2], ["K_OUT_MAX_", 2]], "cta_group": 1, "threads": 32}'))
ROUTE_Q4096_K1_LOWK = 'round176c_r3_cd66_q4096_m20000_k1_qfull_merge8'
ROUTE_Q4096_K2_LOWK = 'round176c_r3_round46_q4096_m20000_k2_split9'
ROUTE_INCUMBENT_K10 = 'round176c_r3_incumbent_mma_split_k_le_10'
ROUTE_TRUEK48 = truek48.ROUTE_Q128_K48_TRUEK48
ROUTE_EXTENDEDK_PORTFOLIO = 'round176c_r3_extendedk_r2_portfolio'
EXTK_SWEEP_HEAD_LABELS: tuple[str, ...] = ('ksweep_q128_m131072_d128_k1', 'ksweep_q128_m131072_d128_k2', 'ksweep_q128_m131072_d128_k5', 'ksweep_q128_m131072_d128_k8', 'ksweep_q128_m131072_d128_k10', 'ksweep_q4096_m20000_d128_k1', 'ksweep_q4096_m20000_d128_k2', 'ksweep_q128_m131072_d128_k11', 'ksweep_q128_m131072_d128_k12', 'blind_k16_q128_m131072_d128_k16', 'ksweep_q128_m131072_d128_k20', 'ksweep_q128_m131072_d128_k30')
EXTK_SWEEP_TAIL_LABELS: tuple[str, ...] = ('blind_k48_q128_m131072_d128_k48', 'blind_k48_q4096_m32768_d128_k48', 'ksweep_q128_m131072_d128_k64', 'ksweep_q4096_m20000_d128_k64', 'blind_k64_q64_m131072_d128_k64', 'blind_k64_q128_m65536_d128_k64', 'blind_k64_q128_m262144_d128_k64', 'blind_k64_q512_m65536_d128_k64', 'blind_k64_q4096_m32768_d128_k64')
EXTK_SWEEP_EVAL_SHAPES = _decode_capture(_json_loads('[{"__dict_items__": [["label", "ksweep_q128_m131072_d128_k1"], ["params", {"__dict_items__": [["B", 1], ["Q", 128], ["M", 131072], ["D", 128], ["K", 1], ["dtype", "bfloat16"], ["seed", 610301], ["self_search", false], ["min_recall", 1.0]]}]]}, {"__dict_items__": [["label", "ksweep_q128_m131072_d128_k2"], ["params", {"__dict_items__": [["B", 1], ["Q", 128], ["M", 131072], ["D", 128], ["K", 2], ["dtype", "bfloat16"], ["seed", 610302], ["self_search", false], ["min_recall", 1.0]]}]]}, {"__dict_items__": [["label", "ksweep_q128_m131072_d128_k5"], ["params", {"__dict_items__": [["B", 1], ["Q", 128], ["M", 131072], ["D", 128], ["K", 5], ["dtype", "bfloat16"], ["seed", 610303], ["self_search", false], ["min_recall", 0.999]]}]]}, {"__dict_items__": [["label", "ksweep_q128_m131072_d128_k8"], ["params", {"__dict_items__": [["B", 1], ["Q", 128], ["M", 131072], ["D", 128], ["K", 8], ["dtype", "bfloat16"], ["seed", 610304], ["self_search", false], ["min_recall", 0.999]]}]]}, {"__dict_items__": [["label", "ksweep_q128_m131072_d128_k10"], ["params", {"__dict_items__": [["B", 1], ["Q", 128], ["M", 131072], ["D", 128], ["K", 10], ["dtype", "bfloat16"], ["seed", 610305], ["self_search", false], ["min_recall", 0.999]]}]]}, {"__dict_items__": [["label", "ksweep_q4096_m20000_d128_k1"], ["params", {"__dict_items__": [["B", 1], ["Q", 4096], ["M", 20000], ["D", 128], ["K", 1], ["dtype", "bfloat16"], ["seed", 610310], ["self_search", false], ["min_recall", 1.0]]}]]}, {"__dict_items__": [["label", "ksweep_q4096_m20000_d128_k2"], ["params", {"__dict_items__": [["B", 1], ["Q", 4096], ["M", 20000], ["D", 128], ["K", 2], ["dtype", "bfloat16"], ["seed", 610311], ["self_search", false], ["min_recall", 1.0]]}]]}, {"__dict_items__": [["label", "ksweep_q128_m131072_d128_k11"], ["params", {"__dict_items__": [["B", 1], ["Q", 128], ["M", 131072], ["D", 128], ["K", 11], ["dtype", "bfloat16"], ["seed", 610306], ["self_search", false], ["min_recall", 0.999]]}]]}, {"__dict_items__": [["label", "ksweep_q128_m131072_d128_k12"], ["params", {"__dict_items__": [["B", 1], ["Q", 128], ["M", 131072], ["D", 128], ["K", 12], ["dtype", "bfloat16"], ["seed", 610307], ["self_search", false], ["min_recall", 0.999]]}]]}, {"__dict_items__": [["label", "blind_k16_q128_m131072_d128_k16"], ["params", {"__dict_items__": [["B", 1], ["Q", 128], ["M", 131072], ["D", 128], ["K", 16], ["dtype", "bfloat16"], ["seed", 610517], ["self_search", false], ["min_recall", 0.999]]}]]}, {"__dict_items__": [["label", "ksweep_q128_m131072_d128_k20"], ["params", {"__dict_items__": [["B", 1], ["Q", 128], ["M", 131072], ["D", 128], ["K", 20], ["dtype", "bfloat16"], ["seed", 610308], ["self_search", false], ["min_recall", 0.999]]}]]}, {"__dict_items__": [["label", "ksweep_q128_m131072_d128_k30"], ["params", {"__dict_items__": [["B", 1], ["Q", 128], ["M", 131072], ["D", 128], ["K", 30], ["dtype", "bfloat16"], ["seed", 610309], ["self_search", false], ["min_recall", 0.999]]}]]}, {"__dict_items__": [["label", "ksweep_q128_m131072_d128_k32"], ["params", {"__dict_items__": [["B", 1], ["Q", 128], ["M", 131072], ["D", 128], ["K", 32], ["dtype", "bfloat16"], ["seed", 610315], ["self_search", false], ["min_recall", 0.999]]}]]}, {"__dict_items__": [["label", "blind_k48_q128_m131072_d128_k48"], ["params", {"__dict_items__": [["B", 1], ["Q", 128], ["M", 131072], ["D", 128], ["K", 48], ["dtype", "bfloat16"], ["seed", 610518], ["self_search", false], ["min_recall", 0.999]]}]]}, {"__dict_items__": [["label", "blind_k48_q4096_m32768_d128_k48"], ["params", {"__dict_items__": [["B", 1], ["Q", 4096], ["M", 32768], ["D", 128], ["K", 48], ["dtype", "bfloat16"], ["seed", 610519], ["self_search", false], ["min_recall", 0.999]]}]]}, {"__dict_items__": [["label", "ksweep_q128_m131072_d128_k64"], ["params", {"__dict_items__": [["B", 1], ["Q", 128], ["M", 131072], ["D", 128], ["K", 64], ["dtype", "bfloat16"], ["seed", 610312], ["self_search", false], ["min_recall", 0.999]]}]]}, {"__dict_items__": [["label", "ksweep_q4096_m20000_d128_k64"], ["params", {"__dict_items__": [["B", 1], ["Q", 4096], ["M", 20000], ["D", 128], ["K", 64], ["dtype", "bfloat16"], ["seed", 610313], ["self_search", false], ["min_recall", 0.999]]}]]}, {"__dict_items__": [["label", "blind_k64_q64_m131072_d128_k64"], ["params", {"__dict_items__": [["B", 1], ["Q", 64], ["M", 131072], ["D", 128], ["K", 64], ["dtype", "bfloat16"], ["seed", 610503], ["self_search", false], ["min_recall", 0.999]]}]]}, {"__dict_items__": [["label", "blind_k64_q128_m65536_d128_k64"], ["params", {"__dict_items__": [["B", 1], ["Q", 128], ["M", 65536], ["D", 128], ["K", 64], ["dtype", "bfloat16"], ["seed", 610504], ["self_search", false], ["min_recall", 0.999]]}]]}, {"__dict_items__": [["label", "blind_k64_q128_m262144_d128_k64"], ["params", {"__dict_items__": [["B", 1], ["Q", 128], ["M", 262144], ["D", 128], ["K", 64], ["dtype", "bfloat16"], ["seed", 610505], ["self_search", false], ["min_recall", 0.999]]}]]}, {"__dict_items__": [["label", "blind_k64_q512_m65536_d128_k64"], ["params", {"__dict_items__": [["B", 1], ["Q", 512], ["M", 65536], ["D", 128], ["K", 64], ["dtype", "bfloat16"], ["seed", 610506], ["self_search", false], ["min_recall", 0.999]]}]]}, {"__dict_items__": [["label", "blind_k64_q4096_m32768_d128_k64"], ["params", {"__dict_items__": [["B", 1], ["Q", 4096], ["M", 32768], ["D", 128], ["K", 64], ["dtype", "bfloat16"], ["seed", 610507], ["self_search", false], ["min_recall", 0.999]]}]]}]'))
SHAPE_DISPATCH_REGISTRY: tuple[dict[str, str], ...] = ({'shape_key': '176c_r3_exact_q4096_m20000_d128_k1_cd66', 'guard': 'B == 1 and Q == 4096 and M == 20000 and D == 128 and K == 1 and tcgen05', 'route': ROUTE_Q4096_K1_LOWK}, {'shape_key': '176c_r3_exact_q4096_m20000_d128_k2_split9', 'guard': 'B == 1 and Q == 4096 and M == 20000 and D == 128 and K == 2 and tcgen05', 'route': ROUTE_Q4096_K2_LOWK}, {'shape_key': '176c_r3_k_le_10_incumbent', 'guard': 'K <= 10; delegate to knn_search_mma_split_v1 incumbent', 'route': ROUTE_INCUMBENT_K10}, {'shape_key': '176c_r3_exact_q128_m131072_d128_k48_truek48', 'guard': 'B == 1 and Q == 128 and M == 131072 and D == 128 and K == 48 and tcgen05', 'route': ROUTE_TRUEK48}, {'shape_key': '176c_r3_d128_k_gt_10_extendedk_r2', 'guard': 'D == 128 and K > 10; delegate to round-2 extended-K Weave portfolio', 'route': ROUTE_EXTENDEDK_PORTFOLIO})

def _use_q4096_k1_lowk(inputs: dict[str, Any]) -> bool:
    return int(inputs['B']) == 1 and int(inputs['Q']) == 4096 and (int(inputs['M']) == 20000) and (int(inputs['D']) == D_STATIC) and (int(inputs['K']) == 1) and lowk_k1._use_qfull_merge8_highq_k1(inputs)

def _use_q4096_k2_lowk(inputs: dict[str, Any]) -> bool:
    return bool(lowk_k2._use_q4096_k2(inputs))

def selected_route(inputs: dict[str, Any]) -> str:
    if _use_q4096_k1_lowk(inputs):
        return ROUTE_Q4096_K1_LOWK
    if _use_q4096_k2_lowk(inputs):
        return ROUTE_Q4096_K2_LOWK
    if int(inputs['K']) <= K10_MAX:
        return ROUTE_INCUMBENT_K10
    if truek48._use_q128_k48_truek48(inputs):
        return ROUTE_TRUEK48
    return ROUTE_EXTENDEDK_PORTFOLIO

def selected_route_name(inputs: dict[str, Any]) -> str:
    return selected_route(inputs)

def _selected_entrypoint(route: str) -> str:
    if route == ROUTE_Q4096_K1_LOWK:
        return 'loom.examples.weave.knn_search_k1_post_merge8_0614_qfull_merge8_cd66_v1:launch_for_eval'
    if route == ROUTE_Q4096_K2_LOWK:
        return 'loom.examples.weave.knn_search_q4096_lowk_k2partial_split9_0613_r46_48e9_v1:launch_for_eval'
    if route == ROUTE_INCUMBENT_K10:
        return 'loom.examples.weave.knn_search_mma_split_v1:launch_for_eval'
    if route == ROUTE_TRUEK48:
        return 'loom.examples.weave.knn_search_k48_q128split512_truek48_kexact_0615_2d9eee_v1:launch_for_eval'
    return 'loom.examples.weave.knn_search_extendedk_dispatch0610_r2_k48k64portfolio_v1:launch_for_eval'

def _selected_guard(route: str) -> str:
    for entry in SHAPE_DISPATCH_REGISTRY:
        if entry.get('route') == route:
            return str(entry['guard'])
    return 'guard miss'

def route_info(inputs: dict[str, Any]) -> dict[str, Any]:
    route = selected_route(inputs)
    guard_order = [entry['shape_key'] for entry in SHAPE_DISPATCH_REGISTRY]
    if route in {ROUTE_Q4096_K1_LOWK, ROUTE_Q4096_K2_LOWK, ROUTE_INCUMBENT_K10}:
        return {'route': route, 'selected_route': route, 'selected_entrypoint': _selected_entrypoint(route), 'route_kind': 'specialized' if route != ROUTE_INCUMBENT_K10 else 'incumbent', 'coverage_class': 'performance_route_high_q_low_k' if route != ROUTE_INCUMBENT_K10 else 'preserved_k_le_10', 'coverage_only': False, 'production_policy': 'weave_only', 'external_fallback': None, 'selected_guard': _selected_guard(route), 'guard_order': guard_order, 'fallback': None}
    if route == ROUTE_TRUEK48:
        info = dict(truek48.route_info(inputs))
        info['guard_order'] = guard_order
        return info
    info = dict(extendedk.route_info(inputs))
    info['route'] = ROUTE_EXTENDEDK_PORTFOLIO
    info['selected_route'] = ROUTE_EXTENDEDK_PORTFOLIO
    info['replaced_route'] = extendedk.selected_route(inputs)
    info['selected_entrypoint'] = _selected_entrypoint(route)
    info['selected_guard'] = _selected_guard(route)
    info['guard_order'] = guard_order
    info.setdefault('production_policy', 'weave_only')
    info.setdefault('external_fallback', None)
    info.setdefault('coverage_only', False)
    return info

def route_trace_entry(label: str, inputs: dict[str, Any]) -> dict[str, Any]:
    return {'label': label, 'B': int(inputs['B']), 'Q': int(inputs['Q']), 'M': int(inputs['M']), 'D': int(inputs['D']), 'K': int(inputs['K']), **route_info(inputs)}

def launch_for_eval(inputs: dict[str, Any]) -> dict[str, Any]:
    route = selected_route(inputs)
    if route == ROUTE_Q4096_K1_LOWK:
        return lowk_k1.launch_for_eval(inputs)
    if route == ROUTE_Q4096_K2_LOWK:
        return lowk_k2.launch_for_eval(inputs)
    if route == ROUTE_INCUMBENT_K10:
        return incumbent.launch_for_eval(inputs)
    if route == ROUTE_TRUEK48:
        return truek48.launch_for_eval(inputs)
    return extendedk.launch_for_eval(inputs)

def knn_search_compile_and_launch_extk_sweep_176c_r3_lowkseed(*, benchmark: bool=True, shapes: list[dict[str, Any]] | None=None) -> dict[str, Any]:
    from .._dispatch_runtime import evaluate
    result = evaluate(launch_for_eval, shapes=EXTK_SWEEP_EVAL_SHAPES if shapes is None else shapes, benchmark=benchmark)
    result['passed'] = bool(result.get('summary', {}).get('all_correct'))
    print(result)
    return result
