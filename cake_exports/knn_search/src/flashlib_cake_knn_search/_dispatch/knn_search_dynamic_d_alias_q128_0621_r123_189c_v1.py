"""Dynamic-D q128 alias-bucket seed wrapper for D63/D129/D257.

Minimum target architecture: sm_100a. This additive bucket-kernel module owns
only ``B=1,Q=128,M=65536,K=10,D in {63,129,257}`` from the 0621 dynamic-D
breakthrough handoff. It does not edit production dispatch; it routes exact
contract rows to existing source-clean Weave tcgen05 seeds so the contract
harness can produce focused same-session evidence for the alias bucket.
"""
from __future__ import annotations
from json import loads as _json_loads
from .._dispatch_runtime import _capture_cuTensorMapEncodeTiled, _decode_capture, _import_dispatch_module, _ir_proxy
from typing import Any
from .._dispatch_runtime import select_named_shapes
from . import knn_search_dynamic_d_high_q128_directstride_tcgen05_0618_ccef_v1 as highd
from . import knn_search_dynamic_d_tiny_q128_m65536_tcgen05_0618_c8b9_v1 as tinyd
from . import knn_search_scalar_capacity_0611_r22_4e96_v1 as scalar_capacity
THREADS = highd.THREADS
MERGE_THREADS = highd.MERGE_THREADS
BLOCK_Q = highd.BLOCK_Q
BLOCK_M = highd.BLOCK_M
K_MAX = highd.K_MAX
ir = _decode_capture(_json_loads('{"__ir__": "knn_search_dynamic_d_high_directstride_tcgen05_partial_0618_ccef_v1", "arg_keys": ["queries", "database", "partial_distances", "partial_indices", "B", "Q", "M", "split_m", "num_q_tiles", "total_m_tiles", "tiles_per_split"], "cluster_dims": [1, 1, 1], "computed_smem_bytes": 126720, "constants": [["K_MAX_", 10], ["D_ORIG_", 512], ["NUM_D_PASSES_", 4], ["Q_NORM_PARTS_", 32]], "cta_group": 1, "threads": 640}'))
partial_ir = _decode_capture(_json_loads('{"__ir__": "knn_search_dynamic_d_high_directstride_tcgen05_partial_0618_ccef_v1", "arg_keys": ["queries", "database", "partial_distances", "partial_indices", "B", "Q", "M", "split_m", "num_q_tiles", "total_m_tiles", "tiles_per_split"], "cluster_dims": [1, 1, 1], "computed_smem_bytes": 126720, "constants": [["K_MAX_", 10], ["D_ORIG_", 512], ["NUM_D_PASSES_", 4], ["Q_NORM_PARTS_", 32]], "cta_group": 1, "threads": 640}'))
merge_ir = _decode_capture(_json_loads('{"__ir__": "knn_search_mma_split_merge_q128_const148_v1", "arg_keys": ["partial_distances", "partial_indices", "out_distances", "out_indices", "B", "Q", "K", "split_m", "num_q_tiles"], "cluster_dims": [1, 1, 1], "computed_smem_bytes": 0, "constants": [["K_MAX_", 10]], "cta_group": 1, "threads": 32}'))
tinyd_partial_ir = _decode_capture(_json_loads('{"__ir__": "knn_search_dynamic_d_tiny_q128_m65536_tcgen05_partial_0618_c8b9_v1", "arg_keys": ["queries", "database", "partial_distances", "partial_indices", "B", "Q", "M", "split_m", "num_q_tiles", "total_m_tiles", "tiles_per_split"], "cluster_dims": [1, 1, 1], "computed_smem_bytes": 112384, "constants": [["K_MAX_", 10], ["D_TOTAL_", 3], ["NUM_D_PASSES_", 1], ["Q_NORM_PARTS_", 1]], "cta_group": 1, "threads": 640}'))
tinyd_merge_ir = _decode_capture(_json_loads('{"__ir__": "knn_search_mma_split_merge_q128_const148_v1", "arg_keys": ["partial_distances", "partial_indices", "out_distances", "out_indices", "B", "Q", "K", "split_m", "num_q_tiles"], "cluster_dims": [1, 1, 1], "computed_smem_bytes": 0, "constants": [["K_MAX_", 10]], "cta_group": 1, "threads": 32}'))
scalar_capacity_ir = _decode_capture(_json_loads('{"__ir__": "knn_search_scalar_capacity_partial_v1", "arg_keys": ["queries", "database", "partial_distances", "partial_indices", "B", "Q", "M", "num_m_tiles"], "cluster_dims": [1, 1, 1], "computed_smem_bytes": 0, "constants": [["D_", 8], ["K_CAP_", 64], ["BLOCK_M_", 512], ["NUM_WARPS_", 8]], "cta_group": 1, "threads": 256}'))
ROUTE_DYNAMIC_D_ALIAS_D63_Q128 = '189c_dynamic_d63_q128_no_pack_tcgen05'
ROUTE_DYNAMIC_D_ALIAS_D129D257_Q128 = highd.ROUTE_HIGH_DYNAMIC_D_TCGEN05
ROUTE_SCALAR_CAPACITY = 'scalar_capacity_parent'
CONSUMED_D63_Q128_SEED = 'weave-evolve-knn-search-c8b9-nopack'
CONSUMED_D129D257_Q128_SEED = highd.CONSUMED_SEED
CONSUMED_SEEDS: tuple[str, ...] = (CONSUMED_D63_Q128_SEED, CONSUMED_D129D257_Q128_SEED)
TARGET_LABELS: tuple[str, ...] = ('blind_dyn_d63_q128_m65536_k10', 'blind_dyn_d129_q128_m65536_k10', 'blind_dyn_d257_q128_m65536_k10')
TARGET_SHAPES = _decode_capture(_json_loads('[{"__dict_items__": [["label", "blind_dyn_d63_q128_m65536_k10"], ["params", {"__dict_items__": [["B", 1], ["Q", 128], ["M", 65536], ["D", 63], ["K", 10], ["dtype", "bfloat16"], ["seed", 610803], ["self_search", false], ["min_recall", 0.999]]}]]}, {"__dict_items__": [["label", "blind_dyn_d129_q128_m65536_k10"], ["params", {"__dict_items__": [["B", 1], ["Q", 128], ["M", 65536], ["D", 129], ["K", 10], ["dtype", "bfloat16"], ["seed", 610804], ["self_search", false], ["min_recall", 0.999]]}]]}, {"__dict_items__": [["label", "blind_dyn_d257_q128_m65536_k10"], ["params", {"__dict_items__": [["B", 1], ["Q", 128], ["M", 65536], ["D", 257], ["K", 10], ["dtype", "bfloat16"], ["seed", 610805], ["self_search", false], ["min_recall", 0.999]]}]]}]'))
REQUESTED_BUT_MISSING_CONTRACT_LABELS: tuple[str, ...] = ('blind_ext_dyn_d63_q128_m65536_k10', 'blind_ext_dyn_d129_q128_m65536_k10', 'blind_ext_dyn_d257_q128_m65536_k10')
_D63_Q128_ENTRY: dict[str, Any] = {'shape_key': '189c_dynamic_d63_q128_m65536_k10', 'label': 'blind_dyn_d63_q128_m65536_k10', 'requested_label_alias': 'blind_ext_dyn_d63_q128_m65536_k10', 'guard': 'B == 1 and Q == 128 and M == 65536 and D == 63 and K == 10 and not self_search and not forced_fallback and arch in {sm_100a,sm_103a}', 'route': ROUTE_DYNAMIC_D_ALIAS_D63_Q128, 'entrypoint': 'loom.examples.weave.knn_search_dynamic_d_alias_q128_0621_r123_189c_v1:launch_for_eval', 'source_entrypoint': 'loom.examples.weave.knn_search_dynamic_d_tiny_q128_m65536_tcgen05_0618_c8b9_v1:launch_for_eval', 'selected_seed': CONSUMED_D63_Q128_SEED, 'source_round_doc': 'design_doc/active/weave_evolve_knn_search_round_2_c8b9_nopack.md', 'coverage_class': 'bucket_seed_dynamic_d63_q128_m65536_k10', 'arch_requirement': 'sm_100a'}
_D129_Q128_ENTRY: dict[str, Any] = {'shape_key': '189c_dynamic_d129_q128_m65536_k10', 'label': 'blind_dyn_d129_q128_m65536_k10', 'requested_label_alias': 'blind_ext_dyn_d129_q128_m65536_k10', 'guard': 'B == 1 and Q == 128 and M == 65536 and D == 129 and K == 10 and not self_search and not forced_fallback and arch in {sm_100a,sm_103a}', 'route': ROUTE_DYNAMIC_D_ALIAS_D129D257_Q128, 'entrypoint': 'loom.examples.weave.knn_search_dynamic_d_alias_q128_0621_r123_189c_v1:launch_for_eval', 'source_entrypoint': 'loom.examples.weave.knn_search_dynamic_d_high_q128_directstride_tcgen05_0618_ccef_v1:launch_for_eval', 'selected_seed': CONSUMED_D129D257_Q128_SEED, 'source_round_doc': 'design_doc/active/weave_evolve_knn_search_round_3_ccef_highd_directstride.md', 'coverage_class': 'bucket_seed_dynamic_d129_q128_m65536_k10', 'arch_requirement': 'sm_100a'}
_D257_Q128_ENTRY: dict[str, Any] = {'shape_key': '189c_dynamic_d257_q128_m65536_k10', 'label': 'blind_dyn_d257_q128_m65536_k10', 'requested_label_alias': 'blind_ext_dyn_d257_q128_m65536_k10', 'guard': 'B == 1 and Q == 128 and M == 65536 and D == 257 and K == 10 and not self_search and not forced_fallback and arch in {sm_100a,sm_103a}', 'route': ROUTE_DYNAMIC_D_ALIAS_D129D257_Q128, 'entrypoint': 'loom.examples.weave.knn_search_dynamic_d_alias_q128_0621_r123_189c_v1:launch_for_eval', 'source_entrypoint': 'loom.examples.weave.knn_search_dynamic_d_high_q128_directstride_tcgen05_0618_ccef_v1:launch_for_eval', 'selected_seed': CONSUMED_D129D257_Q128_SEED, 'source_round_doc': 'design_doc/active/weave_evolve_knn_search_round_3_ccef_highd_directstride.md', 'coverage_class': 'bucket_seed_dynamic_d257_q128_m65536_k10', 'arch_requirement': 'sm_100a'}
SHAPE_DISPATCH_REGISTRY: tuple[dict[str, Any], ...] = (_D63_Q128_ENTRY, _D129_Q128_ENTRY, _D257_Q128_ENTRY)

def _shape_key(inputs: dict[str, Any]) -> tuple[int, int, int, int, int, bool]:
    return (int(inputs.get('B', 1)), int(inputs['Q']), int(inputs['M']), int(inputs['D']), int(inputs['K']), bool(inputs.get('self_search', False)))

def _forced_fallback(inputs: dict[str, Any]) -> bool:
    return bool(inputs.get('force_fallback', False))

def _use_d63_q128(inputs: dict[str, Any]) -> bool:
    return _shape_key(inputs) == (1, 128, 65536, 63, 10, False) and (not _forced_fallback(inputs)) and tinyd.mma._tcgen05_capable_arch()

def _use_d129_q128(inputs: dict[str, Any]) -> bool:
    return _shape_key(inputs) == (1, 128, 65536, 129, 10, False) and (not _forced_fallback(inputs)) and highd.mma._tcgen05_capable_arch()

def _use_d257_q128(inputs: dict[str, Any]) -> bool:
    return _shape_key(inputs) == (1, 128, 65536, 257, 10, False) and (not _forced_fallback(inputs)) and highd.mma._tcgen05_capable_arch()

def _active_entry(inputs: dict[str, Any]) -> dict[str, Any] | None:
    if _use_d63_q128(inputs):
        return _D63_Q128_ENTRY
    if _use_d129_q128(inputs):
        return _D129_Q128_ENTRY
    if _use_d257_q128(inputs):
        return _D257_Q128_ENTRY
    return None

def selected_route(inputs: dict[str, Any]) -> str:
    entry = _active_entry(inputs)
    if entry is None:
        return ROUTE_SCALAR_CAPACITY
    return str(entry['route'])

def selected_route_name(inputs: dict[str, Any]) -> str:
    return selected_route(inputs)

def _fallback_info(inputs: dict[str, Any]) -> dict[str, Any]:
    return {'route': ROUTE_SCALAR_CAPACITY, 'selected_route': ROUTE_SCALAR_CAPACITY, 'selected_entrypoint': 'loom.examples.weave.knn_search_scalar_capacity_0611_r22_4e96_v1:launch_for_eval', 'route_kind': 'general', 'route_source': 'generic-weave-fallback', 'classification': 'guard-miss', 'coverage_only': False, 'production_policy': 'weave_only', 'external_fallback': None, 'guard_order': [str(entry['shape_key']) for entry in SHAPE_DISPATCH_REGISTRY], 'forced_fallback': _forced_fallback(inputs), 'missing_weave_route': False}

def route_info(inputs: dict[str, Any]) -> dict[str, Any]:
    entry = _active_entry(inputs)
    if entry is None:
        return _fallback_info(inputs)
    return {'route': entry['route'], 'selected_route': entry['route'], 'selected_entrypoint': entry['entrypoint'], 'source_entrypoint': entry['source_entrypoint'], 'route_kind': 'specialized', 'route_source': 'shape-specific-seed', 'coverage_class': entry['coverage_class'], 'classification': 'seed-produced', 'coverage_only': False, 'production_policy': 'weave_only', 'external_fallback': None, 'guard_order': [str(item['shape_key']) for item in SHAPE_DISPATCH_REGISTRY], 'guard_id': entry['shape_key'], 'selected_guard': entry['guard'], 'fallback': ROUTE_SCALAR_CAPACITY, 'missing_weave_route': False, 'forced_fallback': False, 'selected_seed': entry['selected_seed'], 'source_round_doc': entry['source_round_doc'], 'arch_requirement': entry['arch_requirement'], 'requested_label_alias': entry['requested_label_alias']}

def route_trace_entry(label: str, inputs: dict[str, Any]) -> dict[str, Any]:
    return {'label': label, 'shape_key': label, 'B': int(inputs['B']), 'Q': int(inputs['Q']), 'M': int(inputs['M']), 'D': int(inputs['D']), 'K': int(inputs['K']), 'self_search': bool(inputs.get('self_search', False)), **route_info(inputs)}

def launch_for_eval(inputs: dict[str, Any]) -> dict[str, Any]:
    entry = _active_entry(inputs)
    if entry is _D63_Q128_ENTRY:
        return tinyd.launch_for_eval(inputs)
    if entry is _D129_Q128_ENTRY or entry is _D257_Q128_ENTRY:
        return highd.launch_for_eval(inputs)
    return scalar_capacity.launch_for_eval(inputs)

def _select_contract_shapes(shape_labels: str | tuple[str, ...] | list[str] | None):
    if shape_labels is None:
        return TARGET_SHAPES
    return select_named_shapes(shape_labels)

def knn_search_compile_and_launch_dynamic_d_alias_q128_0621_r123_189c(*, benchmark: bool=True, shape_labels: str | tuple[str, ...] | list[str] | None=TARGET_LABELS) -> dict[str, Any]:
    from .._dispatch_runtime import evaluate
    result = evaluate(launch_for_eval, shapes=_select_contract_shapes(shape_labels), benchmark=benchmark)
    result['passed'] = bool(result.get('summary', {}).get('all_correct'))
    print(result)
    return result
