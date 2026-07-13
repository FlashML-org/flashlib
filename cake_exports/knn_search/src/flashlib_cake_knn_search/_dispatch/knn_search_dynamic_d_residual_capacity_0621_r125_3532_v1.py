"""Residual dynamic-D scalar-capacity seed bundle.

Minimum target architecture: sm_100a for all tcgen05/TMEM seed routes. This
additive bucket-kernel module targets the round-124 residual dynamic-D
fallback-slow rows and does not edit the production dispatcher. Guard misses
delegate to the r124 dynamic-D replay wrapper so the module can be consumed as a
shape-specific seed bundle by the generalize-auto-tuning dispatcher.
"""
from __future__ import annotations
from json import loads as _json_loads
from .._dispatch_runtime import _capture_cuTensorMapEncodeTiled, _decode_capture, _import_dispatch_module, _ir_proxy
from typing import Any
from .._dispatch_runtime import select_named_shapes
from . import knn_search_dynamic_d_d384_q32_m131072_exact_tcgen05_0618_5847_v1 as d384_q32
from . import knn_search_dynamic_d_high_q128_directstride_tcgen05_0618_ccef_v1 as highd_direct
from . import knn_search_dynamic_d_k64_q64_m65536_tcgen05_0618_ccef_v2 as k64_dynamic
from . import knn_search_dynamic_d_remaining_seeds_0618_ccef_v2 as remaining
from . import knn_search_dynamic_d_scalar_breakthrough_0621_r124_f670_extrows_v1 as parent
from . import knn_search_dynamic_d_tiny_q128_m65536_tcgen05_0618_c8b9_v1 as tinyd_no_pack
THREADS = parent.THREADS
MERGE_THREADS = parent.MERGE_THREADS
BLOCK_Q = parent.BLOCK_Q
BLOCK_M = parent.BLOCK_M
K_MAX = parent.K_MAX
ir = _decode_capture(_json_loads('{"__ir__": "knn_search_dynamic_d_high_directstride_tcgen05_partial_0618_ccef_v1", "arg_keys": ["queries", "database", "partial_distances", "partial_indices", "B", "Q", "M", "split_m", "num_q_tiles", "total_m_tiles", "tiles_per_split"], "cluster_dims": [1, 1, 1], "computed_smem_bytes": 126720, "constants": [["K_MAX_", 10], ["D_ORIG_", 512], ["NUM_D_PASSES_", 4], ["Q_NORM_PARTS_", 32]], "cta_group": 1, "threads": 640}'))
partial_ir = _decode_capture(_json_loads('{"__ir__": "knn_search_dynamic_d_high_directstride_tcgen05_partial_0618_ccef_v1", "arg_keys": ["queries", "database", "partial_distances", "partial_indices", "B", "Q", "M", "split_m", "num_q_tiles", "total_m_tiles", "tiles_per_split"], "cluster_dims": [1, 1, 1], "computed_smem_bytes": 126720, "constants": [["K_MAX_", 10], ["D_ORIG_", 512], ["NUM_D_PASSES_", 4], ["Q_NORM_PARTS_", 32]], "cta_group": 1, "threads": 640}'))
merge_ir = _decode_capture(_json_loads('{"__ir__": "knn_search_mma_split_merge_q128_const148_v1", "arg_keys": ["partial_distances", "partial_indices", "out_distances", "out_indices", "B", "Q", "K", "split_m", "num_q_tiles"], "cluster_dims": [1, 1, 1], "computed_smem_bytes": 0, "constants": [["K_MAX_", 10]], "cta_group": 1, "threads": 32}'))
parent_ir = _decode_capture(_json_loads('{"__ir__": "knn_search_dynamic_d_high_directstride_tcgen05_partial_0618_ccef_v1", "arg_keys": ["queries", "database", "partial_distances", "partial_indices", "B", "Q", "M", "split_m", "num_q_tiles", "total_m_tiles", "tiles_per_split"], "cluster_dims": [1, 1, 1], "computed_smem_bytes": 126720, "constants": [["K_MAX_", 10], ["D_ORIG_", 512], ["NUM_D_PASSES_", 4], ["Q_NORM_PARTS_", 32]], "cta_group": 1, "threads": 640}'))
tinyd_partial_ir = _decode_capture(_json_loads('{"__ir__": "knn_search_dynamic_d_tiny_q128_m65536_tcgen05_partial_0618_c8b9_v1", "arg_keys": ["queries", "database", "partial_distances", "partial_indices", "B", "Q", "M", "split_m", "num_q_tiles", "total_m_tiles", "tiles_per_split"], "cluster_dims": [1, 1, 1], "computed_smem_bytes": 112384, "constants": [["K_MAX_", 10], ["D_TOTAL_", 3], ["NUM_D_PASSES_", 1], ["Q_NORM_PARTS_", 1]], "cta_group": 1, "threads": 640}'))
tinyd_merge_ir = _decode_capture(_json_loads('{"__ir__": "knn_search_mma_split_merge_q128_const148_v1", "arg_keys": ["partial_distances", "partial_indices", "out_distances", "out_indices", "B", "Q", "K", "split_m", "num_q_tiles"], "cluster_dims": [1, 1, 1], "computed_smem_bytes": 0, "constants": [["K_MAX_", 10]], "cta_group": 1, "threads": 32}'))
highd_direct_partial_ir = _decode_capture(_json_loads('{"__ir__": "knn_search_dynamic_d_high_directstride_tcgen05_partial_0618_ccef_v1", "arg_keys": ["queries", "database", "partial_distances", "partial_indices", "B", "Q", "M", "split_m", "num_q_tiles", "total_m_tiles", "tiles_per_split"], "cluster_dims": [1, 1, 1], "computed_smem_bytes": 126720, "constants": [["K_MAX_", 10], ["D_ORIG_", 512], ["NUM_D_PASSES_", 4], ["Q_NORM_PARTS_", 32]], "cta_group": 1, "threads": 640}'))
highd_direct_merge_ir = _decode_capture(_json_loads('{"__ir__": "knn_search_mma_split_merge_q128_const148_v1", "arg_keys": ["partial_distances", "partial_indices", "out_distances", "out_indices", "B", "Q", "K", "split_m", "num_q_tiles"], "cluster_dims": [1, 1, 1], "computed_smem_bytes": 0, "constants": [["K_MAX_", 10]], "cta_group": 1, "threads": 32}'))
d384_q32_partial_ir = _decode_capture(_json_loads('{"__ir__": "knn_search_dynamic_d_d384_q32_m131072_exact_tcgen05_partial_0618_5847_v1", "arg_keys": ["queries", "database", "partial_distances", "partial_indices", "B", "Q", "M", "split_m", "num_q_tiles", "total_m_tiles", "tiles_per_split"], "cluster_dims": [1, 1, 1], "computed_smem_bytes": 122624, "constants": [["K_MAX_", 10]], "cta_group": 1, "threads": 640}'))
d384_q32_merge_ir = _decode_capture(_json_loads('{"__ir__": "knn_search_mma_split_merge_q128_const148_v1", "arg_keys": ["partial_distances", "partial_indices", "out_distances", "out_indices", "B", "Q", "K", "split_m", "num_q_tiles"], "cluster_dims": [1, 1, 1], "computed_smem_bytes": 0, "constants": [["K_MAX_", 10]], "cta_group": 1, "threads": 32}'))
k64_dynamic_partial_ir = _decode_capture(_json_loads('{"__ir__": "knn_search_dynamic_d257_k64_q64_partial_0618_ccef_v1", "arg_keys": ["queries", "database", "partial_distances", "partial_indices", "B", "Q", "M", "split_m", "num_q_tiles", "total_m_tiles", "tiles_per_split", "partial_list_count"], "cluster_dims": [1, 1, 1], "computed_smem_bytes": 151296, "constants": [["K_MAX_", 64]], "cta_group": 1, "threads": 256}'))
k64_dynamic_merge_ir = _decode_capture(_json_loads('{"__ir__": "knn_search_dynamic_d257_k64_q64_merge_0618_ccef_v1", "arg_keys": ["partial_distances", "partial_indices", "out_distances", "out_indices", "B", "Q", "K", "partial_list_count", "num_q_tiles"], "cluster_dims": [1, 1, 1], "computed_smem_bytes": 0, "constants": [["K_MAX_", 64]], "cta_group": 1, "threads": 32}'))
b2_dynamic_ir = _decode_capture(_json_loads('{"__ir__": "knn_search_dynamic_d511_tcgen05_partial_0618_c8b9_v1", "arg_keys": ["queries", "database", "partial_distances", "partial_indices", "B", "Q", "M", "split_m", "num_q_tiles", "total_m_tiles", "tiles_per_split"], "cluster_dims": [1, 1, 1], "computed_smem_bytes": 126720, "constants": [["K_MAX_", 10], ["D_TOTAL_", 512], ["NUM_D_PASSES_", 4], ["Q_NORM_PARTS_", 32]], "cta_group": 1, "threads": 640}'))
ROUTE_DYNAMIC_D7_Q128 = 'c8b9_tiny_dynamic_d_no_pack_guarded_tcgen05'
ROUTE_DYNAMIC_D511_Q128 = highd_direct.ROUTE_HIGH_DYNAMIC_D_TCGEN05
ROUTE_DYNAMIC_D384_Q32 = d384_q32.ROUTE_D384_Q32_EXACT_TCGEN05
ROUTE_DYNAMIC_D257_Q64_K64 = k64_dynamic.ROUTE_DYNAMIC_D257_Q64_K64
ROUTE_DYNAMIC_B2_Q64_D129 = remaining.parent.ROUTE_B2_Q64_D129
ROUTE_PARENT_FALLBACK = 'r124_dynamic_d_replay_parent'
CONSUMED_D7_SEED = 'weave-evolve-knn-search-449d'
CONSUMED_D511_SEED = highd_direct.CONSUMED_SEED
CONSUMED_D384_Q32_SEED = 'weave-evolve-knn-search-5847-d384-q32-exact'
CONSUMED_D257_Q64_K64_SEED = k64_dynamic.CONSUMED_SEED
CONSUMED_B2_Q64_D129_SEED = 'weave-evolve-knn-search-a2ab'
CONSUMED_PARENT_SEEDS: tuple[str, ...] = parent.CONSUMED_SEEDS
CONSUMED_SEEDS: tuple[str, ...] = (*CONSUMED_PARENT_SEEDS, CONSUMED_D7_SEED, CONSUMED_D511_SEED, CONSUMED_D384_Q32_SEED, CONSUMED_D257_Q64_K64_SEED, CONSUMED_B2_Q64_D129_SEED)
TARGET_LABELS: tuple[str, ...] = ('blind_dyn_d7_q128_m65536_k10', 'blind_dyn_d511_q128_m65536_k10', 'blind_dyn_d384_q32_m131072_k10', 'blind_dyn_d257_k64_q64_m65536', 'blind_dyn_b2_q64_m65536_d129_k10')
TARGET_SHAPES = _decode_capture(_json_loads('[{"__dict_items__": [["label", "blind_dyn_d7_q128_m65536_k10"], ["params", {"__dict_items__": [["B", 1], ["Q", 128], ["M", 65536], ["D", 7], ["K", 10], ["dtype", "bfloat16"], ["seed", 610802], ["self_search", false], ["min_recall", 0.999]]}]]}, {"__dict_items__": [["label", "blind_dyn_d511_q128_m65536_k10"], ["params", {"__dict_items__": [["B", 1], ["Q", 128], ["M", 65536], ["D", 511], ["K", 10], ["dtype", "bfloat16"], ["seed", 610806], ["self_search", false], ["min_recall", 0.999]]}]]}, {"__dict_items__": [["label", "blind_dyn_d384_q32_m131072_k10"], ["params", {"__dict_items__": [["B", 1], ["Q", 32], ["M", 131072], ["D", 384], ["K", 10], ["dtype", "bfloat16"], ["seed", 610807], ["self_search", false], ["min_recall", 0.999]]}]]}, {"__dict_items__": [["label", "blind_dyn_d257_k64_q64_m65536"], ["params", {"__dict_items__": [["B", 1], ["Q", 64], ["M", 65536], ["D", 257], ["K", 64], ["dtype", "bfloat16"], ["seed", 610808], ["self_search", false], ["min_recall", 0.999]]}]]}, {"__dict_items__": [["label", "blind_dyn_b2_q64_m65536_d129_k10"], ["params", {"__dict_items__": [["B", 2], ["Q", 64], ["M", 65536], ["D", 129], ["K", 10], ["dtype", "bfloat16"], ["seed", 610809], ["self_search", false], ["min_recall", 0.999]]}]]}]'))
_D7_Q128_ENTRY: dict[str, Any] = {'shape_key': 'r125_3532_dynamic_d7_q128_m65536_k10', 'label': 'blind_dyn_d7_q128_m65536_k10', 'guard': 'B == 1 and Q == 128 and M == 65536 and D == 7 and K == 10 and not self_search and not forced_fallback and arch in {sm_100a,sm_103a}', 'route': ROUTE_DYNAMIC_D7_Q128, 'entrypoint': 'loom.examples.weave.knn_search_dynamic_d_residual_capacity_0621_r125_3532_v1:launch_for_eval', 'source_entrypoint': 'loom.examples.weave.knn_search_dynamic_d_tiny_q128_m65536_tcgen05_0618_c8b9_v1:launch_for_eval', 'selected_seed': CONSUMED_D7_SEED, 'source_round_doc': 'design_doc/active/generalize_auto_tuning_knn_search_round_2_b59b.md', 'coverage_class': 'bucket_seed_dynamic_d7_q128_m65536_k10', 'arch_requirement': 'sm_100a'}
_D511_Q128_ENTRY: dict[str, Any] = {'shape_key': 'r125_3532_dynamic_d511_q128_m65536_k10', 'label': 'blind_dyn_d511_q128_m65536_k10', 'guard': 'B == 1 and Q == 128 and M == 65536 and D == 511 and K == 10 and not self_search and not forced_fallback and arch in {sm_100a,sm_103a}', 'route': ROUTE_DYNAMIC_D511_Q128, 'entrypoint': 'loom.examples.weave.knn_search_dynamic_d_residual_capacity_0621_r125_3532_v1:launch_for_eval', 'source_entrypoint': 'loom.examples.weave.knn_search_dynamic_d_high_q128_directstride_tcgen05_0618_ccef_v1:launch_for_eval', 'selected_seed': CONSUMED_D511_SEED, 'source_round_doc': 'design_doc/active/weave_evolve_knn_search_round_3_ccef_highd_directstride.md', 'coverage_class': 'bucket_seed_dynamic_d511_q128_m65536_k10', 'arch_requirement': 'sm_100a'}
_D384_Q32_ENTRY: dict[str, Any] = {'shape_key': 'r125_3532_dynamic_d384_q32_m131072_k10', 'label': 'blind_dyn_d384_q32_m131072_k10', 'guard': 'B == 1 and Q == 32 and M == 131072 and D == 384 and K == 10 and not self_search and not forced_fallback and arch in {sm_100a,sm_103a}', 'route': ROUTE_DYNAMIC_D384_Q32, 'entrypoint': 'loom.examples.weave.knn_search_dynamic_d_residual_capacity_0621_r125_3532_v1:launch_for_eval', 'source_entrypoint': 'loom.examples.weave.knn_search_dynamic_d_d384_q32_m131072_exact_tcgen05_0618_5847_v1:launch_for_eval', 'selected_seed': CONSUMED_D384_Q32_SEED, 'source_round_doc': 'design_doc/active/weave_evolve_knn_search_round_112_5847_d384_q32_exact.md', 'coverage_class': 'bucket_seed_dynamic_d384_q32_m131072_k10', 'arch_requirement': 'sm_100a'}
_D257_Q64_K64_ENTRY: dict[str, Any] = {'shape_key': 'r125_3532_dynamic_d257_q64_m65536_k64', 'label': 'blind_dyn_d257_k64_q64_m65536', 'guard': 'B == 1 and Q == 64 and M == 65536 and D == 257 and K == 64 and not self_search and not forced_fallback and arch in {sm_100a,sm_103a}', 'route': ROUTE_DYNAMIC_D257_Q64_K64, 'entrypoint': 'loom.examples.weave.knn_search_dynamic_d_residual_capacity_0621_r125_3532_v1:launch_for_eval', 'source_entrypoint': 'loom.examples.weave.knn_search_dynamic_d_k64_q64_m65536_tcgen05_0618_ccef_v2:launch_for_eval', 'selected_seed': CONSUMED_D257_Q64_K64_SEED, 'source_round_doc': 'design_doc/active/weave_evolve_knn_search_round_4_ccef_k64_guard_elision.md', 'coverage_class': 'bucket_seed_dynamic_d257_q64_m65536_k64', 'arch_requirement': 'sm_100a'}
_B2_Q64_D129_ENTRY: dict[str, Any] = {'shape_key': 'r125_3532_dynamic_b2_q64_m65536_d129_k10', 'label': 'blind_dyn_b2_q64_m65536_d129_k10', 'guard': 'B == 2 and Q == 64 and M == 65536 and D == 129 and K == 10 and not self_search and not forced_fallback and arch in {sm_100a,sm_103a}', 'route': ROUTE_DYNAMIC_B2_Q64_D129, 'entrypoint': 'loom.examples.weave.knn_search_dynamic_d_residual_capacity_0621_r125_3532_v1:launch_for_eval', 'source_entrypoint': 'loom.examples.weave.knn_search_dynamic_d_remaining_seeds_0618_ccef_v2:launch_for_eval', 'selected_seed': CONSUMED_B2_Q64_D129_SEED, 'source_round_doc': 'design_doc/active/weave_evolve_knn_search_round_4_ccef_k64_dynamic_d.md', 'coverage_class': 'bucket_seed_dynamic_b2_q64_m65536_d129_k10', 'arch_requirement': 'sm_100a'}
SHAPE_DISPATCH_REGISTRY: tuple[dict[str, Any], ...] = (_D7_Q128_ENTRY, _D511_Q128_ENTRY, _D384_Q32_ENTRY, _D257_Q64_K64_ENTRY, _B2_Q64_D129_ENTRY, *parent.SHAPE_DISPATCH_REGISTRY)

def _forced_fallback(inputs: dict[str, Any]) -> bool:
    return bool(inputs.get('force_fallback', False))

def _active_residual_entry(inputs: dict[str, Any]) -> dict[str, Any] | None:
    if not _forced_fallback(inputs) and tinyd_no_pack._use_tiny_dynamic_d_mma(inputs) and (int(inputs['D']) == 7):
        return _D7_Q128_ENTRY
    if highd_direct._use_high_dynamic_d_tcgen05(inputs) and int(inputs['D']) == 511:
        return _D511_Q128_ENTRY
    if d384_q32._use_d384_q32_exact_tcgen05(inputs):
        return _D384_Q32_ENTRY
    if k64_dynamic._use_dynamic_d257_q64_k64(inputs):
        return _D257_Q64_K64_ENTRY
    if remaining.parent._is_b2_q64_d129(inputs):
        return _B2_Q64_D129_ENTRY
    return None

def _guard_order() -> list[str]:
    return [str(entry['shape_key']) for entry in SHAPE_DISPATCH_REGISTRY]

def selected_route(inputs: dict[str, Any]) -> str:
    entry = _active_residual_entry(inputs)
    if entry is not None:
        return str(entry['route'])
    return parent.selected_route(inputs)

def selected_route_name(inputs: dict[str, Any]) -> str:
    return selected_route(inputs)

def _residual_info(entry: dict[str, Any], inputs: dict[str, Any]) -> dict[str, Any]:
    parent_info = dict(parent.route_info(inputs))
    parent_route = str(parent_info.get('route') or parent_info.get('selected_route') or parent.selected_route(inputs))
    return {'route': entry['route'], 'selected_route': entry['route'], 'selected_entrypoint': entry['entrypoint'], 'source_entrypoint': entry['source_entrypoint'], 'parent_route': parent_route, 'replaced_route': parent_route, 'route_kind': 'specialized', 'route_source': 'shape-specific-seed', 'coverage_class': entry['coverage_class'], 'classification': 'seed-produced', 'coverage_only': False, 'production_policy': 'weave_only', 'external_fallback': None, 'guard_order': _guard_order(), 'guard_id': entry['shape_key'], 'selected_guard': entry['guard'], 'fallback': ROUTE_PARENT_FALLBACK, 'missing_weave_route': False, 'forced_fallback': False, 'selected_seed': entry['selected_seed'], 'source_round_doc': entry['source_round_doc'], 'arch_requirement': entry['arch_requirement']}

def route_info(inputs: dict[str, Any]) -> dict[str, Any]:
    entry = _active_residual_entry(inputs)
    if entry is not None:
        return _residual_info(entry, inputs)
    info = dict(parent.route_info(inputs))
    info['guard_order'] = _guard_order()
    return info

def route_trace_entry(label: str, inputs: dict[str, Any]) -> dict[str, Any]:
    return {'label': label, 'shape_key': label, 'B': int(inputs['B']), 'Q': int(inputs['Q']), 'M': int(inputs['M']), 'D': int(inputs['D']), 'K': int(inputs['K']), 'self_search': bool(inputs.get('self_search', False)), **route_info(inputs)}

def launch_for_eval(inputs: dict[str, Any]) -> dict[str, Any]:
    entry = _active_residual_entry(inputs)
    if entry is _D7_Q128_ENTRY:
        return tinyd_no_pack.launch_for_eval(inputs)
    if entry is _D511_Q128_ENTRY:
        return highd_direct.launch_for_eval(inputs)
    if entry is _D384_Q32_ENTRY:
        return d384_q32.launch_for_eval(inputs)
    if entry is _D257_Q64_K64_ENTRY:
        return k64_dynamic.launch_for_eval(inputs)
    if entry is _B2_Q64_D129_ENTRY:
        return remaining.launch_for_eval(inputs)
    return parent.launch_for_eval(inputs)

def _select_contract_shapes(shape_labels: str | tuple[str, ...] | list[str] | None):
    if shape_labels is None:
        return TARGET_SHAPES
    return select_named_shapes(shape_labels)

def knn_search_compile_and_launch_dynamic_d_residual_capacity_0621_r125_3532(*, benchmark: bool=True, shape_labels: str | tuple[str, ...] | list[str] | None=TARGET_LABELS) -> dict[str, Any]:
    from .._dispatch_runtime import evaluate
    result = evaluate(launch_for_eval, shapes=_select_contract_shapes(shape_labels), benchmark=benchmark)
    result['passed'] = bool(result.get('summary', {}).get('all_correct'))
    print(result)
    return result
