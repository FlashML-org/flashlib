"""Priority dynamic-D seed aggregator for the 4832 auto-tuning lane.

Minimum target architecture: sm_80 for D3 tile-reduce routes; sm_100a for
tcgen05/TMEM dynamic-D routes. This additive bucket-kernel module does not
edit the production dispatcher. It exposes exact guards for the dynamic-D rows
requested by ``generalize_auto_tuning_knn_search_round_122_4832`` and routes
them to previously measured Weave-only seed kernels.
"""
from __future__ import annotations
from json import loads as _json_loads
from .._dispatch_runtime import _capture_cuTensorMapEncodeTiled, _decode_capture, _import_dispatch_module, _ir_proxy
from typing import Any
from .._dispatch_runtime import select_named_shapes
from . import knn_search_dispatch0618_c492_ext_dynamic_highd_9286_v1 as ext_highd
from . import knn_search_dynamic_d512_q64_tcgen05_0618_9286_v1 as d512_q64
from . import knn_search_dynamic_d768d1024_q32q16_tcgen05_0618_9286_v1 as d768d1024
from . import knn_search_dynamic_d_high_q128_directstride_tcgen05_0618_ccef_v1 as highd_q128
from . import knn_search_dynamic_d_remaining_seeds_0618_ccef_v2 as remaining
from . import knn_search_dynamic_d_tiny_q128_m65536_tcgen05_0618_c8b9_v1 as tiny_q128
from . import knn_search_scalar_capacity_0611_r22_4e96_v1 as scalar_capacity
THREADS = _decode_capture(_json_loads('640'))
MERGE_THREADS = _decode_capture(_json_loads('32'))
K_MAX = 10
tiny_q128_ir = _decode_capture(_json_loads('{"__ir__": "knn_search_dynamic_d_tiny_q128_m65536_tcgen05_partial_0618_c8b9_v1", "arg_keys": ["queries", "database", "partial_distances", "partial_indices", "B", "Q", "M", "split_m", "num_q_tiles", "total_m_tiles", "tiles_per_split"], "cluster_dims": [1, 1, 1], "computed_smem_bytes": 112384, "constants": [["K_MAX_", 10], ["D_TOTAL_", 3], ["NUM_D_PASSES_", 1], ["Q_NORM_PARTS_", 1]], "cta_group": 1, "threads": 640}'))
tiny_q128_partial_ir = _decode_capture(_json_loads('{"__ir__": "knn_search_dynamic_d_tiny_q128_m65536_tcgen05_partial_0618_c8b9_v1", "arg_keys": ["queries", "database", "partial_distances", "partial_indices", "B", "Q", "M", "split_m", "num_q_tiles", "total_m_tiles", "tiles_per_split"], "cluster_dims": [1, 1, 1], "computed_smem_bytes": 112384, "constants": [["K_MAX_", 10], ["D_TOTAL_", 3], ["NUM_D_PASSES_", 1], ["Q_NORM_PARTS_", 1]], "cta_group": 1, "threads": 640}'))
tiny_q128_merge_ir = _decode_capture(_json_loads('{"__ir__": "knn_search_mma_split_merge_q128_const148_v1", "arg_keys": ["partial_distances", "partial_indices", "out_distances", "out_indices", "B", "Q", "K", "split_m", "num_q_tiles"], "cluster_dims": [1, 1, 1], "computed_smem_bytes": 0, "constants": [["K_MAX_", 10]], "cta_group": 1, "threads": 32}'))
ext_highd_partial_ir = _decode_capture(_json_loads('{"__ir__": "knn_search_dynamic_d_high_directstride_tcgen05_partial_0618_ccef_v1", "arg_keys": ["queries", "database", "partial_distances", "partial_indices", "B", "Q", "M", "split_m", "num_q_tiles", "total_m_tiles", "tiles_per_split"], "cluster_dims": [1, 1, 1], "computed_smem_bytes": 126720, "constants": [["K_MAX_", 10], ["D_ORIG_", 512], ["NUM_D_PASSES_", 4], ["Q_NORM_PARTS_", 32]], "cta_group": 1, "threads": 640}'))
ext_highd_merge_ir = _decode_capture(_json_loads('{"__ir__": "knn_search_mma_split_merge_q128_const148_v1", "arg_keys": ["partial_distances", "partial_indices", "out_distances", "out_indices", "B", "Q", "K", "split_m", "num_q_tiles"], "cluster_dims": [1, 1, 1], "computed_smem_bytes": 0, "constants": [["K_MAX_", 10]], "cta_group": 1, "threads": 32}'))
highd_q128_partial_ir = _decode_capture(_json_loads('{"__ir__": "knn_search_dynamic_d_high_directstride_tcgen05_partial_0618_ccef_v1", "arg_keys": ["queries", "database", "partial_distances", "partial_indices", "B", "Q", "M", "split_m", "num_q_tiles", "total_m_tiles", "tiles_per_split"], "cluster_dims": [1, 1, 1], "computed_smem_bytes": 126720, "constants": [["K_MAX_", 10], ["D_ORIG_", 512], ["NUM_D_PASSES_", 4], ["Q_NORM_PARTS_", 32]], "cta_group": 1, "threads": 640}'))
highd_q128_merge_ir = _decode_capture(_json_loads('{"__ir__": "knn_search_mma_split_merge_q128_const148_v1", "arg_keys": ["partial_distances", "partial_indices", "out_distances", "out_indices", "B", "Q", "K", "split_m", "num_q_tiles"], "cluster_dims": [1, 1, 1], "computed_smem_bytes": 0, "constants": [["K_MAX_", 10]], "cta_group": 1, "threads": 32}'))
self_d3_ir = _decode_capture(_json_loads('{"__ir__": "knn_search_dynamic_d3_tile_reduce_partial_0618_c8b9_v1", "arg_keys": ["queries", "database", "partial_distances", "partial_indices", "B", "Q", "M", "K", "num_m_tiles"], "cluster_dims": [1, 1, 1], "computed_smem_bytes": 10240, "constants": [["K_MAX_", 10]], "cta_group": 1, "threads": 256}'))
d512_q64_ir = _decode_capture(_json_loads('{"__ir__": "knn_search_dynamic_d512_q64_tcgen05_partial_0618_9286_v1", "arg_keys": ["queries", "database", "partial_distances", "partial_indices", "B", "Q", "M", "split_m", "num_q_tiles", "total_m_tiles"], "cluster_dims": [1, 1, 1], "computed_smem_bytes": 102144, "constants": [["K_MAX_", 10]], "cta_group": 1, "threads": 512}'))
d512_q64_merge_ir = _decode_capture(_json_loads('{"__ir__": "knn_search_mma_split_merge_q128_const148_v1", "arg_keys": ["partial_distances", "partial_indices", "out_distances", "out_indices", "B", "Q", "K", "split_m", "num_q_tiles"], "cluster_dims": [1, 1, 1], "computed_smem_bytes": 0, "constants": [["K_MAX_", 10]], "cta_group": 1, "threads": 32}'))
d768d1024_ir = _decode_capture(_json_loads('{"__ir__": "knn_search_dynamic_d768d1024_q32q16_tcgen05_partial_0618_9286_v1", "arg_keys": ["queries", "database", "partial_distances", "partial_indices", "B", "Q", "M", "split_m", "num_q_tiles", "total_m_tiles", "tiles_per_split"], "cluster_dims": [1, 1, 1], "computed_smem_bytes": 143104, "constants": [["K_MAX_", 10], ["D_ORIG_", 1024], ["NUM_D_PASSES_", 8], ["Q_NORM_PARTS_", 64]], "cta_group": 1, "threads": 640}'))
d768d1024_merge_ir = _decode_capture(_json_loads('{"__ir__": "knn_search_mma_split_merge_q128_const148_v1", "arg_keys": ["partial_distances", "partial_indices", "out_distances", "out_indices", "B", "Q", "K", "split_m", "num_q_tiles"], "cluster_dims": [1, 1, 1], "computed_smem_bytes": 0, "constants": [["K_MAX_", 10]], "cta_group": 1, "threads": 32}'))
scalar_capacity_ir = _decode_capture(_json_loads('{"__ir__": "knn_search_scalar_capacity_partial_v1", "arg_keys": ["queries", "database", "partial_distances", "partial_indices", "B", "Q", "M", "num_m_tiles"], "cluster_dims": [1, 1, 1], "computed_smem_bytes": 0, "constants": [["D_", 8], ["K_CAP_", 64], ["BLOCK_M_", 512], ["NUM_WARPS_", 8]], "cta_group": 1, "threads": 256}'))
ir = _decode_capture(_json_loads('{"__ir__": "knn_search_dynamic_d_tiny_q128_m65536_tcgen05_partial_0618_c8b9_v1", "arg_keys": ["queries", "database", "partial_distances", "partial_indices", "B", "Q", "M", "split_m", "num_q_tiles", "total_m_tiles", "tiles_per_split"], "cluster_dims": [1, 1, 1], "computed_smem_bytes": 112384, "constants": [["K_MAX_", 10], ["D_TOTAL_", 3], ["NUM_D_PASSES_", 1], ["Q_NORM_PARTS_", 1]], "cta_group": 1, "threads": 640}'))
ROUTE_TINY_Q128 = '4832_dynamic_tiny_q128_d3_d63'
ROUTE_EXT_HIGHD = ext_highd.ROUTE_EXT_DYNAMIC_HIGHD_9286
ROUTE_HIGH_Q128 = highd_q128.ROUTE_HIGH_DYNAMIC_D_TCGEN05
ROUTE_SELF_Q2048_D3 = remaining.parent.ROUTE_SELF_Q2048_D3
ROUTE_D512_Q64 = d512_q64.ROUTE_D512_Q64_TCGEN05
ROUTE_D768D1024 = d768d1024.ROUTE_HIGH_DYNAMIC_D_TCGEN05
ROUTE_SCALAR_CAPACITY = 'scalar_capacity_parent'
CONSUMED_TINY_Q128_SEED = 'weave-evolve-knn-search-round-2-c8b9-nopack'
CONSUMED_EXT_HIGHD_SEED = ext_highd.CONSUMED_EXT_HIGHD_SEED
CONSUMED_HIGH_Q128_SEED = highd_q128.CONSUMED_SEED
CONSUMED_SELF_Q2048_D3_SEED = 'weave-evolve-knn-search-ccef-self-q2048-d3-guard-extension'
CONSUMED_D512_Q64_SEED = d512_q64.CONSUMED_SEED
CONSUMED_D768D1024_SEED = d768d1024.CONSUMED_SEED
CONSUMED_SEEDS: tuple[str, ...] = (CONSUMED_TINY_Q128_SEED, CONSUMED_EXT_HIGHD_SEED, CONSUMED_HIGH_Q128_SEED, CONSUMED_SELF_Q2048_D3_SEED, CONSUMED_D512_Q64_SEED, CONSUMED_D768D1024_SEED)
TINY_Q128_LABELS: tuple[str, ...] = ('blind_dyn_d3_q128_m65536_k10', 'blind_dyn_d63_q128_m65536_k10')
EXT_HIGHD_LABELS: tuple[str, ...] = ('blind_ext_dyn_d31_q128_m65536_k10', 'blind_ext_dyn_d65_q128_m65536_k10', 'blind_ext_dyn_d127_q128_m65536_k10', 'blind_ext_dyn_d255_q128_m65536_k10')
HIGH_Q128_LABELS: tuple[str, ...] = ('blind_dyn_d129_q128_m65536_k10', 'blind_dyn_d257_q128_m65536_k10')
SELF_Q2048_D3_LABELS: tuple[str, ...] = ('blind_dyn_self_q2048_m2048_d3_k10',)
D512_Q64_LABELS: tuple[str, ...] = d512_q64.TARGET_LABELS
D768D1024_LABELS: tuple[str, ...] = d768d1024.HIGH_DYNAMIC_D_LABELS
PRIORITY_DYNAMIC_D_LABELS: tuple[str, ...] = (*SELF_Q2048_D3_LABELS, *TINY_Q128_LABELS, *EXT_HIGHD_LABELS, *HIGH_Q128_LABELS, *D512_Q64_LABELS, *D768D1024_LABELS)
PRIORITY_DYNAMIC_D_SHAPES = _decode_capture(_json_loads('[{"__dict_items__": [["label", "blind_dyn_self_q2048_m2048_d3_k10"], ["params", {"__dict_items__": [["B", 1], ["Q", 2048], ["M", 2048], ["D", 3], ["K", 10], ["dtype", "bfloat16"], ["seed", 610810], ["self_search", true], ["min_recall", 0.999]]}]]}, {"__dict_items__": [["label", "blind_dyn_d3_q128_m65536_k10"], ["params", {"__dict_items__": [["B", 1], ["Q", 128], ["M", 65536], ["D", 3], ["K", 10], ["dtype", "bfloat16"], ["seed", 610801], ["self_search", false], ["min_recall", 0.999]]}]]}, {"__dict_items__": [["label", "blind_dyn_d63_q128_m65536_k10"], ["params", {"__dict_items__": [["B", 1], ["Q", 128], ["M", 65536], ["D", 63], ["K", 10], ["dtype", "bfloat16"], ["seed", 610803], ["self_search", false], ["min_recall", 0.999]]}]]}, {"__dict_items__": [["label", "blind_ext_dyn_d31_q128_m65536_k10"], ["params", {"__dict_items__": [["B", 1], ["Q", 128], ["M", 65536], ["D", 31], ["K", 10], ["dtype", "bfloat16"], ["seed", 610916], ["self_search", false], ["min_recall", 0.999]]}]]}, {"__dict_items__": [["label", "blind_ext_dyn_d65_q128_m65536_k10"], ["params", {"__dict_items__": [["B", 1], ["Q", 128], ["M", 65536], ["D", 65], ["K", 10], ["dtype", "bfloat16"], ["seed", 610917], ["self_search", false], ["min_recall", 0.999]]}]]}, {"__dict_items__": [["label", "blind_ext_dyn_d127_q128_m65536_k10"], ["params", {"__dict_items__": [["B", 1], ["Q", 128], ["M", 65536], ["D", 127], ["K", 10], ["dtype", "bfloat16"], ["seed", 610918], ["self_search", false], ["min_recall", 0.999]]}]]}, {"__dict_items__": [["label", "blind_ext_dyn_d255_q128_m65536_k10"], ["params", {"__dict_items__": [["B", 1], ["Q", 128], ["M", 65536], ["D", 255], ["K", 10], ["dtype", "bfloat16"], ["seed", 610920], ["self_search", false], ["min_recall", 0.999]]}]]}, {"__dict_items__": [["label", "blind_dyn_d129_q128_m65536_k10"], ["params", {"__dict_items__": [["B", 1], ["Q", 128], ["M", 65536], ["D", 129], ["K", 10], ["dtype", "bfloat16"], ["seed", 610804], ["self_search", false], ["min_recall", 0.999]]}]]}, {"__dict_items__": [["label", "blind_dyn_d257_q128_m65536_k10"], ["params", {"__dict_items__": [["B", 1], ["Q", 128], ["M", 65536], ["D", 257], ["K", 10], ["dtype", "bfloat16"], ["seed", 610805], ["self_search", false], ["min_recall", 0.999]]}]]}, {"__dict_items__": [["label", "blind_ext_dyn_d512_q64_m65536_k10"], ["params", {"__dict_items__": [["B", 1], ["Q", 64], ["M", 65536], ["D", 512], ["K", 10], ["dtype", "bfloat16"], ["seed", 610922], ["self_search", false], ["min_recall", 0.999]]}]]}, {"__dict_items__": [["label", "blind_ext_dyn_d768_q32_m32768_k10"], ["params", {"__dict_items__": [["B", 1], ["Q", 32], ["M", 32768], ["D", 768], ["K", 10], ["dtype", "bfloat16"], ["seed", 610923], ["self_search", false], ["min_recall", 0.999]]}]]}, {"__dict_items__": [["label", "blind_ext_dyn_d1024_q16_m32768_k10"], ["params", {"__dict_items__": [["B", 1], ["Q", 16], ["M", 32768], ["D", 1024], ["K", 10], ["dtype", "bfloat16"], ["seed", 610924], ["self_search", false], ["min_recall", 0.999]]}]]}]'))
_TINY_Q128_ENTRY: dict[str, Any] = {'shape_key': '4832_dynamic_d3d63_q128_m65536_k10', 'labels': TINY_Q128_LABELS, 'guard': 'B == 1 and Q == 128 and M == 65536 and K == 10 and not self_search and not forced_fallback and (D == 3 or (D == 63 and tcgen05_capable_arch))', 'route': ROUTE_TINY_Q128, 'entrypoint': 'loom.examples.weave.knn_search_dynamic_d_priority_4832_v1:launch_for_eval', 'selected_seed': CONSUMED_TINY_Q128_SEED, 'source_round_doc': 'design_doc/active/weave_evolve_knn_search_round_2_c8b9_nopack.md', 'coverage_class': 'bucket_seed_dynamic_d_tiny_q128_m65536_k10', 'route_source': 'bucket-specific-seed'}
_EXT_HIGHD_ENTRY: dict[str, Any] = {'shape_key': '4832_ext_dynamic_d31_d65_d127_d255_q128_m65536_k10', 'labels': EXT_HIGHD_LABELS, 'guard': 'B == 1 and Q == 128 and M == 65536 and D in {31,65,127,255} and K == 10 and not self_search and not forced_fallback and arch in {sm_100a,sm_103a}', 'route': ROUTE_EXT_HIGHD, 'entrypoint': 'loom.examples.weave.knn_search_dispatch0618_c492_ext_dynamic_highd_9286_v1:launch_for_eval', 'selected_seed': CONSUMED_EXT_HIGHD_SEED, 'source_round_doc': 'design_doc/active/weave_evolve_knn_search_round_3_ccef_highd_directstride.md', 'coverage_class': 'generated_variant_ext_dynamic_d_highd_q128_k10', 'route_source': 'generated-variant'}
_HIGH_Q128_ENTRY: dict[str, Any] = {'shape_key': '4832_dynamic_d129_d257_q128_m65536_k10', 'labels': HIGH_Q128_LABELS, 'guard': 'B == 1 and Q == 128 and M == 65536 and D in {129,257} and K == 10 and not self_search and not forced_fallback and arch in {sm_100a,sm_103a}', 'route': ROUTE_HIGH_Q128, 'entrypoint': 'loom.examples.weave.knn_search_dynamic_d_high_q128_directstride_tcgen05_0618_ccef_v1:launch_for_eval', 'selected_seed': CONSUMED_HIGH_Q128_SEED, 'source_round_doc': 'design_doc/active/weave_evolve_knn_search_round_3_ccef_highd_directstride.md', 'coverage_class': 'bucket_seed_dynamic_d_high_q128_m65536_k10', 'route_source': 'bucket-specific-seed'}
_SELF_Q2048_D3_ENTRY: dict[str, Any] = {'shape_key': '4832_dynamic_self_q2048_m2048_d3_k10', 'labels': SELF_Q2048_D3_LABELS, 'guard': 'B == 1 and Q == 2048 and M == 2048 and D == 3 and K == 10 and self_search and not forced_fallback', 'route': ROUTE_SELF_Q2048_D3, 'entrypoint': 'loom.examples.weave.knn_search_dynamic_d_remaining_seeds_0618_ccef_v2:launch_for_eval', 'selected_seed': CONSUMED_SELF_Q2048_D3_SEED, 'source_round_doc': 'design_doc/active/weave_evolve_knn_search_round_3_ccef_remaining_dynamic_d.md', 'coverage_class': 'bucket_seed_dynamic_d_lowd_self_q2048', 'route_source': 'bucket-specific-seed'}
_D512_Q64_ENTRY: dict[str, Any] = {'shape_key': '4832_dynamic_d512_q64_m65536_k10', 'labels': D512_Q64_LABELS, 'guard': 'B == 1 and Q == 64 and M == 65536 and D == 512 and K == 10 and not self_search and not forced_fallback and arch in {sm_100a,sm_103a}', 'route': ROUTE_D512_Q64, 'entrypoint': 'loom.examples.weave.knn_search_dynamic_d512_q64_tcgen05_0618_9286_v1:launch_for_eval', 'selected_seed': CONSUMED_D512_Q64_SEED, 'source_round_doc': 'design_doc/active/generalize_auto_tuning_knn_search_round_112_c2e0.md', 'coverage_class': 'bucket_seed_dynamic_d512_q64_m65536_k10', 'route_source': 'bucket-specific-seed'}
_D768D1024_ENTRY: dict[str, Any] = {'shape_key': '4832_dynamic_d768d1024_q32q16_m32768_k10', 'labels': D768D1024_LABELS, 'guard': 'B == 1 and M == 32768 and (Q,D) in {(32,768),(16,1024)} and K == 10 and not self_search and not forced_fallback and arch in {sm_100a,sm_103a}', 'route': ROUTE_D768D1024, 'entrypoint': 'loom.examples.weave.knn_search_dynamic_d768d1024_q32q16_tcgen05_0618_9286_v1:launch_for_eval', 'selected_seed': CONSUMED_D768D1024_SEED, 'source_round_doc': 'design_doc/active/generalize_auto_tuning_knn_search_round_113_c1e2.md', 'coverage_class': 'bucket_seed_ext_dynamic_d768d1024_q32q16_m32768_k10', 'route_source': 'bucket-specific-seed'}
_ALL_ENTRIES: tuple[dict[str, Any], ...] = (_SELF_Q2048_D3_ENTRY, _TINY_Q128_ENTRY, _EXT_HIGHD_ENTRY, _HIGH_Q128_ENTRY, _D512_Q64_ENTRY, _D768D1024_ENTRY)
SHAPE_DISPATCH_REGISTRY: tuple[dict[str, Any], ...] = _ALL_ENTRIES

def _shape_key(inputs: dict[str, Any]) -> tuple[int, int, int, int, int, bool]:
    return (int(inputs.get('B', 1)), int(inputs['Q']), int(inputs['M']), int(inputs['D']), int(inputs['K']), bool(inputs.get('self_search', False)))

def _forced_fallback(inputs: dict[str, Any]) -> bool:
    return bool(inputs.get('force_fallback', False))

def _use_tiny_q128(inputs: dict[str, Any]) -> bool:
    if _forced_fallback(inputs):
        return False
    if _shape_key(inputs) == (1, 128, 65536, 3, K_MAX, False):
        return True
    if _shape_key(inputs) == (1, 128, 65536, 63, K_MAX, False):
        return tiny_q128._use_tiny_dynamic_d_mma(inputs)
    return False

def _use_high_q128(inputs: dict[str, Any]) -> bool:
    if _forced_fallback(inputs):
        return False
    return _shape_key(inputs) in {(1, 128, 65536, 129, K_MAX, False), (1, 128, 65536, 257, K_MAX, False)} and highd_q128._use_high_dynamic_d_tcgen05(inputs)

def _entry_for_inputs(inputs: dict[str, Any]) -> dict[str, Any] | None:
    if remaining.parent._is_self_q2048_d3(inputs):
        return _SELF_Q2048_D3_ENTRY
    if _use_tiny_q128(inputs):
        return _TINY_Q128_ENTRY
    if ext_highd._use_ext_dynamic_highd(inputs) and int(inputs['D']) in {31, 65, 127, 255}:
        return _EXT_HIGHD_ENTRY
    if _use_high_q128(inputs):
        return _HIGH_Q128_ENTRY
    if d512_q64._use_d512_q64_tcgen05(inputs):
        return _D512_Q64_ENTRY
    if d768d1024._use_high_dynamic_d_tcgen05(inputs):
        return _D768D1024_ENTRY
    return None

def _guard_order() -> list[str]:
    return [str(entry['shape_key']) for entry in _ALL_ENTRIES]

def _route_for_entry(inputs: dict[str, Any], entry: dict[str, Any]) -> str:
    if entry is _TINY_Q128_ENTRY:
        return tiny_q128.selected_route(inputs)
    return str(entry['route'])

def selected_route(inputs: dict[str, Any]) -> str:
    entry = _entry_for_inputs(inputs)
    if entry is None:
        return ROUTE_SCALAR_CAPACITY
    return _route_for_entry(inputs, entry)

def selected_route_name(inputs: dict[str, Any]) -> str:
    return selected_route(inputs)

def _seed_info(inputs: dict[str, Any], entry: dict[str, Any]) -> dict[str, Any]:
    route = _route_for_entry(inputs, entry)
    return {'route': route, 'selected_route': route, 'selected_entrypoint': entry['entrypoint'], 'route_kind': 'specialized', 'route_source': entry['route_source'], 'coverage_class': entry['coverage_class'], 'classification': 'seed-produced', 'coverage_only': False, 'production_policy': 'weave_only', 'external_fallback': None, 'guard_order': _guard_order(), 'guard_id': entry['shape_key'], 'forced_fallback': False, 'selected_guard': entry['guard'], 'fallback': ROUTE_SCALAR_CAPACITY, 'missing_weave_route': False, 'selected_seed': entry['selected_seed'], 'selected_seed_task': entry['selected_seed'], 'expected_seed': entry['selected_seed'], 'source_round_doc': entry['source_round_doc']}

def route_info(inputs: dict[str, Any]) -> dict[str, Any]:
    entry = _entry_for_inputs(inputs)
    if entry is not None:
        return _seed_info(inputs, entry)
    return {'route': ROUTE_SCALAR_CAPACITY, 'selected_route': ROUTE_SCALAR_CAPACITY, 'selected_entrypoint': 'loom.examples.weave.knn_search_scalar_capacity_0611_r22_4e96_v1:launch_scalar_capacity_for_eval', 'route_kind': 'general', 'route_source': 'generic-weave-fallback', 'coverage_only': False, 'production_policy': 'weave_only', 'external_fallback': None, 'guard_order': _guard_order(), 'forced_fallback': _forced_fallback(inputs), 'missing_weave_route': False, 'selected_seed': 'scalar_capacity_dynamic_d'}

def route_trace_entry(label: str, inputs: dict[str, Any]) -> dict[str, Any]:
    return {'label': label, 'shape_key': label, 'B': int(inputs['B']), 'Q': int(inputs['Q']), 'M': int(inputs['M']), 'D': int(inputs['D']), 'K': int(inputs['K']), 'self_search': bool(inputs.get('self_search', False)), **route_info(inputs)}

def launch_for_eval(inputs: dict[str, Any]) -> dict[str, Any]:
    entry = _entry_for_inputs(inputs)
    if entry is _SELF_Q2048_D3_ENTRY:
        return remaining.launch_for_eval(inputs)
    if entry is _TINY_Q128_ENTRY:
        return tiny_q128.launch_for_eval(inputs)
    if entry is _EXT_HIGHD_ENTRY:
        return ext_highd.launch_for_eval(inputs)
    if entry is _HIGH_Q128_ENTRY:
        return highd_q128.launch_for_eval(inputs)
    if entry is _D512_Q64_ENTRY:
        return d512_q64.launch_for_eval(inputs)
    if entry is _D768D1024_ENTRY:
        return d768d1024.launch_for_eval(inputs)
    return scalar_capacity.launch_scalar_capacity_for_eval(inputs)

def _select_contract_shapes(shape_labels: str | tuple[str, ...] | list[str] | None):
    if shape_labels is None:
        return PRIORITY_DYNAMIC_D_SHAPES
    return select_named_shapes(shape_labels)

def knn_search_compile_and_launch_dynamic_d_priority_4832(*, benchmark: bool=True, shape_labels: str | tuple[str, ...] | list[str] | None=PRIORITY_DYNAMIC_D_LABELS) -> dict[str, Any]:
    from .._dispatch_runtime import evaluate
    result = evaluate(launch_for_eval, shapes=_select_contract_shapes(shape_labels), benchmark=benchmark)
    result['passed'] = bool(result.get('summary', {}).get('all_correct'))
    print(result)
    return result
