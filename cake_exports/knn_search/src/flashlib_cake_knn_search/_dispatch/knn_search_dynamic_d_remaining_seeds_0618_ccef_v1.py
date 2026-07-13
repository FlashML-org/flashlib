"""Dynamic-D blind-spot seed aggregator for the ccef continuation.

Minimum target architecture: sm_100a for inherited tcgen05/TMEM routes. This
additive candidate does not edit the production dispatcher. It exports guarded
Weave-only seed routes for remaining 0618 dynamic-D rows by reusing the
measured high-D padded tcgen05 producer and tiny-D direct tile reducer.
"""
from __future__ import annotations
from json import loads as _json_loads
from .._dispatch_runtime import _capture_cuTensorMapEncodeTiled, _decode_capture, _import_dispatch_module, _ir_proxy
from typing import Any
from .._dispatch_runtime import select_named_shapes
from . import knn_search_dynamic_d_high_q128_tcgen05_0618_c8b9_v1 as highd
from . import knn_search_dynamic_d_tiny_q128_tcgen05_0618_c8b9_v1 as tinyd
from . import knn_search_scalar_capacity_0611_r22_4e96_v1 as scalar_capacity
THREADS = highd.THREADS
BLOCK_Q = highd.BLOCK_Q
BLOCK_M = highd.BLOCK_M
K_MAX = highd.K_MAX
MERGE_THREADS = highd.MERGE_THREADS
HIGH_DYNAMIC_D_SPLIT_M = highd.HIGH_DYNAMIC_D_SPLIT_M
ir = _decode_capture(_json_loads('{"__ir__": "knn_search_dynamic_d511_tcgen05_partial_0618_c8b9_v1", "arg_keys": ["queries", "database", "partial_distances", "partial_indices", "B", "Q", "M", "split_m", "num_q_tiles", "total_m_tiles", "tiles_per_split"], "cluster_dims": [1, 1, 1], "computed_smem_bytes": 126720, "constants": [["K_MAX_", 10], ["D_TOTAL_", 512], ["NUM_D_PASSES_", 4], ["Q_NORM_PARTS_", 32]], "cta_group": 1, "threads": 640}'))
partial_ir = _decode_capture(_json_loads('{"__ir__": "knn_search_dynamic_d511_tcgen05_partial_0618_c8b9_v1", "arg_keys": ["queries", "database", "partial_distances", "partial_indices", "B", "Q", "M", "split_m", "num_q_tiles", "total_m_tiles", "tiles_per_split"], "cluster_dims": [1, 1, 1], "computed_smem_bytes": 126720, "constants": [["K_MAX_", 10], ["D_TOTAL_", 512], ["NUM_D_PASSES_", 4], ["Q_NORM_PARTS_", 32]], "cta_group": 1, "threads": 640}'))
merge_ir = _decode_capture(_json_loads('{"__ir__": "knn_search_mma_split_merge_q128_const148_v1", "arg_keys": ["partial_distances", "partial_indices", "out_distances", "out_indices", "B", "Q", "K", "split_m", "num_q_tiles"], "cluster_dims": [1, 1, 1], "computed_smem_bytes": 0, "constants": [["K_MAX_", 10]], "cta_group": 1, "threads": 32}'))
pack_ir = _decode_capture(_json_loads('{"__ir__": "knn_search_dynamic_d_tiny_pack_bf16_0618_c8b9_v1", "arg_keys": ["queries", "database", "padded_queries", "padded_database", "B", "Q", "M"], "cluster_dims": [1, 1, 1], "computed_smem_bytes": 0, "constants": [["D_ORIG_", 3], ["D_PAD_", 16]], "cta_group": 1, "threads": 256}'))
d3_tile_ir = _decode_capture(_json_loads('{"__ir__": "knn_search_dynamic_d3_tile_reduce_partial_0618_c8b9_v1", "arg_keys": ["queries", "database", "partial_distances", "partial_indices", "B", "Q", "M", "K", "num_m_tiles"], "cluster_dims": [1, 1, 1], "computed_smem_bytes": 10240, "constants": [["K_MAX_", 10]], "cta_group": 1, "threads": 256}'))
scalar_capacity_ir = _decode_capture(_json_loads('{"__ir__": "knn_search_scalar_capacity_partial_v1", "arg_keys": ["queries", "database", "partial_distances", "partial_indices", "B", "Q", "M", "num_m_tiles"], "cluster_dims": [1, 1, 1], "computed_smem_bytes": 0, "constants": [["D_", 8], ["K_CAP_", 64], ["BLOCK_M_", 512], ["NUM_WARPS_", 8]], "cta_group": 1, "threads": 256}'))
ROUTE_HIGH_Q128 = highd.ROUTE_HIGH_DYNAMIC_D_TCGEN05
ROUTE_B2_Q64_D129 = 'ccef_dynamic_d_b2_q64_d129_tcgen05'
ROUTE_D384_Q32 = 'ccef_dynamic_d_d384_q32_tcgen05'
ROUTE_SELF_Q2048_D3 = 'ccef_dynamic_d_self_q2048_d3_tile_reduce'
ROUTE_SCALAR_CAPACITY = 'scalar_capacity_parent'
CONSUMED_HIGH_D_SEED = 'weave-evolve-knn-search-round-2-c8b9-highd'
CONSUMED_TINY_D_SEED = 'weave-evolve-knn-search-round-2-c8b9-tinyd'
CONSUMED_SEEDS: tuple[str, ...] = (CONSUMED_HIGH_D_SEED, CONSUMED_TINY_D_SEED)
HIGH_Q128_LABELS: tuple[str, ...] = highd.HIGH_DYNAMIC_D_LABELS
B2_Q64_D129_LABELS: tuple[str, ...] = ('blind_dyn_b2_q64_m65536_d129_k10',)
D384_Q32_LABELS: tuple[str, ...] = ('blind_dyn_d384_q32_m131072_k10',)
SELF_Q2048_D3_LABELS: tuple[str, ...] = ('blind_dyn_self_q2048_m2048_d3_k10',)
REMAINING_DYNAMIC_D_SEED_LABELS: tuple[str, ...] = (*HIGH_Q128_LABELS, *D384_Q32_LABELS, *B2_Q64_D129_LABELS, *SELF_Q2048_D3_LABELS)
REMAINING_DYNAMIC_D_SEED_SHAPES = _decode_capture(_json_loads('[{"__dict_items__": [["label", "blind_dyn_d129_q128_m65536_k10"], ["params", {"__dict_items__": [["B", 1], ["Q", 128], ["M", 65536], ["D", 129], ["K", 10], ["dtype", "bfloat16"], ["seed", 610804], ["self_search", false], ["min_recall", 0.999]]}]]}, {"__dict_items__": [["label", "blind_dyn_d257_q128_m65536_k10"], ["params", {"__dict_items__": [["B", 1], ["Q", 128], ["M", 65536], ["D", 257], ["K", 10], ["dtype", "bfloat16"], ["seed", 610805], ["self_search", false], ["min_recall", 0.999]]}]]}, {"__dict_items__": [["label", "blind_dyn_d511_q128_m65536_k10"], ["params", {"__dict_items__": [["B", 1], ["Q", 128], ["M", 65536], ["D", 511], ["K", 10], ["dtype", "bfloat16"], ["seed", 610806], ["self_search", false], ["min_recall", 0.999]]}]]}, {"__dict_items__": [["label", "blind_dyn_d384_q32_m131072_k10"], ["params", {"__dict_items__": [["B", 1], ["Q", 32], ["M", 131072], ["D", 384], ["K", 10], ["dtype", "bfloat16"], ["seed", 610807], ["self_search", false], ["min_recall", 0.999]]}]]}, {"__dict_items__": [["label", "blind_dyn_b2_q64_m65536_d129_k10"], ["params", {"__dict_items__": [["B", 2], ["Q", 64], ["M", 65536], ["D", 129], ["K", 10], ["dtype", "bfloat16"], ["seed", 610809], ["self_search", false], ["min_recall", 0.999]]}]]}, {"__dict_items__": [["label", "blind_dyn_self_q2048_m2048_d3_k10"], ["params", {"__dict_items__": [["B", 1], ["Q", 2048], ["M", 2048], ["D", 3], ["K", 10], ["dtype", "bfloat16"], ["seed", 610810], ["self_search", true], ["min_recall", 0.999]]}]]}]'))
_HIGH_Q128_ENTRY: dict[str, Any] = {'overlay': 'ccef_high_q128_dynamic_d', 'shape_key': 'B=1,Q=128,M=65536,D in {129,257,511},K=10', 'labels': HIGH_Q128_LABELS, 'guard': 'B == 1 and Q == 128 and M == 65536 and D in {129,257,511} and K == 10 and not self_search and not forced_fallback', 'route': ROUTE_HIGH_Q128, 'entrypoint': 'loom.examples.weave.knn_search_dynamic_d_high_q128_tcgen05_0618_c8b9_v1:launch_for_eval', 'selected_seed': CONSUMED_HIGH_D_SEED, 'source_round_doc': 'design_doc/active/weave_evolve_knn_search_round_2_c8b9_highd.md', 'coverage_class': 'bucket_seed_dynamic_d_high_q128_m65536_k10'}
_D384_Q32_ENTRY: dict[str, Any] = {'overlay': 'ccef_d384_q32_dynamic_d', 'shape_key': 'B=1,Q=32,M=131072,D=384,K=10', 'labels': D384_Q32_LABELS, 'guard': 'B == 1 and Q == 32 and M == 131072 and D == 384 and K == 10 and not self_search and not forced_fallback', 'route': ROUTE_D384_Q32, 'entrypoint': 'loom.examples.weave.knn_search_dynamic_d_remaining_seeds_0618_ccef_v1:launch_for_eval', 'selected_seed': 'weave-evolve-knn-search-ccef-d384-q32-guard-extension', 'source_round_doc': 'design_doc/active/weave_evolve_knn_search_round_3_ccef_remaining_dynamic_d.md', 'coverage_class': 'bucket_seed_dynamic_d_lowq_d384_m131072_k10'}
_B2_Q64_D129_ENTRY: dict[str, Any] = {'overlay': 'ccef_b2_q64_d129_dynamic_d', 'shape_key': 'B=2,Q=64,M=65536,D=129,K=10', 'labels': B2_Q64_D129_LABELS, 'guard': 'B == 2 and Q == 64 and M == 65536 and D == 129 and K == 10 and not self_search and not forced_fallback', 'route': ROUTE_B2_Q64_D129, 'entrypoint': 'loom.examples.weave.knn_search_dynamic_d_remaining_seeds_0618_ccef_v1:launch_for_eval', 'selected_seed': 'weave-evolve-knn-search-ccef-b2-q64-d129-guard-extension', 'source_round_doc': 'design_doc/active/weave_evolve_knn_search_round_3_ccef_remaining_dynamic_d.md', 'coverage_class': 'bucket_seed_dynamic_d_b2_q64_m65536_d129_k10'}
_SELF_Q2048_D3_ENTRY: dict[str, Any] = {'overlay': 'ccef_self_q2048_d3_dynamic_d', 'shape_key': 'B=1,Q=2048,M=2048,D=3,K=10,self_search=true', 'labels': SELF_Q2048_D3_LABELS, 'guard': 'B == 1 and Q == 2048 and M == 2048 and D == 3 and K == 10 and self_search and not forced_fallback', 'route': ROUTE_SELF_Q2048_D3, 'entrypoint': 'loom.examples.weave.knn_search_dynamic_d_remaining_seeds_0618_ccef_v1:launch_for_eval', 'selected_seed': 'weave-evolve-knn-search-ccef-self-q2048-d3-guard-extension', 'source_round_doc': 'design_doc/active/weave_evolve_knn_search_round_3_ccef_remaining_dynamic_d.md', 'coverage_class': 'bucket_seed_dynamic_d_lowd_self_q2048'}
SHAPE_DISPATCH_REGISTRY: tuple[dict[str, Any], ...] = (_HIGH_Q128_ENTRY, _D384_Q32_ENTRY, _B2_Q64_D129_ENTRY, _SELF_Q2048_D3_ENTRY)

def _shape_key(inputs: dict[str, Any]) -> tuple[int, int, int, int, int, bool]:
    return (int(inputs.get('B', 1)), int(inputs['Q']), int(inputs['M']), int(inputs['D']), int(inputs['K']), bool(inputs.get('self_search', False)))

def _forced_fallback(inputs: dict[str, Any]) -> bool:
    return bool(inputs.get('force_fallback', False))

def _tcgen05_capable() -> bool:
    return highd.mma._tcgen05_capable_arch()

def _is_d384_q32(inputs: dict[str, Any]) -> bool:
    return _shape_key(inputs) == (1, 32, 131072, 384, K_MAX, False) and (not _forced_fallback(inputs)) and _tcgen05_capable()

def _is_b2_q64_d129(inputs: dict[str, Any]) -> bool:
    return _shape_key(inputs) == (2, 64, 65536, 129, K_MAX, False) and (not _forced_fallback(inputs)) and _tcgen05_capable()

def _is_self_q2048_d3(inputs: dict[str, Any]) -> bool:
    return _shape_key(inputs) == (1, 2048, 2048, 3, K_MAX, True) and (not _forced_fallback(inputs))

def _active_entry(inputs: dict[str, Any]) -> dict[str, Any] | None:
    if highd._use_high_dynamic_d_tcgen05(inputs):
        return _HIGH_Q128_ENTRY
    if _is_d384_q32(inputs):
        return _D384_Q32_ENTRY
    if _is_b2_q64_d129(inputs):
        return _B2_Q64_D129_ENTRY
    if _is_self_q2048_d3(inputs):
        return _SELF_Q2048_D3_ENTRY
    return None

def selected_route(inputs: dict[str, Any]) -> str:
    entry = _active_entry(inputs)
    if entry is None:
        return ROUTE_SCALAR_CAPACITY
    return str(entry['route'])

def selected_route_name(inputs: dict[str, Any]) -> str:
    return selected_route(inputs)

def route_info(inputs: dict[str, Any]) -> dict[str, Any]:
    entry = _active_entry(inputs)
    if entry is None:
        return {'route': ROUTE_SCALAR_CAPACITY, 'selected_route': ROUTE_SCALAR_CAPACITY, 'selected_entrypoint': 'loom.examples.weave.knn_search_scalar_capacity_0611_r22_4e96_v1:launch_for_eval', 'route_kind': 'general', 'route_source': 'generic-weave-fallback', 'coverage_only': False, 'production_policy': 'weave_only', 'external_fallback': None, 'forced_fallback': _forced_fallback(inputs)}
    return {'route': entry['route'], 'selected_route': entry['route'], 'selected_entrypoint': entry['entrypoint'], 'route_kind': 'specialized', 'route_source': 'bucket-specific-seed', 'coverage_class': entry['coverage_class'], 'classification': 'seed-produced', 'coverage_only': False, 'production_policy': 'weave_only', 'external_fallback': None, 'guard_id': entry['shape_key'], 'selected_guard': entry['guard'], 'fallback': ROUTE_SCALAR_CAPACITY, 'missing_weave_route': False, 'forced_fallback': False, 'selected_seed': entry['selected_seed'], 'source_round_doc': entry['source_round_doc']}

def route_trace_entry(label: str, inputs: dict[str, Any]) -> dict[str, Any]:
    return {'label': label, 'shape_key': label, 'B': int(inputs['B']), 'Q': int(inputs['Q']), 'M': int(inputs['M']), 'D': int(inputs['D']), 'K': int(inputs['K']), 'self_search': bool(inputs.get('self_search', False)), **route_info(inputs)}

def launch_for_eval(inputs: dict[str, Any]) -> dict[str, Any]:
    entry = _active_entry(inputs)
    if entry is _HIGH_Q128_ENTRY:
        return highd.launch_for_eval(inputs)
    if entry is _D384_Q32_ENTRY or entry is _B2_Q64_D129_ENTRY:
        return highd._launch_high_dynamic_d_tcgen05(inputs)
    if entry is _SELF_Q2048_D3_ENTRY:
        return tinyd._launch_d3_tile_reduce(inputs)
    return scalar_capacity.launch_for_eval(inputs)

def knn_search_compile_and_launch_dynamic_d_remaining_seeds_0618_ccef(*, benchmark: bool=True, shape_labels: str | tuple[str, ...] | list[str] | None=REMAINING_DYNAMIC_D_SEED_LABELS) -> dict[str, Any]:
    from .._dispatch_runtime import evaluate
    shapes = None if shape_labels is None else select_named_shapes(shape_labels)
    result = evaluate(launch_for_eval, shapes=shapes, benchmark=benchmark)
    result['passed'] = bool(result.get('summary', {}).get('all_correct'))
    print(result)
    return result
