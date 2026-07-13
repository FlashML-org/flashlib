"""Residual dynamic-D seed bundle for exact BF16 squared-L2 kNN.

Minimum target architecture: sm_100a for the tcgen05/TMEM routes. This
additive bucket wrapper exposes exact Weave-only routes for residual dynamic-D
contract rows left on scalar-capacity fallback in the ffc4 handoff. It does not
edit the production dispatcher; guard misses remain on the Weave scalar
fallback.
"""
from __future__ import annotations
from json import loads as _json_loads
from .._dispatch_runtime import _capture_cuTensorMapEncodeTiled, _decode_capture, _import_dispatch_module, _ir_proxy
from typing import Any
from .._dispatch_runtime import select_named_shapes
from . import knn_search_dynamic_d_d384_q32_m131072_exact_tcgen05_0618_5847_v1 as d384_exact
from . import knn_search_dynamic_d_high_q128_directstride_tcgen05_0618_ccef_v1 as highd
from . import knn_search_dynamic_d_k64_q64_m65536_tcgen05_0618_ccef_v2 as k64_dynamic
from . import knn_search_dynamic_d_remaining_seeds_0618_ccef_v2 as remaining
from . import knn_search_dynamic_d_tiny_q128_m65536_tcgen05_0618_c8b9_v1 as tinyd_no_pack
from . import knn_search_scalar_capacity_0611_r22_4e96_v1 as scalar_capacity
THREADS = highd.THREADS
BLOCK_Q = highd.BLOCK_Q
BLOCK_M = highd.BLOCK_M
K_MAX = highd.K_MAX
MERGE_THREADS = highd.MERGE_THREADS
ir = _decode_capture(_json_loads('{"__ir__": "knn_search_dynamic_d_high_directstride_tcgen05_partial_0618_ccef_v1", "arg_keys": ["queries", "database", "partial_distances", "partial_indices", "B", "Q", "M", "split_m", "num_q_tiles", "total_m_tiles", "tiles_per_split"], "cluster_dims": [1, 1, 1], "computed_smem_bytes": 126720, "constants": [["K_MAX_", 10], ["D_ORIG_", 512], ["NUM_D_PASSES_", 4], ["Q_NORM_PARTS_", 32]], "cta_group": 1, "threads": 640}'))
highd_partial_ir = _decode_capture(_json_loads('{"__ir__": "knn_search_dynamic_d_high_directstride_tcgen05_partial_0618_ccef_v1", "arg_keys": ["queries", "database", "partial_distances", "partial_indices", "B", "Q", "M", "split_m", "num_q_tiles", "total_m_tiles", "tiles_per_split"], "cluster_dims": [1, 1, 1], "computed_smem_bytes": 126720, "constants": [["K_MAX_", 10], ["D_ORIG_", 512], ["NUM_D_PASSES_", 4], ["Q_NORM_PARTS_", 32]], "cta_group": 1, "threads": 640}'))
highd_merge_ir = _decode_capture(_json_loads('{"__ir__": "knn_search_mma_split_merge_q128_const148_v1", "arg_keys": ["partial_distances", "partial_indices", "out_distances", "out_indices", "B", "Q", "K", "split_m", "num_q_tiles"], "cluster_dims": [1, 1, 1], "computed_smem_bytes": 0, "constants": [["K_MAX_", 10]], "cta_group": 1, "threads": 32}'))
tinyd_partial_ir = _decode_capture(_json_loads('{"__ir__": "knn_search_dynamic_d_tiny_q128_m65536_tcgen05_partial_0618_c8b9_v1", "arg_keys": ["queries", "database", "partial_distances", "partial_indices", "B", "Q", "M", "split_m", "num_q_tiles", "total_m_tiles", "tiles_per_split"], "cluster_dims": [1, 1, 1], "computed_smem_bytes": 112384, "constants": [["K_MAX_", 10], ["D_TOTAL_", 3], ["NUM_D_PASSES_", 1], ["Q_NORM_PARTS_", 1]], "cta_group": 1, "threads": 640}'))
tinyd_merge_ir = _decode_capture(_json_loads('{"__ir__": "knn_search_mma_split_merge_q128_const148_v1", "arg_keys": ["partial_distances", "partial_indices", "out_distances", "out_indices", "B", "Q", "K", "split_m", "num_q_tiles"], "cluster_dims": [1, 1, 1], "computed_smem_bytes": 0, "constants": [["K_MAX_", 10]], "cta_group": 1, "threads": 32}'))
d384_partial_ir = _decode_capture(_json_loads('{"__ir__": "knn_search_dynamic_d_d384_q32_m131072_exact_tcgen05_partial_0618_5847_v1", "arg_keys": ["queries", "database", "partial_distances", "partial_indices", "B", "Q", "M", "split_m", "num_q_tiles", "total_m_tiles", "tiles_per_split"], "cluster_dims": [1, 1, 1], "computed_smem_bytes": 122624, "constants": [["K_MAX_", 10]], "cta_group": 1, "threads": 640}'))
d384_merge_ir = _decode_capture(_json_loads('{"__ir__": "knn_search_mma_split_merge_q128_const148_v1", "arg_keys": ["partial_distances", "partial_indices", "out_distances", "out_indices", "B", "Q", "K", "split_m", "num_q_tiles"], "cluster_dims": [1, 1, 1], "computed_smem_bytes": 0, "constants": [["K_MAX_", 10]], "cta_group": 1, "threads": 32}'))
k64_partial_ir = _decode_capture(_json_loads('{"__ir__": "knn_search_dynamic_d257_k64_q64_partial_0618_ccef_v1", "arg_keys": ["queries", "database", "partial_distances", "partial_indices", "B", "Q", "M", "split_m", "num_q_tiles", "total_m_tiles", "tiles_per_split", "partial_list_count"], "cluster_dims": [1, 1, 1], "computed_smem_bytes": 151296, "constants": [["K_MAX_", 64]], "cta_group": 1, "threads": 256}'))
k64_merge_ir = _decode_capture(_json_loads('{"__ir__": "knn_search_dynamic_d257_k64_q64_merge_0618_ccef_v1", "arg_keys": ["partial_distances", "partial_indices", "out_distances", "out_indices", "B", "Q", "K", "partial_list_count", "num_q_tiles"], "cluster_dims": [1, 1, 1], "computed_smem_bytes": 0, "constants": [["K_MAX_", 64]], "cta_group": 1, "threads": 32}'))
scalar_capacity_ir = _decode_capture(_json_loads('{"__ir__": "knn_search_scalar_capacity_partial_v1", "arg_keys": ["queries", "database", "partial_distances", "partial_indices", "B", "Q", "M", "num_m_tiles"], "cluster_dims": [1, 1, 1], "computed_smem_bytes": 0, "constants": [["D_", 8], ["K_CAP_", 64], ["BLOCK_M_", 512], ["NUM_WARPS_", 8]], "cta_group": 1, "threads": 256}'))
ROUTE_D7_Q128_TCGEN05 = 'ffc4_dynamic_d7_q128_m65536_no_pack_tcgen05'
ROUTE_HIGH_Q128_DIRECTSTRIDE = 'ffc4_dynamic_d129d257d511_q128_m65536_directstride_tcgen05'
ROUTE_D384_Q32_EXACT = d384_exact.ROUTE_D384_Q32_EXACT_TCGEN05
ROUTE_D257_Q64_K64 = k64_dynamic.ROUTE_DYNAMIC_D257_Q64_K64
ROUTE_B2_Q64_D129 = 'ffc4_dynamic_b2_q64_m65536_d129_tcgen05'
ROUTE_SCALAR_CAPACITY = 'scalar_capacity_parent'
CONSUMED_TINY_D_SEED = 'weave-evolve-knn-search-c8b9-tiny-no-pack'
CONSUMED_HIGH_D_SEED = highd.CONSUMED_SEED
CONSUMED_D384_SEED = 'weave-evolve-knn-search-5847-d384-q32-exact'
CONSUMED_K64_SEED = k64_dynamic.CONSUMED_SEED
CONSUMED_B2_SEED = 'weave-evolve-knn-search-ccef-b2-q64-d129-guard-extension'
CONSUMED_SEEDS: tuple[str, ...] = (CONSUMED_TINY_D_SEED, CONSUMED_HIGH_D_SEED, CONSUMED_D384_SEED, CONSUMED_K64_SEED, CONSUMED_B2_SEED)
TARGET_LABELS: tuple[str, ...] = ('blind_dyn_d7_q128_m65536_k10', 'blind_dyn_d129_q128_m65536_k10', 'blind_dyn_d257_q128_m65536_k10', 'blind_dyn_d511_q128_m65536_k10', 'blind_dyn_d384_q32_m131072_k10', 'blind_dyn_d257_k64_q64_m65536', 'blind_dyn_b2_q64_m65536_d129_k10')
TARGET_SHAPES = _decode_capture(_json_loads('[{"__dict_items__": [["label", "blind_dyn_d7_q128_m65536_k10"], ["params", {"__dict_items__": [["B", 1], ["Q", 128], ["M", 65536], ["D", 7], ["K", 10], ["dtype", "bfloat16"], ["seed", 610802], ["self_search", false], ["min_recall", 0.999]]}]]}, {"__dict_items__": [["label", "blind_dyn_d129_q128_m65536_k10"], ["params", {"__dict_items__": [["B", 1], ["Q", 128], ["M", 65536], ["D", 129], ["K", 10], ["dtype", "bfloat16"], ["seed", 610804], ["self_search", false], ["min_recall", 0.999]]}]]}, {"__dict_items__": [["label", "blind_dyn_d257_q128_m65536_k10"], ["params", {"__dict_items__": [["B", 1], ["Q", 128], ["M", 65536], ["D", 257], ["K", 10], ["dtype", "bfloat16"], ["seed", 610805], ["self_search", false], ["min_recall", 0.999]]}]]}, {"__dict_items__": [["label", "blind_dyn_d511_q128_m65536_k10"], ["params", {"__dict_items__": [["B", 1], ["Q", 128], ["M", 65536], ["D", 511], ["K", 10], ["dtype", "bfloat16"], ["seed", 610806], ["self_search", false], ["min_recall", 0.999]]}]]}, {"__dict_items__": [["label", "blind_dyn_d384_q32_m131072_k10"], ["params", {"__dict_items__": [["B", 1], ["Q", 32], ["M", 131072], ["D", 384], ["K", 10], ["dtype", "bfloat16"], ["seed", 610807], ["self_search", false], ["min_recall", 0.999]]}]]}, {"__dict_items__": [["label", "blind_dyn_d257_k64_q64_m65536"], ["params", {"__dict_items__": [["B", 1], ["Q", 64], ["M", 65536], ["D", 257], ["K", 64], ["dtype", "bfloat16"], ["seed", 610808], ["self_search", false], ["min_recall", 0.999]]}]]}, {"__dict_items__": [["label", "blind_dyn_b2_q64_m65536_d129_k10"], ["params", {"__dict_items__": [["B", 2], ["Q", 64], ["M", 65536], ["D", 129], ["K", 10], ["dtype", "bfloat16"], ["seed", 610809], ["self_search", false], ["min_recall", 0.999]]}]]}]'))
REQUESTED_LABEL_ALIASES: dict[str, str] = {'blind_ext_dyn_d129_q128_m65536_k10': 'blind_dyn_d129_q128_m65536_k10', 'blind_ext_dyn_d257_q128_m65536_k10': 'blind_dyn_d257_q128_m65536_k10'}
_D7_ENTRY: dict[str, Any] = {'shape_key': 'ffc4_dynamic_d7_q128_m65536_k10', 'labels': ('blind_dyn_d7_q128_m65536_k10',), 'guard': 'B == 1 and Q == 128 and M == 65536 and D == 7 and K == 10 and not self_search and not forced_fallback and arch in {sm_100a,sm_103a}', 'route': ROUTE_D7_Q128_TCGEN05, 'entrypoint': 'loom.examples.weave.knn_search_dynamic_d_residual_0621_r123_ffc4_v1:launch_for_eval', 'selected_seed': CONSUMED_TINY_D_SEED, 'source_round_doc': 'design_doc/active/weave_evolve_knn_search_round_123_ffc4_residual_dynamicd.md', 'coverage_class': 'bucket_seed_dynamic_d7_q128_m65536_k10'}
_HIGH_Q128_ENTRY: dict[str, Any] = {'shape_key': 'ffc4_dynamic_d129d257d511_q128_m65536_k10', 'labels': ('blind_dyn_d129_q128_m65536_k10', 'blind_dyn_d257_q128_m65536_k10', 'blind_dyn_d511_q128_m65536_k10'), 'guard': 'B == 1 and Q == 128 and M == 65536 and D in {129,257,511} and K == 10 and not self_search and not forced_fallback and arch in {sm_100a,sm_103a}', 'route': ROUTE_HIGH_Q128_DIRECTSTRIDE, 'entrypoint': 'loom.examples.weave.knn_search_dynamic_d_residual_0621_r123_ffc4_v1:launch_for_eval', 'selected_seed': CONSUMED_HIGH_D_SEED, 'source_round_doc': 'design_doc/active/weave_evolve_knn_search_round_123_ffc4_residual_dynamicd.md', 'coverage_class': 'bucket_seed_dynamic_d129d257d511_q128_m65536_k10'}
_D384_ENTRY: dict[str, Any] = {'shape_key': 'ffc4_dynamic_d384_q32_m131072_k10', 'labels': ('blind_dyn_d384_q32_m131072_k10',), 'guard': 'B == 1 and Q == 32 and M == 131072 and D == 384 and K == 10 and not self_search and not forced_fallback and arch in {sm_100a,sm_103a}', 'route': ROUTE_D384_Q32_EXACT, 'entrypoint': 'loom.examples.weave.knn_search_dynamic_d_residual_0621_r123_ffc4_v1:launch_for_eval', 'selected_seed': CONSUMED_D384_SEED, 'source_round_doc': 'design_doc/active/weave_evolve_knn_search_round_123_ffc4_residual_dynamicd.md', 'coverage_class': 'bucket_seed_dynamic_d384_q32_m131072_k10'}
_K64_ENTRY: dict[str, Any] = {'shape_key': 'ffc4_dynamic_d257_q64_m65536_k64', 'labels': ('blind_dyn_d257_k64_q64_m65536',), 'guard': 'B == 1 and Q == 64 and M == 65536 and D == 257 and K == 64 and not self_search and not forced_fallback and arch in {sm_100a,sm_103a}', 'route': ROUTE_D257_Q64_K64, 'entrypoint': 'loom.examples.weave.knn_search_dynamic_d_residual_0621_r123_ffc4_v1:launch_for_eval', 'selected_seed': CONSUMED_K64_SEED, 'source_round_doc': 'design_doc/active/weave_evolve_knn_search_round_123_ffc4_residual_dynamicd.md', 'coverage_class': 'bucket_seed_dynamic_d257_q64_m65536_k64'}
_B2_ENTRY: dict[str, Any] = {'shape_key': 'ffc4_dynamic_b2_q64_m65536_d129_k10', 'labels': ('blind_dyn_b2_q64_m65536_d129_k10',), 'guard': 'B == 2 and Q == 64 and M == 65536 and D == 129 and K == 10 and not self_search and not forced_fallback and arch in {sm_100a,sm_103a}', 'route': ROUTE_B2_Q64_D129, 'entrypoint': 'loom.examples.weave.knn_search_dynamic_d_residual_0621_r123_ffc4_v1:launch_for_eval', 'selected_seed': CONSUMED_B2_SEED, 'source_round_doc': 'design_doc/active/weave_evolve_knn_search_round_123_ffc4_residual_dynamicd.md', 'coverage_class': 'bucket_seed_dynamic_b2_q64_m65536_d129_k10'}
SHAPE_DISPATCH_REGISTRY: tuple[dict[str, Any], ...] = (_D7_ENTRY, _HIGH_Q128_ENTRY, _D384_ENTRY, _K64_ENTRY, _B2_ENTRY)

def _shape_key(inputs: dict[str, Any]) -> tuple[int, int, int, int, int, bool]:
    return (int(inputs.get('B', 1)), int(inputs['Q']), int(inputs['M']), int(inputs['D']), int(inputs['K']), bool(inputs.get('self_search', False)))

def _forced_fallback(inputs: dict[str, Any]) -> bool:
    return bool(inputs.get('force_fallback', False))

def _tcgen05_capable() -> bool:
    return highd.mma._tcgen05_capable_arch()

def _use_d7_q128(inputs: dict[str, Any]) -> bool:
    return _shape_key(inputs) == (1, 128, 65536, 7, 10, False) and (not _forced_fallback(inputs)) and _tcgen05_capable()

def _use_high_q128(inputs: dict[str, Any]) -> bool:
    return highd._use_high_dynamic_d_tcgen05(inputs)

def _use_d384_q32(inputs: dict[str, Any]) -> bool:
    return d384_exact._use_d384_q32_exact_tcgen05(inputs)

def _use_k64_q64(inputs: dict[str, Any]) -> bool:
    return k64_dynamic._use_dynamic_d257_q64_k64(inputs)

def _use_b2_q64_d129(inputs: dict[str, Any]) -> bool:
    return _shape_key(inputs) == (2, 64, 65536, 129, 10, False) and (not _forced_fallback(inputs)) and _tcgen05_capable()

def _active_entry(inputs: dict[str, Any]) -> dict[str, Any] | None:
    if _use_d7_q128(inputs):
        return _D7_ENTRY
    if _use_high_q128(inputs):
        return _HIGH_Q128_ENTRY
    if _use_d384_q32(inputs):
        return _D384_ENTRY
    if _use_k64_q64(inputs):
        return _K64_ENTRY
    if _use_b2_q64_d129(inputs):
        return _B2_ENTRY
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
        return {'route': ROUTE_SCALAR_CAPACITY, 'selected_route': ROUTE_SCALAR_CAPACITY, 'selected_entrypoint': 'loom.examples.weave.knn_search_scalar_capacity_0611_r22_4e96_v1:launch_for_eval', 'route_kind': 'general', 'route_source': 'generic-weave-fallback', 'coverage_only': False, 'production_policy': 'weave_only', 'external_fallback': None, 'forced_fallback': _forced_fallback(inputs), 'missing_weave_route': False}
    return {'route': entry['route'], 'selected_route': entry['route'], 'selected_entrypoint': entry['entrypoint'], 'route_kind': 'specialized', 'route_source': 'bucket-specific-seed', 'coverage_class': entry['coverage_class'], 'classification': 'seed-produced', 'coverage_only': False, 'production_policy': 'weave_only', 'external_fallback': None, 'guard_id': entry['shape_key'], 'selected_guard': entry['guard'], 'fallback': ROUTE_SCALAR_CAPACITY, 'missing_weave_route': False, 'forced_fallback': False, 'selected_seed': entry['selected_seed'], 'source_round_doc': entry['source_round_doc']}

def route_trace_entry(label: str, inputs: dict[str, Any]) -> dict[str, Any]:
    return {'label': label, 'shape_key': label, 'B': int(inputs['B']), 'Q': int(inputs['Q']), 'M': int(inputs['M']), 'D': int(inputs['D']), 'K': int(inputs['K']), 'self_search': bool(inputs.get('self_search', False)), **route_info(inputs)}

def launch_for_eval(inputs: dict[str, Any]) -> dict[str, Any]:
    if _use_d7_q128(inputs):
        return tinyd_no_pack.launch_for_eval(inputs)
    if _use_high_q128(inputs):
        return highd.launch_for_eval(inputs)
    if _use_d384_q32(inputs):
        return d384_exact.launch_for_eval(inputs)
    if _use_k64_q64(inputs):
        return k64_dynamic.launch_for_eval(inputs)
    if _use_b2_q64_d129(inputs):
        return remaining.launch_for_eval(inputs)
    return scalar_capacity.launch_for_eval(inputs)

def knn_search_compile_and_launch_dynamic_d_residual_ffc4(*, benchmark: bool=True, shapes: list[dict[str, Any]] | None=None) -> dict[str, Any]:
    from .._dispatch_runtime import evaluate
    result = evaluate(launch_for_eval, shapes=TARGET_SHAPES if shapes is None else shapes, benchmark=benchmark)
    result['passed'] = bool(result.get('summary', {}).get('all_correct'))
    print(result)
    return result
