"""Dynamic-D blind-spot seed aggregator with K64 coverage.

Minimum target architecture: sm_100a for inherited tcgen05/TMEM routes. This
additive candidate does not edit the production dispatcher. It exports guarded
Weave-only seed routes for the 0618 dynamic-D rows covered by ccef v1 plus the
``B=1,Q=64,M=65536,D=257,K=64`` K64 bucket seed.
"""
from __future__ import annotations
from json import loads as _json_loads
from .._dispatch_runtime import _capture_cuTensorMapEncodeTiled, _decode_capture, _import_dispatch_module, _ir_proxy
from typing import Any
from .._dispatch_runtime import select_named_shapes
from . import knn_search_dynamic_d_k64_q64_m65536_tcgen05_0618_ccef_v1 as k64_dynamic
from . import knn_search_dynamic_d_remaining_seeds_0618_ccef_v1 as parent
THREADS = k64_dynamic.THREADS
BLOCK_Q = k64_dynamic.BLOCK_Q
BLOCK_M = k64_dynamic.BLOCK_M
K_MAX = k64_dynamic.K64_MAX
MERGE_THREADS = k64_dynamic.MERGE_THREADS
ir = _decode_capture(_json_loads('{"__ir__": "knn_search_dynamic_d257_k64_q64_partial_0618_ccef_v1", "arg_keys": ["queries", "database", "partial_distances", "partial_indices", "B", "Q", "M", "split_m", "num_q_tiles", "total_m_tiles", "tiles_per_split", "partial_list_count"], "cluster_dims": [1, 1, 1], "computed_smem_bytes": 151296, "constants": [["K_MAX_", 64]], "cta_group": 1, "threads": 256}'))
partial_ir = _decode_capture(_json_loads('{"__ir__": "knn_search_dynamic_d257_k64_q64_partial_0618_ccef_v1", "arg_keys": ["queries", "database", "partial_distances", "partial_indices", "B", "Q", "M", "split_m", "num_q_tiles", "total_m_tiles", "tiles_per_split", "partial_list_count"], "cluster_dims": [1, 1, 1], "computed_smem_bytes": 151296, "constants": [["K_MAX_", 64]], "cta_group": 1, "threads": 256}'))
merge_ir = _decode_capture(_json_loads('{"__ir__": "knn_search_dynamic_d257_k64_q64_merge_0618_ccef_v1", "arg_keys": ["partial_distances", "partial_indices", "out_distances", "out_indices", "B", "Q", "K", "partial_list_count", "num_q_tiles"], "cluster_dims": [1, 1, 1], "computed_smem_bytes": 0, "constants": [["K_MAX_", 64]], "cta_group": 1, "threads": 32}'))
pack_ir = _decode_capture(_json_loads('{"__ir__": "knn_search_dynamic_d_tiny_pack_bf16_0618_c8b9_v1", "arg_keys": ["queries", "database", "padded_queries", "padded_database", "B", "Q", "M"], "cluster_dims": [1, 1, 1], "computed_smem_bytes": 0, "constants": [["D_ORIG_", 3], ["D_PAD_", 16]], "cta_group": 1, "threads": 256}'))
parent_ir = _decode_capture(_json_loads('{"__ir__": "knn_search_dynamic_d511_tcgen05_partial_0618_c8b9_v1", "arg_keys": ["queries", "database", "partial_distances", "partial_indices", "B", "Q", "M", "split_m", "num_q_tiles", "total_m_tiles", "tiles_per_split"], "cluster_dims": [1, 1, 1], "computed_smem_bytes": 126720, "constants": [["K_MAX_", 10], ["D_TOTAL_", 512], ["NUM_D_PASSES_", 4], ["Q_NORM_PARTS_", 32]], "cta_group": 1, "threads": 640}'))
parent_partial_ir = _decode_capture(_json_loads('{"__ir__": "knn_search_dynamic_d511_tcgen05_partial_0618_c8b9_v1", "arg_keys": ["queries", "database", "partial_distances", "partial_indices", "B", "Q", "M", "split_m", "num_q_tiles", "total_m_tiles", "tiles_per_split"], "cluster_dims": [1, 1, 1], "computed_smem_bytes": 126720, "constants": [["K_MAX_", 10], ["D_TOTAL_", 512], ["NUM_D_PASSES_", 4], ["Q_NORM_PARTS_", 32]], "cta_group": 1, "threads": 640}'))
parent_merge_ir = _decode_capture(_json_loads('{"__ir__": "knn_search_mma_split_merge_q128_const148_v1", "arg_keys": ["partial_distances", "partial_indices", "out_distances", "out_indices", "B", "Q", "K", "split_m", "num_q_tiles"], "cluster_dims": [1, 1, 1], "computed_smem_bytes": 0, "constants": [["K_MAX_", 10]], "cta_group": 1, "threads": 32}'))
parent_pack_ir = _decode_capture(_json_loads('{"__ir__": "knn_search_dynamic_d_tiny_pack_bf16_0618_c8b9_v1", "arg_keys": ["queries", "database", "padded_queries", "padded_database", "B", "Q", "M"], "cluster_dims": [1, 1, 1], "computed_smem_bytes": 0, "constants": [["D_ORIG_", 3], ["D_PAD_", 16]], "cta_group": 1, "threads": 256}'))
d3_tile_ir = _decode_capture(_json_loads('{"__ir__": "knn_search_dynamic_d3_tile_reduce_partial_0618_c8b9_v1", "arg_keys": ["queries", "database", "partial_distances", "partial_indices", "B", "Q", "M", "K", "num_m_tiles"], "cluster_dims": [1, 1, 1], "computed_smem_bytes": 10240, "constants": [["K_MAX_", 10]], "cta_group": 1, "threads": 256}'))
scalar_capacity_ir = _decode_capture(_json_loads('{"__ir__": "knn_search_scalar_capacity_partial_v1", "arg_keys": ["queries", "database", "partial_distances", "partial_indices", "B", "Q", "M", "num_m_tiles"], "cluster_dims": [1, 1, 1], "computed_smem_bytes": 0, "constants": [["D_", 8], ["K_CAP_", 64], ["BLOCK_M_", 512], ["NUM_WARPS_", 8]], "cta_group": 1, "threads": 256}'))
ROUTE_DYNAMIC_D257_Q64_K64 = k64_dynamic.ROUTE_DYNAMIC_D257_Q64_K64
ROUTE_SCALAR_CAPACITY = parent.ROUTE_SCALAR_CAPACITY
CONSUMED_PARENT_SEEDS: tuple[str, ...] = parent.CONSUMED_SEEDS
CONSUMED_K64_SEED = k64_dynamic.CONSUMED_SEED
CONSUMED_SEEDS: tuple[str, ...] = (*CONSUMED_PARENT_SEEDS, CONSUMED_K64_SEED)
DYNAMIC_D257_Q64_K64_LABELS: tuple[str, ...] = k64_dynamic.DYNAMIC_D257_Q64_K64_LABELS
REMAINING_DYNAMIC_D_PLUS_K64_LABELS: tuple[str, ...] = (*parent.REMAINING_DYNAMIC_D_SEED_LABELS, *DYNAMIC_D257_Q64_K64_LABELS)
REMAINING_DYNAMIC_D_PLUS_K64_SHAPES = _decode_capture(_json_loads('[{"__dict_items__": [["label", "blind_dyn_d129_q128_m65536_k10"], ["params", {"__dict_items__": [["B", 1], ["Q", 128], ["M", 65536], ["D", 129], ["K", 10], ["dtype", "bfloat16"], ["seed", 610804], ["self_search", false], ["min_recall", 0.999]]}]]}, {"__dict_items__": [["label", "blind_dyn_d257_q128_m65536_k10"], ["params", {"__dict_items__": [["B", 1], ["Q", 128], ["M", 65536], ["D", 257], ["K", 10], ["dtype", "bfloat16"], ["seed", 610805], ["self_search", false], ["min_recall", 0.999]]}]]}, {"__dict_items__": [["label", "blind_dyn_d511_q128_m65536_k10"], ["params", {"__dict_items__": [["B", 1], ["Q", 128], ["M", 65536], ["D", 511], ["K", 10], ["dtype", "bfloat16"], ["seed", 610806], ["self_search", false], ["min_recall", 0.999]]}]]}, {"__dict_items__": [["label", "blind_dyn_d384_q32_m131072_k10"], ["params", {"__dict_items__": [["B", 1], ["Q", 32], ["M", 131072], ["D", 384], ["K", 10], ["dtype", "bfloat16"], ["seed", 610807], ["self_search", false], ["min_recall", 0.999]]}]]}, {"__dict_items__": [["label", "blind_dyn_b2_q64_m65536_d129_k10"], ["params", {"__dict_items__": [["B", 2], ["Q", 64], ["M", 65536], ["D", 129], ["K", 10], ["dtype", "bfloat16"], ["seed", 610809], ["self_search", false], ["min_recall", 0.999]]}]]}, {"__dict_items__": [["label", "blind_dyn_self_q2048_m2048_d3_k10"], ["params", {"__dict_items__": [["B", 1], ["Q", 2048], ["M", 2048], ["D", 3], ["K", 10], ["dtype", "bfloat16"], ["seed", 610810], ["self_search", true], ["min_recall", 0.999]]}]]}, {"__dict_items__": [["label", "blind_dyn_d257_k64_q64_m65536"], ["params", {"__dict_items__": [["B", 1], ["Q", 64], ["M", 65536], ["D", 257], ["K", 64], ["dtype", "bfloat16"], ["seed", 610808], ["self_search", false], ["min_recall", 0.999]]}]]}]'))
_K64_ENTRY: dict[str, Any] = {'overlay': 'ccef_dynamic_d257_q64_m65536_k64_tcgen05', 'shape_key': 'B=1,Q=64,M=65536,D=257,K=64', 'labels': DYNAMIC_D257_Q64_K64_LABELS, 'guard': 'B == 1 and Q == 64 and M == 65536 and D == 257 and K == 64 and not self_search and not forced_fallback', 'route': ROUTE_DYNAMIC_D257_Q64_K64, 'entrypoint': 'loom.examples.weave.knn_search_dynamic_d_remaining_seeds_0618_ccef_v2:launch_for_eval', 'selected_seed': CONSUMED_K64_SEED, 'source_round_doc': 'design_doc/active/weave_evolve_knn_search_round_4_ccef_k64_dynamic_d.md', 'coverage_class': 'bucket_seed_dynamic_d_k64_q64_m65536'}
SHAPE_DISPATCH_REGISTRY: tuple[dict[str, Any], ...] = (*parent.SHAPE_DISPATCH_REGISTRY, _K64_ENTRY)

def _active_entry(inputs: dict[str, Any]) -> dict[str, Any] | None:
    if k64_dynamic._use_dynamic_d257_q64_k64(inputs):
        return _K64_ENTRY
    return parent._active_entry(inputs)

def selected_route(inputs: dict[str, Any]) -> str:
    entry = _active_entry(inputs)
    if entry is None:
        return ROUTE_SCALAR_CAPACITY
    return str(entry['route'])

def selected_route_name(inputs: dict[str, Any]) -> str:
    return selected_route(inputs)

def route_info(inputs: dict[str, Any]) -> dict[str, Any]:
    if k64_dynamic._use_dynamic_d257_q64_k64(inputs):
        info = k64_dynamic.route_info(inputs)
        info['selected_entrypoint'] = 'loom.examples.weave.knn_search_dynamic_d_remaining_seeds_0618_ccef_v2:launch_for_eval'
        info['source_round_doc'] = _K64_ENTRY['source_round_doc']
        return info
    return parent.route_info(inputs)

def route_trace_entry(label: str, inputs: dict[str, Any]) -> dict[str, Any]:
    return {'label': label, 'shape_key': label, 'B': int(inputs['B']), 'Q': int(inputs['Q']), 'M': int(inputs['M']), 'D': int(inputs['D']), 'K': int(inputs['K']), 'self_search': bool(inputs.get('self_search', False)), **route_info(inputs)}

def launch_for_eval(inputs: dict[str, Any]) -> dict[str, Any]:
    if k64_dynamic._use_dynamic_d257_q64_k64(inputs):
        return k64_dynamic._launch_dynamic_d257_q64_k64(inputs)
    return parent.launch_for_eval(inputs)

def knn_search_compile_and_launch_dynamic_d_remaining_seeds_v2(*, benchmark: bool=True, shapes: list[dict[str, Any]] | None=None) -> dict[str, Any]:
    from .._dispatch_runtime import evaluate
    selected = REMAINING_DYNAMIC_D_PLUS_K64_SHAPES if shapes is None else shapes
    result = evaluate(launch_for_eval, shapes=selected, benchmark=benchmark)
    result['passed'] = bool(result.get('summary', {}).get('all_correct'))
    print(result)
    return result
