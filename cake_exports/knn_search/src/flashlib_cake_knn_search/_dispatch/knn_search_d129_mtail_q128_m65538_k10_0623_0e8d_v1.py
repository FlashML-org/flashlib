"""Round-136/0e8d D129 M65538 K10 seed for kNN search.

Minimum target architecture: sm_100a for the tcgen05/TMEM producer path. This
additive bucket-kernel module targets ``B=1,Q=128,M=65538,D=129,K=10`` by
reusing the source-clean direct-stride dynamic-D tcgen05 producer from ccef.
Guard misses and forced fallback delegate to the current f39e Weave-only
dispatcher.
"""
from __future__ import annotations
from json import loads as _json_loads
from .._dispatch_runtime import _capture_cuTensorMapEncodeTiled, _decode_capture, _import_dispatch_module, _ir_proxy
from typing import Any
from . import knn_search_dispatch0622_dynamicd_prefix4_b2selftail_d129_k1_f39e_v1 as parent
from . import knn_search_dynamic_d_high_q128_directstride_tcgen05_0618_ccef_v1 as highd
THREADS = highd.THREADS
MERGE_THREADS = highd.MERGE_THREADS
BLOCK_Q = highd.BLOCK_Q
BLOCK_M = highd.BLOCK_M
D_STAGE = highd.D_STAGE
K_MAX = highd.K_MAX
HIGH_DYNAMIC_D_SPLIT_M = highd.HIGH_DYNAMIC_D_SPLIT_M
ir = _decode_capture(_json_loads('{"__ir__": "knn_search_dynamic_d_high_directstride_tcgen05_partial_0618_ccef_v1", "arg_keys": ["queries", "database", "partial_distances", "partial_indices", "B", "Q", "M", "split_m", "num_q_tiles", "total_m_tiles", "tiles_per_split"], "cluster_dims": [1, 1, 1], "computed_smem_bytes": 126720, "constants": [["K_MAX_", 10], ["D_ORIG_", 512], ["NUM_D_PASSES_", 4], ["Q_NORM_PARTS_", 32]], "cta_group": 1, "threads": 640}'))
partial_ir = _decode_capture(_json_loads('{"__ir__": "knn_search_dynamic_d_high_directstride_tcgen05_partial_0618_ccef_v1", "arg_keys": ["queries", "database", "partial_distances", "partial_indices", "B", "Q", "M", "split_m", "num_q_tiles", "total_m_tiles", "tiles_per_split"], "cluster_dims": [1, 1, 1], "computed_smem_bytes": 126720, "constants": [["K_MAX_", 10], ["D_ORIG_", 512], ["NUM_D_PASSES_", 4], ["Q_NORM_PARTS_", 32]], "cta_group": 1, "threads": 640}'))
merge_ir = _decode_capture(_json_loads('{"__ir__": "knn_search_mma_split_merge_q128_const148_v1", "arg_keys": ["partial_distances", "partial_indices", "out_distances", "out_indices", "B", "Q", "K", "split_m", "num_q_tiles"], "cluster_dims": [1, 1, 1], "computed_smem_bytes": 0, "constants": [["K_MAX_", 10]], "cta_group": 1, "threads": 32}'))
parent_ir = _decode_capture(_json_loads('{"__ir__": "knn_search_mma_split_partial_v1", "arg_keys": ["queries", "database", "partial_distances", "partial_indices", "B", "Q", "M", "split_m", "num_q_tiles", "total_m_tiles", "tiles_per_split"], "cluster_dims": [1, 1, 1], "computed_smem_bytes": 108800, "constants": [["K_MAX_", 10], ["EXPOSE_COL_COHORTS", 0], ["FULL_M_TILES", 0]], "cta_group": 1, "threads": 640}'))
ROUTE_D129_Q128_M65538 = 'round136_0e8d_d129_q128_m65538_k10_directstride_tcgen05'
ROUTE_PARENT_F39E = 'round136_0e8d_parent_f39e_dispatcher'
CONSUMED_HIGH_D_SEED = highd.CONSUMED_SEED
CONSUMED_PARENT_SEEDS = _decode_capture(_json_loads('{"__tuple__": ["weave-evolve-knn-search-f34c", "weave-evolve-knn-search-d4a9", "weave-evolve-knn-search-4024", "weave-evolve-knn-search-f6df", "weave-evolve-knn-search-49a1", "weave-evolve-knn-search-54a5", "weave-evolve-knn-search-56c2", "weave-evolve-knn-search-0f5b", "weave-evolve-knn-search-r3-extk-lowd-d256", "weave-evolve-knn-search-c08b", "weave-evolve-knn-search-f828", "weave-evolve-knn-search-fbc6", "weave-evolve-knn-search-4944", "weave-evolve-knn-search-2c73", "weave-evolve-knn-search-d44f", "weave-evolve-knn-search-0d0b", "weave-evolve-knn-search-859c", "weave-evolve-knn-search-31af", "weave-evolve-knn-search-abaf", "weave-evolve-knn-search-7ad2", "weave-evolve-knn-search-bbab", "weave-evolve-knn-search-36bd", "weave-evolve-knn-search-9286-d1d5-tile-reduce", "weave-evolve-knn-search-ccef-highd-directstride", "weave-evolve-knn-search-9286-d512-q64-row16-directstride", "weave-evolve-knn-search-9286-d768d1024-directstride-tcgen05", "weave-evolve-knn-search-5698", "weave-evolve-knn-search-4832-d3-k10-self", "weave-evolve-knn-search-189c", "weave-evolve-knn-search-2a44", "weave-evolve-knn-search-c14d", "weave-evolve-knn-search-2f5d", "weave-evolve-knn-search-3053", "current-run-branch-598a", "weave-evolve-knn-search-12d4", "weave-evolve-knn-search-5ded", "floor13_k64_q96_91a3_exact", "floor13_k64_q384_e858_exact", "floor13_k80_prefix8_c3f6", "weave-evolve-knn-search-5a9d", "weave-evolve-knn-search-6055", "floor13_k80_prefix4_4b63", "weave-evolve-knn-search-f31a", "weave-evolve-knn-search-b979", "weave-evolve-knn-search-07a2"]}'))
CONSUMED_SEEDS: tuple[str, ...] = (*CONSUMED_PARENT_SEEDS, CONSUMED_HIGH_D_SEED)
TARGET_LABELS: tuple[str, ...] = ('exp_cov_tail_q128_m65538_d129_k10',)
TARGET_SHAPES: list[dict[str, Any]] = [{'label': 'exp_cov_tail_q128_m65538_d129_k10', 'params': {'B': 1, 'Q': 128, 'M': 65538, 'D': 129, 'K': 10, 'dtype': 'bfloat16', 'seed': 620102, 'self_search': False, 'min_recall': 0.999}}]
ENTRYPOINT = 'loom.examples.weave.knn_search_d129_mtail_q128_m65538_k10_0623_0e8d_v1:launch_for_eval'
SOURCE_ENTRYPOINT = 'loom.examples.weave.knn_search_dynamic_d_high_q128_directstride_tcgen05_0618_ccef_v1:_launch_high_dynamic_d_tcgen05'
SOURCE_ROUND_DOC = 'design_doc/active/weave_evolve_knn_search_round_123_189c_dynamicd_alias_q128.md'
_TARGET_ENTRY: dict[str, Any] = {'shape_key': 'exp_cov_tail_q128_m65538_d129_k10', 'label': 'exp_cov_tail_q128_m65538_d129_k10', 'shape': (1, 128, 65538, 129, 10, False), 'guard': 'B == 1 and Q == 128 and M == 65538 and D == 129 and K == 10 and not self_search and not forced_fallback and arch in {sm_100a,sm_103a}', 'route': ROUTE_D129_Q128_M65538, 'entrypoint': ENTRYPOINT, 'source_entrypoint': SOURCE_ENTRYPOINT, 'selected_seed': CONSUMED_HIGH_D_SEED, 'source_round_doc': SOURCE_ROUND_DOC, 'coverage_class': 'bucket_seed_exp_cov_tail_q128_m65538_d129_k10', 'arch_requirement': 'sm_100a', 'route_source': 'shape-specific-seed'}
SHAPE_DISPATCH_REGISTRY: tuple[dict[str, Any], ...] = (_TARGET_ENTRY, *parent.SHAPE_DISPATCH_REGISTRY)

def _shape_key(inputs: dict[str, Any]) -> tuple[int, int, int, int, int, bool]:
    return (int(inputs.get('B', 1)), int(inputs['Q']), int(inputs['M']), int(inputs['D']), int(inputs['K']), bool(inputs.get('self_search', False)))

def _forced_fallback(inputs: dict[str, Any]) -> bool:
    return bool(inputs.get('force_fallback', False))

def _tcgen05_capable_arch() -> bool:
    return bool(highd.mma._tcgen05_capable_arch())

def _active_entry(inputs: dict[str, Any]) -> dict[str, Any] | None:
    if _forced_fallback(inputs) or not _tcgen05_capable_arch():
        return None
    if _shape_key(inputs) == _TARGET_ENTRY['shape']:
        return _TARGET_ENTRY
    return None

def _guard_order() -> list[str]:
    return [str(entry['shape_key']) for entry in SHAPE_DISPATCH_REGISTRY]

def selected_route(inputs: dict[str, Any]) -> str:
    entry = _active_entry(inputs)
    if entry is not None:
        return str(entry['route'])
    return parent.selected_route(inputs)

def selected_route_name(inputs: dict[str, Any]) -> str:
    return selected_route(inputs)

def _parent_info(inputs: dict[str, Any]) -> dict[str, Any]:
    info = dict(parent.route_info(inputs))
    route = str(info.get('selected_route') or info.get('route') or parent.selected_route(inputs))
    info['route'] = route
    info['selected_route'] = route
    info['guard_order'] = _guard_order()
    info.setdefault('production_policy', 'weave_only')
    info.setdefault('external_fallback', None)
    info.setdefault('missing_weave_route', False)
    return info

def _entry_info(inputs: dict[str, Any], entry: dict[str, Any]) -> dict[str, Any]:
    parent_info = dict(parent.route_info(inputs))
    parent_route = str(parent_info.get('selected_route') or parent_info.get('route') or parent.selected_route(inputs))
    total_m_tiles = (int(inputs['M']) + BLOCK_M - 1) // BLOCK_M
    split_m = min(HIGH_DYNAMIC_D_SPLIT_M, total_m_tiles)
    num_q_tiles = (int(inputs['Q']) + BLOCK_Q - 1) // BLOCK_Q
    tiles_per_split = (total_m_tiles + split_m - 1) // split_m
    return {'route': entry['route'], 'selected_route': entry['route'], 'selected_entrypoint': entry['entrypoint'], 'source_entrypoint': entry['source_entrypoint'], 'parent_route': parent_route, 'replaced_route': parent_route, 'route_kind': 'specialized', 'route_source': 'shape-specific-seed', 'coverage_class': entry['coverage_class'], 'classification': 'seed-produced', 'coverage_only': False, 'production_policy': 'weave_only', 'external_fallback': None, 'guard_order': _guard_order(), 'guard_id': entry['shape_key'], 'selected_guard': entry['guard'], 'fallback': ROUTE_PARENT_F39E, 'missing_weave_route': False, 'forced_fallback': False, 'selected_seed': entry['selected_seed'], 'source_round_doc': entry['source_round_doc'], 'arch_requirement': entry['arch_requirement'], 'split_m': split_m, 'num_q_tiles': num_q_tiles, 'total_m_tiles': total_m_tiles, 'tiles_per_split': tiles_per_split}

def route_info(inputs: dict[str, Any]) -> dict[str, Any]:
    entry = _active_entry(inputs)
    if entry is None:
        return _parent_info(inputs)
    return _entry_info(inputs, entry)

def route_trace_entry(label: str, inputs: dict[str, Any]) -> dict[str, Any]:
    return {'label': label, 'shape_key': label, 'B': int(inputs['B']), 'Q': int(inputs['Q']), 'M': int(inputs['M']), 'D': int(inputs['D']), 'K': int(inputs['K']), 'self_search': bool(inputs.get('self_search', False)), **route_info(inputs)}

def launch_parent_for_eval(inputs: dict[str, Any]) -> dict[str, Any]:
    return parent.launch_for_eval(inputs)

def launch_for_eval(inputs: dict[str, Any]) -> dict[str, Any]:
    if _active_entry(inputs) is not None:
        return highd._launch_high_dynamic_d_tcgen05(inputs)
    return parent.launch_for_eval(inputs)

def knn_search_compile_and_launch_d129_mtail_q128_m65538_k10_0623_0e8d(*, benchmark: bool=True, shapes: list[dict[str, Any]] | None=None) -> dict[str, Any]:
    from .._dispatch_runtime import evaluate
    result = evaluate(launch_for_eval, shapes=TARGET_SHAPES if shapes is None else shapes, benchmark=benchmark)
    result['passed'] = bool(result.get('summary', {}).get('all_correct'))
    print(result)
    return result
