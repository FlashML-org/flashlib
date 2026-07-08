"""Round-133/e472 floor13 B2/self/tail K10 seed bundle for kNN search.

Minimum target architecture: sm_100a for the tcgen05/TMEM producer path. This
additive bucket-kernel module targets the remaining floor13 K10 rows:
``B2/Q96/M196608``, ``self Q1536/M1536``, and ``Q1025/M65537``. It reuses the
source-clean Q-bucket Weave producer with exact shape guards and does not edit
production dispatch.
"""
from __future__ import annotations
from json import loads as _json_loads
from .._dispatch_runtime import _capture_cuTensorMapEncodeTiled, _decode_capture, _import_dispatch_module, _ir_proxy
from typing import Any
from .._dispatch_runtime import select_named_shapes
from . import knn_search_dispatch0622_current_portfolio_e472_v1 as parent
from . import knn_search_highq_midm_qbucket_0611_r19_4e96_v1 as qbucket
THREADS = qbucket.mma.THREADS
MERGE_THREADS = qbucket.mma.MERGE_THREADS
BLOCK_Q = qbucket.mma.BLOCK_Q
BLOCK_M = qbucket.mma.BLOCK_M
D_STATIC = qbucket.mma.D_STATIC
K_MAX = qbucket.mma.K_MAX
ir = _decode_capture(_json_loads('{"__ir__": "knn_search_mma_split_partial_v1", "arg_keys": ["queries", "database", "partial_distances", "partial_indices", "B", "Q", "M", "split_m", "num_q_tiles", "total_m_tiles", "tiles_per_split"], "cluster_dims": [1, 1, 1], "computed_smem_bytes": 108800, "constants": [["K_MAX_", 10], ["EXPOSE_COL_COHORTS", 0], ["FULL_M_TILES", 0]], "cta_group": 1, "threads": 640}'))
partial_ir = _decode_capture(_json_loads('{"__ir__": "knn_search_mma_split_partial_v1", "arg_keys": ["queries", "database", "partial_distances", "partial_indices", "B", "Q", "M", "split_m", "num_q_tiles", "total_m_tiles", "tiles_per_split"], "cluster_dims": [1, 1, 1], "computed_smem_bytes": 108800, "constants": [["K_MAX_", 10], ["EXPOSE_COL_COHORTS", 0], ["FULL_M_TILES", 0]], "cta_group": 1, "threads": 640}'))
merge_ir = _decode_capture(_json_loads('{"__ir__": "knn_search_mma_split_merge_stream_v1", "arg_keys": ["partial_distances", "partial_indices", "out_distances", "out_indices", "B", "Q", "K", "split_m", "num_q_tiles"], "cluster_dims": [1, 1, 1], "computed_smem_bytes": 0, "constants": [["K_MAX_", 10]], "cta_group": 1, "threads": 32}'))
merge_q128_const148_ir = _decode_capture(_json_loads('{"__ir__": "knn_search_mma_split_merge_q128_const148_v1", "arg_keys": ["partial_distances", "partial_indices", "out_distances", "out_indices", "B", "Q", "K", "split_m", "num_q_tiles"], "cluster_dims": [1, 1, 1], "computed_smem_bytes": 0, "constants": [["K_MAX_", 10]], "cta_group": 1, "threads": 32}'))
parent_ir = _decode_capture(_json_loads('{"__ir__": "knn_search_mma_split_partial_v1", "arg_keys": ["queries", "database", "partial_distances", "partial_indices", "B", "Q", "M", "split_m", "num_q_tiles", "total_m_tiles", "tiles_per_split"], "cluster_dims": [1, 1, 1], "computed_smem_bytes": 108800, "constants": [["K_MAX_", 10], ["EXPOSE_COL_COHORTS", 0], ["FULL_M_TILES", 0]], "cta_group": 1, "threads": 640}'))
ROUTE_B2_Q96_M196608 = 'e472_floor13_b2_q96_m196608_k10_qbucket_tcgen05'
ROUTE_SELF_Q1536_M1536 = 'e472_floor13_self_q1536_m1536_k10_qbucket_tcgen05'
ROUTE_TAIL_Q1025_M65537 = 'e472_floor13_tail_q1025_m65537_k10_qbucket_tcgen05'
ROUTE_PARENT_E472 = parent.PROFILE_ALL
CONSUMED_QBUCKET_SEED = 'weave-evolve-knn-search-4e96-qbucket'
CONSUMED_PARENT_SEEDS = _decode_capture(_json_loads('{"__tuple__": ["weave-evolve-knn-search-f34c", "weave-evolve-knn-search-d4a9", "weave-evolve-knn-search-4024", "weave-evolve-knn-search-f6df", "weave-evolve-knn-search-49a1", "weave-evolve-knn-search-54a5", "weave-evolve-knn-search-56c2", "weave-evolve-knn-search-0f5b", "weave-evolve-knn-search-r3-extk-lowd-d256", "weave-evolve-knn-search-c08b", "weave-evolve-knn-search-f828", "weave-evolve-knn-search-fbc6", "weave-evolve-knn-search-4944", "weave-evolve-knn-search-2c73", "weave-evolve-knn-search-d44f", "weave-evolve-knn-search-0d0b", "weave-evolve-knn-search-859c", "weave-evolve-knn-search-31af", "weave-evolve-knn-search-abaf", "weave-evolve-knn-search-7ad2", "weave-evolve-knn-search-bbab", "weave-evolve-knn-search-36bd", "weave-evolve-knn-search-9286-d1d5-tile-reduce", "weave-evolve-knn-search-ccef-highd-directstride", "weave-evolve-knn-search-9286-d512-q64-row16-directstride", "weave-evolve-knn-search-9286-d768d1024-directstride-tcgen05", "weave-evolve-knn-search-5698", "weave-evolve-knn-search-4832-d3-k10-self", "weave-evolve-knn-search-189c", "weave-evolve-knn-search-2a44", "weave-evolve-knn-search-c14d", "weave-evolve-knn-search-2f5d", "weave-evolve-knn-search-3053", "current-run-branch-598a", "weave-evolve-knn-search-12d4", "weave-evolve-knn-search-5ded"]}'))
CONSUMED_SEEDS: tuple[str, ...] = (*CONSUMED_PARENT_SEEDS, CONSUMED_QBUCKET_SEED)
TARGET_LABELS: tuple[str, ...] = ('blind_0622_b2_q96_m196608_d128_k10', 'blind_0622_self_q1536_m1536_d128_k10', 'blind_0622_tail_q1025_m65537_d128_k10')
TARGET_SHAPES = _decode_capture(_json_loads('[{"__dict_items__": [["label", "blind_0622_b2_q96_m196608_d128_k10"], ["params", {"__dict_items__": [["B", 2], ["Q", 96], ["M", 196608], ["D", 128], ["K", 10], ["dtype", "bfloat16"], ["seed", 611009], ["self_search", false], ["min_recall", 0.999]]}]]}, {"__dict_items__": [["label", "blind_0622_self_q1536_m1536_d128_k10"], ["params", {"__dict_items__": [["B", 1], ["Q", 1536], ["M", 1536], ["D", 128], ["K", 10], ["dtype", "bfloat16"], ["seed", 611011], ["self_search", true], ["min_recall", 0.999]]}]]}, {"__dict_items__": [["label", "blind_0622_tail_q1025_m65537_d128_k10"], ["params", {"__dict_items__": [["B", 1], ["Q", 1025], ["M", 65537], ["D", 128], ["K", 10], ["dtype", "bfloat16"], ["seed", 611012], ["self_search", false], ["min_recall", 0.999]]}]]}]'))
_B2_ENTRY: dict[str, Any] = {'shape_key': 'e472_floor13_b2_q96_m196608_d128_k10_qbucket', 'label': 'blind_0622_b2_q96_m196608_d128_k10', 'shape': (2, 96, 196608, 128, 10, False), 'guard': 'B == 2 and Q == 96 and M == 196608 and D == 128 and K == 10 and not self_search and not forced_fallback and arch in {sm_100a,sm_103a}', 'route': ROUTE_B2_Q96_M196608, 'coverage_class': 'bucket_seed_floor13_b2_q96_m196608_d128_k10'}
_SELF_ENTRY: dict[str, Any] = {'shape_key': 'e472_floor13_self_q1536_m1536_d128_k10_qbucket', 'label': 'blind_0622_self_q1536_m1536_d128_k10', 'shape': (1, 1536, 1536, 128, 10, True), 'guard': 'B == 1 and Q == M == 1536 and D == 128 and K == 10 and self_search and not forced_fallback and arch in {sm_100a,sm_103a}', 'route': ROUTE_SELF_Q1536_M1536, 'coverage_class': 'bucket_seed_floor13_self_q1536_m1536_d128_k10'}
_TAIL_ENTRY: dict[str, Any] = {'shape_key': 'e472_floor13_tail_q1025_m65537_d128_k10_qbucket', 'label': 'blind_0622_tail_q1025_m65537_d128_k10', 'shape': (1, 1025, 65537, 128, 10, False), 'guard': 'B == 1 and Q == 1025 and M == 65537 and D == 128 and K == 10 and not self_search and not forced_fallback and arch in {sm_100a,sm_103a}', 'route': ROUTE_TAIL_Q1025_M65537, 'coverage_class': 'bucket_seed_floor13_tail_q1025_m65537_d128_k10'}
_ENTRIES: tuple[dict[str, Any], ...] = (_B2_ENTRY, _SELF_ENTRY, _TAIL_ENTRY)
ENTRYPOINT = 'loom.examples.weave.knn_search_floor13_b2_self_tail_k10_0622_e472_v1:launch_for_eval'
SOURCE_ENTRYPOINT = 'loom.examples.weave.knn_search_highq_midm_qbucket_0611_r19_4e96_v1:_launch_highq_midm_qbucket_mma'
SOURCE_ROUND_DOC = 'design_doc/active/weave_evolve_knn_search_round_1_4e96_scalar_capacity.md'
SHAPE_DISPATCH_REGISTRY: tuple[dict[str, Any], ...] = (*({**entry, 'entrypoint': ENTRYPOINT, 'source_entrypoint': SOURCE_ENTRYPOINT, 'selected_seed': CONSUMED_QBUCKET_SEED, 'source_round_doc': SOURCE_ROUND_DOC, 'route_source': 'shape-specific-seed', 'arch_requirement': 'sm_100a'} for entry in _ENTRIES), *parent.SHAPE_DISPATCH_REGISTRY)

def _shape_key(inputs: dict[str, Any]) -> tuple[int, int, int, int, int, bool]:
    return (int(inputs.get('B', 1)), int(inputs['Q']), int(inputs['M']), int(inputs['D']), int(inputs['K']), bool(inputs.get('self_search', False)))

def _forced_fallback(inputs: dict[str, Any]) -> bool:
    return bool(inputs.get('force_fallback', False))

def _tcgen05_capable_arch() -> bool:
    return bool(qbucket.mma._tcgen05_capable_arch())

def _active_entry(inputs: dict[str, Any]) -> dict[str, Any] | None:
    if _forced_fallback(inputs) or not _tcgen05_capable_arch():
        return None
    shape = _shape_key(inputs)
    for entry in _ENTRIES:
        if shape == entry['shape']:
            return entry
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
    split_m = qbucket._select_highq_midm_split_m(int(inputs['Q']), int(inputs['M']))
    return {'route': entry['route'], 'selected_route': entry['route'], 'selected_entrypoint': ENTRYPOINT, 'source_entrypoint': SOURCE_ENTRYPOINT, 'parent_route': parent_route, 'replaced_route': parent_route, 'route_kind': 'specialized', 'route_source': 'shape-specific-seed', 'coverage_class': entry['coverage_class'], 'classification': 'seed-produced', 'coverage_only': False, 'production_policy': 'weave_only', 'external_fallback': None, 'guard_order': _guard_order(), 'guard_id': entry['shape_key'], 'selected_guard': entry['guard'], 'fallback': ROUTE_PARENT_E472, 'missing_weave_route': False, 'forced_fallback': False, 'selected_seed': CONSUMED_QBUCKET_SEED, 'source_round_doc': SOURCE_ROUND_DOC, 'arch_requirement': 'sm_100a', 'split_m': split_m, 'num_q_tiles': (int(inputs['Q']) + BLOCK_Q - 1) // BLOCK_Q}

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
        return qbucket._launch_highq_midm_qbucket_mma(inputs)
    return parent.launch_for_eval(inputs)

def knn_search_compile_and_launch_floor13_b2_self_tail_k10(*, benchmark: bool=True, shapes: list[dict[str, Any]] | None=None) -> dict[str, Any]:
    from .._dispatch_runtime import evaluate
    result = evaluate(launch_for_eval, shapes=TARGET_SHAPES if shapes is None else shapes, benchmark=benchmark)
    result['passed'] = bool(result.get('summary', {}).get('all_correct'))
    print(result)
    return result
