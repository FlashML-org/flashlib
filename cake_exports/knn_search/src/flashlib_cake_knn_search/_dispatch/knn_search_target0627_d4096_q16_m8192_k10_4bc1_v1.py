"""Target0627 D4096/Q16/M8192/K10 kNN seed.

Minimum target architecture: sm_100a. This additive bucket-kernel wrapper owns
the current target-D floor14 row ``B=1,Q=16,M=8192,D=4096,K=10`` and routes it
to the existing D4096 direct-stride tcgen05 producer/merge path. Guard misses
delegate to the current Weave dispatcher; this module does not edit production
dispatch.
"""
from __future__ import annotations
from json import loads as _json_loads
from .._dispatch_runtime import _capture_cuTensorMapEncodeTiled, _decode_capture, _import_dispatch_module, _ir_proxy
from typing import Any
from . import knn_search_d4096_q4q8_m8192m16384_k10_0623_5ff7_v1 as d4096_seed
from . import knn_search_dispatch0629_cd65_plus_104b_4b52_afe1_6c60_29d8_v1 as parent
THREADS = d4096_seed.THREADS
MERGE_THREADS = d4096_seed.MERGE_THREADS
BLOCK_Q = d4096_seed.BLOCK_Q
PARTIAL_STRIDE_Q = d4096_seed.PARTIAL_STRIDE_Q
BLOCK_M = d4096_seed.BLOCK_M
D_STAGE = d4096_seed.D_STAGE
D_ORIG = d4096_seed.D_ORIG
NUM_D_PASSES = d4096_seed.NUM_D_PASSES
K_MAX = d4096_seed.K_MAX
SPLIT_M = d4096_seed.SPLIT_M
SMEM_BYTES = d4096_seed.SMEM_BYTES
partial_ir = _decode_capture(_json_loads('{"__ir__": "knn_search_d4096_q4q8_m8192m16384_k10_partial_0623_5ff7_v1", "arg_keys": ["queries", "database", "partial_distances", "partial_indices", "B", "Q", "M", "split_m", "num_q_tiles", "total_m_tiles"], "cluster_dims": [1, 1, 1], "computed_smem_bytes": 159488, "constants": [["K_MAX_", 10]], "cta_group": 1, "threads": 512}'))
merge_ir = _decode_capture(_json_loads('{"__ir__": "knn_search_d4096_q4q8_m8192m16384_k10_merge_0623_5ff7_v1", "arg_keys": ["partial_distances", "partial_indices", "out_distances", "out_indices", "B", "Q", "K", "split_m", "num_q_tiles"], "cluster_dims": [1, 1, 1], "computed_smem_bytes": 0, "constants": [["K_MAX_", 10]], "cta_group": 1, "threads": 32}'))
ir = _decode_capture(_json_loads('{"__ir__": "knn_search_d4096_q4q8_m8192m16384_k10_partial_0623_5ff7_v1", "arg_keys": ["queries", "database", "partial_distances", "partial_indices", "B", "Q", "M", "split_m", "num_q_tiles", "total_m_tiles"], "cluster_dims": [1, 1, 1], "computed_smem_bytes": 159488, "constants": [["K_MAX_", 10]], "cta_group": 1, "threads": 512}'))
ROUTE_TARGET0627_D4096_Q16_M8192_K10 = '4bc1_target0627_d4096_q16_m8192_k10_directstride_tcgen05'
SELECTED_SEED = 'weave-evolve-knn-search-4bc1'
PRODUCER_SEED = d4096_seed.CONSUMED_SEED
ENTRYPOINT = 'loom.examples.weave.knn_search_target0627_d4096_q16_m8192_k10_4bc1_v1:launch_for_eval'
ROUND_DOC = 'design_doc/active/weave_evolve_knn_search_round_168_4bc1.md'
TARGET_LABELS: tuple[str, ...] = ('target0627_d4096_q16_m8192_k10',)
TARGET_SHAPES: list[dict[str, Any]] = [{'label': 'target0627_d4096_q16_m8192_k10', 'params': {'B': 1, 'Q': 16, 'M': 8192, 'D': 4096, 'K': 10, 'dtype': 'bfloat16', 'seed': 612115, 'self_search': False, 'min_recall': 0.999}}]
SHAPE_DISPATCH_REGISTRY: tuple[dict[str, Any], ...] = ({'shape_key': 'target0627_d4096_q16_m8192_k10', 'label': 'target0627_d4096_q16_m8192_k10', 'labels': TARGET_LABELS, 'shape': (1, 16, 8192, 4096, 10, False), 'guard': 'B == 1 and Q == 16 and M == 8192 and D == 4096 and K == 10 and not self_search and not force_fallback and arch in {sm_100a,sm_103a}', 'route': ROUTE_TARGET0627_D4096_Q16_M8192_K10, 'entrypoint': ENTRYPOINT, 'source_entrypoint': 'loom.examples.weave.knn_search_d4096_q4q8_m8192m16384_k10_0623_5ff7_v1:_launch_d4096_q4q8_tcgen05', 'selected_seed': SELECTED_SEED, 'producer_seed': PRODUCER_SEED, 'source_round_doc': ROUND_DOC, 'coverage_class': 'bucket_seed_target0627_d4096_q16_m8192_k10', 'workflow_mode': 'generalize_auto_tuning', 'auto_tuning_stage': 'tile-shape-search', 'bucket_id': 'target0628_d4096_q16_m8192_k10', 'arch_requirement': 'sm_100a'},)

def _shape_key(inputs: dict[str, Any]) -> tuple[int, int, int, int, int, bool]:
    return (int(inputs.get('B', 1)), int(inputs['Q']), int(inputs['M']), int(inputs['D']), int(inputs['K']), bool(inputs.get('self_search', False)))

def _forced_fallback(inputs: dict[str, Any]) -> bool:
    return bool(inputs.get('force_fallback', False))

def _tcgen05_capable_arch() -> bool:
    return d4096_seed._tcgen05_capable_arch()

def _active_entry(inputs: dict[str, Any]) -> dict[str, Any] | None:
    if _forced_fallback(inputs) or not _tcgen05_capable_arch():
        return None
    if _shape_key(inputs) == SHAPE_DISPATCH_REGISTRY[0]['shape']:
        return SHAPE_DISPATCH_REGISTRY[0]
    return None

def _parent_guard_order() -> list[str]:
    try:
        return parent._guard_order(parent.PROFILE_ALL)
    except Exception:
        return []

def _guard_order() -> list[str]:
    return [str(SHAPE_DISPATCH_REGISTRY[0]['shape_key']), *_parent_guard_order()]

def selected_route(inputs: dict[str, Any]) -> str:
    entry = _active_entry(inputs)
    if entry is not None:
        return str(entry['route'])
    return parent.selected_route(inputs)

def selected_route_name(inputs: dict[str, Any]) -> str:
    return selected_route(inputs)

def _entry_info(inputs: dict[str, Any], entry: dict[str, Any]) -> dict[str, Any]:
    try:
        parent_info = dict(parent.route_info(inputs))
        parent_route = str(parent_info.get('selected_route') or parent_info.get('route') or parent.selected_route(inputs))
    except Exception:
        parent_route = 'unsupported_or_scalar_capacity_parent'
    return {'route': entry['route'], 'selected_route': entry['route'], 'selected_entrypoint': entry['entrypoint'], 'source_entrypoint': entry['source_entrypoint'], 'parent_route': parent_route, 'replaced_route': parent_route, 'route_kind': 'specialized', 'route_source': 'shape-specific-seed', 'coverage_class': entry['coverage_class'], 'classification': 'seed-produced', 'coverage_only': False, 'production_policy': 'weave_only', 'external_fallback': None, 'guard_order': _guard_order(), 'guard_id': entry['shape_key'], 'selected_guard': entry['guard'], 'guard_condition': entry['guard'], 'fallback': parent_route, 'missing_weave_route': False, 'forced_fallback': False, 'selected_seed': entry['selected_seed'], 'selected_seed_task': entry['selected_seed'], 'producer_seed': entry['producer_seed'], 'producer_seed_task': entry['producer_seed'], 'source_round_doc': entry['source_round_doc'], 'arch_requirement': entry['arch_requirement']}

def route_info(inputs: dict[str, Any]) -> dict[str, Any]:
    entry = _active_entry(inputs)
    if entry is not None:
        return _entry_info(inputs, entry)
    info = dict(parent.route_info(inputs))
    info['guard_order'] = _guard_order()
    info.setdefault('production_policy', 'weave_only')
    info.setdefault('external_fallback', None)
    return info

def route_trace_entry(label: str, inputs: dict[str, Any]) -> dict[str, Any]:
    return {'label': label, 'shape_key': label, 'B': int(inputs['B']), 'Q': int(inputs['Q']), 'M': int(inputs['M']), 'D': int(inputs['D']), 'K': int(inputs['K']), 'self_search': bool(inputs.get('self_search', False)), 'force_fallback': bool(inputs.get('force_fallback', False)), **route_info(inputs)}

def launch_for_eval(inputs: dict[str, Any]) -> dict[str, Any]:
    if _active_entry(inputs) is not None:
        return d4096_seed._launch_d4096_q4q8_tcgen05(inputs)
    return parent.launch_for_eval(inputs)

def knn_search_compile_and_launch_target0627_d4096_q16_m8192_k10_4bc1(*, benchmark: bool=True) -> dict[str, Any]:
    from .._dispatch_runtime import evaluate
    result = evaluate(launch_for_eval, shapes=TARGET_SHAPES, benchmark=benchmark)
    result['passed'] = bool(result.get('summary', {}).get('all_correct'))
    print(result)
    return result
