"""D256/Q1024/M65536/K10 exact-shape tcgen05 kNN route.

Minimum target architecture: sm_100a for the tcgen05 producer path. This
additive candidate owns only ``B=1,Q=1024,M=65536,D=256,K=10`` and delegates
all other shapes to the current target-D dispatcher. It reuses the round-35
D256 K10-capacity tcgen05 producer/merge and retunes split-M for the
target0627 floor14 D256/Q1024 row.
"""
from __future__ import annotations
from json import loads as _json_loads
from .._dispatch_runtime import _capture_cuTensorMapEncodeTiled, _decode_capture, _import_dispatch_module, _ir_proxy
import math
from typing import Any
from .._dispatch_runtime import select_named_shapes
from . import knn_search_dispatch0629_cd65_plus_104b_4b52_afe1_6c60_29d8_v1 as parent
from . import knn_search_lowd_d256_mma64k10_over48e9_0612_r35_48e9_v1 as d256_k10
THREADS = d256_k10.THREADS
MERGE_THREADS = d256_k10.MERGE_THREADS
BLOCK_Q = d256_k10.BLOCK_Q
BLOCK_M = d256_k10.BLOCK_M
D_STATIC = d256_k10.D_STATIC
K_MAX = d256_k10.D256_K10_MAX
SPLIT_M = 64
POST_MMA_COL_COHORTS = d256_k10.MMA_POST_MMA_COL_COHORTS
MMA_SMEM_BYTES = d256_k10.MMA_SMEM_BYTES
MERGE_SMEM_BYTES = d256_k10.MERGE_SMEM_BYTES
partial_ir = _decode_capture(_json_loads('{"__ir__": "knn_search_mma_split_partial_v1", "arg_keys": ["queries", "database", "partial_distances", "partial_indices", "B", "Q", "M", "split_m", "num_q_tiles", "total_m_tiles", "tiles_per_split"], "cluster_dims": [1, 1, 1], "computed_smem_bytes": 143104, "constants": [["K_MAX_", 10], ["EXPOSE_COL_COHORTS", 1]], "cta_group": 1, "threads": 256}'))
merge_ir = _decode_capture(_json_loads('{"__ir__": "knn_search_mma_split_merge_v1", "arg_keys": ["partial_distances", "partial_indices", "out_distances", "out_indices", "B", "Q", "K", "split_m", "num_q_tiles"], "cluster_dims": [1, 1, 1], "computed_smem_bytes": 0, "constants": [["K_MAX_", 10]], "cta_group": 1, "threads": 32}'))
ir = _decode_capture(_json_loads('{"__ir__": "knn_search_mma_split_partial_v1", "arg_keys": ["queries", "database", "partial_distances", "partial_indices", "B", "Q", "M", "split_m", "num_q_tiles", "total_m_tiles", "tiles_per_split"], "cluster_dims": [1, 1, 1], "computed_smem_bytes": 143104, "constants": [["K_MAX_", 10], ["EXPOSE_COL_COHORTS", 1]], "cta_group": 1, "threads": 256}'))
ROUTE_TARGET0628_D256_Q1024_K10 = 'target0628_d256_q1024_m65536_k10_split64_tcgen05_2165'
SELECTED_SEED = 'weave-evolve-knn-search-2165-d256-q1024-k10'
ENTRYPOINT = 'loom.examples.weave.knn_search_target0628_d256_q1024_m65536_k10_2165_v1:launch_for_eval'
TARGET_LABELS: tuple[str, ...] = ('target0627_d256_q1024_m65536_k10',)
TARGET_SHAPES = _decode_capture(_json_loads('[{"__dict_items__": [["label", "target0627_d256_q1024_m65536_k10"], ["params", {"__dict_items__": [["B", 1], ["Q", 1024], ["M", 65536], ["D", 256], ["K", 10], ["dtype", "bfloat16"], ["seed", 612105], ["self_search", false], ["min_recall", 0.999]]}]]}]'))
SHAPE_DISPATCH_REGISTRY: tuple[dict[str, Any], ...] = ({'shape_key': 'target0628_d256_q1024_k10', 'label': 'target0627_d256_q1024_m65536_k10', 'labels': TARGET_LABELS, 'shape': (1, 1024, 65536, 256, 10, False), 'guard': 'B == 1 and Q == 1024 and M == 65536 and D == 256 and K == 10 and not self_search and not forced_fallback and arch in {sm_100a,sm_103a}', 'route': ROUTE_TARGET0628_D256_Q1024_K10, 'entrypoint': ENTRYPOINT, 'source_entrypoint': 'loom.examples.weave.knn_search_lowd_d256_mma64k10_over48e9_0612_r35_48e9_v1:partial/merge', 'selected_seed': SELECTED_SEED, 'producer_seed': 'weave-evolve-knn-search-round35-d256-k10cap', 'source_round_doc': 'design_doc/active/weave_evolve_knn_search_round_35_48e9_d256_k10cap.md', 'coverage_class': 'bucket_seed_target0628_d256_q1024_m65536_k10', 'arch_requirement': 'sm_100a', 'split_m': SPLIT_M, 'partial_list_count': SPLIT_M * POST_MMA_COL_COHORTS, 'padding_tag': 'none', 'workspace_reuse': 'cached torch scratch keyed by exact shape/device/dtype'},)
_KERNELS: dict[str, Any] = {}
_SCRATCH: dict[tuple[int, int, int, int, int, int, int, str], tuple[Any, Any]] = {}

def _shape_key(inputs: dict[str, Any]) -> tuple[int, int, int, int, int, bool]:
    return (int(inputs.get('B', 1)), int(inputs['Q']), int(inputs['M']), int(inputs['D']), int(inputs['K']), bool(inputs.get('self_search', False)))

def _forced_fallback(inputs: dict[str, Any]) -> bool:
    return bool(inputs.get('force_fallback', False))

def _tcgen05_capable_arch() -> bool:
    return d256_k10.parent._tcgen05_capable_arch()

def _use_target(inputs: dict[str, Any]) -> bool:
    return not _forced_fallback(inputs) and _tcgen05_capable_arch() and (_shape_key(inputs) == SHAPE_DISPATCH_REGISTRY[0]['shape'])

def _guard_order() -> list[str]:
    parent_order: list[str] = []
    if hasattr(parent, '_guard_order'):
        try:
            parent_order = list(parent._guard_order())
        except TypeError:
            parent_order = list(parent._guard_order(parent.PROFILE_ALL))
    return [str(SHAPE_DISPATCH_REGISTRY[0]['shape_key']), *parent_order]

def selected_route(inputs: dict[str, Any]) -> str:
    if _use_target(inputs):
        return ROUTE_TARGET0628_D256_Q1024_K10
    return parent.selected_route(inputs)

def selected_route_name(inputs: dict[str, Any]) -> str:
    return selected_route(inputs)

def route_info(inputs: dict[str, Any]) -> dict[str, Any]:
    if not _use_target(inputs):
        info = dict(parent.route_info(inputs))
        info['guard_order'] = _guard_order()
        return info
    entry = SHAPE_DISPATCH_REGISTRY[0]
    parent_info = dict(parent.route_info(inputs))
    parent_route = str(parent_info.get('selected_route') or parent.selected_route(inputs))
    return {'route': entry['route'], 'selected_route': entry['route'], 'selected_entrypoint': entry['entrypoint'], 'source_entrypoint': entry['source_entrypoint'], 'parent_route': parent_route, 'replaced_route': parent_route, 'route_kind': 'specialized', 'route_source': 'shape-specific-seed', 'coverage_class': entry['coverage_class'], 'classification': 'seed-produced', 'coverage_only': False, 'production_policy': 'weave_only', 'external_fallback': None, 'guard_order': _guard_order(), 'guard_id': entry['shape_key'], 'selected_guard': entry['guard'], 'guard_condition': entry['guard'], 'fallback': parent_route, 'missing_weave_route': False, 'forced_fallback': False, 'selected_seed': entry['selected_seed'], 'selected_seed_task': entry['selected_seed'], 'producer_seed': entry['producer_seed'], 'producer_seed_task': entry['producer_seed'], 'source_round_doc': entry['source_round_doc'], 'arch_requirement': entry['arch_requirement'], 'split_m': entry['split_m'], 'partial_list_count': entry['partial_list_count'], 'padding_tag': entry['padding_tag'], 'uses_materialized_padding': False, 'uses_kernel_padding': False, 'padding_overhead_timed': False, 'workspace_reuse': entry['workspace_reuse']}

def route_trace_entry(label: str, inputs: dict[str, Any]) -> dict[str, Any]:
    return {'label': label, 'shape_key': label, 'B': int(inputs['B']), 'Q': int(inputs['Q']), 'M': int(inputs['M']), 'D': int(inputs['D']), 'K': int(inputs['K']), 'self_search': bool(inputs.get('self_search', False)), 'force_fallback': bool(inputs.get('force_fallback', False)), **route_info(inputs)}

def _compile_kernels() -> dict[str, Any]:
    return _decode_capture(_json_loads('{"merge": {"__kernel__": "dispatch_kernel_0567"}, "partial": {"__kernel__": "dispatch_kernel_0566"}}'))

def _ensure_kernels() -> dict[str, Any]:
    if not _KERNELS:
        _KERNELS.update(_compile_kernels())
    return _KERNELS

def _scratch(inputs: dict[str, Any], partial_list_count: int, num_q_tiles: int) -> tuple[Any, Any]:
    import torch
    key = (int(inputs['B']), int(inputs['Q']), int(inputs['M']), int(partial_list_count), int(num_q_tiles), K_MAX, int(inputs['queries'].device.index or 0), str(inputs['queries'].dtype))
    cached = _SCRATCH.get(key)
    if cached is None:
        shape = (int(inputs['B']), int(num_q_tiles), int(partial_list_count), BLOCK_Q, K_MAX)
        cached = (torch.empty(shape, dtype=torch.float32, device=inputs['queries'].device), torch.empty(shape, dtype=torch.int32, device=inputs['queries'].device))
        _SCRATCH[key] = cached
    return cached

def _launch_target(inputs: dict[str, Any]) -> dict[str, Any]:
    kernels = _ensure_kernels()
    bsz = int(inputs['B'])
    q_rows = int(inputs['Q'])
    m_rows = int(inputs['M'])
    k = int(inputs['K'])
    num_q_tiles = math.ceil(q_rows / BLOCK_Q)
    total_m_tiles = math.ceil(m_rows / BLOCK_M)
    split_m = min(SPLIT_M, total_m_tiles)
    tiles_per_split = math.ceil(total_m_tiles / split_m)
    partial_list_count = split_m * POST_MMA_COL_COHORTS
    partial_dist, partial_idx = _scratch(inputs, partial_list_count, num_q_tiles)
    kernels['partial'].launch(grid=(bsz * num_q_tiles * split_m, 1, 1), block=(THREADS, 1, 1), args=[inputs['queries'], inputs['database'], partial_dist, partial_idx, bsz, q_rows, m_rows, split_m, num_q_tiles, total_m_tiles, tiles_per_split], shared_mem=MMA_SMEM_BYTES)
    kernels['merge'].launch(grid=(bsz * q_rows, 1, 1), block=(MERGE_THREADS, 1, 1), args=[partial_dist, partial_idx, inputs['out_distances'], inputs['out_indices'], bsz, q_rows, k, partial_list_count, num_q_tiles], shared_mem=MERGE_SMEM_BYTES)
    return {'distances': inputs['out_distances'], 'indices': inputs['out_indices']}

def launch_for_eval(inputs: dict[str, Any]) -> dict[str, Any]:
    if _use_target(inputs):
        return _launch_target(inputs)
    return parent.launch_for_eval(inputs)

def knn_search_compile_and_launch_target0628_d256_q1024_m65536_k10(*, benchmark: bool=True, shapes: list[dict[str, Any]] | None=None) -> dict[str, Any]:
    from .._dispatch_runtime import evaluate
    result = evaluate(launch_for_eval, shapes=TARGET_SHAPES if shapes is None else shapes, benchmark=benchmark)
    result['passed'] = bool(result.get('summary', {}).get('all_correct'))
    print(result)
    return result
