"""D256/Q128/M262144/K64 exact-shape tcgen05 kNN route.

Minimum target architecture: sm_100a for the tcgen05 producer path. This
additive candidate owns only ``B=1,Q=128,M=262144,D=256,K=64`` and delegates
all other shapes to the current cd65 dispatcher. It reuses the source-clean
round-34 D256 tcgen05 producer/merge and fixes split-M at 128 for the expanded
M262144 target-D floor14 row.
"""
from __future__ import annotations
from json import loads as _json_loads
from .._dispatch_runtime import _capture_cuTensorMapEncodeTiled, _decode_capture, _import_dispatch_module, _ir_proxy
import math
from functools import lru_cache
from types import ModuleType
from typing import Any
from . import knn_search_lowd_d256_mma64_over48e9_0612_r34_48e9_v1 as d256_mma
from .._dispatch_runtime import select_named_shapes
THREADS = d256_mma.THREADS
MERGE_THREADS = d256_mma.MERGE_THREADS
BLOCK_Q = d256_mma.BLOCK_Q
BLOCK_M = d256_mma.BLOCK_M
K_MAX = d256_mma.K_MAX
D_STATIC = d256_mma.D_STATIC
SPLIT_M = 128
POST_MMA_COL_COHORTS = d256_mma.MMA_POST_MMA_COL_COHORTS
MMA_SMEM_BYTES = d256_mma.MMA_SMEM_BYTES
MERGE_SMEM_BYTES = d256_mma.MERGE_SMEM_BYTES
partial_ir = _decode_capture(_json_loads('{"__ir__": "knn_search_mma_split_partial_v1", "arg_keys": ["queries", "database", "partial_distances", "partial_indices", "B", "Q", "M", "split_m", "num_q_tiles", "total_m_tiles", "tiles_per_split"], "cluster_dims": [1, 1, 1], "computed_smem_bytes": 143104, "constants": [["K_MAX_", 64], ["EXPOSE_COL_COHORTS", 1]], "cta_group": 1, "threads": 256}'))
merge_ir = _decode_capture(_json_loads('{"__ir__": "knn_search_mma_split_merge_v1", "arg_keys": ["partial_distances", "partial_indices", "out_distances", "out_indices", "B", "Q", "K", "split_m", "num_q_tiles"], "cluster_dims": [1, 1, 1], "computed_smem_bytes": 0, "constants": [["K_MAX_", 64]], "cta_group": 1, "threads": 32}'))
ir = _decode_capture(_json_loads('{"__ir__": "knn_search_mma_split_partial_v1", "arg_keys": ["queries", "database", "partial_distances", "partial_indices", "B", "Q", "M", "split_m", "num_q_tiles", "total_m_tiles", "tiles_per_split"], "cluster_dims": [1, 1, 1], "computed_smem_bytes": 143104, "constants": [["K_MAX_", 64], ["EXPOSE_COL_COHORTS", 1]], "cta_group": 1, "threads": 256}'))
ROUTE_TARGET0628_D256_Q128_K64 = 'target0628_d256_q128_m262144_k64_split128_tcgen05_dbaf'
SELECTED_SEED = 'weave-evolve-knn-search-dbaf-d256-q128-k64'
ENTRYPOINT = 'loom.examples.weave.knn_search_target0628_d256_q128_m262144_k64_dbaf_v1:launch_for_eval'
TARGET_LABELS: tuple[str, ...] = ('target0627_d256_q128_m262144_k64',)
TARGET_SHAPES = _decode_capture(_json_loads('[{"__dict_items__": [["label", "target0627_d256_q128_m262144_k64"], ["params", {"__dict_items__": [["B", 1], ["Q", 128], ["M", 262144], ["D", 256], ["K", 64], ["dtype", "bfloat16"], ["seed", 612106], ["self_search", false], ["min_recall", 0.999]]}]]}]'))
SHAPE_DISPATCH_REGISTRY: tuple[dict[str, Any], ...] = ({'shape_key': 'target0628_d256_q128_k64', 'label': 'target0627_d256_q128_m262144_k64', 'labels': TARGET_LABELS, 'shape': (1, 128, 262144, 256, 64, False), 'guard': 'B == 1 and Q == 128 and M == 262144 and D == 256 and K == 64 and not self_search and not forced_fallback and arch in {sm_100a,sm_103a}', 'route': ROUTE_TARGET0628_D256_Q128_K64, 'entrypoint': ENTRYPOINT, 'source_entrypoint': 'loom.examples.weave.knn_search_lowd_d256_mma64_over48e9_0612_r34_48e9_v1:partial/merge', 'selected_seed': SELECTED_SEED, 'producer_seed': 'weave-evolve-knn-search-round34-d256-k64', 'source_round_doc': 'design_doc/active/weave_evolve_knn_search_round_34_48e9_lowd_d256_mma64.md', 'coverage_class': 'bucket_seed_target0628_d256_q128_m262144_k64', 'arch_requirement': 'sm_100a', 'split_m': SPLIT_M, 'partial_list_count': SPLIT_M * POST_MMA_COL_COHORTS, 'padding_tag': 'none', 'workspace_reuse': 'cached torch scratch keyed by exact shape/device/dtype'},)
_KERNELS: dict[str, Any] = {}
_SCRATCH: dict[tuple[int, int, int, int, int, int, int, str], tuple[Any, Any]] = {}

@lru_cache(maxsize=1)
def _parent() -> ModuleType:
    from . import knn_search_dispatch0624_full133_eacf_lowd_cd65_v1
    return knn_search_dispatch0624_full133_eacf_lowd_cd65_v1

def _shape_key(inputs: dict[str, Any]) -> tuple[int, int, int, int, int, bool]:
    return (int(inputs.get('B', 1)), int(inputs['Q']), int(inputs['M']), int(inputs['D']), int(inputs['K']), bool(inputs.get('self_search', False)))

def _forced_fallback(inputs: dict[str, Any]) -> bool:
    return bool(inputs.get('force_fallback', False))

def _tcgen05_capable_arch() -> bool:
    return d256_mma._tcgen05_capable_arch()

def _use_target(inputs: dict[str, Any]) -> bool:
    return not _forced_fallback(inputs) and _tcgen05_capable_arch() and (_shape_key(inputs) == SHAPE_DISPATCH_REGISTRY[0]['shape'])

def _guard_order() -> list[str]:
    parent = _parent()
    parent_order: list[str] = []
    if hasattr(parent, '_guard_order'):
        try:
            parent_order = list(parent._guard_order())
        except TypeError:
            parent_order = list(parent._guard_order(parent.PROFILE_ALL))
    return [str(SHAPE_DISPATCH_REGISTRY[0]['shape_key']), *parent_order]

def selected_route(inputs: dict[str, Any]) -> str:
    if _use_target(inputs):
        return ROUTE_TARGET0628_D256_Q128_K64
    return _parent().selected_route(inputs)

def selected_route_name(inputs: dict[str, Any]) -> str:
    return selected_route(inputs)

def route_info(inputs: dict[str, Any]) -> dict[str, Any]:
    parent = _parent()
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
    return _decode_capture(_json_loads('{"merge": {"__kernel__": "dispatch_kernel_0105"}, "partial": {"__kernel__": "dispatch_kernel_0031"}}'))

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
    return _parent().launch_for_eval(inputs)

def knn_search_compile_and_launch_target0628_d256_q128_m262144_k64(*, benchmark: bool=True, shapes: list[dict[str, Any]] | None=None) -> dict[str, Any]:
    from .._dispatch_runtime import evaluate
    result = evaluate(launch_for_eval, shapes=TARGET_SHAPES if shapes is None else shapes, benchmark=benchmark)
    result['passed'] = bool(result.get('summary', {}).get('all_correct'))
    print(result)
    return result
