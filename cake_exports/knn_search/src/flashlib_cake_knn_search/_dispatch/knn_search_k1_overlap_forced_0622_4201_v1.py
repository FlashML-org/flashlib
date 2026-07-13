"""Round-135/4201 K1 overlap and forced-fallback seed for kNN search.

Minimum target architecture: sm_100a for the tcgen05/TMEM producer paths. This
additive bucket module targets the two remaining K1 floor blockers from the
round-134 dispatcher denominator:

* ``B=1,Q=128,M=32768,D=128,K=1`` uses the existing top-1 tcgen05 producer with
  a K1-only split-148 merge so the producer exposes a full B200 wave.
* ``B=1,Q=4096,M=20000,D=128,K=1,force_fallback=True`` uses the validated
  paired-Q top-1 producer under a distinct exact forced-fallback guard.

Guard misses delegate to the round-134/4201 dispatcher. No external fallback is
on the production eval path.
"""
from __future__ import annotations
from json import loads as _json_loads
from .._dispatch_runtime import _capture_cuTensorMapEncodeTiled, _decode_capture, _import_dispatch_module, _ir_proxy
import math
from typing import Any
from . import knn_search_dispatch0622_dynamicd_prefix4_b2selftail_4201_v1 as parent
from . import knn_search_k1_top1_pairq_0622_598a_v1 as pairq_seed
from . import knn_search_q4096_lowk_k1partial_onestage_0614_r2_3ff5_v1 as k1_base
THREADS = k1_base.THREADS
MERGE_THREADS = k1_base.MERGE_THREADS
BLOCK_Q = k1_base.BLOCK_Q
BLOCK_M = k1_base.BLOCK_M
D_STATIC = k1_base.D_STATIC
K_PARTIAL_MAX = k1_base.K_PARTIAL_MAX
LOWK_MMA_SMEM_BYTES = k1_base.LOWK_MMA_SMEM_BYTES
Q128_ROWS = 128
Q128_MID_M = 32768
Q128_K1_SPLIT_M = 148
Q128_K1_SLOT4_LANES = Q128_K1_SPLIT_M - 4 * MERGE_THREADS
Q128_K1_SPLITS_PER_LANE = 5
Q4096_ROWS = pairq_seed.Q4096_ROWS
Q4096_LOWK_M = pairq_seed.Q4096_LOWK_M
FORCED_PAIRQ_SPLIT_M = pairq_seed.TARGET_SPLIT_M
PAIR_Q_TILES = pairq_seed.PAIR_Q_TILES
MERGE_ROWS_PER_CTA = pairq_seed.MERGE_ROWS_PER_CTA
ROUTE_Q128_K1_SPLIT148 = '4201_q128_m32768_k1_split148_top1'
ROUTE_FORCED_PAIRQ_K1 = '4201_forced_fallback_q4096_m20000_k1_pairq'
CONSUMED_Q128_TOP1_SEED = 'q4096_lowk_k1partial_onestage_3ff5'
CONSUMED_PAIRQ_SEED = '598a_k1_top1_pairq'
TARGET_LABELS: tuple[str, ...] = ('exp_guard_overlap_q128_m32768_d128_k1', 'exp_forced_fallback_k1_pairq_target')
TARGET_SHAPES: list[dict[str, Any]] = [{'label': 'exp_guard_overlap_q128_m32768_d128_k1', 'params': {'B': 1, 'Q': Q128_ROWS, 'M': Q128_MID_M, 'D': D_STATIC, 'K': 1, 'dtype': 'bfloat16', 'self_search': False}}, {'label': 'exp_forced_fallback_k1_pairq_target', 'params': {'B': 1, 'Q': Q4096_ROWS, 'M': Q4096_LOWK_M, 'D': D_STATIC, 'K': 1, 'dtype': 'bfloat16', 'self_search': False, 'force_fallback': True}}]
partial_k1_ir = _decode_capture(_json_loads('{"__ir__": "knn_search_q4096_lowk_k1partial_onestage_0614_r2_3ff5_v1", "arg_keys": ["queries", "database", "partial_distances", "partial_indices", "B", "Q", "M", "split_m", "num_q_tiles", "total_m_tiles", "tiles_per_split"], "cluster_dims": [1, 1, 1], "computed_smem_bytes": 108800, "constants": [], "cta_group": 1, "threads": 640}'))
pairq_partial_ir = _decode_capture(_json_loads('{"__ir__": "knn_search_k1_top1_pairq_partial_0622_598a_v1", "arg_keys": ["queries", "database", "partial_distances", "partial_indices", "B", "Q", "M", "split_m", "num_q_tiles", "total_m_tiles", "tiles_per_split"], "cluster_dims": [1, 1, 1], "computed_smem_bytes": 149760, "constants": [], "cta_group": 1, "threads": 640}'))
pairq_merge_ir = _decode_capture(_json_loads('{"__ir__": "knn_search_k1_top1_merge16_d212_v1", "arg_keys": ["partial_distances", "partial_indices", "out_distances", "out_indices", "B", "Q", "K", "split_m", "num_q_tiles"], "cluster_dims": [1, 1, 1], "computed_smem_bytes": 0, "constants": [], "cta_group": 1, "threads": 256}'))
parent_ir = _decode_capture(_json_loads('{"__ir__": "knn_search_mma_split_partial_v1", "arg_keys": ["queries", "database", "partial_distances", "partial_indices", "B", "Q", "M", "split_m", "num_q_tiles", "total_m_tiles", "tiles_per_split"], "cluster_dims": [1, 1, 1], "computed_smem_bytes": 108800, "constants": [["K_MAX_", 10], ["EXPOSE_COL_COHORTS", 0], ["FULL_M_TILES", 0]], "cta_group": 1, "threads": 640}'))
ir = _decode_capture(_json_loads('{"__ir__": "knn_search_q4096_lowk_k1partial_onestage_0614_r2_3ff5_v1", "arg_keys": ["queries", "database", "partial_distances", "partial_indices", "B", "Q", "M", "split_m", "num_q_tiles", "total_m_tiles", "tiles_per_split"], "cluster_dims": [1, 1, 1], "computed_smem_bytes": 108800, "constants": [], "cta_group": 1, "threads": 640}'))
_K1_OVERLAP_KERNELS: dict[str, Any] = {}
_K1_OVERLAP_SCRATCH: dict[tuple[int, int, int, int, int, int, str], tuple[Any, Any]] = {}
knn_search_k1_q128_split148_merge_0622_4201_v1 = _decode_capture(_json_loads('{"__ir__": "knn_search_k1_q128_split148_merge_0622_4201_v1", "arg_keys": ["partial_distances", "partial_indices", "out_distances", "out_indices", "B", "Q", "K", "split_m", "num_q_tiles"], "cluster_dims": [1, 1, 1], "computed_smem_bytes": 0, "constants": [], "cta_group": 1, "threads": 32}'))
merge_q128_k1_ir = _decode_capture(_json_loads('{"__ir__": "knn_search_k1_q128_split148_merge_0622_4201_v1", "arg_keys": ["partial_distances", "partial_indices", "out_distances", "out_indices", "B", "Q", "K", "split_m", "num_q_tiles"], "cluster_dims": [1, 1, 1], "computed_smem_bytes": 0, "constants": [], "cta_group": 1, "threads": 32}'))
SHAPE_DISPATCH_REGISTRY: tuple[dict[str, Any], ...] = ({'shape_key': '4201_q128_m32768_k1_split148_top1', 'labels': ('exp_guard_overlap_q128_m32768_d128_k1',), 'guard': 'B == 1 and Q == 128 and M == 32768 and D == 128 and K == 1 and not self_search and not forced_fallback and tcgen05_capable_arch', 'route': ROUTE_Q128_K1_SPLIT148, 'entrypoint': 'loom.examples.weave.knn_search_k1_overlap_forced_0622_4201_v1:launch_for_eval', 'source_entrypoint': 'loom.examples.weave.knn_search_q4096_lowk_k1partial_onestage_0614_r2_3ff5_v1:_launch_q4096_k1 plus split148 K1 merge', 'selected_seed': CONSUMED_Q128_TOP1_SEED, 'source_task': 'weave-evolve-knn-search-4201-k1overlap', 'coverage_class': 'performance_route_q128_m32768_k1_split148_top1', 'route_source': 'shape-specific-seed'}, {'shape_key': '4201_forced_fallback_q4096_m20000_k1_pairq', 'labels': ('exp_forced_fallback_k1_pairq_target',), 'guard': 'B == 1 and Q == 4096 and M == 20000 and D == 128 and K == 1 and not self_search and forced_fallback and tcgen05_capable_arch', 'route': ROUTE_FORCED_PAIRQ_K1, 'entrypoint': 'loom.examples.weave.knn_search_k1_overlap_forced_0622_4201_v1:launch_for_eval', 'source_entrypoint': 'loom.examples.weave.knn_search_k1_top1_pairq_0622_598a_v1:_launch_pairq_target', 'selected_seed': CONSUMED_PAIRQ_SEED, 'source_task': 'weave-evolve-knn-search-4201-k1overlap', 'coverage_class': 'performance_route_forced_fallback_q4096_m20000_k1_pairq', 'route_source': 'shape-specific-seed'}, *parent.SHAPE_DISPATCH_REGISTRY)

def _tcgen05_capable_arch() -> bool:
    return bool(k1_base.mma._tcgen05_capable_arch())

def _shape_tuple(inputs: dict[str, Any]) -> tuple[int, int, int, int, int, bool, bool]:
    return (int(inputs.get('B', 1)), int(inputs['Q']), int(inputs['M']), int(inputs['D']), int(inputs['K']), bool(inputs.get('self_search', False)), bool(inputs.get('force_fallback', False)))

def _use_q128_overlap_top1(inputs: dict[str, Any]) -> bool:
    return _shape_tuple(inputs) == (1, Q128_ROWS, Q128_MID_M, D_STATIC, 1, False, False) and _tcgen05_capable_arch()

def _use_forced_pairq_top1(inputs: dict[str, Any]) -> bool:
    return _shape_tuple(inputs) == (1, Q4096_ROWS, Q4096_LOWK_M, D_STATIC, 1, False, True) and _tcgen05_capable_arch()

def _active_entry(inputs: dict[str, Any]) -> dict[str, Any] | None:
    if _use_q128_overlap_top1(inputs):
        return SHAPE_DISPATCH_REGISTRY[0]
    if _use_forced_pairq_top1(inputs):
        return SHAPE_DISPATCH_REGISTRY[1]
    return None

def selected_route(inputs: dict[str, Any]) -> str:
    entry = _active_entry(inputs)
    if entry is not None:
        return str(entry['route'])
    return parent.selected_route(inputs)

def selected_route_name(inputs: dict[str, Any]) -> str:
    return selected_route(inputs)

def _guard_order() -> list[str]:
    return [str(entry['shape_key']) for entry in SHAPE_DISPATCH_REGISTRY]

def _parent_info(inputs: dict[str, Any]) -> dict[str, Any]:
    info = dict(parent.route_info(inputs))
    route = str(info.get('selected_route') or info.get('route') or parent.selected_route(inputs))
    info['route'] = route
    info['selected_route'] = route
    info['guard_order'] = _guard_order()
    info.setdefault('production_policy', 'weave_only')
    info.setdefault('external_fallback', None)
    info.setdefault('coverage_only', False)
    info.setdefault('missing_weave_route', False)
    return info

def _entry_info(inputs: dict[str, Any], entry: dict[str, Any]) -> dict[str, Any]:
    parent_info = dict(parent.route_info(inputs))
    parent_route = str(parent_info.get('selected_route') or parent_info.get('route') or parent.selected_route(inputs))
    q_rows = int(inputs['Q'])
    m_rows = int(inputs['M'])
    total_m_tiles = math.ceil(m_rows / BLOCK_M)
    if entry['route'] == ROUTE_Q128_K1_SPLIT148:
        split_m = min(Q128_K1_SPLIT_M, total_m_tiles)
        num_q_tiles = math.ceil(q_rows / BLOCK_Q)
        tiles_per_split = math.ceil(total_m_tiles / split_m)
        extra = {'split_m': split_m, 'num_q_tiles': num_q_tiles, 'total_m_tiles': total_m_tiles, 'tiles_per_split': tiles_per_split, 'merge_kind': 'q128_k1_split148'}
    else:
        split_m = min(FORCED_PAIRQ_SPLIT_M, total_m_tiles)
        num_q_tiles = math.ceil(q_rows / BLOCK_Q)
        extra = {'split_m': split_m, 'num_q_tiles': num_q_tiles, 'total_m_tiles': total_m_tiles, 'tiles_per_split': math.ceil(total_m_tiles / split_m), 'pair_q_tiles': PAIR_Q_TILES, 'merge_rows_per_cta': MERGE_ROWS_PER_CTA, 'merge_kind': 'pairq_merge16'}
    return {'route': entry['route'], 'selected_route': entry['route'], 'selected_entrypoint': entry['entrypoint'], 'source_entrypoint': entry['source_entrypoint'], 'parent_route': parent_route, 'replaced_route': parent_route, 'route_kind': 'specialized', 'route_source': entry['route_source'], 'coverage_class': entry['coverage_class'], 'classification': 'seed-produced', 'coverage_only': False, 'production_policy': 'weave_only', 'external_fallback': None, 'guard_order': _guard_order(), 'guard_id': entry['shape_key'], 'forced_fallback': bool(inputs.get('force_fallback', False)), 'selected_guard': entry['guard'], 'guard_condition': entry['guard'], 'fallback': parent_route, 'missing_weave_route': False, 'source_task': entry['source_task'], 'selected_seed': entry['selected_seed'], 'selected_seed_task': entry['source_task'], 'expected_seed': entry['selected_seed'], 'producer_seed': entry['selected_seed'], **extra}

def route_info(inputs: dict[str, Any]) -> dict[str, Any]:
    entry = _active_entry(inputs)
    if entry is None:
        return _parent_info(inputs)
    return _entry_info(inputs, entry)

def route_trace_entry(label: str, inputs: dict[str, Any]) -> dict[str, Any]:
    return {'label': label, 'shape_key': label, 'B': int(inputs['B']), 'Q': int(inputs['Q']), 'M': int(inputs['M']), 'D': int(inputs['D']), 'K': int(inputs['K']), 'self_search': bool(inputs.get('self_search', False)), 'force_fallback': bool(inputs.get('force_fallback', False)), **route_info(inputs)}

def _compile_q128_kernels() -> dict[str, Any]:
    return _decode_capture(_json_loads('{"merge_q128_k1": {"__kernel__": "dispatch_kernel_0242"}, "partial_k1": {"__kernel__": "dispatch_kernel_0207"}}'))

def _scratch_q128_k1(inputs: dict[str, Any], split_m: int, num_q_tiles: int) -> tuple[Any, Any]:
    import torch
    key = (int(inputs['B']), int(inputs['Q']), int(inputs['M']), int(split_m), int(num_q_tiles), int(inputs['queries'].device.index or 0), str(inputs['queries'].dtype))
    cached = _K1_OVERLAP_SCRATCH.get(key)
    if cached is None:
        shape = (int(inputs['B']), int(num_q_tiles), int(split_m), BLOCK_Q, K_PARTIAL_MAX)
        cached = (torch.empty(shape, dtype=torch.float32, device=inputs['queries'].device), torch.empty(shape, dtype=torch.int32, device=inputs['queries'].device))
        _K1_OVERLAP_SCRATCH[key] = cached
    return cached

def _launch_q128_overlap_top1(inputs: dict[str, Any]) -> dict[str, Any]:
    import torch
    if not _K1_OVERLAP_KERNELS:
        _K1_OVERLAP_KERNELS.update(_compile_q128_kernels())
    bsz = int(inputs['B'])
    q_rows = int(inputs['Q'])
    m_rows = int(inputs['M'])
    num_q_tiles = math.ceil(q_rows / BLOCK_Q)
    total_m_tiles = math.ceil(m_rows / BLOCK_M)
    split_m = min(Q128_K1_SPLIT_M, total_m_tiles)
    tiles_per_split = math.ceil(total_m_tiles / split_m)
    partial_dist, partial_idx = _scratch_q128_k1(inputs, split_m, num_q_tiles)
    _K1_OVERLAP_KERNELS['partial_k1'].launch(grid=(bsz * num_q_tiles * split_m, 1, 1), block=(THREADS, 1, 1), args=[inputs['queries'], inputs['database'], partial_dist, partial_idx, bsz, q_rows, m_rows, split_m, num_q_tiles, total_m_tiles, tiles_per_split], shared_mem=LOWK_MMA_SMEM_BYTES)
    _K1_OVERLAP_KERNELS['merge_q128_k1'].launch(grid=(bsz * q_rows, 1, 1), block=(MERGE_THREADS, 1, 1), args=[partial_dist, partial_idx, inputs['out_distances'], inputs['out_indices'], bsz, q_rows, int(inputs['K']), split_m, num_q_tiles], shared_mem=k1_base.mma.MERGE_SMEM_BYTES)
    return {'distances': inputs['out_distances'], 'indices': inputs['out_indices']}

def launch_parent_for_eval(inputs: dict[str, Any]) -> dict[str, Any]:
    return parent.launch_for_eval(inputs)

def launch_for_eval(inputs: dict[str, Any]) -> dict[str, Any]:
    if _use_q128_overlap_top1(inputs):
        return _launch_q128_overlap_top1(inputs)
    if _use_forced_pairq_top1(inputs):
        return pairq_seed._launch_pairq_target(inputs)
    return parent.launch_for_eval(inputs)

def knn_search_compile_and_launch_k1_overlap_forced_4201(*, benchmark: bool=True, shapes: list[dict[str, Any]] | None=None) -> dict[str, Any]:
    from .._dispatch_runtime import evaluate
    result = evaluate(launch_for_eval, shapes=TARGET_SHAPES if shapes is None else shapes, benchmark=benchmark)
    result['passed'] = bool(result.get('summary', {}).get('all_correct'))
    print(result)
    return result
