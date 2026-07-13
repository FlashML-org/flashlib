"""Round-80a5 high-Q K=1 merge8 bucket seed for exact BF16 kNN.

Minimum target architecture: sm_100a. This additive bucket seed targets only
``B=1, Q=4096, M=20000, D=128, K=1`` and keeps the source-policy-clean
round-93 tcgen05 K=1 partial producer plus merge8 reducer on the eval path.
Guard misses delegate to the inherited 375f Weave K1 seed.
"""
from __future__ import annotations
from json import loads as _json_loads
from .._dispatch_runtime import _capture_cuTensorMapEncodeTiled, _decode_capture, _import_dispatch_module, _ir_proxy
import math
from typing import Any
from .._dispatch_runtime import select_named_shapes
from . import knn_search_k1_top1_0614_375f_v1 as k1_375f
from . import knn_search_k1_top1_margin_0614_r93_merge8_v1 as merge8_seed
THREADS = merge8_seed.THREADS
MERGE_THREADS = merge8_seed.MERGE_THREADS
BLOCK_Q = merge8_seed.BLOCK_Q
BLOCK_M = merge8_seed.BLOCK_M
D_STATIC = merge8_seed.D_STATIC
K_PARTIAL_MAX = merge8_seed.K_PARTIAL_MAX
Q4096_ROWS = merge8_seed.Q4096_ROWS
Q4096_LOWK_M = merge8_seed.Q4096_LOWK_M
TARGET_SPLIT_M = 9
LOWK_MMA_SMEM_BYTES = merge8_seed.LOWK_MMA_SMEM_BYTES
partial_k1_ir = _decode_capture(_json_loads('{"__ir__": "knn_search_q4096_lowk_k1partial_onestage_0614_r2_3ff5_v1", "arg_keys": ["queries", "database", "partial_distances", "partial_indices", "B", "Q", "M", "split_m", "num_q_tiles", "total_m_tiles", "tiles_per_split"], "cluster_dims": [1, 1, 1], "computed_smem_bytes": 108800, "constants": [], "cta_group": 1, "threads": 640}'))
merge_k1_ir = _decode_capture(_json_loads('{"__ir__": "knn_search_k1_top1_margin_0614_r93_merge8_v1", "arg_keys": ["partial_distances", "partial_indices", "out_distances", "out_indices", "B", "Q", "K", "split_m", "num_q_tiles"], "cluster_dims": [1, 1, 1], "computed_smem_bytes": 0, "constants": [], "cta_group": 1, "threads": 128}'))
parent_ir = _decode_capture(_json_loads('{"__ir__": "knn_search_q4096_lowk_k1partial_onestage_0614_r2_3ff5_v1", "arg_keys": ["queries", "database", "partial_distances", "partial_indices", "B", "Q", "M", "split_m", "num_q_tiles", "total_m_tiles", "tiles_per_split"], "cluster_dims": [1, 1, 1], "computed_smem_bytes": 108800, "constants": [], "cta_group": 1, "threads": 640}'))
ir = _decode_capture(_json_loads('{"__ir__": "knn_search_q4096_lowk_k1partial_onestage_0614_r2_3ff5_v1", "arg_keys": ["queries", "database", "partial_distances", "partial_indices", "B", "Q", "M", "split_m", "num_q_tiles", "total_m_tiles", "tiles_per_split"], "cluster_dims": [1, 1, 1], "computed_smem_bytes": 108800, "constants": [], "cta_group": 1, "threads": 640}'))
TARGET_LABELS: tuple[str, ...] = ('ksweep_q4096_m20000_d128_k1',)
TARGET_SHAPES = _decode_capture(_json_loads('[{"__dict_items__": [["label", "ksweep_q4096_m20000_d128_k1"], ["params", {"__dict_items__": [["B", 1], ["Q", 4096], ["M", 20000], ["D", 128], ["K", 1], ["dtype", "bfloat16"], ["seed", 610310], ["self_search", false], ["min_recall", 1.0]]}]]}]'))
K1_80A5_SHAPES = TARGET_SHAPES
ROUTE_K1_MERGE8_80A5 = ''.join(['round80a5_k1_top1_merge8_split', format(TARGET_SPLIT_M, '')])
SHAPE_DISPATCH_REGISTRY: tuple[dict[str, str], ...] = ({'shape_key': 'round80a5_highq_k1_top1_merge8_target', 'guard': 'B == 1 and Q == 4096 and M == 20000 and D == 128 and K == 1 and tcgen05', 'route': ROUTE_K1_MERGE8_80A5}, *k1_375f.SHAPE_DISPATCH_REGISTRY)

def _use_round80a5_merge8_target(inputs: dict[str, Any]) -> bool:
    return int(inputs['B']) == 1 and int(inputs['Q']) == Q4096_ROWS and (int(inputs['M']) == Q4096_LOWK_M) and (int(inputs['D']) == D_STATIC) and (int(inputs['K']) == 1) and merge8_seed.k1_base.mma._tcgen05_capable_arch()

def selected_route(inputs: dict[str, Any]) -> str:
    if _use_round80a5_merge8_target(inputs):
        return ROUTE_K1_MERGE8_80A5
    return k1_375f.selected_route(inputs)

def selected_route_name(inputs: dict[str, Any]) -> str:
    return selected_route(inputs)

def route_info(inputs: dict[str, Any]) -> dict[str, Any]:
    if _use_round80a5_merge8_target(inputs):
        return {'route': ROUTE_K1_MERGE8_80A5, 'selected_route': ROUTE_K1_MERGE8_80A5, 'selected_entrypoint': 'loom.examples.weave.knn_search_k1_top1_merge8_80a5_v1:launch_for_eval', 'parent_route': k1_375f.selected_route(inputs), 'replaced_route': k1_375f.selected_route(inputs), 'route_kind': 'specialized', 'coverage_class': 'performance_route_k1_merge8_80a5', 'coverage_only': False, 'production_policy': 'weave_only', 'external_fallback': None, 'guard_order': [entry['shape_key'] for entry in SHAPE_DISPATCH_REGISTRY], 'forced_fallback': False, 'selected_guard': SHAPE_DISPATCH_REGISTRY[0]['guard'], 'fallback': None}
    parent_info = {'route': k1_375f.selected_route(inputs), 'selected_route': k1_375f.selected_route(inputs), 'route_kind': 'fallback', 'coverage_class': 'inherited_375f_k1_seed', 'coverage_only': False, 'production_policy': 'weave_only', 'external_fallback': None, 'guard_order': [entry['shape_key'] for entry in SHAPE_DISPATCH_REGISTRY], 'forced_fallback': False, 'selected_guard': 'round80a5 target guard miss; delegate to inherited 375f K1 seed', 'fallback': 'round375f_k1_seed'}
    return parent_info

def route_trace_entry(label: str, inputs: dict[str, Any]) -> dict[str, Any]:
    return {'label': label, 'B': int(inputs['B']), 'Q': int(inputs['Q']), 'M': int(inputs['M']), 'D': int(inputs['D']), 'K': int(inputs['K']), **route_info(inputs)}

def _split_m_for_target(total_m_tiles: int) -> int:
    if TARGET_SPLIT_M <= 0:
        raise ValueError('K1 merge8 split_m must be positive')
    if TARGET_SPLIT_M > 16:
        raise ValueError('K1 merge8 split_m must be <= 16')
    return min(TARGET_SPLIT_M, int(total_m_tiles))

def _launch_merge8_target(inputs: dict[str, Any]) -> dict[str, Any]:
    import torch
    if not merge8_seed._K1_MARGIN_KERNELS:
        merge8_seed._K1_MARGIN_KERNELS.update(merge8_seed._compile_k1_margin_kernels())
    bsz = int(inputs['B'])
    q_rows = int(inputs['Q'])
    m_rows = int(inputs['M'])
    num_q_tiles = math.ceil(q_rows / BLOCK_Q)
    total_m_tiles = math.ceil(m_rows / BLOCK_M)
    split_m = _split_m_for_target(total_m_tiles)
    tiles_per_split = math.ceil(total_m_tiles / split_m)
    partial_dist, partial_idx = merge8_seed._scratch_k1(inputs, split_m, num_q_tiles)
    merge8_seed._K1_MARGIN_KERNELS['partial_k1'].launch(grid=(bsz * num_q_tiles * split_m, 1, 1), block=(THREADS, 1, 1), args=[inputs['queries'], inputs['database'], partial_dist, partial_idx, bsz, q_rows, m_rows, split_m, num_q_tiles, total_m_tiles, tiles_per_split], shared_mem=LOWK_MMA_SMEM_BYTES)
    merge8_seed._K1_MARGIN_KERNELS['merge_k1'].launch(grid=(bsz * math.ceil(q_rows / 8), 1, 1), block=(MERGE_THREADS, 1, 1), args=[partial_dist, partial_idx, inputs['out_distances'], inputs['out_indices'], bsz, q_rows, int(inputs['K']), split_m, num_q_tiles], shared_mem=merge8_seed.k1_base.mma.MERGE_SMEM_BYTES)
    return {'distances': inputs['out_distances'], 'indices': inputs['out_indices']}

def launch_for_eval(inputs: dict[str, Any]) -> dict[str, Any]:
    if _use_round80a5_merge8_target(inputs):
        return _launch_merge8_target(inputs)
    return k1_375f.launch_for_eval(inputs)

def knn_search_compile_and_launch_k1_merge8_80a5(*, benchmark: bool=True, shapes: list[dict[str, Any]] | None=None) -> dict[str, Any]:
    from .._dispatch_runtime import evaluate
    result = evaluate(launch_for_eval, shapes=K1_80A5_SHAPES if shapes is None else shapes, benchmark=benchmark)
    result['passed'] = bool(result.get('summary', {}).get('all_correct'))
    print(result)
    return result
