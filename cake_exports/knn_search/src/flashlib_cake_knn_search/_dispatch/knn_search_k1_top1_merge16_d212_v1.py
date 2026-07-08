"""Round-d212 K=1 top-1 merge16 seed for exact BF16 kNN.

Minimum target architecture: sm_100a. This additive bucket seed keeps the
source-clean tcgen05 K=1 partial producer for
``B=1, Q=4096, M=20000, D=128, K=1`` and replaces only the final split merge
with a 256-thread merge kernel that handles sixteen query rows per CTA. Guard
misses delegate to the inherited round-80a5 K1 merge8 seed unchanged.
"""
from __future__ import annotations
from json import loads as _json_loads
from .._dispatch_runtime import _capture_cuTensorMapEncodeTiled, _decode_capture, _import_dispatch_module, _ir_proxy
import math
from typing import Any
from .._dispatch_runtime import select_named_shapes
from . import knn_search_k1_top1_merge8_80a5_v1 as parent_seed
from . import knn_search_q4096_lowk_k1partial_onestage_0614_r2_3ff5_v1 as k1_base
THREADS = parent_seed.THREADS
MERGE_THREADS = 256
MERGE_ROWS_PER_CTA = 16
BLOCK_Q = parent_seed.BLOCK_Q
BLOCK_M = parent_seed.BLOCK_M
D_STATIC = parent_seed.D_STATIC
K_PARTIAL_MAX = parent_seed.K_PARTIAL_MAX
Q4096_ROWS = parent_seed.Q4096_ROWS
Q4096_LOWK_M = parent_seed.Q4096_LOWK_M
TARGET_SPLIT_M = 9
LOWK_MMA_SMEM_BYTES = parent_seed.LOWK_MMA_SMEM_BYTES
partial_k1_ir = _decode_capture(_json_loads('{"__ir__": "knn_search_q4096_lowk_k1partial_onestage_0614_r2_3ff5_v1", "arg_keys": ["queries", "database", "partial_distances", "partial_indices", "B", "Q", "M", "split_m", "num_q_tiles", "total_m_tiles", "tiles_per_split"], "cluster_dims": [1, 1, 1], "computed_smem_bytes": 108800, "constants": [], "cta_group": 1, "threads": 640}'))
parent_ir = _decode_capture(_json_loads('{"__ir__": "knn_search_q4096_lowk_k1partial_onestage_0614_r2_3ff5_v1", "arg_keys": ["queries", "database", "partial_distances", "partial_indices", "B", "Q", "M", "split_m", "num_q_tiles", "total_m_tiles", "tiles_per_split"], "cluster_dims": [1, 1, 1], "computed_smem_bytes": 108800, "constants": [], "cta_group": 1, "threads": 640}'))
ir = _decode_capture(_json_loads('{"__ir__": "knn_search_q4096_lowk_k1partial_onestage_0614_r2_3ff5_v1", "arg_keys": ["queries", "database", "partial_distances", "partial_indices", "B", "Q", "M", "split_m", "num_q_tiles", "total_m_tiles", "tiles_per_split"], "cluster_dims": [1, 1, 1], "computed_smem_bytes": 108800, "constants": [], "cta_group": 1, "threads": 640}'))
_K1_MERGE16_KERNELS: dict[str, Any] = {}
_K1_MERGE16_SCRATCH: dict[tuple[int, int, int, int, int, int, str], tuple[Any, Any]] = {}
knn_search_k1_top1_merge16_d212_v1 = _decode_capture(_json_loads('{"__ir__": "knn_search_k1_top1_merge16_d212_v1", "arg_keys": ["partial_distances", "partial_indices", "out_distances", "out_indices", "B", "Q", "K", "split_m", "num_q_tiles"], "cluster_dims": [1, 1, 1], "computed_smem_bytes": 0, "constants": [], "cta_group": 1, "threads": 256}'))
merge_k1_ir = _decode_capture(_json_loads('{"__ir__": "knn_search_k1_top1_merge16_d212_v1", "arg_keys": ["partial_distances", "partial_indices", "out_distances", "out_indices", "B", "Q", "K", "split_m", "num_q_tiles"], "cluster_dims": [1, 1, 1], "computed_smem_bytes": 0, "constants": [], "cta_group": 1, "threads": 256}'))
TARGET_LABELS: tuple[str, ...] = ('ksweep_q4096_m20000_d128_k1',)
TARGET_SHAPES = _decode_capture(_json_loads('[{"__dict_items__": [["label", "ksweep_q4096_m20000_d128_k1"], ["params", {"__dict_items__": [["B", 1], ["Q", 4096], ["M", 20000], ["D", 128], ["K", 1], ["dtype", "bfloat16"], ["seed", 610310], ["self_search", false], ["min_recall", 1.0]]}]]}]'))
K1_MERGE16_D212_SHAPES = TARGET_SHAPES
ROUTE_K1_MERGE16_D212 = 'round123_d212_k1_top1_merge16_split9'
SHAPE_DISPATCH_REGISTRY: tuple[dict[str, str], ...] = ({'shape_key': 'round123_d212_k1_top1_merge16_target', 'guard': 'B == 1 and Q == 4096 and M == 20000 and D == 128 and K == 1 and not self_search and not forced_fallback and tcgen05_capable_arch', 'route': ROUTE_K1_MERGE16_D212}, *parent_seed.SHAPE_DISPATCH_REGISTRY)

def _use_round123_d212_merge16(inputs: dict[str, Any]) -> bool:
    return int(inputs['B']) == 1 and int(inputs['Q']) == Q4096_ROWS and (int(inputs['M']) == Q4096_LOWK_M) and (int(inputs['D']) == D_STATIC) and (int(inputs['K']) == 1) and (not bool(inputs.get('self_search', False))) and (not bool(inputs.get('force_fallback', False))) and k1_base.mma._tcgen05_capable_arch()

def selected_route(inputs: dict[str, Any]) -> str:
    if _use_round123_d212_merge16(inputs):
        return ROUTE_K1_MERGE16_D212
    return parent_seed.selected_route(inputs)

def selected_route_name(inputs: dict[str, Any]) -> str:
    return selected_route(inputs)

def route_info(inputs: dict[str, Any]) -> dict[str, Any]:
    if _use_round123_d212_merge16(inputs):
        return {'route': ROUTE_K1_MERGE16_D212, 'selected_route': ROUTE_K1_MERGE16_D212, 'selected_entrypoint': 'loom.examples.weave.knn_search_k1_top1_merge16_d212_v1:launch_for_eval', 'parent_route': parent_seed.selected_route(inputs), 'replaced_route': parent_seed.selected_route(inputs), 'route_kind': 'specialized', 'route_source': 'shape-specific-seed', 'coverage_class': 'performance_route_k1_merge16_d212', 'coverage_only': False, 'production_policy': 'weave_only', 'external_fallback': None, 'guard_order': [entry['shape_key'] for entry in SHAPE_DISPATCH_REGISTRY], 'forced_fallback': False, 'selected_guard': SHAPE_DISPATCH_REGISTRY[0]['guard'], 'fallback': None, 'split_m': TARGET_SPLIT_M, 'merge_rows_per_cta': MERGE_ROWS_PER_CTA, 'selected_seed': 'd212_k1_top1_merge16'}
    info = dict(parent_seed.route_info(inputs))
    info['guard_order'] = [entry['shape_key'] for entry in SHAPE_DISPATCH_REGISTRY]
    return info

def route_trace_entry(label: str, inputs: dict[str, Any]) -> dict[str, Any]:
    return {'label': label, 'shape_key': label, 'B': int(inputs['B']), 'Q': int(inputs['Q']), 'M': int(inputs['M']), 'D': int(inputs['D']), 'K': int(inputs['K']), 'self_search': bool(inputs.get('self_search', False)), **route_info(inputs)}

def _compile_k1_merge16_kernels() -> dict[str, Any]:
    return _decode_capture(_json_loads('{"merge_k1": {"__kernel__": "dispatch_kernel_0433"}, "partial_k1": {"__kernel__": "dispatch_kernel_0432"}}'))

def _scratch_k1(inputs: dict[str, Any], split_m: int, num_q_tiles: int) -> tuple[Any, Any]:
    import torch
    key = (int(inputs['B']), int(inputs['Q']), int(inputs['M']), int(split_m), int(num_q_tiles), int(inputs['queries'].device.index or 0), str(inputs['queries'].dtype))
    cached = _K1_MERGE16_SCRATCH.get(key)
    if cached is None:
        shape = (int(inputs['B']), int(num_q_tiles), int(split_m), BLOCK_Q, K_PARTIAL_MAX)
        cached = (torch.empty(shape, dtype=torch.float32, device=inputs['queries'].device), torch.empty(shape, dtype=torch.int32, device=inputs['queries'].device))
        _K1_MERGE16_SCRATCH[key] = cached
    return cached

def _split_m_for_target(total_m_tiles: int) -> int:
    return min(TARGET_SPLIT_M, int(total_m_tiles))

def _launch_merge16_target(inputs: dict[str, Any]) -> dict[str, Any]:
    import torch
    if not _K1_MERGE16_KERNELS:
        _K1_MERGE16_KERNELS.update(_compile_k1_merge16_kernels())
    bsz = int(inputs['B'])
    q_rows = int(inputs['Q'])
    m_rows = int(inputs['M'])
    num_q_tiles = math.ceil(q_rows / BLOCK_Q)
    total_m_tiles = math.ceil(m_rows / BLOCK_M)
    split_m = _split_m_for_target(total_m_tiles)
    tiles_per_split = math.ceil(total_m_tiles / split_m)
    partial_dist, partial_idx = _scratch_k1(inputs, split_m, num_q_tiles)
    _K1_MERGE16_KERNELS['partial_k1'].launch(grid=(bsz * num_q_tiles * split_m, 1, 1), block=(THREADS, 1, 1), args=[inputs['queries'], inputs['database'], partial_dist, partial_idx, bsz, q_rows, m_rows, split_m, num_q_tiles, total_m_tiles, tiles_per_split], shared_mem=LOWK_MMA_SMEM_BYTES)
    _K1_MERGE16_KERNELS['merge_k1'].launch(grid=(bsz * math.ceil(q_rows / MERGE_ROWS_PER_CTA), 1, 1), block=(MERGE_THREADS, 1, 1), args=[partial_dist, partial_idx, inputs['out_distances'], inputs['out_indices'], bsz, q_rows, int(inputs['K']), split_m, num_q_tiles], shared_mem=k1_base.mma.MERGE_SMEM_BYTES)
    return {'distances': inputs['out_distances'], 'indices': inputs['out_indices']}

def launch_for_eval(inputs: dict[str, Any]) -> dict[str, Any]:
    if _use_round123_d212_merge16(inputs):
        return _launch_merge16_target(inputs)
    return parent_seed.launch_for_eval(inputs)

def knn_search_compile_and_launch_k1_top1_merge16_d212(*, benchmark: bool=True, shapes: list[dict[str, Any]] | None=None) -> dict[str, Any]:
    from .._dispatch_runtime import evaluate
    result = evaluate(launch_for_eval, shapes=K1_MERGE16_D212_SHAPES if shapes is None else shapes, benchmark=benchmark)
    result['passed'] = bool(result.get('summary', {}).get('all_correct'))
    print(result)
    return result
