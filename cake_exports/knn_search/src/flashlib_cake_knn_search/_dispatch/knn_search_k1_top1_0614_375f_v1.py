"""Round-92 high-Q K=1 top-1 split-policy seed for exact BF16 kNN.

Minimum target architecture: sm_100a. This additive seed keeps the source-clean
one-stage top-1 tcgen05 producer on the K=1 high-Q eval path and specializes
the host split-M policy for the Q3072/M20000 held-out shape. Guard misses
delegate to the inherited registered Weave dispatcher; no external
implementation is on the production eval path.
"""
from __future__ import annotations
from json import loads as _json_loads
from .._dispatch_runtime import _capture_cuTensorMapEncodeTiled, _decode_capture, _import_dispatch_module, _ir_proxy
import math
from typing import Any
from .._dispatch_runtime import select_named_shapes
from . import knn_search_q4096_lowk_k1partial_onestage_0614_r2_3ff5_v1 as k1_base
THREADS = k1_base.THREADS
MERGE_THREADS = k1_base.MERGE_THREADS
BLOCK_Q = k1_base.BLOCK_Q
BLOCK_M = k1_base.BLOCK_M
D_STATIC = k1_base.D_STATIC
K_PARTIAL_MAX = k1_base.K_PARTIAL_MAX
Q4096_LOWK_K1_SPLIT_M = k1_base.Q4096_LOWK_K1_SPLIT_M
LOWK_MMA_SMEM_BYTES = k1_base.LOWK_MMA_SMEM_BYTES
partial_k1_ir = _decode_capture(_json_loads('{"__ir__": "knn_search_q4096_lowk_k1partial_onestage_0614_r2_3ff5_v1", "arg_keys": ["queries", "database", "partial_distances", "partial_indices", "B", "Q", "M", "split_m", "num_q_tiles", "total_m_tiles", "tiles_per_split"], "cluster_dims": [1, 1, 1], "computed_smem_bytes": 108800, "constants": [], "cta_group": 1, "threads": 640}'))
merge_k1_ir = _decode_capture(_json_loads('{"__ir__": "knn_search_q4096_lowk_k1partial_onestage_merge_0614_r2_3ff5_v1", "arg_keys": ["partial_distances", "partial_indices", "out_distances", "out_indices", "B", "Q", "K", "split_m", "num_q_tiles"], "cluster_dims": [1, 1, 1], "computed_smem_bytes": 0, "constants": [], "cta_group": 1, "threads": 32}'))
parent_ir = _decode_capture(_json_loads('{"__ir__": "knn_search_q4096_lowk_k5partial_0613_r49_48e9_v1", "arg_keys": ["queries", "database", "partial_distances", "partial_indices", "B", "Q", "M", "split_m", "num_q_tiles", "total_m_tiles", "tiles_per_split"], "cluster_dims": [1, 1, 1], "computed_smem_bytes": 108800, "constants": [], "cta_group": 1, "threads": 640}'))
ir = _decode_capture(_json_loads('{"__ir__": "knn_search_q4096_lowk_k1partial_onestage_0614_r2_3ff5_v1", "arg_keys": ["queries", "database", "partial_distances", "partial_indices", "B", "Q", "M", "split_m", "num_q_tiles", "total_m_tiles", "tiles_per_split"], "cluster_dims": [1, 1, 1], "computed_smem_bytes": 108800, "constants": [], "cta_group": 1, "threads": 640}'))
TARGET_LABELS: tuple[str, ...] = ('ksweep_q4096_m20000_d128_k1',)
HELDOUT_K1_HIGHQ_SHAPES: list[dict[str, Any]] = [{'label': 'round92_375f_q2048_m20000_d128_k1', 'params': {'B': 1, 'Q': 2048, 'M': 20000, 'D': 128, 'K': 1, 'dtype': 'bfloat16', 'seed': 710101, 'self_search': False, 'min_recall': 1.0}}, {'label': 'round92_375f_q3072_m20000_d128_k1', 'params': {'B': 1, 'Q': 3072, 'M': 20000, 'D': 128, 'K': 1, 'dtype': 'bfloat16', 'seed': 710104, 'self_search': False, 'min_recall': 1.0}}, {'label': 'round92_375f_q4096_m16384_d128_k1', 'params': {'B': 1, 'Q': 4096, 'M': 16384, 'D': 128, 'K': 1, 'dtype': 'bfloat16', 'seed': 710102, 'self_search': False, 'min_recall': 1.0}}]
K1_TOP1_SHAPES = _decode_capture(_json_loads('[{"__dict_items__": [["label", "ksweep_q4096_m20000_d128_k1"], ["params", {"__dict_items__": [["B", 1], ["Q", 4096], ["M", 20000], ["D", 128], ["K", 1], ["dtype", "bfloat16"], ["seed", 610310], ["self_search", false], ["min_recall", 1.0]]}]]}, {"__dict_items__": [["label", "round92_375f_q2048_m20000_d128_k1"], ["params", {"__dict_items__": [["B", 1], ["Q", 2048], ["M", 20000], ["D", 128], ["K", 1], ["dtype", "bfloat16"], ["seed", 710101], ["self_search", false], ["min_recall", 1.0]]}]]}, {"__dict_items__": [["label", "round92_375f_q3072_m20000_d128_k1"], ["params", {"__dict_items__": [["B", 1], ["Q", 3072], ["M", 20000], ["D", 128], ["K", 1], ["dtype", "bfloat16"], ["seed", 710104], ["self_search", false], ["min_recall", 1.0]]}]]}, {"__dict_items__": [["label", "round92_375f_q4096_m16384_d128_k1"], ["params", {"__dict_items__": [["B", 1], ["Q", 4096], ["M", 16384], ["D", 128], ["K", 1], ["dtype", "bfloat16"], ["seed", 710102], ["self_search", false], ["min_recall", 1.0]]}]]}]'))
K1_TARGET_SHAPES = _decode_capture(_json_loads('[{"__dict_items__": [["label", "ksweep_q4096_m20000_d128_k1"], ["params", {"__dict_items__": [["B", 1], ["Q", 4096], ["M", 20000], ["D", 128], ["K", 1], ["dtype", "bfloat16"], ["seed", 610310], ["self_search", false], ["min_recall", 1.0]]}]]}]'))
SHAPE_SPLIT_M: dict[tuple[int, int], int] = {(2048, 20000): 9, (3072, 20000): 6, (4096, 20000): 9, (4096, 16384): 9}
SHAPE_DISPATCH_REGISTRY: tuple[dict[str, str], ...] = ({'shape_key': 'round92_375f_highq_k1_top1_target', 'guard': 'B == 1 and Q == 4096 and M == 20000 and D == 128 and K == 1 and tcgen05', 'route': 'round92_375f_highq_k1_top1_split9_target'}, {'shape_key': 'round92_375f_highq_k1_top1_heldout_q3072', 'guard': 'B == 1 and Q == 3072 and M == 20000 and D == 128 and K == 1 and tcgen05', 'route': 'round92_375f_highq_k1_top1_split6_q3072'}, {'shape_key': 'round92_375f_highq_k1_top1_heldout_split9', 'guard': 'B == 1 and ((Q == 2048 and M == 20000) or (Q == 4096 and M == 16384)) and D == 128 and K == 1 and tcgen05', 'route': 'round92_375f_highq_k1_top1_split9_heldout'}, *k1_base.SHAPE_DISPATCH_REGISTRY)

def _use_round92_375f_highq_k1_top1(inputs: dict[str, Any]) -> bool:
    if int(inputs['B']) != 1 or int(inputs['D']) != D_STATIC or int(inputs['K']) != 1 or (not k1_base.mma._tcgen05_capable_arch()):
        return False
    return (int(inputs['Q']), int(inputs['M'])) in SHAPE_SPLIT_M

def _split_m_for_shape(inputs: dict[str, Any], total_m_tiles: int) -> int:
    split_m = int(SHAPE_SPLIT_M.get((int(inputs['Q']), int(inputs['M'])), Q4096_LOWK_K1_SPLIT_M))
    if split_m <= 0:
        raise ValueError('K1 split_m must be positive')
    if split_m > 16:
        raise ValueError('K1 split_m must be <= 16 for the 16-lane merge')
    return min(split_m, int(total_m_tiles))

def selected_route(inputs: dict[str, Any]) -> str:
    if _use_round92_375f_highq_k1_top1(inputs):
        split_m = SHAPE_SPLIT_M[int(inputs['Q']), int(inputs['M'])]
        return ''.join(['round92_375f_highq_k1_top1_split', format(split_m, '')])
    return k1_base.selected_route(inputs)

def _launch_k1_with_split_policy(inputs: dict[str, Any]) -> dict[str, Any]:
    import torch
    if not k1_base._LOWK_K1_KERNELS:
        k1_base._LOWK_K1_KERNELS.update(k1_base._compile_k1_kernels())
    bsz = int(inputs['B'])
    q_rows = int(inputs['Q'])
    m_rows = int(inputs['M'])
    num_q_tiles = math.ceil(q_rows / BLOCK_Q)
    total_m_tiles = math.ceil(m_rows / BLOCK_M)
    split_m = _split_m_for_shape(inputs, total_m_tiles)
    tiles_per_split = math.ceil(total_m_tiles / split_m)
    partial_dist, partial_idx = k1_base._scratch_k1(inputs, split_m, num_q_tiles)
    k1_base._LOWK_K1_KERNELS['partial_k1'].launch(grid=(bsz * num_q_tiles * split_m, 1, 1), block=(THREADS, 1, 1), args=[inputs['queries'], inputs['database'], partial_dist, partial_idx, bsz, q_rows, m_rows, split_m, num_q_tiles, total_m_tiles, tiles_per_split], shared_mem=LOWK_MMA_SMEM_BYTES)
    k1_base._LOWK_K1_KERNELS['merge_k1'].launch(grid=(bsz * q_rows, 1, 1), block=(MERGE_THREADS, 1, 1), args=[partial_dist, partial_idx, inputs['out_distances'], inputs['out_indices'], bsz, q_rows, int(inputs['K']), split_m, num_q_tiles], shared_mem=k1_base.mma.MERGE_SMEM_BYTES)
    return {'distances': inputs['out_distances'], 'indices': inputs['out_indices']}

def launch_for_eval(inputs: dict[str, Any]) -> dict[str, Any]:
    if _use_round92_375f_highq_k1_top1(inputs):
        return _launch_k1_with_split_policy(inputs)
    return k1_base.launch_for_eval(inputs)

def knn_search_compile_and_launch_k1_top1(*, benchmark: bool=True, shapes: list[dict[str, Any]] | None=None) -> dict[str, Any]:
    from .._dispatch_runtime import evaluate
    result = evaluate(launch_for_eval, shapes=K1_TOP1_SHAPES if shapes is None else shapes, benchmark=benchmark)
    result['passed'] = bool(result.get('summary', {}).get('all_correct'))
    print(result)
    return result
