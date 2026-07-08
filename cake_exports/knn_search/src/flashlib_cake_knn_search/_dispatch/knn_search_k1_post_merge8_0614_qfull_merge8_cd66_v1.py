"""Round-cd66 K=1 qfull producer plus merge8 reducer seed for exact BF16 kNN.

Minimum target architecture: sm_100a. This additive bucket seed combines the
round-93 full-query-tile tcgen05 K=1 partial producer with the merge8 split
reducer for high-Q K1 shapes. Guard misses delegate to the inherited 375f
Weave seed; no external implementation is on the eval path.
"""
from __future__ import annotations
from json import loads as _json_loads
from .._dispatch_runtime import _capture_cuTensorMapEncodeTiled, _decode_capture, _import_dispatch_module, _ir_proxy
import math
from typing import Any
from .._dispatch_runtime import select_named_shapes
from . import knn_search_k1_top1_0614_375f_v1 as parent_seed
from . import knn_search_k1_top1_margin_0614_qfull_93_v1 as qfull_seed
from . import knn_search_k1_top1_margin_0614_r93_merge8_v1 as merge8_seed
from .knn_search_stream import current_stream_handle
THREADS = qfull_seed.THREADS
MERGE_THREADS = merge8_seed.MERGE_THREADS
BLOCK_Q = qfull_seed.BLOCK_Q
BLOCK_M = qfull_seed.BLOCK_M
D_STATIC = qfull_seed.D_STATIC
K_PARTIAL_MAX = qfull_seed.K_PARTIAL_MAX
LOWK_MMA_SMEM_BYTES = qfull_seed.LOWK_MMA_SMEM_BYTES
partial_k1_ir = _decode_capture(_json_loads('{"__ir__": "knn_search_k1_top1_margin_qfull_partial_0614_r93_v1", "arg_keys": ["queries", "database", "partial_distances", "partial_indices", "B", "Q", "M", "split_m", "num_q_tiles", "total_m_tiles", "tiles_per_split"], "cluster_dims": [1, 1, 1], "computed_smem_bytes": 108800, "constants": [], "cta_group": 1, "threads": 640}'))
merge_k1_ir = _decode_capture(_json_loads('{"__ir__": "knn_search_k1_top1_margin_0614_r93_merge8_v1", "arg_keys": ["partial_distances", "partial_indices", "out_distances", "out_indices", "B", "Q", "K", "split_m", "num_q_tiles"], "cluster_dims": [1, 1, 1], "computed_smem_bytes": 0, "constants": [], "cta_group": 1, "threads": 128}'))
parent_ir = _decode_capture(_json_loads('{"__ir__": "knn_search_q4096_lowk_k1partial_onestage_0614_r2_3ff5_v1", "arg_keys": ["queries", "database", "partial_distances", "partial_indices", "B", "Q", "M", "split_m", "num_q_tiles", "total_m_tiles", "tiles_per_split"], "cluster_dims": [1, 1, 1], "computed_smem_bytes": 108800, "constants": [], "cta_group": 1, "threads": 640}'))
ir = _decode_capture(_json_loads('{"__ir__": "knn_search_k1_top1_margin_qfull_partial_0614_r93_v1", "arg_keys": ["queries", "database", "partial_distances", "partial_indices", "B", "Q", "M", "split_m", "num_q_tiles", "total_m_tiles", "tiles_per_split"], "cluster_dims": [1, 1, 1], "computed_smem_bytes": 108800, "constants": [], "cta_group": 1, "threads": 640}'))
_K1_QFULL_MERGE8_KERNELS: dict[str, Any] = {}
_K1_QFULL_MERGE8_SCRATCH: dict[tuple[int, int, int, int, int, int, str, int], tuple[Any, Any]] = {}
TARGET_LABELS: tuple[str, ...] = ('ksweep_q4096_m20000_d128_k1',)
TARGET_SHAPES = _decode_capture(_json_loads('[{"__dict_items__": [["label", "ksweep_q4096_m20000_d128_k1"], ["params", {"__dict_items__": [["B", 1], ["Q", 4096], ["M", 20000], ["D", 128], ["K", 1], ["dtype", "bfloat16"], ["seed", 610310], ["self_search", false], ["min_recall", 1.0]]}]]}]'))
K1_POST_MERGE8_SHAPES = [*TARGET_SHAPES, *parent_seed.HELDOUT_K1_HIGHQ_SHAPES]
SHAPE_SPLIT_M = _decode_capture(_json_loads('{"__dict_items__": [[{"__tuple__": [2048, 20000]}, 9], [{"__tuple__": [3072, 20000]}, 6], [{"__tuple__": [4096, 20000]}, 9], [{"__tuple__": [4096, 16384]}, 9]]}'))
ROUTE_QFULL_MERGE8 = 'roundcd66_qfull_producer_merge8_highq_k1'
SHAPE_DISPATCH_REGISTRY: tuple[dict[str, str], ...] = ({'shape_key': 'roundcd66_qfull_merge8_highq_k1', 'guard': 'B == 1 and D == 128 and K == 1 and (Q, M) in {(2048,20000), (3072,20000), (4096,20000), (4096,16384)} and Q % 128 == 0 and tcgen05', 'route': ROUTE_QFULL_MERGE8}, *parent_seed.SHAPE_DISPATCH_REGISTRY)

def _use_qfull_merge8_highq_k1(inputs: dict[str, Any]) -> bool:
    if int(inputs['B']) != 1 or int(inputs['D']) != D_STATIC or int(inputs['K']) != 1 or (not qfull_seed.mma._tcgen05_capable_arch()):
        return False
    q_rows = int(inputs['Q'])
    return q_rows % BLOCK_Q == 0 and (q_rows, int(inputs['M'])) in SHAPE_SPLIT_M

def selected_route(inputs: dict[str, Any]) -> str:
    if _use_qfull_merge8_highq_k1(inputs):
        split_m = SHAPE_SPLIT_M[int(inputs['Q']), int(inputs['M'])]
        return ''.join([format(ROUTE_QFULL_MERGE8, ''), '_split', format(split_m, '')])
    return parent_seed.selected_route(inputs)

def _split_m_for_shape(inputs: dict[str, Any], total_m_tiles: int) -> int:
    split_m = int(SHAPE_SPLIT_M[int(inputs['Q']), int(inputs['M'])])
    if split_m <= 0:
        raise ValueError('K1 qfull-merge8 split_m must be positive')
    if split_m > 16:
        raise ValueError('K1 qfull-merge8 split_m must be <= 16 for merge8')
    return min(split_m, int(total_m_tiles))

def _compile_qfull_merge8_kernels() -> dict[str, Any]:
    return _decode_capture(_json_loads('{"merge_k1": {"__kernel__": "dispatch_kernel_0390"}, "partial_k1": {"__kernel__": "dispatch_kernel_0389"}}'))

def _scratch_k1(inputs: dict[str, Any], split_m: int, num_q_tiles: int) -> tuple[Any, Any]:
    import torch
    key = (int(inputs['B']), int(inputs['Q']), int(inputs['M']), int(split_m), int(num_q_tiles), int(inputs['queries'].device.index or 0), str(inputs['queries'].dtype), current_stream_handle(inputs))
    cached = _K1_QFULL_MERGE8_SCRATCH.get(key)
    if cached is None:
        shape = (int(inputs['B']), int(num_q_tiles), int(split_m), BLOCK_Q, K_PARTIAL_MAX)
        cached = (torch.empty(shape, dtype=torch.float32, device=inputs['queries'].device), torch.empty(shape, dtype=torch.int32, device=inputs['queries'].device))
        _K1_QFULL_MERGE8_SCRATCH[key] = cached
    return cached

def _launch_qfull_merge8(inputs: dict[str, Any]) -> dict[str, Any]:
    if not _K1_QFULL_MERGE8_KERNELS:
        _K1_QFULL_MERGE8_KERNELS.update(_compile_qfull_merge8_kernels())
    bsz = int(inputs['B'])
    q_rows = int(inputs['Q'])
    m_rows = int(inputs['M'])
    num_q_tiles = math.ceil(q_rows / BLOCK_Q)
    total_m_tiles = math.ceil(m_rows / BLOCK_M)
    split_m = _split_m_for_shape(inputs, total_m_tiles)
    tiles_per_split = math.ceil(total_m_tiles / split_m)
    partial_dist, partial_idx = _scratch_k1(inputs, split_m, num_q_tiles)
    _K1_QFULL_MERGE8_KERNELS['partial_k1'].launch(grid=(bsz * num_q_tiles * split_m, 1, 1), block=(THREADS, 1, 1), args=[inputs['queries'], inputs['database'], partial_dist, partial_idx, bsz, q_rows, m_rows, split_m, num_q_tiles, total_m_tiles, tiles_per_split], shared_mem=LOWK_MMA_SMEM_BYTES)
    _K1_QFULL_MERGE8_KERNELS['merge_k1'].launch(grid=(bsz * math.ceil(q_rows / 8), 1, 1), block=(MERGE_THREADS, 1, 1), args=[partial_dist, partial_idx, inputs['out_distances'], inputs['out_indices'], bsz, q_rows, int(inputs['K']), split_m, num_q_tiles], shared_mem=qfull_seed.mma.MERGE_SMEM_BYTES)
    return {'distances': inputs['out_distances'], 'indices': inputs['out_indices']}

def launch_for_eval(inputs: dict[str, Any]) -> dict[str, Any]:
    if _use_qfull_merge8_highq_k1(inputs):
        return _launch_qfull_merge8(inputs)
    return parent_seed.launch_for_eval(inputs)

def knn_search_compile_and_launch_k1_post_merge8_qfull_merge8(*, benchmark: bool=True, shapes: list[dict[str, Any]] | None=None) -> dict[str, Any]:
    from .._dispatch_runtime import evaluate
    result = evaluate(launch_for_eval, shapes=K1_POST_MERGE8_SHAPES if shapes is None else shapes, benchmark=benchmark)
    result['passed'] = bool(result.get('summary', {}).get('all_correct'))
    print(result)
    return result
