"""Round-49 Q4096 low-K K5 partial producer for exact BF16 kNN.

Minimum target architecture: sm_100a. This additive shape-kernel candidate
preserves the round-48 dispatcher for guard misses and routes only
``B=1, Q=4096, M=20000, D=128, K=5`` through a split-4 tcgen05 producer that
keeps five partial candidates per split instead of the inherited K10 partial
list.
"""
from __future__ import annotations
from json import loads as _json_loads
from .._dispatch_runtime import _capture_cuTensorMapEncodeTiled, _decode_capture, _import_dispatch_module, _ir_proxy
import math
import os
from typing import Any
from .._dispatch_runtime import select_named_shapes
from . import knn_search_highq_midm_k1k2_k5merge_dispatch_0613_r48_48e9_v1 as parent
from . import knn_search_mma_split_v1 as mma
THREADS = mma.THREADS
MERGE_THREADS = mma.MERGE_THREADS
BLOCK_Q = mma.BLOCK_Q
BLOCK_M = mma.BLOCK_M
D_STATIC = mma.D_STATIC
K_PARTIAL_MAX = 5
Q4096_ROWS = 4096
Q4096_LOWK_M = 20000
Q4096_LOWK_K5_SPLIT_M = 4
LOWK_COHORT_TOPK_D_OFFSET = mma.MMA_SMEM_A_BYTES
LOWK_COHORT_TOPK_I_OFFSET = LOWK_COHORT_TOPK_D_OFFSET + mma.MMA_POST_MMA_COL_COHORTS * BLOCK_Q * K_PARTIAL_MAX * 4
LOWK_COHORT_TOPK_END = LOWK_COHORT_TOPK_I_OFFSET + mma.MMA_POST_MMA_COL_COHORTS * BLOCK_Q * K_PARTIAL_MAX * 4
LOWK_MMA_SMEM_POOL_BYTES = _decode_capture(_json_loads('107776'))
LOWK_MMA_SMEM_BYTES = LOWK_MMA_SMEM_POOL_BYTES + mma.WEAVE_SMEM_SYSTEM_BYTES
ir = _decode_capture(_json_loads('{"__ir__": "knn_search_q4096_lowk_k5partial_0613_r49_48e9_v1", "arg_keys": ["queries", "database", "partial_distances", "partial_indices", "B", "Q", "M", "split_m", "num_q_tiles", "total_m_tiles", "tiles_per_split"], "cluster_dims": [1, 1, 1], "computed_smem_bytes": 108800, "constants": [], "cta_group": 1, "threads": 640}'))
partial_ir = _decode_capture(_json_loads('{"__ir__": "knn_search_q4096_lowk_k5partial_0613_r49_48e9_v1", "arg_keys": ["queries", "database", "partial_distances", "partial_indices", "B", "Q", "M", "split_m", "num_q_tiles", "total_m_tiles", "tiles_per_split"], "cluster_dims": [1, 1, 1], "computed_smem_bytes": 108800, "constants": [], "cta_group": 1, "threads": 640}'))
merge_k5_ir = _decode_capture(_json_loads('{"__ir__": "knn_search_q4096_lowk_k5partial_merge_0613_r49_48e9_v1", "arg_keys": ["partial_distances", "partial_indices", "out_distances", "out_indices", "B", "Q", "K", "split_m", "num_q_tiles"], "cluster_dims": [1, 1, 1], "computed_smem_bytes": 0, "constants": [["K_STRIDE_", 5], ["K_OUT_MAX_", 5]], "cta_group": 1, "threads": 32}'))
parent_ir = _decode_capture(_json_loads('{"__ir__": "knn_search_q4096_lowk_k5_stride10_merge_0613_r48_48e9_v1", "arg_keys": ["partial_distances", "partial_indices", "out_distances", "out_indices", "B", "Q", "K", "split_m", "num_q_tiles"], "cluster_dims": [1, 1, 1], "computed_smem_bytes": 0, "constants": [["K_STRIDE_", 10], ["K_OUT_MAX_", 5]], "cta_group": 1, "threads": 32}'))
_LOWK_K5_KERNELS: dict[str, Any] = {}
_LOWK_K5_SCRATCH: dict[tuple[int, int, int, int, int, int, str], tuple[Any, Any]] = {}
_knn_lowk5_insert_one = _ir_proxy('loom.examples.weave.knn_search_q4096_lowk_k5partial_dispatch_0613_r49_48e9_v1:_knn_lowk5_insert_one', 256)
_knn_lowk5_insert_sorted_pair = _ir_proxy('loom.examples.weave.knn_search_q4096_lowk_k5partial_dispatch_0613_r49_48e9_v1:_knn_lowk5_insert_sorted_pair', 256)
_knn_lowk5_insert_batch_merge = _ir_proxy('loom.examples.weave.knn_search_q4096_lowk_k5partial_dispatch_0613_r49_48e9_v1:_knn_lowk5_insert_batch_merge', 256)
_knn_store_lowk5_pairs = _ir_proxy('loom.examples.weave.knn_search_q4096_lowk_k5partial_dispatch_0613_r49_48e9_v1:_knn_store_lowk5_pairs', 256)
knn_search_q4096_lowk_k5partial_0613_r49_48e9_v1 = _decode_capture(_json_loads('{"__ir__": "knn_search_q4096_lowk_k5partial_0613_r49_48e9_v1", "arg_keys": ["queries", "database", "partial_distances", "partial_indices", "B", "Q", "M", "split_m", "num_q_tiles", "total_m_tiles", "tiles_per_split"], "cluster_dims": [1, 1, 1], "computed_smem_bytes": 108800, "constants": [], "cta_group": 1, "threads": 640}'))
knn_search_q4096_lowk_k5partial_merge_0613_r49_48e9_v1 = _decode_capture(_json_loads('{"__ir__": "knn_search_q4096_lowk_k5partial_merge_0613_r49_48e9_v1", "arg_keys": ["partial_distances", "partial_indices", "out_distances", "out_indices", "B", "Q", "K", "split_m", "num_q_tiles"], "cluster_dims": [1, 1, 1], "computed_smem_bytes": 0, "constants": [["K_STRIDE_", 5], ["K_OUT_MAX_", 5]], "cta_group": 1, "threads": 32}'))
partial_ir = _decode_capture(_json_loads('{"__ir__": "knn_search_q4096_lowk_k5partial_0613_r49_48e9_v1", "arg_keys": ["queries", "database", "partial_distances", "partial_indices", "B", "Q", "M", "split_m", "num_q_tiles", "total_m_tiles", "tiles_per_split"], "cluster_dims": [1, 1, 1], "computed_smem_bytes": 108800, "constants": [], "cta_group": 1, "threads": 640}'))
merge_k5_ir = _decode_capture(_json_loads('{"__ir__": "knn_search_q4096_lowk_k5partial_merge_0613_r49_48e9_v1", "arg_keys": ["partial_distances", "partial_indices", "out_distances", "out_indices", "B", "Q", "K", "split_m", "num_q_tiles"], "cluster_dims": [1, 1, 1], "computed_smem_bytes": 0, "constants": [["K_STRIDE_", 5], ["K_OUT_MAX_", 5]], "cta_group": 1, "threads": 32}'))
ir = _decode_capture(_json_loads('{"__ir__": "knn_search_q4096_lowk_k5partial_0613_r49_48e9_v1", "arg_keys": ["queries", "database", "partial_distances", "partial_indices", "B", "Q", "M", "split_m", "num_q_tiles", "total_m_tiles", "tiles_per_split"], "cluster_dims": [1, 1, 1], "computed_smem_bytes": 108800, "constants": [], "cta_group": 1, "threads": 640}'))
Q4096_K5_LABELS: tuple[str, ...] = ('ksweep_q4096_m20000_d128_k5',)
ROUND49_MEASURE_LABELS: tuple[str, ...] = ('dispatch_q256_m65536_d128_k10', 'dispatch_q512_m65536_d128_k10', 'dispatch_q1024_m65536_d128_k10', 'dispatch_q2048_m65536_d128_k10', 'rag_q4096_m20000_d128_k10', 'rag_batch_q4096_m20000_d128_k10', 'dispatch_q4096_m16384_d128_k10', 'dispatch_q4096_m32768_d128_k10', 'ksweep_q4096_m20000_d128_k1', 'ksweep_q4096_m20000_d128_k2', *Q4096_K5_LABELS)
Q4096_K5_SHAPES = parent.Q4096_K5_SHAPES
ROUND49_MEASURE_SHAPES = parent.HIGHQ_MIDM_K1K2K5_SHAPES
ROUND49_PRESERVE_SHAPES = parent.ROUND48_PRESERVE_SHAPES
SHAPE_DISPATCH_REGISTRY: tuple[dict[str, str], ...] = ({'shape_key': 'd128_q4096_lowk_k5partial_split4', 'guard': 'B == 1 and Q == 4096 and M == 20000 and D == 128 and K == 5 and tcgen05', 'route': 'round49_q4096_lowk_k5partial_split4'}, *parent.SHAPE_DISPATCH_REGISTRY)

def _use_q4096_k5(inputs: dict[str, Any]) -> bool:
    return int(inputs['B']) == 1 and int(inputs['Q']) == Q4096_ROWS and (int(inputs['M']) == Q4096_LOWK_M) and (int(inputs['D']) == D_STATIC) and (int(inputs['K']) == K_PARTIAL_MAX) and mma._tcgen05_capable_arch()

def selected_route(inputs: dict[str, Any]) -> str:
    if _use_q4096_k5(inputs):
        return 'round49_q4096_lowk_k5partial_split4'
    return parent.selected_route(inputs)

def _compile_k5_kernels() -> dict[str, Any]:
    return _decode_capture(_json_loads('{"merge_k5": {"__kernel__": "dispatch_kernel_0196"}, "partial_k5": {"__kernel__": "dispatch_kernel_0158"}}'))

def _scratch_k5(inputs: dict[str, Any], split_m: int, num_q_tiles: int) -> tuple[Any, Any]:
    import torch
    key = (int(inputs['B']), int(inputs['Q']), int(inputs['M']), int(split_m), int(num_q_tiles), int(inputs['queries'].device.index or 0), str(inputs['queries'].dtype))
    cached = _LOWK_K5_SCRATCH.get(key)
    if cached is None:
        shape = (int(inputs['B']), int(num_q_tiles), int(split_m), BLOCK_Q, K_PARTIAL_MAX)
        cached = (torch.empty(shape, dtype=torch.float32, device=inputs['queries'].device), torch.empty(shape, dtype=torch.int32, device=inputs['queries'].device))
        _LOWK_K5_SCRATCH[key] = cached
    return cached

def _k5_split_m(total_m_tiles: int) -> int:
    override = os.environ.get('LOOM_KNN_Q4096_LOWK_K5_SPLIT_M')
    split_m = int(override) if override else Q4096_LOWK_K5_SPLIT_M
    if split_m <= 0:
        raise ValueError('LOOM_KNN_Q4096_LOWK_K5_SPLIT_M must be positive')
    if split_m > 4:
        raise ValueError('LOOM_KNN_Q4096_LOWK_K5_SPLIT_M must be <= 4 for the four-lane merge')
    return min(split_m, int(total_m_tiles))

def _launch_q4096_k5(inputs: dict[str, Any]) -> dict[str, Any]:
    import torch
    if not _LOWK_K5_KERNELS:
        _LOWK_K5_KERNELS.update(_compile_k5_kernels())
    bsz = int(inputs['B'])
    q_rows = int(inputs['Q'])
    m_rows = int(inputs['M'])
    k = int(inputs['K'])
    num_q_tiles = math.ceil(q_rows / BLOCK_Q)
    total_m_tiles = math.ceil(m_rows / BLOCK_M)
    split_m = _k5_split_m(total_m_tiles)
    tiles_per_split = math.ceil(total_m_tiles / split_m)
    partial_dist, partial_idx = _scratch_k5(inputs, split_m, num_q_tiles)
    _LOWK_K5_KERNELS['partial_k5'].launch(grid=(bsz * num_q_tiles * split_m, 1, 1), block=(THREADS, 1, 1), args=[inputs['queries'], inputs['database'], partial_dist, partial_idx, bsz, q_rows, m_rows, split_m, num_q_tiles, total_m_tiles, tiles_per_split], shared_mem=LOWK_MMA_SMEM_BYTES)
    _LOWK_K5_KERNELS['merge_k5'].launch(grid=(bsz * q_rows, 1, 1), block=(MERGE_THREADS, 1, 1), args=[partial_dist, partial_idx, inputs['out_distances'], inputs['out_indices'], bsz, q_rows, k, split_m, num_q_tiles], shared_mem=mma.MERGE_SMEM_BYTES)
    return {'distances': inputs['out_distances'], 'indices': inputs['out_indices']}

def launch_for_eval(inputs: dict[str, Any]) -> dict[str, Any]:
    if _use_q4096_k5(inputs):
        return _launch_q4096_k5(inputs)
    return parent.launch_for_eval(inputs)

def _select_contract_shapes(shape_labels: str | tuple[str, ...] | list[str] | None):
    if shape_labels is None:
        return None
    return select_named_shapes(shape_labels)

def knn_search_compile_and_launch_q4096_k5(*, benchmark: bool=True, shapes: list[dict[str, Any]] | None=None) -> dict[str, Any]:
    from .._dispatch_runtime import evaluate
    result = evaluate(launch_for_eval, shapes=Q4096_K5_SHAPES if shapes is None else shapes, benchmark=benchmark)
    result['passed'] = bool(result.get('summary', {}).get('all_correct'))
    print(result)
    return result

def knn_search_compile_and_launch_round49_measure(*, benchmark: bool=True) -> dict[str, Any]:
    from .._dispatch_runtime import evaluate
    result = evaluate(launch_for_eval, shapes=ROUND49_MEASURE_SHAPES, benchmark=benchmark)
    result['passed'] = bool(result.get('summary', {}).get('all_correct'))
    print(result)
    return result
