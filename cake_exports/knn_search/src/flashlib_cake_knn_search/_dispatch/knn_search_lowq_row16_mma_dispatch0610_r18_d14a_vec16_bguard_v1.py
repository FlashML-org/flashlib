"""Low-Q ROW_16x256B split-M MMA candidate for exact BF16 squared-L2 kNN.

Minimum target architecture: sm_100a. This round-18 additive candidate keeps
the source-policy-clean ROW_16x256B Q64 readback ownership model, 16-wide BF16
Q/database staging, incumbent const-148 split-M merge ABI, and round-17 padded
query row skip. Relative to round 17, it removes redundant `batch_id < B`
guards from the producer; the launch grid is exactly `B * q_tiles * split_m`,
so every producer CTA has a valid batch id.
"""
from __future__ import annotations
from json import loads as _json_loads
from .._dispatch_runtime import _capture_cuTensorMapEncodeTiled, _decode_capture, _import_dispatch_module, _ir_proxy
import math
from typing import Any
from . import knn_search_mma_split_v1 as _mma
THREADS = 512
BLOCK_Q = 64
PARTIAL_STRIDE_Q = _mma.BLOCK_Q
BLOCK_M = 128
D_STATIC = 128
K_MAX = 10
VEC = 16
PACK_WORDS = VEC // 2
Q_STAGE_VECS = BLOCK_Q * D_STATIC // VEC
Q_NORM_PARTS = D_STATIC // VEC
DB_NORM_PARTS = 4
DB_NORM_CHUNK = D_STATIC // DB_NORM_PARTS
DB_NORM_PART_VECS = DB_NORM_CHUNK // VEC
LOCAL_LISTS_PER_ROW = 8
SPLIT_M = _mma.Q128_SPLIT_M
SMEM_A_BYTES = BLOCK_Q * D_STATIC * 2
SMEM_B_BYTES = BLOCK_M * D_STATIC * 2
SMEM_Q_NORM_PART_BYTES = BLOCK_Q * Q_NORM_PARTS * 4
SMEM_DB_NORM_PART_BYTES = BLOCK_M * DB_NORM_PARTS * 4
SMEM_DB_NORM_BYTES = BLOCK_M * 4
SMEM_LOCAL_D_BYTES = BLOCK_Q * LOCAL_LISTS_PER_ROW * K_MAX * 4
SMEM_LOCAL_I_BYTES = BLOCK_Q * LOCAL_LISTS_PER_ROW * K_MAX * 4
Q_NORM_OFFSET = SMEM_A_BYTES + SMEM_B_BYTES
DB_NORM_PART_OFFSET = Q_NORM_OFFSET + SMEM_Q_NORM_PART_BYTES
DB_NORM_OFFSET = DB_NORM_PART_OFFSET + SMEM_DB_NORM_PART_BYTES
LOCAL_D_OFFSET = DB_NORM_OFFSET + SMEM_DB_NORM_BYTES
LOCAL_I_OFFSET = LOCAL_D_OFFSET + SMEM_LOCAL_D_BYTES
SMEM_POOL_BYTES = LOCAL_I_OFFSET + SMEM_LOCAL_I_BYTES + 256
WEAVE_SMEM_SYSTEM_BYTES = 1024
SMEM_BYTES = SMEM_POOL_BYTES + WEAVE_SMEM_SYSTEM_BYTES
_KERNELS: dict[str, Any] = {}
_row16_mma_issue = _ir_proxy('loom.examples.weave.knn_search_lowq_row16_mma_dispatch0610_r18_d14a_vec16_bguard_v1:_row16_mma_issue', 256)
_insert_sorted_pair = _ir_proxy('loom.examples.weave.knn_search_lowq_row16_mma_dispatch0610_r18_d14a_vec16_bguard_v1:_insert_sorted_pair', 256)
_stage_database_tile = _ir_proxy('loom.examples.weave.knn_search_lowq_row16_mma_dispatch0610_r18_d14a_vec16_bguard_v1:_stage_database_tile', 256)
knn_search_lowq_row16_mma_partial_dispatch0610_r18_d14a_vec16_bguard_v1 = _decode_capture(_json_loads('{"__ir__": "knn_search_lowq_row16_mma_partial_dispatch0610_r18_d14a_vec16_bguard_v1", "arg_keys": ["queries", "database", "partial_distances", "partial_indices", "B", "Q", "M", "split_m", "num_q_tiles", "total_m_tiles"], "cluster_dims": [1, 1, 1], "computed_smem_bytes": 96000, "constants": [["K_MAX_", 10]], "cta_group": 1, "threads": 512}'))
ir = _decode_capture(_json_loads('{"__ir__": "knn_search_lowq_row16_mma_partial_dispatch0610_r18_d14a_vec16_bguard_v1", "arg_keys": ["queries", "database", "partial_distances", "partial_indices", "B", "Q", "M", "split_m", "num_q_tiles", "total_m_tiles"], "cluster_dims": [1, 1, 1], "computed_smem_bytes": 96000, "constants": [["K_MAX_", 10]], "cta_group": 1, "threads": 512}'))

def _compile_kernels() -> dict[str, Any]:
    return _decode_capture(_json_loads('{"merge": {"__kernel__": "dispatch_kernel_0009"}, "partial": {"__kernel__": "dispatch_kernel_0014"}}'))

def _tcgen05_capable_arch() -> bool:
    from .._dispatch_runtime import detect_gpu_arch
    return detect_gpu_arch() in {'sm_100a', 'sm_103a'}

def _use_row16_mma(inputs: dict[str, Any]) -> bool:
    q_rows = int(inputs['Q'])
    return _tcgen05_capable_arch() and 8 <= q_rows <= BLOCK_Q and (int(inputs['M']) >= 131072) and (int(inputs['D']) == D_STATIC) and (int(inputs['K']) == K_MAX)

def launch_for_eval(inputs: dict[str, Any]) -> dict[str, Any]:
    import torch
    if not _use_row16_mma(inputs):
        return _mma.launch_for_eval(inputs)
    if not _KERNELS:
        _KERNELS.update(_compile_kernels())
    bsz = int(inputs['B'])
    q_rows = int(inputs['Q'])
    m_rows = int(inputs['M'])
    k = int(inputs['K'])
    num_q_tiles = math.ceil(q_rows / BLOCK_Q)
    total_m_tiles = math.ceil(m_rows / BLOCK_M)
    partial_dist, partial_idx = _mma._scratch(inputs, SPLIT_M, num_q_tiles)
    _KERNELS['partial'].launch(grid=(bsz * num_q_tiles * SPLIT_M, 1, 1), block=(THREADS, 1, 1), args=[inputs['queries'], inputs['database'], partial_dist, partial_idx, bsz, q_rows, m_rows, SPLIT_M, num_q_tiles, total_m_tiles], shared_mem=SMEM_BYTES)
    _KERNELS['merge'].launch(grid=(bsz * q_rows, 1, 1), block=(_mma.MERGE_THREADS, 1, 1), args=[partial_dist, partial_idx, inputs['out_distances'], inputs['out_indices'], bsz, q_rows, k, SPLIT_M, num_q_tiles], shared_mem=_mma.MERGE_SMEM_BYTES)
    return {'distances': inputs['out_distances'], 'indices': inputs['out_indices']}

def lowq_row16_shapes() -> list[dict[str, Any]]:
    from .._dispatch_runtime import load_contract
    labels = {'rag_lowq_q8_m131072_d128_k10', 'rag_lowq_q16_m131072_d128_k10', 'rag_lowq_q32_m131072_d128_k10', 'rag_lowq_q64_m131072_d128_k10'}
    contract = load_contract('knn_search')
    return [{'label': shape['label'], 'params': {key: value for key, value in shape.items() if key != 'label'}} for suite in contract.raw.get('benchmark_shape_suites', {}).get('suites', []) for shape in suite.get('shapes', []) if shape.get('label') in labels]
