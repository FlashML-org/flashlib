"""Round-34 D256 BF16 squared-L2 kNN search over the r33 application wrapper.

Minimum target architecture: sm_100a for the D256 tcgen05 route; r33 fallback
routes keep their original minimum targets. This additive candidate keeps the
round-33 low-D/D256-over-48e9 application dispatcher and attempts to replace
the two D256 scalar-capacity GLM/RAG labels with a source-clean D256
split-M tcgen05 producer using a 128x64x256 dot tile.
"""
from __future__ import annotations
from json import loads as _json_loads
from .._dispatch_runtime import _capture_cuTensorMapEncodeTiled, _decode_capture, _import_dispatch_module, _ir_proxy
import math
from functools import lru_cache
from types import ModuleType
from typing import Any
THREADS = 256
BLOCK_Q = 128
BLOCK_M = 64
D_STATIC = 256
K_MAX = 64
D256_MMA_SPLIT_M = 256
MMA_POST_MMA_COL_COHORTS = 2
MMA_EXTRA_TOPK_COHORTS = MMA_POST_MMA_COL_COHORTS - 1
MMA_SMEM_A_BYTES = BLOCK_Q * D_STATIC * 2
MMA_SMEM_B_BYTES = BLOCK_M * D_STATIC * 2
MMA_DB_NORM_PARTS = 4
MMA_DB_NORM_CHUNK = D_STATIC // MMA_DB_NORM_PARTS
MMA_STAGE_VEC_ELEMS = 16
MMA_STAGE_PACK_WORDS = MMA_STAGE_VEC_ELEMS // 2
MMA_Q_STAGE_VECS = BLOCK_Q * D_STATIC // MMA_STAGE_VEC_ELEMS
MMA_Q_NORM_PARTS = D_STATIC // MMA_STAGE_VEC_ELEMS
MMA_DB_NORM_PART_VECS = MMA_DB_NORM_CHUNK // MMA_STAGE_VEC_ELEMS
MMA_SMEM_Q_NORM_PART_BYTES = BLOCK_Q * MMA_Q_NORM_PARTS * 4
MMA_SMEM_DB_NORM_PART_BYTES = BLOCK_M * MMA_DB_NORM_PARTS * 4
MMA_SMEM_DB_NORM_BYTES = BLOCK_M * 4
MMA_SMEM_B0_OFFSET = MMA_SMEM_A_BYTES
MMA_SMEM_B1_OFFSET = MMA_SMEM_B0_OFFSET + MMA_SMEM_B_BYTES
MMA_SMEM_Q_NORM_PART_OFFSET = MMA_SMEM_B1_OFFSET + MMA_SMEM_B_BYTES
MMA_SMEM_DB_NORM_PART0_OFFSET = MMA_SMEM_Q_NORM_PART_OFFSET + MMA_SMEM_Q_NORM_PART_BYTES
MMA_SMEM_DB_NORM_PART1_OFFSET = MMA_SMEM_DB_NORM_PART0_OFFSET + MMA_SMEM_DB_NORM_PART_BYTES
MMA_SMEM_DB_NORM0_OFFSET = MMA_SMEM_DB_NORM_PART1_OFFSET + MMA_SMEM_DB_NORM_PART_BYTES
MMA_SMEM_DB_NORM1_OFFSET = MMA_SMEM_DB_NORM0_OFFSET + MMA_SMEM_DB_NORM_BYTES
MMA_STAGING_END = MMA_SMEM_DB_NORM1_OFFSET + MMA_SMEM_DB_NORM_BYTES
WEAVE_SMEM_SYSTEM_BYTES = 1024
MMA_SMEM_POOL_BYTES = MMA_STAGING_END + 256
MMA_SMEM_BYTES = MMA_SMEM_POOL_BYTES + WEAVE_SMEM_SYSTEM_BYTES
MERGE_THREADS = 32
MERGE_SPLITS_PER_LANE_MAX = 19
MERGE_SMEM_POOL_BYTES = 0
MERGE_SMEM_BYTES = MERGE_SMEM_POOL_BYTES + WEAVE_SMEM_SYSTEM_BYTES
_D256_MMA_KERNELS: dict[str, Any] = {}
_D256_MMA_SCRATCH: dict[tuple[int, int, int, int, int, int, int, str], tuple[Any, Any]] = {}

@lru_cache(maxsize=1)
def _legacy_parent() -> ModuleType:
    from . import knn_search_lowd_d256_over48e9_0612_r33_blockm_v1
    return knn_search_lowd_d256_over48e9_0612_r33_blockm_v1

def __getattr__(name: str) -> Any:
    if name in {'LOWD_D256_OVER48E9_SHAPES', 'LOWD_D256_PRESERVE_SHAPES'}:
        return getattr(_legacy_parent(), name)
    raise AttributeError(''.join(['module ', format(repr(__name__), ''), ' has no attribute ', format(repr(name), '')]))
_knn_mma_issue = _ir_proxy('loom.examples.weave.knn_search_lowd_d256_mma64_over48e9_0612_r34_48e9_v1:_knn_mma_issue', 256)
_knn_stage_database_tile = _ir_proxy('loom.examples.weave.knn_search_lowd_d256_mma64_over48e9_0612_r34_48e9_v1:_knn_stage_database_tile', 256)
_knn_insert_sorted_pair_assume_first_fits = _ir_proxy('loom.examples.weave.knn_search_lowd_d256_mma64_over48e9_0612_r34_48e9_v1:_knn_insert_sorted_pair_assume_first_fits', 256)
_knn_insert_sorted_pair = _ir_proxy('loom.examples.weave.knn_search_lowd_d256_mma64_over48e9_0612_r34_48e9_v1:_knn_insert_sorted_pair', 256)
_knn_insert_sorted_two_pair_priority_gate = _ir_proxy('loom.examples.weave.knn_search_lowd_d256_mma64_over48e9_0612_r34_48e9_v1:_knn_insert_sorted_two_pair_priority_gate', 256)
_knn_insert_sorted_pair_batch_merge = _ir_proxy('loom.examples.weave.knn_search_lowd_d256_mma64_over48e9_0612_r34_48e9_v1:_knn_insert_sorted_pair_batch_merge', 256)
_knn_merge_two_sorted_lists_to_smem = _ir_proxy('loom.examples.weave.knn_search_lowd_d256_mma64_over48e9_0612_r34_48e9_v1:_knn_merge_two_sorted_lists_to_smem', 256)
_knn_merge_two_sorted_lists_to_regs = _ir_proxy('loom.examples.weave.knn_search_lowd_d256_mma64_over48e9_0612_r34_48e9_v1:_knn_merge_two_sorted_lists_to_regs', 256)
_knn_store_topk_pairs = _ir_proxy('loom.examples.weave.knn_search_lowd_d256_mma64_over48e9_0612_r34_48e9_v1:_knn_store_topk_pairs', 256)
knn_search_mma_split_partial_v1 = _decode_capture(_json_loads('{"__ir__": "knn_search_mma_split_partial_v1", "arg_keys": ["queries", "database", "partial_distances", "partial_indices", "B", "Q", "M", "split_m", "num_q_tiles", "total_m_tiles", "tiles_per_split"], "cluster_dims": [1, 1, 1], "computed_smem_bytes": 143104, "constants": [["K_MAX_", 64], ["EXPOSE_COL_COHORTS", 1]], "cta_group": 1, "threads": 256}'))
knn_search_mma_split_merge_v1 = _decode_capture(_json_loads('{"__ir__": "knn_search_mma_split_merge_v1", "arg_keys": ["partial_distances", "partial_indices", "out_distances", "out_indices", "B", "Q", "K", "split_m", "num_q_tiles"], "cluster_dims": [1, 1, 1], "computed_smem_bytes": 0, "constants": [["K_MAX_", 64]], "cta_group": 1, "threads": 32}'))
knn_search_mma_split_merge_stream_v1 = _decode_capture(_json_loads('{"__ir__": "knn_search_mma_split_merge_stream_v1", "arg_keys": ["partial_distances", "partial_indices", "out_distances", "out_indices", "B", "Q", "K", "split_m", "num_q_tiles"], "cluster_dims": [1, 1, 1], "computed_smem_bytes": 0, "constants": [["K_MAX_", 64]], "cta_group": 1, "threads": 32}'))
partial_ir = _decode_capture(_json_loads('{"__ir__": "knn_search_mma_split_partial_v1", "arg_keys": ["queries", "database", "partial_distances", "partial_indices", "B", "Q", "M", "split_m", "num_q_tiles", "total_m_tiles", "tiles_per_split"], "cluster_dims": [1, 1, 1], "computed_smem_bytes": 143104, "constants": [["K_MAX_", 64], ["EXPOSE_COL_COHORTS", 1]], "cta_group": 1, "threads": 256}'))
merge_ir = _decode_capture(_json_loads('{"__ir__": "knn_search_mma_split_merge_v1", "arg_keys": ["partial_distances", "partial_indices", "out_distances", "out_indices", "B", "Q", "K", "split_m", "num_q_tiles"], "cluster_dims": [1, 1, 1], "computed_smem_bytes": 0, "constants": [["K_MAX_", 64]], "cta_group": 1, "threads": 32}'))
ir = _decode_capture(_json_loads('{"__ir__": "knn_search_mma_split_partial_v1", "arg_keys": ["queries", "database", "partial_distances", "partial_indices", "B", "Q", "M", "split_m", "num_q_tiles", "total_m_tiles", "tiles_per_split"], "cluster_dims": [1, 1, 1], "computed_smem_bytes": 143104, "constants": [["K_MAX_", 64], ["EXPOSE_COL_COHORTS", 1]], "cta_group": 1, "threads": 256}'))

def _compile_kernels() -> dict[str, Any]:
    return _decode_capture(_json_loads('{"merge": {"__kernel__": "dispatch_kernel_0281"}, "partial": {"__kernel__": "dispatch_kernel_0280"}}'))

def _scratch(inputs: dict[str, Any], split_m: int, num_q_tiles: int) -> tuple[Any, Any]:
    import torch
    key = (int(inputs['B']), int(inputs['Q']), int(inputs['M']), int(split_m), int(num_q_tiles), int(inputs['queries'].device.index or 0), str(inputs['queries'].dtype))
    cached = _D256_MMA_SCRATCH.get(key)
    if cached is None:
        shape = (int(inputs['B']), int(num_q_tiles), int(split_m), BLOCK_Q, K_MAX)
        cached = (torch.empty(shape, dtype=torch.float32, device=inputs['queries'].device), torch.empty(shape, dtype=torch.int32, device=inputs['queries'].device))
        _D256_MMA_SCRATCH[key] = cached
    return cached

def _tcgen05_capable_arch() -> bool:
    from .._dispatch_runtime import detect_gpu_arch
    return detect_gpu_arch() in {'sm_100a', 'sm_103a'}

def _use_d256_mma(inputs: dict[str, Any]) -> bool:
    return _tcgen05_capable_arch() and int(inputs['B']) == 1 and (int(inputs['Q']) == 128) and (int(inputs['M']) == 131072) and (int(inputs['D']) == D_STATIC) and (int(inputs['K']) in {10, 64})

def launch_for_eval(inputs: dict[str, Any]) -> dict[str, Any]:
    if not _use_d256_mma(inputs):
        return _legacy_parent().launch_for_eval(inputs)
    return _launch_d256_mma(inputs)

def _launch_d256_mma(inputs: dict[str, Any]) -> dict[str, Any]:
    import torch
    if not _D256_MMA_KERNELS:
        _D256_MMA_KERNELS.update(_compile_kernels())
    bsz = int(inputs['B'])
    q_rows = int(inputs['Q'])
    m_rows = int(inputs['M'])
    k = int(inputs['K'])
    num_q_tiles = math.ceil(q_rows / BLOCK_Q)
    total_m_tiles = math.ceil(m_rows / BLOCK_M)
    split_m = min(D256_MMA_SPLIT_M, total_m_tiles)
    tiles_per_split = math.ceil(total_m_tiles / split_m)
    partial_list_count = split_m * MMA_POST_MMA_COL_COHORTS
    partial_dist, partial_idx = _scratch(inputs, partial_list_count, num_q_tiles)
    _D256_MMA_KERNELS['partial'].launch(grid=(bsz * num_q_tiles * split_m, 1, 1), block=(THREADS, 1, 1), args=[inputs['queries'], inputs['database'], partial_dist, partial_idx, bsz, q_rows, m_rows, split_m, num_q_tiles, total_m_tiles, tiles_per_split], shared_mem=MMA_SMEM_BYTES)
    _D256_MMA_KERNELS['merge'].launch(grid=(bsz * q_rows, 1, 1), block=(MERGE_THREADS, 1, 1), args=[partial_dist, partial_idx, inputs['out_distances'], inputs['out_indices'], bsz, q_rows, k, partial_list_count, num_q_tiles], shared_mem=MERGE_SMEM_BYTES)
    return {'distances': inputs['out_distances'], 'indices': inputs['out_indices']}

def _select_contract_shapes(shape_labels: str | tuple[str, ...] | list[str] | None):
    return _legacy_parent()._select_contract_shapes(shape_labels)

def knn_search_compile_and_launch_mma_split(*, benchmark: bool=True, shape_labels: str | tuple[str, ...] | list[str] | None=None) -> dict[str, Any]:
    from .._dispatch_runtime import evaluate
    result = evaluate(launch_for_eval, shapes=_select_contract_shapes(shape_labels), benchmark=benchmark)
    result['passed'] = bool(result.get('summary', {}).get('all_correct'))
    print(result)
    return result

def knn_search_compile_and_launch_lowd_d256_over48e9(*, benchmark: bool=True, shapes: list[dict[str, Any]] | None=None) -> dict[str, Any]:
    from .._dispatch_runtime import evaluate
    result = evaluate(launch_for_eval, shapes=_legacy_parent().LOWD_D256_OVER48E9_SHAPES if shapes is None else shapes, benchmark=benchmark)
    result['passed'] = bool(result.get('summary', {}).get('all_correct'))
    print(result)
    return result
