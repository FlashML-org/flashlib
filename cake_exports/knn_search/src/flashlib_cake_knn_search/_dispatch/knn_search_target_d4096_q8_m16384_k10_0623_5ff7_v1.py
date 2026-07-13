"""Exact D4096/Q8/M16384/K10 kNN seed using direct-stride tcgen05.

Minimum target architecture: sm_100a. This additive bucket seed owns
``B=1,Q=8,M=16384,D=4096,K=10`` for the target-D frontier. It keeps the
runtime path Weave-only, stages BF16 features directly from the source stride,
uses a Q64 tcgen05 producer with 32 D passes, and feeds the existing Q128
split-M merge ABI.
"""
from __future__ import annotations
from json import loads as _json_loads
from .._dispatch_runtime import _capture_cuTensorMapEncodeTiled, _decode_capture, _import_dispatch_module, _ir_proxy
import math
from typing import Any
from .._dispatch_runtime import select_named_shapes
from . import knn_search_dynamic_d512_q64_tcgen05_0618_9286_v1 as q64
from . import knn_search_lowq_row16_mma_dispatch0610_r18_d14a_vec16_bguard_v1 as row16
from . import knn_search_mma_split_v1 as mma
THREADS = q64.THREADS
MERGE_THREADS = mma.MERGE_THREADS
BLOCK_Q = q64.BLOCK_Q
PARTIAL_STRIDE_Q = mma.BLOCK_Q
BLOCK_M = mma.BLOCK_M
D_STAGE = mma.D_STATIC
D_ORIG = 4096
NUM_D_PASSES = D_ORIG // D_STAGE
K_MAX = mma.K_MAX
SPLIT_M = mma.Q128_SPLIT_M
VEC = mma.MMA_STAGE_VEC_ELEMS
PACK_WORDS = mma.MMA_STAGE_PACK_WORDS
Q_STAGE_VECS = BLOCK_Q * D_STAGE // VEC
Q_NORM_PARTS = D_ORIG // VEC
DB_NORM_PARTS = mma.MMA_DB_NORM_PARTS
DB_NORM_CHUNK = mma.MMA_DB_NORM_CHUNK
DB_NORM_PART_VECS = mma.MMA_DB_NORM_PART_VECS
LOCAL_LISTS_PER_ROW = q64.LOCAL_LISTS_PER_ROW
SMEM_A_BYTES = BLOCK_Q * D_STAGE * 2
SMEM_B_BYTES = BLOCK_M * D_STAGE * 2
SMEM_Q_NORM_PART_BYTES = BLOCK_Q * Q_NORM_PARTS * 4
SMEM_DB_NORM_PART_BYTES = BLOCK_M * DB_NORM_PARTS * 4
SMEM_DB_NORM_BYTES = BLOCK_M * 4
SMEM_LOCAL_D_BYTES = BLOCK_Q * LOCAL_LISTS_PER_ROW * K_MAX * 4
SMEM_LOCAL_I_BYTES = BLOCK_Q * LOCAL_LISTS_PER_ROW * K_MAX * 4
SMEM_B_OFFSET = SMEM_A_BYTES
SMEM_Q_NORM_PART_OFFSET = SMEM_B_OFFSET + SMEM_B_BYTES
SMEM_DB_NORM_PART_OFFSET = SMEM_Q_NORM_PART_OFFSET + SMEM_Q_NORM_PART_BYTES
SMEM_DB_NORM_OFFSET = SMEM_DB_NORM_PART_OFFSET + SMEM_DB_NORM_PART_BYTES
SMEM_LOCAL_D_OFFSET = SMEM_DB_NORM_OFFSET + SMEM_DB_NORM_BYTES
SMEM_LOCAL_I_OFFSET = SMEM_LOCAL_D_OFFSET + SMEM_LOCAL_D_BYTES
SMEM_POOL_BYTES = SMEM_LOCAL_I_OFFSET + SMEM_LOCAL_I_BYTES + 256
SMEM_BYTES = SMEM_POOL_BYTES + mma.WEAVE_SMEM_SYSTEM_BYTES
ROUTE_D4096_Q8_M16384_TCGEN05 = '5ff7_d4096_q8_m16384_k10_q64loop_tcgen05'
CONSUMED_SEED = 'weave-evolve-knn-search-5ff7-d4096-q8'
TARGET_LABELS: tuple[str, ...] = ('target_d4096_q8_m16384_k10',)
TARGET_SHAPES = _decode_capture(_json_loads('[{"__dict_items__": [["label", "target_d4096_q8_m16384_k10"], ["params", {"__dict_items__": [["B", 1], ["Q", 8], ["M", 16384], ["D", 4096], ["K", 10], ["dtype", "bfloat16"], ["seed", 611113], ["self_search", false], ["min_recall", 0.999]]}]]}]'))
_KERNELS: dict[str, Any] = {}
_SCRATCH: dict[tuple[int, int, int, int, int, int, int, str], tuple[Any, Any]] = {}
_accumulate_q_norm_direct_d4096 = _ir_proxy('loom.examples.weave.knn_search_target_d4096_q8_m16384_k10_0623_5ff7_v1:_accumulate_q_norm_direct_d4096', 256)
_stage_q_pass_direct_d4096 = _ir_proxy('loom.examples.weave.knn_search_target_d4096_q8_m16384_k10_0623_5ff7_v1:_stage_q_pass_direct_d4096', 256)
_stage_database_pass_direct_d4096 = _ir_proxy('loom.examples.weave.knn_search_target_d4096_q8_m16384_k10_0623_5ff7_v1:_stage_database_pass_direct_d4096', 256)
_accumulate_db_norm_pass_runtime_init = _ir_proxy('loom.examples.weave.knn_search_target_d4096_q8_m16384_k10_0623_5ff7_v1:_accumulate_db_norm_pass_runtime_init', 256)
knn_search_target_d4096_q8_m16384_k10_tcgen05_partial_0623_5ff7_v1 = _decode_capture(_json_loads('{"__ir__": "knn_search_target_d4096_q8_m16384_k10_tcgen05_partial_0623_5ff7_v1", "arg_keys": ["queries", "database", "partial_distances", "partial_indices", "B", "Q", "M", "split_m", "num_q_tiles", "total_m_tiles"], "cluster_dims": [1, 1, 1], "computed_smem_bytes": 159488, "constants": [["K_MAX_", 10], ["NUM_D_PASSES_", 32]], "cta_group": 1, "threads": 512}'))
partial_ir = _decode_capture(_json_loads('{"__ir__": "knn_search_target_d4096_q8_m16384_k10_tcgen05_partial_0623_5ff7_v1", "arg_keys": ["queries", "database", "partial_distances", "partial_indices", "B", "Q", "M", "split_m", "num_q_tiles", "total_m_tiles"], "cluster_dims": [1, 1, 1], "computed_smem_bytes": 159488, "constants": [["K_MAX_", 10], ["NUM_D_PASSES_", 32]], "cta_group": 1, "threads": 512}'))
merge_ir = _decode_capture(_json_loads('{"__ir__": "knn_search_mma_split_merge_q128_const148_v1", "arg_keys": ["partial_distances", "partial_indices", "out_distances", "out_indices", "B", "Q", "K", "split_m", "num_q_tiles"], "cluster_dims": [1, 1, 1], "computed_smem_bytes": 0, "constants": [["K_MAX_", 10]], "cta_group": 1, "threads": 32}'))
ir = _decode_capture(_json_loads('{"__ir__": "knn_search_target_d4096_q8_m16384_k10_tcgen05_partial_0623_5ff7_v1", "arg_keys": ["queries", "database", "partial_distances", "partial_indices", "B", "Q", "M", "split_m", "num_q_tiles", "total_m_tiles"], "cluster_dims": [1, 1, 1], "computed_smem_bytes": 159488, "constants": [["K_MAX_", 10], ["NUM_D_PASSES_", 32]], "cta_group": 1, "threads": 512}'))
SHAPE_DISPATCH_REGISTRY: tuple[dict[str, Any], ...] = ({'overlay': 'target_d4096_q8_m16384_k10_q64loop_tcgen05_5ff7', 'shape_key': 'target_d4096_q8_m16384_k10', 'labels': TARGET_LABELS, 'guard': 'B == 1 and Q == 8 and M == 16384 and D == 4096 and K == 10 and not self_search and not forced_fallback and arch in {sm_100a,sm_103a}', 'route': ROUTE_D4096_Q8_M16384_TCGEN05, 'entrypoint': 'loom.examples.weave.knn_search_target_d4096_q8_m16384_k10_0623_5ff7_v1:launch_for_eval', 'selected_seed': CONSUMED_SEED, 'source_task': 'weave-evolve-knn-search-5ff7-d4096-q8', 'coverage_class': 'bucket_seed_target_d4096_q8_m16384_k10', 'workflow_mode': 'generalize_auto_tuning', 'auto_tuning_stage': 'bucket-kernel'},)

def _tcgen05_capable_arch() -> bool:
    from .._dispatch_runtime import detect_gpu_arch
    return detect_gpu_arch() in {'sm_100a', 'sm_103a'}

def _use_d4096_q8_tcgen05(inputs: dict[str, Any]) -> bool:
    return int(inputs['B']) == 1 and int(inputs['Q']) == 8 and (int(inputs['M']) == 16384) and (int(inputs['D']) == D_ORIG) and (int(inputs['K']) == K_MAX) and (not bool(inputs.get('self_search', False))) and (not bool(inputs.get('force_fallback', False))) and _tcgen05_capable_arch()

def _compile_kernels() -> dict[str, Any]:
    return _decode_capture(_json_loads('{"merge": {"__kernel__": "dispatch_kernel_0009"}, "partial": {"__kernel__": "dispatch_kernel_0241"}}'))

def _partial_scratch(inputs: dict[str, Any], split_m: int, num_q_tiles: int) -> tuple[Any, Any]:
    import torch
    key = (int(inputs['B']), int(inputs['Q']), int(inputs['M']), int(inputs['D']), int(split_m), int(num_q_tiles), int(inputs['queries'].device.index or 0), str(inputs['queries'].dtype))
    cached = _SCRATCH.get(key)
    if cached is None:
        shape = (int(inputs['B']), int(num_q_tiles), int(split_m), PARTIAL_STRIDE_Q, K_MAX)
        cached = (torch.empty(shape, dtype=torch.float32, device=inputs['queries'].device), torch.empty(shape, dtype=torch.int32, device=inputs['queries'].device))
        _SCRATCH[key] = cached
    return cached

def selected_route(inputs: dict[str, Any]) -> str:
    if _use_d4096_q8_tcgen05(inputs):
        return ROUTE_D4096_Q8_M16384_TCGEN05
    return 'unsupported_shape'

def selected_route_name(inputs: dict[str, Any]) -> str:
    return selected_route(inputs)

def route_info(inputs: dict[str, Any]) -> dict[str, Any]:
    if _use_d4096_q8_tcgen05(inputs):
        return {'route': ROUTE_D4096_Q8_M16384_TCGEN05, 'selected_route': ROUTE_D4096_Q8_M16384_TCGEN05, 'selected_entrypoint': 'loom.examples.weave.knn_search_target_d4096_q8_m16384_k10_0623_5ff7_v1:launch_for_eval', 'route_kind': 'specialized', 'route_source': 'bucket-specific-seed', 'coverage_class': 'bucket_seed_target_d4096_q8_m16384_k10', 'classification': 'seed-produced', 'coverage_only': False, 'production_policy': 'weave_only', 'external_fallback': None, 'guard_id': 'target_d4096_q8_m16384_k10', 'forced_fallback': False, 'selected_guard': SHAPE_DISPATCH_REGISTRY[0]['guard'], 'fallback': None, 'missing_weave_route': False, 'selected_seed': CONSUMED_SEED}
    return {'route': 'unsupported_shape', 'selected_route': 'unsupported_shape', 'selected_entrypoint': None, 'route_kind': 'unsupported', 'route_source': None, 'coverage_only': False, 'production_policy': 'weave_only', 'external_fallback': None, 'forced_fallback': bool(inputs.get('force_fallback', False))}

def route_trace_entry(label: str, inputs: dict[str, Any]) -> dict[str, Any]:
    return {'label': label, 'shape_key': label, 'B': int(inputs['B']), 'Q': int(inputs['Q']), 'M': int(inputs['M']), 'D': int(inputs['D']), 'K': int(inputs['K']), 'self_search': bool(inputs.get('self_search', False)), **route_info(inputs)}

def _launch_d4096_q8_tcgen05(inputs: dict[str, Any]) -> dict[str, Any]:
    import torch
    if not _KERNELS:
        _KERNELS.update(_compile_kernels())
    bsz = int(inputs['B'])
    q_rows = int(inputs['Q'])
    m_rows = int(inputs['M'])
    k = int(inputs['K'])
    num_q_tiles = math.ceil(q_rows / BLOCK_Q)
    total_m_tiles = math.ceil(m_rows / BLOCK_M)
    split_m = SPLIT_M
    partial_dist, partial_idx = _partial_scratch(inputs, split_m, num_q_tiles)
    _KERNELS['partial'].launch(grid=(bsz * num_q_tiles * split_m, 1, 1), block=(THREADS, 1, 1), args=[inputs['queries'], inputs['database'], partial_dist, partial_idx, bsz, q_rows, m_rows, split_m, num_q_tiles, total_m_tiles], shared_mem=SMEM_BYTES)
    _KERNELS['merge'].launch(grid=(bsz * q_rows, 1, 1), block=(MERGE_THREADS, 1, 1), args=[partial_dist, partial_idx, inputs['out_distances'], inputs['out_indices'], bsz, q_rows, k, split_m, num_q_tiles], shared_mem=mma.MERGE_SMEM_BYTES)
    return {'distances': inputs['out_distances'], 'indices': inputs['out_indices']}

def launch_for_eval(inputs: dict[str, Any]) -> dict[str, Any]:
    if _use_d4096_q8_tcgen05(inputs):
        return _launch_d4096_q8_tcgen05(inputs)
    raise ValueError('knn_search_target_d4096_q8_m16384_k10_0623_5ff7_v1 supports only B=1,Q=8,M=16384,D=4096,K=10,self_search=false')

def knn_search_compile_and_launch_target_d4096_q8_m16384_k10_5ff7(*, benchmark: bool=True) -> dict[str, Any]:
    from .._dispatch_runtime import evaluate
    result = evaluate(launch_for_eval, shapes=TARGET_SHAPES, benchmark=benchmark)
    result['passed'] = bool(result.get('summary', {}).get('all_correct'))
    print(result)
    return result
