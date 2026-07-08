"""D512/Q32/K64 tcgen05 repair seed for exact BF16 squared-L2 kNN.

Minimum target architecture: sm_100a. This additive bucket-kernel candidate
targets ``blind_ext_dyn_d512_k64_q32_m32768`` without editing the production
dispatcher. It preserves the K64 split-M producer/merge structure from the
ranked D512 path but reduces the tcgen05 query tile from Q128 to Q64, so the
Q32 target row does not pay for three inactive 32-row cohorts.
"""
from __future__ import annotations
from json import loads as _json_loads
from .._dispatch_runtime import _capture_cuTensorMapEncodeTiled, _decode_capture, _import_dispatch_module, _ir_proxy
import math
from typing import Any
from .._dispatch_runtime import select_named_shapes
from . import knn_search_dynamic_d512_ivf_6bea_r118_v1 as parent
from . import knn_search_k64_q4096split79_twotile_oddevensort_fastmerge_0612_r34_11c1_v1 as sort_parent
from . import knn_search_lowd_d256_mma64_over48e9_0612_r34_48e9_v1 as d256_k64
THREADS = 256
BLOCK_Q = 64
BLOCK_M = 64
D_ORIGINAL = 512
D_STAGE = 256
K64_MAX = 64
Q_ROWS = 32
M_ROWS = 32768
Q32_SPLIT_M = 512
MMA_POST_MMA_COL_COHORTS = 1
Q32_PARTIAL_LISTS = Q32_SPLIT_M * MMA_POST_MMA_COL_COHORTS
MERGE_THREADS = 32
MERGE_SPLITS_PER_LANE_MAX = (Q32_PARTIAL_LISTS + MERGE_THREADS - 1) // MERGE_THREADS
MMA_STAGE_VEC_ELEMS = 16
MMA_STAGE_PACK_WORDS = MMA_STAGE_VEC_ELEMS // 2
MMA_Q_STAGE_VECS = BLOCK_Q * D_STAGE // MMA_STAGE_VEC_ELEMS
MMA_Q_NORM_PARTS = D_ORIGINAL // MMA_STAGE_VEC_ELEMS
MMA_DB_NORM_PARTS = 4
MMA_DB_NORM_CHUNK = D_STAGE // MMA_DB_NORM_PARTS
MMA_DB_NORM_PART_VECS = MMA_DB_NORM_CHUNK // MMA_STAGE_VEC_ELEMS
MMA_SMEM_A_BYTES = BLOCK_Q * D_STAGE * 2
MMA_SMEM_B_BYTES = BLOCK_M * D_STAGE * 2
MMA_SMEM_Q_NORM_PART_BYTES = BLOCK_Q * MMA_Q_NORM_PARTS * 4
MMA_SMEM_DB_NORM_PART_BYTES = BLOCK_M * MMA_DB_NORM_PARTS * 4
MMA_SMEM_DB_NORM_BYTES = BLOCK_M * 4
MMA_SMEM_B_OFFSET = MMA_SMEM_A_BYTES
MMA_SMEM_Q_NORM_PART_OFFSET = MMA_SMEM_B_OFFSET + MMA_SMEM_B_BYTES
MMA_SMEM_DB_NORM_PART_OFFSET = MMA_SMEM_Q_NORM_PART_OFFSET + MMA_SMEM_Q_NORM_PART_BYTES
MMA_SMEM_DB_NORM_OFFSET = MMA_SMEM_DB_NORM_PART_OFFSET + MMA_SMEM_DB_NORM_PART_BYTES
MMA_SMEM_TILE_D_OFFSET = MMA_SMEM_DB_NORM_OFFSET + MMA_SMEM_DB_NORM_BYTES
MMA_SMEM_TILE_I_OFFSET = MMA_SMEM_TILE_D_OFFSET + BLOCK_Q * BLOCK_M * 4
WEAVE_SMEM_SYSTEM_BYTES = 1024
MMA_SMEM_POOL_BYTES = MMA_SMEM_TILE_I_OFFSET + BLOCK_Q * BLOCK_M * 4 + 256
MMA_SMEM_BYTES = MMA_SMEM_POOL_BYTES + WEAVE_SMEM_SYSTEM_BYTES
MERGE_SMEM_BYTES = WEAVE_SMEM_SYSTEM_BYTES
ROUTE_D512_Q32_K64_Q64_TILE = '6bea_r119_d512_q32_k64_q64tile_tcgen05'
CONSUMED_SEED = 'weave-evolve-knn-search-6bea-r119-d512-q32-k64-q64tile'
REPLACED_SEED = parent.REUSED_D512_SEED
TARGET_LABELS: tuple[str, ...] = ('blind_ext_dyn_d512_k64_q32_m32768',)
TARGET_SHAPES = _decode_capture(_json_loads('[{"__dict_items__": [["label", "blind_ext_dyn_d512_k64_q32_m32768"], ["params", {"__dict_items__": [["B", 1], ["Q", 32], ["M", 32768], ["D", 512], ["K", 64], ["dtype", "bfloat16"], ["seed", 610930], ["self_search", false], ["min_recall", 0.999]]}]]}]'))
_KERNELS: dict[str, Any] = {}
_PARTIAL_SCRATCH: dict[tuple[int, int, int, int, int, int, int, int, str], tuple[Any, Any]] = {}
_q64_k64_mma_issue = _ir_proxy('loom.examples.weave.knn_search_dynamic_d512_k64_q32_6bea_r119_v1:_q64_k64_mma_issue', 256)
_accumulate_q_norm_d512_q64 = _ir_proxy('loom.examples.weave.knn_search_dynamic_d512_k64_q32_6bea_r119_v1:_accumulate_q_norm_d512_q64', 256)
_stage_q_pass_d512_q64 = _ir_proxy('loom.examples.weave.knn_search_dynamic_d512_k64_q32_6bea_r119_v1:_stage_q_pass_d512_q64', 256)
_stage_database_pass_d512_q64 = _ir_proxy('loom.examples.weave.knn_search_dynamic_d512_k64_q32_6bea_r119_v1:_stage_database_pass_d512_q64', 256)
_capture_row16_tile_to_smem = _ir_proxy('loom.examples.weave.knn_search_dynamic_d512_k64_q32_6bea_r119_v1:_capture_row16_tile_to_smem', 256)
knn_search_d512_q32_k64_q64tile_partial_6bea_r119_v1 = _decode_capture(_json_loads('{"__ir__": "knn_search_d512_q32_k64_q64tile_partial_6bea_r119_v1", "arg_keys": ["queries", "database", "partial_distances", "partial_indices", "B", "Q", "M", "split_m", "num_q_tiles", "total_m_tiles", "tiles_per_split", "partial_list_count"], "cluster_dims": [1, 1, 1], "computed_smem_bytes": 109056, "constants": [["K_MAX_", 64]], "cta_group": 1, "threads": 256}'))
knn_search_d512_q32_k64_q64tile_merge_6bea_r119_v1 = _decode_capture(_json_loads('{"__ir__": "knn_search_d512_q32_k64_q64tile_merge_6bea_r119_v1", "arg_keys": ["partial_distances", "partial_indices", "out_distances", "out_indices", "B", "Q", "K", "partial_list_count", "num_q_tiles"], "cluster_dims": [1, 1, 1], "computed_smem_bytes": 0, "constants": [["K_MAX_", 64]], "cta_group": 1, "threads": 32}'))
partial_ir = _decode_capture(_json_loads('{"__ir__": "knn_search_d512_q32_k64_q64tile_partial_6bea_r119_v1", "arg_keys": ["queries", "database", "partial_distances", "partial_indices", "B", "Q", "M", "split_m", "num_q_tiles", "total_m_tiles", "tiles_per_split", "partial_list_count"], "cluster_dims": [1, 1, 1], "computed_smem_bytes": 109056, "constants": [["K_MAX_", 64]], "cta_group": 1, "threads": 256}'))
merge_ir = _decode_capture(_json_loads('{"__ir__": "knn_search_d512_q32_k64_q64tile_merge_6bea_r119_v1", "arg_keys": ["partial_distances", "partial_indices", "out_distances", "out_indices", "B", "Q", "K", "partial_list_count", "num_q_tiles"], "cluster_dims": [1, 1, 1], "computed_smem_bytes": 0, "constants": [["K_MAX_", 64]], "cta_group": 1, "threads": 32}'))
ir = _decode_capture(_json_loads('{"__ir__": "knn_search_d512_q32_k64_q64tile_partial_6bea_r119_v1", "arg_keys": ["queries", "database", "partial_distances", "partial_indices", "B", "Q", "M", "split_m", "num_q_tiles", "total_m_tiles", "tiles_per_split", "partial_list_count"], "cluster_dims": [1, 1, 1], "computed_smem_bytes": 109056, "constants": [["K_MAX_", 64]], "cta_group": 1, "threads": 256}'))
SHAPE_DISPATCH_REGISTRY: tuple[dict[str, Any], ...] = ({'overlay': '6bea_r119_d512_q32_k64_q64tile_tcgen05', 'shape_key': 'blind_ext_dyn_d512_k64_q32_m32768', 'labels': TARGET_LABELS, 'guard': 'B == 1 and Q == 32 and M == 32768 and D == 512 and K == 64 and not self_search and not forced_fallback and tcgen05_capable_arch', 'route': ROUTE_D512_Q32_K64_Q64_TILE, 'entrypoint': 'loom.examples.weave.knn_search_dynamic_d512_k64_q32_6bea_r119_v1:launch_for_eval', 'selected_seed': CONSUMED_SEED, 'source_task': CONSUMED_SEED, 'coverage_class': 'bucket_seed_dynamic_d512_q32_k64_q64tile', 'workflow_mode': 'generalize_auto_tuning', 'auto_tuning_stage': 'bucket-kernel'},)

def _tcgen05_capable_arch() -> bool:
    return bool(d256_k64._tcgen05_capable_arch())

def _use_d512_q32_k64_q64tile(inputs: dict[str, Any]) -> bool:
    return int(inputs['B']) == 1 and int(inputs['Q']) == Q_ROWS and (int(inputs['M']) == M_ROWS) and (int(inputs['D']) == D_ORIGINAL) and (int(inputs['K']) == K64_MAX) and (not bool(inputs.get('self_search', False))) and (not bool(inputs.get('force_fallback', False))) and _tcgen05_capable_arch()

def _compile_kernels() -> dict[str, Any]:
    return _decode_capture(_json_loads('{"merge": {"__kernel__": "dispatch_kernel_0444"}, "partial": {"__kernel__": "dispatch_kernel_0443"}}'))

def _ensure_kernels() -> dict[str, Any]:
    if not _KERNELS:
        _KERNELS.update(_compile_kernels())
    return _KERNELS

def _partial_scratch(inputs: dict[str, Any], partial_list_count: int, num_q_tiles: int) -> tuple[Any, Any]:
    import torch
    key = (int(inputs['B']), int(inputs['Q']), int(inputs['M']), int(inputs['D']), int(partial_list_count), int(num_q_tiles), int(inputs['queries'].device.index or 0), id(inputs['queries']), str(inputs['queries'].dtype))
    cached = _PARTIAL_SCRATCH.get(key)
    if cached is None:
        shape = (int(inputs['B']), int(num_q_tiles), int(partial_list_count), BLOCK_Q, K64_MAX)
        cached = (torch.empty(shape, dtype=torch.float32, device=inputs['queries'].device), torch.empty(shape, dtype=torch.int32, device=inputs['queries'].device))
        _PARTIAL_SCRATCH[key] = cached
    return cached

def selected_route(inputs: dict[str, Any]) -> str:
    if _use_d512_q32_k64_q64tile(inputs):
        return ROUTE_D512_Q32_K64_Q64_TILE
    return parent.selected_route(inputs)

def selected_route_name(inputs: dict[str, Any]) -> str:
    return selected_route(inputs)

def route_info(inputs: dict[str, Any]) -> dict[str, Any]:
    if _use_d512_q32_k64_q64tile(inputs):
        parent_info = dict(parent.route_info(inputs))
        return {'route': ROUTE_D512_Q32_K64_Q64_TILE, 'selected_route': ROUTE_D512_Q32_K64_Q64_TILE, 'selected_entrypoint': 'loom.examples.weave.knn_search_dynamic_d512_k64_q32_6bea_r119_v1:launch_for_eval', 'parent_route': parent_info.get('selected_route', parent.selected_route(inputs)), 'replaced_route': parent_info.get('selected_route', parent.selected_route(inputs)), 'route_kind': 'specialized', 'route_source': 'bucket-specific-seed', 'coverage_class': 'bucket_seed_dynamic_d512_q32_k64_q64tile', 'classification': 'seed-produced', 'coverage_only': False, 'production_policy': 'weave_only', 'external_fallback': None, 'guard_id': '6bea_r119_d512_q32_k64_q64tile', 'forced_fallback': False, 'selected_guard': SHAPE_DISPATCH_REGISTRY[0]['guard'], 'fallback': parent_info.get('selected_route', parent.selected_route(inputs)), 'missing_weave_route': False, 'selected_seed': CONSUMED_SEED, 'replaced_seed': REPLACED_SEED}
    return parent.route_info(inputs)

def route_trace_entry(label: str, inputs: dict[str, Any]) -> dict[str, Any]:
    return {'label': label, 'shape_key': label, 'B': int(inputs['B']), 'Q': int(inputs['Q']), 'M': int(inputs['M']), 'D': int(inputs['D']), 'K': int(inputs['K']), 'self_search': bool(inputs.get('self_search', False)), **route_info(inputs)}

def _launch_d512_q32_k64_q64tile(inputs: dict[str, Any]) -> dict[str, Any]:
    kernels = _ensure_kernels()
    bsz = int(inputs['B'])
    q_rows = int(inputs['Q'])
    m_rows = int(inputs['M'])
    k = int(inputs['K'])
    num_q_tiles = math.ceil(q_rows / BLOCK_Q)
    total_m_tiles = math.ceil(m_rows / BLOCK_M)
    split_m = min(Q32_SPLIT_M, total_m_tiles)
    tiles_per_split = math.ceil(total_m_tiles / split_m)
    partial_list_count = split_m * MMA_POST_MMA_COL_COHORTS
    partial_dist, partial_idx = _partial_scratch(inputs, partial_list_count, num_q_tiles)
    kernels['partial'].launch(grid=(bsz * num_q_tiles * split_m, 1, 1), block=(THREADS, 1, 1), args=[inputs['queries'], inputs['database'], partial_dist, partial_idx, bsz, q_rows, m_rows, split_m, num_q_tiles, total_m_tiles, tiles_per_split, partial_list_count], shared_mem=MMA_SMEM_BYTES)
    kernels['merge'].launch(grid=(bsz * q_rows, 1, 1), block=(MERGE_THREADS, 1, 1), args=[partial_dist, partial_idx, inputs['out_distances'], inputs['out_indices'], bsz, q_rows, k, partial_list_count, num_q_tiles], shared_mem=MERGE_SMEM_BYTES)
    return {'distances': inputs['out_distances'], 'indices': inputs['out_indices']}

def launch_for_eval(inputs: dict[str, Any]) -> dict[str, Any]:
    if _use_d512_q32_k64_q64tile(inputs):
        return _launch_d512_q32_k64_q64tile(inputs)
    return parent.launch_for_eval(inputs)

def knn_search_compile_and_launch_dynamic_d512_k64_q32_6bea_r119(*, benchmark: bool=True, shapes: list[dict[str, Any]] | None=None) -> dict[str, Any]:
    from .._dispatch_runtime import evaluate
    selected = TARGET_SHAPES if shapes is None else shapes
    result = evaluate(launch_for_eval, shapes=selected, benchmark=benchmark)
    result['passed'] = bool(result.get('summary', {}).get('all_correct'))
    print(result)
    return result
