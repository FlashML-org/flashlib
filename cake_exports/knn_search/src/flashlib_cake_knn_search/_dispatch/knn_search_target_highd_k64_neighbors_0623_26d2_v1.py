"""High-D K64 compact target-D kNN seed routes.

Minimum target architecture: sm_100a for the tcgen05/TMEM producer path. This
additive bucket-kernel module owns three compact target-D K64 fallback rows:
``D768/Q16/M32768/K64``, ``D1024/Q8/M16384/K64``, and
``D2048/Q4/M8192/K64``. Guard misses delegate to the current ff08 dispatcher;
the production dispatcher is not edited.
"""
from __future__ import annotations
from json import loads as _json_loads
from .._dispatch_runtime import _capture_cuTensorMapEncodeTiled, _decode_capture, _import_dispatch_module, _ir_proxy
import math
from typing import Any
from . import knn_search_dispatch0623_c0f1_plus6da2_41b3_v1 as parent
from . import knn_search_k64_q4096split79_twotile_oddevensort_fastmerge_0612_r34_11c1_v1 as sort64
from . import knn_search_lowd_d256_mma64_over48e9_0612_r34_48e9_v1 as d256_k64
THREADS = 256
MERGE_THREADS = 32
BLOCK_Q = 64
BLOCK_M = 64
D_STAGE = 256
D_MAX = 2048
K64_MAX = 64
MAX_SPLIT_M = 512
MERGE_SPLITS_PER_LANE_MAX = MAX_SPLIT_M // MERGE_THREADS
HIERMERGE_GROUPS = 8
HIERMERGE_LISTS_PER_GROUP_MAX = MAX_SPLIT_M // HIERMERGE_GROUPS
HIERMERGE_SPLITS_PER_LANE_MAX = HIERMERGE_LISTS_PER_GROUP_MAX // MERGE_THREADS
VEC = 16
PACK_WORDS = VEC // 2
Q_STAGE_VECS = BLOCK_Q * D_STAGE // VEC
Q_NORM_PARTS_MAX = D_MAX // VEC
DB_NORM_PARTS = 4
DB_NORM_CHUNK = D_STAGE // DB_NORM_PARTS
DB_NORM_PART_VECS = DB_NORM_CHUNK // VEC
SMEM_A_BYTES = BLOCK_Q * D_STAGE * 2
SMEM_B_BYTES = BLOCK_M * D_STAGE * 2
SMEM_Q_NORM_PART_BYTES = BLOCK_Q * Q_NORM_PARTS_MAX * 4
SMEM_DB_NORM_PART_BYTES = BLOCK_M * DB_NORM_PARTS * 4
SMEM_DB_NORM_BYTES = BLOCK_M * 4
SMEM_TILE_D_BYTES = BLOCK_Q * BLOCK_M * 4
SMEM_TILE_I_BYTES = BLOCK_Q * BLOCK_M * 4
SMEM_B_OFFSET = SMEM_A_BYTES
SMEM_Q_NORM_PART_OFFSET = SMEM_B_OFFSET + SMEM_B_BYTES
SMEM_DB_NORM_PART_OFFSET = SMEM_Q_NORM_PART_OFFSET + SMEM_Q_NORM_PART_BYTES
SMEM_DB_NORM_OFFSET = SMEM_DB_NORM_PART_OFFSET + SMEM_DB_NORM_PART_BYTES
SMEM_TILE_D_OFFSET = SMEM_DB_NORM_OFFSET + SMEM_DB_NORM_BYTES
SMEM_TILE_I_OFFSET = SMEM_TILE_D_OFFSET + SMEM_TILE_D_BYTES
SMEM_POOL_BYTES = SMEM_TILE_I_OFFSET + SMEM_TILE_I_BYTES + 256
WEAVE_SMEM_SYSTEM_BYTES = 1024
SMEM_BYTES = SMEM_POOL_BYTES + WEAVE_SMEM_SYSTEM_BYTES
MERGE_SMEM_BYTES = WEAVE_SMEM_SYSTEM_BYTES
ROUTE_TARGET_HIGHD_K64 = '26d2_target_highd_k64_compact_tcgen05'
SELECTED_SEED = 'weave-evolve-knn-search-26d2-highd-k64-neighbors'
ENTRYPOINT = 'loom.examples.weave.knn_search_target_highd_k64_neighbors_0623_26d2_v1:launch_for_eval'
TARGET_LABELS: tuple[str, ...] = ('target_d768_q16_m32768_k64', 'target_d1024_q8_m16384_k64', 'target_d2048_q4_m8192_k64')
TARGET_SHAPES: list[dict[str, Any]] = [{'label': 'target_d768_q16_m32768_k64', 'params': {'B': 1, 'Q': 16, 'M': 32768, 'D': 768, 'K': 64, 'dtype': 'bfloat16', 'seed': 611215, 'self_search': False, 'min_recall': 0.999}}, {'label': 'target_d1024_q8_m16384_k64', 'params': {'B': 1, 'Q': 8, 'M': 16384, 'D': 1024, 'K': 64, 'dtype': 'bfloat16', 'seed': 611216, 'self_search': False, 'min_recall': 0.999}}, {'label': 'target_d2048_q4_m8192_k64', 'params': {'B': 1, 'Q': 4, 'M': 8192, 'D': 2048, 'K': 64, 'dtype': 'bfloat16', 'seed': 611217, 'self_search': False, 'min_recall': 0.999}}]
SHAPE_DISPATCH_REGISTRY: tuple[dict[str, Any], ...] = ({'shape_key': '26d2_target_d768_q16_m32768_k64', 'label': 'target_d768_q16_m32768_k64', 'labels': ('target_d768_q16_m32768_k64',), 'shape': (1, 16, 32768, 768, 64, False), 'guard': 'B == 1 and Q == 16 and M == 32768 and D == 768 and K == 64 and not self_search and not forced_fallback and arch in {sm_100a,sm_103a}', 'route': ROUTE_TARGET_HIGHD_K64, 'entrypoint': ENTRYPOINT, 'source_entrypoint': 'direct-stride high-D K64 tcgen05 producer/merge in this module', 'selected_seed': SELECTED_SEED, 'producer_seed': SELECTED_SEED, 'source_round_doc': 'design_doc/active/weave_evolve_knn_search_round_148_26d2_highdk64.md', 'coverage_class': 'bucket_seed_target_d768_q16_m32768_k64', 'arch_requirement': 'sm_100a'}, {'shape_key': '26d2_target_d1024_q8_m16384_k64', 'label': 'target_d1024_q8_m16384_k64', 'labels': ('target_d1024_q8_m16384_k64',), 'shape': (1, 8, 16384, 1024, 64, False), 'guard': 'B == 1 and Q == 8 and M == 16384 and D == 1024 and K == 64 and not self_search and not forced_fallback and arch in {sm_100a,sm_103a}', 'route': ROUTE_TARGET_HIGHD_K64, 'entrypoint': ENTRYPOINT, 'source_entrypoint': 'direct-stride high-D K64 tcgen05 producer/merge in this module', 'selected_seed': SELECTED_SEED, 'producer_seed': SELECTED_SEED, 'source_round_doc': 'design_doc/active/weave_evolve_knn_search_round_148_26d2_highdk64.md', 'coverage_class': 'bucket_seed_target_d1024_q8_m16384_k64', 'arch_requirement': 'sm_100a'}, {'shape_key': '26d2_target_d2048_q4_m8192_k64', 'label': 'target_d2048_q4_m8192_k64', 'labels': ('target_d2048_q4_m8192_k64',), 'shape': (1, 4, 8192, 2048, 64, False), 'guard': 'B == 1 and Q == 4 and M == 8192 and D == 2048 and K == 64 and not self_search and not forced_fallback and arch in {sm_100a,sm_103a}', 'route': ROUTE_TARGET_HIGHD_K64, 'entrypoint': ENTRYPOINT, 'source_entrypoint': 'direct-stride high-D K64 tcgen05 producer/merge in this module', 'selected_seed': SELECTED_SEED, 'producer_seed': SELECTED_SEED, 'source_round_doc': 'design_doc/active/weave_evolve_knn_search_round_148_26d2_highdk64.md', 'coverage_class': 'bucket_seed_target_d2048_q4_m8192_k64', 'arch_requirement': 'sm_100a'})
_KERNELS: dict[int, dict[str, Any]] = {}
_SCRATCH: dict[tuple[int, int, int, int, int, int, int, str], tuple[Any, Any]] = {}
_GROUP_SCRATCH: dict[tuple[int, int, int, int, int, int, str], tuple[Any, Any]] = {}
_q64_mma_issue = _ir_proxy('loom.examples.weave.knn_search_target_highd_k64_neighbors_0623_26d2_v1:_q64_mma_issue', 256)
_accumulate_q_norm_direct_d = _ir_proxy('loom.examples.weave.knn_search_target_highd_k64_neighbors_0623_26d2_v1:_accumulate_q_norm_direct_d', 256)
_stage_q_pass_direct_d = _ir_proxy('loom.examples.weave.knn_search_target_highd_k64_neighbors_0623_26d2_v1:_stage_q_pass_direct_d', 256)
_stage_database_pass_direct_d = _ir_proxy('loom.examples.weave.knn_search_target_highd_k64_neighbors_0623_26d2_v1:_stage_database_pass_direct_d', 256)
_accumulate_db_norm_pass = _ir_proxy('loom.examples.weave.knn_search_target_highd_k64_neighbors_0623_26d2_v1:_accumulate_db_norm_pass', 256)
_capture_row16_tile_to_smem = _ir_proxy('loom.examples.weave.knn_search_target_highd_k64_neighbors_0623_26d2_v1:_capture_row16_tile_to_smem', 256)
knn_search_target_highd_k64_partial_26d2_v1 = _decode_capture(_json_loads('{"__ir__": "knn_search_target_highd_k64_partial_26d2_v1", "arg_keys": ["queries", "database", "partial_distances", "partial_indices", "B", "Q", "M", "split_m", "num_q_tiles"], "cluster_dims": [1, 1, 1], "computed_smem_bytes": 133632, "constants": [["D_ORIG_", 2048], ["NUM_D_PASSES_", 8], ["Q_NORM_PARTS_", 128], ["K_MAX_", 64]], "cta_group": 1, "threads": 256}'))
knn_search_target_highd_k64_merge_26d2_v1 = _decode_capture(_json_loads('{"__ir__": "knn_search_target_highd_k64_merge_26d2_v1", "arg_keys": ["partial_distances", "partial_indices", "out_distances", "out_indices", "B", "Q", "K", "split_m", "num_q_tiles"], "cluster_dims": [1, 1, 1], "computed_smem_bytes": 0, "constants": [["K_MAX_", 64]], "cta_group": 1, "threads": 32}'))
knn_search_target_highd_k64_group_merge_26d2_v1 = _decode_capture(_json_loads('{"__ir__": "knn_search_target_highd_k64_group_merge_26d2_v1", "arg_keys": ["partial_distances", "partial_indices", "group_distances", "group_indices", "B", "Q", "partial_list_count", "num_q_tiles"], "cluster_dims": [1, 1, 1], "computed_smem_bytes": 0, "constants": [["K_MAX_", 64]], "cta_group": 1, "threads": 32}'))
knn_search_target_highd_k64_final_merge_26d2_v1 = _decode_capture(_json_loads('{"__ir__": "knn_search_target_highd_k64_final_merge_26d2_v1", "arg_keys": ["group_distances", "group_indices", "out_distances", "out_indices", "B", "Q", "K"], "cluster_dims": [1, 1, 1], "computed_smem_bytes": 0, "constants": [["K_MAX_", 64]], "cta_group": 1, "threads": 32}'))
partial_ir = _decode_capture(_json_loads('{"__ir__": "knn_search_target_highd_k64_partial_26d2_v1", "arg_keys": ["queries", "database", "partial_distances", "partial_indices", "B", "Q", "M", "split_m", "num_q_tiles"], "cluster_dims": [1, 1, 1], "computed_smem_bytes": 133632, "constants": [["D_ORIG_", 2048], ["NUM_D_PASSES_", 8], ["Q_NORM_PARTS_", 128], ["K_MAX_", 64]], "cta_group": 1, "threads": 256}'))
flat_merge_ir = _decode_capture(_json_loads('{"__ir__": "knn_search_target_highd_k64_merge_26d2_v1", "arg_keys": ["partial_distances", "partial_indices", "out_distances", "out_indices", "B", "Q", "K", "split_m", "num_q_tiles"], "cluster_dims": [1, 1, 1], "computed_smem_bytes": 0, "constants": [["K_MAX_", 64]], "cta_group": 1, "threads": 32}'))
group_merge_ir = _decode_capture(_json_loads('{"__ir__": "knn_search_target_highd_k64_group_merge_26d2_v1", "arg_keys": ["partial_distances", "partial_indices", "group_distances", "group_indices", "B", "Q", "partial_list_count", "num_q_tiles"], "cluster_dims": [1, 1, 1], "computed_smem_bytes": 0, "constants": [["K_MAX_", 64]], "cta_group": 1, "threads": 32}'))
final_merge_ir = _decode_capture(_json_loads('{"__ir__": "knn_search_target_highd_k64_final_merge_26d2_v1", "arg_keys": ["group_distances", "group_indices", "out_distances", "out_indices", "B", "Q", "K"], "cluster_dims": [1, 1, 1], "computed_smem_bytes": 0, "constants": [["K_MAX_", 64]], "cta_group": 1, "threads": 32}'))
merge_ir = _decode_capture(_json_loads('{"__ir__": "knn_search_target_highd_k64_group_merge_26d2_v1", "arg_keys": ["partial_distances", "partial_indices", "group_distances", "group_indices", "B", "Q", "partial_list_count", "num_q_tiles"], "cluster_dims": [1, 1, 1], "computed_smem_bytes": 0, "constants": [["K_MAX_", 64]], "cta_group": 1, "threads": 32}'))
ir = _decode_capture(_json_loads('{"__ir__": "knn_search_target_highd_k64_partial_26d2_v1", "arg_keys": ["queries", "database", "partial_distances", "partial_indices", "B", "Q", "M", "split_m", "num_q_tiles"], "cluster_dims": [1, 1, 1], "computed_smem_bytes": 133632, "constants": [["D_ORIG_", 2048], ["NUM_D_PASSES_", 8], ["Q_NORM_PARTS_", 128], ["K_MAX_", 64]], "cta_group": 1, "threads": 256}'))

def _shape_key(inputs: dict[str, Any]) -> tuple[int, int, int, int, int, bool]:
    return (int(inputs.get('B', 1)), int(inputs['Q']), int(inputs['M']), int(inputs['D']), int(inputs['K']), bool(inputs.get('self_search', False)))

def _forced_fallback(inputs: dict[str, Any]) -> bool:
    return bool(inputs.get('force_fallback', False))

def _tcgen05_capable_arch() -> bool:
    return d256_k64._tcgen05_capable_arch()

def _active_entry(inputs: dict[str, Any]) -> dict[str, Any] | None:
    if _forced_fallback(inputs) or not _tcgen05_capable_arch():
        return None
    shape = _shape_key(inputs)
    for entry in SHAPE_DISPATCH_REGISTRY:
        if shape == entry['shape']:
            return entry
    return None

def _guard_order() -> list[str]:
    return [*(str(entry['shape_key']) for entry in SHAPE_DISPATCH_REGISTRY), *parent._guard_order(parent.PROFILE_ALL)]

def selected_route(inputs: dict[str, Any]) -> str:
    entry = _active_entry(inputs)
    if entry is not None:
        return str(entry['route'])
    return parent.selected_route(inputs)

def selected_route_name(inputs: dict[str, Any]) -> str:
    return selected_route(inputs)

def _entry_info(inputs: dict[str, Any], entry: dict[str, Any]) -> dict[str, Any]:
    parent_info = dict(parent.route_info(inputs))
    parent_route = str(parent_info.get('selected_route') or parent_info.get('route') or parent.selected_route(inputs))
    return {'route': entry['route'], 'selected_route': entry['route'], 'selected_entrypoint': entry['entrypoint'], 'source_entrypoint': entry['source_entrypoint'], 'parent_route': parent_route, 'replaced_route': parent_route, 'route_kind': 'specialized', 'route_source': 'shape-specific-seed', 'coverage_class': entry['coverage_class'], 'classification': 'seed-produced', 'coverage_only': False, 'production_policy': 'weave_only', 'external_fallback': None, 'guard_order': _guard_order(), 'guard_id': entry['shape_key'], 'selected_guard': entry['guard'], 'guard_condition': entry['guard'], 'fallback': parent_route, 'missing_weave_route': False, 'forced_fallback': False, 'selected_seed': entry['selected_seed'], 'selected_seed_task': entry['selected_seed'], 'producer_seed': entry['producer_seed'], 'producer_seed_task': entry['producer_seed'], 'source_round_doc': entry['source_round_doc'], 'arch_requirement': entry['arch_requirement']}

def route_info(inputs: dict[str, Any]) -> dict[str, Any]:
    entry = _active_entry(inputs)
    if entry is not None:
        return _entry_info(inputs, entry)
    info = dict(parent.route_info(inputs))
    info['guard_order'] = _guard_order()
    info.setdefault('production_policy', 'weave_only')
    info.setdefault('external_fallback', None)
    return info

def route_trace_entry(label: str, inputs: dict[str, Any]) -> dict[str, Any]:
    return {'label': label, 'shape_key': label, 'B': int(inputs['B']), 'Q': int(inputs['Q']), 'M': int(inputs['M']), 'D': int(inputs['D']), 'K': int(inputs['K']), 'self_search': bool(inputs.get('self_search', False)), 'force_fallback': bool(inputs.get('force_fallback', False)), **route_info(inputs)}

def _compile_kernels(original_d: int) -> dict[str, Any]:
    from .._dispatch_runtime import generate_kernel
    from .._dispatch_runtime import _cuda_include_dirs
    from .._dispatch_runtime import compile_cuda, detect_gpu_arch
    from .._dispatch_runtime import CUDAKernel
    arch = detect_gpu_arch()
    include_dirs = _cuda_include_dirs()
    partial_source = generate_kernel(partial_ir, validate=False, smem_bytes=SMEM_BYTES, D_ORIG_=int(original_d), NUM_D_PASSES_=int(math.ceil(original_d / D_STAGE)), Q_NORM_PARTS_=int(math.ceil(original_d / VEC)), K_MAX_=K64_MAX)
    merge_source = generate_kernel(group_merge_ir, validate=False, smem_bytes=MERGE_SMEM_BYTES, K_MAX_=K64_MAX)
    final_source = generate_kernel(final_merge_ir, validate=False, smem_bytes=MERGE_SMEM_BYTES, K_MAX_=K64_MAX)
    partial_cubin = compile_cuda(partial_source, arch=arch, options=['--use_fast_math'], include_dirs=include_dirs)
    merge_cubin = compile_cuda(merge_source, arch=arch, options=['--use_fast_math'], include_dirs=include_dirs)
    final_cubin = compile_cuda(final_source, arch=arch, options=['--use_fast_math'], include_dirs=include_dirs)
    return {'partial': CUDAKernel(partial_cubin, ''.join(['kernel_', format(partial_ir.symbol, '')])), 'group_merge': CUDAKernel(merge_cubin, ''.join(['kernel_', format(group_merge_ir.symbol, '')])), 'final_merge': CUDAKernel(final_cubin, ''.join(['kernel_', format(final_merge_ir.symbol, '')]))}

def _ensure_kernels(original_d: int) -> dict[str, Any]:
    kernels = _KERNELS.get(original_d)
    if kernels is None:
        kernels = _compile_kernels(original_d)
        _KERNELS[original_d] = kernels
    return kernels

def _scratch(inputs: dict[str, Any], split_m: int, num_q_tiles: int) -> tuple[Any, Any]:
    import torch
    key = (int(inputs['B']), int(inputs['Q']), int(inputs['M']), int(inputs['D']), int(split_m), int(num_q_tiles), int(inputs['queries'].device.index or 0), str(inputs['queries'].dtype))
    cached = _SCRATCH.get(key)
    if cached is None:
        shape = (int(inputs['B']), int(num_q_tiles), int(split_m), BLOCK_Q, K64_MAX)
        cached = (torch.empty(shape, dtype=torch.float32, device=inputs['queries'].device), torch.empty(shape, dtype=torch.int32, device=inputs['queries'].device))
        _SCRATCH[key] = cached
    return cached

def _group_scratch(inputs: dict[str, Any]) -> tuple[Any, Any]:
    import torch
    key = (int(inputs['B']), int(inputs['Q']), int(inputs['K']), int(inputs['queries'].device.index or 0), id(inputs['queries']), str(inputs['queries'].dtype))
    cached = _GROUP_SCRATCH.get(key)
    if cached is None:
        shape = (int(inputs['B']), int(inputs['Q']), HIERMERGE_GROUPS, K64_MAX)
        cached = (torch.empty(shape, dtype=torch.float32, device=inputs['queries'].device), torch.empty(shape, dtype=torch.int32, device=inputs['queries'].device))
        _GROUP_SCRATCH[key] = cached
    return cached

def _launch_highd_k64(inputs: dict[str, Any]) -> dict[str, Any]:
    kernels = _ensure_kernels(int(inputs['D']))
    bsz = int(inputs['B'])
    q_rows = int(inputs['Q'])
    m_rows = int(inputs['M'])
    k = int(inputs['K'])
    num_q_tiles = math.ceil(q_rows / BLOCK_Q)
    split_m = math.ceil(m_rows / BLOCK_M)
    partial_dist, partial_idx = _scratch(inputs, split_m, num_q_tiles)
    group_dist, group_idx = _group_scratch(inputs)
    kernels['partial'].launch(grid=(bsz * num_q_tiles * split_m, 1, 1), block=(THREADS, 1, 1), args=[inputs['queries'], inputs['database'], partial_dist, partial_idx, bsz, q_rows, m_rows, split_m, num_q_tiles], shared_mem=SMEM_BYTES)
    kernels['group_merge'].launch(grid=(bsz * q_rows * HIERMERGE_GROUPS, 1, 1), block=(MERGE_THREADS, 1, 1), args=[partial_dist, partial_idx, group_dist, group_idx, bsz, q_rows, split_m, num_q_tiles], shared_mem=MERGE_SMEM_BYTES)
    kernels['final_merge'].launch(grid=(bsz * q_rows, 1, 1), block=(MERGE_THREADS, 1, 1), args=[group_dist, group_idx, inputs['out_distances'], inputs['out_indices'], bsz, q_rows, k], shared_mem=MERGE_SMEM_BYTES)
    return {'distances': inputs['out_distances'], 'indices': inputs['out_indices']}

def launch_for_eval(inputs: dict[str, Any]) -> dict[str, Any]:
    if _active_entry(inputs) is not None:
        return _launch_highd_k64(inputs)
    return parent.launch_for_eval(inputs)

def knn_search_compile_and_launch_target_highd_k64_neighbors_0623_26d2(*, benchmark: bool=True, shapes: list[dict[str, Any]] | None=None) -> dict[str, Any]:
    from .._dispatch_runtime import evaluate
    result = evaluate(launch_for_eval, shapes=TARGET_SHAPES if shapes is None else shapes, benchmark=benchmark)
    result['passed'] = bool(result.get('summary', {}).get('all_correct'))
    print(result)
    return result
