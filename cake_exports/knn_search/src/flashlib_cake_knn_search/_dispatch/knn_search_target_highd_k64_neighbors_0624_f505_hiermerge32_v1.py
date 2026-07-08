"""High-D K64 compact target-D 32-group merge kNN seed route.

Minimum target architecture: sm_100a for the tcgen05/TMEM producer path. This
additive bucket-kernel module continues the 26d2 high-D K64 lane for the exact
``D768/Q16/M32768/K64``, ``D1024/Q8/M16384/K64``, and
``D2048/Q4/M8192/K64`` rows. It keeps the proven 26d2 tcgen05 producer and
changes only the hierarchical merge fan-in from 8 groups to 32 groups.
"""
from __future__ import annotations
from json import loads as _json_loads
from .._dispatch_runtime import _capture_cuTensorMapEncodeTiled, _decode_capture, _import_dispatch_module, _ir_proxy
import math
from typing import Any
from . import knn_search_target_highd_k64_neighbors_0623_26d2_v1 as parent_seed
THREADS = parent_seed.THREADS
MERGE_THREADS = parent_seed.MERGE_THREADS
BLOCK_Q = parent_seed.BLOCK_Q
BLOCK_M = parent_seed.BLOCK_M
D_STAGE = parent_seed.D_STAGE
D_MAX = parent_seed.D_MAX
K64_MAX = parent_seed.K64_MAX
MAX_SPLIT_M = parent_seed.MAX_SPLIT_M
HIERMERGE_GROUPS = 32
HIERMERGE_LISTS_PER_GROUP_MAX = MAX_SPLIT_M // HIERMERGE_GROUPS
HIERMERGE_SPLITS_PER_LANE_MAX = (HIERMERGE_LISTS_PER_GROUP_MAX + MERGE_THREADS - 1) // MERGE_THREADS
SMEM_BYTES = parent_seed.SMEM_BYTES
MERGE_SMEM_BYTES = parent_seed.MERGE_SMEM_BYTES
ROUTE_TARGET_HIGHD_K64_HIERMERGE32 = 'f505_target_highd_k64_compact_hiermerge32_tcgen05'
SELECTED_SEED = 'weave-evolve-knn-search-f505-highd-k64-hiermerge32'
PARENT_SEED = parent_seed.SELECTED_SEED
ENTRYPOINT = 'loom.examples.weave.knn_search_target_highd_k64_neighbors_0624_f505_hiermerge32_v1:launch_for_eval'
ROUND_DOC = 'design_doc/active/weave_evolve_knn_search_round_149_f505_highdk64_hiermerge32.md'
TARGET_LABELS = parent_seed.TARGET_LABELS
TARGET_SHAPES = parent_seed.TARGET_SHAPES
SHAPE_DISPATCH_REGISTRY = _decode_capture(_json_loads('{"__tuple__": [{"__dict_items__": [["shape_key", "f505_hiermerge32_target_d768_q16_m32768_k64"], ["label", "target_d768_q16_m32768_k64"], ["labels", {"__tuple__": ["target_d768_q16_m32768_k64"]}], ["shape", {"__tuple__": [1, 16, 32768, 768, 64, false]}], ["guard", "B == 1 and Q == 16 and M == 32768 and D == 768 and K == 64 and not self_search and not forced_fallback and arch in {sm_100a,sm_103a}"], ["route", "f505_target_highd_k64_compact_hiermerge32_tcgen05"], ["entrypoint", "loom.examples.weave.knn_search_target_highd_k64_neighbors_0624_f505_hiermerge32_v1:launch_for_eval"], ["source_entrypoint", "26d2 high-D K64 tcgen05 producer with 32-group merge in this module"], ["selected_seed", "weave-evolve-knn-search-f505-highd-k64-hiermerge32"], ["producer_seed", "weave-evolve-knn-search-f505-highd-k64-hiermerge32"], ["source_round_doc", "design_doc/active/weave_evolve_knn_search_round_149_f505_highdk64_hiermerge32.md"], ["coverage_class", "bucket_seed_target_d768_q16_m32768_k64"], ["arch_requirement", "sm_100a"], ["parent_seed", "weave-evolve-knn-search-26d2-highd-k64-neighbors"]]}, {"__dict_items__": [["shape_key", "f505_hiermerge32_target_d1024_q8_m16384_k64"], ["label", "target_d1024_q8_m16384_k64"], ["labels", {"__tuple__": ["target_d1024_q8_m16384_k64"]}], ["shape", {"__tuple__": [1, 8, 16384, 1024, 64, false]}], ["guard", "B == 1 and Q == 8 and M == 16384 and D == 1024 and K == 64 and not self_search and not forced_fallback and arch in {sm_100a,sm_103a}"], ["route", "f505_target_highd_k64_compact_hiermerge32_tcgen05"], ["entrypoint", "loom.examples.weave.knn_search_target_highd_k64_neighbors_0624_f505_hiermerge32_v1:launch_for_eval"], ["source_entrypoint", "26d2 high-D K64 tcgen05 producer with 32-group merge in this module"], ["selected_seed", "weave-evolve-knn-search-f505-highd-k64-hiermerge32"], ["producer_seed", "weave-evolve-knn-search-f505-highd-k64-hiermerge32"], ["source_round_doc", "design_doc/active/weave_evolve_knn_search_round_149_f505_highdk64_hiermerge32.md"], ["coverage_class", "bucket_seed_target_d1024_q8_m16384_k64"], ["arch_requirement", "sm_100a"], ["parent_seed", "weave-evolve-knn-search-26d2-highd-k64-neighbors"]]}, {"__dict_items__": [["shape_key", "f505_hiermerge32_target_d2048_q4_m8192_k64"], ["label", "target_d2048_q4_m8192_k64"], ["labels", {"__tuple__": ["target_d2048_q4_m8192_k64"]}], ["shape", {"__tuple__": [1, 4, 8192, 2048, 64, false]}], ["guard", "B == 1 and Q == 4 and M == 8192 and D == 2048 and K == 64 and not self_search and not forced_fallback and arch in {sm_100a,sm_103a}"], ["route", "f505_target_highd_k64_compact_hiermerge32_tcgen05"], ["entrypoint", "loom.examples.weave.knn_search_target_highd_k64_neighbors_0624_f505_hiermerge32_v1:launch_for_eval"], ["source_entrypoint", "26d2 high-D K64 tcgen05 producer with 32-group merge in this module"], ["selected_seed", "weave-evolve-knn-search-f505-highd-k64-hiermerge32"], ["producer_seed", "weave-evolve-knn-search-f505-highd-k64-hiermerge32"], ["source_round_doc", "design_doc/active/weave_evolve_knn_search_round_149_f505_highdk64_hiermerge32.md"], ["coverage_class", "bucket_seed_target_d2048_q4_m8192_k64"], ["arch_requirement", "sm_100a"], ["parent_seed", "weave-evolve-knn-search-26d2-highd-k64-neighbors"]]}]}'))
_KERNELS: dict[int, dict[str, Any]] = {}
_SCRATCH: dict[tuple[int, int, int, int, int, int, int, str], tuple[Any, Any]] = {}
_GROUP_SCRATCH: dict[tuple[int, int, int, int, int, int, str], tuple[Any, Any]] = {}
knn_search_target_highd_k64_group_merge_f505_hiermerge32_v1 = _decode_capture(_json_loads('{"__ir__": "knn_search_target_highd_k64_group_merge_f505_hiermerge32_v1", "arg_keys": ["partial_distances", "partial_indices", "group_distances", "group_indices", "B", "Q", "partial_list_count", "num_q_tiles"], "cluster_dims": [1, 1, 1], "computed_smem_bytes": 0, "constants": [["K_MAX_", 64]], "cta_group": 1, "threads": 32}'))
knn_search_target_highd_k64_final_merge_f505_hiermerge32_v1 = _decode_capture(_json_loads('{"__ir__": "knn_search_target_highd_k64_final_merge_f505_hiermerge32_v1", "arg_keys": ["group_distances", "group_indices", "out_distances", "out_indices", "B", "Q", "K"], "cluster_dims": [1, 1, 1], "computed_smem_bytes": 0, "constants": [["K_MAX_", 64]], "cta_group": 1, "threads": 32}'))
partial_ir = _decode_capture(_json_loads('{"__ir__": "knn_search_target_highd_k64_partial_26d2_v1", "arg_keys": ["queries", "database", "partial_distances", "partial_indices", "B", "Q", "M", "split_m", "num_q_tiles"], "cluster_dims": [1, 1, 1], "computed_smem_bytes": 133632, "constants": [["D_ORIG_", 2048], ["NUM_D_PASSES_", 8], ["Q_NORM_PARTS_", 128], ["K_MAX_", 64]], "cta_group": 1, "threads": 256}'))
group_merge_ir = _decode_capture(_json_loads('{"__ir__": "knn_search_target_highd_k64_group_merge_f505_hiermerge32_v1", "arg_keys": ["partial_distances", "partial_indices", "group_distances", "group_indices", "B", "Q", "partial_list_count", "num_q_tiles"], "cluster_dims": [1, 1, 1], "computed_smem_bytes": 0, "constants": [["K_MAX_", 64]], "cta_group": 1, "threads": 32}'))
final_merge_ir = _decode_capture(_json_loads('{"__ir__": "knn_search_target_highd_k64_final_merge_f505_hiermerge32_v1", "arg_keys": ["group_distances", "group_indices", "out_distances", "out_indices", "B", "Q", "K"], "cluster_dims": [1, 1, 1], "computed_smem_bytes": 0, "constants": [["K_MAX_", 64]], "cta_group": 1, "threads": 32}'))
merge_ir = _decode_capture(_json_loads('{"__ir__": "knn_search_target_highd_k64_group_merge_f505_hiermerge32_v1", "arg_keys": ["partial_distances", "partial_indices", "group_distances", "group_indices", "B", "Q", "partial_list_count", "num_q_tiles"], "cluster_dims": [1, 1, 1], "computed_smem_bytes": 0, "constants": [["K_MAX_", 64]], "cta_group": 1, "threads": 32}'))
ir = _decode_capture(_json_loads('{"__ir__": "knn_search_target_highd_k64_group_merge_f505_hiermerge32_v1", "arg_keys": ["partial_distances", "partial_indices", "group_distances", "group_indices", "B", "Q", "partial_list_count", "num_q_tiles"], "cluster_dims": [1, 1, 1], "computed_smem_bytes": 0, "constants": [["K_MAX_", 64]], "cta_group": 1, "threads": 32}'))

def _shape_key(inputs: dict[str, Any]) -> tuple[int, int, int, int, int, bool]:
    return parent_seed._shape_key(inputs)

def _forced_fallback(inputs: dict[str, Any]) -> bool:
    return parent_seed._forced_fallback(inputs)

def _tcgen05_capable_arch() -> bool:
    return parent_seed._tcgen05_capable_arch()

def _active_entry(inputs: dict[str, Any]) -> dict[str, Any] | None:
    if _forced_fallback(inputs) or not _tcgen05_capable_arch():
        return None
    shape = _shape_key(inputs)
    for entry in SHAPE_DISPATCH_REGISTRY:
        if shape == entry['shape']:
            return entry
    return None

def _guard_order() -> list[str]:
    return [*(str(entry['shape_key']) for entry in SHAPE_DISPATCH_REGISTRY), *parent_seed._guard_order()]

def selected_route(inputs: dict[str, Any]) -> str:
    entry = _active_entry(inputs)
    if entry is not None:
        return str(entry['route'])
    return parent_seed.selected_route(inputs)

def selected_route_name(inputs: dict[str, Any]) -> str:
    return selected_route(inputs)

def route_info(inputs: dict[str, Any]) -> dict[str, Any]:
    entry = _active_entry(inputs)
    if entry is not None:
        info = parent_seed._entry_info(inputs, entry)
        info['parent_seed'] = PARENT_SEED
        info['merge_groups'] = HIERMERGE_GROUPS
        return info
    info = dict(parent_seed.route_info(inputs))
    info['guard_order'] = _guard_order()
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
    partial_source = generate_kernel(partial_ir, validate=False, smem_bytes=SMEM_BYTES, D_ORIG_=int(original_d), NUM_D_PASSES_=int(math.ceil(original_d / D_STAGE)), Q_NORM_PARTS_=int(math.ceil(original_d / parent_seed.VEC)), K_MAX_=K64_MAX)
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

def _launch_highd_k64_hiermerge32(inputs: dict[str, Any]) -> dict[str, Any]:
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
        return _launch_highd_k64_hiermerge32(inputs)
    return parent_seed.launch_for_eval(inputs)

def knn_search_compile_and_launch_target_highd_k64_neighbors_0624_f505_hiermerge32(*, benchmark: bool=True, shapes: list[dict[str, Any]] | None=None) -> dict[str, Any]:
    from .._dispatch_runtime import evaluate
    result = evaluate(launch_for_eval, shapes=TARGET_SHAPES if shapes is None else shapes, benchmark=benchmark)
    result['passed'] = bool(result.get('summary', {}).get('all_correct'))
    print(result)
    return result
