"""Dynamic low-D and K64 capacity seeds for exact BF16 squared-L2 kNN.

Minimum target architecture: sm_80 for the D3 direct CUDA-core route; sm_100a
for the D130/D512 K64 tcgen05 routes. This additive bucket module targets the
round-116 ``dynamic_low_d_and_k64_capacity`` continuation without editing the
production dispatcher. D130/D512 K64 rows are packed with a Weave kernel to a
zero-padded D512 stride, then scanned by the existing CCEF D512/K64 tcgen05
producer and merge consumer. The D3 DBSCAN row uses one CTA per query with a
direct in-block top-K32 merge.
"""
from __future__ import annotations
from json import loads as _json_loads
from .._dispatch_runtime import _capture_cuTensorMapEncodeTiled, _decode_capture, _import_dispatch_module, _ir_proxy
import math
from typing import Any
from .._dispatch_runtime import select_named_shapes
from . import knn_search_dynamic_d_k64_q64_m65536_tcgen05_0618_ccef_v2 as k64_parent
from . import knn_search_dynamic_d_tiny_q128_tcgen05_0618_c8b9_v1 as pack_parent
from . import knn_search_scalar_capacity_0611_r22_4e96_v1 as scalar_capacity
PACK_THREADS = pack_parent.PACK_THREADS
K64_THREADS = k64_parent.THREADS
K64_BLOCK_Q = k64_parent.BLOCK_Q
K64_BLOCK_M = k64_parent.BLOCK_M
K64_D_PAD = k64_parent.D_PAD
K64_MAX = k64_parent.K64_MAX
K64_MERGE_THREADS = k64_parent.MERGE_THREADS
K64_MMA_POST_MMA_COL_COHORTS = k64_parent.MMA_POST_MMA_COL_COHORTS
K64_MMA_SMEM_BYTES = k64_parent.MMA_SMEM_BYTES
K64_MERGE_SMEM_BYTES = k64_parent.MERGE_SMEM_BYTES
D3_THREADS = 256
D3_NUM_WARPS = D3_THREADS // 32
D3_STATIC = 3
D3_Q = 4096
D3_M = 4096
D3_K = 32
D3_LOCAL_LIST_CAP = (D3_M + D3_THREADS - 1) // D3_THREADS
D3_LOCAL_CANDIDATES = D3_THREADS * D3_LOCAL_LIST_CAP
D3_LOCAL_DIST_BYTES = D3_LOCAL_CANDIDATES * 4
D3_LOCAL_IDX_BYTES = D3_LOCAL_CANDIDATES * 4
D3_WARP_DIST_OFFSET = D3_LOCAL_DIST_BYTES + D3_LOCAL_IDX_BYTES
D3_WARP_IDX_OFFSET = D3_WARP_DIST_OFFSET + D3_NUM_WARPS * 4
D3_WARP_THREAD_OFFSET = D3_WARP_IDX_OFFSET + D3_NUM_WARPS * 4
D3_DIRECT_SMEM_BYTES = D3_WARP_THREAD_OFFSET + D3_NUM_WARPS * 4
ROUTE_D3_DBSCAN_DIRECT = '9d5c_r117_d3_dbscan_q4096_k32_direct'
ROUTE_D130_K64_D512_PACKED = '9d5c_r117_dynamic_d130_q64_k64_d512packed_tcgen05'
ROUTE_D512_K64_D512_PACKED = '9d5c_r117_dynamic_d512_q32_k64_d512packed_tcgen05'
ROUTE_SCALAR_CAPACITY = 'scalar_capacity_parent'
CONSUMED_CCEF_K64_SEED = 'weave-evolve-knn-search-ccef-dynamic-d257-q64-k64-v2'
CONSUMED_SEED = 'weave-evolve-knn-search-9d5c-dynamic-lowd-k64-r117'
TARGET_LABELS: tuple[str, ...] = ('blind_ext_dbscan_self_q4096_m4096_d3_k32', 'blind_ext_dyn_d130_k64_q64_m65536', 'blind_ext_dyn_d512_k64_q32_m32768')
TARGET_SHAPES = _decode_capture(_json_loads('[{"__dict_items__": [["label", "blind_ext_dbscan_self_q4096_m4096_d3_k32"], ["params", {"__dict_items__": [["B", 1], ["Q", 4096], ["M", 4096], ["D", 3], ["K", 32], ["dtype", "bfloat16"], ["seed", 610932], ["self_search", true], ["min_recall", 0.999]]}]]}, {"__dict_items__": [["label", "blind_ext_dyn_d130_k64_q64_m65536"], ["params", {"__dict_items__": [["B", 1], ["Q", 64], ["M", 65536], ["D", 130], ["K", 64], ["dtype", "bfloat16"], ["seed", 610929], ["self_search", false], ["min_recall", 0.999]]}]]}, {"__dict_items__": [["label", "blind_ext_dyn_d512_k64_q32_m32768"], ["params", {"__dict_items__": [["B", 1], ["Q", 32], ["M", 32768], ["D", 512], ["K", 64], ["dtype", "bfloat16"], ["seed", 610930], ["self_search", false], ["min_recall", 0.999]]}]]}]'))
pack_ir = _decode_capture(_json_loads('{"__ir__": "knn_search_dynamic_d_tiny_pack_bf16_0618_c8b9_v1", "arg_keys": ["queries", "database", "padded_queries", "padded_database", "B", "Q", "M"], "cluster_dims": [1, 1, 1], "computed_smem_bytes": 0, "constants": [["D_ORIG_", 3], ["D_PAD_", 16]], "cta_group": 1, "threads": 256}'))
k64_partial_ir = _decode_capture(_json_loads('{"__ir__": "knn_search_dynamic_d257_k64_q64_partial_0618_ccef_v1", "arg_keys": ["queries", "database", "partial_distances", "partial_indices", "B", "Q", "M", "split_m", "num_q_tiles", "total_m_tiles", "tiles_per_split", "partial_list_count"], "cluster_dims": [1, 1, 1], "computed_smem_bytes": 151296, "constants": [["K_MAX_", 64]], "cta_group": 1, "threads": 256}'))
k64_merge_ir = _decode_capture(_json_loads('{"__ir__": "knn_search_dynamic_d257_k64_q64_merge_0618_ccef_v1", "arg_keys": ["partial_distances", "partial_indices", "out_distances", "out_indices", "B", "Q", "K", "partial_list_count", "num_q_tiles"], "cluster_dims": [1, 1, 1], "computed_smem_bytes": 0, "constants": [["K_MAX_", 64]], "cta_group": 1, "threads": 32}'))
scalar_capacity_ir = _decode_capture(_json_loads('{"__ir__": "knn_search_scalar_capacity_partial_v1", "arg_keys": ["queries", "database", "partial_distances", "partial_indices", "B", "Q", "M", "num_m_tiles"], "cluster_dims": [1, 1, 1], "computed_smem_bytes": 0, "constants": [["D_", 8], ["K_CAP_", 64], ["BLOCK_M_", 512], ["NUM_WARPS_", 8]], "cta_group": 1, "threads": 256}'))
_D3_DIRECT_KERNEL: dict[str, Any] = {}
_PACK_KERNELS: dict[int, Any] = {}
_PADDED_INPUTS: dict[tuple[int, int, int, int, int, int, int, int, str], tuple[Any, Any]] = {}
_K64_PARTIAL_SCRATCH: dict[tuple[int, int, int, int, int, int, int, int, str], tuple[Any, Any]] = {}
knn_search_d3_dbscan_q4096_k32_direct_9d5c_r117_v1 = _decode_capture(_json_loads('{"__ir__": "knn_search_d3_dbscan_q4096_k32_direct_9d5c_r117_v1", "arg_keys": ["queries", "database", "out_distances", "out_indices", "B", "Q", "M", "K"], "cluster_dims": [1, 1, 1], "computed_smem_bytes": 32896, "constants": [["D_", 3], ["K_MAX_", 32], ["LOCAL_LIST_CAP_", 16], ["NUM_WARPS_", 8]], "cta_group": 1, "threads": 256}'))
ir = _decode_capture(_json_loads('{"__ir__": "knn_search_d3_dbscan_q4096_k32_direct_9d5c_r117_v1", "arg_keys": ["queries", "database", "out_distances", "out_indices", "B", "Q", "M", "K"], "cluster_dims": [1, 1, 1], "computed_smem_bytes": 32896, "constants": [["D_", 3], ["K_MAX_", 32], ["LOCAL_LIST_CAP_", 16], ["NUM_WARPS_", 8]], "cta_group": 1, "threads": 256}'))

def _tcgen05_capable_arch() -> bool:
    return bool(k64_parent.d256_k64._tcgen05_capable_arch())

def _use_d3_dbscan_direct(inputs: dict[str, Any]) -> bool:
    return int(inputs['B']) == 1 and int(inputs['Q']) == D3_Q and (int(inputs['M']) == D3_M) and (int(inputs['D']) == D3_STATIC) and (int(inputs['K']) == D3_K) and bool(inputs.get('self_search', False)) and (not bool(inputs.get('force_fallback', False)))

def _use_dynamic_d130_k64(inputs: dict[str, Any]) -> bool:
    return int(inputs['B']) == 1 and int(inputs['Q']) == 64 and (int(inputs['M']) == 65536) and (int(inputs['D']) == 130) and (int(inputs['K']) == K64_MAX) and (not bool(inputs.get('self_search', False))) and (not bool(inputs.get('force_fallback', False))) and _tcgen05_capable_arch()

def _use_dynamic_d512_k64(inputs: dict[str, Any]) -> bool:
    return int(inputs['B']) == 1 and int(inputs['Q']) == 32 and (int(inputs['M']) == 32768) and (int(inputs['D']) == K64_D_PAD) and (int(inputs['K']) == K64_MAX) and (not bool(inputs.get('self_search', False))) and (not bool(inputs.get('force_fallback', False))) and _tcgen05_capable_arch()

def _use_padded_k64(inputs: dict[str, Any]) -> bool:
    return _use_dynamic_d130_k64(inputs) or _use_dynamic_d512_k64(inputs)

def _compile_d3_kernel():
    return _decode_capture(_json_loads('{"__kernel__": "dispatch_kernel_0446"}'))

def _ensure_d3_kernel():
    kernel = _D3_DIRECT_KERNEL.get('direct')
    if kernel is None:
        kernel = _compile_d3_kernel()
        _D3_DIRECT_KERNEL['direct'] = kernel
    return kernel

def _compile_pack_kernel(original_d: int):
    kernel = _PACK_KERNELS.get(int(original_d))
    if kernel is None:
        kernel = pack_parent._compile_pack_kernel(int(original_d), K64_D_PAD)
        _PACK_KERNELS[int(original_d)] = kernel
    return kernel

def _padded_buffers(inputs: dict[str, Any]) -> tuple[Any, Any]:
    import torch
    key = (int(inputs['B']), int(inputs['Q']), int(inputs['M']), int(inputs['D']), K64_D_PAD, int(inputs['queries'].device.index or 0), id(inputs['queries']), id(inputs['database']), str(inputs['queries'].dtype))
    cached = _PADDED_INPUTS.get(key)
    if cached is None:
        cached = (torch.empty((int(inputs['B']), int(inputs['Q']), K64_D_PAD), dtype=inputs['queries'].dtype, device=inputs['queries'].device), torch.empty((int(inputs['B']), int(inputs['M']), K64_D_PAD), dtype=inputs['database'].dtype, device=inputs['database'].device))
        _PADDED_INPUTS[key] = cached
    return cached

def _k64_partial_scratch(inputs: dict[str, Any], partial_list_count: int, num_q_tiles: int) -> tuple[Any, Any]:
    import torch
    key = (int(inputs['B']), int(inputs['Q']), int(inputs['M']), int(inputs['D']), int(partial_list_count), int(num_q_tiles), int(inputs['queries'].device.index or 0), id(inputs['queries']), str(inputs['queries'].dtype))
    cached = _K64_PARTIAL_SCRATCH.get(key)
    if cached is None:
        shape = (int(inputs['B']), int(num_q_tiles), int(partial_list_count), K64_BLOCK_Q, K64_MAX)
        cached = (torch.empty(shape, dtype=torch.float32, device=inputs['queries'].device), torch.empty(shape, dtype=torch.int32, device=inputs['queries'].device))
        _K64_PARTIAL_SCRATCH[key] = cached
    return cached

def _launch_d3_dbscan_direct(inputs: dict[str, Any]) -> dict[str, Any]:
    kernel = _ensure_d3_kernel()
    kernel.launch(grid=(int(inputs['B']) * int(inputs['Q']), 1, 1), block=(D3_THREADS, 1, 1), args=[inputs['queries'], inputs['database'], inputs['out_distances'], inputs['out_indices'], int(inputs['B']), int(inputs['Q']), int(inputs['M']), int(inputs['K'])], shared_mem=D3_DIRECT_SMEM_BYTES)
    return {'distances': inputs['out_distances'], 'indices': inputs['out_indices']}

def _launch_padded_k64(inputs: dict[str, Any]) -> dict[str, Any]:
    kernels = k64_parent._ensure_kernels()
    bsz = int(inputs['B'])
    q_rows = int(inputs['Q'])
    m_rows = int(inputs['M'])
    k = int(inputs['K'])
    if int(inputs['D']) == K64_D_PAD:
        padded_queries = inputs['queries']
        padded_database = inputs['database']
    else:
        pack_kernel = _compile_pack_kernel(int(inputs['D']))
        padded_queries, padded_database = _padded_buffers(inputs)
        pack_elems = bsz * (q_rows + m_rows) * K64_D_PAD
        pack_kernel.launch(grid=(math.ceil(pack_elems / PACK_THREADS), 1, 1), block=(PACK_THREADS, 1, 1), args=[inputs['queries'], inputs['database'], padded_queries, padded_database, bsz, q_rows, m_rows], shared_mem=0)
    num_q_tiles = math.ceil(q_rows / K64_BLOCK_Q)
    total_m_tiles = math.ceil(m_rows / K64_BLOCK_M)
    split_m = math.ceil(total_m_tiles / 2)
    tiles_per_split = math.ceil(total_m_tiles / split_m)
    partial_list_count = split_m * K64_MMA_POST_MMA_COL_COHORTS
    partial_dist, partial_idx = _k64_partial_scratch(inputs, partial_list_count, num_q_tiles)
    kernels['partial'].launch(grid=(bsz * num_q_tiles * split_m, 1, 1), block=(K64_THREADS, 1, 1), args=[padded_queries, padded_database, partial_dist, partial_idx, bsz, q_rows, m_rows, split_m, num_q_tiles, total_m_tiles, tiles_per_split, partial_list_count], shared_mem=K64_MMA_SMEM_BYTES)
    kernels['merge'].launch(grid=(bsz * q_rows, 1, 1), block=(K64_MERGE_THREADS, 1, 1), args=[partial_dist, partial_idx, inputs['out_distances'], inputs['out_indices'], bsz, q_rows, k, partial_list_count, num_q_tiles], shared_mem=K64_MERGE_SMEM_BYTES)
    return {'distances': inputs['out_distances'], 'indices': inputs['out_indices']}

def selected_route(inputs: dict[str, Any]) -> str:
    if _use_d3_dbscan_direct(inputs):
        return ROUTE_D3_DBSCAN_DIRECT
    if _use_dynamic_d130_k64(inputs):
        return ROUTE_D130_K64_D512_PACKED
    if _use_dynamic_d512_k64(inputs):
        return ROUTE_D512_K64_D512_PACKED
    return ROUTE_SCALAR_CAPACITY

def selected_route_name(inputs: dict[str, Any]) -> str:
    return selected_route(inputs)

def route_info(inputs: dict[str, Any]) -> dict[str, Any]:
    route = selected_route(inputs)
    if route == ROUTE_D3_DBSCAN_DIRECT:
        return {'route': route, 'selected_route': route, 'selected_entrypoint': 'loom.examples.weave.knn_search_dynamic_lowd_k64_capacity_9d5c_r117_v1:launch_for_eval', 'route_kind': 'specialized', 'route_source': 'bucket-specific-seed', 'coverage_class': 'bucket_seed_lowd_dbscan_d3_k32', 'classification': 'seed-produced', 'coverage_only': False, 'production_policy': 'weave_only', 'external_fallback': None, 'guard_id': '9d5c_r117_d3_dbscan_q4096_m4096_k32', 'selected_seed': CONSUMED_SEED}
    if route in {ROUTE_D130_K64_D512_PACKED, ROUTE_D512_K64_D512_PACKED}:
        pack_route = 'direct_d512_stride' if int(inputs['D']) == K64_D_PAD else 'weave_bf16_zero_pad_to_d512'
        materialized_padding = pack_route != 'direct_d512_stride'
        return {'route': route, 'selected_route': route, 'selected_entrypoint': 'loom.examples.weave.knn_search_dynamic_lowd_k64_capacity_9d5c_r117_v1:launch_for_eval', 'route_kind': 'specialized', 'route_source': 'bucket-specific-seed', 'coverage_class': 'bucket_seed_dynamic_d_k64_capacity', 'classification': 'seed-produced', 'coverage_only': False, 'production_policy': 'weave_only', 'external_fallback': None, 'guard_id': '9d5c_r117_dynamic_d_k64_d512packed', 'pack_route': pack_route, 'scan_route': k64_parent.ROUTE_DYNAMIC_D257_Q64_K64, 'selected_seed': CONSUMED_SEED, 'reused_seed': CONSUMED_CCEF_K64_SEED, 'padding_tag': 'materialized_d512_pack' if materialized_padding else 'none', 'uses_materialized_padding': materialized_padding, 'uses_kernel_padding': False, 'padding_overhead_timed': materialized_padding, 'original_D': int(inputs['D']), 'padded_D': K64_D_PAD, 'workspace_reuse': 'module_cache_by_shape_device_input_identity'}
    return {'route': ROUTE_SCALAR_CAPACITY, 'selected_route': ROUTE_SCALAR_CAPACITY, 'selected_entrypoint': 'loom.examples.weave.knn_search_scalar_capacity_0611_r22_4e96_v1:launch_for_eval', 'route_kind': 'general', 'route_source': 'generic-weave-fallback', 'coverage_only': False, 'production_policy': 'weave_only', 'external_fallback': None, 'forced_fallback': bool(inputs.get('force_fallback', False))}

def route_trace_entry(label: str, inputs: dict[str, Any]) -> dict[str, Any]:
    return {'label': label, 'shape_key': label, 'B': int(inputs['B']), 'Q': int(inputs['Q']), 'M': int(inputs['M']), 'D': int(inputs['D']), 'K': int(inputs['K']), 'self_search': bool(inputs.get('self_search', False)), **route_info(inputs)}
SHAPE_DISPATCH_REGISTRY: tuple[dict[str, Any], ...] = ({'overlay': '9d5c_r117_lowd_dbscan_d3_k32_direct', 'shape_key': 'blind_ext_dbscan_self_q4096_m4096_d3_k32', 'labels': ('blind_ext_dbscan_self_q4096_m4096_d3_k32',), 'guard': 'B == 1 and Q == 4096 and M == 4096 and D == 3 and K == 32 and self_search and not forced_fallback', 'route': ROUTE_D3_DBSCAN_DIRECT, 'entrypoint': 'loom.examples.weave.knn_search_dynamic_lowd_k64_capacity_9d5c_r117_v1:launch_for_eval', 'selected_seed': CONSUMED_SEED, 'source_task': CONSUMED_SEED, 'coverage_class': 'bucket_seed_lowd_dbscan_d3_k32'}, {'overlay': '9d5c_r117_dynamic_d130_q64_k64_d512packed', 'shape_key': 'blind_ext_dyn_d130_k64_q64_m65536', 'labels': ('blind_ext_dyn_d130_k64_q64_m65536',), 'guard': 'B == 1 and Q == 64 and M == 65536 and D == 130 and K == 64 and not self_search and not forced_fallback and tcgen05_capable_arch', 'route': ROUTE_D130_K64_D512_PACKED, 'entrypoint': 'loom.examples.weave.knn_search_dynamic_lowd_k64_capacity_9d5c_r117_v1:launch_for_eval', 'selected_seed': CONSUMED_SEED, 'source_task': CONSUMED_SEED, 'coverage_class': 'bucket_seed_dynamic_d130_k64_d512packed', 'reused_seed': CONSUMED_CCEF_K64_SEED}, {'overlay': '9d5c_r117_dynamic_d512_q32_k64_d512packed', 'shape_key': 'blind_ext_dyn_d512_k64_q32_m32768', 'labels': ('blind_ext_dyn_d512_k64_q32_m32768',), 'guard': 'B == 1 and Q == 32 and M == 32768 and D == 512 and K == 64 and not self_search and not forced_fallback and tcgen05_capable_arch', 'route': ROUTE_D512_K64_D512_PACKED, 'entrypoint': 'loom.examples.weave.knn_search_dynamic_lowd_k64_capacity_9d5c_r117_v1:launch_for_eval', 'selected_seed': CONSUMED_SEED, 'source_task': CONSUMED_SEED, 'coverage_class': 'bucket_seed_dynamic_d512_k64_d512packed', 'reused_seed': CONSUMED_CCEF_K64_SEED})

def launch_for_eval(inputs: dict[str, Any]) -> dict[str, Any]:
    if _use_d3_dbscan_direct(inputs):
        return _launch_d3_dbscan_direct(inputs)
    if _use_padded_k64(inputs):
        return _launch_padded_k64(inputs)
    return scalar_capacity.launch_for_eval(inputs)

def knn_search_compile_and_launch_dynamic_lowd_k64_capacity(*, benchmark: bool=True, shapes: list[dict[str, Any]] | None=None) -> dict[str, Any]:
    from .._dispatch_runtime import evaluate
    selected = TARGET_SHAPES if shapes is None else shapes
    result = evaluate(launch_for_eval, shapes=selected, benchmark=benchmark)
    result['passed'] = bool(result.get('summary', {}).get('all_correct'))
    print(result)
    return result
