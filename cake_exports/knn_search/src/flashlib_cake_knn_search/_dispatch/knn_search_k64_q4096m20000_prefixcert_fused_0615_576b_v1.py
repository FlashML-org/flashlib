"""Round-20/576b Q4096/M20000/K64 fused prefix6-cert route.

Minimum target architecture: sm_100a. This additive shape kernel keeps the
round-20/245d prefix6 certified tcgen05 producer, but folds the omitted-sentinel
certification pass into the prefix merge kernel.  The merge writes one device
flag per query row and a following exact CUDA-core repair kernel recomputes only
unsafe rows.  The entire decision stays on the GPU: there is no flag readback,
device synchronization, or host-conditional launch in the production path.
"""
from __future__ import annotations
from json import loads as _json_loads
from .._dispatch_runtime import _capture_cuTensorMapEncodeTiled, _decode_capture, _import_dispatch_module, _ir_proxy
import math
from typing import Any
from . import knn_search_k64_q4096split79_localprefix6_certfallback_0615_245d_v1 as prefix6_parent
from . import knn_search_scalar_capacity_0611_r22_4e96_v1 as scalar_parent
from .knn_search_stream import current_stream_handle
K64_MAX = prefix6_parent.K64_MAX
LOCAL_PREFIX_K = prefix6_parent.LOCAL_PREFIX_K
K_PARTIAL_STRIDE = prefix6_parent.K_PARTIAL_STRIDE
Q4096_ROWS = prefix6_parent.Q4096_ROWS
Q4096_M_ROWS = prefix6_parent.Q4096_M_ROWS
Q4096_K64_SPLIT_M = prefix6_parent.Q4096_K64_SPLIT_M
Q4096_K64_PARTIAL_LISTS = prefix6_parent.Q4096_K64_PARTIAL_LISTS
MERGE10_SPLITS_PER_LANE_MAX = prefix6_parent.MERGE10_SPLITS_PER_LANE_MAX
THREADS = prefix6_parent.THREADS
BLOCK_Q = prefix6_parent.BLOCK_Q
BLOCK_M = prefix6_parent.BLOCK_M
D_STATIC = prefix6_parent.D_STATIC
MERGE_THREADS = prefix6_parent.MERGE_THREADS
MMA_POST_MMA_COL_COHORTS = prefix6_parent.MMA_POST_MMA_COL_COHORTS
MMA_SMEM_BYTES = prefix6_parent.MMA_SMEM_BYTES
MERGE_SMEM_BYTES = prefix6_parent.MERGE_SMEM_BYTES
ROUTE_Q4096_M20000_K64_PREFIX6CERT_FUSED = 'round20_576b_q4096_m20000_k64_prefix6cert_fused'
K64_Q4096_PREFIXCERT_FUSED_SHAPES: list[dict[str, Any]] = [{'label': 'ksweep_q4096_m20000_d128_k64', 'params': {'B': 1, 'Q': 4096, 'M': 20000, 'D': 128, 'K': 64, 'dtype': 'bfloat16', 'seed': 610313, 'self_search': False, 'min_recall': 0.999}}]
_KNN_SEARCH_K64_Q4096_PREFIXCERT_FUSED_KERNELS: dict[str, Any] = {}
_KNN_SEARCH_K64_Q4096_PREFIXCERT_FUSED_SCRATCH: dict[tuple[int, int, int, int, int, int, int, int, str, int], tuple[Any, Any]] = {}
_KNN_SEARCH_K64_Q4096_PREFIXCERT_FUSED_FLAGS: dict[tuple[int, int, int], Any] = {}
_KNN_SEARCH_K64_Q4096_PREFIXCERT_FUSED_STATS: dict[str, int | bool | None] = {'attempts': 0, 'certified_count': 0, 'fallback_count': 0, 'last_overflow': None}
knn_search_k64_q4096m20000_prefixcert_fused_merge_0615_576b_v1 = _decode_capture(_json_loads('{"__ir__": "knn_search_k64_q4096m20000_prefixcert_fused_merge_0615_576b_v1", "arg_keys": ["partial_distances", "partial_indices", "out_distances", "out_indices", "overflow_rows", "B", "Q", "K", "partial_list_count", "num_q_tiles"], "cluster_dims": [1, 1, 1], "computed_smem_bytes": 0, "constants": [["K_MAX_", 64], ["K_PREFIX_READ_", 6], ["K_STRIDE_", 7]], "cta_group": 1, "threads": 32}'))
knn_search_k64_q4096m20000_prefixcert_gpu_repair_0705_v1 = _decode_capture(_json_loads('{"__ir__": "knn_search_k64_q4096m20000_prefixcert_gpu_repair_0705_v1", "arg_keys": ["queries", "database", "out_distances", "out_indices", "overflow_rows", "B", "Q", "M"], "cluster_dims": [1, 1, 1], "computed_smem_bytes": 4096, "constants": [["D_", 128], ["K_CAP_", 64], ["NUM_WARPS_", 8]], "cta_group": 1, "threads": 256}'))
partial_ir = _decode_capture(_json_loads('{"__ir__": "knn_search_k64_q4096split79_localprefix7_partial_0615_245d_v1", "arg_keys": ["queries", "database", "partial_distances", "partial_indices", "B", "Q", "M", "split_m", "num_q_tiles", "total_m_tiles", "tiles_per_split"], "cluster_dims": [1, 1, 1], "computed_smem_bytes": 165120, "constants": [["K_MAX_", 64], ["K_PREFIX_", 7]], "cta_group": 1, "threads": 512}'))
merge_ir = _decode_capture(_json_loads('{"__ir__": "knn_search_k64_q4096m20000_prefixcert_fused_merge_0615_576b_v1", "arg_keys": ["partial_distances", "partial_indices", "out_distances", "out_indices", "overflow_rows", "B", "Q", "K", "partial_list_count", "num_q_tiles"], "cluster_dims": [1, 1, 1], "computed_smem_bytes": 0, "constants": [["K_MAX_", 64], ["K_PREFIX_READ_", 6], ["K_STRIDE_", 7]], "cta_group": 1, "threads": 32}'))
repair_ir = _decode_capture(_json_loads('{"__ir__": "knn_search_k64_q4096m20000_prefixcert_gpu_repair_0705_v1", "arg_keys": ["queries", "database", "out_distances", "out_indices", "overflow_rows", "B", "Q", "M"], "cluster_dims": [1, 1, 1], "computed_smem_bytes": 4096, "constants": [["D_", 128], ["K_CAP_", 64], ["NUM_WARPS_", 8]], "cta_group": 1, "threads": 256}'))
certflag_init_ir = _decode_capture(_json_loads('{"__ir__": "knn_search_k64_q4096split79_localprefix6_certflag_init_0615_245d_v1", "arg_keys": ["overflow_flag"], "cluster_dims": [1, 1, 1], "computed_smem_bytes": 0, "constants": [], "cta_group": 1, "threads": 32}'))
ir = _decode_capture(_json_loads('{"__ir__": "knn_search_k64_q4096split79_localprefix7_partial_0615_245d_v1", "arg_keys": ["queries", "database", "partial_distances", "partial_indices", "B", "Q", "M", "split_m", "num_q_tiles", "total_m_tiles", "tiles_per_split"], "cluster_dims": [1, 1, 1], "computed_smem_bytes": 165120, "constants": [["K_MAX_", 64], ["K_PREFIX_", 7]], "cta_group": 1, "threads": 512}'))

def _scratch_prefixcert_fused(inputs: dict[str, Any], partial_list_count: int, num_q_tiles: int) -> tuple[Any, Any]:
    import torch
    key = (int(inputs['B']), int(inputs['Q']), int(inputs['M']), int(partial_list_count), int(num_q_tiles), LOCAL_PREFIX_K, K_PARTIAL_STRIDE, int(inputs['queries'].device.index or 0), str(inputs['queries'].dtype), current_stream_handle(inputs))
    cached = _KNN_SEARCH_K64_Q4096_PREFIXCERT_FUSED_SCRATCH.get(key)
    if cached is None:
        shape = (int(inputs['B']), int(num_q_tiles), int(partial_list_count), BLOCK_Q, K_PARTIAL_STRIDE)
        cached = (torch.empty(shape, dtype=torch.float32, device=inputs['queries'].device), torch.empty(shape, dtype=torch.int32, device=inputs['queries'].device))
        _KNN_SEARCH_K64_Q4096_PREFIXCERT_FUSED_SCRATCH[key] = cached
    return cached

def _cert_rows_fused(inputs: dict[str, Any], row_count: int):
    import torch
    device_index = int(inputs['queries'].device.index or 0)
    key = (device_index, current_stream_handle(inputs), int(row_count))
    rows = _KNN_SEARCH_K64_Q4096_PREFIXCERT_FUSED_FLAGS.get(key)
    if rows is None:
        rows = torch.empty((row_count,), dtype=torch.int32, device=inputs['queries'].device)
        _KNN_SEARCH_K64_Q4096_PREFIXCERT_FUSED_FLAGS[key] = rows
    return rows

def _compile_k64_q4096_prefixcert_fused_kernels() -> dict[str, Any]:
    return _decode_capture(_json_loads('{"merge": {"__kernel__": "dispatch_kernel_0590"}, "partial": {"__kernel__": "dispatch_kernel_0589"}, "repair": {"__kernel__": "dispatch_kernel_0591"}}'))

def _use_q4096_k64_prefixcert_fused(inputs: dict[str, Any]) -> bool:
    return prefix6_parent._use_q4096_k64_prefix6cert(inputs)

def _launch_q4096_k64_prefixcert_fused(inputs: dict[str, Any]) -> dict[str, Any]:
    if not _KNN_SEARCH_K64_Q4096_PREFIXCERT_FUSED_KERNELS:
        _KNN_SEARCH_K64_Q4096_PREFIXCERT_FUSED_KERNELS.update(_compile_k64_q4096_prefixcert_fused_kernels())
    bsz = int(inputs['B'])
    q_rows = int(inputs['Q'])
    m_rows = int(inputs['M'])
    k = int(inputs['K'])
    num_q_tiles = math.ceil(q_rows / BLOCK_Q)
    total_m_tiles = math.ceil(m_rows / BLOCK_M)
    split_m = min(Q4096_K64_SPLIT_M, total_m_tiles)
    tiles_per_split = math.ceil(total_m_tiles / split_m)
    partial_list_count = split_m * MMA_POST_MMA_COL_COHORTS
    partial_dist, partial_idx = _scratch_prefixcert_fused(inputs, partial_list_count, num_q_tiles)
    overflow_rows = _cert_rows_fused(inputs, bsz * q_rows)
    _KNN_SEARCH_K64_Q4096_PREFIXCERT_FUSED_KERNELS['partial'].launch(grid=(bsz * num_q_tiles * split_m, 1, 1), block=(THREADS, 1, 1), args=[inputs['queries'], inputs['database'], partial_dist, partial_idx, bsz, q_rows, m_rows, split_m, num_q_tiles, total_m_tiles, tiles_per_split], shared_mem=MMA_SMEM_BYTES)
    _KNN_SEARCH_K64_Q4096_PREFIXCERT_FUSED_KERNELS['merge'].launch(grid=(bsz * q_rows, 1, 1), block=(MERGE_THREADS, 1, 1), args=[partial_dist, partial_idx, inputs['out_distances'], inputs['out_indices'], overflow_rows, bsz, q_rows, k, partial_list_count, num_q_tiles], shared_mem=MERGE_SMEM_BYTES)
    _KNN_SEARCH_K64_Q4096_PREFIXCERT_FUSED_KERNELS['repair'].launch(grid=(bsz * q_rows, 1, 1), block=(scalar_parent.THREADS, 1, 1), args=[inputs['queries'], inputs['database'], inputs['out_distances'], inputs['out_indices'], overflow_rows, bsz, q_rows, m_rows], shared_mem=scalar_parent.DIRECT_SMEM_BYTES)
    _KNN_SEARCH_K64_Q4096_PREFIXCERT_FUSED_STATS['attempts'] = int(_KNN_SEARCH_K64_Q4096_PREFIXCERT_FUSED_STATS['attempts'] or 0) + 1
    _KNN_SEARCH_K64_Q4096_PREFIXCERT_FUSED_STATS['last_overflow'] = None
    return {'distances': inputs['out_distances'], 'indices': inputs['out_indices']}

def launch_for_eval(inputs: dict[str, Any]) -> dict[str, Any]:
    if _use_q4096_k64_prefixcert_fused(inputs):
        return _launch_q4096_k64_prefixcert_fused(inputs)
    return prefix6_parent.parent.exact_parent.launch_for_eval(inputs)

def _select_contract_shapes(shape_labels: str | tuple[str, ...] | list[str] | None):
    return prefix6_parent.parent._select_contract_shapes(shape_labels)

def selected_route_name(inputs: dict[str, Any]) -> str:
    if _use_q4096_k64_prefixcert_fused(inputs):
        return ROUTE_Q4096_M20000_K64_PREFIX6CERT_FUSED
    return prefix6_parent.selected_route_name(inputs)

def route_info(inputs: dict[str, Any]) -> dict[str, Any]:
    route = selected_route_name(inputs)
    if route == ROUTE_Q4096_M20000_K64_PREFIX6CERT_FUSED:
        return {'route': route, 'selected_route': route, 'selected_entrypoint': 'loom.examples.weave.knn_search_k64_q4096m20000_prefixcert_fused_0615_576b_v1:launch_for_eval', 'route_kind': 'specialized', 'coverage_only': False, 'production_policy': 'weave_only', 'external_fallback': None, 'selected_guard': 'B == 1 and Q == 4096 and M == 20000 and D == 128 and K == 64', 'certification_policy': 'prefix6_plus_sentinel_fused_merge_with_gpu_row_exact_repair', 'fallback_entrypoint': 'loom.examples.weave.knn_search_k64_q4096m20000_prefixcert_fused_0615_576b_v1:knn_search_k64_q4096m20000_prefixcert_gpu_repair_0705_v1'}
    return prefix6_parent.route_info(inputs)

def reset_stats() -> None:
    _KNN_SEARCH_K64_Q4096_PREFIXCERT_FUSED_STATS['attempts'] = 0
    _KNN_SEARCH_K64_Q4096_PREFIXCERT_FUSED_STATS['certified_count'] = 0
    _KNN_SEARCH_K64_Q4096_PREFIXCERT_FUSED_STATS['fallback_count'] = 0
    _KNN_SEARCH_K64_Q4096_PREFIXCERT_FUSED_STATS['last_overflow'] = None

def knn_search_compile_and_launch_k64_q4096_prefixcert_fused(*, benchmark: bool=True, shapes: list[dict[str, Any]] | None=None) -> dict[str, Any]:
    from .._dispatch_runtime import evaluate
    result = evaluate(launch_for_eval, shapes=K64_Q4096_PREFIXCERT_FUSED_SHAPES if shapes is None else shapes, benchmark=benchmark)
    result['passed'] = bool(result.get('summary', {}).get('all_correct'))
    print(result)
    return result
