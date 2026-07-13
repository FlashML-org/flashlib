"""Round-36/e4cb Q4096/K64 row-flag fused prefix6-cert route.

Minimum target architecture: sm_100a. This additive shape kernel keeps the
round-20/245d prefix6-certified tcgen05 producer for
``B=1,Q=4096,M=20000,D=128,K=64``, folds the omitted-sentinel certification
check into the merge consumer, and writes one overflow flag per query row. The
row-local flags remove the separate global flag initialization launch while
preserving exact fallback to the full-K64 Weave parent when a skipped local
entry can beat the emitted top-64 threshold.
"""
from __future__ import annotations
from json import loads as _json_loads
from .._dispatch_runtime import _capture_cuTensorMapEncodeTiled, _decode_capture, _import_dispatch_module, _ir_proxy
from typing import Any
from . import knn_search_k64_q4096split79_localprefix6_certfallback_0615_245d_v1 as cert_parent
from . import knn_search_k64_q4096split79_twotile_oddevensort_fastmerge_0612_r34_11c1_v1 as exact_parent
K64_MAX = cert_parent.K64_MAX
LOCAL_PREFIX_K = cert_parent.LOCAL_PREFIX_K
K_PARTIAL_STRIDE = cert_parent.K_PARTIAL_STRIDE
Q4096_ROWS = cert_parent.Q4096_ROWS
Q4096_M_ROWS = cert_parent.Q4096_M_ROWS
Q4096_K64_SPLIT_M = cert_parent.Q4096_K64_SPLIT_M
Q4096_K64_PARTIAL_LISTS = cert_parent.Q4096_K64_PARTIAL_LISTS
MERGE10_SPLITS_PER_LANE_MAX = cert_parent.MERGE10_SPLITS_PER_LANE_MAX
THREADS = cert_parent.THREADS
BLOCK_Q = cert_parent.BLOCK_Q
BLOCK_M = cert_parent.BLOCK_M
D_STATIC = cert_parent.D_STATIC
MERGE_THREADS = cert_parent.MERGE_THREADS
MMA_POST_MMA_COL_COHORTS = cert_parent.MMA_POST_MMA_COL_COHORTS
MMA_SMEM_BYTES = cert_parent.MMA_SMEM_BYTES
MERGE_SMEM_BYTES = cert_parent.MERGE_SMEM_BYTES
ROUTE_Q4096_M20000_K64_PREFIX6_ROWFLAG_FUSEDCERT = 'round36_e4cb_q4096_m20000_k64_prefix6_rowflag_fusedcert'
K64_Q4096_ROWFLAG_FUSEDCERT_E4CB_SHAPES: list[dict[str, Any]] = [{'label': 'ksweep_q4096_m20000_d128_k64', 'params': {'B': 1, 'Q': 4096, 'M': 20000, 'D': 128, 'K': 64, 'dtype': 'bfloat16', 'seed': 610313, 'self_search': False, 'min_recall': 0.999}}]
_KNN_SEARCH_K64_Q4096_ROWFLAG_FUSEDCERT_E4CB_KERNELS: dict[str, Any] = {}
_KNN_SEARCH_K64_Q4096_ROWFLAG_FUSEDCERT_E4CB_SCRATCH: dict[tuple[int, int, int, int, int, int, int, int, str], tuple[Any, Any]] = {}
_KNN_SEARCH_K64_Q4096_ROWFLAG_FUSEDCERT_E4CB_FLAGS: dict[tuple[int, int], Any] = {}
_KNN_SEARCH_K64_Q4096_ROWFLAG_FUSEDCERT_E4CB_STATS: dict[str, int | bool | None] = {'attempts': 0, 'certified_count': 0, 'fallback_count': 0, 'last_overflow': None}
knn_search_k64_q4096split79_localprefix6_rowflag_fusedcert_merge_0615_r36_e4cb_v1 = _decode_capture(_json_loads('{"__ir__": "knn_search_k64_q4096split79_localprefix6_rowflag_fusedcert_merge_0615_r36_e4cb_v1", "arg_keys": ["partial_distances", "partial_indices", "out_distances", "out_indices", "overflow_rows", "B", "Q", "K", "partial_list_count", "num_q_tiles"], "cluster_dims": [1, 1, 1], "computed_smem_bytes": 0, "constants": [["K_MAX_", 64], ["K_PREFIX_READ_", 6], ["K_STRIDE_", 7]], "cta_group": 1, "threads": 32}'))
partial_ir = _decode_capture(_json_loads('{"__ir__": "knn_search_k64_q4096split79_localprefix7_partial_0615_245d_v1", "arg_keys": ["queries", "database", "partial_distances", "partial_indices", "B", "Q", "M", "split_m", "num_q_tiles", "total_m_tiles", "tiles_per_split"], "cluster_dims": [1, 1, 1], "computed_smem_bytes": 165120, "constants": [["K_MAX_", 64], ["K_PREFIX_", 7]], "cta_group": 1, "threads": 512}'))
merge_ir = _decode_capture(_json_loads('{"__ir__": "knn_search_k64_q4096split79_localprefix6_rowflag_fusedcert_merge_0615_r36_e4cb_v1", "arg_keys": ["partial_distances", "partial_indices", "out_distances", "out_indices", "overflow_rows", "B", "Q", "K", "partial_list_count", "num_q_tiles"], "cluster_dims": [1, 1, 1], "computed_smem_bytes": 0, "constants": [["K_MAX_", 64], ["K_PREFIX_READ_", 6], ["K_STRIDE_", 7]], "cta_group": 1, "threads": 32}'))
ir = _decode_capture(_json_loads('{"__ir__": "knn_search_k64_q4096split79_localprefix7_partial_0615_245d_v1", "arg_keys": ["queries", "database", "partial_distances", "partial_indices", "B", "Q", "M", "split_m", "num_q_tiles", "total_m_tiles", "tiles_per_split"], "cluster_dims": [1, 1, 1], "computed_smem_bytes": 165120, "constants": [["K_MAX_", 64], ["K_PREFIX_", 7]], "cta_group": 1, "threads": 512}'))

def _scratch_rowflag_fusedcert(inputs: dict[str, Any], partial_list_count: int, num_q_tiles: int) -> tuple[Any, Any]:
    import torch
    key = (int(inputs['B']), int(inputs['Q']), int(inputs['M']), int(partial_list_count), int(num_q_tiles), LOCAL_PREFIX_K, K_PARTIAL_STRIDE, int(inputs['queries'].device.index or 0), str(inputs['queries'].dtype))
    cached = _KNN_SEARCH_K64_Q4096_ROWFLAG_FUSEDCERT_E4CB_SCRATCH.get(key)
    if cached is None:
        shape = (int(inputs['B']), int(num_q_tiles), int(partial_list_count), BLOCK_Q, K_PARTIAL_STRIDE)
        cached = (torch.empty(shape, dtype=torch.float32, device=inputs['queries'].device), torch.empty(shape, dtype=torch.int32, device=inputs['queries'].device))
        _KNN_SEARCH_K64_Q4096_ROWFLAG_FUSEDCERT_E4CB_SCRATCH[key] = cached
    return cached

def _row_flags(inputs: dict[str, Any], row_count: int):
    import torch
    device_index = int(inputs['queries'].device.index or 0)
    key = (device_index, int(row_count))
    flags = _KNN_SEARCH_K64_Q4096_ROWFLAG_FUSEDCERT_E4CB_FLAGS.get(key)
    if flags is None:
        flags = torch.empty((int(row_count),), dtype=torch.int32, device=inputs['queries'].device)
        _KNN_SEARCH_K64_Q4096_ROWFLAG_FUSEDCERT_E4CB_FLAGS[key] = flags
    return flags

def _compile_k64_q4096_rowflag_fusedcert_kernels() -> dict[str, Any]:
    return _decode_capture(_json_loads('{"merge": {"__kernel__": "dispatch_kernel_0218"}, "partial": {"__kernel__": "dispatch_kernel_0217"}}'))

def _use_q4096_k64_rowflag_fusedcert(inputs: dict[str, Any]) -> bool:
    return cert_parent._use_q4096_k64_prefix6cert(inputs)

def _launch_q4096_k64_rowflag_fusedcert(inputs: dict[str, Any]) -> dict[str, Any]:
    """Bypass the retired host-certified route with its exact same-stream fallback."""
    _KNN_SEARCH_K64_Q4096_ROWFLAG_FUSEDCERT_E4CB_STATS['attempts'] = int(_KNN_SEARCH_K64_Q4096_ROWFLAG_FUSEDCERT_E4CB_STATS['attempts'] or 0) + 1
    _KNN_SEARCH_K64_Q4096_ROWFLAG_FUSEDCERT_E4CB_STATS['last_overflow'] = None
    _KNN_SEARCH_K64_Q4096_ROWFLAG_FUSEDCERT_E4CB_STATS['fallback_count'] = int(_KNN_SEARCH_K64_Q4096_ROWFLAG_FUSEDCERT_E4CB_STATS['fallback_count'] or 0) + 1
    return exact_parent._launch_q4096_k64_split79_oddevensort_fastmerge(inputs)

def launch_for_eval(inputs: dict[str, Any]) -> dict[str, Any]:
    if _use_q4096_k64_rowflag_fusedcert(inputs):
        return _launch_q4096_k64_rowflag_fusedcert(inputs)
    return exact_parent.launch_for_eval(inputs)

def _select_contract_shapes(shape_labels: str | tuple[str, ...] | list[str] | None):
    return exact_parent._select_contract_shapes(shape_labels)

def selected_route_name(inputs: dict[str, Any]) -> str:
    if _use_q4096_k64_rowflag_fusedcert(inputs):
        return ROUTE_Q4096_M20000_K64_PREFIX6_ROWFLAG_FUSEDCERT
    return cert_parent.selected_route_name(inputs)

def stats() -> dict[str, int | bool | None]:
    return dict(_KNN_SEARCH_K64_Q4096_ROWFLAG_FUSEDCERT_E4CB_STATS)

def route_info(inputs: dict[str, Any]) -> dict[str, Any]:
    if _use_q4096_k64_rowflag_fusedcert(inputs):
        return {'route': ROUTE_Q4096_M20000_K64_PREFIX6_ROWFLAG_FUSEDCERT, 'selected_route': ROUTE_Q4096_M20000_K64_PREFIX6_ROWFLAG_FUSEDCERT, 'selected_entrypoint': 'loom.examples.weave.knn_search_k64_q4096split79_localprefix6_rowflag_fusedcert_0615_r36_e4cb_v1:launch_for_eval', 'route_kind': 'specialized', 'coverage_only': False, 'production_policy': 'weave_only', 'external_fallback': None, 'selected_guard': 'B == 1 and Q == 4096 and M == 20000 and D == 128 and K == 64', 'route_status': 'retired_bypassed_to_exact', 'certification_policy': 'legacy_prefix_cert_retired_direct_same_stream_exact_fallback', 'fallback_entrypoint': 'loom.examples.weave.knn_search_k64_q4096split79_twotile_oddevensort_fastmerge_0612_r34_11c1_v1:launch_for_eval'}
    if hasattr(cert_parent, 'route_info'):
        return cert_parent.route_info(inputs)
    return {'route': 'parent'}

def route_trace_entry(label: str, inputs: dict[str, Any]) -> dict[str, Any]:
    info = route_info(inputs)
    return {'label': label, 'B': int(inputs['B']), 'Q': int(inputs['Q']), 'M': int(inputs['M']), 'D': int(inputs['D']), 'K': int(inputs['K']), **info}

def knn_search_compile_and_launch_k64_q4096_rowflag_fusedcert(*, benchmark: bool=True, shapes: list[dict[str, Any]] | None=None) -> dict[str, Any]:
    from .._dispatch_runtime import evaluate
    result = evaluate(launch_for_eval, shapes=K64_Q4096_ROWFLAG_FUSEDCERT_E4CB_SHAPES if shapes is None else shapes, benchmark=benchmark)
    result['passed'] = bool(result.get('summary', {}).get('all_correct'))
    print(result)
    return result
