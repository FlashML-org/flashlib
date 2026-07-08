"""Round-32 Q4096/K64 certified local-prefix route.

Minimum target architecture: sm_100a. This additive shape kernel keeps the
round-36 local-prefix tcgen05 producer for
``B=1,Q=4096,M=20000,D=128,K=64``, but stores one extra local entry per
producer list. A cert kernel checks whether omitted local entries can beat the
emitted K64 threshold; uncertified rows trigger the existing full-K64 Weave
split79 fallback for exactness.
"""
from __future__ import annotations
from json import loads as _json_loads
from .._dispatch_runtime import _capture_cuTensorMapEncodeTiled, _decode_capture, _import_dispatch_module, _ir_proxy
from typing import Any
from .._dispatch_runtime import dc as dc
from . import knn_search_k64_q4096split79_localprefix_0614_r36_edd7_v1 as prefix_parent
from . import knn_search_k64_q4096split79_twotile_oddevensort_fastmerge_0612_r34_11c1_v1 as exact_parent
K64_MAX = prefix_parent.K64_MAX
LOCAL_PREFIX_K = prefix_parent.LOCAL_PREFIX_K
K_PARTIAL_STRIDE = LOCAL_PREFIX_K + 1
Q4096_ROWS = prefix_parent.Q4096_ROWS
Q4096_M_ROWS = prefix_parent.Q4096_M_ROWS
Q4096_K64_SPLIT_M = prefix_parent.Q4096_K64_SPLIT_M
Q4096_K64_PARTIAL_LISTS = prefix_parent.Q4096_K64_PARTIAL_LISTS
MERGE10_SPLITS_PER_LANE_MAX = prefix_parent.MERGE10_SPLITS_PER_LANE_MAX
THREADS = prefix_parent.THREADS
BLOCK_Q = prefix_parent.BLOCK_Q
BLOCK_M = prefix_parent.BLOCK_M
D_STATIC = prefix_parent.D_STATIC
MERGE_THREADS = prefix_parent.MERGE_THREADS
MMA_POST_MMA_COL_COHORTS = prefix_parent.MMA_POST_MMA_COL_COHORTS
MMA_SMEM_BYTES = prefix_parent.MMA_SMEM_BYTES
MERGE_SMEM_BYTES = prefix_parent.MERGE_SMEM_BYTES
K64_Q4096_PREFIXCERT_SHAPES: list[dict[str, Any]] = [{'label': 'ksweep_q128_m131072_d128_k64', 'params': {'B': 1, 'Q': 128, 'M': 131072, 'D': 128, 'K': 64, 'dtype': 'bfloat16', 'seed': 610312, 'self_search': False, 'min_recall': 0.999}}, {'label': 'ksweep_q4096_m20000_d128_k64', 'params': {'B': 1, 'Q': 4096, 'M': 20000, 'D': 128, 'K': 64, 'dtype': 'bfloat16', 'seed': 610313, 'self_search': False, 'min_recall': 0.999}}]
_KNN_SEARCH_K64_Q4096_PREFIXCERT_KERNELS: dict[str, Any] = {}
_KNN_SEARCH_K64_Q4096_PREFIXCERT_SCRATCH: dict[tuple[int, int, int, int, int, int, int, int, str], tuple[Any, Any]] = {}
_KNN_SEARCH_K64_Q4096_PREFIXCERT_FLAGS: dict[int, Any] = {}
_KNN_SEARCH_K64_Q4096_PREFIXCERT_STATS: dict[str, int | bool | None] = {'attempts': 0, 'certified_count': 0, 'fallback_count': 0, 'last_overflow': None}
knn_search_k64_q4096split79_localprefix_certmerge_0615_r32_edd7_v1 = _decode_capture(_json_loads('{"__ir__": "knn_search_k64_q4096split79_localprefix_certmerge_0615_r32_edd7_v1", "arg_keys": ["partial_distances", "partial_indices", "out_distances", "out_indices", "B", "Q", "K", "partial_list_count", "num_q_tiles"], "cluster_dims": [1, 1, 1], "computed_smem_bytes": 0, "constants": [["K_MAX_", 64], ["K_PREFIX_READ_", 8], ["K_STRIDE_", 9]], "cta_group": 1, "threads": 32}'))
knn_search_k64_q4096split79_localprefix_certflag_init_0615_r32_edd7_v1 = _decode_capture(_json_loads('{"__ir__": "knn_search_k64_q4096split79_localprefix_certflag_init_0615_r32_edd7_v1", "arg_keys": ["overflow_flag"], "cluster_dims": [1, 1, 1], "computed_smem_bytes": 0, "constants": [], "cta_group": 1, "threads": 32}'))
knn_search_k64_q4096split79_localprefix_cert_0615_r32_edd7_v1 = _decode_capture(_json_loads('{"__ir__": "knn_search_k64_q4096split79_localprefix_cert_0615_r32_edd7_v1", "arg_keys": ["partial_distances", "partial_indices", "out_distances", "out_indices", "overflow_flag", "B", "Q", "K", "partial_list_count", "num_q_tiles"], "cluster_dims": [1, 1, 1], "computed_smem_bytes": 0, "constants": [["K_MAX_", 64], ["K_PREFIX_READ_", 8], ["K_STRIDE_", 9]], "cta_group": 1, "threads": 32}'))
partial_ir = _decode_capture(_json_loads('{"__ir__": "knn_search_k64_q4096split79_localprefix9_partial_0615_r32_edd7_v1", "arg_keys": ["queries", "database", "partial_distances", "partial_indices", "B", "Q", "M", "split_m", "num_q_tiles", "total_m_tiles", "tiles_per_split"], "cluster_dims": [1, 1, 1], "computed_smem_bytes": 165120, "constants": [["K_MAX_", 64], ["K_PREFIX_", 9]], "cta_group": 1, "threads": 512}'))
merge_ir = _decode_capture(_json_loads('{"__ir__": "knn_search_k64_q4096split79_localprefix_certmerge_0615_r32_edd7_v1", "arg_keys": ["partial_distances", "partial_indices", "out_distances", "out_indices", "B", "Q", "K", "partial_list_count", "num_q_tiles"], "cluster_dims": [1, 1, 1], "computed_smem_bytes": 0, "constants": [["K_MAX_", 64], ["K_PREFIX_READ_", 8], ["K_STRIDE_", 9]], "cta_group": 1, "threads": 32}'))
certflag_init_ir = _decode_capture(_json_loads('{"__ir__": "knn_search_k64_q4096split79_localprefix_certflag_init_0615_r32_edd7_v1", "arg_keys": ["overflow_flag"], "cluster_dims": [1, 1, 1], "computed_smem_bytes": 0, "constants": [], "cta_group": 1, "threads": 32}'))
cert_ir = _decode_capture(_json_loads('{"__ir__": "knn_search_k64_q4096split79_localprefix_cert_0615_r32_edd7_v1", "arg_keys": ["partial_distances", "partial_indices", "out_distances", "out_indices", "overflow_flag", "B", "Q", "K", "partial_list_count", "num_q_tiles"], "cluster_dims": [1, 1, 1], "computed_smem_bytes": 0, "constants": [["K_MAX_", 64], ["K_PREFIX_READ_", 8], ["K_STRIDE_", 9]], "cta_group": 1, "threads": 32}'))
ir = _decode_capture(_json_loads('{"__ir__": "knn_search_k64_q4096split79_localprefix9_partial_0615_r32_edd7_v1", "arg_keys": ["queries", "database", "partial_distances", "partial_indices", "B", "Q", "M", "split_m", "num_q_tiles", "total_m_tiles", "tiles_per_split"], "cluster_dims": [1, 1, 1], "computed_smem_bytes": 165120, "constants": [["K_MAX_", 64], ["K_PREFIX_", 9]], "cta_group": 1, "threads": 512}'))

def _scratch_prefixcert(inputs: dict[str, Any], partial_list_count: int, num_q_tiles: int) -> tuple[Any, Any]:
    import torch
    key = (int(inputs['B']), int(inputs['Q']), int(inputs['M']), int(partial_list_count), int(num_q_tiles), LOCAL_PREFIX_K, K_PARTIAL_STRIDE, int(inputs['queries'].device.index or 0), str(inputs['queries'].dtype))
    cached = _KNN_SEARCH_K64_Q4096_PREFIXCERT_SCRATCH.get(key)
    if cached is None:
        shape = (int(inputs['B']), int(num_q_tiles), int(partial_list_count), BLOCK_Q, K_PARTIAL_STRIDE)
        cached = (torch.empty(shape, dtype=torch.float32, device=inputs['queries'].device), torch.empty(shape, dtype=torch.int32, device=inputs['queries'].device))
        _KNN_SEARCH_K64_Q4096_PREFIXCERT_SCRATCH[key] = cached
    return cached

def _cert_flag(inputs: dict[str, Any]):
    import torch
    device_index = int(inputs['queries'].device.index or 0)
    flag = _KNN_SEARCH_K64_Q4096_PREFIXCERT_FLAGS.get(device_index)
    if flag is None:
        flag = torch.empty((1,), dtype=torch.int32, device=inputs['queries'].device)
        _KNN_SEARCH_K64_Q4096_PREFIXCERT_FLAGS[device_index] = flag
    return flag

def _compile_k64_q4096_prefixcert_kernels() -> dict[str, Any]:
    return _decode_capture(_json_loads('{"cert": {"__kernel__": "dispatch_kernel_0360"}, "certflag_init": {"__kernel__": "dispatch_kernel_0359"}, "merge": {"__kernel__": "dispatch_kernel_0358"}, "partial": {"__kernel__": "dispatch_kernel_0357"}}'))

def _use_q4096_k64_prefixcert(inputs: dict[str, Any]) -> bool:
    return int(inputs['B']) == 1 and int(inputs['K']) == K64_MAX and (int(inputs['Q']) == Q4096_ROWS) and (int(inputs['M']) == Q4096_M_ROWS) and (int(inputs['D']) == D_STATIC) and prefix_parent.base._tcgen05_capable_arch()

def _launch_q4096_k64_prefixcert(inputs: dict[str, Any]) -> dict[str, Any]:
    """Bypass the retired host-certified route with its exact same-stream fallback."""
    _KNN_SEARCH_K64_Q4096_PREFIXCERT_STATS['attempts'] = int(_KNN_SEARCH_K64_Q4096_PREFIXCERT_STATS['attempts'] or 0) + 1
    _KNN_SEARCH_K64_Q4096_PREFIXCERT_STATS['last_overflow'] = None
    _KNN_SEARCH_K64_Q4096_PREFIXCERT_STATS['fallback_count'] = int(_KNN_SEARCH_K64_Q4096_PREFIXCERT_STATS['fallback_count'] or 0) + 1
    return exact_parent._launch_q4096_k64_split79_oddevensort_fastmerge(inputs)

def launch_for_eval(inputs: dict[str, Any]) -> dict[str, Any]:
    if _use_q4096_k64_prefixcert(inputs):
        return _launch_q4096_k64_prefixcert(inputs)
    return exact_parent.launch_for_eval(inputs)

def _select_contract_shapes(shape_labels: str | tuple[str, ...] | list[str] | None):
    return exact_parent._select_contract_shapes(shape_labels)

def knn_search_compile_and_launch_k64_q4096_prefixcert(*, benchmark: bool=True, shapes: list[dict[str, Any]] | None=None) -> dict[str, Any]:
    from .._dispatch_runtime import evaluate
    result = evaluate(launch_for_eval, shapes=K64_Q4096_PREFIXCERT_SHAPES if shapes is None else shapes, benchmark=benchmark)
    result['passed'] = bool(result.get('summary', {}).get('all_correct'))
    print(result)
    return result
