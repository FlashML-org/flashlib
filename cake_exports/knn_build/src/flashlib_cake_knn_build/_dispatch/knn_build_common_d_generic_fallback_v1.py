"""Coverage-only generic kNN build/search fallback for v11 common-D misses.

Minimum target architecture: sm_80. This fallback is intentionally simple:
one CTA computes one query row, each thread scans a strided subset of database
rows, and thread 0 merges the per-thread top-k lists. It exists only to keep
the v11 common-D dispatcher Weave-only and correct for uncovered high-D rows;
hot shapes should still return to shape-specific evolution for performance.
"""
from __future__ import annotations
from json import loads as _json_loads
from .._dispatch_runtime import _capture_cuTensorMapEncodeTiled, _decode_capture, _import_dispatch_module, _ir_proxy
from functools import lru_cache
from typing import Any
THREADS = 256
K_MAX = 10
SMEM_DIST_BYTES = THREADS * K_MAX * 4
SMEM_IDX_BYTES = THREADS * K_MAX * 4
SMEM_BYTES = SMEM_DIST_BYTES + SMEM_IDX_BYTES
MODULE = 'loom.examples.weave.knn_build_common_d_generic_fallback_v1'
ROUTE_ENTRYPOINT = ''.join([format(MODULE, ''), ':launch_from_contract_inputs'])
ROUTE_ID = 'knn_build_common_d_generic_fallback_v1:direct_scalar_topk'
SEED_ID = 'coverage_only_common_d_generic_fallback_v1'
knn_build_common_d_generic_direct_v1 = _decode_capture(_json_loads('{"__ir__": "knn_build_common_d_generic_direct_v1", "arg_keys": ["query", "database", "out_dists", "out_indices", "B", "Q", "M", "K", "D"], "cluster_dims": [1, 1, 1], "computed_smem_bytes": 20480, "constants": [["K_MAX_", 10], ["THREADS_", 256]], "cta_group": 1, "threads": 256}'))
ir = _decode_capture(_json_loads('{"__ir__": "knn_build_common_d_generic_direct_v1", "arg_keys": ["query", "database", "out_dists", "out_indices", "B", "Q", "M", "K", "D"], "cluster_dims": [1, 1, 1], "computed_smem_bytes": 20480, "constants": [["K_MAX_", 10], ["THREADS_", 256]], "cta_group": 1, "threads": 256}'))

def _compile_kernel():
    return _decode_capture(_json_loads('{"__kernel__": "dispatch_kernel_0189"}'))

def _compiled_kernel():
    return _decode_capture(_json_loads('{"__kernel__": "dispatch_kernel_0189"}'))

def _dtype_name(inputs: dict[str, Any]) -> str:
    query = inputs.get('query')
    if query is not None:
        return str(query.dtype).replace('torch.', '')
    return str(inputs.get('dtype', '')).replace('torch.', '')

def _eligible_common_d_fallback(inputs: dict[str, Any]) -> bool:
    return _dtype_name(inputs) == 'bfloat16' and int(inputs.get('K', -1)) <= K_MAX

def route_for_contract_inputs(inputs: dict[str, Any]) -> str:
    if not _eligible_common_d_fallback(inputs):
        raise ValueError('common-D generic fallback supports bfloat16 inputs with K <= 10')
    return ROUTE_ID

def launch_from_contract_inputs(inputs: dict[str, Any]) -> None:
    if not _eligible_common_d_fallback(inputs):
        raise ValueError('common-D generic fallback supports bfloat16 inputs with K <= 10')
    _compiled_kernel().launch(grid=(int(inputs['B']) * int(inputs['Q']), 1, 1), block=(THREADS, 1, 1), args=[inputs['query'], inputs['database'], inputs['out_dists'], inputs['out_indices'], int(inputs['B']), int(inputs['Q']), int(inputs['M']), int(inputs['K']), int(inputs['D'])], shared_mem=SMEM_BYTES)

def candidate(inputs: dict[str, Any]) -> None:
    launch_from_contract_inputs(inputs)
