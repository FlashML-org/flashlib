"""Round-22 low-D DBSCAN capacity route for exact BF16 squared-L2 kNN.

Minimum target architecture: sm_80. This additive clean-room candidate adds a
D=2, K<=64 direct-search route for small DBSCAN/self-search shapes and delegates
all other shapes to the round-21 composed dispatcher.
"""
from __future__ import annotations
from json import loads as _json_loads
from .._dispatch_runtime import _capture_cuTensorMapEncodeTiled, _decode_capture, _import_dispatch_module, _ir_proxy
from typing import Any
from . import knn_search_dispatch_compose_0611_r21_4e96_v1 as parent
THREADS = 128
D_STATIC = 2
K_MAX = 64
M_MAX = 2048
LOCAL_CAP = 16
PARTIAL_DIST_BYTES = THREADS * LOCAL_CAP * 4
PARTIAL_IDX_BYTES = THREADS * LOCAL_CAP * 4
DIRECT_SMEM_BYTES = PARTIAL_DIST_BYTES + PARTIAL_IDX_BYTES
_KNN_LOWD_KERNELS: dict[str, Any] = {}
knn_search_lowd_dbscan_d2_direct_0611_r22_6e85_v1 = _decode_capture(_json_loads('{"__ir__": "knn_search_lowd_dbscan_d2_direct_0611_r22_6e85_v1", "arg_keys": ["queries", "database", "out_distances", "out_indices", "B", "Q", "M", "K"], "cluster_dims": [1, 1, 1], "computed_smem_bytes": 16384, "constants": [["D_", 2], ["K_MAX_", 64], ["LOCAL_CAP_", 16]], "cta_group": 1, "threads": 128}'))
ir = _decode_capture(_json_loads('{"__ir__": "knn_search_lowd_dbscan_d2_direct_0611_r22_6e85_v1", "arg_keys": ["queries", "database", "out_distances", "out_indices", "B", "Q", "M", "K"], "cluster_dims": [1, 1, 1], "computed_smem_bytes": 16384, "constants": [["D_", 2], ["K_MAX_", 64], ["LOCAL_CAP_", 16]], "cta_group": 1, "threads": 128}'))
DBSCAN_D2_SHAPES: list[dict[str, Any]] = [{'label': 'dbscan_lowd_self_q1500_m1500_d2_k32', 'params': {'B': 1, 'Q': 1500, 'M': 1500, 'D': 2, 'K': 32, 'dtype': 'bfloat16', 'seed': 610405, 'self_search': True, 'min_recall': 0.999}}, {'label': 'dbscan_lowd_self_q1500_m1500_d2_k64', 'params': {'B': 1, 'Q': 1500, 'M': 1500, 'D': 2, 'K': 64, 'dtype': 'bfloat16', 'seed': 610407, 'self_search': True, 'min_recall': 0.999}}]

def _compile_lowd_kernels() -> dict[str, Any]:
    return _decode_capture(_json_loads('{"direct": {"__kernel__": "dispatch_kernel_0178"}}'))

def _use_d2_dbscan_direct(inputs: dict[str, Any]) -> bool:
    return int(inputs['B']) == 1 and int(inputs['D']) == D_STATIC and (int(inputs['Q']) <= M_MAX) and (int(inputs['M']) <= M_MAX) and (int(inputs['K']) <= K_MAX)

def launch_for_eval(inputs: dict[str, Any]) -> dict[str, Any]:
    import torch
    if not _use_d2_dbscan_direct(inputs):
        return parent.launch_for_eval(inputs)
    if not _KNN_LOWD_KERNELS:
        _KNN_LOWD_KERNELS.update(_compile_lowd_kernels())
    _KNN_LOWD_KERNELS['direct'].launch(grid=(int(inputs['B']) * int(inputs['Q']), 1, 1), block=(THREADS, 1, 1), args=[inputs['queries'], inputs['database'], inputs['out_distances'], inputs['out_indices'], int(inputs['B']), int(inputs['Q']), int(inputs['M']), int(inputs['K'])], shared_mem=DIRECT_SMEM_BYTES)
    return {'distances': inputs['out_distances'], 'indices': inputs['out_indices']}

def _select_contract_shapes(shape_labels: str | tuple[str, ...] | list[str] | None):
    return parent._select_contract_shapes(shape_labels)

def knn_search_compile_and_launch_lowd_dbscan(*, benchmark: bool=True, shapes: list[dict[str, Any]] | None=None) -> dict[str, Any]:
    from .._dispatch_runtime import evaluate
    result = evaluate(launch_for_eval, shapes=shapes, benchmark=benchmark)
    result['passed'] = bool(result.get('summary', {}).get('all_correct'))
    print(result)
    return result
