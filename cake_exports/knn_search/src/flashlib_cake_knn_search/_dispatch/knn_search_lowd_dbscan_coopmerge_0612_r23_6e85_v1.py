"""Round-23 low-D DBSCAN cooperative-merge route for exact BF16 squared-L2 kNN.

Minimum target architecture: sm_80. This additive clean-room candidate handles
D=2, K<=64, M<=2048 DBSCAN/self-search shapes with CUDA-core scalar distance
and a block-cooperative top-K merge, then delegates all other shapes to the
round-22 low-D wrapper.
"""
from __future__ import annotations
from json import loads as _json_loads
from .._dispatch_runtime import _capture_cuTensorMapEncodeTiled, _decode_capture, _import_dispatch_module, _ir_proxy
from typing import Any
from . import knn_search_lowd_dbscan_0611_r22_6e85_v1 as parent
THREADS = 256
NUM_WARPS = THREADS // 32
D_STATIC = 2
K_MAX = 64
M_MAX = 2048
LOCAL_LIST_CAP = (M_MAX + THREADS - 1) // THREADS
LOCAL_CANDIDATES = THREADS * LOCAL_LIST_CAP
LOCAL_DIST_BYTES = LOCAL_CANDIDATES * 4
LOCAL_IDX_BYTES = LOCAL_CANDIDATES * 4
WARP_DIST_OFFSET = LOCAL_DIST_BYTES + LOCAL_IDX_BYTES
WARP_IDX_OFFSET = WARP_DIST_OFFSET + NUM_WARPS * 4
WARP_THREAD_OFFSET = WARP_IDX_OFFSET + NUM_WARPS * 4
DIRECT_SMEM_BYTES = WARP_THREAD_OFFSET + NUM_WARPS * 4
_KNN_LOWD_COOP_KERNELS: dict[str, Any] = {}
knn_search_lowd_dbscan_d2_coopmerge_0612_r23_6e85_v1 = _decode_capture(_json_loads('{"__ir__": "knn_search_lowd_dbscan_d2_coopmerge_0612_r23_6e85_v1", "arg_keys": ["queries", "database", "out_distances", "out_indices", "B", "Q", "M", "K"], "cluster_dims": [1, 1, 1], "computed_smem_bytes": 16512, "constants": [["D_", 2], ["K_MAX_", 64], ["LOCAL_LIST_CAP_", 8], ["NUM_WARPS_", 8]], "cta_group": 1, "threads": 256}'))
ir = _decode_capture(_json_loads('{"__ir__": "knn_search_lowd_dbscan_d2_coopmerge_0612_r23_6e85_v1", "arg_keys": ["queries", "database", "out_distances", "out_indices", "B", "Q", "M", "K"], "cluster_dims": [1, 1, 1], "computed_smem_bytes": 16512, "constants": [["D_", 2], ["K_MAX_", 64], ["LOCAL_LIST_CAP_", 8], ["NUM_WARPS_", 8]], "cta_group": 1, "threads": 256}'))
DBSCAN_D2_SHAPES: list[dict[str, Any]] = parent.DBSCAN_D2_SHAPES

def _compile_lowd_coop_kernels() -> dict[str, Any]:
    return _decode_capture(_json_loads('{"direct": {"__kernel__": "dispatch_kernel_0177"}}'))

def _use_d2_dbscan_coop(inputs: dict[str, Any]) -> bool:
    return int(inputs['B']) == 1 and int(inputs['D']) == D_STATIC and (int(inputs['Q']) <= M_MAX) and (int(inputs['M']) <= M_MAX) and (int(inputs['K']) <= K_MAX)

def launch_for_eval(inputs: dict[str, Any]) -> dict[str, Any]:
    import torch
    if not _use_d2_dbscan_coop(inputs):
        return parent.launch_for_eval(inputs)
    if not _KNN_LOWD_COOP_KERNELS:
        _KNN_LOWD_COOP_KERNELS.update(_compile_lowd_coop_kernels())
    _KNN_LOWD_COOP_KERNELS['direct'].launch(grid=(int(inputs['B']) * int(inputs['Q']), 1, 1), block=(THREADS, 1, 1), args=[inputs['queries'], inputs['database'], inputs['out_distances'], inputs['out_indices'], int(inputs['B']), int(inputs['Q']), int(inputs['M']), int(inputs['K'])], shared_mem=DIRECT_SMEM_BYTES)
    return {'distances': inputs['out_distances'], 'indices': inputs['out_indices']}

def _select_contract_shapes(shape_labels: str | tuple[str, ...] | list[str] | None):
    return parent._select_contract_shapes(shape_labels)

def knn_search_compile_and_launch_lowd_dbscan_coopmerge(*, benchmark: bool=True, shapes: list[dict[str, Any]] | None=None) -> dict[str, Any]:
    from .._dispatch_runtime import evaluate
    result = evaluate(launch_for_eval, shapes=shapes, benchmark=benchmark)
    result['passed'] = bool(result.get('summary', {}).get('all_correct'))
    print(result)
    return result
