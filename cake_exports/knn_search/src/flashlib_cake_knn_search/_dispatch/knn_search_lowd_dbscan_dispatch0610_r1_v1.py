"""Exact BF16 squared-L2 kNN capacity path for low-D DBSCAN shapes.

Minimum target architecture: sm_80. This additive dispatch candidate handles
``D=2, Q%4==0, M%4==0, M<=2048, K<=64`` directly in Weave and delegates the
existing K<=10 frontier to ``knn_search_mma_split_v1``. It is clean-room code
for the expanded DBSCAN coverage labels, not a FlashLib port.
"""
from __future__ import annotations
from json import loads as _json_loads
from .._dispatch_runtime import _capture_cuTensorMapEncodeTiled, _decode_capture, _import_dispatch_module, _ir_proxy
from typing import Any
from .knn_search_mma_split_v1 import launch_for_eval as _incumbent_launch_for_eval
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
SMEM_POOL_BYTES = WARP_THREAD_OFFSET + NUM_WARPS * 4
_KERNELS: dict[str, Any] = {}
knn_search_lowd_dbscan_direct_dispatch0610_r1_v1 = _decode_capture(_json_loads('{"__ir__": "knn_search_lowd_dbscan_direct_dispatch0610_r1_v1", "arg_keys": ["queries", "database", "out_distances", "out_indices", "B", "Q", "M", "K"], "cluster_dims": [1, 1, 1], "computed_smem_bytes": 16512, "constants": [["D_", 2], ["K_MAX_", 64], ["LOCAL_LIST_CAP_", 8], ["NUM_WARPS_", 8]], "cta_group": 1, "threads": 256}'))
ir = _decode_capture(_json_loads('{"__ir__": "knn_search_lowd_dbscan_direct_dispatch0610_r1_v1", "arg_keys": ["queries", "database", "out_distances", "out_indices", "B", "Q", "M", "K"], "cluster_dims": [1, 1, 1], "computed_smem_bytes": 16512, "constants": [["D_", 2], ["K_MAX_", 64], ["LOCAL_LIST_CAP_", 8], ["NUM_WARPS_", 8]], "cta_group": 1, "threads": 256}'))

def _compile_kernels() -> dict[str, Any]:
    return _decode_capture(_json_loads('{"direct": {"__kernel__": "dispatch_kernel_0283"}}'))

def _use_lowd_capacity(inputs: dict[str, Any]) -> bool:
    return int(inputs['D']) == D_STATIC and int(inputs['Q']) % 4 == 0 and (int(inputs['M']) % 4 == 0) and (int(inputs['M']) <= M_MAX) and (int(inputs['K']) <= K_MAX)

def launch_for_eval(inputs: dict[str, Any]) -> dict[str, Any]:
    import torch
    if not _use_lowd_capacity(inputs):
        if int(inputs['K']) <= 10:
            return _incumbent_launch_for_eval(inputs)
        raise ValueError(''.join(['knn_search_lowd_dbscan_dispatch0610_r1_v1 supports K>10 only for D=', format(D_STATIC, ''), ', Q%4==0, M%4==0, M<=', format(M_MAX, ''), '; got Q=', format(inputs['Q'], ''), ', D=', format(inputs['D'], ''), ', M=', format(inputs['M'], ''), ', K=', format(inputs['K'], '')]))
    if not _KERNELS:
        _KERNELS.update(_compile_kernels())
    bsz = int(inputs['B'])
    q_rows = int(inputs['Q'])
    k = int(inputs['K'])
    _KERNELS['direct'].launch(grid=(bsz * q_rows, 1, 1), block=(THREADS, 1, 1), args=[inputs['queries'], inputs['database'], inputs['out_distances'], inputs['out_indices'], bsz, q_rows, int(inputs['M']), k], shared_mem=SMEM_POOL_BYTES)
    return {'distances': inputs['out_distances'], 'indices': inputs['out_indices']}
