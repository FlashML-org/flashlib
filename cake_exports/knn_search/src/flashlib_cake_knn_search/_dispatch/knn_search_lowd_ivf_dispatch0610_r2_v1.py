"""Exact BF16 squared-L2 kNN path for small-M IVF-like low-D shapes.

Minimum target architecture: sm_80. This additive dispatch candidate handles
``D in {32, 48}, M<=32, K<=10`` directly in Weave for the expanded IVF-like
coverage labels. Other inherited labels delegate to the round-1 low-D DBSCAN
wrapper, which in turn delegates the incumbent K<=10 frontier.
"""
from __future__ import annotations
from json import loads as _json_loads
from .._dispatch_runtime import _capture_cuTensorMapEncodeTiled, _decode_capture, _import_dispatch_module, _ir_proxy
from typing import Any
from .knn_search_lowd_dbscan_dispatch0610_r1_v1 import launch_for_eval as _round1_launch_for_eval
THREADS = 32
D_MAX = 48
M_MAX = 32
K_MAX = 10
DIST_BYTES = M_MAX * 4
IDX_BYTES = M_MAX * 4
SMEM_POOL_BYTES = DIST_BYTES + IDX_BYTES
_KERNELS: dict[str, Any] = {}
knn_search_lowd_ivf_direct_dispatch0610_r2_v1 = _decode_capture(_json_loads('{"__ir__": "knn_search_lowd_ivf_direct_dispatch0610_r2_v1", "arg_keys": ["queries", "database", "out_distances", "out_indices", "B", "Q", "M", "K", "D"], "cluster_dims": [1, 1, 1], "computed_smem_bytes": 256, "constants": [["K_MAX_", 10], ["M_MAX_", 32]], "cta_group": 1, "threads": 32}'))
ir = _decode_capture(_json_loads('{"__ir__": "knn_search_lowd_ivf_direct_dispatch0610_r2_v1", "arg_keys": ["queries", "database", "out_distances", "out_indices", "B", "Q", "M", "K", "D"], "cluster_dims": [1, 1, 1], "computed_smem_bytes": 256, "constants": [["K_MAX_", 10], ["M_MAX_", 32]], "cta_group": 1, "threads": 32}'))

def _compile_kernels() -> dict[str, Any]:
    return _decode_capture(_json_loads('{"direct": {"__kernel__": "dispatch_kernel_0175"}}'))

def _use_ivf_small(inputs: dict[str, Any]) -> bool:
    dim = int(inputs['D'])
    return dim in {32, 48} and int(inputs['M']) <= M_MAX and (int(inputs['K']) <= K_MAX) and (dim % 8 == 0)

def launch_for_eval(inputs: dict[str, Any]) -> dict[str, Any]:
    import torch
    if not _use_ivf_small(inputs):
        return _round1_launch_for_eval(inputs)
    if not _KERNELS:
        _KERNELS.update(_compile_kernels())
    bsz = int(inputs['B'])
    q_rows = int(inputs['Q'])
    _KERNELS['direct'].launch(grid=(bsz * q_rows, 1, 1), block=(THREADS, 1, 1), args=[inputs['queries'], inputs['database'], inputs['out_distances'], inputs['out_indices'], bsz, q_rows, int(inputs['M']), int(inputs['K']), int(inputs['D'])], shared_mem=SMEM_POOL_BYTES)
    return {'distances': inputs['out_distances'], 'indices': inputs['out_indices']}
