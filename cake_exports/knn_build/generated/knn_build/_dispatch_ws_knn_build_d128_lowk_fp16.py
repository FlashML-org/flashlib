"""Generated workspace-shape table for the dispatch-family .so loader."""

from __future__ import annotations

WORKSPACE_KEYS = ('partial_dists', 'partial_indices')
WORKSPACE_DTYPES = {'partial_dists': 'float32', 'partial_indices': 'int32'}
SCALAR_NAMES = ('B', 'Q', 'M', 'D', 'K', 'dtype', 'build')
ROUTE_IDS = ('knn_build_d128_lowk_k10_fp16_q2048_fd37_s8',)
ROUTE_LAUNCH_COUNTS = (2,)
ROUTE_INPUT_TENSORS = (('database', 'database_sq', 'out_dists', 'out_indices', 'queries', 'query_sq'),)

def workspace_shapes(route_index: int, B, Q, M, D, K, dtype, build):
    if route_index == 0:
        return {"partial_dists": (8, B, Q, K,), "partial_indices": (8, B, Q, K,)}
    return {}
