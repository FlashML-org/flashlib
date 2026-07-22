"""Generated workspace-shape table for the dispatch-family .so loader."""

from __future__ import annotations

WORKSPACE_KEYS = ('partial_dists', 'partial_indices')
WORKSPACE_DTYPES = {'partial_dists': 'float32', 'partial_indices': 'int32'}
SCALAR_NAMES = ('B', 'Q', 'M', 'D', 'K', 'dtype', 'build')
ROUTE_IDS = ('knn_build_d128_largek_dd2b_k48_q16', 'knn_build_d128_largek_k48_7dc5_d03c', 'knn_build_d128_largek_k96_s2', 'knn_build_d128_largek_k96_s4', 'knn_build_d128_largek_k64_continuous')
ROUTE_LAUNCH_COUNTS = (2, 2, 2, 2, 2)
ROUTE_INPUT_TENSORS = (('database', 'database_sq', 'out_dists', 'out_indices', 'queries', 'query_sq'), ('database', 'database_sq', 'out_dists', 'out_indices', 'queries', 'query_sq'), ('database', 'database_sq', 'out_dists', 'out_indices', 'queries', 'query_sq'), ('database', 'database_sq', 'out_dists', 'out_indices', 'queries', 'query_sq'), ('database', 'database_sq', 'out_dists', 'out_indices', 'queries', 'query_sq'))

def workspace_shapes(route_index: int, B, Q, M, D, K, dtype, build):
    if route_index == 0:
        return {"partial_dists": (144, B, Q, K,), "partial_indices": (144, B, Q, K,)}
    if route_index == 1:
        return {"partial_dists": (4, B, Q, K,), "partial_indices": (4, B, Q, K,)}
    if route_index == 2:
        return {"partial_dists": (2, B, Q, K,), "partial_indices": (2, B, Q, K,)}
    if route_index == 3:
        return {"partial_dists": (4, B, Q, K,), "partial_indices": (4, B, Q, K,)}
    if route_index == 4:
        return {"partial_dists": (8, B, Q, K,), "partial_indices": (8, B, Q, K,)}
    return {}
