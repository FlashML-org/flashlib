"""Generated workspace-shape table for the dispatch-family .so loader."""

from __future__ import annotations

WORKSPACE_KEYS = ('partial_dists', 'partial_indices')
WORKSPACE_DTYPES = {'partial_dists': 'float32', 'partial_indices': 'int32'}
SCALAR_NAMES = ('B', 'Q', 'M', 'D', 'K', 'dtype', 'build')
ROUTE_IDS = ('knn_build_d256_e2df_q4', 'knn_build_d256_rag_q16', 'knn_build_d256_q128_longm', 'knn_build_d256_build_q1024', 'knn_build_d256_search_q1024', 'knn_build_d256_df2f_q2048', 'knn_build_d256_k32_tail_q8', 'knn_build_d256_k32_tail_q128')
ROUTE_LAUNCH_COUNTS = (2, 2, 2, 2, 2, 2, 2, 2)
ROUTE_INPUT_TENSORS = (('database', 'database_sq', 'out_dists', 'out_indices', 'queries', 'query_sq'), ('database', 'database_sq', 'out_dists', 'out_indices', 'queries', 'query_sq'), ('database', 'database_sq', 'out_dists', 'out_indices', 'queries', 'query_sq'), ('database', 'database_sq', 'out_dists', 'out_indices', 'queries', 'query_sq'), ('database', 'database_sq', 'out_dists', 'out_indices', 'queries', 'query_sq'), ('database', 'database_sq', 'out_dists', 'out_indices', 'queries', 'query_sq'), ('database', 'database_sq', 'out_dists', 'out_indices', 'queries', 'query_sq'), ('database', 'database_sq', 'out_dists', 'out_indices', 'queries', 'query_sq'))

def workspace_shapes(route_index: int, B, Q, M, D, K, dtype, build):
    if route_index == 0:
        return {"partial_dists": (144, B, Q, K,), "partial_indices": (144, B, Q, K,)}
    if route_index == 1:
        return {"partial_dists": (144, B, Q, K,), "partial_indices": (144, B, Q, K,)}
    if route_index == 2:
        return {"partial_dists": (144, B, Q, K,), "partial_indices": (144, B, Q, K,)}
    if route_index == 3:
        return {"partial_dists": (16, B, Q, K,), "partial_indices": (16, B, Q, K,)}
    if route_index == 4:
        return {"partial_dists": (16, B, Q, K,), "partial_indices": (16, B, Q, K,)}
    if route_index == 5:
        return {"partial_dists": (8, B, Q, K,), "partial_indices": (8, B, Q, K,)}
    if route_index == 6:
        return {"partial_dists": (144, B, Q, K,), "partial_indices": (144, B, Q, K,)}
    if route_index == 7:
        return {"partial_dists": (64, B, Q, K,), "partial_indices": (64, B, Q, K,)}
    return {}
