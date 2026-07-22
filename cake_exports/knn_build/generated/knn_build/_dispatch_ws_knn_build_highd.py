"""Generated workspace-shape table for the dispatch-family .so loader."""

from __future__ import annotations

WORKSPACE_KEYS = ('partial_dists', 'partial_indices')
WORKSPACE_DTYPES = {'partial_dists': 'float32', 'partial_indices': 'int32'}
SCALAR_NAMES = ('B', 'Q', 'M', 'D', 'K', 'dtype', 'build')
ROUTE_IDS = ('knn_build_highd_d96_build', 'knn_build_highd_d320_build', 'knn_build_highd_d320_search_recurrence', 'knn_build_highd_d768_rag_q8', 'knn_build_highd_d768_rag_q16', 'knn_build_highd_d768_search', 'knn_build_highd_d768_build', 'knn_build_highd_d1024_rag_continuous', 'knn_build_highd_d4096_rag_continuous', 'knn_build_highd_d1024_search', 'knn_build_highd_d4096_search', 'knn_build_highd_d1024_build', 'knn_build_highd_d4096_build')
ROUTE_LAUNCH_COUNTS = (2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2)
ROUTE_INPUT_TENSORS = (('database', 'database_sq', 'out_dists', 'out_indices', 'queries', 'query_sq'), ('database', 'database_sq', 'out_dists', 'out_indices', 'queries', 'query_sq'), ('database', 'database_sq', 'out_dists', 'out_indices', 'queries', 'query_sq'), ('database', 'database_sq', 'out_dists', 'out_indices', 'queries', 'query_sq'), ('database', 'database_sq', 'out_dists', 'out_indices', 'queries', 'query_sq'), ('database', 'database_sq', 'out_dists', 'out_indices', 'queries', 'query_sq'), ('database', 'database_sq', 'out_dists', 'out_indices', 'queries', 'query_sq'), ('database', 'database_sq', 'out_dists', 'out_indices', 'queries', 'query_sq'), ('database', 'database_sq', 'out_dists', 'out_indices', 'queries', 'query_sq'), ('database', 'database_sq', 'out_dists', 'out_indices', 'queries', 'query_sq'), ('database', 'database_sq', 'out_dists', 'out_indices', 'queries', 'query_sq'), ('database', 'database_sq', 'out_dists', 'out_indices', 'queries', 'query_sq'), ('database', 'database_sq', 'out_dists', 'out_indices', 'queries', 'query_sq'))

def workspace_shapes(route_index: int, B, Q, M, D, K, dtype, build):
    if route_index == 0:
        return {"partial_dists": (8, B, Q, K,), "partial_indices": (8, B, Q, K,)}
    if route_index == 1:
        return {"partial_dists": (8, B, Q, K,), "partial_indices": (8, B, Q, K,)}
    if route_index == 2:
        return {"partial_dists": (48, B, Q, K,), "partial_indices": (48, B, Q, K,)}
    if route_index == 3:
        return {"partial_dists": (144, B, Q, K,), "partial_indices": (144, B, Q, K,)}
    if route_index == 4:
        return {"partial_dists": (72, B, Q, K,), "partial_indices": (72, B, Q, K,)}
    if route_index == 5:
        return {"partial_dists": (32, B, Q, K,), "partial_indices": (32, B, Q, K,)}
    if route_index == 6:
        return {"partial_dists": (16, B, Q, K,), "partial_indices": (16, B, Q, K,)}
    if route_index == 7:
        return {"partial_dists": (144, B, Q, K,), "partial_indices": (144, B, Q, K,)}
    if route_index == 8:
        return {"partial_dists": (128, B, Q, K,), "partial_indices": (128, B, Q, K,)}
    if route_index == 9:
        return {"partial_dists": (64, B, Q, K,), "partial_indices": (64, B, Q, K,)}
    if route_index == 10:
        return {"partial_dists": (64, B, Q, K,), "partial_indices": (64, B, Q, K,)}
    if route_index == 11:
        return {"partial_dists": (8, B, Q, K,), "partial_indices": (8, B, Q, K,)}
    if route_index == 12:
        return {"partial_dists": (8, B, Q, K,), "partial_indices": (8, B, Q, K,)}
    return {}
