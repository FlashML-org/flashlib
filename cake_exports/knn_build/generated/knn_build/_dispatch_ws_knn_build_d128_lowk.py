"""Generated workspace-shape table for the dispatch-family .so loader."""

from __future__ import annotations

WORKSPACE_KEYS = ('partial_dists', 'partial_indices')
WORKSPACE_DTYPES = {'partial_dists': 'float32', 'partial_indices': 'int32'}
SCALAR_NAMES = ('B', 'Q', 'M', 'D', 'K', 'dtype', 'build')
ROUTE_IDS = ('knn_build_d128_lowk_k1_q512_group2_s2', 'knn_build_d128_lowk_k2456_q512_lowk_s4', 'knn_build_d128_lowk_k5_q256_flashml_s4', 'knn_build_d128_lowk_k8_q512_q2048_s8', 'knn_build_d128_lowk_k8_q1024_195e_s16', 'knn_build_d128_lowk_k8_q4096_fd9b_s4', 'knn_build_d128_lowk_k10_q1_ragonline_s144g12', 'knn_build_d128_lowk_k10_q1_m524287_ea43_s147g7', 'knn_build_d128_lowk_k10_q4_q8_microbatch_s144g12', 'knn_build_d128_lowk_k10_q16_m64_s136g8', 'knn_build_d128_lowk_k10_q32_q64_m64_s128g8', 'knn_build_d128_lowk_k10_q128_rowld_s74', 'knn_build_d128_lowk_k10_frontier_s72', 'knn_build_d128_lowk_k10_build_rowbase_s4', 'knn_build_d128_lowk_k10_build_rowbase_s7', 'knn_build_d128_lowk_k10_rect_q1024_d15e_s16', 'knn_build_d128_lowk_k10_rect_q2048_4452_s8', 'knn_build_d128_lowk_k10_evolve_7bfc_continuous')
ROUTE_LAUNCH_COUNTS = (2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2)
ROUTE_INPUT_TENSORS = (('database', 'database_sq', 'out_dists', 'out_indices', 'queries', 'query_sq'), ('database', 'database_sq', 'out_dists', 'out_indices', 'queries', 'query_sq'), ('database', 'database_sq', 'out_dists', 'out_indices', 'queries', 'query_sq'), ('database', 'database_sq', 'out_dists', 'out_indices', 'queries', 'query_sq'), ('database', 'database_sq', 'out_dists', 'out_indices', 'queries', 'query_sq'), ('database', 'database_sq', 'out_dists', 'out_indices', 'queries', 'query_sq'), ('database', 'database_sq', 'out_dists', 'out_indices', 'queries', 'query_sq'), ('database', 'database_sq', 'out_dists', 'out_indices', 'queries', 'query_sq'), ('database', 'database_sq', 'out_dists', 'out_indices', 'queries', 'query_sq'), ('database', 'database_sq', 'out_dists', 'out_indices', 'queries', 'query_sq'), ('database', 'database_sq', 'out_dists', 'out_indices', 'queries', 'query_sq'), ('database', 'database_sq', 'out_dists', 'out_indices', 'queries', 'query_sq'), ('database', 'database_sq', 'out_dists', 'out_indices', 'queries', 'query_sq'), ('database', 'database_sq', 'out_dists', 'out_indices', 'queries', 'query_sq'), ('database', 'database_sq', 'out_dists', 'out_indices', 'queries', 'query_sq'), ('database', 'database_sq', 'out_dists', 'out_indices', 'queries', 'query_sq'), ('database', 'database_sq', 'out_dists', 'out_indices', 'queries', 'query_sq'), ('database', 'database_sq', 'out_dists', 'out_indices', 'queries', 'query_sq'))

def workspace_shapes(route_index: int, B, Q, M, D, K, dtype, build):
    if route_index == 0:
        return {"partial_dists": (2, B, Q, K,), "partial_indices": (2, B, Q, K,)}
    if route_index == 1:
        return {"partial_dists": (4, B, Q, K,), "partial_indices": (4, B, Q, K,)}
    if route_index == 2:
        return {"partial_dists": (4, B, Q, K,), "partial_indices": (4, B, Q, K,)}
    if route_index == 3:
        return {"partial_dists": (8, B, Q, K,), "partial_indices": (8, B, Q, K,)}
    if route_index == 4:
        return {"partial_dists": (16, B, Q, K,), "partial_indices": (16, B, Q, K,)}
    if route_index == 5:
        return {"partial_dists": (4, B, Q, K,), "partial_indices": (4, B, Q, K,)}
    if route_index == 6:
        return {"partial_dists": (144, B, Q, K,), "partial_indices": (144, B, Q, K,)}
    if route_index == 7:
        return {"partial_dists": (147, B, Q, K,), "partial_indices": (147, B, Q, K,)}
    if route_index == 8:
        return {"partial_dists": (144, B, Q, K,), "partial_indices": (144, B, Q, K,)}
    if route_index == 9:
        return {"partial_dists": (136, B, Q, K,), "partial_indices": (136, B, Q, K,)}
    if route_index == 10:
        return {"partial_dists": (128, B, Q, K,), "partial_indices": (128, B, Q, K,)}
    if route_index == 11:
        return {"partial_dists": (74, B, Q, K,), "partial_indices": (74, B, Q, K,)}
    if route_index == 12:
        return {"partial_dists": (72, B, Q, K,), "partial_indices": (72, B, Q, K,)}
    if route_index == 13:
        return {"partial_dists": (4, B, Q, K,), "partial_indices": (4, B, Q, K,)}
    if route_index == 14:
        return {"partial_dists": (7, B, Q, K,), "partial_indices": (7, B, Q, K,)}
    if route_index == 15:
        return {"partial_dists": (16, B, Q, K,), "partial_indices": (16, B, Q, K,)}
    if route_index == 16:
        return {"partial_dists": (8, B, Q, K,), "partial_indices": (8, B, Q, K,)}
    if route_index == 17:
        return {"partial_dists": (4, B, Q, K,), "partial_indices": (4, B, Q, K,)}
    return {}
