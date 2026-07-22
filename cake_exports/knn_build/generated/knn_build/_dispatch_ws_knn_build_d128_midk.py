"""Generated workspace-shape table for the dispatch-family .so loader."""

from __future__ import annotations

WORKSPACE_KEYS = ('partial_dists', 'partial_indices')
WORKSPACE_DTYPES = {'partial_dists': 'float32', 'partial_indices': 'int32'}
SCALAR_NAMES = ('B', 'Q', 'M', 'D', 'K', 'dtype', 'build')
ROUTE_IDS = ('knn_build_d128_midk_k11_q2048_e080', 'knn_build_d128_midk_k12_q1024_s16', 'knn_build_d128_midk_k12_q2048_s8', 'knn_build_d128_midk_k12_q4096_unordered_s4', 'knn_build_d128_midk_k13_q2048_e080', 'knn_build_d128_midk_k13_q4096_2c1c_unordered_s4', 'knn_build_d128_midk_k16_q1024_f8c3_s16', 'knn_build_d128_midk_k20_q1024_s16', 'knn_build_d128_midk_k20_q2048_s8', 'knn_build_d128_midk_k20_q3072_ordered_s4', 'knn_build_d128_midk_k20_q4096_unordered_s4', 'knn_build_d128_midk_k20_warpselect_s4_continuous', 'knn_build_d128_midk_k20_q8192_square_s2warp8', 'knn_build_d128_midk_k20_q8192_lowfanout_s2', 'knn_build_d128_midk_k20_q1536_rect_s12warp4', 'knn_build_d128_midk_k24_q2048_s8', 'knn_build_d128_midk_k24_q4096_1074_unordered_s4', 'knn_build_d128_midk_k28_q2048_s8', 'knn_build_d128_midk_k28_q4096_unordered_s4', 'knn_build_d128_midk_k30_q4096_6998_unordered_s4', 'knn_build_d128_midk_k32_q8192_square_s2warp8', 'knn_build_d128_midk_k32_q4096_overk_unordered_s4', 'knn_build_d128_midk_k32_q16_m100000_56ed_s144', 'knn_build_d128_midk_k32_q16_m131071_56ed_s148', 'knn_build_d128_midk_k32_q16_m250000_56ed_s288', 'knn_build_d128_midk_k32_q24_24dc_s144', 'knn_build_d128_midk_k32_q32_f653_s141', 'knn_build_d128_midk_k32_q8_q8half_s144', 'knn_build_d128_midk_k32_q128_m100000_staticn128_s72', 'knn_build_d128_midk_k32_q128_m131071_e5db_s72', 'knn_build_d128_midk_k11_32_build_k32split_continuous', 'knn_build_d128_midk_k11_32_search_evolve_7bfc_v1_continuous')
ROUTE_LAUNCH_COUNTS = (2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 1)
ROUTE_INPUT_TENSORS = (('database', 'database_sq', 'out_dists', 'out_indices', 'queries', 'query_sq'), ('database', 'database_sq', 'out_dists', 'out_indices', 'queries', 'query_sq'), ('database', 'database_sq', 'out_dists', 'out_indices', 'queries', 'query_sq'), ('database', 'database_sq', 'out_dists', 'out_indices', 'queries', 'query_sq'), ('database', 'database_sq', 'out_dists', 'out_indices', 'queries', 'query_sq'), ('database', 'database_sq', 'out_dists', 'out_indices', 'queries', 'query_sq'), ('database', 'database_sq', 'out_dists', 'out_indices', 'queries', 'query_sq'), ('database', 'database_sq', 'out_dists', 'out_indices', 'queries', 'query_sq'), ('database', 'database_sq', 'out_dists', 'out_indices', 'queries', 'query_sq'), ('database', 'database_sq', 'out_dists', 'out_indices', 'queries', 'query_sq'), ('database', 'database_sq', 'out_dists', 'out_indices', 'queries', 'query_sq'), ('database', 'database_sq', 'out_dists', 'out_indices', 'queries', 'query_sq'), ('database', 'database_sq', 'out_dists', 'out_indices', 'queries', 'query_sq'), ('database', 'database_sq', 'out_dists', 'out_indices', 'queries', 'query_sq'), ('database', 'database_sq', 'out_dists', 'out_indices', 'queries', 'query_sq'), ('database', 'database_sq', 'out_dists', 'out_indices', 'queries', 'query_sq'), ('database', 'database_sq', 'out_dists', 'out_indices', 'queries', 'query_sq'), ('database', 'database_sq', 'out_dists', 'out_indices', 'queries', 'query_sq'), ('database', 'database_sq', 'out_dists', 'out_indices', 'queries', 'query_sq'), ('database', 'database_sq', 'out_dists', 'out_indices', 'queries', 'query_sq'), ('database', 'database_sq', 'out_dists', 'out_indices', 'queries', 'query_sq'), ('database', 'database_sq', 'out_dists', 'out_indices', 'queries', 'query_sq'), ('database', 'database_sq', 'out_dists', 'out_indices', 'queries', 'query_sq'), ('database', 'database_sq', 'out_dists', 'out_indices', 'queries', 'query_sq'), ('database', 'database_sq', 'out_dists', 'out_indices', 'queries', 'query_sq'), ('database', 'database_sq', 'out_dists', 'out_indices', 'queries', 'query_sq'), ('database', 'database_sq', 'out_dists', 'out_indices', 'queries', 'query_sq'), ('database', 'database_sq', 'out_dists', 'out_indices', 'queries', 'query_sq'), ('database', 'database_sq', 'out_dists', 'out_indices', 'queries', 'query_sq'), ('database', 'database_sq', 'out_dists', 'out_indices', 'queries', 'query_sq'), ('database', 'database_sq', 'out_dists', 'out_indices', 'queries', 'query_sq'), ('database', 'database_sq', 'out_dists', 'out_indices', 'queries', 'query_sq'))

def workspace_shapes(route_index: int, B, Q, M, D, K, dtype, build):
    if route_index == 0:
        return {"partial_dists": (8, B, Q, K,), "partial_indices": (8, B, Q, K,)}
    if route_index == 1:
        return {"partial_dists": (16, B, Q, K,), "partial_indices": (16, B, Q, K,)}
    if route_index == 2:
        return {"partial_dists": (8, B, Q, K,), "partial_indices": (8, B, Q, K,)}
    if route_index == 3:
        return {"partial_dists": (4, B, Q, K,), "partial_indices": (4, B, Q, K,)}
    if route_index == 4:
        return {"partial_dists": (8, B, Q, K,), "partial_indices": (8, B, Q, K,)}
    if route_index == 5:
        return {"partial_dists": (4, B, Q, K,), "partial_indices": (4, B, Q, K,)}
    if route_index == 6:
        return {"partial_dists": (16, B, Q, K,), "partial_indices": (16, B, Q, K,)}
    if route_index == 7:
        return {"partial_dists": (16, B, Q, K,), "partial_indices": (16, B, Q, K,)}
    if route_index == 8:
        return {"partial_dists": (8, B, Q, K,), "partial_indices": (8, B, Q, K,)}
    if route_index == 9:
        return {"partial_dists": (4, B, Q, K,), "partial_indices": (4, B, Q, K,)}
    if route_index == 10:
        return {"partial_dists": (4, B, Q, K,), "partial_indices": (4, B, Q, K,)}
    if route_index == 11:
        return {"partial_dists": (4, B, Q, K,), "partial_indices": (4, B, Q, K,)}
    if route_index == 12:
        return {"partial_dists": (2, B, Q, K,), "partial_indices": (2, B, Q, K,)}
    if route_index == 13:
        return {"partial_dists": (2, B, Q, K,), "partial_indices": (2, B, Q, K,)}
    if route_index == 14:
        return {"partial_dists": (12, B, Q, K,), "partial_indices": (12, B, Q, K,)}
    if route_index == 15:
        return {"partial_dists": (8, B, Q, K,), "partial_indices": (8, B, Q, K,)}
    if route_index == 16:
        return {"partial_dists": (4, B, Q, K,), "partial_indices": (4, B, Q, K,)}
    if route_index == 17:
        return {"partial_dists": (8, B, Q, K,), "partial_indices": (8, B, Q, K,)}
    if route_index == 18:
        return {"partial_dists": (4, B, Q, K,), "partial_indices": (4, B, Q, K,)}
    if route_index == 19:
        return {"partial_dists": (4, B, Q, K,), "partial_indices": (4, B, Q, K,)}
    if route_index == 20:
        return {"partial_dists": (2, B, Q, K,), "partial_indices": (2, B, Q, K,)}
    if route_index == 21:
        return {"partial_dists": (4, B, Q, K,), "partial_indices": (4, B, Q, K,)}
    if route_index == 22:
        return {"partial_dists": (144, B, Q, K,), "partial_indices": (144, B, Q, K,)}
    if route_index == 23:
        return {"partial_dists": (148, B, Q, K,), "partial_indices": (148, B, Q, K,)}
    if route_index == 24:
        return {"partial_dists": (288, B, Q, K,), "partial_indices": (288, B, Q, K,)}
    if route_index == 25:
        return {"partial_dists": (144, B, Q, K,), "partial_indices": (144, B, Q, K,)}
    if route_index == 26:
        return {"partial_dists": (141, B, Q, K,), "partial_indices": (141, B, Q, K,)}
    if route_index == 27:
        return {"partial_dists": (144, B, Q, K,), "partial_indices": (144, B, Q, K,)}
    if route_index == 28:
        return {"partial_dists": (72, B, Q, K,), "partial_indices": (72, B, Q, K,)}
    if route_index == 29:
        return {"partial_dists": (72, B, Q, K,), "partial_indices": (72, B, Q, K,)}
    if route_index == 30:
        return {"partial_dists": (4, B, Q, K,), "partial_indices": (4, B, Q, K,)}
    return {}
