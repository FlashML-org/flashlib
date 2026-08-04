"""Generated workspace-shape table for the dispatch-family .so loader."""

from __future__ import annotations

WORKSPACE_KEYS = ('partial_distances', 'partial_indices', 'group_distances', 'group_indices', 'tail_partial_distances', 'tail_partial_indices', 'tail_group_distances', 'tail_group_indices')
WORKSPACE_DTYPES = {'partial_distances': 'float32', 'partial_indices': 'int32', 'group_distances': 'float32', 'group_indices': 'int32', 'tail_partial_distances': 'float32', 'tail_partial_indices': 'int32', 'tail_group_distances': 'float32', 'tail_group_indices': 'int32'}
SCALAR_NAMES = ('B', 'Q', 'M', 'D', 'K', 'self_search')
ROUTE_IDS = ('round46_q4096_lowk_k2partial_split9', 'round51_registered_q4096_lowk_k5partial_split9', 'round47_registered_q4096_lowk_k8_stride10_out8_split4', 'round54_q4096_m16384_lowk_k8_stride10_out8_split4', 'round54_q4096_m32768_lowk_k8_stride10_out8_split4', 'afe6_dynamic_d_scalar_capacity', 'afe6_dynamic_d_scalar_capacity', 'round55_lowq_q8q16q32q64_row16_registered', 'round36_d256_k10_k64_tcgen05', 'round56_d2_dbscan_t128_m1536_cd72', 'residual0705_q63_masked_query_structural_reset_f392_tcgen05', 'residual0705_q64_tail_split152_full_wave_3dee_tcgen05', 'residual0705_q64_tail_plus_812c_sentinel_warp_state_361b_tcgen05', 'residual0705_q65_distinct_topology_structural_reset_d436_tcgen05', 'residual_q128_stability_split304_fullwaves_a8f5_tcgen05')
ROUTE_LAUNCH_COUNTS = (2, 2, 2, 2, 2, 1, 1, 2, 2, 1, 3, 3, 3, 5, 3)
ROUTE_INPUT_TENSORS = (('database', 'out_distances', 'out_indices', 'queries'), ('database', 'out_distances', 'out_indices', 'queries'), ('database', 'out_distances', 'out_indices', 'queries'), ('database', 'out_distances', 'out_indices', 'queries'), ('database', 'out_distances', 'out_indices', 'queries'), ('database', 'out_distances', 'out_indices', 'queries'), ('database', 'out_distances', 'out_indices', 'queries'), ('database', 'out_distances', 'out_indices', 'queries'), ('database', 'out_distances', 'out_indices', 'queries'), ('database', 'out_distances', 'out_indices', 'queries'), ('database', 'out_distances', 'out_indices', 'queries'), ('database', 'out_distances', 'out_indices', 'queries'), ('database', 'out_distances', 'out_indices', 'queries'), ('database', 'out_distances', 'out_indices', 'queries'), ('database', 'out_distances', 'out_indices', 'queries'))

def workspace_shapes(route_index: int, B, Q, M, D, K, self_search):
    if route_index == 0:
        return {"partial_distances": (B, 32, 9, 128, 2,), "partial_indices": (B, 32, 9, 128, 2,)}
    if route_index == 1:
        return {"partial_distances": (B, 32, 9, 128, 5,), "partial_indices": (B, 32, 9, 128, 5,)}
    if route_index == 2:
        return {"partial_distances": (B, 32, 4, 128, 10,), "partial_indices": (B, 32, 4, 128, 10,)}
    if route_index == 3:
        return {"partial_distances": (B, 32, 4, 128, 10,), "partial_indices": (B, 32, 4, 128, 10,)}
    if route_index == 4:
        return {"partial_distances": (B, 32, 4, 128, 10,), "partial_indices": (B, 32, 4, 128, 10,)}
    if route_index == 7:
        return {"partial_distances": (B, 1, 148, 128, 10,), "partial_indices": (B, 1, 148, 128, 10,)}
    if route_index == 8:
        return {"partial_distances": (B, 1, 296, 128, 10,), "partial_indices": (B, 1, 296, 128, 10,)}
    if route_index == 10:
        return {"partial_distances": (B, 1, 256, 64, 64,), "partial_indices": (B, 1, 256, 64, 64,), "group_distances": (B, Q, 4, 64,), "group_indices": (B, Q, 4, 64,)}
    if route_index == 11:
        return {"partial_distances": (B, 1, 304, 64, 64,), "partial_indices": (B, 1, 304, 64, 64,), "group_distances": (B, Q, 4, 64,), "group_indices": (B, Q, 4, 64,)}
    if route_index == 12:
        return {"partial_distances": (B, 1, 256, 64, 64,), "partial_indices": (B, 1, 256, 64, 64,), "group_distances": (B, Q, 4, 64,), "group_indices": (B, Q, 4, 64,)}
    if route_index == 13:
        return {"partial_distances": (B, 1, 256, 64, 64,), "partial_indices": (B, 1, 256, 64, 64,), "group_distances": (B, Q, 4, 64,), "group_indices": (B, Q, 4, 64,), "tail_partial_distances": (B, 2048, 64,), "tail_partial_indices": (B, 2048, 64,), "tail_group_distances": (B, 32, 64,), "tail_group_indices": (B, 32, 64,)}
    if route_index == 14:
        return {"partial_distances": (B, 1, 608, 128, 64,), "partial_indices": (B, 1, 608, 128, 64,), "group_distances": (B, Q, 8, 64,), "group_indices": (B, Q, 8, 64,)}
    return {}
