"""Generated workspace-shape table for the dispatch-family .so loader."""

from __future__ import annotations

WORKSPACE_KEYS = ('partial_distances', 'partial_indices', 'group_distances', 'group_indices')
WORKSPACE_DTYPES = {'partial_distances': 'float32', 'partial_indices': 'int32', 'group_distances': 'float32', 'group_indices': 'int32'}
SCALAR_NAMES = ('B', 'Q', 'M', 'D', 'K', 'self_search')
ROUTE_IDS = ('0268_high_d_low_q_d768_q64_n256_tcgen05', '0268_high_d_low_q_d640_q32_directstride_tcgen05', '0268_high_d_low_q_d2048_q8_directstride_tcgen05', '0268_high_d_low_q_d4096_q4_directstride_tcgen05', '0268_high_d_low_q_d4096_q8_directstride_tcgen05', 'cf73_legacy_dyn_d160_alignedvec_directstride_tcgen05', 'cf73_legacy_dyn_d33_scalar_directstride_tcgen05', 'cf73_legacy_dyn_d80_alignedvec_directstride_tcgen05', 'dbbe_irregular_b2_q96_m196608_split_wave_fullm_tcgen05', 'dbbe_irregular_q768_m98304_split24_fullm_tcgen05', 'dbbe_irregular_q1025_m65537_split16_masked_tail_tcgen05', 'f561_v2_d1024_q32_k64_hiermerge8_tcgen05', 'target_d64_q128_m131072_k64_split512_d64_tcgen05', 'residual_full198_d256_k64_fused_hier8x64_ownership_e92c_tcgen05')
ROUTE_LAUNCH_COUNTS = (2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 3, 3, 2)
ROUTE_INPUT_TENSORS = (('database', 'out_distances', 'out_indices', 'queries'), ('database', 'out_distances', 'out_indices', 'queries'), ('database', 'out_distances', 'out_indices', 'queries'), ('database', 'out_distances', 'out_indices', 'queries'), ('database', 'out_distances', 'out_indices', 'queries'), ('database', 'out_distances', 'out_indices', 'queries'), ('database', 'out_distances', 'out_indices', 'queries'), ('database', 'out_distances', 'out_indices', 'queries'), ('database', 'out_distances', 'out_indices', 'queries'), ('database', 'out_distances', 'out_indices', 'queries'), ('database', 'out_distances', 'out_indices', 'queries'), ('database', 'out_distances', 'out_indices', 'queries'), ('database', 'out_distances', 'out_indices', 'queries'), ('database', 'out_distances', 'out_indices', 'queries'))

def workspace_shapes(route_index: int, B, Q, M, D, K, self_search):
    if route_index == 0:
        return {"partial_distances": (B, 1, 148, 128, 10,), "partial_indices": (B, 1, 148, 128, 10,)}
    if route_index == 1:
        return {"partial_distances": (B, 1, 148, 128, 10,), "partial_indices": (B, 1, 148, 128, 10,)}
    if route_index == 2:
        return {"partial_distances": (B, 1, 128, 128, 10,), "partial_indices": (B, 1, 128, 128, 10,)}
    if route_index == 3:
        return {"partial_distances": (B, 1, 64, 128, 10,), "partial_indices": (B, 1, 64, 128, 10,)}
    if route_index == 4:
        return {"partial_distances": (B, 1, 128, 128, 10,), "partial_indices": (B, 1, 128, 128, 10,)}
    if route_index == 5:
        return {"partial_distances": (B, 1, 148, 128, 10,), "partial_indices": (B, 1, 148, 128, 10,)}
    if route_index == 6:
        return {"partial_distances": (B, 1, 148, 128, 10,), "partial_indices": (B, 1, 148, 128, 10,)}
    if route_index == 7:
        return {"partial_distances": (B, 2, 148, 128, 10,), "partial_indices": (B, 2, 148, 128, 10,)}
    if route_index == 8:
        return {"partial_distances": (B, 1, 72, 128, 10,), "partial_indices": (B, 1, 72, 128, 10,)}
    if route_index == 9:
        return {"partial_distances": (B, 6, 24, 128, 10,), "partial_indices": (B, 6, 24, 128, 10,)}
    if route_index == 10:
        return {"partial_distances": (B, 9, 16, 128, 10,), "partial_indices": (B, 9, 16, 128, 10,)}
    if route_index == 11:
        return {"partial_distances": (B, 1, 512, 32, 64,), "partial_indices": (B, 1, 512, 32, 64,), "group_distances": (B, Q, 8, 64,), "group_indices": (B, Q, 8, 64,)}
    if route_index == 12:
        return {"partial_distances": (B, 1, 2048, 128, 64,), "partial_indices": (B, 1, 2048, 128, 64,), "group_distances": (B, Q, 32, 64,), "group_indices": (B, Q, 32, 64,)}
    if route_index == 13:
        return {"partial_distances": (B, 1, 512, 128, 64,), "partial_indices": (B, 1, 512, 128, 64,)}
    return {}
