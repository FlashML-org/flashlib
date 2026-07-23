"""Generated workspace-shape table for the dispatch-family .so loader."""

from __future__ import annotations

WORKSPACE_KEYS = ('partial_scores', 'partial_indices', 'partial_keys')
WORKSPACE_DTYPES = {'partial_scores': 'float32', 'partial_indices': 'int32', 'partial_keys': 'uint64'}
SCALAR_NAMES = ('B', 'N', 'D', 'K')
ROUTE_IDS = ('d64_direct_single64_1p2gap_9f2a_v1', 'microdim_pad64_d64_direct_v1', 'microdim_pipeline4_08f9_v4', 'microdim_pipeline4_08f9_v4', 'microdim_hybrid_9c0d_v1', 'microdim_hybrid_9c0d_v1', 'd288_parent_splitk_hybrid_20260629_v1', 'd288_parent_splitk_hybrid_20260629_v1', 'd288_parent_splitk_hybrid_20260629_v1', 'd352_exactd_splitk_c95c_v2', 'd352_exactd_splitk_c95c_v2', 'd480_splitk_k1024_eac2_v1', 'd224_tmem_abi_repair_d17c_v4', 'd112_f826_c829_full_bucket_weave_evolve_flash_kmeans_assign_a262_v1', 'gapfused_oob_v1', 'gapfused_oob_v1', 'gapfused_oob_v1', 'gapfused_oob_v1', 'gapfused_oob_v1', 'gapfused_oob_v1', 'gapfused_oob_v1', 'gapfused_oob_v1', 'gapfused_oob_v1', 'gapfused_oob_v1', 'gapfused_oob_v1', 'gapfused_oob_v1', 'gapfused_oob_v1', 'lowdim_e50c_v1', 'd144_d160_d176_pad192_tail_repair_f9b2_v1', 'd144_d160_d176_pad192_tail_repair_f9b2_v1', 'd160_padded_single_repeated_mma_v2', 'd192_single_repeated_mma_v1', 'd192_paired_repeated_mma_v1', 'd256_single_repeated_mma_v1', 'd768_no_padding_splitk_priority_d768_exact_seed_v1', 'd128_splitk_priority_575c_v1', 'flash_kmeans_assign_no_padding_portfolio_r63_g1_d512n512_v1', 'flash_kmeans_assign_no_padding_portfolio_r63_g1_d512n512_v1', 'flash_kmeans_assign_no_padding_portfolio_r63_g1_d512n512_v1', 'flash_kmeans_assign_no_padding_portfolio_r63_g1_d512n512_v1', 'highd_splitk_8de8_v1', 'highd_splitd_single_tile_6fcf_v1', 'd128_even_near_floor_v10_repair', 'small_grid_single_tile_v10', 'paired_large_v15', 'aligned_weave_v10_fallback')
ROUTE_LAUNCH_COUNTS = (1, 1, 1, 1, 1, 1, 1, 2, 1, 1, 2, 2, 1, 1, 1, 1, 1, 1, 1, 2, 1, 2, 1, 2, 1, 2, 1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 2, 1, 1, 1, 1, 1)
ROUTE_INPUT_TENSORS = (('c_sq', 'centroids', 'out', 'x', 'x_sq'), ('c_sq', 'centroids', 'out', 'x', 'x_sq'), ('c_sq', 'centroids', 'out', 'x'), ('c_sq', 'centroids', 'out', 'x', 'x_sq'), ('c_sq', 'centroids', 'out', 'x', 'x_sq'), ('c_sq', 'centroids', 'out', 'x', 'x_sq'), ('c_sq', 'centroids', 'out', 'x', 'x_sq'), ('c_sq', 'centroids', 'out', 'x'), ('c_sq', 'centroids', 'out', 'x', 'x_sq'), ('c_sq', 'centroids', 'out', 'x', 'x_sq'), ('c_sq', 'centroids', 'out', 'x'), ('c_sq', 'centroids', 'out', 'x'), ('c_sq', 'centroids', 'out', 'x', 'x_sq'), ('c_sq', 'centroids', 'out', 'x'), ('c_sq', 'centroids', 'out', 'x', 'x_sq'), ('c_sq', 'centroids', 'out', 'x', 'x_sq'), ('c_sq', 'centroids', 'out', 'x', 'x_sq'), ('c_sq', 'centroids', 'out', 'x', 'x_sq'), ('c_sq', 'centroids', 'out', 'x', 'x_sq'), ('c_sq', 'centroids', 'out', 'x'), ('c_sq', 'centroids', 'out', 'x', 'x_sq'), ('c_sq', 'centroids', 'out', 'x'), ('c_sq', 'centroids', 'out', 'x', 'x_sq'), ('c_sq', 'centroids', 'out', 'x'), ('c_sq', 'centroids', 'out', 'x', 'x_sq'), ('c_sq', 'centroids', 'out', 'x'), ('c_sq', 'centroids', 'out', 'x', 'x_sq'), ('c_sq', 'centroids', 'out', 'x', 'x_sq'), ('c_sq', 'centroids', 'out', 'x', 'x_sq'), ('c_sq', 'centroids', 'out', 'x', 'x_sq'), ('c_sq', 'centroids', 'out', 'x', 'x_sq'), ('c_sq', 'centroids', 'out', 'x', 'x_sq'), ('c_sq', 'centroids', 'out', 'x', 'x_sq'), ('c_sq', 'centroids', 'out', 'x', 'x_sq'), ('c_sq', 'centroids', 'out', 'x'), ('c_sq', 'centroids', 'out', 'x'), ('c_sq', 'centroids', 'out', 'x'), ('c_sq', 'centroids', 'out', 'x'), ('c_sq', 'centroids', 'out', 'x'), ('c_sq', 'centroids', 'out', 'x'), ('c_sq', 'centroids', 'out', 'x'), ('c_sq', 'centroids', 'out', 'x', 'x_sq'), ('c_sq', 'centroids', 'out', 'x', 'x_sq'), ('c_sq', 'centroids', 'out', 'x', 'x_sq'), ('c_sq', 'centroids', 'out', 'x', 'x_sq'), ('c_sq', 'centroids', 'out', 'x', 'x_sq'))

def workspace_shapes(route_index: int, B, N, D, K):
    if route_index == 7:
        return {"partial_scores": (((B * (N // 128)) * (K // 256)), 128,), "partial_indices": (((B * (N // 128)) * (K // 256)), 128,)}
    if route_index == 10:
        return {"partial_scores": (((B * (N // 64)) * (K // 256)), 64,), "partial_indices": (((B * (N // 64)) * (K // 256)), 64,)}
    if route_index == 11:
        return {"partial_scores": (((B * (N // 64)) * (K // 256)), 64,), "partial_indices": (((B * (N // 64)) * (K // 256)), 64,)}
    if route_index == 19:
        return {"partial_scores": (((B * (N // 128)) * (K // 256)), 128,), "partial_indices": (((B * (N // 128)) * (K // 256)), 128,)}
    if route_index == 21:
        return {"partial_scores": (((B * (N // 128)) * (K // 256)), 128,), "partial_indices": (((B * (N // 128)) * (K // 256)), 128,)}
    if route_index == 23:
        return {"partial_scores": (((B * (N // 128)) * (K // 256)), 128,), "partial_indices": (((B * (N // 128)) * (K // 256)), 128,)}
    if route_index == 25:
        return {"partial_scores": (((B * (N // 128)) * (K // 256)), 128,), "partial_indices": (((B * (N // 128)) * (K // 256)), 128,)}
    if route_index == 34:
        return {"partial_scores": (((B * (N // 64)) * (K // 256)), 64,), "partial_indices": (((B * (N // 64)) * (K // 256)), 64,)}
    if route_index == 35:
        return {"partial_scores": (((B * (N // 64)) * (K // 256)), 64,), "partial_indices": (((B * (N // 64)) * (K // 256)), 64,)}
    if route_index == 36:
        return {"partial_scores": (((B * (N // 64)) * ((K // 256) // 2)), 64,), "partial_indices": (((B * (N // 64)) * ((K // 256) // 2)), 64,)}
    if route_index == 37:
        return {"partial_scores": (((B * (N // 128)) * (K // 256)), 128,), "partial_indices": (((B * (N // 128)) * (K // 256)), 128,)}
    if route_index == 38:
        return {"partial_keys": (((B * (N // 64)) * ((K // 256) // 2)), 64,)}
    if route_index == 39:
        return {"partial_keys": (((B * (N // 64)) * ((K // 256) // 2)), 64,)}
    if route_index == 40:
        return {"partial_scores": (((B * (N // 128)) * (K // 256)), 128,), "partial_indices": (((B * (N // 128)) * (K // 256)), 128,)}
    return {}
