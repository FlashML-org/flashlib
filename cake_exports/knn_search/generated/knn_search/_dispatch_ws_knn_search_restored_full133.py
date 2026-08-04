"""Generated workspace-shape table for the dispatch-family .so loader."""

from __future__ import annotations

WORKSPACE_KEYS = ('partial_distances', 'partial_indices', 'padded_queries', 'padded_database')
WORKSPACE_DTYPES = {'partial_distances': 'float32', 'partial_indices': 'int32', 'padded_queries': 'bfloat16', 'padded_database': 'bfloat16'}
SCALAR_NAMES = ('B', 'Q', 'M', 'D', 'K', 'self_search')
ROUTE_IDS = ('round127_598a_k1_top1_pairq_split9_merge16', 'round156_7e60_dynamic_lowd_d1_tile_reduce', 'round156_1b18_dynamic_lowd_d1_tail_tile_reduce', 'round157_7e60_dynamic_lowd_d3d5_tile_reduce', 'round157_7e60_dynamic_lowd_d3d5_tile_reduce', 'round157_199f_self_d3_single_tile', 'ffc4_dynamic_d7_q128_m65536_no_pack_tcgen05', 'c8b9_tiny_dynamic_d_no_pack_guarded_tcgen05', '9286_ext_dynamic_d_highd_generated_variants', '9286_ext_dynamic_d_highd_generated_variants', '9286_ext_dynamic_d_highd_generated_variants', '9286_ext_dynamic_d_highd_generated_variants', '9286_ext_dynamic_d_highd_generated_variants', '9286_ext_dynamic_d_highd_generated_variants', '9286_ext_dynamic_d_highd_generated_variants', 'ffc4_dynamic_d129d257d511_q128_m65536_directstride_tcgen05', 'ffc4_dynamic_d129d257d511_q128_m65536_directstride_tcgen05', 'ffc4_dynamic_d129d257d511_q128_m65536_directstride_tcgen05', '5847_dynamic_d384_q32_m131072_exact_tcgen05', '9286_d512_q64_row16_directstride_tcgen05', '9286_dynamic_d768d1024_q32q16_directstride_tcgen05', '9286_dynamic_d768d1024_q32q16_directstride_tcgen05', '6bea_r118_ivf_q12_m100_d64_k20_direct', 'round117_9d5c_d3_k32_self_tile', 'round149_3c6e_q4096_m32768_k32_prefix8_emitted_tie', 'round149_3c6e_q4096_m32768_k64_prefix8_emitted_tie', 'round147_e36b_q4096_m32768_k48_prefix8', 'round147_3363_q128_m131072_k64_prefix8', 'round113_c2e0_q4096_m49152_k64_prefix8', 'round20_245d_q4096_m20000_k64_prefix6cert', '4b95_d130_q64_m65536_k64_direct_d256pad_tcgen05', '177e_r121_d512_q32_k64_distonlymerge_tcgen05', 'ffc4_dynamic_b2_q64_m65536_d129_tcgen05', 'ccef_dynamic_d257_q64_m65536_k64_tcgen05_v2', 'round116_455f_q127_m131071_split148')
ROUTE_LAUNCH_COUNTS = (2, 2, 2, 2, 2, 1, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 1, 1, 2, 2, 2, 2, 2, 2, 2, 2, 3, 3, 2)
ROUTE_INPUT_TENSORS = (('database', 'out_distances', 'out_indices', 'queries'), ('database', 'out_distances', 'out_indices', 'queries'), ('database', 'out_distances', 'out_indices', 'queries'), ('database', 'out_distances', 'out_indices', 'queries'), ('database', 'out_distances', 'out_indices', 'queries'), ('database', 'out_distances', 'out_indices', 'queries'), ('database', 'out_distances', 'out_indices', 'queries'), ('database', 'out_distances', 'out_indices', 'queries'), ('database', 'out_distances', 'out_indices', 'queries'), ('database', 'out_distances', 'out_indices', 'queries'), ('database', 'out_distances', 'out_indices', 'queries'), ('database', 'out_distances', 'out_indices', 'queries'), ('database', 'out_distances', 'out_indices', 'queries'), ('database', 'out_distances', 'out_indices', 'queries'), ('database', 'out_distances', 'out_indices', 'queries'), ('database', 'out_distances', 'out_indices', 'queries'), ('database', 'out_distances', 'out_indices', 'queries'), ('database', 'out_distances', 'out_indices', 'queries'), ('database', 'out_distances', 'out_indices', 'queries'), ('database', 'out_distances', 'out_indices', 'queries'), ('database', 'out_distances', 'out_indices', 'queries'), ('database', 'out_distances', 'out_indices', 'queries'), ('database', 'out_distances', 'out_indices', 'queries'), ('database', 'out_distances', 'out_indices', 'queries'), ('database', 'out_distances', 'out_indices', 'queries'), ('database', 'out_distances', 'out_indices', 'queries'), ('database', 'out_distances', 'out_indices', 'queries'), ('database', 'out_distances', 'out_indices', 'queries'), ('database', 'out_distances', 'out_indices', 'queries'), ('database', 'out_distances', 'out_indices', 'queries'), ('database', 'out_distances', 'out_indices', 'queries'), ('database', 'out_distances', 'out_indices', 'queries'), ('database', 'out_distances', 'out_indices', 'queries'), ('database', 'out_distances', 'out_indices', 'queries'), ('database', 'out_distances', 'out_indices', 'queries'))

def workspace_shapes(route_index: int, B, Q, M, D, K, self_search):
    if route_index == 0:
        return {"partial_distances": (B, 32, 9, 128, 1,), "partial_indices": (B, 32, 9, 128, 1,)}
    if route_index == 1:
        return {"partial_distances": (B, Q, 16, 10,), "partial_indices": (B, Q, 16, 10,)}
    if route_index == 2:
        return {"partial_distances": (B, Q, ((M + 4095) // 4096), 10,), "partial_indices": (B, Q, ((M + 4095) // 4096), 10,)}
    if route_index == 3:
        return {"partial_distances": (B, Q, 16, 10,), "partial_indices": (B, Q, 16, 10,)}
    if route_index == 4:
        return {"partial_distances": (B, Q, 16, 10,), "partial_indices": (B, Q, 16, 10,)}
    if route_index == 6:
        return {"partial_distances": (B, 1, 148, 128, 10,), "partial_indices": (B, 1, 148, 128, 10,)}
    if route_index == 7:
        return {"partial_distances": (B, 1, 148, 128, 10,), "partial_indices": (B, 1, 148, 128, 10,)}
    if route_index == 8:
        return {"partial_distances": (B, 1, 148, 128, 10,), "partial_indices": (B, 1, 148, 128, 10,)}
    if route_index == 9:
        return {"partial_distances": (B, 1, 148, 128, 10,), "partial_indices": (B, 1, 148, 128, 10,)}
    if route_index == 10:
        return {"partial_distances": (B, 1, 148, 128, 10,), "partial_indices": (B, 1, 148, 128, 10,)}
    if route_index == 11:
        return {"partial_distances": (B, 1, 148, 128, 10,), "partial_indices": (B, 1, 148, 128, 10,)}
    if route_index == 12:
        return {"partial_distances": (B, 1, 148, 128, 10,), "partial_indices": (B, 1, 148, 128, 10,)}
    if route_index == 13:
        return {"partial_distances": (B, 1, 148, 128, 10,), "partial_indices": (B, 1, 148, 128, 10,)}
    if route_index == 14:
        return {"partial_distances": (B, 1, 148, 128, 10,), "partial_indices": (B, 1, 148, 128, 10,)}
    if route_index == 15:
        return {"partial_distances": (B, 1, 148, 128, 10,), "partial_indices": (B, 1, 148, 128, 10,)}
    if route_index == 16:
        return {"partial_distances": (B, 1, 148, 128, 10,), "partial_indices": (B, 1, 148, 128, 10,)}
    if route_index == 17:
        return {"partial_distances": (B, 1, 148, 128, 10,), "partial_indices": (B, 1, 148, 128, 10,)}
    if route_index == 18:
        return {"partial_distances": (B, 1, 148, 128, 10,), "partial_indices": (B, 1, 148, 128, 10,)}
    if route_index == 19:
        return {"partial_distances": (B, 1, 148, 128, 10,), "partial_indices": (B, 1, 148, 128, 10,)}
    if route_index == 20:
        return {"partial_distances": (B, 1, 148, 128, 10,), "partial_indices": (B, 1, 148, 128, 10,)}
    if route_index == 21:
        return {"partial_distances": (B, 1, 148, 128, 10,), "partial_indices": (B, 1, 148, 128, 10,)}
    if route_index == 24:
        return {"partial_distances": (B, 32, 512, 128, 8,), "partial_indices": (B, 32, 512, 128, 8,)}
    if route_index == 25:
        return {"partial_distances": (B, 32, 512, 128, 8,), "partial_indices": (B, 32, 512, 128, 8,)}
    if route_index == 26:
        return {"partial_distances": (B, 32, 512, 128, 8,), "partial_indices": (B, 32, 512, 128, 8,)}
    if route_index == 27:
        return {"partial_distances": (B, 1, 2048, 128, 8,), "partial_indices": (B, 1, 2048, 128, 8,)}
    if route_index == 28:
        return {"partial_distances": (B, 32, 768, 128, 8,), "partial_indices": (B, 32, 768, 128, 8,)}
    if route_index == 29:
        return {"partial_distances": (B, 32, 316, 128, 64,), "partial_indices": (B, 32, 316, 128, 64,)}
    if route_index == 30:
        return {"partial_distances": (B, 1, 296, 128, 64,), "partial_indices": (B, 1, 296, 128, 64,)}
    if route_index == 31:
        return {"partial_distances": (B, 1, 512, 32, 64,), "partial_indices": (B, 1, 512, 32, 64,)}
    if route_index == 32:
        return {"padded_queries": (B, Q, 144,), "padded_database": (B, M, 144,), "partial_distances": (B, 1, 148, 128, 10,), "partial_indices": (B, 1, 148, 128, 10,)}
    if route_index == 33:
        return {"padded_queries": (B, Q, 512,), "padded_database": (B, M, 512,), "partial_distances": (B, 1, 1024, 128, 64,), "partial_indices": (B, 1, 1024, 128, 64,)}
    if route_index == 34:
        return {"partial_distances": (B, 1, 148, 128, 10,), "partial_indices": (B, 1, 148, 128, 10,)}
    return {}
