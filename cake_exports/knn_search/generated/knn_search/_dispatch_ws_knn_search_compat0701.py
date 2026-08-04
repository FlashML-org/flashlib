"""Generated workspace-shape table for the dispatch-family .so loader."""

from __future__ import annotations

WORKSPACE_KEYS = ('partial_distances', 'partial_indices', 'group_distances', 'group_indices', 'temp_distances', 'temp_indices')
WORKSPACE_DTYPES = {'partial_distances': 'float32', 'partial_indices': 'int32', 'group_distances': 'float32', 'group_indices': 'int32', 'temp_distances': 'float32', 'temp_indices': 'int32'}
SCALAR_NAMES = ('B', 'Q', 'M', 'D', 'K', 'self_search')
ROUTE_IDS = ('4bc1_target0627_d4096_q16_m8192_k10_directstride_tcgen05', '974f_target0629_d1024_q64_m32768_k10_directstride_tcgen05', 'c3bc_target0629_d768_q128_m32768_k10_directstride_tcgen05', 'codex0616_q4096_k10_split4_exact_m16384_20000_32768_65536', 'codex0616_q4096_k10_split4_exact_m16384_20000_32768_65536', 'codex0616_q4096_k10_split4_exact_m16384_20000_32768_65536', 'codex0616_q4096_lowk_k3_k7_split4_exact_m20000', 'codex0616_v2_highq_qbucket_exact_q256_q512_q1024_q2048_m65536_k10', 'codex0616_v2_highq_qbucket_exact_q256_q512_q1024_q2048_m65536_k10', 'codex0616_v2_highq_qbucket_exact_q256_q512_q1024_q2048_m65536_k10', 'codex0616_v2_highq_qbucket_exact_q256_q512_q1024_q2048_m65536_k10', 'dispatch0610_r2_f94e_blind_d384_exact_tcgen05', 'f939_target0629_d4096_q4_m8192_k64_native_n64_g8_final8_grid132_tcgen05', 'profiled_target0630_d4096_q4_m32768_k10_qreuse_tcgen05', 'q8_d1024_exact_tcgen05_seed_repair_v1', 'round115_80a5_q256_m65536_k64_split256_twotileproducer_hiermerge16', 'round116_455f_q3072_m49152_split6_full', 'round116_455f_self_q3072_m3072_split6_full', 'round125_f4bc_self_q1024_m1024_d128_k10_split8', 'round131_f3ce_floor13_k64_prefix8', 'round131_f3ce_q384_m98304_k64_prefix8', 'round131_f3ce_q96_m131072_k64_twotile_hiermerge32', 'round132_f3ce_floor13_k80_prefix8', 'round132_f3ce_floor13_k80_prefix8', 'round14_k12_tcgen05_capacity', 'round14_k12_tcgen05_capacity', 'round20_k20_k30_tcgen05_capacity', 'round20_k20_k30_tcgen05_capacity', 'round20_k20_k30_tcgen05_capacity', 'round20c7_22d9_q128_e2eb_guard_miss_kle10', 'round20c7_22d9_q128_e2eb_guard_miss_kle10', 'round20c7_22d9_q128_e2eb_guard_miss_kle10', 'round20c7_22d9_q128_e2eb_guard_miss_kle10', 'round20c7_22d9_q128_e2eb_guard_miss_kle10', 'round20c7_22d9_q128_e2eb_guard_miss_kle10', 'round20c7_22d9_q128_e2eb_guard_miss_kle10', 'round20c7_22d9_q128_e2eb_guard_miss_kle10', 'round20c7_22d9_q128_e2eb_guard_miss_kle10', 'round2_6389_blind_k64_portfolio', 'round2_6389_blind_k64_portfolio', 'round2_6389_blind_k64_portfolio', 'round2_6389_blind_k64_portfolio', 'round2d9eee_q128_k48_split512_truek48_kexact', 'round39_lowq_tcgen05_mma_split', 'round39_lowq_tcgen05_mma_split', 'round39_lowq_tcgen05_mma_split', 'round39_lowq_tcgen05_mma_split', 'round39_lowq_tile_reduce', 'round39_lowq_tile_reduce', 'round39_lowq_tile_reduce', 'round39_q1_tile_reduce', 'round39_q1_tile_reduce', 'round39_q1_tile_reduce', 'round43_highq_midm_qbucket', 'round43_highq_midm_qbucket', 'round43_highq_midm_qbucket', 'round43_highq_midm_qbucket', 'round455f_d128_k10_q513_m98304_full_split32_f828', 'round5_54ff_b2_q128_qbucket_exact_m65536', 'round7_28ec_q128_m131072_k40_via_k64_truncate', 'round7_28ec_q128_m65536_k56_via_k64_truncate', 'round80a5_b2_q128_m65536_k64_twotile_hiermerge16', 'round80a5_d256_q512_m65536_k10_tcgen05', 'round80a5_d384_q256_m32768_k10_tcgen05_split148', 'round80a5_q1_m262144_k10_flashdecode', 'round80a5_self_k5_q2048_m2048_split16_partial', 'round98_7b4c_midq_0e99_blind_split', 'round98_7b4c_midq_0e99_blind_split', 'round98_7b4c_midq_0e99_blind_split', 'round99_blind_lowd_non_d128_padded_tcgen05', 'round99_blind_lowd_non_d128_padded_tcgen05', 'round99_blind_lowd_non_d128_padded_tcgen05', 'round99_blind_lowd_non_d128_padded_tcgen05', 'round_q1024split8_self_k5_q1024_m1024_split8_partial', 'round_selfk5direct0616_q256_q512_direct_k5', 'round_selfk5direct0616_q256_q512_direct_k5', 'rounda7f3_q4096_m65536_k10_split4_fulltile', 'roundb2fb_r8_lowq_q3_m131072_exact_blockm640', 'roundb72b_ann_q10000_m100000_split7', 'roundcc76_lowq_q2q4q7_m131072_blockm640_exact', 'roundcc76_lowq_q2q4q7_m131072_blockm640_exact', 'roundcc76_lowq_q2q4q7_m131072_blockm640_exact', 'target0627_d768_q64_m65536_k64_6472_v2_direct_tcgen05', 'target0628_d1024_q32_m65536_k64_9571_hiermerge16_tcgen05', 'target0628_d128_q4096_m20000_k3_104b_k3partial_split4', 'target0628_d256_q1024_m65536_k10_split64_tcgen05_2165', 'target0628_d64_q256_m131072_k10_split512_hmerge8_d64_tcgen05_885d', 'target0628_d64_q512_m65536_k64_6c60_split256_d64_tcgen05_8group', 'target0628_d768_q16_m65536_k10_4950_directstride_n256_split2_tcgen05', 'target0630_d4096_q1_m65536_k10_restore296_tcgen05', 'round_b08d_q128_midtail_e2eb_split_m', 'round40_q4096_split8', 'round34_application_wrapper', 'round34_application_wrapper', 'round34_application_wrapper', 'round34_application_wrapper', 'round34_application_wrapper', 'round34_application_wrapper', 'round34_application_wrapper', 'round34_application_wrapper')
ROUTE_LAUNCH_COUNTS = (2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 3, 2, 2, 3, 2, 2, 2, 2, 2, 3, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 3, 3, 2, 3, 3, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 4, 4, 3, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 1, 1, 2, 2, 2, 2, 2, 2, 2, 3, 2, 2, 3, 3, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2)
ROUTE_INPUT_TENSORS = (('database', 'out_distances', 'out_indices', 'queries'), ('database', 'out_distances', 'out_indices', 'queries'), ('database', 'out_distances', 'out_indices', 'queries'), ('database', 'out_distances', 'out_indices', 'queries'), ('database', 'out_distances', 'out_indices', 'queries'), ('database', 'out_distances', 'out_indices', 'queries'), ('database', 'out_distances', 'out_indices', 'queries'), ('database', 'out_distances', 'out_indices', 'queries'), ('database', 'out_distances', 'out_indices', 'queries'), ('database', 'out_distances', 'out_indices', 'queries'), ('database', 'out_distances', 'out_indices', 'queries'), ('database', 'out_distances', 'out_indices', 'queries'), ('database', 'out_distances', 'out_indices', 'queries'), ('database', 'out_distances', 'out_indices', 'queries'), ('database', 'out_distances', 'out_indices', 'queries'), ('database', 'out_distances', 'out_indices', 'queries'), ('database', 'out_distances', 'out_indices', 'queries'), ('out_distances', 'out_indices', 'queries'), ('out_distances', 'out_indices', 'queries'), ('database', 'out_distances', 'out_indices', 'queries'), ('database', 'out_distances', 'out_indices', 'queries'), ('database', 'out_distances', 'out_indices', 'queries'), ('database', 'out_distances', 'out_indices', 'queries'), ('database', 'out_distances', 'out_indices', 'queries'), ('database', 'out_distances', 'out_indices', 'queries'), ('database', 'out_distances', 'out_indices', 'queries'), ('database', 'out_distances', 'out_indices', 'queries'), ('database', 'out_distances', 'out_indices', 'queries'), ('database', 'out_distances', 'out_indices', 'queries'), ('database', 'out_distances', 'out_indices', 'queries'), ('database', 'out_distances', 'out_indices', 'queries'), ('database', 'out_distances', 'out_indices', 'queries'), ('database', 'out_distances', 'out_indices', 'queries'), ('database', 'out_distances', 'out_indices', 'queries'), ('database', 'out_distances', 'out_indices', 'queries'), ('database', 'out_distances', 'out_indices', 'queries'), ('database', 'out_distances', 'out_indices', 'queries'), ('database', 'out_distances', 'out_indices', 'queries'), ('database', 'out_distances', 'out_indices', 'queries'), ('database', 'out_distances', 'out_indices', 'queries'), ('database', 'out_distances', 'out_indices', 'queries'), ('database', 'out_distances', 'out_indices', 'queries'), ('database', 'out_distances', 'out_indices', 'queries'), ('database', 'out_distances', 'out_indices', 'queries'), ('database', 'out_distances', 'out_indices', 'queries'), ('database', 'out_distances', 'out_indices', 'queries'), ('database', 'out_distances', 'out_indices', 'queries'), ('database', 'out_distances', 'out_indices', 'queries'), ('database', 'out_distances', 'out_indices', 'queries'), ('database', 'out_distances', 'out_indices', 'queries'), ('database', 'out_distances', 'out_indices', 'queries'), ('database', 'out_distances', 'out_indices', 'queries'), ('database', 'out_distances', 'out_indices', 'queries'), ('database', 'out_distances', 'out_indices', 'queries'), ('database', 'out_distances', 'out_indices', 'queries'), ('database', 'out_distances', 'out_indices', 'queries'), ('database', 'out_distances', 'out_indices', 'queries'), ('database', 'out_distances', 'out_indices', 'queries'), ('database', 'out_distances', 'out_indices', 'queries'), ('database', 'out_distances', 'out_indices', 'queries'), ('database', 'out_distances', 'out_indices', 'queries'), ('database', 'out_distances', 'out_indices', 'queries'), ('database', 'out_distances', 'out_indices', 'queries'), ('database', 'out_distances', 'out_indices', 'queries'), ('database', 'out_distances', 'out_indices', 'queries'), ('out_distances', 'out_indices', 'queries'), ('database', 'out_distances', 'out_indices', 'queries'), ('database', 'out_distances', 'out_indices', 'queries'), ('database', 'out_distances', 'out_indices', 'queries'), ('database', 'out_distances', 'out_indices', 'queries'), ('database', 'out_distances', 'out_indices', 'queries'), ('database', 'out_distances', 'out_indices', 'queries'), ('database', 'out_distances', 'out_indices', 'queries'), ('out_distances', 'out_indices', 'queries'), ('out_distances', 'out_indices', 'queries'), ('out_distances', 'out_indices', 'queries'), ('database', 'out_distances', 'out_indices', 'queries'), ('database', 'out_distances', 'out_indices', 'queries'), ('database', 'out_distances', 'out_indices', 'queries'), ('database', 'out_distances', 'out_indices', 'queries'), ('database', 'out_distances', 'out_indices', 'queries'), ('database', 'out_distances', 'out_indices', 'queries'), ('database', 'out_distances', 'out_indices', 'queries'), ('database', 'out_distances', 'out_indices', 'queries'), ('database', 'out_distances', 'out_indices', 'queries'), ('database', 'out_distances', 'out_indices', 'queries'), ('database', 'out_distances', 'out_indices', 'queries'), ('database', 'out_distances', 'out_indices', 'queries'), ('database', 'out_distances', 'out_indices', 'queries'), ('database', 'out_distances', 'out_indices', 'queries'), ('database', 'out_distances', 'out_indices', 'queries'), ('database', 'out_distances', 'out_indices', 'queries'), ('out_distances', 'out_indices', 'queries'), ('database', 'out_distances', 'out_indices', 'queries'), ('database', 'out_distances', 'out_indices', 'queries'), ('database', 'out_distances', 'out_indices', 'queries'), ('database', 'out_distances', 'out_indices', 'queries'), ('database', 'out_distances', 'out_indices', 'queries'), ('out_distances', 'out_indices', 'queries'), ('database', 'out_distances', 'out_indices', 'queries'))

def workspace_shapes(route_index: int, B, Q, M, D, K, self_search):
    if route_index == 0:
        return {"partial_distances": (1, 1, 64, 128, 10,), "partial_indices": (1, 1, 64, 128, 10,)}
    if route_index == 1:
        return {"partial_distances": (1, 1, 148, 128, 10,), "partial_indices": (1, 1, 148, 128, 10,)}
    if route_index == 2:
        return {"partial_distances": (1, 1, 148, 128, 10,), "partial_indices": (1, 1, 148, 128, 10,)}
    if route_index == 3:
        return {"partial_distances": (1, 32, 4, 128, 10,), "partial_indices": (1, 32, 4, 128, 10,)}
    if route_index == 4:
        return {"partial_distances": (1, 32, 4, 128, 10,), "partial_indices": (1, 32, 4, 128, 10,)}
    if route_index == 5:
        return {"partial_distances": (1, 32, 4, 128, 10,), "partial_indices": (1, 32, 4, 128, 10,)}
    if route_index == 6:
        return {"partial_distances": (1, 32, 4, 128, 10,), "partial_indices": (1, 32, 4, 128, 10,)}
    if route_index == 7:
        return {"partial_distances": (1, 8, 16, 128, 10,), "partial_indices": (1, 8, 16, 128, 10,)}
    if route_index == 8:
        return {"partial_distances": (1, 16, 16, 128, 10,), "partial_indices": (1, 16, 16, 128, 10,)}
    if route_index == 9:
        return {"partial_distances": (1, 2, 72, 128, 10,), "partial_indices": (1, 2, 72, 128, 10,)}
    if route_index == 10:
        return {"partial_distances": (1, 4, 32, 128, 10,), "partial_indices": (1, 4, 32, 128, 10,)}
    if route_index == 11:
        return {"partial_distances": (1, 1, 148, 128, 10,), "partial_indices": (1, 1, 148, 128, 10,)}
    if route_index == 12:
        return {"partial_distances": (1, 1, 132, 64, 64,), "partial_indices": (1, 1, 132, 64, 64,), "group_distances": (1, 4, 8, 64,), "group_indices": (1, 4, 8, 64,)}
    if route_index == 13:
        return {"partial_distances": (1, 1, 128, 128, 10,), "partial_indices": (1, 1, 128, 128, 10,)}
    if route_index == 14:
        return {"partial_distances": (1, 1, 148, 128, 10,), "partial_indices": (1, 1, 148, 128, 10,)}
    if route_index == 15:
        return {"partial_distances": (1, 2, 1024, 128, 64,), "partial_indices": (1, 2, 1024, 128, 64,), "group_distances": (1, 256, 16, 64,), "group_indices": (1, 256, 16, 64,)}
    if route_index == 16:
        return {"partial_distances": (1, 24, 6, 128, 10,), "partial_indices": (1, 24, 6, 128, 10,)}
    if route_index == 17:
        return {"partial_distances": (1, 24, 6, 128, 10,), "partial_indices": (1, 24, 6, 128, 10,)}
    if route_index == 18:
        return {"partial_distances": (1, 8, 8, 128, 10,), "partial_indices": (1, 8, 8, 128, 10,)}
    if route_index == 19:
        return {"partial_distances": (2, 2, 1536, 128, 8,), "partial_indices": (2, 2, 1536, 128, 8,)}
    if route_index == 20:
        return {"partial_distances": (1, 3, 1536, 128, 8,), "partial_indices": (1, 3, 1536, 128, 8,)}
    if route_index == 21:
        return {"partial_distances": (1, 1, 2048, 128, 64,), "partial_indices": (1, 1, 2048, 128, 64,), "group_distances": (1, 96, 32, 64,), "group_indices": (1, 96, 32, 64,)}
    if route_index == 22:
        return {"partial_distances": (1, 1, 1024, 128, 8,), "partial_indices": (1, 1, 1024, 128, 8,)}
    if route_index == 23:
        return {"partial_distances": (1, 32, 512, 128, 8,), "partial_indices": (1, 32, 512, 128, 8,)}
    if route_index == 24:
        return {"partial_distances": (1, 1, 592, 128, 12,), "partial_indices": (1, 1, 592, 128, 12,)}
    if route_index == 25:
        return {"partial_distances": (1, 1, 592, 128, 12,), "partial_indices": (1, 1, 592, 128, 12,)}
    if route_index == 26:
        return {"partial_distances": (1, 1, 148, 128, 20,), "partial_indices": (1, 1, 148, 128, 20,)}
    if route_index == 27:
        return {"partial_distances": (1, 1, 148, 128, 20,), "partial_indices": (1, 1, 148, 128, 20,)}
    if route_index == 28:
        return {"partial_distances": (1, 1, 148, 128, 30,), "partial_indices": (1, 1, 148, 128, 30,)}
    if route_index == 29:
        return {"partial_distances": (1, 1, 128, 128, 10,), "partial_indices": (1, 1, 128, 128, 10,)}
    if route_index == 30:
        return {"partial_distances": (1, 1, 148, 128, 10,), "partial_indices": (1, 1, 148, 128, 10,)}
    if route_index == 31:
        return {"partial_distances": (1, 1, 148, 128, 10,), "partial_indices": (1, 1, 148, 128, 10,)}
    if route_index == 32:
        return {"partial_distances": (1, 1, 64, 128, 10,), "partial_indices": (1, 1, 64, 128, 10,)}
    if route_index == 33:
        return {"partial_distances": (1, 1, 148, 128, 10,), "partial_indices": (1, 1, 148, 128, 10,)}
    if route_index == 34:
        return {"partial_distances": (1, 1, 148, 128, 10,), "partial_indices": (1, 1, 148, 128, 10,)}
    if route_index == 35:
        return {"partial_distances": (1, 1, 148, 128, 10,), "partial_indices": (1, 1, 148, 128, 10,)}
    if route_index == 36:
        return {"partial_distances": (1, 1, 148, 128, 10,), "partial_indices": (1, 1, 148, 128, 10,)}
    if route_index == 37:
        return {"partial_distances": (1, 1, 148, 128, 10,), "partial_indices": (1, 1, 148, 128, 10,)}
    if route_index == 38:
        return {"partial_distances": (1, 1, 4096, 128, 64,), "partial_indices": (1, 1, 4096, 128, 64,), "group_distances": (1, 128, 32, 64,), "group_indices": (1, 128, 32, 64,)}
    if route_index == 39:
        return {"partial_distances": (1, 1, 1024, 128, 64,), "partial_indices": (1, 1, 1024, 128, 64,), "group_distances": (1, 128, 16, 64,), "group_indices": (1, 128, 16, 64,)}
    if route_index == 40:
        return {"partial_distances": (1, 4, 1024, 128, 64,), "partial_indices": (1, 4, 1024, 128, 64,)}
    if route_index == 41:
        return {"partial_distances": (1, 1, 2048, 128, 64,), "partial_indices": (1, 1, 2048, 128, 64,), "group_distances": (1, 64, 32, 64,), "group_indices": (1, 64, 32, 64,)}
    if route_index == 42:
        return {"partial_distances": (1, 1, 2048, 128, 48,), "partial_indices": (1, 1, 2048, 128, 48,), "group_distances": (1, 128, 32, 48,), "group_indices": (1, 128, 32, 48,)}
    if route_index == 43:
        return {"partial_distances": (1, 1, 148, 128, 10,), "partial_indices": (1, 1, 148, 128, 10,)}
    if route_index == 44:
        return {"partial_distances": (1, 1, 148, 128, 10,), "partial_indices": (1, 1, 148, 128, 10,)}
    if route_index == 45:
        return {"partial_distances": (2, 1, 148, 128, 10,), "partial_indices": (2, 1, 148, 128, 10,)}
    if route_index == 46:
        return {"partial_distances": (1, 1, 148, 128, 10,), "partial_indices": (1, 1, 148, 128, 10,)}
    if route_index == 47:
        return {"partial_distances": (2, 3, 256, 10,), "partial_indices": (2, 3, 256, 10,)}
    if route_index == 48:
        return {"partial_distances": (1, 5, 512, 10,), "partial_indices": (1, 5, 512, 10,)}
    if route_index == 49:
        return {"partial_distances": (1, 6, 512, 10,), "partial_indices": (1, 6, 512, 10,)}
    if route_index == 50:
        return {"partial_distances": (2, 512, 10,), "partial_indices": (2, 512, 10,)}
    if route_index == 51:
        return {"partial_distances": (1, 391, 10,), "partial_indices": (1, 391, 10,)}
    if route_index == 52:
        return {"partial_distances": (1, 512, 10,), "partial_indices": (1, 512, 10,)}
    if route_index == 53:
        return {"partial_distances": (2, 3, 32, 128, 10,), "partial_indices": (2, 3, 32, 128, 10,)}
    if route_index == 54:
        return {"partial_distances": (1, 3, 32, 128, 10,), "partial_indices": (1, 3, 32, 128, 10,)}
    if route_index == 55:
        return {"partial_distances": (1, 6, 16, 128, 10,), "partial_indices": (1, 6, 16, 128, 10,)}
    if route_index == 56:
        return {"partial_distances": (1, 12, 16, 128, 10,), "partial_indices": (1, 12, 16, 128, 10,)}
    if route_index == 57:
        return {"partial_distances": (1, 5, 32, 128, 10,), "partial_indices": (1, 5, 32, 128, 10,)}
    if route_index == 58:
        return {"partial_distances": (2, 1, 72, 128, 10,), "partial_indices": (2, 1, 72, 128, 10,)}
    if route_index == 59:
        return {"partial_distances": (1, 1, 2048, 128, 64,), "partial_indices": (1, 1, 2048, 128, 64,), "group_distances": (1, 128, 32, 64,), "group_indices": (1, 128, 32, 64,), "temp_distances": (1, 128, 64,), "temp_indices": (1, 128, 64,)}
    if route_index == 60:
        return {"partial_distances": (1, 1, 1024, 128, 64,), "partial_indices": (1, 1, 1024, 128, 64,), "group_distances": (1, 128, 16, 64,), "group_indices": (1, 128, 16, 64,), "temp_distances": (1, 128, 64,), "temp_indices": (1, 128, 64,)}
    if route_index == 61:
        return {"partial_distances": (2, 1, 1024, 128, 64,), "partial_indices": (2, 1, 1024, 128, 64,), "group_distances": (2, 128, 16, 64,), "group_indices": (2, 128, 16, 64,)}
    if route_index == 62:
        return {"partial_distances": (1, 4, 148, 128, 10,), "partial_indices": (1, 4, 148, 128, 10,)}
    if route_index == 63:
        return {"partial_distances": (1, 2, 148, 128, 10,), "partial_indices": (1, 2, 148, 128, 10,)}
    if route_index == 64:
        return {"partial_distances": (1, 1024, 10,), "partial_indices": (1, 1024, 10,)}
    if route_index == 65:
        return {"partial_distances": (1, 16, 16, 128, 10,), "partial_indices": (1, 16, 16, 128, 10,)}
    if route_index == 66:
        return {"partial_distances": (1, 4, 37, 128, 10,), "partial_indices": (1, 4, 37, 128, 10,)}
    if route_index == 67:
        return {"partial_distances": (1, 2, 74, 128, 10,), "partial_indices": (1, 2, 74, 128, 10,)}
    if route_index == 68:
        return {"partial_distances": (1, 1, 148, 128, 10,), "partial_indices": (1, 1, 148, 128, 10,)}
    if route_index == 69:
        return {"partial_distances": (1, 1, 148, 128, 10,), "partial_indices": (1, 1, 148, 128, 10,)}
    if route_index == 70:
        return {"partial_distances": (1, 1, 148, 128, 10,), "partial_indices": (1, 1, 148, 128, 10,)}
    if route_index == 71:
        return {"partial_distances": (1, 1, 148, 128, 10,), "partial_indices": (1, 1, 148, 128, 10,)}
    if route_index == 72:
        return {"partial_distances": (1, 1, 148, 128, 10,), "partial_indices": (1, 1, 148, 128, 10,)}
    if route_index == 73:
        return {"partial_distances": (1, 8, 8, 128, 10,), "partial_indices": (1, 8, 8, 128, 10,)}
    if route_index == 76:
        return {"partial_distances": (1, 32, 4, 128, 10,), "partial_indices": (1, 32, 4, 128, 10,)}
    if route_index == 77:
        return {"partial_distances": (3, 205, 10,), "partial_indices": (3, 205, 10,)}
    if route_index == 78:
        return {"partial_distances": (1, 79, 7, 128, 10,), "partial_indices": (1, 79, 7, 128, 10,)}
    if route_index == 79:
        return {"partial_distances": (7, 205, 10,), "partial_indices": (7, 205, 10,)}
    if route_index == 80:
        return {"partial_distances": (2, 205, 10,), "partial_indices": (2, 205, 10,)}
    if route_index == 81:
        return {"partial_distances": (4, 205, 10,), "partial_indices": (4, 205, 10,)}
    if route_index == 82:
        return {"partial_distances": (1, 1, 1024, 128, 64,), "partial_indices": (1, 1, 1024, 128, 64,)}
    if route_index == 83:
        return {"partial_distances": (1, 1, 1024, 32, 64,), "partial_indices": (1, 1, 1024, 32, 64,), "group_distances": (1, 32, 16, 64,), "group_indices": (1, 32, 16, 64,)}
    if route_index == 84:
        return {"partial_distances": (1, 32, 4, 128, 3,), "partial_indices": (1, 32, 4, 128, 3,)}
    if route_index == 85:
        return {"partial_distances": (1, 8, 128, 128, 10,), "partial_indices": (1, 8, 128, 128, 10,)}
    if route_index == 86:
        return {"partial_distances": (1, 2, 2048, 128, 10,), "partial_indices": (1, 2, 2048, 128, 10,), "group_distances": (1, 256, 8, 10,), "group_indices": (1, 256, 8, 10,)}
    if route_index == 87:
        return {"partial_distances": (1, 4, 1024, 128, 64,), "partial_indices": (1, 4, 1024, 128, 64,), "group_distances": (1, 512, 8, 64,), "group_indices": (1, 512, 8, 64,)}
    if route_index == 88:
        return {"partial_distances": (1, 1, 148, 128, 10,), "partial_indices": (1, 1, 148, 128, 10,)}
    if route_index == 89:
        return {"partial_distances": (1, 1, 296, 128, 10,), "partial_indices": (1, 1, 296, 128, 10,)}
    if route_index == 90:
        return {"partial_distances": (B, ((Q + 127) // 128), 148, 128, 10,), "partial_indices": (B, ((Q + 127) // 128), 148, 128, 10,)}
    if route_index == 91:
        return {"partial_distances": (B, ((Q + 127) // 128), 8, 128, 10,), "partial_indices": (B, ((Q + 127) // 128), 8, 128, 10,)}
    if route_index == 92:
        return {"partial_distances": (1, 12, 1, 128, 10,), "partial_indices": (1, 12, 1, 128, 10,)}
    if route_index == 93:
        return {"partial_distances": (1, 32, 4, 128, 10,), "partial_indices": (1, 32, 4, 128, 10,)}
    if route_index == 94:
        return {"partial_distances": (1, 1, 148, 128, 30,), "partial_indices": (1, 1, 148, 128, 30,)}
    if route_index == 95:
        return {"partial_distances": (1, 2, 148, 128, 10,), "partial_indices": (1, 2, 148, 128, 10,)}
    if route_index == 96:
        return {"partial_distances": (1, 2, 148, 128, 10,), "partial_indices": (1, 2, 148, 128, 10,)}
    if route_index == 97:
        return {"partial_distances": (1, 3, 148, 128, 10,), "partial_indices": (1, 3, 148, 128, 10,)}
    if route_index == 98:
        return {"partial_distances": (1, 32, 72, 128, 10,), "partial_indices": (1, 32, 72, 128, 10,)}
    if route_index == 99:
        return {"partial_distances": (1, 1, 148, 128, 32,), "partial_indices": (1, 1, 148, 128, 32,)}
    return {}
