from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from functools import lru_cache
from typing import Any

from ._dispatch import knn_search_registry_b653_compat0701_v1 as _root
from ._dispatch_runtime import _import_dispatch_module, dispatch_launch_options

_WEAVE_PREFIX = 'loom.examples.weave.'
_ROOT_MODULE = 'knn_search_registry_b653_compat0701_v1'
_ROOT_CALLABLE = 'launch_for_eval'
_EXACT_LAUNCH_SPECS = {(1, 1, 65536, 4096, 10, 'bfloat16', False, False): ('target0630_d4096_q1_m65536_k10_restore296_tcgen05', 'knn_search_target0630_d4096_q1_m65536_k10_restore296_92e6_v1', 'launch_for_eval'), (1, 1, 100000, 128, 10, 'bfloat16', False, False): ('round39_q1_tile_reduce', 'knn_search_lowq_q8q16q32q64_row16_registered_0613_r55_48e9_v1', 'launch_for_eval'), (1, 1, 131072, 128, 10, 'bfloat16', False, False): ('round39_q1_tile_reduce', 'knn_search_lowq_q8q16q32q64_row16_registered_0613_r55_48e9_v1', 'launch_for_eval'), (1, 1, 262144, 128, 10, 'bfloat16', False, False): ('round80a5_q1_m262144_k10_flashdecode', 'knn_search_80a5_blocker_q1_m262144_54ff_v1', 'launch_for_eval'), (1, 2, 131072, 128, 10, 'bfloat16', False, False): ('roundcc76_lowq_q2q4q7_m131072_blockm640_exact', 'knn_search_lowq_q247_m131072_exact_0617_cc76_v1', 'launch_for_eval'), (1, 3, 131072, 128, 10, 'bfloat16', False, False): ('roundb2fb_r8_lowq_q3_m131072_exact_blockm640', 'knn_search_lowq_q3_m131072_exact_0617_b2fb_r8_v1', 'launch_for_eval'), (1, 4, 8192, 4096, 10, 'bfloat16', False, False): ('0268_high_d_low_q_d4096_q4_directstride_tcgen05', 'knn_search_registry_b653_compat0701_v1', 'launch_for_eval'), (1, 4, 8192, 4096, 64, 'bfloat16', False, False): ('f939_target0629_d4096_q4_m8192_k64_native_n64_g8_final8_grid132_tcgen05', 'knn_search_target0629_d4096_q4_m8192_k64_f939_n64_g8_final8_grid132_v1', 'launch_for_eval'), (1, 4, 32768, 4096, 10, 'bfloat16', False, False): ('profiled_target0630_d4096_q4_m32768_k10_qreuse_tcgen05', 'knn_search_target0630_d4096_q4_m32768_k10_profiled_weave_evolve_v1', 'launch_for_eval'), (1, 4, 131072, 128, 10, 'bfloat16', False, False): ('roundcc76_lowq_q2q4q7_m131072_blockm640_exact', 'knn_search_lowq_q247_m131072_exact_0617_cc76_v1', 'launch_for_eval'), (1, 5, 131072, 128, 10, 'bfloat16', False, False): ('round39_lowq_tile_reduce', 'knn_search_lowq_q8q16q32q64_row16_registered_0613_r55_48e9_v1', 'launch_for_eval'), (1, 6, 262144, 128, 10, 'bfloat16', False, False): ('round39_lowq_tile_reduce', 'knn_search_lowq_q8q16q32q64_row16_registered_0613_r55_48e9_v1', 'launch_for_eval'), (1, 7, 131072, 128, 10, 'bfloat16', False, False): ('roundcc76_lowq_q2q4q7_m131072_blockm640_exact', 'knn_search_lowq_q247_m131072_exact_0617_cc76_v1', 'launch_for_eval'), (1, 8, 10, 32, 10, 'bfloat16', False, False): ('afe6_dynamic_d_scalar_capacity', 'knn_search_scalar_capacity_0611_r22_4e96_v1', 'launch_scalar_capacity_for_eval'), (1, 8, 20, 48, 10, 'bfloat16', False, False): ('afe6_dynamic_d_scalar_capacity', 'knn_search_scalar_capacity_0611_r22_4e96_v1', 'launch_scalar_capacity_for_eval'), (1, 8, 16384, 2048, 10, 'bfloat16', False, False): ('0268_high_d_low_q_d2048_q8_directstride_tcgen05', 'knn_search_registry_b653_compat0701_v1', 'launch_for_eval'), (1, 8, 16384, 4096, 10, 'bfloat16', False, False): ('0268_high_d_low_q_d4096_q8_directstride_tcgen05', 'knn_search_registry_b653_compat0701_v1', 'launch_for_eval'), (1, 8, 65536, 1024, 10, 'bfloat16', False, False): ('q8_d1024_exact_tcgen05_seed_repair_v1', 'knn_search_q8_d1024_repair_q8_d1024_seed_v1', 'launch_for_eval'), (1, 8, 131072, 128, 10, 'bfloat16', False, False): ('round55_lowq_q8q16q32q64_row16_registered', 'knn_search_lowq_q8q16q32q64_row16_registered_0613_r55_48e9_v1', 'launch_for_eval'), (1, 9, 196608, 128, 10, 'bfloat16', False, False): ('round39_lowq_tcgen05_mma_split', 'knn_search_lowq_q8q16q32q64_row16_registered_0613_r55_48e9_v1', 'launch_for_eval'), (1, 12, 100, 64, 20, 'bfloat16', False, False): ('6bea_r118_ivf_q12_m100_d64_k20_direct', 'knn_search_dynamic_d512_ivf_6bea_r118_v1', 'launch_for_eval'), (1, 16, 8192, 4096, 10, 'bfloat16', False, False): ('4bc1_target0627_d4096_q16_m8192_k10_directstride_tcgen05', 'knn_search_target0627_d4096_q16_m8192_k10_4bc1_v1', 'launch_for_eval'), (1, 16, 32768, 1024, 10, 'bfloat16', False, False): ('9286_dynamic_d768d1024_q32q16_directstride_tcgen05', 'knn_search_dynamic_d768d1024_q32q16_tcgen05_0618_9286_v1', 'launch_for_eval'), (1, 16, 65536, 768, 10, 'bfloat16', False, False): ('target0628_d768_q16_m65536_k10_4950_directstride_n256_split2_tcgen05', 'knn_search_target0627_d768_q16_m65536_k10_4950_split2_v1', 'launch_for_eval'), (1, 16, 131072, 128, 10, 'bfloat16', False, False): ('round55_lowq_q8q16q32q64_row16_registered', 'knn_search_lowq_q8q16q32q64_row16_registered_0613_r55_48e9_v1', 'launch_for_eval'), (1, 17, 196608, 128, 10, 'bfloat16', False, False): ('round39_lowq_tcgen05_mma_split', 'knn_search_lowq_q8q16q32q64_row16_registered_0613_r55_48e9_v1', 'launch_for_eval'), (1, 24, 196608, 128, 10, 'bfloat16', False, False): ('round39_lowq_tcgen05_mma_split', 'knn_search_lowq_q8q16q32q64_row16_registered_0613_r55_48e9_v1', 'launch_for_eval'), (1, 32, 32768, 512, 64, 'bfloat16', False, False): ('177e_r121_d512_q32_k64_distonlymerge_tcgen05', 'knn_search_dynamic_d512_k64_q32_6bea_r121_177e_distonlymerge_v1', 'launch_for_eval'), (1, 32, 32768, 640, 10, 'bfloat16', False, False): ('0268_high_d_low_q_d640_q32_directstride_tcgen05', 'knn_search_registry_b653_compat0701_v1', 'launch_for_eval'), (1, 32, 32768, 768, 10, 'bfloat16', False, False): ('9286_dynamic_d768d1024_q32q16_directstride_tcgen05', 'knn_search_dynamic_d768d1024_q32q16_tcgen05_0618_9286_v1', 'launch_for_eval'), (1, 32, 32768, 1024, 64, 'bfloat16', False, False): ('f561_v2_d1024_q32_k64_hiermerge8_tcgen05', 'knn_search_registry_b653_compat0701_v1', 'launch_for_eval'), (1, 32, 65536, 1024, 64, 'bfloat16', False, False): ('target0628_d1024_q32_m65536_k64_9571_hiermerge16_tcgen05', 'knn_search_target0628_d1024_q32_m65536_k64_9571_hiermerge16_v2', 'launch_for_eval'), (1, 32, 131072, 128, 10, 'bfloat16', False, False): ('round55_lowq_q8q16q32q64_row16_registered', 'knn_search_lowq_q8q16q32q64_row16_registered_0613_r55_48e9_v1', 'launch_for_eval'), (1, 32, 131072, 384, 10, 'bfloat16', False, False): ('5847_dynamic_d384_q32_m131072_exact_tcgen05', 'knn_search_dynamic_d_residual_0621_r123_ffc4_v1', 'launch_for_eval'), (1, 63, 262144, 256, 64, 'bfloat16', False, False): ('residual0705_q63_masked_query_structural_reset_f392_tcgen05', 'knn_search_residual0705_q63_masked_query_structural_reset_f392_v1', 'launch_for_eval'), (1, 64, 32768, 1024, 10, 'bfloat16', False, False): ('974f_target0629_d1024_q64_m32768_k10_directstride_tcgen05', 'knn_search_target0629_d1024_q64_m32768_k10_974f_v1', 'launch_for_eval'), (1, 64, 65536, 1, 10, 'bfloat16', False, False): ('round156_7e60_dynamic_lowd_d1_tile_reduce', 'knn_search_dynamic_lowd_d1_bucket_tile_reduce_0625_7e60_v1', 'launch_for_eval'), (1, 64, 65536, 130, 64, 'bfloat16', False, False): ('4b95_d130_q64_m65536_k64_direct_d256pad_tcgen05', 'knn_search_dynamic_d130_k64_q64_m65536_directpad_0625_4b95_v1', 'launch_for_eval'), (1, 64, 65536, 257, 64, 'bfloat16', False, False): ('ccef_dynamic_d257_q64_m65536_k64_tcgen05_v2', 'knn_search_dynamic_d_residual_0621_r123_ffc4_v1', 'launch_for_eval'), (1, 64, 65536, 512, 10, 'bfloat16', False, False): ('9286_d512_q64_row16_directstride_tcgen05', 'knn_search_dynamic_d512_q64_tcgen05_0618_9286_v1', 'launch_for_eval'), (1, 64, 65536, 768, 10, 'bfloat16', False, False): ('0268_high_d_low_q_d768_q64_n256_tcgen05', 'knn_search_registry_b653_compat0701_v1', 'launch_for_eval'), (1, 64, 65536, 768, 64, 'bfloat16', False, False): ('target0627_d768_q64_m65536_k64_6472_v2_direct_tcgen05', 'knn_search_target0627_d768_q64_m65536_k64_6472_v2', 'launch_for_eval'), (1, 64, 131072, 128, 10, 'bfloat16', False, False): ('round55_lowq_q8q16q32q64_row16_registered', 'knn_search_lowq_q8q16q32q64_row16_registered_0613_r55_48e9_v1', 'launch_for_eval'), (1, 64, 131072, 128, 64, 'bfloat16', False, False): ('round2_6389_blind_k64_portfolio', 'knn_search_blind_k64_m262144_portfolio_0614_r19_6389_v1', 'launch_for_eval'), (1, 64, 131072, 160, 10, 'bfloat16', False, False): ('cf73_legacy_dyn_d160_alignedvec_directstride_tcgen05', 'knn_search_registry_b653_compat0701_v1', 'launch_for_eval'), (1, 64, 262143, 256, 64, 'bfloat16', False, False): ('residual0705_q64_tail_split152_full_wave_3dee_tcgen05', 'knn_search_residual0705_q64_tail_split152_full_wave_3dee_v1', 'launch_for_eval'), (1, 64, 262144, 256, 64, 'bfloat16', False, False): ('residual0707_d256_q64_split152_warp_state_7ad3_tcgen05', 'knn_search_registry_b653_compat0701_v1', 'launch_for_eval'), (1, 64, 262145, 256, 64, 'bfloat16', False, False): ('residual0705_q64_tail_plus_812c_sentinel_warp_state_361b_tcgen05', 'knn_search_residual0705_q64_tail_plus_812c_sentinel_warp_state_361b_v2', 'launch_for_eval'), (1, 65, 262144, 256, 64, 'bfloat16', False, False): ('residual0705_q65_distinct_topology_structural_reset_d436_tcgen05', 'knn_search_residual0705_q65_distinct_topology_structural_reset_d436_v1', 'launch_for_eval'), (1, 96, 98304, 128, 10, 'bfloat16', False, False): ('round98_7b4c_midq_0e99_blind_split', 'knn_search_dispatch0610_midq0e99_0615_r98_7b4c_v1', 'launch_for_eval'), (1, 96, 131072, 128, 64, 'bfloat16', False, False): ('round131_f3ce_q96_m131072_k64_twotile_hiermerge32', 'knn_search_dispatch0622_current_portfolio_e472_v1', 'launch_for_eval'), (1, 127, 65536, 1, 10, 'bfloat16', False, False): ('round156_7e60_dynamic_lowd_d1_tile_reduce', 'knn_search_dynamic_lowd_d1_bucket_tile_reduce_0625_7e60_v1', 'launch_for_eval'), (1, 127, 131071, 128, 10, 'bfloat16', False, False): ('round116_455f_q127_m131071_split148', 'knn_search_d128_k10_subfloor_455f_r116_v1', 'launch_for_eval'), (1, 128, 8192, 128, 10, 'bfloat16', False, False): ('round20c7_22d9_q128_e2eb_guard_miss_kle10', 'knn_search_dispatch0610_highq_midq_q128_qbucket_split4_codex0616_v4', 'launch_q128_e2eb_for_eval'), (1, 128, 16384, 128, 10, 'bfloat16', False, False): ('round20c7_22d9_q128_e2eb_guard_miss_kle10', 'knn_search_dispatch0610_highq_midq_q128_qbucket_split4_codex0616_v4', 'launch_q128_e2eb_for_eval'), (1, 128, 32768, 128, 10, 'bfloat16', False, False): ('round20c7_22d9_q128_e2eb_guard_miss_kle10', 'knn_search_dispatch0610_highq_midq_q128_qbucket_split4_codex0616_v4', 'launch_q128_e2eb_for_eval'), (1, 128, 32768, 768, 10, 'bfloat16', False, False): ('c3bc_target0629_d768_q128_m32768_k10_directstride_tcgen05', 'knn_search_target0629_d768_q128_m32768_k10_c3bc_v1', 'launch_for_eval'), (1, 128, 65535, 1, 10, 'bfloat16', False, False): ('round156_1b18_dynamic_lowd_d1_tail_tile_reduce', 'knn_search_dynamic_lowd_d1_tail_tile_reduce_0625_1b18_v1', 'launch_for_eval'), (1, 128, 65535, 128, 10, 'bfloat16', False, False): ('round_b08d_q128_midtail_e2eb_split_m', 'knn_search_q128_midtail_current_dispatch0610_b08d_v1', 'launch_for_eval'), (1, 128, 65536, 1, 10, 'bfloat16', False, False): ('round156_7e60_dynamic_lowd_d1_tile_reduce', 'knn_search_dynamic_lowd_d1_bucket_tile_reduce_0625_7e60_v1', 'launch_for_eval'), (1, 128, 65536, 1, 10, 'bfloat16', False, True): ('round156_7e60_dynamic_lowd_d1_tile_reduce_forced_fallback', 'knn_search_dispatch0624_full133_eacf_lowd_cd65_v1', 'launch_for_eval'), (1, 128, 65536, 3, 10, 'bfloat16', False, False): ('round157_7e60_dynamic_lowd_d3d5_tile_reduce', 'knn_search_dynamic_lowd_d3d5_bucket_tile_reduce_0625_7e60_v1', 'launch_for_eval'), (1, 128, 65536, 5, 10, 'bfloat16', False, False): ('round157_7e60_dynamic_lowd_d3d5_tile_reduce', 'knn_search_dynamic_lowd_d3d5_bucket_tile_reduce_0625_7e60_v1', 'launch_for_eval'), (1, 128, 65536, 7, 10, 'bfloat16', False, False): ('ffc4_dynamic_d7_q128_m65536_no_pack_tcgen05', 'knn_search_dynamic_d_residual_0621_r123_ffc4_v1', 'launch_for_eval'), (1, 128, 65536, 15, 10, 'bfloat16', False, False): ('9286_ext_dynamic_d_highd_generated_variants', 'knn_search_dispatch0618_c492_ext_dynamic_highd_9286_v1', 'launch_for_eval'), (1, 128, 65536, 31, 10, 'bfloat16', False, False): ('9286_ext_dynamic_d_highd_generated_variants', 'knn_search_dispatch0618_c492_ext_dynamic_highd_9286_v1', 'launch_for_eval'), (1, 128, 65536, 63, 10, 'bfloat16', False, False): ('c8b9_tiny_dynamic_d_no_pack_guarded_tcgen05', 'knn_search_dynamic_d_priority_4832_v1', 'launch_for_eval'), (1, 128, 65536, 64, 10, 'bfloat16', False, False): ('round99_blind_lowd_non_d128_padded_tcgen05', 'knn_search_blind_lowd_non_d128_tcgen05_dispatch0610_r99_ec7c_v1', 'launch_for_eval'), (1, 128, 65536, 65, 10, 'bfloat16', False, False): ('9286_ext_dynamic_d_highd_generated_variants', 'knn_search_dispatch0618_c492_ext_dynamic_highd_9286_v1', 'launch_for_eval'), (1, 128, 65536, 96, 10, 'bfloat16', False, False): ('round99_blind_lowd_non_d128_padded_tcgen05', 'knn_search_blind_lowd_non_d128_tcgen05_dispatch0610_r99_ec7c_v1', 'launch_for_eval'), (1, 128, 65536, 127, 10, 'bfloat16', False, False): ('9286_ext_dynamic_d_highd_generated_variants', 'knn_search_dispatch0618_c492_ext_dynamic_highd_9286_v1', 'launch_for_eval'), (1, 128, 65536, 128, 10, 'bfloat16', False, False): ('round_b08d_q128_midtail_e2eb_split_m', 'knn_search_q128_midtail_current_dispatch0610_b08d_v1', 'launch_for_eval'), (1, 128, 65536, 128, 56, 'bfloat16', False, False): ('round7_28ec_q128_m65536_k56_via_k64_truncate', 'knn_search_ext_k_capacity_0618_28ec_v1', 'launch_for_eval'), (1, 128, 65536, 128, 64, 'bfloat16', False, False): ('round2_6389_blind_k64_portfolio', 'knn_search_blind_k64_m262144_portfolio_0614_r19_6389_v1', 'launch_for_eval'), (1, 128, 65536, 128, 80, 'bfloat16', False, False): ('round132_f3ce_floor13_k80_prefix8', 'knn_search_dispatch0622_current_portfolio_e472_v1', 'launch_for_eval'), (1, 128, 65536, 129, 10, 'bfloat16', False, False): ('ffc4_dynamic_d129d257d511_q128_m65536_directstride_tcgen05', 'knn_search_dynamic_d_residual_0621_r123_ffc4_v1', 'launch_for_eval'), (1, 128, 65536, 130, 10, 'bfloat16', False, False): ('9286_ext_dynamic_d_highd_generated_variants', 'knn_search_dispatch0618_c492_ext_dynamic_highd_9286_v1', 'launch_for_eval'), (1, 128, 65536, 192, 10, 'bfloat16', False, False): ('round99_blind_lowd_non_d128_padded_tcgen05', 'knn_search_blind_lowd_non_d128_tcgen05_dispatch0610_r99_ec7c_v1', 'launch_for_eval'), (1, 128, 65536, 255, 10, 'bfloat16', False, False): ('9286_ext_dynamic_d_highd_generated_variants', 'knn_search_dispatch0618_c492_ext_dynamic_highd_9286_v1', 'launch_for_eval'), (1, 128, 65536, 257, 10, 'bfloat16', False, False): ('ffc4_dynamic_d129d257d511_q128_m65536_directstride_tcgen05', 'knn_search_dynamic_d_residual_0621_r123_ffc4_v1', 'launch_for_eval'), (1, 128, 65536, 258, 10, 'bfloat16', False, False): ('9286_ext_dynamic_d_highd_generated_variants', 'knn_search_dispatch0618_c492_ext_dynamic_highd_9286_v1', 'launch_for_eval'), (1, 128, 65536, 320, 10, 'bfloat16', False, False): ('round99_blind_lowd_non_d128_padded_tcgen05', 'knn_search_blind_lowd_non_d128_tcgen05_dispatch0610_r99_ec7c_v1', 'launch_for_eval'), (1, 128, 65536, 384, 10, 'bfloat16', False, False): ('dispatch0610_r2_f94e_blind_d384_exact_tcgen05', 'knn_search_blind_d384_tcgen05_dispatch0610_r2_f94e_v1', 'launch_for_eval'), (1, 128, 65536, 511, 10, 'bfloat16', False, False): ('ffc4_dynamic_d129d257d511_q128_m65536_directstride_tcgen05', 'knn_search_dynamic_d_residual_0621_r123_ffc4_v1', 'launch_for_eval'), (1, 128, 65537, 1, 10, 'bfloat16', False, False): ('round156_1b18_dynamic_lowd_d1_tail_tile_reduce', 'knn_search_dynamic_lowd_d1_tail_tile_reduce_0625_1b18_v1', 'launch_for_eval'), (1, 128, 65537, 128, 10, 'bfloat16', False, False): ('round_b08d_q128_midtail_e2eb_split_m', 'knn_search_q128_midtail_current_dispatch0610_b08d_v1', 'launch_for_eval'), (1, 128, 98304, 128, 10, 'bfloat16', False, False): ('round_b08d_q128_midtail_e2eb_split_m', 'knn_search_q128_midtail_current_dispatch0610_b08d_v1', 'launch_for_eval'), (1, 128, 131072, 33, 10, 'bfloat16', False, False): ('cf73_legacy_dyn_d33_scalar_directstride_tcgen05', 'knn_search_registry_b653_compat0701_v1', 'launch_for_eval'), (1, 128, 131072, 64, 64, 'bfloat16', False, False): ('target_d64_q128_m131072_k64_split512_d64_tcgen05', 'knn_search_registry_b653_compat0701_v1', 'launch_for_eval'), (1, 128, 131072, 128, 1, 'bfloat16', False, False): ('round20c7_22d9_q128_e2eb_guard_miss_kle10', 'knn_search_dispatch0610_highq_midq_q128_qbucket_split4_codex0616_v4', 'launch_q128_e2eb_for_eval'), (1, 128, 131072, 128, 2, 'bfloat16', False, False): ('round20c7_22d9_q128_e2eb_guard_miss_kle10', 'knn_search_dispatch0610_highq_midq_q128_qbucket_split4_codex0616_v4', 'launch_q128_e2eb_for_eval'), (1, 128, 131072, 128, 5, 'bfloat16', False, False): ('round20c7_22d9_q128_e2eb_guard_miss_kle10', 'knn_search_dispatch0610_highq_midq_q128_qbucket_split4_codex0616_v4', 'launch_q128_e2eb_for_eval'), (1, 128, 131072, 128, 8, 'bfloat16', False, False): ('round20c7_22d9_q128_e2eb_guard_miss_kle10', 'knn_search_dispatch0610_highq_midq_q128_qbucket_split4_codex0616_v4', 'launch_q128_e2eb_for_eval'), (1, 128, 131072, 128, 10, 'bfloat16', False, False): ('round20c7_22d9_q128_e2eb_guard_miss_kle10', 'knn_search_dispatch0610_highq_midq_q128_qbucket_split4_codex0616_v4', 'launch_q128_e2eb_for_eval'), (1, 128, 131072, 128, 11, 'bfloat16', False, False): ('round14_k12_tcgen05_capacity', 'knn_search_q128_extendedk_dispatch_0613_r39_48e9_v1', 'launch_for_eval'), (1, 128, 131072, 128, 12, 'bfloat16', False, False): ('round14_k12_tcgen05_capacity', 'knn_search_q128_extendedk_dispatch_0613_r39_48e9_v1', 'launch_for_eval'), (1, 128, 131072, 128, 16, 'bfloat16', False, False): ('round20_k20_k30_tcgen05_capacity', 'knn_search_q128_extendedk_dispatch_0613_r39_48e9_v1', 'launch_for_eval'), (1, 128, 131072, 128, 20, 'bfloat16', False, False): ('round20_k20_k30_tcgen05_capacity', 'knn_search_q128_extendedk_dispatch_0613_r39_48e9_v1', 'launch_for_eval'), (1, 128, 131072, 128, 24, 'bfloat16', False, False): ('round34_application_wrapper', 'knn_search_lowq_q8q16q32q64_row16_registered_0613_r55_48e9_v1', 'launch_for_eval'), (1, 128, 131072, 128, 30, 'bfloat16', False, False): ('round20_k20_k30_tcgen05_capacity', 'knn_search_q128_extendedk_dispatch_0613_r39_48e9_v1', 'launch_for_eval'), (1, 128, 131072, 128, 32, 'bfloat16', False, False): ('round34_application_wrapper', 'knn_search_lowq_q8q16q32q64_row16_registered_0613_r55_48e9_v1', 'launch_for_eval'), (1, 128, 131072, 128, 40, 'bfloat16', False, False): ('round7_28ec_q128_m131072_k40_via_k64_truncate', 'knn_search_ext_k_capacity_0618_28ec_v1', 'launch_for_eval'), (1, 128, 131072, 128, 48, 'bfloat16', False, False): ('round2d9eee_q128_k48_split512_truek48_kexact', 'knn_search_k48_q128split512_truek48_kexact_0615_2d9eee_v1', 'launch_for_eval'), (1, 128, 131072, 128, 64, 'bfloat16', False, False): ('round147_3363_q128_m131072_k64_prefix8', 'knn_search_q128_m131072_k64_prefix8_0624_3363_v1', 'launch_for_eval'), (1, 128, 131072, 256, 10, 'bfloat16', False, False): ('round36_d256_k10_k64_tcgen05', 'knn_search_extendedk_lowd_d256_dispatch0610_r3_extk_v1', 'launch_for_eval'), (1, 128, 131072, 256, 64, 'bfloat16', False, False): ('residual_full198_d256_k64_fused_hier8x64_ownership_e92c_tcgen05', 'knn_search_registry_b653_compat0701_v1', 'launch_for_eval'), (1, 128, 262144, 128, 10, 'bfloat16', False, False): ('round20c7_22d9_q128_e2eb_guard_miss_kle10', 'knn_search_dispatch0610_highq_midq_q128_qbucket_split4_codex0616_v4', 'launch_q128_e2eb_for_eval'), (1, 128, 262144, 128, 64, 'bfloat16', False, False): ('round2_6389_blind_k64_portfolio', 'knn_search_blind_k64_m262144_portfolio_0614_r19_6389_v1', 'launch_for_eval'), (1, 128, 262144, 256, 64, 'bfloat16', False, False): ('residual_q128_stability_split304_fullwaves_a8f5_tcgen05', 'knn_search_residual_convergence_q128_stability_then_kernel_round301_a8f5_v1', 'launch_for_eval'), (1, 129, 65536, 1, 10, 'bfloat16', False, False): ('round156_7e60_dynamic_lowd_d1_tile_reduce', 'knn_search_dynamic_lowd_d1_bucket_tile_reduce_0625_7e60_v1', 'launch_for_eval'), (1, 129, 131073, 128, 10, 'bfloat16', False, False): ('round34_application_wrapper', 'knn_search_lowq_q8q16q32q64_row16_registered_0613_r55_48e9_v1', 'launch_for_eval'), (1, 192, 98304, 128, 10, 'bfloat16', False, False): ('round98_7b4c_midq_0e99_blind_split', 'knn_search_dispatch0610_midq0e99_0615_r98_7b4c_v1', 'launch_for_eval'), (1, 255, 262143, 128, 10, 'bfloat16', False, False): ('round34_application_wrapper', 'knn_search_lowq_q8q16q32q64_row16_registered_0613_r55_48e9_v1', 'launch_for_eval'), (1, 256, 256, 128, 5, 'bfloat16', True, False): ('round_selfk5direct0616_q256_q512_direct_k5', 'knn_search_self_k5_q1024_split8_0616_q1024split8_v1', 'launch_for_eval'), (1, 256, 32768, 384, 10, 'bfloat16', False, False): ('round80a5_d384_q256_m32768_k10_tcgen05_split148', 'knn_search_80a5_blocker_d384_q256_tcgen05_v1', 'launch_for_eval'), (1, 256, 65536, 1, 10, 'bfloat16', False, False): ('round156_7e60_dynamic_lowd_d1_tile_reduce', 'knn_search_dynamic_lowd_d1_bucket_tile_reduce_0625_7e60_v1', 'launch_for_eval'), (1, 256, 65536, 80, 10, 'bfloat16', False, False): ('cf73_legacy_dyn_d80_alignedvec_directstride_tcgen05', 'knn_search_registry_b653_compat0701_v1', 'launch_for_eval'), (1, 256, 65536, 128, 10, 'bfloat16', False, False): ('codex0616_v2_highq_qbucket_exact_q256_q512_q1024_q2048_m65536_k10', 'knn_search_highq_midm_qbucket_0611_r19_4e96_v1', 'launch_for_eval'), (1, 256, 65536, 128, 64, 'bfloat16', False, False): ('round115_80a5_q256_m65536_k64_split256_twotileproducer_hiermerge16', 'knn_search_80a5_blocker_k64_q256_m65536_twotileproducer_v1', 'launch_for_eval'), (1, 256, 131072, 64, 10, 'bfloat16', False, False): ('target0628_d64_q256_m131072_k10_split512_hmerge8_d64_tcgen05_885d', 'knn_search_target0628_d64_q256_m131072_k10_885d_hmerge8_v1', 'launch_for_eval'), (1, 257, 262145, 128, 10, 'bfloat16', False, False): ('round34_application_wrapper', 'knn_search_lowq_q8q16q32q64_row16_registered_0613_r55_48e9_v1', 'launch_for_eval'), (1, 384, 49152, 128, 10, 'bfloat16', False, False): ('round43_highq_midm_qbucket', 'knn_search_lowq_q8q16q32q64_row16_registered_0613_r55_48e9_v1', 'launch_for_eval'), (1, 384, 98304, 128, 64, 'bfloat16', False, False): ('round131_f3ce_q384_m98304_k64_prefix8', 'knn_search_dispatch0622_current_portfolio_e472_v1', 'launch_for_eval'), (1, 512, 512, 128, 5, 'bfloat16', True, False): ('round_selfk5direct0616_q256_q512_direct_k5', 'knn_search_self_k5_q1024_split8_0616_q1024split8_v1', 'launch_for_eval'), (1, 512, 65536, 64, 64, 'bfloat16', False, False): ('target0628_d64_q512_m65536_k64_6c60_split256_d64_tcgen05_8group', 'knn_search_target0628_d64_q512_m65536_k64_dbaf_v1', 'launch_for_eval'), (1, 512, 65536, 128, 10, 'bfloat16', False, False): ('codex0616_v2_highq_qbucket_exact_q256_q512_q1024_q2048_m65536_k10', 'knn_search_highq_midm_qbucket_0611_r19_4e96_v1', 'launch_for_eval'), (1, 512, 65536, 128, 64, 'bfloat16', False, False): ('round2_6389_blind_k64_portfolio', 'knn_search_blind_k64_m262144_portfolio_0614_r19_6389_v1', 'launch_for_eval'), (1, 512, 65536, 256, 10, 'bfloat16', False, False): ('round80a5_d256_q512_m65536_k10_tcgen05', 'knn_search_80a5_blocker_d256_q512_m65536_v1', 'launch_for_eval'), (1, 512, 98304, 128, 10, 'bfloat16', False, False): ('round98_7b4c_midq_0e99_blind_split', 'knn_search_dispatch0610_midq0e99_0615_r98_7b4c_v1', 'launch_for_eval'), (1, 513, 98304, 128, 10, 'bfloat16', False, False): ('round455f_d128_k10_q513_m98304_full_split32_f828', 'knn_search_dispatch0618_d128_k10_c08b_f828_synth_9d5c_v1', 'launch_for_eval'), (1, 768, 49152, 128, 10, 'bfloat16', False, False): ('round43_highq_midm_qbucket', 'knn_search_lowq_q8q16q32q64_row16_registered_0613_r55_48e9_v1', 'launch_for_eval'), (1, 768, 98304, 128, 10, 'bfloat16', False, False): ('dbbe_irregular_q768_m98304_split24_fullm_tcgen05', 'knn_search_registry_b653_compat0701_v1', 'launch_for_eval'), (1, 1024, 1024, 128, 5, 'bfloat16', True, False): ('round_q1024split8_self_k5_q1024_m1024_split8_partial', 'knn_search_self_k5_q1024_split8_0616_q1024split8_v1', 'launch_for_eval'), (1, 1024, 1024, 128, 10, 'bfloat16', True, False): ('round125_f4bc_self_q1024_m1024_d128_k10_split8', 'knn_search_self_k10_q1024_split8_0622_f4bc_v1', 'launch_for_eval'), (1, 1024, 65536, 128, 10, 'bfloat16', False, False): ('codex0616_v2_highq_qbucket_exact_q256_q512_q1024_q2048_m65536_k10', 'knn_search_highq_midm_qbucket_0611_r19_4e96_v1', 'launch_for_eval'), (1, 1024, 65536, 256, 10, 'bfloat16', False, False): ('target0628_d256_q1024_m65536_k10_split64_tcgen05_2165', 'knn_search_target0628_d256_q1024_m65536_k10_2165_v1', 'launch_for_eval'), (1, 1025, 65537, 128, 10, 'bfloat16', False, False): ('dbbe_irregular_q1025_m65537_split16_masked_tail_tcgen05', 'knn_search_registry_b653_compat0701_v1', 'launch_for_eval'), (1, 1500, 1500, 2, 32, 'bfloat16', True, False): ('round56_d2_dbscan_t128_m1536_cd72', 'knn_search_extendedk_lowd_d256_dispatch0610_r3_extk_v1', 'launch_for_eval'), (1, 1500, 1500, 2, 64, 'bfloat16', True, False): ('round56_d2_dbscan_t128_m1536_cd72', 'knn_search_extendedk_lowd_d256_dispatch0610_r3_extk_v1', 'launch_for_eval'), (1, 1536, 1536, 128, 10, 'bfloat16', True, False): ('round34_application_wrapper', 'knn_search_lowq_q8q16q32q64_row16_registered_0613_r55_48e9_v1', 'launch_for_eval'), (1, 1536, 49152, 128, 10, 'bfloat16', False, False): ('round43_highq_midm_qbucket', 'knn_search_lowq_q8q16q32q64_row16_registered_0613_r55_48e9_v1', 'launch_for_eval'), (1, 2048, 2048, 3, 10, 'bfloat16', True, False): ('round157_199f_self_d3_single_tile', 'knn_search_dynamic_self_d3_single_tile_0625_199f_v1', 'launch_for_eval'), (1, 2048, 2048, 128, 5, 'bfloat16', True, False): ('round80a5_self_k5_q2048_m2048_split16_partial', 'knn_search_self_k5_q2048_split16_0617_80a5_v1', 'launch_for_eval'), (1, 2048, 65536, 128, 10, 'bfloat16', False, False): ('codex0616_v2_highq_qbucket_exact_q256_q512_q1024_q2048_m65536_k10', 'knn_search_highq_midm_qbucket_0611_r19_4e96_v1', 'launch_for_eval'), (1, 3072, 3072, 128, 10, 'bfloat16', True, False): ('round116_455f_self_q3072_m3072_split6_full', 'knn_search_d128_k10_subfloor_455f_r116_v1', 'launch_for_eval'), (1, 3072, 49152, 128, 10, 'bfloat16', False, False): ('round116_455f_q3072_m49152_split6_full', 'knn_search_d128_k10_subfloor_455f_r116_v1', 'launch_for_eval'), (1, 4096, 4096, 3, 32, 'bfloat16', True, False): ('round117_9d5c_d3_k32_self_tile', 'knn_search_dynamic_d3_k32_self_tile_0620_9d5c_v1', 'launch_for_eval'), (1, 4096, 4096, 128, 10, 'bfloat16', True, False): ('round34_application_wrapper', 'knn_search_lowq_q8q16q32q64_row16_registered_0613_r55_48e9_v1', 'launch_for_eval'), (1, 4096, 16384, 128, 8, 'bfloat16', False, False): ('round54_q4096_m16384_lowk_k8_stride10_out8_split4', 'knn_search_lowq_q8q16q32q64_row16_registered_0613_r55_48e9_v1', 'launch_for_eval'), (1, 4096, 16384, 128, 10, 'bfloat16', False, False): ('codex0616_q4096_k10_split4_exact_m16384_20000_32768_65536', 'knn_search_dispatch0610_highq_qbucket_split4_codex0616_v2', 'launch_for_eval'), (1, 4096, 20000, 128, 1, 'bfloat16', False, False): ('round127_598a_k1_top1_pairq_split9_merge16', 'knn_search_k1_top1_pairq_0622_598a_v1', 'launch_for_eval'), (1, 4096, 20000, 128, 2, 'bfloat16', False, False): ('round46_q4096_lowk_k2partial_split9', 'knn_search_lowq_q8q16q32q64_row16_registered_0613_r55_48e9_v1', 'launch_for_eval'), (1, 4096, 20000, 128, 3, 'bfloat16', False, False): ('target0628_d128_q4096_m20000_k3_104b_k3partial_split4', 'knn_search_target0628_d128_q4096_m20000_k3_e8f1_k3partial_v1', 'launch_for_eval'), (1, 4096, 20000, 128, 5, 'bfloat16', False, False): ('round51_registered_q4096_lowk_k5partial_split9', 'knn_search_lowq_q8q16q32q64_row16_registered_0613_r55_48e9_v1', 'launch_for_eval'), (1, 4096, 20000, 128, 7, 'bfloat16', False, False): ('codex0616_q4096_lowk_k3_k7_split4_exact_m20000', 'knn_search_q4096_split4_tiestable_0612_r15_4e2c_v1', 'launch_for_eval'), (1, 4096, 20000, 128, 8, 'bfloat16', False, False): ('round47_registered_q4096_lowk_k8_stride10_out8_split4', 'knn_search_lowq_q8q16q32q64_row16_registered_0613_r55_48e9_v1', 'launch_for_eval'), (1, 4096, 20000, 128, 10, 'bfloat16', False, False): ('codex0616_q4096_k10_split4_exact_m16384_20000_32768_65536', 'knn_search_dispatch0610_highq_qbucket_split4_codex0616_v2', 'launch_for_eval'), (1, 4096, 20000, 128, 64, 'bfloat16', False, False): ('round20_245d_q4096_m20000_k64_prefix6cert', 'knn_search_k64_q4096split79_localprefix6_certfallback_0615_245d_v1', 'launch_for_eval'), (1, 4096, 32768, 128, 8, 'bfloat16', False, False): ('round54_q4096_m32768_lowk_k8_stride10_out8_split4', 'knn_search_lowq_q8q16q32q64_row16_registered_0613_r55_48e9_v1', 'launch_for_eval'), (1, 4096, 32768, 128, 10, 'bfloat16', False, False): ('codex0616_q4096_k10_split4_exact_m16384_20000_32768_65536', 'knn_search_dispatch0610_highq_qbucket_split4_codex0616_v2', 'launch_for_eval'), (1, 4096, 32768, 128, 32, 'bfloat16', False, False): ('round149_3c6e_q4096_m32768_k32_prefix8_emitted_tie', 'knn_search_dispatch0624_3c6e_k32k64_prefix_tie_v1', 'launch_for_eval'), (1, 4096, 32768, 128, 48, 'bfloat16', False, False): ('round147_e36b_q4096_m32768_k48_prefix8', 'knn_search_k48_q4096_m32768_prefix8_0623_e36b_v1', 'launch_for_eval'), (1, 4096, 32768, 128, 64, 'bfloat16', False, False): ('round149_3c6e_q4096_m32768_k64_prefix8_emitted_tie', 'knn_search_dispatch0624_3c6e_k32k64_prefix_tie_v1', 'launch_for_eval'), (1, 4096, 32768, 128, 80, 'bfloat16', False, False): ('round132_f3ce_floor13_k80_prefix8', 'knn_search_dispatch0622_current_portfolio_e472_v1', 'launch_for_eval'), (1, 4096, 49152, 128, 5, 'bfloat16', False, False): ('round34_application_wrapper', 'knn_search_lowq_q8q16q32q64_row16_registered_0613_r55_48e9_v1', 'launch_for_eval'), (1, 4096, 49152, 128, 64, 'bfloat16', False, False): ('round113_c2e0_q4096_m49152_k64_prefix8', 'knn_search_ext_k64_highq_prefix8_0618_c2e0_v1', 'launch_for_eval'), (1, 4096, 65536, 128, 10, 'bfloat16', False, False): ('rounda7f3_q4096_m65536_k10_split4_fulltile', 'knn_search_highq_q4096_m65536_k10_fulltile_0615_a7f3_v1', 'launch_for_eval'), (1, 10000, 100000, 128, 10, 'bfloat16', False, False): ('roundb72b_ann_q10000_m100000_split7', 'knn_search_7ce1_plus_ann_split7_dispatch_0615_b72b_v1', 'launch_for_eval'), (2, 1, 131072, 128, 10, 'bfloat16', False, False): ('round39_q1_tile_reduce', 'knn_search_lowq_q8q16q32q64_row16_registered_0613_r55_48e9_v1', 'launch_for_eval'), (2, 3, 131072, 128, 10, 'bfloat16', False, False): ('round39_lowq_tile_reduce', 'knn_search_lowq_q8q16q32q64_row16_registered_0613_r55_48e9_v1', 'launch_for_eval'), (2, 64, 65536, 129, 10, 'bfloat16', False, False): ('ffc4_dynamic_b2_q64_m65536_d129_tcgen05', 'knn_search_dynamic_d_residual_0621_r123_ffc4_v1', 'launch_for_eval'), (2, 64, 262144, 128, 10, 'bfloat16', False, False): ('round39_lowq_tcgen05_mma_split', 'knn_search_lowq_q8q16q32q64_row16_registered_0613_r55_48e9_v1', 'launch_for_eval'), (2, 96, 196608, 128, 10, 'bfloat16', False, False): ('dbbe_irregular_b2_q96_m196608_split_wave_fullm_tcgen05', 'knn_search_registry_b653_compat0701_v1', 'launch_for_eval'), (2, 128, 65536, 128, 10, 'bfloat16', False, False): ('round5_54ff_b2_q128_qbucket_exact_m65536', 'knn_search_b2_q128_blind_dispatch0616_54ff_v1', 'launch_for_eval'), (2, 128, 65536, 128, 64, 'bfloat16', False, False): ('round80a5_b2_q128_m65536_k64_twotile_hiermerge16', 'knn_search_80a5_blocker_k64_b2_twotile_74f4_v1', 'launch_for_eval'), (2, 256, 98304, 128, 64, 'bfloat16', False, False): ('round131_f3ce_floor13_k64_prefix8', 'knn_search_floor13_k64_prefix8_0622_f3ce_v1', 'launch_for_eval'), (2, 257, 65536, 128, 10, 'bfloat16', False, False): ('round43_highq_midm_qbucket', 'knn_search_lowq_q8q16q32q64_row16_registered_0613_r55_48e9_v1', 'launch_for_eval'), (2, 4096, 20000, 128, 10, 'bfloat16', False, False): ('round40_q4096_split8', 'knn_search_lowq_q8q16q32q64_row16_registered_0613_r55_48e9_v1', 'launch_for_eval')}


@dataclass(frozen=True)
class RouteDecision:
    """Resolved semantic route with a launcher that can be reused directly."""

    route_id: str
    launch_entrypoint: str
    launcher: Callable[[dict[str, Any]], Any] = field(repr=False, compare=False)
    exact_contract: bool = False

    def launch(
        self,
        inputs: dict[str, Any],
        *,
        stream: Any = None,
        timeout_ms: float | None = None,
    ) -> Any:
        with dispatch_launch_options(stream=stream, timeout_ms=timeout_ms):
            return self.launcher(inputs)


def _route_key(inputs: dict[str, Any]) -> tuple[int, int, int, int, int, str, bool, bool]:
    dtype = str(inputs.get("dtype", "bfloat16"))
    if dtype.startswith("torch."):
        dtype = dtype[6:]
    return (
        *(int(inputs[name]) for name in ("B", "Q", "M", "D", "K")),
        dtype,
        bool(inputs.get("self_search", False)),
        bool(inputs.get("force_fallback", False)),
    )


@lru_cache(maxsize=None)
def _load_launcher(module_name: str, callable_name: str) -> Callable[[dict[str, Any]], Any]:
    module = _import_dispatch_module(module_name)
    launcher = getattr(module, callable_name, None)
    if not callable(launcher):
        raise RuntimeError(f"resolved dispatcher launcher is not callable: {module_name}:{callable_name}")
    return launcher


@lru_cache(maxsize=None)
def _make_decision(
    route_id: str,
    module_name: str,
    callable_name: str,
    exact_contract: bool,
) -> RouteDecision:
    return RouteDecision(
        route_id=route_id,
        launch_entrypoint=f"{_WEAVE_PREFIX}{module_name}:{callable_name}",
        launcher=_load_launcher(module_name, callable_name),
        exact_contract=exact_contract,
    )


def _entrypoint_spec(entrypoint: object) -> tuple[str, str] | None:
    if not isinstance(entrypoint, str):
        return None
    module_name, separator, callable_name = entrypoint.partition(":")
    if not separator or not module_name.startswith(_WEAVE_PREFIX) or not callable_name.isidentifier():
        return None
    return module_name.removeprefix(_WEAVE_PREFIX), callable_name


def _generic_decision(inputs: dict[str, Any]) -> RouteDecision:
    info_fn = getattr(_root, "route_info", None)
    info = dict(info_fn(inputs)) if callable(info_fn) else {}
    route_id = info.get("selected_route", info.get("route"))
    if route_id is None:
        select_route = getattr(_root, "selected_route", None)
        route_id = select_route(inputs) if callable(select_route) else _ROOT_CALLABLE
    # ``resolved_launch_entrypoint`` is an explicit launch contract and may
    # bypass the root. ``selected_entrypoint`` is seed provenance: it can name
    # a narrower module whose own guards would silently miss for this
    # signature (the K11 prefix route reports its K64 seed there, and
    # launching the seed with K=11 inputs falls through to a slower exact
    # parent). Signatures without a launch contract go through the root
    # dispatcher, which reproduces the frozen guard cascade exactly.
    spec = _entrypoint_spec(info.get("resolved_launch_entrypoint"))
    if spec is None:
        spec = (_ROOT_MODULE, _ROOT_CALLABLE)
    return _make_decision(str(route_id), *spec, False)


def resolve_route(inputs: dict[str, Any]) -> RouteDecision:
    """Resolve once; exact exported shapes never re-enter the root dispatcher."""

    spec = _EXACT_LAUNCH_SPECS.get(_route_key(inputs))
    if spec is None:
        return _generic_decision(inputs)
    return _make_decision(*spec, True)


class LaunchPlan:
    """Per-signature resolved execution state for the exported hot path.

    Migration step 2 of ``PLAN_BASED_EXPORT_RUNTIME.md``: the guard cascade
    (``resolve_route`` plus one captured dispatcher traversal) runs exactly
    once, at construction. A hot call overwrites the recorded 8-byte pointer
    carriers in place, refreshes device tensor-map descriptors only when a
    bound pointer actually changed, and submits the already-marshalled
    launches on their construction-time stream — no re-marshalling, no
    per-launch stream query, no dispatcher re-entry.
    """

    __slots__ = (
        "route",
        "sequence",
        "torch_stream",
        "stream_handle",
        "_launches",
        "_pointer_writers",
        "_tma_bindings",
    )

    def __init__(self, route: RouteDecision, sequence: Any, *, torch_stream: Any, stream_handle: int):
        launches = tuple(sequence._launches)
        input_bindings = tuple(sequence._input_bindings)
        if input_bindings and len(input_bindings) != len(launches):
            raise RuntimeError("launch plan capture has corrupt input bindings")
        writers: dict[str, list[Any]] = {}
        for launch, bindings in zip(launches, input_bindings):
            carriers = launch._packed._prevent_gc
            for index, key in bindings:
                writers.setdefault(key, []).append(carriers[index])
        self.route = route
        self.sequence = sequence
        self.torch_stream = torch_stream
        self.stream_handle = int(stream_handle)
        self._launches = launches
        self._pointer_writers = tuple((key, tuple(items)) for key, items in writers.items())
        self._tma_bindings = tuple(sequence._tensor_map_bindings)

    @property
    def launch_count(self) -> int:
        return len(self._launches)

    @property
    def bound_input_keys(self) -> tuple[str, ...]:
        direct = {key for key, _carriers in self._pointer_writers}
        derived = {binding.input_key for binding in self._tma_bindings}
        return tuple(sorted(direct | derived))

    def bind_hot(self, bindings: dict[str, Any]) -> None:
        """Refresh tensor maps and overwrite bound pointer carriers in place.

        This is the host-side half of a hot call. Callers that enqueue their
        own support launches between the plan's pointer binding and its
        kernel submission (the KNN-build fused row norms) must call this
        BEFORE those launches: a fresh-pointer tensor-map re-encode costs
        host time, and paying it after any kernel is already enqueued turns
        that host time into a GPU inter-kernel gap.

        Tensor maps go through the per-pointer variant bank: a pointer the
        bank has seen keeps its device-resident descriptor, so re-activating
        it is a handful of carrier writes with no ``cuTensorMapEncodeTiled``,
        no pinned-staging refresh, and no H2D copy. The plan's signature slot
        is stream-keyed and alias-keyed by the caller, which is the safety
        contract ``rebind_stream_bound`` requires.
        """

        for binding in self._tma_bindings:
            binding.rebind_stream_bound(bindings[binding.input_key], stream=self.torch_stream)
        for key, carriers in self._pointer_writers:
            pointer = bindings[key].data_ptr()
            for carrier in carriers:
                carrier.value = pointer

    def submit_hot(self, *, timeout_ms: float | None = None) -> None:
        """Submit every prepared launch on the plan's construction stream."""

        launches = self._launches
        last_index = len(launches) - 1
        for index, launch in enumerate(launches):
            launch.launch(stream=None, timeout_ms=timeout_ms if index == last_index else None)

    def launch_hot(self, bindings: dict[str, Any], *, timeout_ms: float | None = None) -> None:
        """Patch bound pointer carriers from ``bindings`` and submit every launch."""

        self.bind_hot(bindings)
        self.submit_hot(timeout_ms=timeout_ms)

    def record_stream(self, stream: Any) -> None:
        """Tie every plan-held launch argument to ``stream`` before release."""

        self.sequence.record_stream(stream)


class PerCallRoutePlan:
    """Per-signature plan for routes whose host logic reads device results.

    Capture observed the route reading GPU memory (for example an
    ``overflow_flag.item()`` certification) while its kernels were only being
    recorded, so a frozen launch list cannot reproduce the route's per-call
    branch decisions — freezing would bake whichever branch the capture-time
    garbage selected. These signatures keep the generic per-call launcher:
    route resolution stays cached and the guard cascade still ran exactly
    once, but every hot call re-executes the resolved route's host program.
    Device-side repair (Migration step 3) makes such routes replayable.
    """

    __slots__ = (
        "route",
        "torch_stream",
        "stream_handle",
        "launch_count",
        "host_data_reads",
        "_static_inputs",
        "_pending_bindings",
    )

    def __init__(
        self,
        route: RouteDecision,
        *,
        torch_stream: Any,
        stream_handle: int,
        static_inputs: dict[str, Any],
        launch_count: int,
        host_data_reads: int,
    ):
        self.route = route
        self.torch_stream = torch_stream
        self.stream_handle = int(stream_handle)
        self.launch_count = int(launch_count)
        self.host_data_reads = int(host_data_reads)
        self._static_inputs = dict(static_inputs)
        self._pending_bindings = None

    def bind_hot(self, bindings: dict[str, Any]) -> None:
        """Stage this call's tensor bindings for ``submit_hot``.

        Mirrors ``LaunchPlan``'s two-phase hot call so callers with support
        launches use one code path. The caller's per-signature slot lock
        serializes bind/submit pairs on a plan instance.
        """

        self._pending_bindings = dict(bindings)

    def submit_hot(self, *, timeout_ms: float | None = None) -> None:
        """Re-execute the resolved route with the staged tensor bindings."""

        bindings = self._pending_bindings
        if bindings is None:
            raise RuntimeError("PerCallRoutePlan.submit_hot requires a preceding bind_hot")
        self._pending_bindings = None
        self.launch_hot(bindings, timeout_ms=timeout_ms)

    def launch_hot(self, bindings: dict[str, Any], *, timeout_ms: float | None = None) -> None:
        """Re-execute the resolved route with this call's tensor bindings.

        The torch stream context keeps the route's tensor operations (scratch
        fills, the certification read-back) ordered with its kernel launches
        on the plan's stream, exactly as the live evaluation path runs it.
        """

        import torch

        inputs = dict(self._static_inputs)
        inputs.update(bindings)
        with torch.cuda.stream(self.torch_stream):
            self.route.launch(inputs, stream=self.torch_stream, timeout_ms=timeout_ms)

    def record_stream(self, stream: Any) -> None:
        """Per-call plans hold no launch arguments; nothing to record."""


def build_launch_plan(
    inputs: dict[str, Any],
    *,
    stream: Any,
    arch: str,
    validate_result: Callable[[Any, dict[str, Any]], None] | None = None,
    route: Any = None,
) -> LaunchPlan | PerCallRoutePlan:
    """Run the guard cascade once and freeze its launches into a LaunchPlan.

    This is the per-signature slow path; ``resolve_route`` remains the single
    source of routing truth. ``validate_result(result, inputs)`` must raise
    when the resolved route's outputs cannot be retargeted by pointer
    rebinding (for example outputs that are not caller-owned tensors).
    Routes whose host logic read device memory during capture resolve to a
    ``PerCallRoutePlan`` instead of a frozen launch list.

    ``route`` accepts an already-resolved decision from a sibling routing
    layer with the same contract (``route_id``/``launch_entrypoint``/
    ``exact_contract`` plus ``launch(inputs, stream=..., timeout_ms=...)``),
    for workloads whose exact-contract table lives outside this module (the
    KNN-build direct-manifest resolver). It must come from that workload's
    frozen routing surface, never from re-guessing the cascade.
    """

    from ._dispatch_runtime import capture_kernel_launches
    from ._runtime import launch_context

    if stream is None:
        raise ValueError("build_launch_plan requires a resolved torch CUDA stream, not None")
    if route is None:
        route = resolve_route(inputs)
    with capture_kernel_launches(stream=stream, arch=arch, inputs=inputs) as captured:
        with launch_context(arch=arch, stream=stream, timeout_ms=None):
            result = route.launch(inputs, stream=stream, timeout_ms=None)
    if validate_result is not None:
        validate_result(result, inputs)
    if captured.host_data_dependent:
        static_inputs = {
            key: value for key, value in inputs.items() if not callable(getattr(value, "data_ptr", None))
        }
        return PerCallRoutePlan(
            route,
            torch_stream=stream,
            stream_handle=int(stream.cuda_stream),
            static_inputs=static_inputs,
            launch_count=len(captured._launches),
            host_data_reads=captured.host_data_reads,
        )
    sequence = captured.bind(result)
    return LaunchPlan(
        route,
        sequence,
        torch_stream=stream,
        stream_handle=int(stream.cuda_stream),
    )


class GraphCaptureUnsupported(RuntimeError):
    """A plan's launches have no validated CUDA-graph capture path."""


def _check_cu(err: Any, message: str) -> None:
    code = err[0] if isinstance(err, tuple) else err
    if int(code) != 0:
        raise RuntimeError(f"{message}: CUresult={int(code)}")


class GraphExecPlan:
    """One per-signature CUDA graph over a LaunchPlan plus support launches.

    Migration step 3 of ``PLAN_BASED_EXPORT_RUNTIME.md``: the signature's
    stable kernel chain (support launches such as fused row norms, then the
    frozen route launches) is stream-captured once at plan construction. A
    hot call is host-only binding (the caller's ``plan.bind_hot`` plus
    support pointer writes into the same persistent packed argument buffers),
    then ``submit_hot``: every kernel node's packed buffer is pushed through
    ``cuGraphExecKernelNodeSetParams`` and the chain replays with one
    ``cuGraphLaunch`` on the plan's construction-time stream. Kernel-node
    launch attributes recorded at capture (cluster dimensions, scheduling
    preference) persist across exec-node parameter updates.
    """

    __slots__ = (
        "plan",
        "_launches",
        "_graph",
        "_graph_exec",
        "_node_params",
        "_cu_stream",
        "_set_params",
        "_graph_launch",
        "_cu_success",
        "_destroyed",
    )

    def __init__(self, plan: LaunchPlan, launches, graph, graph_exec, node_params, cu_stream):
        from cuda.bindings import driver

        self.plan = plan
        self._launches = tuple(launches)
        self._graph = graph
        self._graph_exec = graph_exec
        self._node_params = tuple(node_params)
        self._cu_stream = cu_stream
        self._set_params = driver.cuGraphExecKernelNodeSetParams
        self._graph_launch = driver.cuGraphLaunch
        self._cu_success = driver.CUresult.CUDA_SUCCESS
        self._destroyed = False

    @property
    def launch_count(self) -> int:
        return len(self._launches)

    def submit_hot(self, *, timeout_ms: float | None = None) -> None:
        """Push the persistent packed argument buffers and replay the graph.

        The caller must have completed every pointer/tensor-map bind for this
        call (``plan.bind_hot`` plus support-launch binds) first; parameter
        values are copied out of the packed buffers here.
        """

        if self._destroyed:
            raise RuntimeError("graph plan was destroyed by a runtime clear()")
        set_params = self._set_params
        graph_exec = self._graph_exec
        success = self._cu_success
        for node, params in self._node_params:
            (err,) = set_params(graph_exec, node, params)
            if err != success:
                _check_cu(err, "cuGraphExecKernelNodeSetParams failed")
        (err,) = self._graph_launch(graph_exec, self._cu_stream)
        if err != success:
            _check_cu(err, "cuGraphLaunch failed")
        if timeout_ms is not None:
            self._launches[-1]._kernel._wait_with_timeout(self._cu_stream, timeout_ms)

    def destroy(self) -> None:
        """Release the driver graph handles. Device work must be complete."""

        if self._destroyed:
            return
        from cuda.bindings import driver

        self._destroyed = True
        driver.cuGraphExecDestroy(self._graph_exec)
        driver.cuGraphDestroy(self._graph)


def build_graph_exec_plan(plan: Any, *, support_launches: tuple = ()) -> GraphExecPlan:
    """Capture ``support_launches`` then ``plan``'s launches into one graph.

    The per-signature slow path, run once at plan construction. Launches are
    replayed onto a dedicated capture stream (graph construction only — no
    kernel executes), each launch is mapped to its kernel node through the
    capture stream's leaf-dependency query, and the captured topology is
    hard-checked to contain exactly the expected kernel nodes so foreign
    work (for example watchdog event records) can never silently ride along.

    Raises ``GraphCaptureUnsupported`` for plans that cannot replay from a
    frozen kernel chain (per-call routes) and for launch modes without a
    validated capture path (cooperative). Any other failure propagates —
    a capture that should work but does not is an error, not a fallback.
    """

    import ctypes
    import sys
    from contextlib import nullcontext

    import torch
    from cuda.bindings import driver

    if not isinstance(plan, LaunchPlan):
        raise GraphCaptureUnsupported(
            "only frozen LaunchPlans are graph-capturable; per-call routes re-execute host logic"
        )
    launches = tuple(support_launches) + tuple(plan._launches)
    for launch in launches:
        if launch._mode not in ("regular", "cluster"):
            raise GraphCaptureUnsupported(
                f"launch mode {launch._mode!r} has no validated graph-capture path"
            )

    # Captured launches build graph nodes and do not execute, so loom's CUDA
    # watchdog (present only when the in-repo runtime shares this process)
    # must not record completion events for them: the event record would be
    # captured as a foreign node and the poller would query a captured event.
    watchdog = sys.modules.get("loom.runtime.cuda_watchdog")
    suspend = getattr(watchdog, "suspend_tracking", None)
    suspension = suspend() if callable(suspend) else nullcontext()

    capture_stream = torch.cuda.Stream(device=plan.torch_stream.device)
    cu_capture = driver.CUstream(capture_stream.cuda_stream)
    nodes = []
    with suspension:
        (err,) = driver.cuStreamBeginCapture(
            cu_capture, driver.CUstreamCaptureMode.CU_STREAM_CAPTURE_MODE_THREAD_LOCAL
        )
        _check_cu(err, "cuStreamBeginCapture failed")
        try:
            for launch in launches:
                launch.launch(stream=capture_stream, timeout_ms=None)
                info = driver.cuStreamGetCaptureInfo(cu_capture)
                _check_cu(info[0], "cuStreamGetCaptureInfo failed")
                status, leaves = info[1], info[4]
                if (
                    status != driver.CUstreamCaptureStatus.CU_STREAM_CAPTURE_STATUS_ACTIVE
                    or len(leaves) != 1
                ):
                    raise RuntimeError(
                        "graph capture did not add exactly one leaf node for a prepared launch"
                    )
                nodes.append(leaves[0])
        except BaseException:
            driver.cuStreamEndCapture(cu_capture)  # abandon the partial capture
            raise
        err, graph = driver.cuStreamEndCapture(cu_capture)
        _check_cu(err, "cuStreamEndCapture failed")

    try:
        err, _probe, total_nodes = driver.cuGraphGetNodes(graph, 0)
        _check_cu(err, "cuGraphGetNodes failed")
        if int(total_nodes) != len(launches):
            raise RuntimeError(
                f"captured graph has {int(total_nodes)} nodes, expected {len(launches)}; "
                "foreign work was injected into the capture"
            )
        for node in nodes:
            err, node_type = driver.cuGraphNodeGetType(node)
            _check_cu(err, "cuGraphNodeGetType failed")
            if node_type != driver.CUgraphNodeType.CU_GRAPH_NODE_TYPE_KERNEL:
                raise RuntimeError("captured graph node is not a kernel node")
        err, graph_exec = driver.cuGraphInstantiate(graph, 0)
        _check_cu(err, "cuGraphInstantiate failed")
    except BaseException:
        driver.cuGraphDestroy(graph)
        raise

    node_params = []
    for launch, node in zip(launches, nodes):
        params = driver.CUDA_KERNEL_NODE_PARAMS()
        params.func = launch._kernel._func
        params.gridDimX, params.gridDimY, params.gridDimZ = launch._grid
        params.blockDimX, params.blockDimY, params.blockDimZ = launch._block
        params.sharedMemBytes = launch._shared_mem
        params.kernelParams = ctypes.addressof(launch._packed)
        params.extra = 0
        node_params.append((node, params))
    return GraphExecPlan(
        plan,
        launches,
        graph,
        graph_exec,
        node_params,
        driver.CUstream(plan.stream_handle),
    )
