from __future__ import annotations
_KERNEL_ALIAS_BY_IR_NAME = {'knn_build_evolve_7bfc_split_cg2_stage1_batch8_cond4_vmin_threshold_k5': 'dispatch_kernel_0000', 'knn_build_evolve_7bfc_k5_merge_s4_tree': 'dispatch_kernel_0001', 'knn_build_evolve_7bfc_split_merge': 'dispatch_kernel_0003', 'knn_build_evolve_7bfc_split_cg2_u2_stage1_k8split': 'dispatch_kernel_0004', 'knn_build_evolve_7bfc_k30_merge_s8_rowbase_cache_k8s8': 'dispatch_kernel_0005', 'knn_build_evolve_7bfc_k10_merge_s4_rowbase_cache': 'dispatch_kernel_0006', 'knn_build_evolve_7bfc_split_cg2_u2_stage1_k16split': 'dispatch_kernel_0008', 'knn_build_evolve_7bfc_k30_merge_s8_rowbase_cache_f8c3lowk_k16s16': 'dispatch_kernel_0009', 'knn_build_evolve_7bfc_split_cg2_u2_stage1_k12split': 'dispatch_kernel_0010', 'knn_build_evolve_7bfc_k30_merge_s8_rowbase_cache_k12s16': 'dispatch_kernel_0011', 'knn_build_evolve_7bfc_split_cg2_u2_stage1_k20split': 'dispatch_kernel_0012', 'knn_build_evolve_7bfc_k30_merge_s8_rowbase_cache_k20s16': 'dispatch_kernel_0013', 'knn_build_evolve_7bfc_k30_merge_s8_rowbase_cache_195e_q1024k8s16': 'dispatch_kernel_0014', 'knn_build_q4096_k8_fd9b_stage1_unordered_exact_prefill': 'dispatch_kernel_0015', 'knn_build_q4096_k8_fd9b_merge_s4_unordered_warp_select': 'dispatch_kernel_0016', 'knn_build_dim_midk_73a9_d64_split_stage1': 'dispatch_kernel_0017', 'knn_build_d64_build_aa88_k10_merge_s8_rowbase_cache': 'dispatch_kernel_0018', 'knn_build_d64_q4096_c271_stage1_unordered_syncdrop': 'dispatch_kernel_0019', 'knn_build_d64_build_aa88_k10_merge_s8_rowbase_cache_s4': 'dispatch_kernel_0020', 'knn_build_non128_frontier_4be7_d96exact_stage1': 'dispatch_kernel_0021', 'knn_build_non128_frontier_3d5a_k10_merge_s8_rowbase_cache': 'dispatch_kernel_0022', 'knn_build_dim_midk_df2f_d256_split_stage1': 'dispatch_kernel_0023', 'knn_build_non128_frontier_7231_pad_bf16_rows_d256': 'dispatch_kernel_0024', 'knn_build_common_d_56f3_d256_q1024_k10_merge_rowbase_cache_s16': 'dispatch_kernel_0025', 'knn_build_common_d768_build_eeff_m64split_stage1': 'dispatch_kernel_0026', 'knn_build_non128_frontier_4be7_d768fused_merge_s16g8_4be7_d768fused_v1': 'dispatch_kernel_0027', 'knn_build_non128_frontier_7231_stage1_d1024': 'dispatch_kernel_0028', 'knn_build_common_d_56f3_k10_merge_rowbase_cache_s8': 'dispatch_kernel_0029', 'knn_build_non128_frontier_7231_stage1_d4096': 'dispatch_kernel_0030', 'knn_build_non128_frontier_8227_d320tail_stage1': 'dispatch_kernel_0031', 'knn_build_dim_midk_df2f_fp16_split_stage1': 'dispatch_kernel_0032', 'knn_build_fp16_d128_lowfloor_fd37_k10_s8_merge': 'dispatch_kernel_0033', 'knn_build_evolve_7bfc_split_cg2_u2_stage1_e080k11exact': 'dispatch_kernel_0034', 'knn_build_evolve_7bfc_k30_merge_s8_rowbase_cache_e080k11s8exact': 'dispatch_kernel_0035', 'knn_build_evolve_7bfc_k30_merge_s8_rowbase_cache_k12s8': 'dispatch_kernel_0037', 'knn_build_evolve_7bfc_split_cg2_u2_stage1_e080k13exact': 'dispatch_kernel_0038', 'knn_build_evolve_7bfc_k30_merge_s8_rowbase_cache_e080k13s8exact': 'dispatch_kernel_0039', 'knn_build_evolve_7bfc_k30_merge_s8_rowbase_cache_k20s8': 'dispatch_kernel_0040', 'knn_build_evolve_7bfc_split_cg2_u2_stage1_bad5midks8k24': 'dispatch_kernel_0041', 'knn_build_evolve_7bfc_k30_merge_s8_rowbase_cache_bad5midks8k24': 'dispatch_kernel_0042', 'knn_build_evolve_7bfc_split_cg2_u2_stage1_bad5midks8k28': 'dispatch_kernel_0043', 'knn_build_evolve_7bfc_k30_merge_s8_rowbase_cache_bad5midks8k28': 'dispatch_kernel_0044', 'knn_build_evolve_7bfc_k32_merge_s4_rowbase_cache_k20split': 'dispatch_kernel_0046', 'knn_build_evolve_7bfc_split_cg2_u2_stage1_k32_unordered_k12unordered': 'dispatch_kernel_0047', 'knn_build_evolve_7bfc_k32_merge_s4_unordered_k12unordered': 'dispatch_kernel_0048', 'knn_build_evolve_7bfc_split_cg2_u2_stage1_k32_unordered_2c1ck13unordered': 'dispatch_kernel_0049', 'knn_build_evolve_7bfc_k32_merge_s4_unordered_2c1ck13unordered': 'dispatch_kernel_0050', 'knn_build_evolve_7bfc_split_cg2_u2_stage1_k32_unordered_k20unordered': 'dispatch_kernel_0051', 'knn_build_evolve_7bfc_k32_merge_s4_unordered_k20unordered': 'dispatch_kernel_0052', 'knn_build_evolve_7bfc_split_cg2_u2_stage1_k32_unordered_k30unordered_1074k24unordered': 'dispatch_kernel_0053', 'knn_build_1074_k24_q4096_merge_s4_unordered_warp_select': 'dispatch_kernel_0054', 'knn_build_evolve_7bfc_split_cg2_u2_stage1_k32_unordered_k30unordered_bad5k28unordered': 'dispatch_kernel_0055', 'knn_build_evolve_7bfc_k32_merge_s4_unordered_k30unordered_bad5k28unordered': 'dispatch_kernel_0056', 'knn_build_evolve_7bfc_split_cg2_u2_stage1_k32_unordered': 'dispatch_kernel_0057', 'knn_build_evolve_7bfc_k32_merge_s4_unordered_warp_select': 'dispatch_kernel_0058', 'knn_build_evolve_7bfc_split_cg2_u2_stage1_k32_unordered_k30unordered': 'dispatch_kernel_0059', 'knn_build_k30_q4096_6998_merge_s4_unordered_warp_select': 'dispatch_kernel_0060', 'knn_build_evolve_7bfc_split_cg2_u2_stage1_k32_unordered_k48over32': 'dispatch_kernel_0061', 'knn_build_k48_merge_s4_unordered_warp_select': 'dispatch_kernel_0062', 'knn_build_k64_stage1_tailinf_k64over32tailinfsplitgrid': 'dispatch_kernel_0063', 'knn_build_k64_merge_s8_unordered_warp_select_k64over32s8warpselect': 'dispatch_kernel_0064', 'knn_build_k20_mergeown_08ec_warp8_select_s2warp8': 'dispatch_kernel_0066', 'knn_build_large_square_k32_stage1_chunkworst': 'dispatch_kernel_0067', 'knn_build_large_square_k32_s2_warp8_merge': 'dispatch_kernel_0068', 'knn_build_evolve_7bfc_k10_merge_s7_rowbase_cache_rectd15e_s16': 'dispatch_kernel_0069', 'knn_build_rect_d64_23be_s16_cached_merge': 'dispatch_kernel_0070', 'knn_build_rect_d64_23be_unordered_stage1': 'dispatch_kernel_0071', 'knn_build_d128_rag_q128_k10_s74_warp_merge_d320_s48_f556_v2': 'dispatch_kernel_0072', 'knn_build_non128_frontier_7231_stage1_d256': 'dispatch_kernel_0073', 'knn_build_non128_frontier_7231_stage1_d768': 'dispatch_kernel_0074', 'knn_build_non128_frontier_4be7_d768fused_merge_s32g8_4be7_d768fused_v1': 'dispatch_kernel_0075', 'knn_build_evolve_7bfc_k20_merge_s4_unordered_warp_select': 'dispatch_kernel_0076', 'knn_build_rect_d128_k20_q1536_warp4_merge': 'dispatch_kernel_0077', 'knn_build_ragonline_mbucket_4fc7_q1m262_v2_stage1_q1_k10_m64_halfrow': 'dispatch_kernel_0080', 'knn_build_rag_microbatch_4a72_k10_fused_group_final_merge_s144g12_4a72_v1': 'dispatch_kernel_0081', 'knn_build_ragonline_mbucket_ea43_q1m524_n128_stage1': 'dispatch_kernel_0082', 'knn_build_rag_microbatch_4a72_k10_fused_group_final_merge_s147g7_4a72_v1': 'dispatch_kernel_0083', 'knn_build_rag_stream_k10_q128_1bed_rowld_stage1': 'dispatch_kernel_0084', 'knn_build_d128_rag_q128_k10_s74_warp_merge_rowld_s74_1bed_v1': 'dispatch_kernel_0085', 'knn_build_k20_large_lowfanout_s2_warp_select': 'dispatch_kernel_0086', 'knn_build_rag_microbatch_4a72_v2_stage1_k10_cta1_maxtree': 'dispatch_kernel_0087', 'knn_build_rag_microbatch_4a72_v2_k10_fused_group_final_merge_s144g12_4a72_v2': 'dispatch_kernel_0088', 'knn_build_rag_microbatch_m64_d4f7_stage1': 'dispatch_kernel_0089', 'knn_build_rag_microbatch_4a72_k10_fused_group_final_merge_s136g8_4a72_v1': 'dispatch_kernel_0090', 'knn_build_non128_frontier_7ee5_m64rag_stage1': 'dispatch_kernel_0091', 'knn_build_non128_frontier_4be7_d768fused_merge_s72g8_4be7_d768fused_v1': 'dispatch_kernel_0092', 'knn_build_common_d_1438_rag_d64_m128_stage1': 'dispatch_kernel_0093', 'knn_build_non128_frontier_4be7_d768fused_merge_s136g8_4be7_d768fused_v1': 'dispatch_kernel_0094', 'knn_build_non128_frontier_7ee5_m64rag_stage1_d256_5e7f_rag_d64d256_v1': 'dispatch_kernel_0095', 'knn_build_non128_frontier_4be7_d768fused_merge_s144g8_4be7_d768fused_v1': 'dispatch_kernel_0096', 'knn_build_non128_frontier_7ee5_m64rag_stage1_d1024_5e7f_highd_v1': 'dispatch_kernel_0097', 'knn_build_non128_frontier_4be7_d768fused_merge_s144g12_4be7_d768fused_v1': 'dispatch_kernel_0098', 'knn_build_non128_frontier_7ee5_m64rag_stage1_d4096_5e7f_highd_v1': 'dispatch_kernel_0099', 'knn_build_non128_frontier_4be7_d768fused_merge_s128g8_4be7_d768fused_v1': 'dispatch_kernel_0100', 'knn_build_rag_microbatch_4a72_k10_fused_group_final_merge_s128g8_4a72_v1': 'dispatch_kernel_0101', 'knn_build_rag_microbucket_k32q8half_0077_v1_stage1_q8_k32_m64_halfrow_q8half_0077_v1': 'dispatch_kernel_0102', 'knn_build_rag_microbucket_k32warpmerge_0077_v1_warp_row_merge_k32s144_0077_v1': 'dispatch_kernel_0103', 'knn_build_rag_microbucket_k32_q16irreg2warp_a444_v2_stage1_q16_rowld1_2warp_q16dual2warp_56ed_v1': 'dispatch_kernel_0104', 'knn_build_rag_microbucket_k32warpmerge_0077_v1_warp_row_merge_k32s144r4_56ed_v1': 'dispatch_kernel_0105', 'knn_build_rag_microbucket_k32rowld1warp_0077_v1_stage1_q32_k32_m64_rowld2_q24rowld2_24dc_v1': 'dispatch_kernel_0106', 'knn_build_rag_microbucket_k32warpmerge_0077_v1_warp_row_merge_k32q24s144r4_24dc_v1': 'dispatch_kernel_0107', 'knn_build_rag_microbucket_k32warpmerge_0077_v1_warp_row_merge_k32s288r4_56ed_v1': 'dispatch_kernel_0108', 'knn_build_rag_microbucket_k32_q32rowld2exact_s141_72d1_v1_stage1_q32rowld2exact_f653_v1': 'dispatch_kernel_0109', 'knn_build_rag_microbucket_k32warpmerge_0077_v1_warp_row_merge_k32q32exact_s141r4_f653_v1': 'dispatch_kernel_0110', 'knn_build_rag_microbucket_k32warpmerge_0077_v1_warp_row_merge_k32s148r4_56ed_v1': 'dispatch_kernel_0111', 'knn_build_rag_stream_k32_q128m100000_staticn128_664a_stage1': 'dispatch_kernel_0112', 'knn_build_rag_frontier_7399_k32_fused_group_final_merge_k32s72g8_4fbf_v6': 'dispatch_kernel_0113', 'knn_build_rag_microbucket_q32rowld_e5db_v1_stage1_q32_k32_m64': 'dispatch_kernel_0114', 'knn_build_rag_microbucket_k32warpmerge_0077_v1_warp_row_merge_k32s72_0077_v1': 'dispatch_kernel_0115', 'knn_build_evolve_7bfc_k30_merge_s8_rowbase_cache_k10s72_4e09': 'dispatch_kernel_0116', 'knn_build_evolve_7bfc_k10_merge_s7_rowbase_cache_rect4452_s8': 'dispatch_kernel_0117', 'knn_build_k96_stage1_exact_prefill_q1024_k96over64exactprefillq1024_e5db': 'dispatch_kernel_0120', 'knn_build_k96_merge_s8_unordered_chunkprefill_k96over64s2chunkprefill_f9d1': 'dispatch_kernel_0121', 'knn_build_k96_merge_s8_unordered_chunkprefill_k96over64s4chunkprefill_f9d1': 'dispatch_kernel_0122', 'knn_build_v12_d64_tail_017a_stage1': 'dispatch_kernel_0123', 'knn_build_common_d768_build_eeff_m64split_stage1_d256_q128_k10_59fe_v1': 'dispatch_kernel_0124', 'knn_build_non128_frontier_7ee5_m64rag_stage1_d768_5e7f_highd_v1': 'dispatch_kernel_0125', 'knn_build_common_d768_build_eeff_m64split_stage1_d1024_be66_search_v1': 'dispatch_kernel_0126', 'knn_build_non128_frontier_4be7_d768fused_merge_s64g8_4be7_d768fused_v1': 'dispatch_kernel_0127', 'knn_build_common_d768_build_eeff_m64split_stage1_d4096_be66_search_v1': 'dispatch_kernel_0128', 'knn_build_v12_d256_k32_tail_59fe_v1_stage1_rowld': 'dispatch_kernel_0129', 'knn_build_rag_microbucket_k32warpmerge_0077_v1_warp_row_merge_k32s64_0077_v1': 'dispatch_kernel_0130', 'knn_build_v12_d128_q16_k48_dd2b_v1_stage1': 'dispatch_kernel_0131', 'knn_build_rag_microbucket_k32warpmerge_0077_v1_warp_row_merge_k48s144r4_dd2b_v1': 'dispatch_kernel_0132', 'knn_build_rag_stream_k10_s72_warp_row_merge_34da': 'dispatch_kernel_0133', 'knn_build_evolve_7bfc_k30_merge_s8_rowbase_cache_k10s64_3d97': 'dispatch_kernel_0134', 'knn_build_evolve_7bfc_v1': 'dispatch_kernel_0135', 'knn_build_evolve_7bfc_split_stage1': 'dispatch_kernel_0136', 'knn_build_evolve_7bfc_split_cg2_u2_stage1': 'dispatch_kernel_0137', 'knn_build_evolve_7bfc_k10_merge_s4': 'dispatch_kernel_0138', 'knn_build_evolve_7bfc_k10_merge_s7': 'dispatch_kernel_0139', 'knn_build_evolve_7bfc_split_cg2_stage1_batch8_cond4_vmin_threshold_k5_mintree': 'dispatch_kernel_0140', 'knn_build_evolve_7bfc_k5_merge_s4_tree_rowbase': 'dispatch_kernel_0141', 'knn_build_evolve_7bfc_k32_merge_s4_unordered': 'dispatch_kernel_0148', 'knn_build_evolve_7bfc_k30_merge_s8_rowbase_cache': 'dispatch_kernel_0149', 'knn_build_evolve_7bfc_k10_merge_s7_rowbase_cache_k8s7': 'dispatch_kernel_0152', 'knn_build_evolve_7bfc_fp16_d128_base': 'dispatch_kernel_0154', 'knn_build_evolve_7bfc_d256_twomma_base': 'dispatch_kernel_0155', 'knn_build_evolve_7bfc_d64_tcgen05_base': 'dispatch_kernel_0156', 'knn_build_evolve_7bfc_k20_merge_s4_unordered_warp_select_splitmajor': 'dispatch_kernel_0157', 'knn_build_k20_mergeown_08ec_s4_rowbase_lane': 'dispatch_kernel_0159', 'knn_build_k20_large_rect_s3_warp_select': 'dispatch_kernel_0160', 'knn_build_rag_frontier_b6d4_stage1_k32_sort4earlystop': 'dispatch_kernel_0162', 'knn_build_rag_frontier_7399_k32_fused_group_final_merge': 'dispatch_kernel_0163', 'knn_build_evolve_7bfc_k30_merge_s8_rowbase_cache_k32s32_4b5c': 'dispatch_kernel_0164', 'knn_build_evolve_7bfc_split_cg2_u2_stage1_k32_unordered_k96over64': 'dispatch_kernel_0165', 'knn_build_k96_merge_s8_unordered_chunkprefill_k96over64s8chunkprefill': 'dispatch_kernel_0166', 'knn_build_rag_frontier_b6d4_stage1_k32_chunked': 'dispatch_kernel_0167', 'knn_build_rag_frontier_4fbf_v7_stage1_k32_sort4earlystop_tailinf': 'dispatch_kernel_0176', 'knn_build_rag_frontier_4fbf_stage1_k32_sort4earlystop_tailinf': 'dispatch_kernel_0177', 'knn_build_evolve_7bfc_split_cg2_u2_stage1_bad5k24s8': 'dispatch_kernel_0178', 'knn_build_evolve_7bfc_split_cg2_u2_stage1_bad5k28s8': 'dispatch_kernel_0179', 'knn_build_evolve_7bfc_k30_merge_s8_rowbase_cache_bad5k24s8': 'dispatch_kernel_0180', 'knn_build_evolve_7bfc_k30_merge_s8_rowbase_cache_bad5k28s8': 'dispatch_kernel_0181', 'knn_build_large_square_k32_s2_warp_select': 'dispatch_kernel_0182', 'knn_build_ragonline_mbucket_aa88_q1m_s72_k10_coop_merge': 'dispatch_kernel_0184', 'knn_build_ragonline_mbucket_aa88_q1m_s72_k10_coop_merge_s74_m250': 'dispatch_kernel_0185', 'knn_build_rag_microbucket_5093_v1_stage1_k32_tailinf_cta1_compactwarp': 'dispatch_kernel_0186', 'knn_build_rag_microbucket_3505_v3_stage1_k32_tailinf_cta1': 'dispatch_kernel_0187', 'knn_build_k96_stage1_sort4_chunked_k96over64sort4chunked': 'dispatch_kernel_0189', 'knn_build_rag_microbucket_3505_v9_stage1_q8_k32_m64': 'dispatch_kernel_0191', 'knn_build_d128_rag_q128_k10_s74_warp_merge': 'dispatch_kernel_0192', 'knn_build_evolve_7bfc_k32_merge_s4_rowbase_cache_e080k11s4exact': 'dispatch_kernel_0194', 'knn_build_evolve_7bfc_k32_merge_s4_rowbase_cache_e080k13s4exact': 'dispatch_kernel_0195', 'knn_build_rag_microbucket_k32rowld1warp_0077_v1_stage1_q16_k32_m64_rowld1_q16rowld1_0077_v1': 'dispatch_kernel_0199', 'knn_build_rag_microbucket_k32rowld1warp_0077_v1_stage1_q32_k32_m64_rowld2_q32rowld2_0077_v1': 'dispatch_kernel_0200', 'knn_build_k96_stage1_sort4_prefill_q1024_k96over64sort4prefillq1024_8c56': 'dispatch_kernel_0202', 'knn_build_non128_frontier_8199_d384_stage1': 'dispatch_kernel_0203', 'knn_build_rag_microbucket_q32_k31_c3d2_v1_stage1_q32k31_c3d2_v1': 'dispatch_kernel_0204', 'knn_build_rag_microbucket_k32_f590_q32exact_v1_stage1_q32exact_f590_v1': 'dispatch_kernel_0205', 'knn_build_rag_microbucket_k12_2f22_q48exact_v1_stage1': 'dispatch_kernel_0206', 'knn_build_rag_microbucket_k32_q32rowld2uneven_f653_v1_stage1_q32rowld2uneven_f653_v1': 'dispatch_kernel_0207', 'knn_build_rag_microbucket_k32rowld1warp_0077_v1_stage1_q32_k32_m64_rowld2_q32rowld2_f653_v1': 'dispatch_kernel_0208', 'knn_build_rag_microbucket_k32_0cb5_q31tail_v2_stage1_q31exact_0cb5_v2': 'dispatch_kernel_0209', 'knn_build_rag_microbucket_k32_q16irreg2warp_a444_v2_stage1_q16_rowld1_2warp_q16irreg2warp_a444_v2': 'dispatch_kernel_0210', 'knn_build_rag_microbucket_q32rowld_e5db_v1_stage1_q32_k32_m64_q128rowld_60fb_v1': 'dispatch_kernel_0211', 'knn_build_common_d_5e7f_rag_d64_m64_stage1': 'dispatch_kernel_0212', 'knn_build_rag_microbucket_3505_v2_stage1_k32_tailinf_cta1': 'dispatch_kernel_0213', 'knn_build_evolve_7bfc_split_cg2_u2_stage1_k32_unordered_fd9b_k8unordered': 'dispatch_kernel_0220', 'knn_build_d64_q4096_c271_twostage_group_reduce': 'dispatch_kernel_0221', 'knn_build_d64_build_aa88_k10_merge_s8_rowbase_cache_c271_s5': 'dispatch_kernel_0222', 'knn_build_d64_build_aa88_k10_merge_s8_rowbase_cache_c271_s6': 'dispatch_kernel_0223', 'knn_build_d64_q4096_c271_stage1_syncdrop': 'dispatch_kernel_0224', 'knn_build_common_d_generic_direct_v1': 'dispatch_kernel_0227', 'knn_build_common_d_5e7f_rag_d64_repair_stage1': 'dispatch_kernel_0229', 'knn_build_rag_microbucket_k32_q32rowld2exact_f653_v1_stage1_q32rowld2exact_f653_v1': 'dispatch_kernel_0230', 'knn_build_k96_merge_s2_unordered_warp_select': 'dispatch_kernel_0231', 'knn_build_q1m524_workfeed_s147_g21_register_merge': 'dispatch_kernel_0232'}
_KERNEL_ALIAS_BY_REQUEST = {'{"computed_smem_bytes":50432,"constants":[["BLOCK_Q",128],["BLOCK_M",64],["FEAT_D",128],["TOP_K_SMALL",5]],"ir_name":"knn_build_evolve_7bfc_split_cg2_stage1_batch8_cond4_vmin_threshold_k5","kwargs":{"smem_bytes":50432,"validate":false},"threads":192}': 'dispatch_kernel_0000', '{"computed_smem_bytes":0,"constants":[["TOP_K_SMALL",5]],"ir_name":"knn_build_evolve_7bfc_k5_merge_s4_tree","kwargs":{"smem_bytes":0,"validate":false},"threads":256}': 'dispatch_kernel_0001', '{"computed_smem_bytes":50432,"constants":[["BLOCK_Q",128],["BLOCK_M",64],["FEAT_D",128],["TOP_K_MAX",10]],"ir_name":"knn_build_evolve_7bfc_split_cg2_stage1_batch8_cond4_vmin_maxtree","kwargs":{"smem_bytes":50432,"validate":false},"threads":192}': 'dispatch_kernel_0002', '{"computed_smem_bytes":0,"constants":[["TOP_K_MAX",10]],"ir_name":"knn_build_evolve_7bfc_split_merge","kwargs":{"smem_bytes":0,"validate":false},"threads":256}': 'dispatch_kernel_0003', '{"computed_smem_bytes":50432,"constants":[["BLOCK_Q",128],["BLOCK_M",64],["FEAT_D",128],["TOP_K_MAX",8]],"ir_name":"knn_build_evolve_7bfc_split_cg2_u2_stage1_k8split","kwargs":{"smem_bytes":50432,"validate":false},"threads":192}': 'dispatch_kernel_0004', '{"computed_smem_bytes":0,"constants":[["TOP_K_MAX",8],["SPLIT_COUNT",8]],"ir_name":"knn_build_evolve_7bfc_k30_merge_s8_rowbase_cache_k8s8","kwargs":{"smem_bytes":0,"validate":false},"threads":32}': 'dispatch_kernel_0005', '{"computed_smem_bytes":0,"constants":[["TOP_K_MAX",10],["SPLIT_COUNT",4]],"ir_name":"knn_build_evolve_7bfc_k10_merge_s4_rowbase_cache","kwargs":{"smem_bytes":0,"validate":false},"threads":32}': 'dispatch_kernel_0006', '{"computed_smem_bytes":0,"constants":[["TOP_K_MAX",10],["SPLIT_COUNT",7]],"ir_name":"knn_build_evolve_7bfc_k10_merge_s7_rowbase_cache","kwargs":{"smem_bytes":0,"validate":false},"threads":32}': 'dispatch_kernel_0007', '{"computed_smem_bytes":50432,"constants":[["BLOCK_Q",128],["BLOCK_M",64],["FEAT_D",128],["TOP_K_MAX",16]],"ir_name":"knn_build_evolve_7bfc_split_cg2_u2_stage1_k16split","kwargs":{"smem_bytes":50432,"validate":false},"threads":192}': 'dispatch_kernel_0008', '{"computed_smem_bytes":0,"constants":[["TOP_K_MAX",16],["SPLIT_COUNT",16]],"ir_name":"knn_build_evolve_7bfc_k30_merge_s8_rowbase_cache_f8c3lowk_k16s16","kwargs":{"smem_bytes":0,"validate":false},"threads":32}': 'dispatch_kernel_0009', '{"computed_smem_bytes":50432,"constants":[["BLOCK_Q",128],["BLOCK_M",64],["FEAT_D",128],["TOP_K_MAX",12]],"ir_name":"knn_build_evolve_7bfc_split_cg2_u2_stage1_k12split","kwargs":{"smem_bytes":50432,"validate":false},"threads":192}': 'dispatch_kernel_0010', '{"computed_smem_bytes":0,"constants":[["TOP_K_MAX",12],["SPLIT_COUNT",16]],"ir_name":"knn_build_evolve_7bfc_k30_merge_s8_rowbase_cache_k12s16","kwargs":{"smem_bytes":0,"validate":false},"threads":32}': 'dispatch_kernel_0011', '{"computed_smem_bytes":50432,"constants":[["BLOCK_Q",128],["BLOCK_M",64],["FEAT_D",128],["TOP_K_MAX",20]],"ir_name":"knn_build_evolve_7bfc_split_cg2_u2_stage1_k20split","kwargs":{"smem_bytes":50432,"validate":false},"threads":192}': 'dispatch_kernel_0012', '{"computed_smem_bytes":0,"constants":[["TOP_K_MAX",20],["SPLIT_COUNT",16]],"ir_name":"knn_build_evolve_7bfc_k30_merge_s8_rowbase_cache_k20s16","kwargs":{"smem_bytes":0,"validate":false},"threads":32}': 'dispatch_kernel_0013', '{"computed_smem_bytes":0,"constants":[["TOP_K_MAX",8],["SPLIT_COUNT",16]],"ir_name":"knn_build_evolve_7bfc_k30_merge_s8_rowbase_cache_195e_q1024k8s16","kwargs":{"smem_bytes":0,"validate":false},"threads":32}': 'dispatch_kernel_0014', '{"computed_smem_bytes":50432,"constants":[["BLOCK_Q",128],["BLOCK_M",64],["FEAT_D",128],["TOP_K_MAX",8]],"ir_name":"knn_build_q4096_k8_fd9b_stage1_unordered_exact_prefill","kwargs":{"smem_bytes":50432,"validate":false},"threads":192}': 'dispatch_kernel_0015', '{"computed_smem_bytes":0,"constants":[["TOP_K_MAX",8],["SPLIT_COUNT",4]],"ir_name":"knn_build_q4096_k8_fd9b_merge_s4_unordered_warp_select","kwargs":{"smem_bytes":0,"validate":false},"threads":128}': 'dispatch_kernel_0016', '{"computed_smem_bytes":25856,"constants":[["BLOCK_Q",128],["BLOCK_M",64],["TOP_K_MAX",10]],"ir_name":"knn_build_dim_midk_73a9_d64_split_stage1","kwargs":{"smem_bytes":25856,"validate":false},"threads":192}': 'dispatch_kernel_0017', '{"computed_smem_bytes":0,"constants":[["TOP_K_MAX",10],["SPLIT_COUNT",8]],"ir_name":"knn_build_d64_build_aa88_k10_merge_s8_rowbase_cache","kwargs":{"smem_bytes":0,"validate":false},"threads":32}': 'dispatch_kernel_0018', '{"computed_smem_bytes":25856,"constants":[["BLOCK_Q",128],["BLOCK_M",64],["TOP_K_MAX",10]],"ir_name":"knn_build_d64_q4096_c271_stage1_unordered_syncdrop","kwargs":{"smem_bytes":25856,"validate":false},"threads":192}': 'dispatch_kernel_0019', '{"computed_smem_bytes":0,"constants":[["TOP_K_MAX",10],["SPLIT_COUNT",4]],"ir_name":"knn_build_d64_build_aa88_k10_merge_s8_rowbase_cache_s4","kwargs":{"smem_bytes":0,"validate":false},"threads":32}': 'dispatch_kernel_0020', '{"computed_smem_bytes":38144,"constants":[["BLOCK_Q",128],["BLOCK_M",64],["TOP_K_MAX",10]],"ir_name":"knn_build_non128_frontier_4be7_d96exact_stage1","kwargs":{"smem_bytes":38144,"validate":false},"threads":192}': 'dispatch_kernel_0021', '{"computed_smem_bytes":0,"constants":[["TOP_K_MAX",10],["SPLIT_COUNT",8]],"ir_name":"knn_build_non128_frontier_3d5a_k10_merge_s8_rowbase_cache","kwargs":{"smem_bytes":0,"validate":false},"threads":32}': 'dispatch_kernel_0022', '{"computed_smem_bytes":99584,"constants":[["BLOCK_Q",128],["BLOCK_M",64],["TOP_K_MAX",10]],"ir_name":"knn_build_dim_midk_df2f_d256_split_stage1","kwargs":{"smem_bytes":99584,"validate":false},"threads":192}': 'dispatch_kernel_0023', '{"computed_smem_bytes":0,"constants":[["D_PAD",256]],"ir_name":"knn_build_non128_frontier_7231_pad_bf16_rows_d256","kwargs":{"smem_bytes":0,"validate":false},"threads":256}': 'dispatch_kernel_0024', '{"computed_smem_bytes":0,"constants":[["TOP_K_MAX",10],["SPLIT_COUNT",16]],"ir_name":"knn_build_common_d_56f3_d256_q1024_k10_merge_rowbase_cache_s16","kwargs":{"smem_bytes":0,"validate":false},"threads":32}': 'dispatch_kernel_0025', '{"computed_smem_bytes":50432,"constants":[["FEATURE_CHUNKS",6]],"ir_name":"knn_build_common_d768_build_eeff_m64split_stage1","kwargs":{"smem_bytes":50432,"validate":false},"threads":192}': 'dispatch_kernel_0026', '{"computed_smem_bytes":1024,"constants":[["TOP_K_MAX",10],["GROUP_COUNT",8],["GROUP_SPLITS",2]],"ir_name":"knn_build_non128_frontier_4be7_d768fused_merge_s16g8_4be7_d768fused_v1","kwargs":{"smem_bytes":1024,"validate":false},"threads":32}': 'dispatch_kernel_0027', '{"computed_smem_bytes":50432,"constants":[["BLOCK_Q",128],["BLOCK_M",64],["K_TILE",128],["TOP_K_MAX",10],["FEATURE_CHUNKS",8]],"ir_name":"knn_build_non128_frontier_7231_stage1_d1024","kwargs":{"smem_bytes":50432,"validate":false},"threads":192}': 'dispatch_kernel_0028', '{"computed_smem_bytes":0,"constants":[["TOP_K_MAX",10],["SPLIT_COUNT",8]],"ir_name":"knn_build_common_d_56f3_k10_merge_rowbase_cache_s8","kwargs":{"smem_bytes":0,"validate":false},"threads":32}': 'dispatch_kernel_0029', '{"computed_smem_bytes":50432,"constants":[["BLOCK_Q",128],["BLOCK_M",64],["K_TILE",128],["TOP_K_MAX",10],["FEATURE_CHUNKS",32]],"ir_name":"knn_build_non128_frontier_7231_stage1_d4096","kwargs":{"smem_bytes":50432,"validate":false},"threads":192}': 'dispatch_kernel_0030', '{"computed_smem_bytes":124160,"constants":[["BLOCK_Q",128],["BLOCK_M",64],["TOP_K_MAX",10]],"ir_name":"knn_build_non128_frontier_8227_d320tail_stage1","kwargs":{"smem_bytes":124160,"validate":false},"threads":192}': 'dispatch_kernel_0031', '{"computed_smem_bytes":50432,"constants":[["BLOCK_Q",128],["BLOCK_M",64],["TOP_K_MAX",10]],"ir_name":"knn_build_dim_midk_df2f_fp16_split_stage1","kwargs":{"smem_bytes":50432,"validate":false},"threads":192}': 'dispatch_kernel_0032', '{"computed_smem_bytes":0,"constants":[["TOP_K_MAX",10],["SPLIT_COUNT",8]],"ir_name":"knn_build_fp16_d128_lowfloor_fd37_k10_s8_merge","kwargs":{"smem_bytes":0,"validate":false},"threads":32}': 'dispatch_kernel_0033', '{"computed_smem_bytes":50432,"constants":[["BLOCK_Q",128],["BLOCK_M",64],["FEAT_D",128],["TOP_K_MAX",11]],"ir_name":"knn_build_evolve_7bfc_split_cg2_u2_stage1_e080k11exact","kwargs":{"smem_bytes":50432,"validate":false},"threads":192}': 'dispatch_kernel_0034', '{"computed_smem_bytes":0,"constants":[["TOP_K_MAX",11],["SPLIT_COUNT",8]],"ir_name":"knn_build_evolve_7bfc_k30_merge_s8_rowbase_cache_e080k11s8exact","kwargs":{"smem_bytes":0,"validate":false},"threads":32}': 'dispatch_kernel_0035', '{"computed_smem_bytes":0,"constants":[["TOP_K_MAX",12],["SPLIT_COUNT",8]],"ir_name":"knn_build_evolve_7bfc_k30_merge_s8_rowbase_cache_k12s8","kwargs":{"smem_bytes":0,"validate":false},"threads":32}': 'dispatch_kernel_0037', '{"computed_smem_bytes":50432,"constants":[["BLOCK_Q",128],["BLOCK_M",64],["FEAT_D",128],["TOP_K_MAX",13]],"ir_name":"knn_build_evolve_7bfc_split_cg2_u2_stage1_e080k13exact","kwargs":{"smem_bytes":50432,"validate":false},"threads":192}': 'dispatch_kernel_0038', '{"computed_smem_bytes":0,"constants":[["TOP_K_MAX",13],["SPLIT_COUNT",8]],"ir_name":"knn_build_evolve_7bfc_k30_merge_s8_rowbase_cache_e080k13s8exact","kwargs":{"smem_bytes":0,"validate":false},"threads":32}': 'dispatch_kernel_0039', '{"computed_smem_bytes":0,"constants":[["TOP_K_MAX",20],["SPLIT_COUNT",8]],"ir_name":"knn_build_evolve_7bfc_k30_merge_s8_rowbase_cache_k20s8","kwargs":{"smem_bytes":0,"validate":false},"threads":32}': 'dispatch_kernel_0040', '{"computed_smem_bytes":50432,"constants":[["BLOCK_Q",128],["BLOCK_M",64],["FEAT_D",128],["TOP_K_MAX",24]],"ir_name":"knn_build_evolve_7bfc_split_cg2_u2_stage1_bad5midks8k24","kwargs":{"smem_bytes":50432,"validate":false},"threads":192}': 'dispatch_kernel_0041', '{"computed_smem_bytes":0,"constants":[["TOP_K_MAX",24],["SPLIT_COUNT",8]],"ir_name":"knn_build_evolve_7bfc_k30_merge_s8_rowbase_cache_bad5midks8k24","kwargs":{"smem_bytes":0,"validate":false},"threads":32}': 'dispatch_kernel_0042', '{"computed_smem_bytes":50432,"constants":[["BLOCK_Q",128],["BLOCK_M",64],["FEAT_D",128],["TOP_K_MAX",28]],"ir_name":"knn_build_evolve_7bfc_split_cg2_u2_stage1_bad5midks8k28","kwargs":{"smem_bytes":50432,"validate":false},"threads":192}': 'dispatch_kernel_0043', '{"computed_smem_bytes":0,"constants":[["TOP_K_MAX",28],["SPLIT_COUNT",8]],"ir_name":"knn_build_evolve_7bfc_k30_merge_s8_rowbase_cache_bad5midks8k28","kwargs":{"smem_bytes":0,"validate":false},"threads":32}': 'dispatch_kernel_0044', '{"computed_smem_bytes":0,"constants":[["TOP_K_MAX",20]],"ir_name":"knn_build_evolve_7bfc_k32_merge_s4_rowbase_cache_k20split","kwargs":{"smem_bytes":0,"validate":false},"threads":32}': 'dispatch_kernel_0046', '{"computed_smem_bytes":50432,"constants":[["BLOCK_Q",128],["BLOCK_M",64],["FEAT_D",128],["TOP_K_MAX",12]],"ir_name":"knn_build_evolve_7bfc_split_cg2_u2_stage1_k32_unordered_k12unordered","kwargs":{"smem_bytes":50432,"validate":false},"threads":192}': 'dispatch_kernel_0047', '{"computed_smem_bytes":0,"constants":[["TOP_K_MAX",12],["SPLIT_COUNT",4]],"ir_name":"knn_build_evolve_7bfc_k32_merge_s4_unordered_k12unordered","kwargs":{"smem_bytes":0,"validate":false},"threads":32}': 'dispatch_kernel_0048', '{"computed_smem_bytes":50432,"constants":[["BLOCK_Q",128],["BLOCK_M",64],["FEAT_D",128],["TOP_K_MAX",13]],"ir_name":"knn_build_evolve_7bfc_split_cg2_u2_stage1_k32_unordered_2c1ck13unordered","kwargs":{"smem_bytes":50432,"validate":false},"threads":192}': 'dispatch_kernel_0049', '{"computed_smem_bytes":0,"constants":[["TOP_K_MAX",13],["SPLIT_COUNT",4]],"ir_name":"knn_build_evolve_7bfc_k32_merge_s4_unordered_2c1ck13unordered","kwargs":{"smem_bytes":0,"validate":false},"threads":32}': 'dispatch_kernel_0050', '{"computed_smem_bytes":50432,"constants":[["BLOCK_Q",128],["BLOCK_M",64],["FEAT_D",128],["TOP_K_MAX",20]],"ir_name":"knn_build_evolve_7bfc_split_cg2_u2_stage1_k32_unordered_k20unordered","kwargs":{"smem_bytes":50432,"validate":false},"threads":192}': 'dispatch_kernel_0051', '{"computed_smem_bytes":0,"constants":[["TOP_K_MAX",20],["SPLIT_COUNT",4]],"ir_name":"knn_build_evolve_7bfc_k32_merge_s4_unordered_k20unordered","kwargs":{"smem_bytes":0,"validate":false},"threads":32}': 'dispatch_kernel_0052', '{"computed_smem_bytes":50432,"constants":[["BLOCK_Q",128],["BLOCK_M",64],["FEAT_D",128],["TOP_K_MAX",24]],"ir_name":"knn_build_evolve_7bfc_split_cg2_u2_stage1_k32_unordered_k30unordered_1074k24unordered","kwargs":{"smem_bytes":50432,"validate":false},"threads":192}': 'dispatch_kernel_0053', '{"computed_smem_bytes":0,"constants":[["TOP_K_MAX",24],["SPLIT_COUNT",4]],"ir_name":"knn_build_1074_k24_q4096_merge_s4_unordered_warp_select","kwargs":{"smem_bytes":0,"validate":false},"threads":128}': 'dispatch_kernel_0054', '{"computed_smem_bytes":50432,"constants":[["BLOCK_Q",128],["BLOCK_M",64],["FEAT_D",128],["TOP_K_MAX",28]],"ir_name":"knn_build_evolve_7bfc_split_cg2_u2_stage1_k32_unordered_k30unordered_bad5k28unordered","kwargs":{"smem_bytes":50432,"validate":false},"threads":192}': 'dispatch_kernel_0055', '{"computed_smem_bytes":0,"constants":[["TOP_K_MAX",28],["SPLIT_COUNT",4]],"ir_name":"knn_build_evolve_7bfc_k32_merge_s4_unordered_k30unordered_bad5k28unordered","kwargs":{"smem_bytes":0,"validate":false},"threads":32}': 'dispatch_kernel_0056', '{"computed_smem_bytes":50432,"constants":[["BLOCK_Q",128],["BLOCK_M",64],["FEAT_D",128],["TOP_K_MAX",32]],"ir_name":"knn_build_evolve_7bfc_split_cg2_u2_stage1_k32_unordered","kwargs":{"smem_bytes":50432,"validate":false},"threads":192}': 'dispatch_kernel_0057', '{"computed_smem_bytes":0,"constants":[["TOP_K_MAX",32],["SPLIT_COUNT",4]],"ir_name":"knn_build_evolve_7bfc_k32_merge_s4_unordered_warp_select","kwargs":{"smem_bytes":0,"validate":false},"threads":128}': 'dispatch_kernel_0058', '{"computed_smem_bytes":50432,"constants":[["BLOCK_Q",128],["BLOCK_M",64],["FEAT_D",128],["TOP_K_MAX",30]],"ir_name":"knn_build_evolve_7bfc_split_cg2_u2_stage1_k32_unordered_k30unordered","kwargs":{"smem_bytes":50432,"validate":false},"threads":192}': 'dispatch_kernel_0059', '{"computed_smem_bytes":0,"constants":[["TOP_K_MAX",30],["SPLIT_COUNT",4]],"ir_name":"knn_build_k30_q4096_6998_merge_s4_unordered_warp_select","kwargs":{"smem_bytes":0,"validate":false},"threads":128}': 'dispatch_kernel_0060', '{"computed_smem_bytes":50432,"constants":[["BLOCK_Q",128],["BLOCK_M",64],["FEAT_D",128],["TOP_K_MAX",48]],"ir_name":"knn_build_evolve_7bfc_split_cg2_u2_stage1_k32_unordered_k48over32","kwargs":{"smem_bytes":50432,"validate":false},"threads":192}': 'dispatch_kernel_0061', '{"computed_smem_bytes":0,"constants":[["TOP_K_MAX",48],["SPLIT_COUNT",4]],"ir_name":"knn_build_k48_merge_s4_unordered_warp_select","kwargs":{"smem_bytes":0,"validate":false},"threads":128}': 'dispatch_kernel_0062', '{"computed_smem_bytes":50432,"constants":[["BLOCK_Q",128],["BLOCK_M",64],["FEAT_D",128],["TOP_K_MAX",64]],"ir_name":"knn_build_k64_stage1_tailinf_k64over32tailinfsplitgrid","kwargs":{"smem_bytes":50432,"validate":false},"threads":192}': 'dispatch_kernel_0063', '{"computed_smem_bytes":0,"constants":[["TOP_K_MAX",64],["SPLIT_COUNT",8]],"ir_name":"knn_build_k64_merge_s8_unordered_warp_select_k64over32s8warpselect","kwargs":{"smem_bytes":0,"validate":false},"threads":128}': 'dispatch_kernel_0064', '{"computed_smem_bytes":0,"constants":[["TOP_K_MAX",20],["SPLIT_COUNT",2]],"ir_name":"knn_build_k20_mergeown_08ec_warp8_select_s2warp8","kwargs":{"smem_bytes":0,"validate":false},"threads":256}': 'dispatch_kernel_0066', '{"computed_smem_bytes":50432,"constants":[["BLOCK_Q",128],["BLOCK_M",64],["FEAT_D",128],["TOP_K_MAX",32]],"ir_name":"knn_build_large_square_k32_stage1_chunkworst","kwargs":{"smem_bytes":50432,"validate":false},"threads":192}': 'dispatch_kernel_0067', '{"computed_smem_bytes":0,"constants":[["TOP_K_MAX",32]],"ir_name":"knn_build_large_square_k32_s2_warp8_merge","kwargs":{"smem_bytes":0,"validate":false},"threads":256}': 'dispatch_kernel_0068', '{"computed_smem_bytes":0,"constants":[["TOP_K_MAX",10],["SPLIT_COUNT",16]],"ir_name":"knn_build_evolve_7bfc_k10_merge_s7_rowbase_cache_rectd15e_s16","kwargs":{"smem_bytes":0,"validate":false},"threads":32}': 'dispatch_kernel_0069', '{"computed_smem_bytes":0,"constants":[["TOP_K_MAX",10],["SPLIT_COUNT",16]],"ir_name":"knn_build_rect_d64_23be_s16_cached_merge","kwargs":{"smem_bytes":0,"validate":false},"threads":8}': 'dispatch_kernel_0070', '{"computed_smem_bytes":25856,"constants":[["BLOCK_Q",128],["BLOCK_M",64],["TOP_K_MAX",10]],"ir_name":"knn_build_rect_d64_23be_unordered_stage1","kwargs":{"smem_bytes":25856,"validate":false},"threads":192}': 'dispatch_kernel_0071', '{"computed_smem_bytes":0,"constants":[["TOP_K_MAX",10],["SPLIT_COUNT",48]],"ir_name":"knn_build_d128_rag_q128_k10_s74_warp_merge_d320_s48_f556_v2","kwargs":{"smem_bytes":0,"validate":false},"threads":128}': 'dispatch_kernel_0072', '{"computed_smem_bytes":50432,"constants":[["BLOCK_Q",128],["BLOCK_M",64],["K_TILE",128],["TOP_K_MAX",10],["FEATURE_CHUNKS",2]],"ir_name":"knn_build_non128_frontier_7231_stage1_d256","kwargs":{"smem_bytes":50432,"validate":false},"threads":192}': 'dispatch_kernel_0073', '{"computed_smem_bytes":50432,"constants":[["BLOCK_Q",128],["BLOCK_M",64],["K_TILE",128],["TOP_K_MAX",10],["FEATURE_CHUNKS",6]],"ir_name":"knn_build_non128_frontier_7231_stage1_d768","kwargs":{"smem_bytes":50432,"validate":false},"threads":192}': 'dispatch_kernel_0074', '{"computed_smem_bytes":1024,"constants":[["TOP_K_MAX",10],["GROUP_COUNT",8],["GROUP_SPLITS",4]],"ir_name":"knn_build_non128_frontier_4be7_d768fused_merge_s32g8_4be7_d768fused_v1","kwargs":{"smem_bytes":1024,"validate":false},"threads":32}': 'dispatch_kernel_0075', '{"computed_smem_bytes":0,"constants":[["TOP_K_MAX",20],["SPLIT_COUNT",4]],"ir_name":"knn_build_evolve_7bfc_k20_merge_s4_unordered_warp_select","kwargs":{"smem_bytes":0,"validate":false},"threads":128}': 'dispatch_kernel_0076', '{"computed_smem_bytes":0,"constants":[["TOP_K_MAX",20],["SPLIT_COUNT_CONST",12]],"ir_name":"knn_build_rect_d128_k20_q1536_warp4_merge","kwargs":{"smem_bytes":0,"validate":false},"threads":128}': 'dispatch_kernel_0077', '{"computed_smem_bytes":36608,"constants":[["BLOCK_Q",64],["BLOCK_M",64],["FEAT_D",128],["TOP_K_MAX",10],["ROWS_COVERED",1]],"ir_name":"knn_build_ragonline_mbucket_4fc7_q1m262_v2_stage1_q1_k10_m64_halfrow","kwargs":{"smem_bytes":36608,"validate":false},"threads":96}': 'dispatch_kernel_0080', '{"computed_smem_bytes":1024,"constants":[["TOP_K_MAX",10],["GROUP_COUNT",12],["GROUP_SPLITS",12]],"ir_name":"knn_build_rag_microbatch_4a72_k10_fused_group_final_merge_s144g12_4a72_v1","kwargs":{"smem_bytes":1024,"validate":false},"threads":32}': 'dispatch_kernel_0081', '{"computed_smem_bytes":52992,"constants":[["BLOCK_Q",64],["BLOCK_M",128],["FEAT_D",128],["TOP_K_MAX",10],["ROWS_COVERED",1]],"ir_name":"knn_build_ragonline_mbucket_ea43_q1m524_n128_stage1","kwargs":{"smem_bytes":52992,"validate":false},"threads":96}': 'dispatch_kernel_0082', '{"computed_smem_bytes":1024,"constants":[["TOP_K_MAX",10],["GROUP_COUNT",7],["GROUP_SPLITS",21]],"ir_name":"knn_build_rag_microbatch_4a72_k10_fused_group_final_merge_s147g7_4a72_v1","kwargs":{"smem_bytes":1024,"validate":false},"threads":32}': 'dispatch_kernel_0083', '{"computed_smem_bytes":54528,"constants":[["BLOCK_Q",64],["BLOCK_M",64],["FEAT_D",128],["TOP_K_MAX",10]],"ir_name":"knn_build_rag_stream_k10_q128_1bed_rowld_stage1","kwargs":{"smem_bytes":54528,"validate":false},"threads":192}': 'dispatch_kernel_0084', '{"computed_smem_bytes":0,"constants":[["TOP_K_MAX",10],["SPLIT_COUNT",74]],"ir_name":"knn_build_d128_rag_q128_k10_s74_warp_merge_rowld_s74_1bed_v1","kwargs":{"smem_bytes":0,"validate":false},"threads":128}': 'dispatch_kernel_0085', '{"computed_smem_bytes":0,"constants":[["TOP_K_MAX",20]],"ir_name":"knn_build_k20_large_lowfanout_s2_warp_select","kwargs":{"smem_bytes":0,"validate":false},"threads":128}': 'dispatch_kernel_0086', '{"computed_smem_bytes":50432,"constants":[["BLOCK_Q",128],["BLOCK_M",64],["FEAT_D",128],["TOP_K_MAX",10]],"ir_name":"knn_build_rag_microbatch_4a72_v2_stage1_k10_cta1_maxtree","kwargs":{"smem_bytes":50432,"validate":false},"threads":192}': 'dispatch_kernel_0087', '{"computed_smem_bytes":1024,"constants":[["TOP_K_MAX",10],["GROUP_COUNT",12],["GROUP_SPLITS",12]],"ir_name":"knn_build_rag_microbatch_4a72_v2_k10_fused_group_final_merge_s144g12_4a72_v2","kwargs":{"smem_bytes":1024,"validate":false},"threads":32}': 'dispatch_kernel_0088', '{"computed_smem_bytes":91392,"constants":[],"ir_name":"knn_build_rag_microbatch_m64_d4f7_stage1","kwargs":{"smem_bytes":91392,"validate":false},"threads":512}': 'dispatch_kernel_0089', '{"computed_smem_bytes":1024,"constants":[["TOP_K_MAX",10],["GROUP_COUNT",8],["GROUP_SPLITS",17]],"ir_name":"knn_build_rag_microbatch_4a72_k10_fused_group_final_merge_s136g8_4a72_v1","kwargs":{"smem_bytes":1024,"validate":false},"threads":32}': 'dispatch_kernel_0090', '{"computed_smem_bytes":34048,"constants":[["FEATURE_CHUNKS",6]],"ir_name":"knn_build_non128_frontier_7ee5_m64rag_stage1","kwargs":{"smem_bytes":34048,"validate":false},"threads":96}': 'dispatch_kernel_0091', '{"computed_smem_bytes":1024,"constants":[["TOP_K_MAX",10],["GROUP_COUNT",8],["GROUP_SPLITS",9]],"ir_name":"knn_build_non128_frontier_4be7_d768fused_merge_s72g8_4be7_d768fused_v1","kwargs":{"smem_bytes":1024,"validate":false},"threads":32}': 'dispatch_kernel_0092', '{"computed_smem_bytes":66816,"constants":[],"ir_name":"knn_build_common_d_1438_rag_d64_m128_stage1","kwargs":{"smem_bytes":66816,"validate":false},"threads":512}': 'dispatch_kernel_0093', '{"computed_smem_bytes":1024,"constants":[["TOP_K_MAX",10],["GROUP_COUNT",8],["GROUP_SPLITS",17]],"ir_name":"knn_build_non128_frontier_4be7_d768fused_merge_s136g8_4be7_d768fused_v1","kwargs":{"smem_bytes":1024,"validate":false},"threads":32}': 'dispatch_kernel_0094', '{"computed_smem_bytes":34048,"constants":[["FEATURE_CHUNKS",2]],"ir_name":"knn_build_non128_frontier_7ee5_m64rag_stage1_d256_5e7f_rag_d64d256_v1","kwargs":{"smem_bytes":34048,"validate":false},"threads":96}': 'dispatch_kernel_0095', '{"computed_smem_bytes":1024,"constants":[["TOP_K_MAX",10],["GROUP_COUNT",8],["GROUP_SPLITS",18]],"ir_name":"knn_build_non128_frontier_4be7_d768fused_merge_s144g8_4be7_d768fused_v1","kwargs":{"smem_bytes":1024,"validate":false},"threads":32}': 'dispatch_kernel_0096', '{"computed_smem_bytes":34048,"constants":[["FEATURE_CHUNKS",8]],"ir_name":"knn_build_non128_frontier_7ee5_m64rag_stage1_d1024_5e7f_highd_v1","kwargs":{"smem_bytes":34048,"validate":false},"threads":96}': 'dispatch_kernel_0097', '{"computed_smem_bytes":1024,"constants":[["TOP_K_MAX",10],["GROUP_COUNT",12],["GROUP_SPLITS",12]],"ir_name":"knn_build_non128_frontier_4be7_d768fused_merge_s144g12_4be7_d768fused_v1","kwargs":{"smem_bytes":1024,"validate":false},"threads":32}': 'dispatch_kernel_0098', '{"computed_smem_bytes":34048,"constants":[["FEATURE_CHUNKS",32]],"ir_name":"knn_build_non128_frontier_7ee5_m64rag_stage1_d4096_5e7f_highd_v1","kwargs":{"smem_bytes":34048,"validate":false},"threads":96}': 'dispatch_kernel_0099', '{"computed_smem_bytes":1024,"constants":[["TOP_K_MAX",10],["GROUP_COUNT",8],["GROUP_SPLITS",16]],"ir_name":"knn_build_non128_frontier_4be7_d768fused_merge_s128g8_4be7_d768fused_v1","kwargs":{"smem_bytes":1024,"validate":false},"threads":32}': 'dispatch_kernel_0100', '{"computed_smem_bytes":1024,"constants":[["TOP_K_MAX",10],["GROUP_COUNT",8],["GROUP_SPLITS",16]],"ir_name":"knn_build_rag_microbatch_4a72_k10_fused_group_final_merge_s128g8_4a72_v1","kwargs":{"smem_bytes":1024,"validate":false},"threads":32}': 'dispatch_kernel_0101', '{"computed_smem_bytes":42240,"constants":[["BLOCK_Q",64],["BLOCK_M",64],["FEAT_D",128],["TOP_K_MAX",32],["ROWS_COVERED",8]],"ir_name":"knn_build_rag_microbucket_k32q8half_0077_v1_stage1_q8_k32_m64_halfrow_q8half_0077_v1","kwargs":{"smem_bytes":42240,"validate":false},"threads":96}': 'dispatch_kernel_0102', '{"computed_smem_bytes":0,"constants":[["TOP_K_MAX",32],["SPLIT_COUNT",144],["SPLITS_PER_LANE",5],["ROWS_PER_CTA",1]],"ir_name":"knn_build_rag_microbucket_k32warpmerge_0077_v1_warp_row_merge_k32s144_0077_v1","kwargs":{"smem_bytes":0,"validate":false},"threads":128}': 'dispatch_kernel_0103', '{"computed_smem_bytes":52480,"constants":[["BLOCK_Q",64],["BLOCK_M",64],["FEAT_D",128],["TOP_K_MAX",32],["ROWS_COVERED",16]],"ir_name":"knn_build_rag_microbucket_k32_q16irreg2warp_a444_v2_stage1_q16_rowld1_2warp_q16dual2warp_56ed_v1","kwargs":{"smem_bytes":52480,"validate":false},"threads":128}': 'dispatch_kernel_0104', '{"computed_smem_bytes":0,"constants":[["TOP_K_MAX",32],["SPLIT_COUNT",144],["SPLITS_PER_LANE",5],["ROWS_PER_CTA",4]],"ir_name":"knn_build_rag_microbucket_k32warpmerge_0077_v1_warp_row_merge_k32s144r4_56ed_v1","kwargs":{"smem_bytes":0,"validate":false},"threads":128}': 'dispatch_kernel_0105', '{"computed_smem_bytes":66816,"constants":[["BLOCK_Q",64],["BLOCK_M",64],["FEAT_D",128],["TOP_K_MAX",32],["ROWS_COVERED",24]],"ir_name":"knn_build_rag_microbucket_k32rowld1warp_0077_v1_stage1_q32_k32_m64_rowld2_q24rowld2_24dc_v1","kwargs":{"smem_bytes":66816,"validate":false},"threads":128}': 'dispatch_kernel_0106', '{"computed_smem_bytes":0,"constants":[["TOP_K_MAX",32],["SPLIT_COUNT",144],["SPLITS_PER_LANE",5],["ROWS_PER_CTA",4]],"ir_name":"knn_build_rag_microbucket_k32warpmerge_0077_v1_warp_row_merge_k32q24s144r4_24dc_v1","kwargs":{"smem_bytes":0,"validate":false},"threads":128}': 'dispatch_kernel_0107', '{"computed_smem_bytes":0,"constants":[["TOP_K_MAX",32],["SPLIT_COUNT",288],["SPLITS_PER_LANE",9],["ROWS_PER_CTA",4]],"ir_name":"knn_build_rag_microbucket_k32warpmerge_0077_v1_warp_row_merge_k32s288r4_56ed_v1","kwargs":{"smem_bytes":0,"validate":false},"threads":128}': 'dispatch_kernel_0108', '{"computed_smem_bytes":66816,"constants":[["BLOCK_Q",64],["BLOCK_M",64],["FEAT_D",128],["TOP_K_MAX",32],["ROWS_COVERED",32],["SPLIT_COUNT_CONST",141],["NUM_DB_TILES_CONST",1563],["TILES_FLOOR_CONST",11],["EXTRA_SPLITS_CONST",12],["DB_TILES_PER_SPLIT_CONST",12],["M_LIMIT",100000]],"ir_name":"knn_build_rag_microbucket_k32_q32rowld2exact_s141_72d1_v1_stage1_q32rowld2exact_f653_v1","kwargs":{"smem_bytes":66816,"validate":false},"threads":128}': 'dispatch_kernel_0109', '{"computed_smem_bytes":0,"constants":[["TOP_K_MAX",32],["SPLIT_COUNT",141],["SPLITS_PER_LANE",5],["ROWS_PER_CTA",4]],"ir_name":"knn_build_rag_microbucket_k32warpmerge_0077_v1_warp_row_merge_k32q32exact_s141r4_f653_v1","kwargs":{"smem_bytes":0,"validate":false},"threads":128}': 'dispatch_kernel_0110', '{"computed_smem_bytes":0,"constants":[["TOP_K_MAX",32],["SPLIT_COUNT",148],["SPLITS_PER_LANE",5],["ROWS_PER_CTA",4]],"ir_name":"knn_build_rag_microbucket_k32warpmerge_0077_v1_warp_row_merge_k32s148r4_56ed_v1","kwargs":{"smem_bytes":0,"validate":false},"threads":128}': 'dispatch_kernel_0111', '{"computed_smem_bytes":181504,"constants":[],"ir_name":"knn_build_rag_stream_k32_q128m100000_staticn128_664a_stage1","kwargs":{"smem_bytes":181504,"validate":false},"threads":256}': 'dispatch_kernel_0112', '{"computed_smem_bytes":2048,"constants":[["TOP_K_MAX",32],["GROUP_COUNT",8],["GROUP_SPLITS",9]],"ir_name":"knn_build_rag_frontier_7399_k32_fused_group_final_merge_k32s72g8_4fbf_v6","kwargs":{"smem_bytes":2048,"validate":false},"threads":32}': 'dispatch_kernel_0113', '{"computed_smem_bytes":99584,"constants":[["BLOCK_Q",64],["BLOCK_M",64],["FEAT_D",128],["TOP_K_MAX",32]],"ir_name":"knn_build_rag_microbucket_q32rowld_e5db_v1_stage1_q32_k32_m64","kwargs":{"smem_bytes":99584,"validate":false},"threads":192}': 'dispatch_kernel_0114', '{"computed_smem_bytes":0,"constants":[["TOP_K_MAX",32],["SPLIT_COUNT",72],["SPLITS_PER_LANE",3],["ROWS_PER_CTA",1]],"ir_name":"knn_build_rag_microbucket_k32warpmerge_0077_v1_warp_row_merge_k32s72_0077_v1","kwargs":{"smem_bytes":0,"validate":false},"threads":128}': 'dispatch_kernel_0115', '{"computed_smem_bytes":0,"constants":[["TOP_K_MAX",10],["SPLIT_COUNT",72]],"ir_name":"knn_build_evolve_7bfc_k30_merge_s8_rowbase_cache_k10s72_4e09","kwargs":{"smem_bytes":0,"validate":false},"threads":32}': 'dispatch_kernel_0116', '{"computed_smem_bytes":0,"constants":[["TOP_K_MAX",10],["SPLIT_COUNT",8]],"ir_name":"knn_build_evolve_7bfc_k10_merge_s7_rowbase_cache_rect4452_s8","kwargs":{"smem_bytes":0,"validate":false},"threads":32}': 'dispatch_kernel_0117', '{"computed_smem_bytes":50432,"constants":[["BLOCK_Q",128],["BLOCK_M",64],["FEAT_D",128],["TOP_K_MAX",96]],"ir_name":"knn_build_k96_stage1_exact_prefill_q1024_k96over64exactprefillq1024_e5db","kwargs":{"smem_bytes":50432,"validate":false},"threads":192}': 'dispatch_kernel_0120', '{"computed_smem_bytes":0,"constants":[["TOP_K_MAX",96],["SPLIT_COUNT",2]],"ir_name":"knn_build_k96_merge_s8_unordered_chunkprefill_k96over64s2chunkprefill_f9d1","kwargs":{"smem_bytes":0,"validate":false},"threads":32}': 'dispatch_kernel_0121', '{"computed_smem_bytes":0,"constants":[["TOP_K_MAX",96],["SPLIT_COUNT",4]],"ir_name":"knn_build_k96_merge_s8_unordered_chunkprefill_k96over64s4chunkprefill_f9d1","kwargs":{"smem_bytes":0,"validate":false},"threads":32}': 'dispatch_kernel_0122', '{"computed_smem_bytes":20224,"constants":[["BLOCK_Q",64],["BLOCK_M",64],["FEAT_D",64],["TOP_K_MAX",10],["ROWS_COVERED",4]],"ir_name":"knn_build_v12_d64_tail_017a_stage1","kwargs":{"smem_bytes":20224,"validate":false},"threads":96}': 'dispatch_kernel_0123', '{"computed_smem_bytes":50432,"constants":[["FEATURE_CHUNKS",2]],"ir_name":"knn_build_common_d768_build_eeff_m64split_stage1_d256_q128_k10_59fe_v1","kwargs":{"smem_bytes":50432,"validate":false},"threads":192}': 'dispatch_kernel_0124', '{"computed_smem_bytes":34048,"constants":[["FEATURE_CHUNKS",6]],"ir_name":"knn_build_non128_frontier_7ee5_m64rag_stage1_d768_5e7f_highd_v1","kwargs":{"smem_bytes":34048,"validate":false},"threads":96}': 'dispatch_kernel_0125', '{"computed_smem_bytes":50432,"constants":[["FEATURE_CHUNKS",8]],"ir_name":"knn_build_common_d768_build_eeff_m64split_stage1_d1024_be66_search_v1","kwargs":{"smem_bytes":50432,"validate":false},"threads":192}': 'dispatch_kernel_0126', '{"computed_smem_bytes":1024,"constants":[["TOP_K_MAX",10],["GROUP_COUNT",8],["GROUP_SPLITS",8]],"ir_name":"knn_build_non128_frontier_4be7_d768fused_merge_s64g8_4be7_d768fused_v1","kwargs":{"smem_bytes":1024,"validate":false},"threads":32}': 'dispatch_kernel_0127', '{"computed_smem_bytes":50432,"constants":[["FEATURE_CHUNKS",32]],"ir_name":"knn_build_common_d768_build_eeff_m64split_stage1_d4096_be66_search_v1","kwargs":{"smem_bytes":50432,"validate":false},"threads":192}': 'dispatch_kernel_0128', '{"computed_smem_bytes":99584,"constants":[["BLOCK_Q",64],["BLOCK_M",64],["K_TILE",128],["FEATURE_CHUNKS",2],["TOP_K_MAX",32]],"ir_name":"knn_build_v12_d256_k32_tail_59fe_v1_stage1_rowld","kwargs":{"smem_bytes":99584,"validate":false},"threads":192}': 'dispatch_kernel_0129', '{"computed_smem_bytes":0,"constants":[["TOP_K_MAX",32],["SPLIT_COUNT",64],["SPLITS_PER_LANE",2],["ROWS_PER_CTA",1]],"ir_name":"knn_build_rag_microbucket_k32warpmerge_0077_v1_warp_row_merge_k32s64_0077_v1","kwargs":{"smem_bytes":0,"validate":false},"threads":128}': 'dispatch_kernel_0130', '{"computed_smem_bytes":60672,"constants":[["BLOCK_Q_CONST",64],["BLOCK_M_CONST",64],["FEAT_D_CONST",128],["TOP_K_MAX",48],["ROWS_COVERED_CONST",16]],"ir_name":"knn_build_v12_d128_q16_k48_dd2b_v1_stage1","kwargs":{"smem_bytes":60672,"validate":false},"threads":128}': 'dispatch_kernel_0131', '{"computed_smem_bytes":0,"constants":[["TOP_K_MAX",48],["SPLIT_COUNT",144],["SPLITS_PER_LANE",5],["ROWS_PER_CTA",4]],"ir_name":"knn_build_rag_microbucket_k32warpmerge_0077_v1_warp_row_merge_k48s144r4_dd2b_v1","kwargs":{"smem_bytes":0,"validate":false},"threads":128}': 'dispatch_kernel_0132', '{"computed_smem_bytes":0,"constants":[["TOP_K",10],["SPLITS",72],["LANESLOTS",3],["ROWS",4]],"ir_name":"knn_build_rag_stream_k10_s72_warp_row_merge_34da","kwargs":{"smem_bytes":0,"validate":false},"threads":128}': 'dispatch_kernel_0133', '{"computed_smem_bytes":0,"constants":[["TOP_K_MAX",10],["SPLIT_COUNT",64]],"ir_name":"knn_build_evolve_7bfc_k30_merge_s8_rowbase_cache_k10s64_3d97","kwargs":{"smem_bytes":0,"validate":false},"threads":32}': 'dispatch_kernel_0134', '{"computed_smem_bytes":50176,"constants":[["BLOCK_Q",128],["BLOCK_M",64],["FEAT_D",128],["TOP_K_MAX",10]],"ir_name":"knn_build_evolve_7bfc_v1","kwargs":{"smem_bytes":50176,"validate":false},"threads":192}': 'dispatch_kernel_0135', '{"computed_smem_bytes":50432,"constants":[["BLOCK_Q",128],["BLOCK_M",64],["FEAT_D",128],["TOP_K_MAX",10]],"ir_name":"knn_build_evolve_7bfc_split_stage1","kwargs":{"smem_bytes":50432,"validate":false},"threads":192}': 'dispatch_kernel_0136', '{"computed_smem_bytes":50432,"constants":[["BLOCK_Q",128],["BLOCK_M",64],["FEAT_D",128],["TOP_K_MAX",10]],"ir_name":"knn_build_evolve_7bfc_split_cg2_u2_stage1","kwargs":{"smem_bytes":50432,"validate":false},"threads":192}': 'dispatch_kernel_0137', '{"computed_smem_bytes":0,"constants":[["TOP_K_MAX",10],["SPLIT_COUNT",4]],"ir_name":"knn_build_evolve_7bfc_k10_merge_s4","kwargs":{"smem_bytes":0,"validate":false},"threads":256}': 'dispatch_kernel_0138', '{"computed_smem_bytes":0,"constants":[["TOP_K_MAX",10],["SPLIT_COUNT",7]],"ir_name":"knn_build_evolve_7bfc_k10_merge_s7","kwargs":{"smem_bytes":0,"validate":false},"threads":256}': 'dispatch_kernel_0139', '{"computed_smem_bytes":50432,"constants":[["BLOCK_Q",128],["BLOCK_M",64],["FEAT_D",128],["TOP_K_SMALL",5]],"ir_name":"knn_build_evolve_7bfc_split_cg2_stage1_batch8_cond4_vmin_threshold_k5_mintree","kwargs":{"smem_bytes":50432,"validate":false},"threads":192}': 'dispatch_kernel_0140', '{"computed_smem_bytes":0,"constants":[["TOP_K_SMALL",5]],"ir_name":"knn_build_evolve_7bfc_k5_merge_s4_tree_rowbase","kwargs":{"smem_bytes":0,"validate":false},"threads":256}': 'dispatch_kernel_0141', '{"computed_smem_bytes":0,"constants":[["TOP_K_MAX",10],["SPLIT_COUNT",7]],"ir_name":"knn_build_evolve_7bfc_k10_merge_s7_rowbase_cache","kwargs":{"smem_bytes":0,"validate":false},"threads":64}': 'dispatch_kernel_0142', '{"computed_smem_bytes":0,"constants":[["TOP_K_MAX",32],["SPLIT_COUNT",4]],"ir_name":"knn_build_evolve_7bfc_k32_merge_s4_unordered","kwargs":{"smem_bytes":0,"validate":false},"threads":32}': 'dispatch_kernel_0148', '{"computed_smem_bytes":0,"constants":[["TOP_K_MAX",30],["SPLIT_COUNT",8]],"ir_name":"knn_build_evolve_7bfc_k30_merge_s8_rowbase_cache","kwargs":{"smem_bytes":0,"validate":false},"threads":32}': 'dispatch_kernel_0149', '{"computed_smem_bytes":0,"constants":[["TOP_K_MAX",8],["SPLIT_COUNT",7]],"ir_name":"knn_build_evolve_7bfc_k10_merge_s7_rowbase_cache_k8s7","kwargs":{"smem_bytes":0,"validate":false},"threads":32}': 'dispatch_kernel_0152', '{"computed_smem_bytes":50176,"constants":[["BLOCK_Q",128],["BLOCK_M",64],["FEAT_D",128],["TOP_K_MAX",10]],"ir_name":"knn_build_evolve_7bfc_fp16_d128_base","kwargs":{"smem_bytes":50176,"validate":false},"threads":192}': 'dispatch_kernel_0154', '{"computed_smem_bytes":99328,"constants":[["BLOCK_Q",128],["BLOCK_M",64],["TOP_K_MAX",10]],"ir_name":"knn_build_evolve_7bfc_d256_twomma_base","kwargs":{"smem_bytes":99328,"validate":false},"threads":192}': 'dispatch_kernel_0155', '{"computed_smem_bytes":25600,"constants":[["BLOCK_Q",128],["BLOCK_M",64],["TOP_K_MAX",10]],"ir_name":"knn_build_evolve_7bfc_d64_tcgen05_base","kwargs":{"smem_bytes":25600,"validate":false},"threads":192}': 'dispatch_kernel_0156', '{"computed_smem_bytes":0,"constants":[["TOP_K_MAX",20],["SPLIT_COUNT",4]],"ir_name":"knn_build_evolve_7bfc_k20_merge_s4_unordered_warp_select_splitmajor","kwargs":{"smem_bytes":0,"validate":false},"threads":128}': 'dispatch_kernel_0157', '{"computed_smem_bytes":0,"constants":[["TOP_K_MAX",20],["SPLIT_COUNT",4]],"ir_name":"knn_build_k20_mergeown_08ec_s4_rowbase_lane","kwargs":{"smem_bytes":0,"validate":false},"threads":32}': 'dispatch_kernel_0159', '{"computed_smem_bytes":0,"constants":[["TOP_K_MAX",20]],"ir_name":"knn_build_k20_large_rect_s3_warp_select","kwargs":{"smem_bytes":0,"validate":false},"threads":128}': 'dispatch_kernel_0160', '{"computed_smem_bytes":50432,"constants":[["BLOCK_Q",128],["BLOCK_M",64],["FEAT_D",128],["TOP_K_MAX",32]],"ir_name":"knn_build_rag_frontier_b6d4_stage1_k32_sort4earlystop","kwargs":{"smem_bytes":50432,"validate":false},"threads":192}': 'dispatch_kernel_0162', '{"computed_smem_bytes":2048,"constants":[["TOP_K_MAX",32],["GROUP_COUNT",8],["GROUP_SPLITS",9]],"ir_name":"knn_build_rag_frontier_7399_k32_fused_group_final_merge","kwargs":{"smem_bytes":2048,"validate":false},"threads":32}': 'dispatch_kernel_0163', '{"computed_smem_bytes":0,"constants":[["TOP_K_MAX",32],["SPLIT_COUNT",32]],"ir_name":"knn_build_evolve_7bfc_k30_merge_s8_rowbase_cache_k32s32_4b5c","kwargs":{"smem_bytes":0,"validate":false},"threads":32}': 'dispatch_kernel_0164', '{"computed_smem_bytes":50432,"constants":[["BLOCK_Q",128],["BLOCK_M",64],["FEAT_D",128],["TOP_K_MAX",96]],"ir_name":"knn_build_evolve_7bfc_split_cg2_u2_stage1_k32_unordered_k96over64","kwargs":{"smem_bytes":50432,"validate":false},"threads":192}': 'dispatch_kernel_0165', '{"computed_smem_bytes":0,"constants":[["TOP_K_MAX",96],["SPLIT_COUNT",8]],"ir_name":"knn_build_k96_merge_s8_unordered_chunkprefill_k96over64s8chunkprefill","kwargs":{"smem_bytes":0,"validate":false},"threads":32}': 'dispatch_kernel_0166', '{"computed_smem_bytes":50432,"constants":[["BLOCK_Q",128],["BLOCK_M",64],["FEAT_D",128],["TOP_K_MAX",32]],"ir_name":"knn_build_rag_frontier_b6d4_stage1_k32_chunked","kwargs":{"smem_bytes":50432,"validate":false},"threads":192}': 'dispatch_kernel_0167', '{"computed_smem_bytes":50432,"constants":[["BLOCK_Q",128],["BLOCK_M",64],["FEAT_D",128],["TOP_K_MAX",32]],"ir_name":"knn_build_rag_frontier_4fbf_v7_stage1_k32_sort4earlystop_tailinf","kwargs":{"smem_bytes":50432,"validate":false},"threads":192}': 'dispatch_kernel_0176', '{"computed_smem_bytes":50432,"constants":[["BLOCK_Q",128],["BLOCK_M",64],["FEAT_D",128],["TOP_K_MAX",32]],"ir_name":"knn_build_rag_frontier_4fbf_stage1_k32_sort4earlystop_tailinf","kwargs":{"smem_bytes":50432,"validate":false},"threads":192}': 'dispatch_kernel_0177', '{"computed_smem_bytes":50432,"constants":[["BLOCK_Q",128],["BLOCK_M",64],["FEAT_D",128],["TOP_K_MAX",24]],"ir_name":"knn_build_evolve_7bfc_split_cg2_u2_stage1_bad5k24s8","kwargs":{"smem_bytes":50432,"validate":false},"threads":192}': 'dispatch_kernel_0178', '{"computed_smem_bytes":50432,"constants":[["BLOCK_Q",128],["BLOCK_M",64],["FEAT_D",128],["TOP_K_MAX",28]],"ir_name":"knn_build_evolve_7bfc_split_cg2_u2_stage1_bad5k28s8","kwargs":{"smem_bytes":50432,"validate":false},"threads":192}': 'dispatch_kernel_0179', '{"computed_smem_bytes":0,"constants":[["TOP_K_MAX",24],["SPLIT_COUNT",8]],"ir_name":"knn_build_evolve_7bfc_k30_merge_s8_rowbase_cache_bad5k24s8","kwargs":{"smem_bytes":0,"validate":false},"threads":32}': 'dispatch_kernel_0180', '{"computed_smem_bytes":0,"constants":[["TOP_K_MAX",28],["SPLIT_COUNT",8]],"ir_name":"knn_build_evolve_7bfc_k30_merge_s8_rowbase_cache_bad5k28s8","kwargs":{"smem_bytes":0,"validate":false},"threads":32}': 'dispatch_kernel_0181', '{"computed_smem_bytes":0,"constants":[["TOP_K_MAX",32]],"ir_name":"knn_build_large_square_k32_s2_warp_select","kwargs":{"smem_bytes":0,"validate":false},"threads":128}': 'dispatch_kernel_0182', '{"computed_smem_bytes":0,"constants":[["TOP_K_MAX",10],["SPLIT_COUNT",16]],"ir_name":"knn_build_rect_d64_cf49_s16_cached_merge","kwargs":{"smem_bytes":0,"validate":false},"threads":64}': 'dispatch_kernel_0183', '{"computed_smem_bytes":512,"constants":[["TOP_K_MAX",10],["SPLIT_COUNT",72],["MERGE_GROUPS",4],["SPLITS_PER_GROUP",18]],"ir_name":"knn_build_ragonline_mbucket_aa88_q1m_s72_k10_coop_merge","kwargs":{"smem_bytes":512,"validate":false},"threads":128}': 'dispatch_kernel_0184', '{"computed_smem_bytes":512,"constants":[["TOP_K_MAX",10],["SPLIT_COUNT",74],["MERGE_GROUPS",4],["SPLITS_PER_GROUP",19]],"ir_name":"knn_build_ragonline_mbucket_aa88_q1m_s72_k10_coop_merge_s74_m250","kwargs":{"smem_bytes":512,"validate":false},"threads":128}': 'dispatch_kernel_0185', '{"computed_smem_bytes":50432,"constants":[["BLOCK_Q",128],["BLOCK_M",64],["FEAT_D",128],["TOP_K_MAX",32]],"ir_name":"knn_build_rag_microbucket_5093_v1_stage1_k32_tailinf_cta1_compactwarp","kwargs":{"smem_bytes":50432,"validate":false},"threads":96}': 'dispatch_kernel_0186', '{"computed_smem_bytes":50432,"constants":[["BLOCK_Q",128],["BLOCK_M",64],["FEAT_D",128],["TOP_K_MAX",32]],"ir_name":"knn_build_rag_microbucket_3505_v3_stage1_k32_tailinf_cta1","kwargs":{"smem_bytes":50432,"validate":false},"threads":192}': 'dispatch_kernel_0187', '{"computed_smem_bytes":0,"constants":[["TOP_K_MAX",10],["SPLIT_COUNT",16]],"ir_name":"knn_build_rect_d64_cf49_s16_cached_merge","kwargs":{"smem_bytes":0,"validate":false},"threads":8}': 'dispatch_kernel_0188', '{"computed_smem_bytes":50432,"constants":[["BLOCK_Q",128],["BLOCK_M",64],["FEAT_D",128],["TOP_K_MAX",96]],"ir_name":"knn_build_k96_stage1_sort4_chunked_k96over64sort4chunked","kwargs":{"smem_bytes":50432,"validate":false},"threads":192}': 'dispatch_kernel_0189', '{"computed_smem_bytes":34048,"constants":[["BLOCK_Q",64],["BLOCK_M",64],["FEAT_D",128],["TOP_K_MAX",32]],"ir_name":"knn_build_rag_microbucket_3505_v9_stage1_q8_k32_m64","kwargs":{"smem_bytes":34048,"validate":false},"threads":96}': 'dispatch_kernel_0191', '{"computed_smem_bytes":0,"constants":[["TOP_K_MAX",10],["SPLIT_COUNT",74]],"ir_name":"knn_build_d128_rag_q128_k10_s74_warp_merge","kwargs":{"smem_bytes":0,"validate":false},"threads":128}': 'dispatch_kernel_0192', '{"computed_smem_bytes":0,"constants":[["TOP_K_MAX",11]],"ir_name":"knn_build_evolve_7bfc_k32_merge_s4_rowbase_cache_e080k11s4exact","kwargs":{"smem_bytes":0,"validate":false},"threads":32}': 'dispatch_kernel_0194', '{"computed_smem_bytes":0,"constants":[["TOP_K_MAX",13]],"ir_name":"knn_build_evolve_7bfc_k32_merge_s4_rowbase_cache_e080k13s4exact","kwargs":{"smem_bytes":0,"validate":false},"threads":32}': 'dispatch_kernel_0195', '{"computed_smem_bytes":50432,"constants":[["BLOCK_Q",64],["BLOCK_M",64],["FEAT_D",128],["TOP_K_MAX",32],["ROWS_COVERED",16]],"ir_name":"knn_build_rag_microbucket_k32rowld1warp_0077_v1_stage1_q16_k32_m64_rowld1_q16rowld1_0077_v1","kwargs":{"smem_bytes":50432,"validate":false},"threads":96}': 'dispatch_kernel_0199', '{"computed_smem_bytes":66816,"constants":[["BLOCK_Q",64],["BLOCK_M",64],["FEAT_D",128],["TOP_K_MAX",32],["ROWS_COVERED",32]],"ir_name":"knn_build_rag_microbucket_k32rowld1warp_0077_v1_stage1_q32_k32_m64_rowld2_q32rowld2_0077_v1","kwargs":{"smem_bytes":66816,"validate":false},"threads":128}': 'dispatch_kernel_0200', '{"computed_smem_bytes":50432,"constants":[["BLOCK_Q",128],["BLOCK_M",64],["FEAT_D",128],["TOP_K_MAX",96]],"ir_name":"knn_build_k96_stage1_sort4_prefill_q1024_k96over64sort4prefillq1024_8c56","kwargs":{"smem_bytes":50432,"validate":false},"threads":192}': 'dispatch_kernel_0202', '{"computed_smem_bytes":148736,"constants":[["BLOCK_Q",128],["BLOCK_M",64],["TOP_K_MAX",10]],"ir_name":"knn_build_non128_frontier_8199_d384_stage1","kwargs":{"smem_bytes":148736,"validate":false},"threads":192}': 'dispatch_kernel_0203', '{"computed_smem_bytes":97536,"constants":[["BLOCK_Q",64],["BLOCK_M",64],["FEAT_D",128],["TOP_K_MAX",31],["ROWS_COVERED",32]],"ir_name":"knn_build_rag_microbucket_q32_k31_c3d2_v1_stage1_q32k31_c3d2_v1","kwargs":{"smem_bytes":97536,"validate":false},"threads":128}': 'dispatch_kernel_0204', '{"computed_smem_bytes":66816,"constants":[["BLOCK_Q",64],["BLOCK_M",64],["FEAT_D",128],["TOP_K_MAX",32],["ROWS_COVERED",32]],"ir_name":"knn_build_rag_microbucket_k32_f590_q32exact_v1_stage1_q32exact_f590_v1","kwargs":{"smem_bytes":66816,"validate":false},"threads":128}': 'dispatch_kernel_0205', '{"computed_smem_bytes":58624,"constants":[["BLOCK_Q",64],["BLOCK_M",64],["FEAT_D",128],["TOP_K_MAX",12]],"ir_name":"knn_build_rag_microbucket_k12_2f22_q48exact_v1_stage1","kwargs":{"smem_bytes":58624,"validate":false},"threads":192}': 'dispatch_kernel_0206', '{"computed_smem_bytes":66816,"constants":[["BLOCK_Q",64],["BLOCK_M",64],["FEAT_D",128],["TOP_K_MAX",32],["ROWS_COVERED",32]],"ir_name":"knn_build_rag_microbucket_k32_q32rowld2uneven_f653_v1_stage1_q32rowld2uneven_f653_v1","kwargs":{"smem_bytes":66816,"validate":false},"threads":128}': 'dispatch_kernel_0207', '{"computed_smem_bytes":66816,"constants":[["BLOCK_Q",64],["BLOCK_M",64],["FEAT_D",128],["TOP_K_MAX",32],["ROWS_COVERED",32]],"ir_name":"knn_build_rag_microbucket_k32rowld1warp_0077_v1_stage1_q32_k32_m64_rowld2_q32rowld2_f653_v1","kwargs":{"smem_bytes":66816,"validate":false},"threads":128}': 'dispatch_kernel_0208', '{"computed_smem_bytes":66816,"constants":[["BLOCK_Q",64],["BLOCK_M",64],["FEAT_D",128],["TOP_K_MAX",32],["ROWS_COVERED",31]],"ir_name":"knn_build_rag_microbucket_k32_0cb5_q31tail_v2_stage1_q31exact_0cb5_v2","kwargs":{"smem_bytes":66816,"validate":false},"threads":128}': 'dispatch_kernel_0209', '{"computed_smem_bytes":52480,"constants":[["BLOCK_Q",64],["BLOCK_M",64],["FEAT_D",128],["TOP_K_MAX",32],["ROWS_COVERED",16]],"ir_name":"knn_build_rag_microbucket_k32_q16irreg2warp_a444_v2_stage1_q16_rowld1_2warp_q16irreg2warp_a444_v2","kwargs":{"smem_bytes":52480,"validate":false},"threads":128}': 'dispatch_kernel_0210', '{"computed_smem_bytes":99584,"constants":[["BLOCK_Q",64],["BLOCK_M",64],["FEAT_D",128],["TOP_K_MAX",32]],"ir_name":"knn_build_rag_microbucket_q32rowld_e5db_v1_stage1_q32_k32_m64_q128rowld_60fb_v1","kwargs":{"smem_bytes":99584,"validate":false},"threads":192}': 'dispatch_kernel_0211', '{"computed_smem_bytes":17664,"constants":[],"ir_name":"knn_build_common_d_5e7f_rag_d64_m64_stage1","kwargs":{"smem_bytes":17664,"validate":false},"threads":96}': 'dispatch_kernel_0212', '{"computed_smem_bytes":50432,"constants":[["BLOCK_Q",128],["BLOCK_M",64],["FEAT_D",128],["TOP_K_MAX",32]],"ir_name":"knn_build_rag_microbucket_3505_v2_stage1_k32_tailinf_cta1","kwargs":{"smem_bytes":50432,"validate":false},"threads":192}': 'dispatch_kernel_0213', '{"computed_smem_bytes":50432,"constants":[["BLOCK_Q",128],["BLOCK_M",64],["FEAT_D",128],["TOP_K_MAX",8]],"ir_name":"knn_build_evolve_7bfc_split_cg2_u2_stage1_k32_unordered_fd9b_k8unordered","kwargs":{"smem_bytes":50432,"validate":false},"threads":192}': 'dispatch_kernel_0220', '{"computed_smem_bytes":0,"constants":[["TOP_K_MAX",10],["SPLIT_COUNT",8],["GROUP_COUNT",4],["GROUP_SPLITS",2],["GROUPS_PER_CTA",4]],"ir_name":"knn_build_d64_q4096_c271_twostage_group_reduce","kwargs":{"smem_bytes":0,"validate":false},"threads":128}': 'dispatch_kernel_0221', '{"computed_smem_bytes":0,"constants":[["TOP_K_MAX",10],["SPLIT_COUNT",5]],"ir_name":"knn_build_d64_build_aa88_k10_merge_s8_rowbase_cache_c271_s5","kwargs":{"smem_bytes":0,"validate":false},"threads":32}': 'dispatch_kernel_0222', '{"computed_smem_bytes":0,"constants":[["TOP_K_MAX",10],["SPLIT_COUNT",6]],"ir_name":"knn_build_d64_build_aa88_k10_merge_s8_rowbase_cache_c271_s6","kwargs":{"smem_bytes":0,"validate":false},"threads":32}': 'dispatch_kernel_0223', '{"computed_smem_bytes":25856,"constants":[["BLOCK_Q",128],["BLOCK_M",64],["TOP_K_MAX",10]],"ir_name":"knn_build_d64_q4096_c271_stage1_syncdrop","kwargs":{"smem_bytes":25856,"validate":false},"threads":192}': 'dispatch_kernel_0224', '{"computed_smem_bytes":20480,"constants":[["K_MAX_",10],["THREADS_",256]],"ir_name":"knn_build_common_d_generic_direct_v1","kwargs":{"smem_bytes":20480,"validate":false},"threads":256}': 'dispatch_kernel_0227', '{"computed_smem_bytes":17664,"constants":[["BLOCK_Q",64],["BLOCK_M",64],["K_TILE",64],["TOP_K_MAX",10]],"ir_name":"knn_build_common_d_5e7f_rag_d64_repair_stage1","kwargs":{"smem_bytes":17664,"validate":false},"threads":96}': 'dispatch_kernel_0229', '{"computed_smem_bytes":66816,"constants":[["BLOCK_Q",64],["BLOCK_M",64],["FEAT_D",128],["TOP_K_MAX",32],["ROWS_COVERED",32],["SPLIT_COUNT_CONST",141],["NUM_DB_TILES_CONST",1563],["TILES_FLOOR_CONST",11],["EXTRA_SPLITS_CONST",12],["DB_TILES_PER_SPLIT_CONST",12],["M_LIMIT",100000]],"ir_name":"knn_build_rag_microbucket_k32_q32rowld2exact_f653_v1_stage1_q32rowld2exact_f653_v1","kwargs":{"smem_bytes":66816,"validate":false},"threads":128}': 'dispatch_kernel_0230', '{"computed_smem_bytes":0,"constants":[["TOP_K_MAX",96],["SPLIT_COUNT",2]],"ir_name":"knn_build_k96_merge_s2_unordered_warp_select","kwargs":{"smem_bytes":0,"validate":false},"threads":128}': 'dispatch_kernel_0231', '{"computed_smem_bytes":0,"constants":[["TOP_K_MAX",10],["GROUP_COUNT",21],["GROUP_SPLITS",7]],"ir_name":"knn_build_q1m524_workfeed_s147_g21_register_merge","kwargs":{"smem_bytes":0,"validate":false},"threads":32}': 'dispatch_kernel_0232'}

import ctypes
import importlib
import json
import threading
import sys
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass, field, replace as _dataclass_replace
from importlib import resources
from types import SimpleNamespace

from .kernels import get_kernel
from ._runtime import launch_stream_context, resolve_launch_defaults


_DISPATCH_LAUNCH_OPTIONS = ContextVar("dispatch_launch_options", default=(None, None))


@contextmanager
def dispatch_launch_options(*, stream=None, timeout_ms=None):
    token = _DISPATCH_LAUNCH_OPTIONS.set((stream, timeout_ms))
    try:
        yield
    finally:
        _DISPATCH_LAUNCH_OPTIONS.reset(token)


def _resolved_launch_options(stream, timeout_ms):
    default_stream, default_timeout_ms = _DISPATCH_LAUNCH_OPTIONS.get()
    return (
        default_stream if stream is None else stream,
        default_timeout_ms if timeout_ms is None else timeout_ms,
    )


_active_launch_capture = ContextVar("flashlib_active_launch_capture", default=None)
_pending_tensor_map_recipe = ContextVar("flashlib_pending_tensor_map_recipe", default=None)
_launch_capture_prepare_lock = threading.RLock()


def _replace(value, /, **changes):
    replacer = getattr(value, "__replace__", None)
    if callable(replacer):
        return replacer(**changes)
    return _dataclass_replace(value, **changes)


dc = SimpleNamespace(replace=_replace)


def _import_dispatch_module(short_name):
    return importlib.import_module(f"{__package__}._dispatch.{short_name}")


_DISPATCH_OWNED_DICT_SUFFIXES = ("CACHE", "SCRATCH", "INPUTS", "OUTPUTS", "FLAGS")


def _cache_value_references_owned_object(value, owned_ids, seen):
    identity = id(value)
    if identity in owned_ids:
        return True
    if identity in seen:
        return False
    seen.add(identity)
    if isinstance(value, dict):
        return any(
            _cache_value_references_owned_object(item, owned_ids, seen)
            for pair in value.items()
            for item in pair
        )
    if isinstance(value, (tuple, list, set, frozenset)):
        return any(
            _cache_value_references_owned_object(item, owned_ids, seen)
            for item in value
        )
    return False


def release_dispatch_caches(owned_objects):
    '''Clear route-owned tensor dictionaries after a prepared sequence binds.

    Generated dispatch modules may temporarily cache tensor-map descriptors and
    workspaces while a route is prepared.  A bound ``PreparedKernelSequence``
    retains every CUDA argument, so those module globals are no longer owners.
    To avoid clearing dispatch registries or scalar statistics, this contract
    is limited to dict-valued, private, uppercase names with an explicit
    workspace/cache suffix below this generated package's ``_dispatch``
    namespace.
    '''

    prefix = f"{__package__}._dispatch."
    owned_ids = {
        id(value)
        for value in owned_objects
        if callable(getattr(value, "data_ptr", None))
    }
    if not owned_ids:
        return 0
    cleared = 0
    for module_name, module in tuple(sys.modules.items()):
        if module is None or not module_name.startswith(prefix):
            continue
        for name, value in tuple(vars(module).items()):
            if (
                name.startswith("_")
                and name.endswith(_DISPATCH_OWNED_DICT_SUFFIXES)
                and name.isupper()
                and isinstance(value, dict)
            ):
                removed = False
                for key, item in tuple(value.items()):
                    if _cache_value_references_owned_object(item, owned_ids, set()):
                        value.pop(key, None)
                        removed = True
                cleared += int(removed)
    return cleared


def _decode_capture(value):
    if isinstance(value, dict) and "__ir__" in value:
        return _ir_proxy(
            value["__ir__"],
            value.get("threads", 256),
            value.get("computed_smem_bytes", 0),
            value.get("cluster_dims", (1, 1, 1)),
            value.get("cta_group", 1),
            value.get("constants", ()),
            value.get("arg_keys", ()),
        )
    if isinstance(value, dict) and set(value) == {"__kernel__"}:
        return DispatchKernel(value["__kernel__"])
    if isinstance(value, dict) and set(value) == {"__kernel_source__"}:
        return value["__kernel_source__"]
    if isinstance(value, dict) and set(value) == {"__tuple__"}:
        return tuple(_decode_capture(item) for item in value["__tuple__"])
    if isinstance(value, dict) and set(value) == {"__dict_items__"}:
        return {
            _decode_capture(key): _decode_capture(item)
            for key, item in value["__dict_items__"]
        }
    if isinstance(value, dict):
        return {key: _decode_capture(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_decode_capture(item) for item in value]
    return value


@dataclass(frozen=True)
class _IRProxy:
    symbol: str
    threads: int = 256
    computed_smem_bytes: int = 0
    constants: tuple = ()
    grid: object = None
    arg_keys: tuple = ()

    def __replace__(self, /, **changes):
        values = {
            "symbol": self.symbol,
            "threads": self.threads,
            "computed_smem_bytes": self.computed_smem_bytes,
            "constants": self.constants,
            "grid": self.grid,
            "arg_keys": self.arg_keys,
        }
        unknown = sorted(set(changes) - set(values))
        if unknown:
            raise TypeError(f"unknown frozen WeaveIR field(s): {unknown}")
        values.update(changes)
        return _IRProxy(**values)


def _ir_proxy(
    name, threads=256, computed_smem_bytes=0, cluster_dims=(1, 1, 1),
    cta_group=1, constants=(), arg_keys=(),
):
    return _IRProxy(
        name.rpartition(":")[2], int(threads), int(computed_smem_bytes),
        tuple(tuple(item) for item in constants),
        SimpleNamespace(cluster_dims=tuple(cluster_dims), cta_group=int(cta_group)),
        tuple(arg_keys),
    )


def pack_kernel_args(schedule, /, **bindings):
    expected = tuple(schedule.arg_keys)
    missing = sorted(set(expected) - set(bindings))
    unexpected = sorted(set(bindings) - set(expected))
    if missing or unexpected:
        raise ValueError(
            f"kernel argument bindings do not match frozen WeaveIR.args: "
            f"missing={missing!r}, unexpected={unexpected!r}"
        )
    return [bindings[key] for key in expected]


class PreparedKernelSequence:
    def __init__(
        self,
        launches,
        result,
        input_bindings=(),
        result_template=None,
        tensor_map_bindings=(),
        input_alias_topology=(),
        stream=None,
    ):
        if not launches:
            raise RuntimeError("prepared semantic route did not capture a CUDA launch")
        self._launches = tuple(launches)
        self._result = result
        self._input_bindings = tuple(tuple(bindings) for bindings in input_bindings)
        if self._input_bindings and len(self._input_bindings) != len(self._launches):
            raise RuntimeError("prepared semantic route has corrupt input bindings")
        self._result_template = result_template
        self._input_alias_topology = tuple(
            tuple(group) for group in input_alias_topology
        )
        direct_input_keys = {key for bindings in self._input_bindings for _, key in bindings}
        self._direct_input_keys = tuple(sorted(direct_input_keys))
        self._tensor_map_bindings = _own_tensor_map_bindings(
            self._launches,
            tuple(tensor_map_bindings),
            stream=stream,
        )
        self._bound_input_keys = tuple(
            sorted(direct_input_keys | {binding.input_key for binding in self._tensor_map_bindings})
        )
        self._input_references_retained = True

    @property
    def launch_count(self):
        return len(self._launches)

    @property
    def bound_input_keys(self):
        return self._bound_input_keys

    def rebind_inputs(
        self,
        inputs,
        *,
        stream=None,
        materialize_result=True,
        preserve_prepared_stream=False,
        retain_input_references=True,
    ):
        if not isinstance(materialize_result, bool):
            raise TypeError("materialize_result must be a bool")
        if not isinstance(preserve_prepared_stream, bool):
            raise TypeError("preserve_prepared_stream must be a bool")
        if not isinstance(retain_input_references, bool):
            raise TypeError("retain_input_references must be a bool")
        if preserve_prepared_stream and stream is not None:
            raise ValueError("preserve_prepared_stream requires stream=None")
        if materialize_result and not retain_input_references:
            raise ValueError(
                "materialize_result requires retain_input_references so the result remains valid"
            )
        if not any(self._input_bindings) and not self._tensor_map_bindings:
            raise RuntimeError(
                "prepared semantic route has no input bindings; "
                "capture it with capture_kernel_launches(inputs=...)"
            )
        missing = sorted(set(self.bound_input_keys) - set(inputs))
        if missing:
            raise KeyError(f"missing prepared semantic input binding(s): {missing!r}")
        _validate_public_tensor_alias_topology(inputs, self._input_alias_topology)
        pointer_values = None
        inputs_already_scrubbed = False
        if not retain_input_references:
            pointer_values = {}
            for key in self._direct_input_keys:
                value = inputs[key]
                data_ptr = getattr(value, "data_ptr", None)
                if not callable(data_ptr):
                    raise TypeError(f"prepared CUDA tensor binding {key!r} is not tensor-like")
                pointer_values[key] = int(data_ptr())
            inputs_already_scrubbed = not self._input_references_retained
        with launch_stream_context(stream):
            for binding in self._tensor_map_bindings:
                binding.refresh(inputs[binding.input_key])
            for launch, bindings in zip(self._launches, self._input_bindings, strict=True):
                launch.rebind_tensor_arguments(
                    bindings,
                    inputs,
                    stream=stream,
                    preserve_stream=preserve_prepared_stream,
                    retain_inputs=retain_input_references,
                    pointer_values=pointer_values,
                    inputs_already_scrubbed=inputs_already_scrubbed,
                )
        self._input_references_retained = retain_input_references
        # Stateful public runtimes may own the output independently of the
        # semantic return tree.  Let those callers skip recursively rebuilding
        # a result they will not observe while retaining the default behavior
        # for normal prepared-dispatch callers.
        if materialize_result and self._result_template is not None:
            self._result = _materialize_result_template(self._result_template, inputs)
        return self

    def _rebind_stream_bound_scrubbed_inputs(self, inputs, *, stream):
        '''Rebind one validated fixed-stream runtime slot without generic checks.

        This private path is valid only after a stateful wrapper selected the
        sequence through a cache key containing the complete public pointer
        alias topology, recorded every caller-owned tensor, and scrubbed the
        sequence's caller references. Public prepared callers continue to use
        :meth:`rebind_inputs` and its full validation.
        '''
        if self._input_references_retained:
            raise RuntimeError(
                "fixed-stream semantic rebind requires scrubbed input references"
            )
        if stream is None:
            raise ValueError("fixed-stream semantic rebind requires an explicit stream")
        pointer_values = {}
        for key in self._direct_input_keys:
            value = inputs[key]
            data_ptr = getattr(value, "data_ptr", None)
            if not callable(data_ptr):
                raise TypeError(f"prepared CUDA tensor binding {key!r} is not tensor-like")
            pointer_values[key] = int(data_ptr())
        for binding in self._tensor_map_bindings:
            binding.rebind_stream_bound(inputs[binding.input_key], stream=stream)
        for launch, bindings in zip(self._launches, self._input_bindings, strict=True):
            launch.rebind_tensor_arguments(
                bindings,
                inputs,
                preserve_stream=True,
                retain_inputs=False,
                pointer_values=pointer_values,
                inputs_already_scrubbed=True,
            )
        return self

    def release_bound_inputs(self):
        '''Drop caller tensor references after their launch stream was recorded.'''
        if self._input_references_retained:
            for launch, bindings in zip(self._launches, self._input_bindings, strict=True):
                keepalive = list(launch._keepalive)
                for index, _key in bindings:
                    value = keepalive[index]
                    data_ptr = getattr(value, "data_ptr", None)
                    if callable(data_ptr):
                        keepalive[index] = int(data_ptr())
                launch._keepalive = tuple(keepalive)
            self._input_references_retained = False
        self._result = None

    def record_stream(self, stream):
        '''Tie every tensor launch argument, including private scratch, to a stream.'''
        if stream is None:
            raise ValueError("prepared semantic record_stream requires an explicit stream")
        seen = set()
        for launch in self._launches:
            for value in launch._keepalive:
                identity = id(value)
                record_stream = getattr(value, "record_stream", None)
                if identity not in seen and callable(record_stream):
                    seen.add(identity)
                    record_stream(stream)
        # Variant-bank descriptor tensors are slot-owned but only the active
        # variant appears in a launch keepalive; record every variant so a
        # non-synchronizing release stays allocator-safe.
        for binding in self._tensor_map_bindings:
            for value in binding.variants.values():
                identity = id(value)
                record_stream = getattr(value, "record_stream", None)
                if identity not in seen and callable(record_stream):
                    seen.add(identity)
                    record_stream(stream)

    def _finish_rebind(self, result):
        self._result = result
        return self

    def __call__(self, _inputs=None, *, stream=None, timeout_ms=None):
        last = len(self._launches) - 1
        for index, launch in enumerate(self._launches):
            launch.launch(stream=stream, timeout_ms=timeout_ms if index == last else None)
        return self._result


class KernelLaunchCapture:
    def __init__(self, *, stream=None, arch=None, inputs=None, rebind=None):
        if rebind is not None and not isinstance(rebind, PreparedKernelSequence):
            raise TypeError("rebind must be a PreparedKernelSequence")
        if inputs is not None and rebind is not None:
            raise ValueError("inputs and rebind are mutually exclusive capture modes")
        if rebind is not None:
            raise RuntimeError(
                "capture(rebind=...) is unsupported because an in-place topology "
                "update cannot be transactional; capture a new sequence instead"
            )
        self.stream = stream
        self.arch = arch
        self._launches = []
        self._input_bindings = []
        self._input_key_by_identity = _public_tensor_input_identities(inputs)
        self._input_key_by_pointer = _public_tensor_input_pointers(inputs)
        self._input_alias_topology = _public_tensor_alias_topology(inputs)
        self._tensor_map_bindings = {}
        self._route_caches_released = False
        self._rebind = rebind
        self._rebind_index = 0
        self.host_data_reads = 0

    @property
    def host_data_dependent(self):
        '''True when the route read device memory while its kernels were only
        being recorded — its host branch decisions cannot be frozen.'''
        return self.host_data_reads > 0

    @property
    def rebinding(self):
        return self._rebind is not None

    def add(self, launch):
        if self.rebinding:
            raise RuntimeError("rebind capture requires launch topology, not a newly prepared launch")
        self._launches.append(launch)
        self._input_bindings.append(
            ()
            if not self._input_key_by_identity
            else tuple(
                (index, self._input_key_by_identity[id(arg)])
                for index, arg in enumerate(launch._keepalive)
                if id(arg) in self._input_key_by_identity
            )
        )
        for arg in launch._keepalive:
            recipe = getattr(arg, "_loom_tensor_map_recipe", None)
            if recipe is None:
                continue
            source_pointer = int(recipe[2])
            input_key = self._input_key_by_pointer.get(source_pointer)
            if input_key is None:
                continue
            self._tensor_map_bindings.setdefault(
                id(arg),
                _TensorMapBinding(
                    input_key=input_key,
                    tensor=arg,
                    recipe=tuple(recipe),
                    pointer=source_pointer,
                ),
            )

    def add_kernel_launch(
        self,
        exported,
        *,
        mode,
        grid,
        block,
        args,
        arg_types,
        shared_mem,
        stream,
        cluster_dims=None,
    ):
        resolved_arch, resolved_stream, _ = resolve_launch_defaults(
            arch=self.arch,
            stream=self.stream if self.stream is not None else stream,
            timeout_ms=None,
        )
        with launch_stream_context(resolved_stream):
            kernel = exported.compile(arch=resolved_arch, options=["--use_fast_math"])
            kwargs = {
                "grid": grid,
                "block": block,
                "args": tuple(args),
                "arg_types": arg_types,
                "shared_mem": shared_mem,
                "stream": resolved_stream,
            }
            if self.rebinding:
                if self._rebind_index >= self._rebind.launch_count:
                    raise RuntimeError(
                        "prepared semantic route launch-count mismatch: "
                        f"expected {self._rebind.launch_count}, captured more launches"
                    )
                prepared = self._rebind._launches[self._rebind_index]
                if mode == "cluster":
                    kernel.rebind_launch_cluster(
                        prepared, cluster_dims=cluster_dims, **kwargs
                    )
                elif mode == "cooperative":
                    kernel.rebind_launch_cooperative(prepared, **kwargs)
                elif mode == "regular":
                    kernel.rebind_launch(prepared, **kwargs)
                else:
                    raise RuntimeError(f"unsupported captured launch mode: {mode!r}")
                self._rebind_index += 1
                return
            if mode == "cluster":
                prepared = kernel.prepare_launch_cluster(
                    cluster_dims=cluster_dims, **kwargs
                )
            elif mode == "cooperative":
                prepared = kernel.prepare_launch_cooperative(**kwargs)
            elif mode == "regular":
                prepared = kernel.prepare_launch(**kwargs)
            else:
                raise RuntimeError(f"unsupported captured launch mode: {mode!r}")
            self.add(prepared)

    def bind(self, result):
        if self.rebinding:
            if self._rebind_index != self._rebind.launch_count:
                raise RuntimeError(
                    "prepared semantic route launch-count mismatch: "
                    f"expected {self._rebind.launch_count}, captured {self._rebind_index}"
                )
            return self._rebind._finish_rebind(result)
        result_template = _capture_result_template(result, self._input_key_by_identity)
        route_cache_owned_objects = self._route_cache_owned_objects()
        sequence = PreparedKernelSequence(
            self._launches,
            result,
            self._input_bindings,
            result_template,
            tuple(self._tensor_map_bindings.values()),
            self._input_alias_topology,
            self.stream,
        )
        self.release_route_caches(route_cache_owned_objects)
        return sequence

    def _route_cache_owned_objects(self):
        return tuple(arg for launch in self._launches for arg in launch._keepalive)

    def release_route_caches(self, owned_objects=None):
        if self._route_caches_released:
            return 0
        if owned_objects is None:
            owned_objects = self._route_cache_owned_objects()
        self._route_caches_released = True
        return release_dispatch_caches(tuple(owned_objects))


@dataclass(frozen=True)
class _BoundInputResult:
    key: str


@dataclass
class _TensorMapBinding:
    input_key: str
    tensor: object
    recipe: tuple
    pointer: int
    pointer_carriers: tuple = ()
    staging_slots: list = field(default_factory=list)
    variants: dict = field(default_factory=dict)
    variant_capacity: int = 4

    def __post_init__(self):
        if not self.variants:
            self.variants[self.pointer] = self.tensor

    def _acquire_staging_slot(self, torch):
        for slot in self.staging_slots:
            if slot.event.query():
                return slot
        slot = _TensorMapStagingSlot(
            host_buffer=torch.empty(128, dtype=torch.uint8, pin_memory=True),
            event=torch.cuda.Event(blocking=False, interprocess=False),
        )
        self.staging_slots.append(slot)
        return slot

    def _encode_into(self, pointer, tensor, *, stream=None):
        from cuda.bindings import driver
        import torch

        arguments = list(self.recipe)
        arguments[2] = pointer
        err, tmap = driver.cuTensorMapEncodeTiled(*arguments)
        if err != 0:
            raise RuntimeError(f"cuTensorMapEncodeTiled rebind failed: CUresult={err}")
        slot = self._acquire_staging_slot(torch)
        ctypes.memmove(
            int(slot.host_buffer.data_ptr()),
            int(tmap.getPtr()),
            128,
        )
        tensor.copy_(slot.host_buffer, non_blocking=True)
        slot.event.record(torch.cuda.current_stream() if stream is None else stream)
        self.tensor = tensor
        self.pointer = pointer
        self.recipe = tuple(arguments)
        tensor._loom_tensor_map_recipe = self.recipe

    def _activate(self, pointer, tensor):
        descriptor_pointer = int(tensor.data_ptr())
        for carrier in self.pointer_carriers:
            carrier.value = descriptor_pointer
        arguments = list(self.recipe)
        arguments[2] = pointer
        self.tensor = tensor
        self.pointer = pointer
        self.recipe = tuple(arguments)
        tensor._loom_tensor_map_recipe = self.recipe

    def refresh(self, source, *, stream=None):
        pointer = int(source.data_ptr())
        if pointer == self.pointer:
            return
        self._encode_into(pointer, self.tensor, stream=stream)
        # Generic public rebinding mutates the active descriptor in place.
        # Reset the private variant bank so no stale pointer key can name it.
        self.variants.clear()
        self.variants[pointer] = self.tensor

    def rebind_stream_bound(self, source, *, stream):
        if stream is None:
            raise ValueError("stream-bound tensor-map rebind requires an explicit stream")
        pointer = int(source.data_ptr())
        if pointer == self.pointer:
            return
        cached = self.variants.pop(pointer, None)
        if cached is not None:
            # LRU recency: re-insert the hit so eviction removes the
            # least-recently-activated variant, not the newest one.
            self.variants[pointer] = cached
            self._activate(pointer, cached)
            return

        import torch

        if len(self.variants) < self.variant_capacity:
            tensor = torch.empty_like(self.tensor)
            record_stream = getattr(tensor, "record_stream", None)
            if callable(record_stream):
                record_stream(stream)
        else:
            tensor = self.variants.pop(next(iter(self.variants)))
        self._encode_into(pointer, tensor, stream=stream)
        self.variants[pointer] = tensor
        self._activate(pointer, tensor)


@dataclass
class _TensorMapStagingSlot:
    host_buffer: object
    event: object


def _own_tensor_map_bindings(launches, bindings, *, stream):
    '''Clone cached descriptors and patch every launch to slot-owned storage.'''

    if not bindings:
        return ()
    owned_by_identity = {}
    owned_bindings = []
    with launch_stream_context(stream):
        for binding in bindings:
            original = binding.tensor
            owned = original.clone()
            owned._loom_tensor_map_recipe = binding.recipe
            metadata = getattr(original, "_loom_tma_metadata", None)
            if metadata is not None:
                owned._loom_tma_metadata = metadata
            owned_by_identity[id(original)] = owned
            pointer_carriers = tuple(
                launch._packed._prevent_gc[index]
                for launch in launches
                for index, arg in enumerate(launch._keepalive)
                if id(arg) == id(original)
            )
            if not pointer_carriers or any(
                type(carrier) is not ctypes.c_void_p for carrier in pointer_carriers
            ):
                raise RuntimeError("captured tensor-map binding has invalid pointer carriers")
            owned_bindings.append(
                _TensorMapBinding(
                    input_key=binding.input_key,
                    tensor=owned,
                    recipe=binding.recipe,
                    pointer=binding.pointer,
                    pointer_carriers=pointer_carriers,
                )
            )
        for launch in launches:
            replacements = {
                index: owned_by_identity[id(arg)]
                for index, arg in enumerate(launch._keepalive)
                if id(arg) in owned_by_identity
            }
            if replacements:
                launch.rebind_arguments(replacements, stream=stream)
    return tuple(owned_bindings)


def _public_tensor_input_identities(inputs):
    if inputs is None:
        return {}
    if not hasattr(inputs, "items"):
        raise TypeError("capture inputs must be a mapping")
    identities = {}
    for key, value in inputs.items():
        if (
            isinstance(key, str)
            and not key.startswith("_")
            and callable(getattr(value, "data_ptr", None))
        ):
            identities.setdefault(id(value), key)
    return identities


def _public_tensor_input_pointers(inputs):
    if inputs is None:
        return {}
    pointers = {}
    for key, value in inputs.items():
        if (
            isinstance(key, str)
            and not key.startswith("_")
            and callable(getattr(value, "data_ptr", None))
        ):
            pointers.setdefault(int(value.data_ptr()), key)
    return pointers


def _public_tensor_alias_topology(inputs, keys=None):
    '''Return the complete pointer-equality partition of public tensor inputs.'''

    if inputs is None:
        return ()
    if not hasattr(inputs, "items"):
        raise TypeError("capture inputs must be a mapping")
    if keys is None:
        selected = [
            key
            for key, value in inputs.items()
            if (
                isinstance(key, str)
                and not key.startswith("_")
                and callable(getattr(value, "data_ptr", None))
            )
        ]
    else:
        selected = list(keys)
        missing = sorted(set(selected) - set(inputs))
        if missing:
            raise KeyError(f"missing prepared semantic alias binding(s): {missing!r}")
        invalid = sorted(
            key
            for key in selected
            if not callable(getattr(inputs[key], "data_ptr", None))
        )
        if invalid:
            raise TypeError(
                f"prepared semantic alias binding(s) must be tensor-like: {invalid!r}"
            )
    groups = {}
    for key in selected:
        groups.setdefault(int(inputs[key].data_ptr()), []).append(key)
    return tuple(sorted(tuple(sorted(group)) for group in groups.values()))


def _validate_public_tensor_alias_topology(inputs, expected):
    if not expected:
        return
    keys = tuple(key for group in expected for key in group)
    actual = _public_tensor_alias_topology(inputs, keys)
    if actual != expected:
        raise RuntimeError(
            "prepared semantic public tensor alias topology changed: "
            f"expected {expected!r}, got {actual!r}; capture a new sequence"
        )


def _capture_result_template(value, input_key_by_identity):
    key = input_key_by_identity.get(id(value))
    if key is not None:
        return _BoundInputResult(key)
    if isinstance(value, tuple):
        return tuple(_capture_result_template(item, input_key_by_identity) for item in value)
    if isinstance(value, list):
        return [_capture_result_template(item, input_key_by_identity) for item in value]
    if isinstance(value, dict):
        return {
            key: _capture_result_template(item, input_key_by_identity)
            for key, item in value.items()
        }
    return value


def _materialize_result_template(value, inputs):
    if isinstance(value, _BoundInputResult):
        return inputs[value.key]
    if isinstance(value, tuple):
        return tuple(_materialize_result_template(item, inputs) for item in value)
    if isinstance(value, list):
        return [_materialize_result_template(item, inputs) for item in value]
    if isinstance(value, dict):
        return {key: _materialize_result_template(item, inputs) for key, item in value.items()}
    return value


@contextmanager
def capture_kernel_launches(*, stream=None, arch=None, inputs=None, rebind=None):
    import torch

    with _launch_capture_prepare_lock:
        if _active_launch_capture.get() is not None:
            raise RuntimeError("nested kernel launch capture is not supported")
        capture = KernelLaunchCapture(stream=stream, arch=arch, inputs=inputs, rebind=rebind)
        token = _active_launch_capture.set(capture)
        # Captured launches are marshalled, not run, so any device read the
        # route performs mid-traversal (for example an ``overflow_flag.item()``
        # certification) observes memory its recorded kernels never wrote.
        # Interpose ``torch.Tensor.item`` for the capture's duration and count
        # CUDA-tensor reads; a nonzero count marks the capture
        # ``host_data_dependent`` so the plan builder keeps that signature on
        # the per-call launcher instead of freezing an unreproducible branch.
        # Captures are serialized by ``_launch_capture_prepare_lock``, so the
        # process-global interpose cannot nest. ``.item()`` is the only device
        # read the vendored dispatch modules perform on their launch paths.
        # Torch test doubles without a ``Tensor.item`` skip the interpose.
        original_tensor_item = getattr(getattr(torch, "Tensor", None), "item", None)

        def _observed_item(tensor):
            if getattr(tensor, "is_cuda", False):
                capture.host_data_reads += 1
            return original_tensor_item(tensor)

        if original_tensor_item is not None:
            torch.Tensor.item = _observed_item
        try:
            if stream is None:
                yield capture
            else:
                with torch.cuda.stream(stream):
                    yield capture
        finally:
            if original_tensor_item is not None:
                torch.Tensor.item = original_tensor_item
            capture.release_route_caches()
            _active_launch_capture.reset(token)


class DispatchKernel:
    def __init__(self, alias, symbol=None):
        self.exported = get_kernel(alias)
        self.symbol = symbol or self.exported.spec.symbol

    def __enter__(self):
        return self

    def __exit__(self, *args):
        return None

    def launch(self, *, grid, block, args, shared_mem=0, stream=None, timeout_ms=None, **kwargs):
        stream, timeout_ms = _resolved_launch_options(stream, timeout_ms)
        capture = _active_launch_capture.get()
        if capture is not None:
            capture.add_kernel_launch(
                self.exported,
                mode="regular",
                grid=grid,
                block=block,
                args=args,
                arg_types=self.exported.arg_types,
                shared_mem=shared_mem,
                stream=stream,
            )
            return
        self.exported.launch(
            *args, grid=grid, block=block, shared_mem=shared_mem, stream=stream,
            timeout_ms=timeout_ms, options=["--use_fast_math"],
        )

    def launch_cluster(
        self, *, grid, block, args, cluster_dims, shared_mem=0, stream=None,
        timeout_ms=None, **kwargs
    ):
        stream, timeout_ms = _resolved_launch_options(stream, timeout_ms)
        capture = _active_launch_capture.get()
        if capture is not None:
            capture.add_kernel_launch(
                self.exported,
                mode="cluster",
                grid=grid,
                block=block,
                args=args,
                arg_types=self.exported.arg_types,
                cluster_dims=cluster_dims,
                shared_mem=shared_mem,
                stream=stream,
            )
            return
        arch, stream, timeout_ms = resolve_launch_defaults(
            arch=None,
            stream=stream,
            timeout_ms=timeout_ms,
        )
        with launch_stream_context(stream):
            kernel = self.exported.compile(arch=arch, options=["--use_fast_math"])
            kernel.launch_cluster(
                grid=grid, block=block, args=tuple(args),
                arg_types=self.exported.arg_types,
                cluster_dims=cluster_dims, shared_mem=shared_mem, stream=stream,
                timeout_ms=timeout_ms,
            )

    def launch_cooperative(
        self, *, grid, block, args, shared_mem=0, stream=None, timeout_ms=None, **kwargs
    ):
        stream, timeout_ms = _resolved_launch_options(stream, timeout_ms)
        capture = _active_launch_capture.get()
        if capture is not None:
            capture.add_kernel_launch(
                self.exported,
                mode="cooperative",
                grid=grid,
                block=block,
                args=args,
                arg_types=self.exported.arg_types,
                shared_mem=shared_mem,
                stream=stream,
            )
            return
        arch, stream, timeout_ms = resolve_launch_defaults(
            arch=None,
            stream=stream,
            timeout_ms=timeout_ms,
        )
        with launch_stream_context(stream):
            kernel = self.exported.compile(arch=arch, options=["--use_fast_math"])
            kernel.launch_cooperative(
                grid=grid, block=block, args=tuple(args),
                arg_types=self.exported.arg_types,
                shared_mem=shared_mem, stream=stream, timeout_ms=timeout_ms,
            )

    def prepare_launch(self, **kwargs):
        return _PreparedDispatchLaunch(self.launch, kwargs)

    def prepare_launch_cluster(self, **kwargs):
        return _PreparedDispatchLaunch(self.launch_cluster, kwargs)

    def prepare_launch_cooperative(self, **kwargs):
        return _PreparedDispatchLaunch(self.launch_cooperative, kwargs)


class _PreparedDispatchLaunch:
    def __init__(self, launch, kwargs):
        self._launch = launch
        self._kwargs = dict(kwargs)

    def launch(self, timeout_ms=None):
        kwargs = dict(self._kwargs)
        if timeout_ms is not None:
            kwargs["timeout_ms"] = timeout_ms
        return self._launch(**kwargs)


CUDAKernel = DispatchKernel


def compile_cuda(source, **kwargs):
    return source


def detect_gpu_arch():
    import torch
    major, minor = torch.cuda.get_device_capability()
    return f"sm_{major}{minor}a"


def _cuda_include_dirs():
    return []


def arch_flag_for_cc(major, minor):
    sm = int(major) * 10 + int(minor)
    return f"sm_{sm}a" if sm >= 90 else f"sm_{sm}"


def _capture_cuTensorMapEncodeTiled(*arguments):
    '''Encode a tensor map while retaining a pointer-rebind recipe.'''
    from cuda.bindings import driver

    result = driver.cuTensorMapEncodeTiled(*arguments)
    if result[0] == 0:
        _pending_tensor_map_recipe.set(tuple(arguments))
    else:
        _pending_tensor_map_recipe.set(None)
    return result


def _tmap_to_device(tmap, metadata=None):
    import torch
    del metadata
    recipe = _pending_tensor_map_recipe.get()
    _pending_tensor_map_recipe.set(None)
    host_ptr = tmap.getPtr()
    raw = bytes((ctypes.c_ubyte * 128).from_address(host_ptr))
    host = torch.frombuffer(bytearray(raw), dtype=torch.uint8)
    device = torch.empty(128, dtype=torch.uint8, device="cuda")
    device.copy_(host)
    if recipe is not None:
        device._loom_tensor_map_recipe = recipe
    return device


class Swizzle:
    # Standalone spellings used only by stripped TMA metadata helpers.
    SZ_128B = "128B"
    SZ_64B = "64B"
    SZ_32B = "32B"
    NONE = "none"


class TensorMapMetadata:
    # Compatibility carrier; frozen launch packing needs no metadata.
    def __init__(self, **values):
        self.__dict__.update(values)


def attach_tma_metadata(tensor, metadata):
    tensor._loom_tensor_map_metadata = metadata
    return tensor


def create_tensor_map(data_ptr, dim0, dim1, box0, box1, stride1_bytes):
    '''Create a rank-2 BF16 128B-swizzled map with a rebind recipe.'''
    from cuda.bindings import driver

    err, tmap = _capture_cuTensorMapEncodeTiled(
        driver.CUtensorMapDataType.CU_TENSOR_MAP_DATA_TYPE_BFLOAT16,
        2,
        data_ptr,
        [driver.cuuint64_t(dim0), driver.cuuint64_t(dim1)],
        [driver.cuuint64_t(stride1_bytes)],
        [driver.cuuint32_t(box0), driver.cuuint32_t(box1)],
        [driver.cuuint32_t(1), driver.cuuint32_t(1)],
        driver.CUtensorMapInterleave.CU_TENSOR_MAP_INTERLEAVE_NONE,
        driver.CUtensorMapSwizzle.CU_TENSOR_MAP_SWIZZLE_128B,
        driver.CUtensorMapL2promotion.CU_TENSOR_MAP_L2_PROMOTION_NONE,
        driver.CUtensorMapFloatOOBfill.CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE,
    )
    if err != 0:
        raise RuntimeError(f"cuTensorMapEncodeTiled (2D BF16) failed: CUresult={err}")
    return _tmap_to_device(tmap)


def _create_tensor_map_3d(data_ptr, global_height, shared_height, width, block_width, swizzle):
    from cuda.bindings import driver
    atoms = {"128B": 64, "64B": 32, "32B": 16}
    swizzles = {
        "128B": driver.CUtensorMapSwizzle.CU_TENSOR_MAP_SWIZZLE_128B,
        "64B": driver.CUtensorMapSwizzle.CU_TENSOR_MAP_SWIZZLE_64B,
        "32B": driver.CUtensorMapSwizzle.CU_TENSOR_MAP_SWIZZLE_32B,
    }
    try:
        atom = atoms[swizzle]
        swizzle_value = swizzles[swizzle]
    except KeyError as exc:
        raise ValueError(f"unsupported 3D tensor-map swizzle: {swizzle}") from exc
    err, tmap = _capture_cuTensorMapEncodeTiled(
        driver.CUtensorMapDataType.CU_TENSOR_MAP_DATA_TYPE_BFLOAT16,
        3,
        data_ptr,
        [driver.cuuint64_t(atom), driver.cuuint64_t(global_height), driver.cuuint64_t(width // atom)],
        [driver.cuuint64_t(width * 2), driver.cuuint64_t(atom * 2)],
        [driver.cuuint32_t(atom), driver.cuuint32_t(shared_height), driver.cuuint32_t(block_width // atom)],
        [driver.cuuint32_t(1), driver.cuuint32_t(1), driver.cuuint32_t(1)],
        driver.CUtensorMapInterleave.CU_TENSOR_MAP_INTERLEAVE_NONE,
        swizzle_value,
        driver.CUtensorMapL2promotion.CU_TENSOR_MAP_L2_PROMOTION_NONE,
        driver.CUtensorMapFloatOOBfill.CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE,
    )
    if err != 0:
        raise RuntimeError(f"cuTensorMapEncodeTiled failed: CUresult={err}")
    return _tmap_to_device(tmap)


def create_tensor_map_3d(data_ptr, global_height, shared_height, width, block_width):
    return _create_tensor_map_3d(data_ptr, global_height, shared_height, width, block_width, "128B")


def create_tensor_map_3d_64b(data_ptr, global_height, shared_height, width, block_width):
    return _create_tensor_map_3d(data_ptr, global_height, shared_height, width, block_width, "64B")


def create_tensor_map_3d_32b(data_ptr, global_height, shared_height, width, block_width):
    return _create_tensor_map_3d(data_ptr, global_height, shared_height, width, block_width, "32B")


def generate_kernel(ir, **kwargs):
    request_key = json.dumps(
        {
            "ir_name": ir.symbol,
            "constants": [[str(name), value] for name, value in ir.constants],
            "threads": int(ir.threads),
            "computed_smem_bytes": int(ir.computed_smem_bytes),
            "kwargs": kwargs,
        },
        sort_keys=True, separators=(",", ":"), default=repr,
    )
    alias = _KERNEL_ALIAS_BY_REQUEST.get(request_key)
    if alias is None:
        alias = _KERNEL_ALIAS_BY_IR_NAME.get(ir.symbol)
    if alias is None:
        raise RuntimeError(f"uncaptured dispatcher specialization for {ir.symbol}")
    return alias


def generate_kernel_bundle(*args, **kwargs):
    raise RuntimeError("uncaptured dispatcher bundle specialization")


def _all_shapes():
    package = __package__ or __name__.rpartition(".")[0]
    text = resources.files(package).joinpath("_dispatch_shapes.json").read_text(encoding="utf-8")
    return json.loads(text)


class _CanonicalShapes:
    '''Lazy contract-shape view backed by the exported plan ledger.'''

    def __iter__(self):
        return iter(_all_shapes())

    def __len__(self):
        return len(_all_shapes())

    def __getitem__(self, index):
        return _all_shapes()[index]


CANONICAL_SHAPES = _CanonicalShapes()


def select_named_shapes(labels):
    labels = [labels] if isinstance(labels, str) else list(labels)
    by_label = {row["label"]: row for row in _all_shapes()}
    return [by_label[label] for label in labels]


def evaluate(*args, **kwargs):
    raise RuntimeError("Cake eval harness is not part of the standalone runtime")
