from __future__ import annotations
_KERNEL_ALIAS_BY_IR_NAME = {'knn_search_self_k5_direct_v1': 'dispatch_kernel_0000', 'knn_search_q1_tile_reduce_partial_v1': 'dispatch_kernel_0001', 'knn_search_q1_tile_reduce_merge_v1': 'dispatch_kernel_0002', 'knn_search_mma_split_merge_stream_v1': 'dispatch_kernel_0008', 'knn_search_mma_split_merge_q128_const148_v1': 'dispatch_kernel_0009', 'knn_search_mma_split_merge_q4096_pairlocal_v1': 'dispatch_kernel_0010', 'knn_search_q1_irregular_m_tail_partial_v1': 'dispatch_kernel_0011', 'knn_search_lowq_tile_reduce_partial_m131072_exact_0617_cc76_v1': 'dispatch_kernel_0012', 'knn_search_lowq_tile_reduce_merge_m131072_exact_0617_cc76_v1': 'dispatch_kernel_0013', 'knn_search_lowq_row16_mma_partial_dispatch0610_r18_d14a_vec16_bguard_v1': 'dispatch_kernel_0014', 'knn_search_floor13_k64_prefix8_partial_0622_f3ce_v1': 'dispatch_kernel_0021', 'knn_search_floor13_k64_prefix8_merge_0622_f3ce_v1': 'dispatch_kernel_0022', 'knn_search_k1_top1_pairq_partial_0622_598a_v1': 'dispatch_kernel_0023', 'knn_search_k1_top1_merge16_d212_v1': 'dispatch_kernel_0024', 'knn_search_q4096_lowk_k2partial_split9_merge_0613_r46_48e9_v1': 'dispatch_kernel_0026', 'knn_search_k64_q4096split79_twotile_oddevensort_partial_0612_r34_11c1_v1': 'dispatch_kernel_0027', 'knn_search_k64_q4096split79_indexfastmerge10_guarded_0612_r34_11c1_v1': 'dispatch_kernel_0028', 'knn_search_residual_full198_d256_k64_fused_hier8x64_e92c_v1': 'dispatch_kernel_0032', 'knn_search_scalar_capacity_merge_v1': 'dispatch_kernel_0034', 'knn_search_lowd_dbscan_d2_t128_m1536_0613_r56_cd72_v1': 'dispatch_kernel_0038', 'knn_search_blind_k64_twotile_partial_0614_50cc_v1': 'dispatch_kernel_0039', 'knn_search_blind_k64_highq_merge32_0614_50cc_v1': 'dispatch_kernel_0040', 'knn_search_k64_q128split512_twotile_partial_0614_r26_k64thin_v1': 'dispatch_kernel_0043', 'knn_search_k64_q128m65536_groupmerge64_kexact_0614_r27_k64thin_v1': 'dispatch_kernel_0044', 'knn_search_k64_q128m65536_finalmerge16_kexact_0614_r27_k64thin_v1': 'dispatch_kernel_0045', 'knn_search_blind_k64_q128m262144_groupmerge64_0614_r19_6389_v1': 'dispatch_kernel_0046', 'knn_search_blind_k64_q4096_m32768_prefix8_partial_5132_v1': 'dispatch_kernel_0047', 'knn_search_q4096_m32768_k32_prefix8_merge_tie_3c6e_v1': 'dispatch_kernel_0048', 'knn_search_q4096_m32768_k64_prefix8_merge_tie_3c6e_v1': 'dispatch_kernel_0049', 'knn_search_k48_q4096_m32768_prefix8_partial_0623_e36b_v1': 'dispatch_kernel_0055', 'knn_search_k48_q4096_m32768_prefix8_merge_0623_e36b_v1': 'dispatch_kernel_0056', 'knn_search_target0628_d128_q4096_m20000_k3_e8f1_k3partial_v1': 'dispatch_kernel_0059', 'knn_search_target0628_d128_q4096_m20000_k3_e8f1_k3partial_merge_v1': 'dispatch_kernel_0060', 'knn_search_k31k32_q128_split148_static_lateidx_merge_r68_v1': 'dispatch_kernel_0062', 'knn_search_blind_d384_tcgen05_partial_dispatch0610_r2_f94e_v1': 'dispatch_kernel_0063', 'knn_search_q1_flashdecode_merge128_0614_r92_v1': 'dispatch_kernel_0064', 'knn_search_lowq_tile_reduce_partial_dispatch0610_r3_v1': 'dispatch_kernel_0065', 'knn_search_lowq_tile_reduce_merge_dispatch0610_r3_v1': 'dispatch_kernel_0066', 'knn_search_lowq_tile_reduce_partial_dispatch0610_r8_blockm512_v1': 'dispatch_kernel_0067', 'knn_search_lowq_tile_reduce_merge_dispatch0610_r8_blockm512_v1': 'dispatch_kernel_0068', 'knn_search_d256_mma_split_partial_0612_r34_v1': 'dispatch_kernel_0069', 'knn_search_d384_mma_split_partial_0612_r34_v1': 'dispatch_kernel_0070', 'knn_search_80a5_blocker_k64_q256_m65536_twotile_partial_v1': 'dispatch_kernel_0071', 'knn_search_80a5_b2_q128m65536_k64_twotile_partial_74f4_v1': 'dispatch_kernel_0072', 'knn_search_dynamic_tinyd_tile_reduce_merge_0618_c8b9_v1': 'dispatch_kernel_0074', 'knn_search_dynamic_d_d384_q32_m131072_exact_tcgen05_partial_0618_5847_v1': 'dispatch_kernel_0080', 'knn_search_dynamic_self_d3_single_tile_0625_199f_v1': 'dispatch_kernel_0086', 'knn_search_dynamic_d512_q64_tcgen05_partial_0618_9286_v1': 'dispatch_kernel_0096', 'knn_search_ext_k_capacity_truncate64_to_k_0618_28ec_v1': 'dispatch_kernel_0099', 'knn_search_ext_k_capacity_q4096_m49152_merge24_0618_28ec_v1': 'dispatch_kernel_0100', 'knn_search_ext_k_capacity_q4096_m49152_partial_0618_28ec_v1': 'dispatch_kernel_0101', 'knn_search_ext_k64_highq_prefix8_partial_0618_c2e0_v1': 'dispatch_kernel_0102', 'knn_search_ext_k64_highq_prefix8_merge_0618_c2e0_v1': 'dispatch_kernel_0103', 'knn_search_d130_q64_k64_directpad_partial_0625_4b95_v1': 'dispatch_kernel_0104', 'knn_search_d512_q32_k64_mergefast_partial_83da_r121_v1': 'dispatch_kernel_0106', 'knn_search_d512_q32_k64_distonlymerge_merge_177e_r121_v1': 'dispatch_kernel_0107', 'knn_search_ivf_q12_m100_d64_k20_direct_6bea_r118_v1': 'dispatch_kernel_0108', 'knn_search_dynamic_d3_k32_self_tile_0620_9d5c_v1': 'dispatch_kernel_0109', 'knn_search_floor13_k64_q384_prefix8_partial_0622_f3ce_v1': 'dispatch_kernel_0110', 'knn_search_floor13_k64_q384_prefix8_merge_0622_f3ce_v1': 'dispatch_kernel_0111', 'knn_search_floor13_k80_prefix8_merge_0622_f3ce_v1': 'dispatch_kernel_0112', 'knn_search_d64_q128_m131072_k64_partial_0623_e157_v1': 'dispatch_kernel_0117', 'knn_search_high_d_low_q_d768_q64_n256_partial_0268_v1': 'dispatch_kernel_0118', 'knn_search_high_d_low_q_d768_q64_norm_merge_0268_v1': 'dispatch_kernel_0119', 'knn_search_d1024_q32_k64_targetd_partial_67ec_v1': 'dispatch_kernel_0120', 'knn_search_d1024_q32_k64_hiermerge8_group_f561_v2': 'dispatch_kernel_0121', 'knn_search_d1024_q32_k64_hiermerge8_final_f561_v2': 'dispatch_kernel_0122', 'knn_search_d2048_q8_m16384_k10_partial_0623_6185_d51b_v1': 'dispatch_kernel_0123', 'knn_search_d2048_q8_m16384_k10_merge_0623_6185_d51b_v1': 'dispatch_kernel_0124', 'knn_search_d4096_q4q8_m8192m16384_k10_partial_0623_5ff7_v1': 'dispatch_kernel_0125', 'knn_search_d4096_q4q8_m8192m16384_k10_merge_0623_5ff7_v1': 'dispatch_kernel_0126', 'knn_search_target0628_d64_q256_m131072_k10_partial_885d_hmerge8_v1': 'dispatch_kernel_0127', 'knn_search_target0628_d64_q256_m131072_k10_groupmerge_885d_hmerge8_v1': 'dispatch_kernel_0128', 'knn_search_target0628_d64_q256_m131072_k10_finalmerge_885d_hmerge8_v1': 'dispatch_kernel_0129', 'knn_search_target0628_d64_q512_m65536_k64_partial_e8f1_v1': 'dispatch_kernel_0130', 'knn_search_target0628_d64_q512_m65536_k64_groupmerge_dbaf_v1': 'dispatch_kernel_0131', 'knn_search_target0628_d64_q512_m65536_k64_finalmerge_dbaf_v1': 'dispatch_kernel_0132', 'knn_search_residual_q128_groupmerge76_fanin8_a8f5_v1': 'dispatch_kernel_0134', 'knn_search_d256_split256_rows8_finalmerge8compact_1056_v1': 'dispatch_kernel_0135', 'knn_search_target0627_d768_q64_m65536_k64_partial_6472_v2': 'dispatch_kernel_0137', 'knn_search_target0627_d768_q64_m65536_k64_merge_6472_v2': 'dispatch_kernel_0138', 'knn_search_target0628_d1024_q32_m65536_k64_group_merge16_9571_v2': 'dispatch_kernel_0140', 'knn_search_target0628_d1024_q32_m65536_k64_final_merge16_9571_v2': 'dispatch_kernel_0141', 'knn_search_d4096_q1_m65536_k10_partial_qcache_pipe_69ea_v1': 'dispatch_kernel_0142', 'knn_search_target0630_d4096_q4_m32768_k10_partial_qreuse_v1': 'dispatch_kernel_0144', 'knn_search_target0627_d4096_q4_m32768_k10_merge128_r221_v1': 'dispatch_kernel_0145', 'knn_search_target0628_d4096_q4_m8192_k64_partial_e750_v1': 'dispatch_kernel_0146', 'group_merge_ir': 'dispatch_kernel_0147', 'knn_search_q64_warp_distributed_state_04b4_v1': 'dispatch_kernel_0149', 'knn_search_q64_tail_groupmerge76_split152_3dee_v1': 'dispatch_kernel_0150', 'knn_search_q64_pairedowner_finalmerge_cce0_v1': 'dispatch_kernel_0151', 'knn_search_q64_tail_split152_full_wave_partial_3dee_v1': 'dispatch_kernel_0152', 'knn_search_q64_tail_plus_812c_sentinel_warp_state_361b_v2': 'dispatch_kernel_0153', 'knn_search_q64_pairedowner_groupmerge_cce0_v1': 'dispatch_kernel_0154', 'knn_search_q65_fused_04b4_low_depth_tail_d436_v1': 'dispatch_kernel_0155', 'knn_search_q65_tail_groupmerge_d436_v1': 'dispatch_kernel_0156', 'knn_search_q65_tail_finalmerge_d436_v1': 'dispatch_kernel_0157', 'knn_search_q4096_lowk_k5partial_0613_r49_48e9_v1': 'dispatch_kernel_0158', 'knn_search_q4096_lowk_k5partial_split9_merge_0613_r51_48e9_v1': 'dispatch_kernel_0159', 'knn_search_q4096_lowk_k8_stride10_out8_merge_0613_r52_48e9_v1': 'dispatch_kernel_0160', 'knn_search_scalar_capacity_pad_bf16_0705_v1': 'dispatch_kernel_0161', 'knn_search_warp_split_partial_v1': 'dispatch_kernel_0164', 'knn_search_warp_split_merge_v1': 'dispatch_kernel_0165', 'knn_search_warp_direct_v1': 'dispatch_kernel_0166', 'knn_search_q64_native_pairedowner_partial_cce0_v1': 'dispatch_kernel_0167', 'knn_search_d256_split256_groupmerge64_ownerless_7ce6_v1': 'dispatch_kernel_0168', 'knn_search_d256_split256_groupmerge64_tile64_d51ts_v1': 'dispatch_kernel_0169', 'knn_search_d256_split256_groupmerge64_warp3136_v1': 'dispatch_kernel_0170', 'knn_search_d256_split256_rows8_finalmerge8_hier9c25_v1': 'dispatch_kernel_0171', 'knn_search_d256_split256_groupmerge64_hier9c25_v1': 'dispatch_kernel_0172', 'knn_search_d256_split256_rows8_merge_v1': 'dispatch_kernel_0173', 'knn_search_d256_localcta_rows8_merge_v1': 'dispatch_kernel_0174', 'knn_search_lowd_ivf_direct_dispatch0610_r2_v1': 'dispatch_kernel_0175', 'knn_search_lowd_dbscan_direct_dispatch0610_r1_v1': 'dispatch_kernel_0176', 'knn_search_lowd_dbscan_d2_coopmerge_0612_r23_6e85_v1': 'dispatch_kernel_0177', 'knn_search_lowd_dbscan_d2_direct_0611_r22_6e85_v1': 'dispatch_kernel_0178', 'knn_search_k64_q4096split80_twotile_distanceonly_branchpruned_partial_0612_r30_11c1_v1': 'dispatch_kernel_0179', 'knn_search_k64_q4096split80_merge10_0612_r22_4e2c_v1': 'dispatch_kernel_0180', 'knn_search_k64_stable_merge_0612_r23_4e96_v1': 'dispatch_kernel_0182', 'knn_search_k64_q4096split80_twotile_partial_0612_r25_4e2c_v1': 'dispatch_kernel_0183', 'knn_search_d768_q64_m65536_k10_partial_0623_e35f_v1': 'dispatch_kernel_0187', 'knn_search_dynamic_d3_tile_reduce_partial_0618_c8b9_v1': 'dispatch_kernel_0188', 'knn_search_lowd_non128_tile_reduce_merge_0615_7d36_v2': 'dispatch_kernel_0189', 'knn_search_k64_q4096split80_twotile_distanceonly_stagingunrolled_partial_0612_r31_11c1_v1': 'dispatch_kernel_0192', 'knn_search_k64_q4096split80_indexfastmerge10_0612_r32_11c1_v1': 'dispatch_kernel_0193', 'knn_search_q4096_lowk_k1partial_minpair_0613_r46_48e9_lowk_k1top1_v1': 'dispatch_kernel_0194', 'knn_search_q4096_lowk_k1partial_minpair_merge_0613_r46_48e9_lowk_k1top1_v1': 'dispatch_kernel_0195', 'knn_search_q4096_lowk_k5partial_merge_0613_r49_48e9_v1': 'dispatch_kernel_0196', 'knn_search_q4096_lowk_k5_stride10_merge_0613_r48_48e9_v1': 'dispatch_kernel_0197', 'knn_search_q4096_lowk_k1partial_0613_r44_48e9_v1': 'dispatch_kernel_0200', 'knn_search_q4096_lowk_k1partial_merge_0613_r44_48e9_v1': 'dispatch_kernel_0201', 'knn_search_k64_q128split512_groupmerge64_0613_r43_11c1_v1': 'dispatch_kernel_0204', 'knn_search_k64_q128split512_finalmerge32_0613_r43_11c1_v1': 'dispatch_kernel_0205', 'knn_search_k64_q128split256_merge1024_0613_r37_11c1_v1': 'dispatch_kernel_0206', 'knn_search_q4096_lowk_k1partial_onestage_0614_r2_3ff5_v1': 'dispatch_kernel_0207', 'knn_search_q4096_lowk_k1partial_onestage_merge_0614_r2_3ff5_v1': 'dispatch_kernel_0208', 'knn_search_k64_q128split512_groupmerge64_indexfast_0614_r24_k64thin_v1': 'dispatch_kernel_0209', 'knn_search_k64_q128split512_finalmerge32_indexfast_0614_r24_k64thin_v1': 'dispatch_kernel_0210', 'knn_search_k64_q4096split79_localprefix9_partial_0615_r32_edd7_v1': 'dispatch_kernel_0211', 'knn_search_k64_q4096split79_localprefix_certmerge_0615_r32_edd7_v1': 'dispatch_kernel_0212', 'knn_search_k64_q4096split79_localprefix_certflag_init_0615_r32_edd7_v1': 'dispatch_kernel_0213', 'knn_search_k64_q4096split79_localprefix_cert_0615_r32_edd7_v1': 'dispatch_kernel_0214', 'knn_search_k64_q4096split79_localprefix_partial_0614_r36_edd7_v1': 'dispatch_kernel_0215', 'knn_search_k64_q4096split79_localprefix_merge_0614_r36_edd7_v1': 'dispatch_kernel_0216', 'knn_search_k64_q4096split79_localprefix7_partial_0615_245d_v1': 'dispatch_kernel_0217', 'knn_search_k64_q4096split79_localprefix6_rowflag_fusedcert_merge_0615_r36_e4cb_v1': 'dispatch_kernel_0218', 'knn_search_k64_q4096split79_localprefix6_certmerge_0615_245d_v1': 'dispatch_kernel_0219', 'knn_search_k64_q4096split79_localprefix6_certflag_init_0615_245d_v1': 'dispatch_kernel_0220', 'knn_search_k64_q4096split79_localprefix6_cert_0615_245d_v1': 'dispatch_kernel_0221', 'knn_search_k48_q4096split128_m32768_k48scratch_partial_0614_ddbc_q4096k48_v2': 'dispatch_kernel_0222', 'knn_search_k48_q4096split128_m32768_k48scratch_merge16_0614_ddbc_q4096k48_v2': 'dispatch_kernel_0223', 'knn_search_k48_q128split512_finalmerge32_strided_0614_ddbc_v1': 'dispatch_kernel_0224', 'knn_search_blind_k64_q4096_m32768_merge16_0614_1968_v1': 'dispatch_kernel_0225', 'knn_search_k1_top1_margin_qfull_partial_0614_r93_v1': 'dispatch_kernel_0226', 'knn_search_k1_top1_margin_0614_r93_merge8_v1': 'dispatch_kernel_0227', 'knn_search_k1_top1_margin_qfull_merge_0614_r93_v1': 'dispatch_kernel_0228', 'knn_search_lowq_tile_reduce_partial_0614_r10_e864_blockm640_v1': 'dispatch_kernel_0229', 'knn_search_lowq_tile_reduce_merge_0614_r10_e864_blockm640_v1': 'dispatch_kernel_0230', 'knn_search_lowq_tile_reduce_partial_0615_196e_blockm640_tailguard_v1': 'dispatch_kernel_0231', 'knn_search_k64_q4096split79_localprefix6_partial_0615_r5_9a85_v1': 'dispatch_kernel_0232', 'knn_search_k64_q4096split79_localprefix5_rowflag_fusedcert_merge_0615_r5_9a85_v1': 'dispatch_kernel_0233', 'knn_search_lowq_tile_reduce_partial_0614_r11_e864_blockm896_v1': 'dispatch_kernel_0234', 'knn_search_lowq_tile_reduce_merge_0614_r11_e864_blockm896_v1': 'dispatch_kernel_0235', 'knn_search_80a5_blocker_k32_q4096_m32768_split128_k34scratch_partial_v1': 'dispatch_kernel_0236', 'knn_search_80a5_blocker_k32_q4096_m32768_split128_k34scratch_merge16_v1': 'dispatch_kernel_0237', 'knn_search_dynamic_lowd_d1_warpmerge_0624_05a2_v1': 'dispatch_kernel_0238', 'knn_search_dynamic_lowd_direct_0624_3676_v1': 'dispatch_kernel_0239', 'knn_search_dynamic_lowd_d1_serialmerge_0624_3676_v1': 'dispatch_kernel_0240', 'knn_search_target_d4096_q8_m16384_k10_tcgen05_partial_0623_5ff7_v1': 'dispatch_kernel_0241', 'knn_search_k1_q128_split148_merge_0622_4201_v1': 'dispatch_kernel_0242', 'knn_search_q4096_m32768_k32_prefix8_merge_3053_v1': 'dispatch_kernel_0243', 'knn_search_blind_k64_q4096_m32768_prefix8_merge_5132_v1': 'dispatch_kernel_0244', 'knn_search_d512_q32_k64_mergefast_merge_83da_r121_v1': 'dispatch_kernel_0245', 'knn_search_d512_q32_k64_q32active_partial_abaf_r120_v1': 'dispatch_kernel_0246', 'knn_search_d512_q32_k64_q32active_merge_abaf_r120_v1': 'dispatch_kernel_0247', 'knn_search_d512_q32_k64_q64tile_partial_6bea_r119_v1': 'dispatch_kernel_0248', 'knn_search_d512_q32_k64_q64tile_merge_6bea_r119_v1': 'dispatch_kernel_0249', 'knn_search_d3_dbscan_q4096_k32_direct_9d5c_r117_v1': 'dispatch_kernel_0250', 'knn_search_lowk_k3_k5partial_a79b_merge_v1': 'dispatch_kernel_0251', 'knn_search_ext_k64_q4096_m49152_split192_partial_0618_28ec_v2': 'dispatch_kernel_0252', 'knn_search_ext_k64_q4096_m49152_prefix7_partial_0618_28ec_v2': 'dispatch_kernel_0253', 'knn_search_ext_k64_q4096_m49152_merge24_0618_28ec_v2': 'dispatch_kernel_0254', 'knn_search_ext_k64_q4096_m49152_prefix6cert_merge_0618_28ec_v2': 'dispatch_kernel_0255', 'knn_search_ext_k64_q4096_m49152_certflag_init_0618_28ec_v2': 'dispatch_kernel_0256', 'knn_search_dynamic_d3_k10_self_tile_0621_4832_v1': 'dispatch_kernel_0257', 'knn_search_q4096_m32769_k32_tailinsert_132_67a5_v1': 'dispatch_kernel_0260', 'knn_search_d1024_q32_k64_targetd_merge_67ec_v1': 'dispatch_kernel_0263', 'knn_search_dynamic_d3_self_q2048_r124_c16f_direct_v1': 'dispatch_kernel_0264', 'knn_search_dynamic_d3_self_q2048_r123_7d2a_direct_v1': 'dispatch_kernel_0265', 'knn_search_expanded_d1_tail_warpmerge_0624_025e_v1': 'dispatch_kernel_0266', 'knn_search_q64_m_tail_plus_extra_row_merge_ca90_v1': 'dispatch_kernel_0267', 'knn_search_d256_groupmerge64_fanin8cta_blockm64_891a_v1': 'dispatch_kernel_0268', 'knn_search_d256_groupmerge64_fanin4cta_814e_v1': 'dispatch_kernel_0269', 'knn_search_q64_tail_last_tile_handoff_partial_812c_v1': 'dispatch_kernel_0270', 'knn_search_d4096_q1_m65536_k10_partial_qcache_69ea_v1': 'dispatch_kernel_0271', 'knn_search_d4096_q1_m65536_k10_partial_compactsmem_69ea_v1': 'dispatch_kernel_0272', 'knn_search_d4096_q1_m65536_k10_partial_q1stage_v1': 'dispatch_kernel_0273', 'knn_search_target0627_d4096_q4_m32768_k10_merge148_r217_v1': 'dispatch_kernel_0274', 'knn_search_target0627_d4096_q4_m32768_k10_merge256_3737_v1': 'dispatch_kernel_0275', 'knn_search_target0628_d4096_q4_m8192_k64_group16_merge_2ced_v1': 'dispatch_kernel_0277', 'knn_search_target0628_d4096_q4_m8192_k64_group16_final_2ced_v1': 'dispatch_kernel_0278', 'knn_search_target_highd_k64_group_merge_f505_hiermerge32_v1': 'dispatch_kernel_0279', 'knn_search_target_highd_k64_final_merge_f505_hiermerge32_v1': 'dispatch_kernel_0280', 'knn_search_target0628_d4096_q4_m8192_k64_partial_7738_v1': 'dispatch_kernel_0281', 'knn_search_target0628_d1024_q32_m65536_k64_merge_9571_v1': 'dispatch_kernel_0282', 'knn_search_target0628_d64_q512_m65536_k64_groupmerge_e8f1_v1': 'dispatch_kernel_0283', 'knn_search_target0628_d64_q512_m65536_k64_finalmerge_e8f1_v1': 'dispatch_kernel_0284', 'knn_search_d1024_q8_tcgen05_partial_16warp_b3fc_v1': 'dispatch_kernel_0287', 'knn_search_target0630_d4096_q4_m32768_k10_merge_independent_warp_q4tail237_v1': 'dispatch_kernel_0288', 'knn_search_target0630_d4096_q4_m32768_k10_merge_q4warp_d759_v2': 'dispatch_kernel_0289', 'knn_search_q65_rows8_bounded_finalmerge_21e6_v1': 'dispatch_kernel_0290', 'knn_search_k64_prefix_to_k11_copy_0705_v1': 'dispatch_kernel_0291', 'knn_search_k64_q4096m20000_prefixcert_fused_merge_0615_576b_v1': 'dispatch_kernel_0292', 'knn_search_k64_q4096m20000_prefixcert_gpu_repair_0705_v1': 'dispatch_kernel_0293'}
_KERNEL_ALIAS_BY_REQUEST = {'{"computed_smem_bytes":384,"constants":[["D_",128],["K_MAX_",5],["NUM_WARPS_",8]],"ir_name":"knn_search_self_k5_direct_v1","kwargs":{"smem_bytes":320,"validate":false},"threads":256}': 'dispatch_kernel_0000', '{"computed_smem_bytes":5120,"constants":[["D_",128],["K_MAX_",10],["BLOCK_M_",256],["NUM_WARPS_",8],["NUM_ROW_WORKERS_",64],["SUBWARP_WIDTH_",4],["SUBWARPS_PER_WARP_",8],["LOCAL_LIST_CAP_",4]],"ir_name":"knn_search_q1_tile_reduce_partial_v1","kwargs":{"smem_bytes":5120,"validate":false},"threads":256}': 'dispatch_kernel_0001', '{"computed_smem_bytes":640,"constants":[["K_MAX_",10]],"ir_name":"knn_search_q1_tile_reduce_merge_v1","kwargs":{"smem_bytes":640,"validate":false},"threads":256}': 'dispatch_kernel_0002', '{"computed_smem_bytes":108800,"constants":[["K_MAX_",10],["EXPOSE_COL_COHORTS",0],["FULL_M_TILES",0]],"ir_name":"knn_search_mma_split_partial_v1","kwargs":{"smem_bytes":108800,"validate":false},"threads":640}': 'dispatch_kernel_0003', '{"computed_smem_bytes":108800,"constants":[["K_MAX_",10],["EXPOSE_COL_COHORTS",0],["FULL_M_TILES",0]],"ir_name":"knn_search_mma_split_partial_v1","kwargs":{"EXPOSE_COL_COHORTS":1,"smem_bytes":108800,"validate":false},"threads":640}': 'dispatch_kernel_0004', '{"computed_smem_bytes":108800,"constants":[["K_MAX_",10],["EXPOSE_COL_COHORTS",0],["FULL_M_TILES",0]],"ir_name":"knn_search_mma_split_partial_v1","kwargs":{"FULL_M_TILES":1,"smem_bytes":108800,"validate":false},"threads":640}': 'dispatch_kernel_0005', '{"computed_smem_bytes":108800,"constants":[["K_MAX_",10],["EXPOSE_COL_COHORTS",0],["FULL_M_TILES",0]],"ir_name":"knn_search_mma_split_partial_v1","kwargs":{"EXPOSE_COL_COHORTS":1,"FULL_M_TILES":1,"smem_bytes":108800,"validate":false},"threads":640}': 'dispatch_kernel_0006', '{"computed_smem_bytes":0,"constants":[["K_MAX_",10],["MERGE_SLOTS_",5]],"ir_name":"knn_search_mma_split_merge_v1","kwargs":{"smem_bytes":1024,"validate":false},"threads":32}': 'dispatch_kernel_0007', '{"computed_smem_bytes":0,"constants":[["K_MAX_",10]],"ir_name":"knn_search_mma_split_merge_stream_v1","kwargs":{"smem_bytes":1024,"validate":false},"threads":32}': 'dispatch_kernel_0008', '{"computed_smem_bytes":0,"constants":[["K_MAX_",10]],"ir_name":"knn_search_mma_split_merge_q128_const148_v1","kwargs":{"smem_bytes":1024,"validate":false},"threads":32}': 'dispatch_kernel_0009', '{"computed_smem_bytes":0,"constants":[["K_MAX_",10]],"ir_name":"knn_search_mma_split_merge_q4096_pairlocal_v1","kwargs":{"smem_bytes":1024,"validate":false},"threads":32}': 'dispatch_kernel_0010', '{"computed_smem_bytes":5120,"constants":[["D_",128],["K_MAX_",10],["BLOCK_M_",256],["NUM_WARPS_",8],["NUM_ROW_WORKERS_",64],["SUBWARP_WIDTH_",4],["SUBWARPS_PER_WARP_",8],["LOCAL_LIST_CAP_",4]],"ir_name":"knn_search_q1_irregular_m_tail_partial_v1","kwargs":{"smem_bytes":5120,"validate":false},"threads":256}': 'dispatch_kernel_0011', '{"computed_smem_bytes":5120,"constants":[["D_",128],["K_MAX_",10],["BLOCK_M_",640],["ROUTED_M_",131072],["NUM_M_TILES_",205],["NUM_ROW_WORKERS_",64],["SUBWARP_WIDTH_",4],["SUBWARPS_PER_WARP_",8],["LOCAL_LIST_CAP_",10]],"ir_name":"knn_search_lowq_tile_reduce_partial_m131072_exact_0617_cc76_v1","kwargs":{"smem_bytes":5120,"validate":false},"threads":256}': 'dispatch_kernel_0012', '{"computed_smem_bytes":384,"constants":[["K_MAX_",10],["NUM_M_TILES_",205],["NUM_GROUPS_",4],["TILES_PER_GROUP_",64]],"ir_name":"knn_search_lowq_tile_reduce_merge_m131072_exact_0617_cc76_v1","kwargs":{"smem_bytes":320,"validate":false},"threads":128}': 'dispatch_kernel_0013', '{"computed_smem_bytes":96000,"constants":[["K_MAX_",10]],"ir_name":"knn_search_lowq_row16_mma_partial_dispatch0610_r18_d14a_vec16_bguard_v1","kwargs":{"smem_bytes":96000,"validate":false},"threads":512}': 'dispatch_kernel_0014', '{"computed_smem_bytes":165120,"constants":[["K_MAX_",32],["EXPOSE_COL_COHORTS",0]],"ir_name":"knn_search_mma_split_partial_v1","kwargs":{"EXPOSE_COL_COHORTS":1,"K_MAX_":12,"smem_bytes":165120,"validate":false},"threads":512}': 'dispatch_kernel_0015', '{"computed_smem_bytes":0,"constants":[["K_MAX_",32]],"ir_name":"knn_search_mma_split_merge_v1","kwargs":{"K_MAX_":12,"smem_bytes":1024,"validate":false},"threads":32}': 'dispatch_kernel_0016', '{"computed_smem_bytes":165120,"constants":[["K_MAX_",32],["EXPOSE_COL_COHORTS",0]],"ir_name":"knn_search_mma_split_partial_v1","kwargs":{"EXPOSE_COL_COHORTS":0,"K_MAX_":20,"smem_bytes":165120,"validate":false},"threads":512}': 'dispatch_kernel_0017', '{"computed_smem_bytes":0,"constants":[["K_MAX_",30]],"ir_name":"knn_search_k20k30_q128_split148_guarded_merge_v1","kwargs":{"K_MAX_":20,"smem_bytes":1024,"validate":false},"threads":32}': 'dispatch_kernel_0018', '{"computed_smem_bytes":165120,"constants":[["K_MAX_",32],["EXPOSE_COL_COHORTS",0]],"ir_name":"knn_search_mma_split_partial_v1","kwargs":{"EXPOSE_COL_COHORTS":0,"K_MAX_":30,"smem_bytes":165120,"validate":false},"threads":512}': 'dispatch_kernel_0019', '{"computed_smem_bytes":0,"constants":[["K_MAX_",30]],"ir_name":"knn_search_k20k30_q128_split148_guarded_merge_v1","kwargs":{"K_MAX_":30,"smem_bytes":1024,"validate":false},"threads":32}': 'dispatch_kernel_0020', '{"computed_smem_bytes":165120,"constants":[["K_MAX_",64],["K_PREFIX_",8]],"ir_name":"knn_search_floor13_k64_prefix8_partial_0622_f3ce_v1","kwargs":{"K_MAX_":64,"K_PREFIX_":8,"smem_bytes":165120,"validate":false},"threads":512}': 'dispatch_kernel_0021', '{"computed_smem_bytes":0,"constants":[["K_MAX_",64],["K_PREFIX_",8]],"ir_name":"knn_search_floor13_k64_prefix8_merge_0622_f3ce_v1","kwargs":{"K_MAX_":64,"K_PREFIX_":8,"smem_bytes":1024,"validate":false},"threads":32}': 'dispatch_kernel_0022', '{"computed_smem_bytes":149760,"constants":[],"ir_name":"knn_search_k1_top1_pairq_partial_0622_598a_v1","kwargs":{"smem_bytes":149760,"validate":false},"threads":640}': 'dispatch_kernel_0023', '{"computed_smem_bytes":0,"constants":[],"ir_name":"knn_search_k1_top1_merge16_d212_v1","kwargs":{"smem_bytes":1024,"validate":false},"threads":256}': 'dispatch_kernel_0024', '{"computed_smem_bytes":108800,"constants":[],"ir_name":"knn_search_q4096_lowk_k2partial_0613_r45_48e9_v1","kwargs":{"smem_bytes":108800,"validate":false},"threads":640}': 'dispatch_kernel_0025', '{"computed_smem_bytes":0,"constants":[["K_STRIDE_",2],["K_OUT_MAX_",2]],"ir_name":"knn_search_q4096_lowk_k2partial_split9_merge_0613_r46_48e9_v1","kwargs":{"K_OUT_MAX_":2,"K_STRIDE_":2,"smem_bytes":1024,"validate":false},"threads":32}': 'dispatch_kernel_0026', '{"computed_smem_bytes":165120,"constants":[["K_MAX_",64]],"ir_name":"knn_search_k64_q4096split79_twotile_oddevensort_partial_0612_r34_11c1_v1","kwargs":{"K_MAX_":64,"smem_bytes":165120,"validate":false},"threads":512}': 'dispatch_kernel_0027', '{"computed_smem_bytes":0,"constants":[["K_MAX_",64]],"ir_name":"knn_search_k64_q4096split79_indexfastmerge10_guarded_0612_r34_11c1_v1","kwargs":{"K_MAX_":64,"smem_bytes":1024,"validate":false},"threads":32}': 'dispatch_kernel_0028', '{"computed_smem_bytes":143104,"constants":[["K_MAX_",64],["EXPOSE_COL_COHORTS",1]],"ir_name":"knn_search_mma_split_partial_v1","kwargs":{"EXPOSE_COL_COHORTS":1,"K_MAX_":10,"smem_bytes":143104,"validate":false},"threads":256}': 'dispatch_kernel_0029', '{"computed_smem_bytes":0,"constants":[["K_MAX_",64]],"ir_name":"knn_search_mma_split_merge_v1","kwargs":{"K_MAX_":10,"smem_bytes":1024,"validate":false},"threads":32}': 'dispatch_kernel_0030', '{"computed_smem_bytes":143104,"constants":[["K_MAX_",64],["EXPOSE_COL_COHORTS",1]],"ir_name":"knn_search_mma_split_partial_v1","kwargs":{"EXPOSE_COL_COHORTS":1,"K_MAX_":64,"smem_bytes":143104,"validate":false},"threads":256}': 'dispatch_kernel_0031', '{"computed_smem_bytes":4096,"constants":[["K_MAX_",64]],"ir_name":"knn_search_residual_full198_d256_k64_fused_hier8x64_e92c_v1","kwargs":{"K_MAX_":64,"smem_bytes":4096,"validate":false},"threads":256}': 'dispatch_kernel_0032', '{"computed_smem_bytes":0,"constants":[["D_",8],["K_CAP_",64],["BLOCK_M_",512],["NUM_WARPS_",8]],"ir_name":"knn_search_scalar_capacity_partial_v1","kwargs":{"D_":32,"K_CAP_":10,"smem_bytes":0,"validate":false},"threads":256}': 'dispatch_kernel_0033', '{"computed_smem_bytes":131072,"constants":[["K_CAP_",64],["NUM_WARPS_",8]],"ir_name":"knn_search_scalar_capacity_merge_v1","kwargs":{"K_CAP_":10,"smem_bytes":131072,"validate":false},"threads":256}': 'dispatch_kernel_0034', '{"computed_smem_bytes":4096,"constants":[["D_",8],["K_CAP_",64],["NUM_WARPS_",8]],"ir_name":"knn_search_scalar_capacity_direct_v1","kwargs":{"D_":32,"K_CAP_":10,"smem_bytes":4096,"validate":false},"threads":256}': 'dispatch_kernel_0035', '{"computed_smem_bytes":0,"constants":[["D_",8],["K_CAP_",64],["BLOCK_M_",512],["NUM_WARPS_",8]],"ir_name":"knn_search_scalar_capacity_partial_v1","kwargs":{"D_":48,"K_CAP_":10,"smem_bytes":0,"validate":false},"threads":256}': 'dispatch_kernel_0036', '{"computed_smem_bytes":4096,"constants":[["D_",8],["K_CAP_",64],["NUM_WARPS_",8]],"ir_name":"knn_search_scalar_capacity_direct_v1","kwargs":{"D_":48,"K_CAP_":10,"smem_bytes":4096,"validate":false},"threads":256}': 'dispatch_kernel_0037', '{"computed_smem_bytes":12416,"constants":[["D_",2],["K_MAX_",64],["LOCAL_LIST_CAP_",12],["NUM_WARPS_",4]],"ir_name":"knn_search_lowd_dbscan_d2_t128_m1536_0613_r56_cd72_v1","kwargs":{"smem_bytes":12336,"validate":false},"threads":128}': 'dispatch_kernel_0038', '{"computed_smem_bytes":165120,"constants":[["K_MAX_",64]],"ir_name":"knn_search_blind_k64_twotile_partial_0614_50cc_v1","kwargs":{"K_MAX_":64,"smem_bytes":165120,"validate":false},"threads":512}': 'dispatch_kernel_0039', '{"computed_smem_bytes":0,"constants":[["K_MAX_",64]],"ir_name":"knn_search_blind_k64_highq_merge32_0614_50cc_v1","kwargs":{"K_MAX_":64,"smem_bytes":1024,"validate":false},"threads":32}': 'dispatch_kernel_0040', '{"computed_smem_bytes":0,"constants":[["K_MAX_",64]],"ir_name":"knn_search_k64_q128split512_groupmerge64_kexact_0614_r25_k64thin_v1","kwargs":{"K_MAX_":64,"smem_bytes":1024,"validate":false},"threads":32}': 'dispatch_kernel_0041', '{"computed_smem_bytes":0,"constants":[["K_MAX_",64]],"ir_name":"knn_search_k64_q128split512_finalmerge32_kexact_0614_r25_k64thin_v1","kwargs":{"K_MAX_":64,"smem_bytes":1024,"validate":false},"threads":32}': 'dispatch_kernel_0042', '{"computed_smem_bytes":165120,"constants":[["K_MAX_",64]],"ir_name":"knn_search_k64_q128split512_twotile_partial_0614_r26_k64thin_v1","kwargs":{"K_MAX_":64,"smem_bytes":165120,"validate":false},"threads":512}': 'dispatch_kernel_0043', '{"computed_smem_bytes":0,"constants":[["K_MAX_",64]],"ir_name":"knn_search_k64_q128m65536_groupmerge64_kexact_0614_r27_k64thin_v1","kwargs":{"K_MAX_":64,"smem_bytes":1024,"validate":false},"threads":32}': 'dispatch_kernel_0044', '{"computed_smem_bytes":0,"constants":[["K_MAX_",64]],"ir_name":"knn_search_k64_q128m65536_finalmerge16_kexact_0614_r27_k64thin_v1","kwargs":{"K_MAX_":64,"smem_bytes":1024,"validate":false},"threads":32}': 'dispatch_kernel_0045', '{"computed_smem_bytes":0,"constants":[["K_MAX_",64]],"ir_name":"knn_search_blind_k64_q128m262144_groupmerge64_0614_r19_6389_v1","kwargs":{"K_MAX_":64,"smem_bytes":1024,"validate":false},"threads":32}': 'dispatch_kernel_0046', '{"computed_smem_bytes":165120,"constants":[["K_MAX_",64],["K_PREFIX_",8]],"ir_name":"knn_search_blind_k64_q4096_m32768_prefix8_partial_5132_v1","kwargs":{"K_MAX_":64,"K_PREFIX_":8,"smem_bytes":165120,"validate":false},"threads":512}': 'dispatch_kernel_0047', '{"computed_smem_bytes":0,"constants":[["K_OUT_",32],["K_PREFIX_",8]],"ir_name":"knn_search_q4096_m32768_k32_prefix8_merge_tie_3c6e_v1","kwargs":{"K_OUT_":32,"K_PREFIX_":8,"smem_bytes":1024,"validate":false},"threads":32}': 'dispatch_kernel_0048', '{"computed_smem_bytes":0,"constants":[["K_MAX_",64],["K_PREFIX_",8]],"ir_name":"knn_search_q4096_m32768_k64_prefix8_merge_tie_3c6e_v1","kwargs":{"K_MAX_":64,"K_PREFIX_":8,"smem_bytes":1024,"validate":false},"threads":32}': 'dispatch_kernel_0049', '{"computed_smem_bytes":120576,"constants":[["K_MAX_",10],["D_TOTAL_",64],["NUM_D_PASSES_",1],["Q_NORM_PARTS_",4]],"ir_name":"knn_search_blind_lowd_non_d128_tcgen05_partial_0615_r99_v1","kwargs":{"D_TOTAL_":64,"NUM_D_PASSES_":1,"Q_NORM_PARTS_":4,"smem_bytes":120576,"validate":false},"threads":640}': 'dispatch_kernel_0050', '{"computed_smem_bytes":120576,"constants":[["K_MAX_",10],["D_TOTAL_",64],["NUM_D_PASSES_",1],["Q_NORM_PARTS_",4]],"ir_name":"knn_search_blind_lowd_non_d128_tcgen05_partial_0615_r99_v1","kwargs":{"D_TOTAL_":320,"NUM_D_PASSES_":3,"Q_NORM_PARTS_":20,"smem_bytes":120576,"validate":false},"threads":640}': 'dispatch_kernel_0051', '{"computed_smem_bytes":165120,"constants":[["K_MAX_",32],["EXPOSE_COL_COHORTS",0]],"ir_name":"knn_search_mma_split_partial_v1","kwargs":{"EXPOSE_COL_COHORTS":1,"K_MAX_":48,"smem_bytes":165120,"validate":false},"threads":512}': 'dispatch_kernel_0052', '{"computed_smem_bytes":0,"constants":[["K_MAX_",64]],"ir_name":"knn_search_k64_q128split512_groupmerge64_kexact_0614_r25_k64thin_v1","kwargs":{"K_MAX_":48,"smem_bytes":1024,"validate":false},"threads":32}': 'dispatch_kernel_0053', '{"computed_smem_bytes":0,"constants":[["K_MAX_",64]],"ir_name":"knn_search_k64_q128split512_finalmerge32_kexact_0614_r25_k64thin_v1","kwargs":{"K_MAX_":48,"smem_bytes":1024,"validate":false},"threads":32}': 'dispatch_kernel_0054', '{"computed_smem_bytes":165120,"constants":[["K_MAX_",64],["K_PREFIX_",8]],"ir_name":"knn_search_k48_q4096_m32768_prefix8_partial_0623_e36b_v1","kwargs":{"K_MAX_":64,"K_PREFIX_":8,"smem_bytes":165120,"validate":false},"threads":512}': 'dispatch_kernel_0055', '{"computed_smem_bytes":0,"constants":[["K_MAX_",48],["K_PREFIX_",8]],"ir_name":"knn_search_k48_q4096_m32768_prefix8_merge_0623_e36b_v1","kwargs":{"K_MAX_":48,"K_PREFIX_":8,"smem_bytes":1024,"validate":false},"threads":32}': 'dispatch_kernel_0056', '{"computed_smem_bytes":120576,"constants":[["K_MAX_",10],["D_TOTAL_",64],["NUM_D_PASSES_",1],["Q_NORM_PARTS_",4]],"ir_name":"knn_search_blind_lowd_non_d128_tcgen05_partial_0615_r99_v1","kwargs":{"D_TOTAL_":96,"NUM_D_PASSES_":1,"Q_NORM_PARTS_":6,"smem_bytes":120576,"validate":false},"threads":640}': 'dispatch_kernel_0057', '{"computed_smem_bytes":120576,"constants":[["K_MAX_",10],["D_TOTAL_",64],["NUM_D_PASSES_",1],["Q_NORM_PARTS_",4]],"ir_name":"knn_search_blind_lowd_non_d128_tcgen05_partial_0615_r99_v1","kwargs":{"D_TOTAL_":192,"NUM_D_PASSES_":2,"Q_NORM_PARTS_":12,"smem_bytes":120576,"validate":false},"threads":640}': 'dispatch_kernel_0058', '{"computed_smem_bytes":108800,"constants":[],"ir_name":"knn_search_target0628_d128_q4096_m20000_k3_e8f1_k3partial_v1","kwargs":{"smem_bytes":108800,"validate":false},"threads":640}': 'dispatch_kernel_0059', '{"computed_smem_bytes":0,"constants":[["K_STRIDE_",3],["K_OUT_MAX_",3]],"ir_name":"knn_search_target0628_d128_q4096_m20000_k3_e8f1_k3partial_merge_v1","kwargs":{"K_OUT_MAX_":3,"K_STRIDE_":3,"smem_bytes":1024,"validate":false},"threads":32}': 'dispatch_kernel_0060', '{"computed_smem_bytes":165120,"constants":[["K_MAX_",32],["EXPOSE_COL_COHORTS",0]],"ir_name":"knn_search_mma_split_partial_v1","kwargs":{"EXPOSE_COL_COHORTS":0,"K_MAX_":32,"smem_bytes":165120,"validate":false},"threads":512}': 'dispatch_kernel_0061', '{"computed_smem_bytes":0,"constants":[["K_MAX_",32]],"ir_name":"knn_search_k31k32_q128_split148_static_lateidx_merge_r68_v1","kwargs":{"K_MAX_":32,"smem_bytes":1024,"validate":false},"threads":32}': 'dispatch_kernel_0062', '{"computed_smem_bytes":122624,"constants":[["K_MAX_",10]],"ir_name":"knn_search_blind_d384_tcgen05_partial_dispatch0610_r2_f94e_v1","kwargs":{"smem_bytes":122624,"validate":false},"threads":640}': 'dispatch_kernel_0063', '{"computed_smem_bytes":640,"constants":[["K_MAX_",10]],"ir_name":"knn_search_q1_flashdecode_merge128_0614_r92_v1","kwargs":{"smem_bytes":640,"validate":false},"threads":256}': 'dispatch_kernel_0064', '{"computed_smem_bytes":5120,"constants":[["D_",128],["K_MAX_",10],["BLOCK_M_",256],["NUM_ROW_WORKERS_",64],["SUBWARP_WIDTH_",4],["SUBWARPS_PER_WARP_",8],["LOCAL_LIST_CAP_",4]],"ir_name":"knn_search_lowq_tile_reduce_partial_dispatch0610_r3_v1","kwargs":{"smem_bytes":5120,"validate":false},"threads":256}': 'dispatch_kernel_0065', '{"computed_smem_bytes":640,"constants":[["K_MAX_",10]],"ir_name":"knn_search_lowq_tile_reduce_merge_dispatch0610_r3_v1","kwargs":{"smem_bytes":640,"validate":false},"threads":256}': 'dispatch_kernel_0066', '{"computed_smem_bytes":5120,"constants":[["D_",128],["K_MAX_",10],["BLOCK_M_",512],["NUM_ROW_WORKERS_",64],["SUBWARP_WIDTH_",4],["SUBWARPS_PER_WARP_",8],["LOCAL_LIST_CAP_",8]],"ir_name":"knn_search_lowq_tile_reduce_partial_dispatch0610_r8_blockm512_v1","kwargs":{"smem_bytes":5120,"validate":false},"threads":256}': 'dispatch_kernel_0067', '{"computed_smem_bytes":640,"constants":[["K_MAX_",10]],"ir_name":"knn_search_lowq_tile_reduce_merge_dispatch0610_r8_blockm512_v1","kwargs":{"smem_bytes":640,"validate":false},"threads":256}': 'dispatch_kernel_0068', '{"computed_smem_bytes":120576,"constants":[["K_MAX_",10]],"ir_name":"knn_search_d256_mma_split_partial_0612_r34_v1","kwargs":{"smem_bytes":120576,"validate":false},"threads":640}': 'dispatch_kernel_0069', '{"computed_smem_bytes":126720,"constants":[["K_MAX_",10]],"ir_name":"knn_search_d384_mma_split_partial_0612_r34_v1","kwargs":{"smem_bytes":126720,"validate":false},"threads":640}': 'dispatch_kernel_0070', '{"computed_smem_bytes":165120,"constants":[["K_MAX_",64]],"ir_name":"knn_search_80a5_blocker_k64_q256_m65536_twotile_partial_v1","kwargs":{"K_MAX_":64,"smem_bytes":165120,"validate":false},"threads":512}': 'dispatch_kernel_0071', '{"computed_smem_bytes":165120,"constants":[["K_MAX_",64]],"ir_name":"knn_search_80a5_b2_q128m65536_k64_twotile_partial_74f4_v1","kwargs":{"K_MAX_":64,"smem_bytes":165120,"validate":false},"threads":512}': 'dispatch_kernel_0072', '{"computed_smem_bytes":10240,"constants":[["D_",3],["BLOCK_M_",4096],["ROWS_PER_WORKER_",32],["K_MAX_",10]],"ir_name":"knn_search_dynamic_tinyd_tile_reduce_partial_0618_c8b9_v1","kwargs":{"BLOCK_M_":4096,"D_":3,"ROWS_PER_WORKER_":32,"smem_bytes":10240,"validate":false},"threads":128}': 'dispatch_kernel_0073', '{"computed_smem_bytes":384,"constants":[["K_MAX_",10]],"ir_name":"knn_search_dynamic_tinyd_tile_reduce_merge_0618_c8b9_v1","kwargs":{"smem_bytes":320,"validate":false},"threads":128}': 'dispatch_kernel_0074', '{"computed_smem_bytes":112384,"constants":[["K_MAX_",10],["D_TOTAL_",3],["NUM_D_PASSES_",1],["Q_NORM_PARTS_",1]],"ir_name":"knn_search_dynamic_d_tiny_q128_m65536_tcgen05_partial_0618_c8b9_v1","kwargs":{"D_TOTAL_":7,"NUM_D_PASSES_":1,"Q_NORM_PARTS_":1,"smem_bytes":112384,"validate":false},"threads":640}': 'dispatch_kernel_0075', '{"computed_smem_bytes":112384,"constants":[["K_MAX_",10],["D_TOTAL_",3],["NUM_D_PASSES_",1],["Q_NORM_PARTS_",1]],"ir_name":"knn_search_dynamic_d_tiny_q128_m65536_tcgen05_partial_0618_c8b9_v1","kwargs":{"D_TOTAL_":63,"NUM_D_PASSES_":1,"Q_NORM_PARTS_":4,"smem_bytes":112384,"validate":false},"threads":640}': 'dispatch_kernel_0076', '{"computed_smem_bytes":126720,"constants":[["K_MAX_",10],["D_ORIG_",512],["NUM_D_PASSES_",4],["Q_NORM_PARTS_",32]],"ir_name":"knn_search_dynamic_d_high_directstride_tcgen05_partial_0618_ccef_v1","kwargs":{"D_ORIG_":129,"NUM_D_PASSES_":2,"Q_NORM_PARTS_":9,"smem_bytes":126720,"validate":false},"threads":640}': 'dispatch_kernel_0077', '{"computed_smem_bytes":126720,"constants":[["K_MAX_",10],["D_ORIG_",512],["NUM_D_PASSES_",4],["Q_NORM_PARTS_",32]],"ir_name":"knn_search_dynamic_d_high_directstride_tcgen05_partial_0618_ccef_v1","kwargs":{"D_ORIG_":257,"NUM_D_PASSES_":3,"Q_NORM_PARTS_":17,"smem_bytes":126720,"validate":false},"threads":640}': 'dispatch_kernel_0078', '{"computed_smem_bytes":126720,"constants":[["K_MAX_",10],["D_ORIG_",512],["NUM_D_PASSES_",4],["Q_NORM_PARTS_",32]],"ir_name":"knn_search_dynamic_d_high_directstride_tcgen05_partial_0618_ccef_v1","kwargs":{"D_ORIG_":511,"NUM_D_PASSES_":4,"Q_NORM_PARTS_":32,"smem_bytes":126720,"validate":false},"threads":640}': 'dispatch_kernel_0079', '{"computed_smem_bytes":122624,"constants":[["K_MAX_",10]],"ir_name":"knn_search_dynamic_d_d384_q32_m131072_exact_tcgen05_partial_0618_5847_v1","kwargs":{"smem_bytes":122624,"validate":false},"threads":640}': 'dispatch_kernel_0080', '{"computed_smem_bytes":0,"constants":[["D_ORIG_",3],["D_PAD_",16]],"ir_name":"knn_search_dynamic_d_tiny_pack_bf16_0618_c8b9_v1","kwargs":{"D_ORIG_":257,"D_PAD_":512,"smem_bytes":0,"validate":false},"threads":256}': 'dispatch_kernel_0081', '{"computed_smem_bytes":151296,"constants":[["K_MAX_",64]],"ir_name":"knn_search_dynamic_d257_k64_q64_partial_0618_ccef_v1","kwargs":{"K_MAX_":64,"smem_bytes":151296,"validate":false},"threads":256}': 'dispatch_kernel_0082', '{"computed_smem_bytes":0,"constants":[["K_MAX_",64]],"ir_name":"knn_search_dynamic_d257_k64_q64_merge_0618_ccef_v1","kwargs":{"K_MAX_":64,"smem_bytes":1024,"validate":false},"threads":32}': 'dispatch_kernel_0083', '{"computed_smem_bytes":0,"constants":[["D_ORIG_",3],["D_PAD_",16]],"ir_name":"knn_search_dynamic_d_tiny_pack_bf16_0618_c8b9_v1","kwargs":{"D_ORIG_":129,"D_PAD_":144,"smem_bytes":0,"validate":false},"threads":256}': 'dispatch_kernel_0084', '{"computed_smem_bytes":120576,"constants":[["K_MAX_",10],["D_TOTAL_",64],["NUM_D_PASSES_",1],["Q_NORM_PARTS_",4]],"ir_name":"knn_search_blind_lowd_non_d128_tcgen05_partial_0615_r99_v1","kwargs":{"D_TOTAL_":144,"NUM_D_PASSES_":2,"Q_NORM_PARTS_":9,"smem_bytes":120576,"validate":false},"threads":640}': 'dispatch_kernel_0085', '{"computed_smem_bytes":10240,"constants":[["K_MAX_",10],["ROWS_PER_WORKER_",16]],"ir_name":"knn_search_dynamic_self_d3_single_tile_0625_199f_v1","kwargs":{"smem_bytes":10240,"validate":false},"threads":256}': 'dispatch_kernel_0086', '{"computed_smem_bytes":10240,"constants":[["D_",3],["BLOCK_M_",4096],["ROWS_PER_WORKER_",32],["K_MAX_",10]],"ir_name":"knn_search_dynamic_tinyd_tile_reduce_partial_0618_c8b9_v1","kwargs":{"BLOCK_M_":4096,"D_":1,"ROWS_PER_WORKER_":32,"smem_bytes":10240,"validate":false},"threads":128}': 'dispatch_kernel_0087', '{"computed_smem_bytes":10240,"constants":[["D_",3],["BLOCK_M_",4096],["ROWS_PER_WORKER_",32],["K_MAX_",10]],"ir_name":"knn_search_dynamic_tinyd_tile_reduce_partial_0618_c8b9_v1","kwargs":{"BLOCK_M_":4096,"D_":5,"ROWS_PER_WORKER_":32,"smem_bytes":10240,"validate":false},"threads":128}': 'dispatch_kernel_0088', '{"computed_smem_bytes":126720,"constants":[["K_MAX_",10],["D_ORIG_",512],["NUM_D_PASSES_",4],["Q_NORM_PARTS_",32]],"ir_name":"knn_search_dynamic_d_high_directstride_tcgen05_partial_0618_ccef_v1","kwargs":{"D_ORIG_":15,"NUM_D_PASSES_":1,"Q_NORM_PARTS_":1,"smem_bytes":126720,"validate":false},"threads":640}': 'dispatch_kernel_0089', '{"computed_smem_bytes":126720,"constants":[["K_MAX_",10],["D_ORIG_",512],["NUM_D_PASSES_",4],["Q_NORM_PARTS_",32]],"ir_name":"knn_search_dynamic_d_high_directstride_tcgen05_partial_0618_ccef_v1","kwargs":{"D_ORIG_":31,"NUM_D_PASSES_":1,"Q_NORM_PARTS_":2,"smem_bytes":126720,"validate":false},"threads":640}': 'dispatch_kernel_0090', '{"computed_smem_bytes":126720,"constants":[["K_MAX_",10],["D_ORIG_",512],["NUM_D_PASSES_",4],["Q_NORM_PARTS_",32]],"ir_name":"knn_search_dynamic_d_high_directstride_tcgen05_partial_0618_ccef_v1","kwargs":{"D_ORIG_":65,"NUM_D_PASSES_":1,"Q_NORM_PARTS_":5,"smem_bytes":126720,"validate":false},"threads":640}': 'dispatch_kernel_0091', '{"computed_smem_bytes":126720,"constants":[["K_MAX_",10],["D_ORIG_",512],["NUM_D_PASSES_",4],["Q_NORM_PARTS_",32]],"ir_name":"knn_search_dynamic_d_high_directstride_tcgen05_partial_0618_ccef_v1","kwargs":{"D_ORIG_":127,"NUM_D_PASSES_":1,"Q_NORM_PARTS_":8,"smem_bytes":126720,"validate":false},"threads":640}': 'dispatch_kernel_0092', '{"computed_smem_bytes":126720,"constants":[["K_MAX_",10],["D_ORIG_",512],["NUM_D_PASSES_",4],["Q_NORM_PARTS_",32]],"ir_name":"knn_search_dynamic_d_high_directstride_tcgen05_partial_0618_ccef_v1","kwargs":{"D_ORIG_":130,"NUM_D_PASSES_":2,"Q_NORM_PARTS_":9,"smem_bytes":126720,"validate":false},"threads":640}': 'dispatch_kernel_0093', '{"computed_smem_bytes":126720,"constants":[["K_MAX_",10],["D_ORIG_",512],["NUM_D_PASSES_",4],["Q_NORM_PARTS_",32]],"ir_name":"knn_search_dynamic_d_high_directstride_tcgen05_partial_0618_ccef_v1","kwargs":{"D_ORIG_":255,"NUM_D_PASSES_":2,"Q_NORM_PARTS_":16,"smem_bytes":126720,"validate":false},"threads":640}': 'dispatch_kernel_0094', '{"computed_smem_bytes":126720,"constants":[["K_MAX_",10],["D_ORIG_",512],["NUM_D_PASSES_",4],["Q_NORM_PARTS_",32]],"ir_name":"knn_search_dynamic_d_high_directstride_tcgen05_partial_0618_ccef_v1","kwargs":{"D_ORIG_":258,"NUM_D_PASSES_":3,"Q_NORM_PARTS_":17,"smem_bytes":126720,"validate":false},"threads":640}': 'dispatch_kernel_0095', '{"computed_smem_bytes":102144,"constants":[["K_MAX_",10]],"ir_name":"knn_search_dynamic_d512_q64_tcgen05_partial_0618_9286_v1","kwargs":{"smem_bytes":102144,"validate":false},"threads":512}': 'dispatch_kernel_0096', '{"computed_smem_bytes":143104,"constants":[["K_MAX_",10],["D_ORIG_",1024],["NUM_D_PASSES_",8],["Q_NORM_PARTS_",64]],"ir_name":"knn_search_dynamic_d768d1024_q32q16_tcgen05_partial_0618_9286_v1","kwargs":{"D_ORIG_":768,"NUM_D_PASSES_":6,"Q_NORM_PARTS_":48,"smem_bytes":143104,"validate":false},"threads":640}': 'dispatch_kernel_0097', '{"computed_smem_bytes":143104,"constants":[["K_MAX_",10],["D_ORIG_",1024],["NUM_D_PASSES_",8],["Q_NORM_PARTS_",64]],"ir_name":"knn_search_dynamic_d768d1024_q32q16_tcgen05_partial_0618_9286_v1","kwargs":{"D_ORIG_":1024,"NUM_D_PASSES_":8,"Q_NORM_PARTS_":64,"smem_bytes":143104,"validate":false},"threads":640}': 'dispatch_kernel_0098', '{"computed_smem_bytes":0,"constants":[["K_MAX_",64]],"ir_name":"knn_search_ext_k_capacity_truncate64_to_k_0618_28ec_v1","kwargs":{"K_MAX_":64,"smem_bytes":0,"validate":false},"threads":256}': 'dispatch_kernel_0099', '{"computed_smem_bytes":0,"constants":[["K_MAX_",64]],"ir_name":"knn_search_ext_k_capacity_q4096_m49152_merge24_0618_28ec_v1","kwargs":{"K_MAX_":64,"smem_bytes":0,"validate":false},"threads":32}': 'dispatch_kernel_0100', '{"computed_smem_bytes":165120,"constants":[["K_MAX_",64]],"ir_name":"knn_search_ext_k_capacity_q4096_m49152_partial_0618_28ec_v1","kwargs":{"K_MAX_":64,"smem_bytes":165120,"validate":false},"threads":512}': 'dispatch_kernel_0101', '{"computed_smem_bytes":165120,"constants":[["K_MAX_",64],["K_PREFIX_",8]],"ir_name":"knn_search_ext_k64_highq_prefix8_partial_0618_c2e0_v1","kwargs":{"K_MAX_":64,"K_PREFIX_":8,"smem_bytes":165120,"validate":false},"threads":512}': 'dispatch_kernel_0102', '{"computed_smem_bytes":0,"constants":[["K_MAX_",64],["K_PREFIX_",8]],"ir_name":"knn_search_ext_k64_highq_prefix8_merge_0618_c2e0_v1","kwargs":{"K_MAX_":64,"K_PREFIX_":8,"smem_bytes":1024,"validate":false},"threads":32}': 'dispatch_kernel_0103', '{"computed_smem_bytes":143104,"constants":[["K_MAX_",64]],"ir_name":"knn_search_d130_q64_k64_directpad_partial_0625_4b95_v1","kwargs":{"K_MAX_":64,"smem_bytes":143104,"validate":false},"threads":256}': 'dispatch_kernel_0104', '{"computed_smem_bytes":0,"constants":[["K_MAX_",64]],"ir_name":"knn_search_mma_split_merge_v1","kwargs":{"K_MAX_":64,"smem_bytes":1024,"validate":false},"threads":32}': 'dispatch_kernel_0105', '{"computed_smem_bytes":88576,"constants":[["K_MAX_",64]],"ir_name":"knn_search_d512_q32_k64_mergefast_partial_83da_r121_v1","kwargs":{"K_MAX_":64,"smem_bytes":88576,"validate":false},"threads":256}': 'dispatch_kernel_0106', '{"computed_smem_bytes":0,"constants":[["K_MAX_",64]],"ir_name":"knn_search_d512_q32_k64_distonlymerge_merge_177e_r121_v1","kwargs":{"K_MAX_":64,"smem_bytes":1024,"validate":false},"threads":32}': 'dispatch_kernel_0107', '{"computed_smem_bytes":128,"constants":[["D_",64],["K_MAX_",20],["NUM_WARPS_",4]],"ir_name":"knn_search_ivf_q12_m100_d64_k20_direct_6bea_r118_v1","kwargs":{"smem_bytes":48,"validate":false},"threads":128}': 'dispatch_kernel_0108', '{"computed_smem_bytes":32896,"constants":[["D_",3],["K_MAX_",32],["LOCAL_LIST_CAP_",16],["NUM_WARPS_",8]],"ir_name":"knn_search_dynamic_d3_k32_self_tile_0620_9d5c_v1","kwargs":{"smem_bytes":32864,"validate":false},"threads":256}': 'dispatch_kernel_0109', '{"computed_smem_bytes":165120,"constants":[["K_MAX_",64],["K_PREFIX_",8]],"ir_name":"knn_search_floor13_k64_q384_prefix8_partial_0622_f3ce_v1","kwargs":{"K_MAX_":64,"K_PREFIX_":8,"smem_bytes":165120,"validate":false},"threads":512}': 'dispatch_kernel_0110', '{"computed_smem_bytes":0,"constants":[["K_MAX_",64],["K_PREFIX_",8]],"ir_name":"knn_search_floor13_k64_q384_prefix8_merge_0622_f3ce_v1","kwargs":{"K_MAX_":64,"K_PREFIX_":8,"smem_bytes":1024,"validate":false},"threads":32}': 'dispatch_kernel_0111', '{"computed_smem_bytes":0,"constants":[["K_OUT_MAX_",80],["K_PREFIX_",8]],"ir_name":"knn_search_floor13_k80_prefix8_merge_0622_f3ce_v1","kwargs":{"K_OUT_MAX_":80,"K_PREFIX_":8,"smem_bytes":1024,"validate":false},"threads":32}': 'dispatch_kernel_0112', '{"computed_smem_bytes":112384,"constants":[["K_MAX_",10],["D_TOTAL_",3],["NUM_D_PASSES_",1],["Q_NORM_PARTS_",1]],"ir_name":"knn_search_dynamic_d_tiny_q128_m65536_tcgen05_partial_0618_c8b9_v1","kwargs":{"D_TOTAL_":33,"NUM_D_PASSES_":1,"Q_NORM_PARTS_":3,"smem_bytes":112384,"validate":false},"threads":640}': 'dispatch_kernel_0113', '{"computed_smem_bytes":143104,"constants":[["K_MAX_",10],["D_ORIG_",1024],["NUM_D_PASSES_",8],["Q_NORM_PARTS_",64]],"ir_name":"knn_search_dynamic_d768d1024_q32q16_tcgen05_partial_0618_9286_v1","kwargs":{"D_ORIG_":80,"NUM_D_PASSES_":1,"Q_NORM_PARTS_":5,"smem_bytes":143104,"validate":false},"threads":640}': 'dispatch_kernel_0114', '{"computed_smem_bytes":143104,"constants":[["K_MAX_",10],["D_ORIG_",1024],["NUM_D_PASSES_",8],["Q_NORM_PARTS_",64]],"ir_name":"knn_search_dynamic_d768d1024_q32q16_tcgen05_partial_0618_9286_v1","kwargs":{"D_ORIG_":160,"NUM_D_PASSES_":2,"Q_NORM_PARTS_":10,"smem_bytes":143104,"validate":false},"threads":640}': 'dispatch_kernel_0115', '{"computed_smem_bytes":143104,"constants":[["K_MAX_",10],["D_ORIG_",1024],["NUM_D_PASSES_",8],["Q_NORM_PARTS_",64]],"ir_name":"knn_search_dynamic_d768d1024_q32q16_tcgen05_partial_0618_9286_v1","kwargs":{"D_ORIG_":640,"NUM_D_PASSES_":5,"Q_NORM_PARTS_":40,"smem_bytes":143104,"validate":false},"threads":640}': 'dispatch_kernel_0116', '{"computed_smem_bytes":57600,"constants":[["K_MAX_",64]],"ir_name":"knn_search_d64_q128_m131072_k64_partial_0623_e157_v1","kwargs":{"K_MAX_":64,"smem_bytes":57600,"validate":false},"threads":512}': 'dispatch_kernel_0117', '{"computed_smem_bytes":145664,"constants":[["K_MAX_",10],["D_ORIG_",768],["NUM_D_PASSES_",6]],"ir_name":"knn_search_high_d_low_q_d768_q64_n256_partial_0268_v1","kwargs":{"D_ORIG_":768,"NUM_D_PASSES_":6,"smem_bytes":145664,"validate":false},"threads":640}': 'dispatch_kernel_0118', '{"computed_smem_bytes":0,"constants":[["K_MAX_",10]],"ir_name":"knn_search_high_d_low_q_d768_q64_norm_merge_0268_v1","kwargs":{"smem_bytes":1024,"validate":false},"threads":32}': 'dispatch_kernel_0119', '{"computed_smem_bytes":92672,"constants":[["K_MAX_",64]],"ir_name":"knn_search_d1024_q32_k64_targetd_partial_67ec_v1","kwargs":{"K_MAX_":64,"smem_bytes":92672,"validate":false},"threads":256}': 'dispatch_kernel_0120', '{"computed_smem_bytes":0,"constants":[["K_MAX_",64]],"ir_name":"knn_search_d1024_q32_k64_hiermerge8_group_f561_v2","kwargs":{"K_MAX_":64,"smem_bytes":1024,"validate":false},"threads":32}': 'dispatch_kernel_0121', '{"computed_smem_bytes":0,"constants":[["K_MAX_",64]],"ir_name":"knn_search_d1024_q32_k64_hiermerge8_final_f561_v2","kwargs":{"K_MAX_":64,"smem_bytes":1024,"validate":false},"threads":32}': 'dispatch_kernel_0122', '{"computed_smem_bytes":126720,"constants":[["K_MAX_",10]],"ir_name":"knn_search_d2048_q8_m16384_k10_partial_0623_6185_d51b_v1","kwargs":{"smem_bytes":126720,"validate":false},"threads":512}': 'dispatch_kernel_0123', '{"computed_smem_bytes":0,"constants":[["K_MAX_",10]],"ir_name":"knn_search_d2048_q8_m16384_k10_merge_0623_6185_d51b_v1","kwargs":{"smem_bytes":1024,"validate":false},"threads":32}': 'dispatch_kernel_0124', '{"computed_smem_bytes":159488,"constants":[["K_MAX_",10]],"ir_name":"knn_search_d4096_q4q8_m8192m16384_k10_partial_0623_5ff7_v1","kwargs":{"smem_bytes":159488,"validate":false},"threads":512}': 'dispatch_kernel_0125', '{"computed_smem_bytes":0,"constants":[["K_MAX_",10]],"ir_name":"knn_search_d4096_q4q8_m8192m16384_k10_merge_0623_5ff7_v1","kwargs":{"smem_bytes":1024,"validate":false},"threads":32}': 'dispatch_kernel_0126', '{"computed_smem_bytes":57600,"constants":[["K_MAX_",10]],"ir_name":"knn_search_target0628_d64_q256_m131072_k10_partial_885d_hmerge8_v1","kwargs":{"K_MAX_":10,"smem_bytes":57600,"validate":false},"threads":512}': 'dispatch_kernel_0127', '{"computed_smem_bytes":0,"constants":[["K_MAX_",10]],"ir_name":"knn_search_target0628_d64_q256_m131072_k10_groupmerge_885d_hmerge8_v1","kwargs":{"K_MAX_":10,"smem_bytes":1024,"validate":false},"threads":32}': 'dispatch_kernel_0128', '{"computed_smem_bytes":0,"constants":[["K_MAX_",10]],"ir_name":"knn_search_target0628_d64_q256_m131072_k10_finalmerge_885d_hmerge8_v1","kwargs":{"K_MAX_":10,"smem_bytes":1024,"validate":false},"threads":32}': 'dispatch_kernel_0129', '{"computed_smem_bytes":57600,"constants":[["K_MAX_",64]],"ir_name":"knn_search_target0628_d64_q512_m65536_k64_partial_e8f1_v1","kwargs":{"K_MAX_":64,"smem_bytes":57600,"validate":false},"threads":512}': 'dispatch_kernel_0130', '{"computed_smem_bytes":0,"constants":[["K_MAX_",64]],"ir_name":"knn_search_target0628_d64_q512_m65536_k64_groupmerge_dbaf_v1","kwargs":{"K_MAX_":64,"smem_bytes":1024,"validate":false},"threads":32}': 'dispatch_kernel_0131', '{"computed_smem_bytes":0,"constants":[["K_MAX_",64]],"ir_name":"knn_search_target0628_d64_q512_m65536_k64_finalmerge_dbaf_v1","kwargs":{"K_MAX_":64,"smem_bytes":1024,"validate":false},"threads":32}': 'dispatch_kernel_0132', '{"computed_smem_bytes":0,"constants":[["K_MAX_",64]],"ir_name":"knn_search_residual_q128_groupmerge76_fanin8_a8f5_v1","kwargs":{"K_MAX_":64,"smem_bytes":0,"validate":false},"threads":256}': 'dispatch_kernel_0134', '{"computed_smem_bytes":0,"constants":[["K_MAX_",64]],"ir_name":"knn_search_d256_split256_rows8_finalmerge8compact_1056_v1","kwargs":{"K_MAX_":64,"smem_bytes":1024,"validate":false},"threads":256}': 'dispatch_kernel_0135', '{"computed_smem_bytes":170240,"constants":[["K_MAX_",10],["D_ORIG_",768],["NUM_D_PASSES_",8],["Q_NORM_PARTS_",48]],"ir_name":"knn_search_q8_blockm256_two_stripe_tmem_8d4fe4ead6cd_v1","kwargs":{"D_ORIG_":768,"NUM_D_PASSES_":6,"Q_NORM_PARTS_":48,"smem_bytes":170240,"validate":false},"threads":640}': 'dispatch_kernel_0136', '{"computed_smem_bytes":159488,"constants":[["K_MAX_",64]],"ir_name":"knn_search_target0627_d768_q64_m65536_k64_partial_6472_v2","kwargs":{"K_MAX_":64,"smem_bytes":159488,"validate":false},"threads":256}': 'dispatch_kernel_0137', '{"computed_smem_bytes":0,"constants":[["K_MAX_",64]],"ir_name":"knn_search_target0627_d768_q64_m65536_k64_merge_6472_v2","kwargs":{"K_MAX_":64,"smem_bytes":1024,"validate":false},"threads":32}': 'dispatch_kernel_0138', '{"computed_smem_bytes":178432,"constants":[["K_MAX_",10],["D_ORIG_",1024],["NUM_D_PASSES_",8],["Q_NORM_PARTS_",64]],"ir_name":"knn_search_q8_blockm256_two_stripe_tmem_8d4fe4ead6cd_v1","kwargs":{"D_ORIG_":1024,"NUM_D_PASSES_":8,"Q_NORM_PARTS_":64,"smem_bytes":178432,"validate":false},"threads":640}': 'dispatch_kernel_0139', '{"computed_smem_bytes":0,"constants":[["K_MAX_",64]],"ir_name":"knn_search_target0628_d1024_q32_m65536_k64_group_merge16_9571_v2","kwargs":{"K_MAX_":64,"smem_bytes":1024,"validate":false},"threads":32}': 'dispatch_kernel_0140', '{"computed_smem_bytes":0,"constants":[["K_MAX_",64]],"ir_name":"knn_search_target0628_d1024_q32_m65536_k64_final_merge16_9571_v2","kwargs":{"K_MAX_":64,"smem_bytes":1024,"validate":false},"threads":32}': 'dispatch_kernel_0141', '{"computed_smem_bytes":62848,"constants":[["K_MAX_",10]],"ir_name":"knn_search_d4096_q1_m65536_k10_partial_qcache_pipe_69ea_v1","kwargs":{"smem_bytes":62848,"validate":false},"threads":512}': 'dispatch_kernel_0142', '{"computed_smem_bytes":0,"constants":[["K_MAX_",10],["MERGE_SLOTS_",5]],"ir_name":"knn_search_mma_split_merge_v1","kwargs":{"MERGE_SLOTS_":10,"smem_bytes":1024,"validate":false},"threads":32}': 'dispatch_kernel_0143', '{"computed_smem_bytes":192768,"constants":[["K_MAX_",10]],"ir_name":"knn_search_target0630_d4096_q4_m32768_k10_partial_qreuse_v1","kwargs":{"smem_bytes":192768,"validate":false},"threads":512}': 'dispatch_kernel_0144', '{"computed_smem_bytes":0,"constants":[["K_MAX_",10]],"ir_name":"knn_search_target0627_d4096_q4_m32768_k10_merge128_r221_v1","kwargs":{"smem_bytes":1024,"validate":false},"threads":32}': 'dispatch_kernel_0145', '{"computed_smem_bytes":105984,"constants":[["D_ORIG_",4096],["NUM_D_PASSES_",16],["K_MAX_",64]],"ir_name":"knn_search_target0628_d4096_q4_m8192_k64_partial_e750_v1","kwargs":{"smem_bytes":105984,"validate":false},"threads":256}': 'dispatch_kernel_0146', '{"computed_smem_bytes":0,"constants":[["K_MAX_",64]],"ir_name":"group_merge_ir","kwargs":{"smem_bytes":1024,"validate":false},"threads":32}': 'dispatch_kernel_0147', '{"computed_smem_bytes":0,"constants":[["K_MAX_",64]],"ir_name":"final_merge_ir","kwargs":{"smem_bytes":1024,"validate":false},"threads":32}': 'dispatch_kernel_0148', '{"computed_smem_bytes":106240,"constants":[["K_MAX_",64]],"ir_name":"knn_search_q64_warp_distributed_state_04b4_v1","kwargs":{"K_MAX_":64,"smem_bytes":106240,"validate":false},"threads":256}': 'dispatch_kernel_0149', '{"computed_smem_bytes":0,"constants":[["K_MAX_",64]],"ir_name":"knn_search_q64_tail_groupmerge76_split152_3dee_v1","kwargs":{"K_MAX_":64,"smem_bytes":0,"validate":false},"threads":128}': 'dispatch_kernel_0150', '{"computed_smem_bytes":0,"constants":[["K_MAX_",64]],"ir_name":"knn_search_q64_pairedowner_finalmerge_cce0_v1","kwargs":{"K_MAX_":64,"smem_bytes":1024,"validate":false},"threads":256}': 'dispatch_kernel_0151', '{"computed_smem_bytes":188160,"constants":[["K_MAX_",64]],"ir_name":"knn_search_q64_tail_split152_full_wave_partial_3dee_v1","kwargs":{"K_MAX_":64,"smem_bytes":188160,"validate":false},"threads":256}': 'dispatch_kernel_0152', '{"computed_smem_bytes":106240,"constants":[["K_MAX_",64]],"ir_name":"knn_search_q64_tail_plus_812c_sentinel_warp_state_361b_v2","kwargs":{"K_MAX_":64,"smem_bytes":106240,"validate":false},"threads":256}': 'dispatch_kernel_0153', '{"computed_smem_bytes":0,"constants":[["K_MAX_",64]],"ir_name":"knn_search_q64_pairedowner_groupmerge_cce0_v1","kwargs":{"K_MAX_":64,"smem_bytes":0,"validate":false},"threads":128}': 'dispatch_kernel_0154', '{"computed_smem_bytes":109440,"constants":[["K_MAX_",64]],"ir_name":"knn_search_q65_fused_04b4_low_depth_tail_d436_v1","kwargs":{"K_MAX_":64,"smem_bytes":109348,"validate":false},"threads":256}': 'dispatch_kernel_0155', '{"computed_smem_bytes":0,"constants":[["K_MAX_",64]],"ir_name":"knn_search_q65_tail_groupmerge_d436_v1","kwargs":{"K_MAX_":64,"smem_bytes":0,"validate":false},"threads":128}': 'dispatch_kernel_0156', '{"computed_smem_bytes":0,"constants":[["K_MAX_",64]],"ir_name":"knn_search_q65_tail_finalmerge_d436_v1","kwargs":{"K_MAX_":64,"smem_bytes":0,"validate":false},"threads":32}': 'dispatch_kernel_0157', '{"computed_smem_bytes":108800,"constants":[],"ir_name":"knn_search_q4096_lowk_k5partial_0613_r49_48e9_v1","kwargs":{"smem_bytes":108800,"validate":false},"threads":640}': 'dispatch_kernel_0158', '{"computed_smem_bytes":0,"constants":[["K_STRIDE_",5],["K_OUT_MAX_",5]],"ir_name":"knn_search_q4096_lowk_k5partial_split9_merge_0613_r51_48e9_v1","kwargs":{"K_OUT_MAX_":5,"K_STRIDE_":5,"smem_bytes":1024,"validate":false},"threads":32}': 'dispatch_kernel_0159', '{"computed_smem_bytes":0,"constants":[["K_STRIDE_",10],["K_OUT_MAX_",8]],"ir_name":"knn_search_q4096_lowk_k8_stride10_out8_merge_0613_r52_48e9_v1","kwargs":{"K_OUT_MAX_":8,"K_STRIDE_":10,"smem_bytes":1024,"validate":false},"threads":32}': 'dispatch_kernel_0160', '{"computed_smem_bytes":0,"constants":[],"ir_name":"knn_search_scalar_capacity_pad_bf16_0705_v1","kwargs":{"smem_bytes":0,"validate":false},"threads":256}': 'dispatch_kernel_0161', '{"computed_smem_bytes":165120,"constants":[["K_MAX_",32],["EXPOSE_COL_COHORTS",0]],"ir_name":"knn_search_mma_split_partial_v1","kwargs":{"EXPOSE_COL_COHORTS":1,"smem_bytes":165120,"validate":false},"threads":512}': 'dispatch_kernel_0162', '{"computed_smem_bytes":0,"constants":[["K_MAX_",32]],"ir_name":"knn_search_mma_split_merge_v1","kwargs":{"smem_bytes":1024,"validate":false},"threads":32}': 'dispatch_kernel_0163', '{"computed_smem_bytes":0,"constants":[["D_",128],["K_MAX_",10],["BLOCK_M_",512],["NUM_WARPS_",8]],"ir_name":"knn_search_warp_split_partial_v1","kwargs":{"smem_bytes":0,"validate":false},"threads":256}': 'dispatch_kernel_0164', '{"computed_smem_bytes":20480,"constants":[["K_MAX_",10],["NUM_WARPS_",8],["PARTIAL_ELEMS_PER_TILE_",80]],"ir_name":"knn_search_warp_split_merge_v1","kwargs":{"smem_bytes":20480,"validate":false},"threads":256}': 'dispatch_kernel_0165', '{"computed_smem_bytes":640,"constants":[["D_",128],["K_MAX_",10],["NUM_WARPS_",8]],"ir_name":"knn_search_warp_direct_v1","kwargs":{"smem_bytes":640,"validate":false},"threads":256}': 'dispatch_kernel_0166', '{"computed_smem_bytes":106240,"constants":[["K_MAX_",64]],"ir_name":"knn_search_q64_native_pairedowner_partial_cce0_v1","kwargs":{"K_MAX_":64,"smem_bytes":106240,"validate":false},"threads":256}': 'dispatch_kernel_0167', '{"computed_smem_bytes":0,"constants":[["K_MAX_",64]],"ir_name":"knn_search_d256_split256_groupmerge64_ownerless_7ce6_v1","kwargs":{"K_MAX_":64,"smem_bytes":0,"validate":false},"threads":32}': 'dispatch_kernel_0168', '{"computed_smem_bytes":0,"constants":[["K_MAX_",64]],"ir_name":"knn_search_d256_split256_groupmerge64_tile64_d51ts_v1","kwargs":{"K_MAX_":64,"smem_bytes":0,"validate":false},"threads":64}': 'dispatch_kernel_0169', '{"computed_smem_bytes":0,"constants":[["K_MAX_",64]],"ir_name":"knn_search_d256_split256_groupmerge64_warp3136_v1","kwargs":{"K_MAX_":64,"smem_bytes":0,"validate":false},"threads":32}': 'dispatch_kernel_0170', '{"computed_smem_bytes":0,"constants":[["K_MAX_",64]],"ir_name":"knn_search_d256_split256_rows8_finalmerge8_hier9c25_v1","kwargs":{"K_MAX_":64,"smem_bytes":1024,"validate":false},"threads":256}': 'dispatch_kernel_0171', '{"computed_smem_bytes":0,"constants":[["K_MAX_",64]],"ir_name":"knn_search_d256_split256_groupmerge64_hier9c25_v1","kwargs":{"K_MAX_":64,"smem_bytes":1024,"validate":false},"threads":256}': 'dispatch_kernel_0172', '{"computed_smem_bytes":0,"constants":[["K_MAX_",64]],"ir_name":"knn_search_d256_split256_rows8_merge_v1","kwargs":{"K_MAX_":64,"smem_bytes":1024,"validate":false},"threads":256}': 'dispatch_kernel_0173', '{"computed_smem_bytes":0,"constants":[["K_MAX_",64]],"ir_name":"knn_search_d256_localcta_rows8_merge_v1","kwargs":{"K_MAX_":64,"smem_bytes":1024,"validate":false},"threads":256}': 'dispatch_kernel_0174', '{"computed_smem_bytes":256,"constants":[["K_MAX_",10],["M_MAX_",32]],"ir_name":"knn_search_lowd_ivf_direct_dispatch0610_r2_v1","kwargs":{"smem_bytes":256,"validate":false},"threads":32}': 'dispatch_kernel_0175', '{"computed_smem_bytes":16512,"constants":[["D_",2],["K_MAX_",64],["LOCAL_LIST_CAP_",8],["NUM_WARPS_",8]],"ir_name":"knn_search_lowd_dbscan_direct_dispatch0610_r1_v1","kwargs":{"smem_bytes":16480,"validate":false},"threads":256}': 'dispatch_kernel_0176', '{"computed_smem_bytes":16512,"constants":[["D_",2],["K_MAX_",64],["LOCAL_LIST_CAP_",8],["NUM_WARPS_",8]],"ir_name":"knn_search_lowd_dbscan_d2_coopmerge_0612_r23_6e85_v1","kwargs":{"smem_bytes":16480,"validate":false},"threads":256}': 'dispatch_kernel_0177', '{"computed_smem_bytes":16384,"constants":[["D_",2],["K_MAX_",64],["LOCAL_CAP_",16]],"ir_name":"knn_search_lowd_dbscan_d2_direct_0611_r22_6e85_v1","kwargs":{"smem_bytes":16384,"validate":false},"threads":128}': 'dispatch_kernel_0178', '{"computed_smem_bytes":165120,"constants":[["K_MAX_",64]],"ir_name":"knn_search_k64_q4096split80_twotile_distanceonly_branchpruned_partial_0612_r30_11c1_v1","kwargs":{"K_MAX_":64,"smem_bytes":165120,"validate":false},"threads":512}': 'dispatch_kernel_0179', '{"computed_smem_bytes":0,"constants":[["K_MAX_",64]],"ir_name":"knn_search_k64_q4096split80_merge10_0612_r22_4e2c_v1","kwargs":{"K_MAX_":64,"smem_bytes":1024,"validate":false},"threads":32}': 'dispatch_kernel_0180', '{"computed_smem_bytes":165120,"constants":[["K_MAX_",32],["EXPOSE_COL_COHORTS",0]],"ir_name":"knn_search_mma_split_partial_v1","kwargs":{"EXPOSE_COL_COHORTS":1,"K_MAX_":64,"smem_bytes":165120,"validate":false},"threads":512}': 'dispatch_kernel_0181', '{"computed_smem_bytes":0,"constants":[["K_MAX_",64]],"ir_name":"knn_search_k64_stable_merge_0612_r23_4e96_v1","kwargs":{"K_MAX_":64,"smem_bytes":1024,"validate":false},"threads":32}': 'dispatch_kernel_0182', '{"computed_smem_bytes":0,"constants":[["K_MAX_",32]],"ir_name":"knn_search_mma_split_merge_v1","kwargs":{"K_MAX_":64,"smem_bytes":1024,"validate":false},"threads":32}': 'dispatch_kernel_0105', '{"computed_smem_bytes":165120,"constants":[["K_MAX_",64]],"ir_name":"knn_search_k64_q4096split80_twotile_partial_0612_r25_4e2c_v1","kwargs":{"K_MAX_":64,"smem_bytes":165120,"validate":false},"threads":512}': 'dispatch_kernel_0183', '{"computed_smem_bytes":165120,"constants":[["K_MAX_",32],["EXPOSE_COL_COHORTS",0]],"ir_name":"knn_search_mma_split_partial_v1","kwargs":{"EXPOSE_COL_COHORTS":0,"K_MAX_":31,"smem_bytes":165120,"validate":false},"threads":512}': 'dispatch_kernel_0184', '{"computed_smem_bytes":0,"constants":[["K_MAX_",32]],"ir_name":"knn_search_k32_q128_split148_guarded_merge_r60_v1","kwargs":{"K_MAX_":31,"smem_bytes":1024,"validate":false},"threads":32}': 'dispatch_kernel_0185', '{"computed_smem_bytes":0,"constants":[["K_MAX_",32]],"ir_name":"knn_search_k32_q128_split148_guarded_merge_r60_v1","kwargs":{"K_MAX_":32,"smem_bytes":1024,"validate":false},"threads":32}': 'dispatch_kernel_0186', '{"computed_smem_bytes":106240,"constants":[["K_MAX_",10]],"ir_name":"knn_search_d768_q64_m65536_k10_partial_0623_e35f_v1","kwargs":{"smem_bytes":106240,"validate":false},"threads":512}': 'dispatch_kernel_0187', '{"computed_smem_bytes":10240,"constants":[["K_MAX_",10]],"ir_name":"knn_search_dynamic_d3_tile_reduce_partial_0618_c8b9_v1","kwargs":{"smem_bytes":10240,"validate":false},"threads":256}': 'dispatch_kernel_0188', '{"computed_smem_bytes":640,"constants":[["K_MAX_",10]],"ir_name":"knn_search_lowd_non128_tile_reduce_merge_0615_7d36_v2","kwargs":{"smem_bytes":640,"validate":false},"threads":256}': 'dispatch_kernel_0189', '{"computed_smem_bytes":165120,"constants":[["K_MAX_",64]],"ir_name":"knn_search_k64_q4096split80_twotile_distanceonly_stagingunrolled_partial_0612_r31_11c1_v1","kwargs":{"K_MAX_":64,"smem_bytes":165120,"validate":false},"threads":512}': 'dispatch_kernel_0192', '{"computed_smem_bytes":0,"constants":[["K_MAX_",64]],"ir_name":"knn_search_k64_q4096split80_indexfastmerge10_0612_r32_11c1_v1","kwargs":{"K_MAX_":64,"smem_bytes":1024,"validate":false},"threads":32}': 'dispatch_kernel_0193', '{"computed_smem_bytes":108800,"constants":[],"ir_name":"knn_search_q4096_lowk_k1partial_minpair_0613_r46_48e9_lowk_k1top1_v1","kwargs":{"smem_bytes":108800,"validate":false},"threads":640}': 'dispatch_kernel_0194', '{"computed_smem_bytes":0,"constants":[],"ir_name":"knn_search_q4096_lowk_k1partial_minpair_merge_0613_r46_48e9_lowk_k1top1_v1","kwargs":{"smem_bytes":1024,"validate":false},"threads":32}': 'dispatch_kernel_0195', '{"computed_smem_bytes":0,"constants":[["K_STRIDE_",5],["K_OUT_MAX_",5]],"ir_name":"knn_search_q4096_lowk_k5partial_merge_0613_r49_48e9_v1","kwargs":{"K_OUT_MAX_":5,"K_STRIDE_":5,"smem_bytes":1024,"validate":false},"threads":32}': 'dispatch_kernel_0196', '{"computed_smem_bytes":0,"constants":[["K_STRIDE_",10],["K_OUT_MAX_",5]],"ir_name":"knn_search_q4096_lowk_k5_stride10_merge_0613_r48_48e9_v1","kwargs":{"smem_bytes":1024,"validate":false},"threads":32}': 'dispatch_kernel_0197', '{"computed_smem_bytes":0,"constants":[["K_STRIDE_",2],["K_OUT_MAX_",2]],"ir_name":"knn_search_q4096_lowk_k2partial_merge_0613_r45_48e9_v1","kwargs":{"K_OUT_MAX_":1,"K_STRIDE_":2,"smem_bytes":1024,"validate":false},"threads":32}': 'dispatch_kernel_0198', '{"computed_smem_bytes":0,"constants":[["K_STRIDE_",2],["K_OUT_MAX_",2]],"ir_name":"knn_search_q4096_lowk_k2partial_merge_0613_r45_48e9_v1","kwargs":{"K_OUT_MAX_":2,"K_STRIDE_":2,"smem_bytes":1024,"validate":false},"threads":32}': 'dispatch_kernel_0199', '{"computed_smem_bytes":108800,"constants":[],"ir_name":"knn_search_q4096_lowk_k1partial_0613_r44_48e9_v1","kwargs":{"smem_bytes":108800,"validate":false},"threads":640}': 'dispatch_kernel_0200', '{"computed_smem_bytes":0,"constants":[],"ir_name":"knn_search_q4096_lowk_k1partial_merge_0613_r44_48e9_v1","kwargs":{"smem_bytes":1024,"validate":false},"threads":32}': 'dispatch_kernel_0201', '{"computed_smem_bytes":0,"constants":[["K_OUT_MAX_",2]],"ir_name":"knn_search_mma_split_merge_q4096_lowk_v1","kwargs":{"K_OUT_MAX_":1,"smem_bytes":1024,"validate":false},"threads":32}': 'dispatch_kernel_0202', '{"computed_smem_bytes":0,"constants":[["K_OUT_MAX_",2]],"ir_name":"knn_search_mma_split_merge_q4096_lowk_v1","kwargs":{"K_OUT_MAX_":2,"smem_bytes":1024,"validate":false},"threads":32}': 'dispatch_kernel_0203', '{"computed_smem_bytes":0,"constants":[["K_MAX_",64]],"ir_name":"knn_search_k64_q128split512_groupmerge64_0613_r43_11c1_v1","kwargs":{"K_MAX_":64,"smem_bytes":1024,"validate":false},"threads":32}': 'dispatch_kernel_0204', '{"computed_smem_bytes":0,"constants":[["K_MAX_",64]],"ir_name":"knn_search_k64_q128split512_finalmerge32_0613_r43_11c1_v1","kwargs":{"K_MAX_":64,"smem_bytes":1024,"validate":false},"threads":32}': 'dispatch_kernel_0205', '{"computed_smem_bytes":0,"constants":[["K_MAX_",64]],"ir_name":"knn_search_k64_q128split256_merge1024_0613_r37_11c1_v1","kwargs":{"K_MAX_":64,"smem_bytes":1024,"validate":false},"threads":32}': 'dispatch_kernel_0206', '{"computed_smem_bytes":108800,"constants":[],"ir_name":"knn_search_q4096_lowk_k1partial_onestage_0614_r2_3ff5_v1","kwargs":{"smem_bytes":108800,"validate":false},"threads":640}': 'dispatch_kernel_0207', '{"computed_smem_bytes":0,"constants":[],"ir_name":"knn_search_q4096_lowk_k1partial_onestage_merge_0614_r2_3ff5_v1","kwargs":{"smem_bytes":1024,"validate":false},"threads":32}': 'dispatch_kernel_0208', '{"computed_smem_bytes":0,"constants":[["K_MAX_",64]],"ir_name":"knn_search_k64_q128split512_groupmerge64_indexfast_0614_r24_k64thin_v1","kwargs":{"K_MAX_":64,"smem_bytes":1024,"validate":false},"threads":32}': 'dispatch_kernel_0209', '{"computed_smem_bytes":0,"constants":[["K_MAX_",64]],"ir_name":"knn_search_k64_q128split512_finalmerge32_indexfast_0614_r24_k64thin_v1","kwargs":{"K_MAX_":64,"smem_bytes":1024,"validate":false},"threads":32}': 'dispatch_kernel_0210', '{"computed_smem_bytes":165120,"constants":[["K_MAX_",64],["K_PREFIX_",9]],"ir_name":"knn_search_k64_q4096split79_localprefix9_partial_0615_r32_edd7_v1","kwargs":{"smem_bytes":165120,"validate":false},"threads":512}': 'dispatch_kernel_0211', '{"computed_smem_bytes":0,"constants":[["K_MAX_",64],["K_PREFIX_READ_",8],["K_STRIDE_",9]],"ir_name":"knn_search_k64_q4096split79_localprefix_certmerge_0615_r32_edd7_v1","kwargs":{"smem_bytes":1024,"validate":false},"threads":32}': 'dispatch_kernel_0212', '{"computed_smem_bytes":0,"constants":[],"ir_name":"knn_search_k64_q4096split79_localprefix_certflag_init_0615_r32_edd7_v1","kwargs":{"smem_bytes":1024,"validate":false},"threads":32}': 'dispatch_kernel_0213', '{"computed_smem_bytes":0,"constants":[["K_MAX_",64],["K_PREFIX_READ_",8],["K_STRIDE_",9]],"ir_name":"knn_search_k64_q4096split79_localprefix_cert_0615_r32_edd7_v1","kwargs":{"smem_bytes":1024,"validate":false},"threads":32}': 'dispatch_kernel_0214', '{"computed_smem_bytes":165120,"constants":[["K_MAX_",64],["K_PREFIX_",8]],"ir_name":"knn_search_k64_q4096split79_localprefix_partial_0614_r36_edd7_v1","kwargs":{"K_MAX_":64,"K_PREFIX_":8,"smem_bytes":165120,"validate":false},"threads":512}': 'dispatch_kernel_0215', '{"computed_smem_bytes":0,"constants":[["K_MAX_",64],["K_PREFIX_",8]],"ir_name":"knn_search_k64_q4096split79_localprefix_merge_0614_r36_edd7_v1","kwargs":{"K_MAX_":64,"K_PREFIX_":8,"smem_bytes":1024,"validate":false},"threads":32}': 'dispatch_kernel_0216', '{"computed_smem_bytes":165120,"constants":[["K_MAX_",64],["K_PREFIX_",7]],"ir_name":"knn_search_k64_q4096split79_localprefix7_partial_0615_245d_v1","kwargs":{"smem_bytes":165120,"validate":false},"threads":512}': 'dispatch_kernel_0217', '{"computed_smem_bytes":0,"constants":[["K_MAX_",64],["K_PREFIX_READ_",6],["K_STRIDE_",7]],"ir_name":"knn_search_k64_q4096split79_localprefix6_rowflag_fusedcert_merge_0615_r36_e4cb_v1","kwargs":{"smem_bytes":1024,"validate":false},"threads":32}': 'dispatch_kernel_0218', '{"computed_smem_bytes":0,"constants":[["K_MAX_",64],["K_PREFIX_READ_",6],["K_STRIDE_",7]],"ir_name":"knn_search_k64_q4096split79_localprefix6_certmerge_0615_245d_v1","kwargs":{"smem_bytes":1024,"validate":false},"threads":32}': 'dispatch_kernel_0219', '{"computed_smem_bytes":0,"constants":[],"ir_name":"knn_search_k64_q4096split79_localprefix6_certflag_init_0615_245d_v1","kwargs":{"smem_bytes":1024,"validate":false},"threads":32}': 'dispatch_kernel_0220', '{"computed_smem_bytes":0,"constants":[["K_MAX_",64],["K_PREFIX_READ_",6],["K_STRIDE_",7]],"ir_name":"knn_search_k64_q4096split79_localprefix6_cert_0615_245d_v1","kwargs":{"smem_bytes":1024,"validate":false},"threads":32}': 'dispatch_kernel_0221', '{"computed_smem_bytes":165120,"constants":[["K_MAX_",64]],"ir_name":"knn_search_k48_q4096split128_m32768_k48scratch_partial_0614_ddbc_q4096k48_v2","kwargs":{"K_MAX_":64,"smem_bytes":165120,"validate":false},"threads":512}': 'dispatch_kernel_0222', '{"computed_smem_bytes":0,"constants":[["K_MAX_",48]],"ir_name":"knn_search_k48_q4096split128_m32768_k48scratch_merge16_0614_ddbc_q4096k48_v2","kwargs":{"K_MAX_":48,"smem_bytes":1024,"validate":false},"threads":32}': 'dispatch_kernel_0223', '{"computed_smem_bytes":165120,"constants":[["K_MAX_",64],["EXPOSE_COL_COHORTS",1]],"ir_name":"knn_search_mma_split_partial_v1","kwargs":{"EXPOSE_COL_COHORTS":1,"K_MAX_":64,"smem_bytes":165120,"validate":false},"threads":512}': 'dispatch_kernel_0181', '{"computed_smem_bytes":0,"constants":[["K_MAX_",64]],"ir_name":"knn_search_k48_q128split512_finalmerge32_strided_0614_ddbc_v1","kwargs":{"K_MAX_":64,"smem_bytes":1024,"validate":false},"threads":32}': 'dispatch_kernel_0224', '{"computed_smem_bytes":0,"constants":[["K_MAX_",64],["PARTIAL_LISTS_",512],["SPLITS_PER_LANE_",16]],"ir_name":"knn_search_blind_k64_q4096_m32768_merge16_0614_1968_v1","kwargs":{"K_MAX_":64,"PARTIAL_LISTS_":512,"SPLITS_PER_LANE_":16,"smem_bytes":0,"validate":false},"threads":32}': 'dispatch_kernel_0225', '{"computed_smem_bytes":108800,"constants":[],"ir_name":"knn_search_k1_top1_margin_qfull_partial_0614_r93_v1","kwargs":{"smem_bytes":108800,"validate":false},"threads":640}': 'dispatch_kernel_0226', '{"computed_smem_bytes":0,"constants":[],"ir_name":"knn_search_k1_top1_margin_0614_r93_merge8_v1","kwargs":{"smem_bytes":1024,"validate":false},"threads":128}': 'dispatch_kernel_0227', '{"computed_smem_bytes":0,"constants":[],"ir_name":"knn_search_k1_top1_margin_qfull_merge_0614_r93_v1","kwargs":{"smem_bytes":1024,"validate":false},"threads":32}': 'dispatch_kernel_0228', '{"computed_smem_bytes":5120,"constants":[["D_",128],["K_MAX_",10],["BLOCK_M_",640],["NUM_ROW_WORKERS_",64],["SUBWARP_WIDTH_",4],["SUBWARPS_PER_WARP_",8],["LOCAL_LIST_CAP_",10]],"ir_name":"knn_search_lowq_tile_reduce_partial_0614_r10_e864_blockm640_v1","kwargs":{"smem_bytes":5120,"validate":false},"threads":256}': 'dispatch_kernel_0229', '{"computed_smem_bytes":640,"constants":[["K_MAX_",10]],"ir_name":"knn_search_lowq_tile_reduce_merge_0614_r10_e864_blockm640_v1","kwargs":{"smem_bytes":640,"validate":false},"threads":256}': 'dispatch_kernel_0230', '{"computed_smem_bytes":5120,"constants":[["D_",128],["K_MAX_",10],["BLOCK_M_",640],["NUM_ROW_WORKERS_",64],["SUBWARP_WIDTH_",4],["SUBWARPS_PER_WARP_",8],["LOCAL_LIST_CAP_",10]],"ir_name":"knn_search_lowq_tile_reduce_partial_0615_196e_blockm640_tailguard_v1","kwargs":{"smem_bytes":5120,"validate":false},"threads":256}': 'dispatch_kernel_0231', '{"computed_smem_bytes":165120,"constants":[["K_MAX_",64],["K_PREFIX_",6]],"ir_name":"knn_search_k64_q4096split79_localprefix6_partial_0615_r5_9a85_v1","kwargs":{"smem_bytes":165120,"validate":false},"threads":512}': 'dispatch_kernel_0232', '{"computed_smem_bytes":0,"constants":[["K_MAX_",64],["K_PREFIX_READ_",5],["K_STRIDE_",6]],"ir_name":"knn_search_k64_q4096split79_localprefix5_rowflag_fusedcert_merge_0615_r5_9a85_v1","kwargs":{"smem_bytes":1024,"validate":false},"threads":32}': 'dispatch_kernel_0233', '{"computed_smem_bytes":5120,"constants":[["D_",128],["K_MAX_",10],["BLOCK_M_",896],["NUM_ROW_WORKERS_",64],["SUBWARP_WIDTH_",4],["SUBWARPS_PER_WARP_",8],["LOCAL_LIST_CAP_",14]],"ir_name":"knn_search_lowq_tile_reduce_partial_0614_r11_e864_blockm896_v1","kwargs":{"smem_bytes":5120,"validate":false},"threads":256}': 'dispatch_kernel_0234', '{"computed_smem_bytes":640,"constants":[["K_MAX_",10]],"ir_name":"knn_search_lowq_tile_reduce_merge_0614_r11_e864_blockm896_v1","kwargs":{"smem_bytes":640,"validate":false},"threads":256}': 'dispatch_kernel_0235', '{"computed_smem_bytes":165120,"constants":[["K_MAX_",64]],"ir_name":"knn_search_80a5_blocker_k32_q4096_m32768_split128_k34scratch_partial_v1","kwargs":{"K_MAX_":64,"smem_bytes":165120,"validate":false},"threads":512}': 'dispatch_kernel_0236', '{"computed_smem_bytes":0,"constants":[["K_MAX_",34]],"ir_name":"knn_search_80a5_blocker_k32_q4096_m32768_split128_k34scratch_merge16_v1","kwargs":{"K_MAX_":34,"smem_bytes":1024,"validate":false},"threads":32}': 'dispatch_kernel_0237', '{"computed_smem_bytes":21120,"constants":[["K_MAX_",10],["LOCAL_LIST_CAP_",10],["ROWS_PER_THREAD_",256],["NUM_GROUPS_",8],["GROUP_SIZE_",32]],"ir_name":"knn_search_dynamic_lowd_d1_warpmerge_0624_05a2_v1","kwargs":{"smem_bytes":21120,"validate":false},"threads":256}': 'dispatch_kernel_0238', '{"computed_smem_bytes":20608,"constants":[["K_MAX_",10],["LOCAL_LIST_CAP_",10],["ROWS_PER_THREAD_",256],["NUM_WARPS_",8]],"ir_name":"knn_search_dynamic_lowd_direct_0624_3676_v1","kwargs":{"smem_bytes":20576,"validate":false},"threads":256}': 'dispatch_kernel_0239', '{"computed_smem_bytes":20608,"constants":[["K_MAX_",10],["LOCAL_LIST_CAP_",10],["ROWS_PER_THREAD_",256]],"ir_name":"knn_search_dynamic_lowd_d1_serialmerge_0624_3676_v1","kwargs":{"smem_bytes":20576,"validate":false},"threads":256}': 'dispatch_kernel_0240', '{"computed_smem_bytes":159488,"constants":[["K_MAX_",10],["NUM_D_PASSES_",32]],"ir_name":"knn_search_target_d4096_q8_m16384_k10_tcgen05_partial_0623_5ff7_v1","kwargs":{"smem_bytes":159488,"validate":false},"threads":512}': 'dispatch_kernel_0241', '{"computed_smem_bytes":0,"constants":[],"ir_name":"knn_search_k1_q128_split148_merge_0622_4201_v1","kwargs":{"smem_bytes":1024,"validate":false},"threads":32}': 'dispatch_kernel_0242', '{"computed_smem_bytes":0,"constants":[["K_OUT_",32],["K_PREFIX_",8]],"ir_name":"knn_search_q4096_m32768_k32_prefix8_merge_3053_v1","kwargs":{"K_OUT_":32,"K_PREFIX_":8,"smem_bytes":1024,"validate":false},"threads":32}': 'dispatch_kernel_0243', '{"computed_smem_bytes":0,"constants":[["K_MAX_",64],["K_PREFIX_",8]],"ir_name":"knn_search_blind_k64_q4096_m32768_prefix8_merge_5132_v1","kwargs":{"K_MAX_":64,"K_PREFIX_":8,"smem_bytes":1024,"validate":false},"threads":32}': 'dispatch_kernel_0244', '{"computed_smem_bytes":0,"constants":[["K_MAX_",64]],"ir_name":"knn_search_d512_q32_k64_mergefast_merge_83da_r121_v1","kwargs":{"K_MAX_":64,"smem_bytes":1024,"validate":false},"threads":32}': 'dispatch_kernel_0245', '{"computed_smem_bytes":88576,"constants":[["K_MAX_",64]],"ir_name":"knn_search_d512_q32_k64_q32active_partial_abaf_r120_v1","kwargs":{"K_MAX_":64,"smem_bytes":88576,"validate":false},"threads":256}': 'dispatch_kernel_0246', '{"computed_smem_bytes":0,"constants":[["K_MAX_",64]],"ir_name":"knn_search_d512_q32_k64_q32active_merge_abaf_r120_v1","kwargs":{"K_MAX_":64,"smem_bytes":1024,"validate":false},"threads":32}': 'dispatch_kernel_0247', '{"computed_smem_bytes":109056,"constants":[["K_MAX_",64]],"ir_name":"knn_search_d512_q32_k64_q64tile_partial_6bea_r119_v1","kwargs":{"K_MAX_":64,"smem_bytes":109056,"validate":false},"threads":256}': 'dispatch_kernel_0248', '{"computed_smem_bytes":0,"constants":[["K_MAX_",64]],"ir_name":"knn_search_d512_q32_k64_q64tile_merge_6bea_r119_v1","kwargs":{"K_MAX_":64,"smem_bytes":1024,"validate":false},"threads":32}': 'dispatch_kernel_0249', '{"computed_smem_bytes":32896,"constants":[["D_",3],["K_MAX_",32],["LOCAL_LIST_CAP_",16],["NUM_WARPS_",8]],"ir_name":"knn_search_d3_dbscan_q4096_k32_direct_9d5c_r117_v1","kwargs":{"smem_bytes":32864,"validate":false},"threads":256}': 'dispatch_kernel_0250', '{"computed_smem_bytes":0,"constants":[["K_STRIDE_",5],["K_OUT_MAX_",3]],"ir_name":"knn_search_lowk_k3_k5partial_a79b_merge_v1","kwargs":{"K_OUT_MAX_":3,"K_STRIDE_":5,"smem_bytes":1024,"validate":false},"threads":32}': 'dispatch_kernel_0251', '{"computed_smem_bytes":165120,"constants":[["K_MAX_",64],["K_STORE_",64]],"ir_name":"knn_search_ext_k64_q4096_m49152_split192_partial_0618_28ec_v2","kwargs":{"K_MAX_":64,"K_STORE_":64,"smem_bytes":165120,"validate":false},"threads":512}': 'dispatch_kernel_0252', '{"computed_smem_bytes":165120,"constants":[["K_MAX_",64],["K_STORE_",7]],"ir_name":"knn_search_ext_k64_q4096_m49152_prefix7_partial_0618_28ec_v2","kwargs":{"K_MAX_":64,"K_STORE_":7,"smem_bytes":165120,"validate":false},"threads":512}': 'dispatch_kernel_0253', '{"computed_smem_bytes":0,"constants":[["K_MAX_",64]],"ir_name":"knn_search_ext_k64_q4096_m49152_merge24_0618_28ec_v2","kwargs":{"K_MAX_":64,"smem_bytes":1024,"validate":false},"threads":32}': 'dispatch_kernel_0254', '{"computed_smem_bytes":0,"constants":[["K_MAX_",64],["K_PREFIX_READ_",6],["K_STRIDE_",7]],"ir_name":"knn_search_ext_k64_q4096_m49152_prefix6cert_merge_0618_28ec_v2","kwargs":{"K_MAX_":64,"K_PREFIX_READ_":6,"K_STRIDE_":7,"smem_bytes":1024,"validate":false},"threads":32}': 'dispatch_kernel_0255', '{"computed_smem_bytes":0,"constants":[],"ir_name":"knn_search_ext_k64_q4096_m49152_certflag_init_0618_28ec_v2","kwargs":{"smem_bytes":1024,"validate":false},"threads":32}': 'dispatch_kernel_0256', '{"computed_smem_bytes":16512,"constants":[["D_",3],["K_MAX_",10],["LOCAL_LIST_CAP_",8],["NUM_WARPS_",8]],"ir_name":"knn_search_dynamic_d3_k10_self_tile_0621_4832_v1","kwargs":{"smem_bytes":16480,"validate":false},"threads":256}': 'dispatch_kernel_0257', '{"computed_smem_bytes":165120,"constants":[["K_MAX_",64],["K_PREFIX_",16]],"ir_name":"knn_search_floor13_k80_prefix16_partial_0622_f3ce_v1","kwargs":{"K_MAX_":64,"K_PREFIX_":16,"smem_bytes":165120,"validate":false},"threads":512}': 'dispatch_kernel_0258', '{"computed_smem_bytes":0,"constants":[["K_MAX_",80],["K_PREFIX_",16]],"ir_name":"knn_search_floor13_k80_prefix16_merge_0622_f3ce_v1","kwargs":{"K_MAX_":80,"K_PREFIX_":16,"smem_bytes":1024,"validate":false},"threads":32}': 'dispatch_kernel_0259', '{"computed_smem_bytes":0,"constants":[["K_OUT_",32],["D_",128],["M_MAIN_",32768]],"ir_name":"knn_search_q4096_m32769_k32_tailinsert_132_67a5_v1","kwargs":{"D_":128,"K_OUT_":32,"M_MAIN_":32768,"smem_bytes":0,"validate":false},"threads":32}': 'dispatch_kernel_0260', '{"computed_smem_bytes":165120,"constants":[["K_MAX_",64],["K_PREFIX_",16]],"ir_name":"knn_search_floor13_k80_prefix16_partial_0622_f3ce_v1","kwargs":{"K_MAX_":64,"K_PREFIX_":4,"smem_bytes":165120,"validate":false},"threads":512}': 'dispatch_kernel_0261', '{"computed_smem_bytes":0,"constants":[["K_MAX_",80],["K_PREFIX_",16]],"ir_name":"knn_search_floor13_k80_prefix16_merge_0622_f3ce_v1","kwargs":{"K_MAX_":80,"K_PREFIX_":4,"smem_bytes":1024,"validate":false},"threads":32}': 'dispatch_kernel_0262', '{"computed_smem_bytes":0,"constants":[["K_MAX_",64]],"ir_name":"knn_search_d1024_q32_k64_targetd_merge_67ec_v1","kwargs":{"K_MAX_":64,"smem_bytes":1024,"validate":false},"threads":32}': 'dispatch_kernel_0263', '{"computed_smem_bytes":10368,"constants":[["D_",3],["K_MAX_",10],["LOCAL_LIST_CAP_",10],["NUM_WARPS_",4]],"ir_name":"knn_search_dynamic_d3_self_q2048_r124_c16f_direct_v1","kwargs":{"smem_bytes":10288,"validate":false},"threads":128}': 'dispatch_kernel_0264', '{"computed_smem_bytes":10368,"constants":[["D_",3],["K_MAX_",10],["LOCAL_LIST_CAP_",10],["ROWS_PER_THREAD_",16],["NUM_WARPS_",4]],"ir_name":"knn_search_dynamic_d3_self_q2048_r123_7d2a_direct_v1","kwargs":{"smem_bytes":10288,"validate":false},"threads":128}': 'dispatch_kernel_0265', '{"computed_smem_bytes":21120,"constants":[["K_MAX_",10],["LOCAL_LIST_CAP_",10],["ROWS_PER_THREAD_",257],["NUM_GROUPS_",8],["GROUP_SIZE_",32]],"ir_name":"knn_search_expanded_d1_tail_warpmerge_0624_025e_v1","kwargs":{"smem_bytes":21120,"validate":false},"threads":256}': 'dispatch_kernel_0266', '{"computed_smem_bytes":0,"constants":[["K_MAX_",64]],"ir_name":"knn_search_q64_m_tail_plus_extra_row_merge_ca90_v1","kwargs":{"K_MAX_":64,"smem_bytes":0,"validate":false},"threads":256}': 'dispatch_kernel_0267', '{"computed_smem_bytes":0,"constants":[["K_MAX_",64]],"ir_name":"knn_search_d256_groupmerge64_fanin8cta_blockm64_891a_v1","kwargs":{"K_MAX_":64,"smem_bytes":0,"validate":false},"threads":256}': 'dispatch_kernel_0268', '{"computed_smem_bytes":0,"constants":[["K_MAX_",64]],"ir_name":"knn_search_d256_groupmerge64_fanin4cta_814e_v1","kwargs":{"K_MAX_":64,"smem_bytes":0,"validate":false},"threads":128}': 'dispatch_kernel_0269', '{"computed_smem_bytes":188160,"constants":[["K_MAX_",64]],"ir_name":"knn_search_q64_tail_last_tile_handoff_partial_812c_v1","kwargs":{"K_MAX_":64,"smem_bytes":188160,"validate":false},"threads":256}': 'dispatch_kernel_0270', '{"computed_smem_bytes":62848,"constants":[["K_MAX_",10]],"ir_name":"knn_search_d4096_q1_m65536_k10_partial_qcache_69ea_v1","kwargs":{"smem_bytes":62848,"validate":false},"threads":512}': 'dispatch_kernel_0271', '{"computed_smem_bytes":54656,"constants":[["K_MAX_",10]],"ir_name":"knn_search_d4096_q1_m65536_k10_partial_compactsmem_69ea_v1","kwargs":{"smem_bytes":54656,"validate":false},"threads":512}': 'dispatch_kernel_0272', '{"computed_smem_bytes":159488,"constants":[["K_MAX_",10]],"ir_name":"knn_search_d4096_q1_m65536_k10_partial_q1stage_v1","kwargs":{"smem_bytes":159488,"validate":false},"threads":512}': 'dispatch_kernel_0273', '{"computed_smem_bytes":0,"constants":[["K_MAX_",10]],"ir_name":"knn_search_target0627_d4096_q4_m32768_k10_merge148_r217_v1","kwargs":{"smem_bytes":1024,"validate":false},"threads":32}': 'dispatch_kernel_0274', '{"computed_smem_bytes":0,"constants":[["K_MAX_",10]],"ir_name":"knn_search_target0627_d4096_q4_m32768_k10_merge256_3737_v1","kwargs":{"smem_bytes":1024,"validate":false},"threads":32}': 'dispatch_kernel_0275', '{"computed_smem_bytes":0,"constants":[["K_MAX_",64]],"ir_name":"knn_search_target0628_d4096_q4_m8192_k64_group16_merge_2ced_v1","kwargs":{"smem_bytes":1024,"validate":false},"threads":32}': 'dispatch_kernel_0277', '{"computed_smem_bytes":0,"constants":[["K_MAX_",64]],"ir_name":"knn_search_target0628_d4096_q4_m8192_k64_group16_final_2ced_v1","kwargs":{"smem_bytes":1024,"validate":false},"threads":32}': 'dispatch_kernel_0278', '{"computed_smem_bytes":0,"constants":[["K_MAX_",64]],"ir_name":"knn_search_target_highd_k64_group_merge_f505_hiermerge32_v1","kwargs":{"smem_bytes":1024,"validate":false},"threads":32}': 'dispatch_kernel_0279', '{"computed_smem_bytes":0,"constants":[["K_MAX_",64]],"ir_name":"knn_search_target_highd_k64_final_merge_f505_hiermerge32_v1","kwargs":{"smem_bytes":1024,"validate":false},"threads":32}': 'dispatch_kernel_0280', '{"computed_smem_bytes":166400,"constants":[["D_ORIG_",4096],["NUM_D_PASSES_",16],["K_MAX_",64]],"ir_name":"knn_search_target0628_d4096_q4_m8192_k64_partial_7738_v1","kwargs":{"smem_bytes":166400,"validate":false},"threads":256}': 'dispatch_kernel_0281', '{"computed_smem_bytes":0,"constants":[["K_MAX_",64]],"ir_name":"knn_search_target0628_d1024_q32_m65536_k64_merge_9571_v1","kwargs":{"K_MAX_":64,"smem_bytes":1024,"validate":false},"threads":32}': 'dispatch_kernel_0282', '{"computed_smem_bytes":0,"constants":[["K_MAX_",64]],"ir_name":"knn_search_target0628_d64_q512_m65536_k64_groupmerge_e8f1_v1","kwargs":{"K_MAX_":64,"smem_bytes":1024,"validate":false},"threads":32}': 'dispatch_kernel_0283', '{"computed_smem_bytes":0,"constants":[["K_MAX_",64]],"ir_name":"knn_search_target0628_d64_q512_m65536_k64_finalmerge_e8f1_v1","kwargs":{"K_MAX_":64,"smem_bytes":1024,"validate":false},"threads":32}': 'dispatch_kernel_0284', '{"computed_smem_bytes":143104,"constants":[["K_MAX_",10],["D_ORIG_",1024],["NUM_D_PASSES_",8],["Q_NORM_PARTS_",64]],"ir_name":"knn_search_d1024_q8_tcgen05_partial_16warp_b3fc_v1","kwargs":{"smem_bytes":143104,"validate":false},"threads":512}': 'dispatch_kernel_0287', '{"computed_smem_bytes":0,"constants":[["K_MAX_",10]],"ir_name":"knn_search_target0630_d4096_q4_m32768_k10_merge_independent_warp_q4tail237_v1","kwargs":{"smem_bytes":1024,"validate":false},"threads":32}': 'dispatch_kernel_0288', '{"computed_smem_bytes":0,"constants":[["K_MAX_",10]],"ir_name":"knn_search_target0630_d4096_q4_m32768_k10_merge_q4warp_d759_v2","kwargs":{"smem_bytes":1024,"validate":false},"threads":128}': 'dispatch_kernel_0289', '{"computed_smem_bytes":0,"constants":[["K_MAX_",64]],"ir_name":"knn_search_q65_rows8_bounded_finalmerge_21e6_v1","kwargs":{"K_MAX_":64,"smem_bytes":1024,"validate":false},"threads":256}': 'dispatch_kernel_0290', '{"computed_smem_bytes":0,"constants":[["SOURCE_K_",64],["OUTPUT_K_",11]],"ir_name":"knn_search_k64_prefix_to_k11_copy_0705_v1","kwargs":{"OUTPUT_K_":11,"SOURCE_K_":64,"validate":false},"threads":256}': 'dispatch_kernel_0291', '{"computed_smem_bytes":0,"constants":[["K_MAX_",64],["K_PREFIX_READ_",6],["K_STRIDE_",7]],"ir_name":"knn_search_k64_q4096m20000_prefixcert_fused_merge_0615_576b_v1","kwargs":{"smem_bytes":1024,"validate":false},"threads":32}': 'dispatch_kernel_0292', '{"computed_smem_bytes":4096,"constants":[["D_",128],["K_CAP_",64],["NUM_WARPS_",8]],"ir_name":"knn_search_k64_q4096m20000_prefixcert_gpu_repair_0705_v1","kwargs":{"D_":128,"K_CAP_":64,"NUM_WARPS_":8,"smem_bytes":4096,"validate":false},"threads":256}': 'dispatch_kernel_0293'}

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
