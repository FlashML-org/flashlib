"""v11 common-D synthesized seed portfolio dispatcher.

Minimum target architecture: sm_100a for the consumed tcgen05/TMA seeds. The
fallback route remains the existing Weave-only common-D dispatcher and may run
generic coverage fallback code on sm_80, but this portfolio is intended for
Blackwell v11 common-D validation.

This wrapper only adds guards around validated seed entrypoints. It does not
retune seed schedules. Production dispatch stays Weave-only.
"""
from __future__ import annotations
from json import loads as _json_loads
from .._dispatch_runtime import _capture_cuTensorMapEncodeTiled, _decode_capture, _import_dispatch_module, _ir_proxy
import argparse
import json
from pathlib import Path
from typing import Any, Callable
from . import knn_build_common_d256_q1024_56f3_v1 as d256_build_a4ec
from . import knn_build_common_d_56f3_build_d256_q1024_v1 as d256_build
from . import knn_build_common_d_eeff_search_d768_v1 as d768_search
from . import knn_build_common_d_5e7f_rag_highd_v1 as highd_rag
from . import knn_build_common_d_5e7f_rag_d64_d256_v1 as rag_d64d256
from . import knn_build_common_d_5e7f_rag_d64_repair_v1 as rag_d64_repair
from . import knn_build_common_d_1438_rag_d64_m128_v1 as rag_d64_m128
from . import knn_build_common_d_5e7f_search_d256_v1 as search_d256
from . import knn_build_common_d768_build_eeff_m64split_v1 as d768_build_fast
from . import knn_build_common_d_56f3_build_highd_v1 as highd_build
from . import knn_build_build_lowfloor_2c1c_v3 as seed_k13
from . import knn_build_d64_q4096_c271_twostage_v1 as d64_q4096_c271
from . import knn_build_dispatch_common_d_v11_fallback_v1 as base_current
from . import knn_build_ragonline_mbucket_ea43_q1m524_n128_v1 as q1_m524_n128
from . import knn_build_d128_rag_q128_k10_df0f_warpmerge_v1 as q128_k10_warpmerge
from . import knn_build_k48_k96_floor_repair_d03c_v2 as seed_k48
from . import knn_build_large_square_k32_efe4_prodcache_v1 as large_square_k32_efe4
from . import knn_build_non128_frontier_4be7_d768fused_v1 as d768_rag
from . import knn_build_rag_microbucket_k32_q32rowld2exact_f653_v1 as q32exact
from . import knn_build_v12_d256_k10_longm_e2df_v1 as d256_q4_e2df
from . import knn_build_v12_d256_k32_tail_59fe_v1 as d256_k32_59fe
from . import knn_build_v12_d256_q128_k10_longm_59fe_v1 as d256_q128_59fe
from . import knn_build_v12_d64_tail_017a_v1 as d64_tail_017a
from . import knn_build_v12_d128_q16_k48_dd2b_v1 as d128_k48_dd2b
from . import knn_build_v12_highd_rag_22e9_v1 as highd_rag_22e9
from . import knn_build_v12_highd_search_be66_v1 as highd_search_be66
from . import knn_build_rect_d128_k20_q1536_s12warp4_7768_v1 as rect_d128_k20_s12warp4
MODULE = 'loom.examples.weave.knn_build_dispatch_v11_common_d_seed_portfolio_a4ec_v1'
CANDIDATE_A4EC_BASE = 'v11_common_d_seed_portfolio_a4ec_mixed_v1'
CANDIDATE_FA04_BASE = 'v11_common_d_seed_portfolio_fa04_cda9_6164_v1'
CANDIDATE_F328_BASE = 'v11_common_d_seed_portfolio_4cf7_highd_rag_v1'
CANDIDATE_MIXED = 'v11_common_d_seed_portfolio_8dbc_ba22_d256_v1'
CANDIDATE_D64_REPAIR = 'v11_common_d_seed_portfolio_0474_d64_rag_repair_v1'
CANDIDATE_D64_M128 = 'v11_common_d_seed_portfolio_1438_d64_rag_m128_backfill_v1'
CANDIDATE_D64_Q4096_C271 = 'v11_common_d_seed_portfolio_c271_d64_q4096_v1'
CANDIDATE_C271_7DC5_K13_K48 = 'v11_common_d_seed_portfolio_c271_7dc5_k13_k48_v1'
CANDIDATE_FLOOR_SEEDS_Q128_5698 = 'v11_common_d_seed_portfolio_k13_q128dualm_q16_v1'
CANDIDATE_FLOOR_SEEDS_Q128_681B = 'v11_common_d_seed_portfolio_k13_q128s72r2_q16_v1'
CANDIDATE_FLOOR_SEEDS_Q128_MIXED = 'v11_common_d_seed_portfolio_k13_q128mixed_q16_v1'
CANDIDATE_PRE_Q128_K10_WARPMERGE = 'v11_common_d_seed_portfolio_c271_7dc5_k48_q128mixed_q16_v1'
CANDIDATE_2498_BASELINE = CANDIDATE_PRE_Q128_K10_WARPMERGE
CANDIDATE_Q128_K10_WARPMERGE = 'v11_common_d_seed_portfolio_c271_7dc5_k48_q128mixed_q16_q128k10wm_v1'
CANDIDATE_PRE_Q1_M524_N128 = 'v11_common_d_seed_portfolio_c271_7dc5_k48_q128mixed_q16_efe4_34da_v1'
CANDIDATE_PRE_Q32_EXACT = 'v11_common_d_seed_portfolio_c271_7dc5_k48_q128mixed_q16_efe4_34da_q1m524n128_v1'
CANDIDATE_PRE_Q32TAIL = 'v11_common_d_seed_portfolio_c271_7dc5_k48_q128mixed_q16_efe4_34da_q1m524n128_q32exact_v1'
CANDIDATE_PRE_EXPANDED_K32_Q48 = 'v11_common_d_seed_portfolio_c271_7dc5_k48_q128mixed_q16_efe4_34da_q1m524n128_q32exact_q32tail_v1'
CANDIDATE_PRE_LOWK_C3D2 = 'v11_common_d_seed_portfolio_c271_7dc5_k48_q128mixed_q16_efe4_34da_q1m524n128_q32exact_q32tail_q31_q33_q40_q48_v1'
CANDIDATE_D665_LOWK_BASELINE = 'v11_common_d_seed_portfolio_c271_7dc5_k48_q128mixed_q16_efe4_34da_q1m524n128_q32exact_q32tail_q31_q33_q40_q48_lowk_c3d2_v1'
CANDIDATE_PRE_D64_TAIL_017A = 'v11_common_d_seed_portfolio_c271_7dc5_k48_q128mixed_q16_efe4_34da_q1m524n128_q32exact_q32tail_q31_q33_q40_q48_lowk_c3d2_k31_eaf7_v1'
CANDIDATE_PRE_D256_Q4_E2DF = 'v11_common_d_seed_portfolio_c271_7dc5_k48_q128mixed_q16_efe4_34da_q1m524n128_q32exact_q32tail_q31_q33_q40_q48_lowk_c3d2_k31_eaf7_017a_d64tail_v1'
CANDIDATE_SELECTED_SYNTHESIS = 'v11_common_d_seed_portfolio_c271_7dc5_k48_q128mixed_q16_efe4_34da_q1m524n128_q32exact_q32tail_q31_q33_q40_q48_lowk_c3d2_k31_eaf7_017a_d64tail_e2df_d256q4_v1'
CANDIDATE_D256_Q128_59FE = 'v11_common_d_seed_portfolio_c271_7dc5_k48_q128mixed_q16_efe4_34da_q1m524n128_q32exact_q32tail_q31_q33_q40_q48_lowk_c3d2_k31_eaf7_017a_d64tail_e2df_d256q4_59fe_d256q128_v1'
CANDIDATE_D256_K32_59FE = 'v11_common_d_seed_portfolio_c271_7dc5_k48_q128mixed_q16_efe4_34da_q1m524n128_q32exact_q32tail_q31_q33_q40_q48_lowk_c3d2_k31_eaf7_017a_d64tail_e2df_d256q4_59fe_d256q128_9203_d256k32_v1'
CANDIDATE_HIGHD_RAG_22E9 = 'v11_common_d_seed_portfolio_c271_7dc5_k48_q128mixed_q16_efe4_34da_q1m524n128_q32exact_q32tail_q31_q33_q40_q48_lowk_c3d2_k31_eaf7_017a_d64tail_e2df_d256q4_59fe_d256q128_9203_d256k32_7902_highd_rag_v1'
CANDIDATE_HIGHD_SEARCH_BE66 = 'v11_common_d_seed_portfolio_c271_7dc5_k48_q128mixed_q16_efe4_34da_q1m524n128_q32exact_q32tail_q31_q33_q40_q48_lowk_c3d2_k31_eaf7_017a_d64tail_e2df_d256q4_59fe_d256q128_9203_d256k32_7902_highd_rag_ad73_highd_search_v1'
CANDIDATE_D128_K48_DD2B = 'v11_common_d_seed_portfolio_c271_7dc5_k48_q128mixed_q16_efe4_34da_q1m524n128_q32exact_q32tail_q31_q33_q40_q48_lowk_c3d2_k31_eaf7_017a_d64tail_e2df_d256q4_59fe_d256q128_9203_d256k32_7902_highd_rag_ad73_highd_search_dd2b_d128q16k48_v1'
CANDIDATE_RECT_D128_K20_S12WARP4 = 'v11_common_d_seed_portfolio_c271_7dc5_k48_q128mixed_q16_efe4_34da_q1m524n128_q32exact_q32tail_q31_q33_q40_q48_lowk_c3d2_k31_eaf7_017a_d64tail_e2df_d256q4_59fe_d256q128_9203_d256k32_7902_highd_rag_ad73_highd_search_dd2b_d128q16k48_7768_rectd128k20q1536_s12warp4_v1'
CANDIDATE_Q128_K10_ROWLD_1BED = 'v11_common_d_seed_portfolio_c271_7dc5_k48_q128mixed_q16_efe4_34da_q1m524n128_q32exact_q32tail_q31_q33_q40_q48_lowk_c3d2_k31_eaf7_017a_d64tail_e2df_d256q4_59fe_d256q128_9203_d256k32_7902_highd_rag_ad73_highd_search_dd2b_d128q16k48_7768_rectd128k20q1536_s12warp4_1bed_q128k10_rowld_v1'
CANDIDATE_DEFAULT = CANDIDATE_Q128_K10_ROWLD_1BED
CANDIDATE_F1D9_BUILD = 'v11_common_d_seed_portfolio_a4ec_f1d9_build_v1'
CANDIDATE_BASE = 'base_common_d_v11_fallback'
SPEEDUP_FLOOR = 1.2
FLOOR_SEED_PORTFOLIOS = (CANDIDATE_FLOOR_SEEDS_Q128_5698, CANDIDATE_FLOOR_SEEDS_Q128_681B, CANDIDATE_FLOOR_SEEDS_Q128_MIXED, CANDIDATE_2498_BASELINE, CANDIDATE_Q128_K10_WARPMERGE, CANDIDATE_PRE_Q1_M524_N128, CANDIDATE_PRE_Q32_EXACT, CANDIDATE_PRE_Q32TAIL, CANDIDATE_PRE_EXPANDED_K32_Q48, CANDIDATE_PRE_LOWK_C3D2, CANDIDATE_D665_LOWK_BASELINE, CANDIDATE_PRE_D64_TAIL_017A, CANDIDATE_PRE_D256_Q4_E2DF, CANDIDATE_SELECTED_SYNTHESIS)
PORTFOLIOS_WITH_K13_K48 = (CANDIDATE_C271_7DC5_K13_K48, *FLOOR_SEED_PORTFOLIOS)
PORTFOLIOS_WITH_D768_FAST = (CANDIDATE_F328_BASE, CANDIDATE_MIXED, CANDIDATE_D64_REPAIR, CANDIDATE_D64_M128, CANDIDATE_D64_Q4096_C271, *PORTFOLIOS_WITH_K13_K48)
PORTFOLIOS_WITH_HIGHD_RAG = PORTFOLIOS_WITH_D768_FAST
PORTFOLIOS_WITH_SEARCH_D256 = (CANDIDATE_MIXED, CANDIDATE_D64_REPAIR, CANDIDATE_D64_M128, CANDIDATE_D64_Q4096_C271, *PORTFOLIOS_WITH_K13_K48)
PORTFOLIOS_WITH_RAG_D64_M128 = (CANDIDATE_D64_M128, CANDIDATE_D64_Q4096_C271, *PORTFOLIOS_WITH_K13_K48)
PORTFOLIOS_WITH_D64_Q4096_C271 = (CANDIDATE_D64_Q4096_C271, *PORTFOLIOS_WITH_K13_K48)
PORTFOLIOS_WITH_LARGE_SQUARE_K32_EFE4 = (CANDIDATE_2498_BASELINE, CANDIDATE_Q128_K10_WARPMERGE, CANDIDATE_PRE_Q1_M524_N128, CANDIDATE_PRE_Q32_EXACT, CANDIDATE_PRE_Q32TAIL, CANDIDATE_PRE_EXPANDED_K32_Q48, CANDIDATE_PRE_LOWK_C3D2, CANDIDATE_D665_LOWK_BASELINE, CANDIDATE_PRE_D64_TAIL_017A, CANDIDATE_PRE_D256_Q4_E2DF, CANDIDATE_SELECTED_SYNTHESIS)
PORTFOLIOS_WITH_Q128_K10_WARPMERGE = (CANDIDATE_Q128_K10_WARPMERGE,)
PORTFOLIOS_WITH_RAG_STREAM_K10_34DA = (CANDIDATE_PRE_Q1_M524_N128, CANDIDATE_PRE_Q32_EXACT, CANDIDATE_PRE_Q32TAIL, CANDIDATE_PRE_EXPANDED_K32_Q48, CANDIDATE_PRE_LOWK_C3D2, CANDIDATE_D665_LOWK_BASELINE, CANDIDATE_PRE_D64_TAIL_017A, CANDIDATE_PRE_D256_Q4_E2DF, CANDIDATE_SELECTED_SYNTHESIS)
PORTFOLIOS_WITH_Q1_M524_N128 = (CANDIDATE_PRE_Q32_EXACT, CANDIDATE_PRE_Q32TAIL, CANDIDATE_PRE_EXPANDED_K32_Q48, CANDIDATE_PRE_LOWK_C3D2, CANDIDATE_D665_LOWK_BASELINE, CANDIDATE_PRE_D64_TAIL_017A, CANDIDATE_PRE_D256_Q4_E2DF, CANDIDATE_SELECTED_SYNTHESIS)
PORTFOLIOS_WITH_Q32_EXACT = (CANDIDATE_PRE_Q32TAIL, CANDIDATE_PRE_EXPANDED_K32_Q48, CANDIDATE_PRE_D64_TAIL_017A, CANDIDATE_PRE_D256_Q4_E2DF, CANDIDATE_SELECTED_SYNTHESIS)
PORTFOLIOS_WITH_Q32_TAIL = (CANDIDATE_PRE_EXPANDED_K32_Q48, CANDIDATE_PRE_LOWK_C3D2, CANDIDATE_D665_LOWK_BASELINE, CANDIDATE_PRE_D64_TAIL_017A, CANDIDATE_PRE_D256_Q4_E2DF, CANDIDATE_SELECTED_SYNTHESIS)
PORTFOLIOS_WITH_EXPANDED_K32_Q31_Q33_Q40 = (CANDIDATE_PRE_LOWK_C3D2, CANDIDATE_D665_LOWK_BASELINE, CANDIDATE_PRE_D64_TAIL_017A, CANDIDATE_PRE_D256_Q4_E2DF, CANDIDATE_SELECTED_SYNTHESIS)
PORTFOLIOS_WITH_Q48_K12 = (CANDIDATE_PRE_LOWK_C3D2, CANDIDATE_D665_LOWK_BASELINE, CANDIDATE_PRE_D64_TAIL_017A, CANDIDATE_PRE_D256_Q4_E2DF, CANDIDATE_SELECTED_SYNTHESIS)
PORTFOLIOS_WITH_Q32_LOWK_C3D2 = (CANDIDATE_D665_LOWK_BASELINE, CANDIDATE_PRE_D64_TAIL_017A, CANDIDATE_PRE_D256_Q4_E2DF, CANDIDATE_SELECTED_SYNTHESIS)
PORTFOLIOS_WITH_Q32_K31_C3D2 = (CANDIDATE_PRE_D64_TAIL_017A, CANDIDATE_PRE_D256_Q4_E2DF, CANDIDATE_SELECTED_SYNTHESIS)
PORTFOLIOS_WITH_D64_TAIL_017A = (CANDIDATE_PRE_D256_Q4_E2DF, CANDIDATE_SELECTED_SYNTHESIS)
PORTFOLIOS_WITH_D256_Q4_E2DF = (CANDIDATE_SELECTED_SYNTHESIS, CANDIDATE_D256_Q128_59FE, CANDIDATE_D256_K32_59FE)
PORTFOLIOS_WITH_D256_Q128_59FE = (CANDIDATE_D256_Q128_59FE, CANDIDATE_D256_K32_59FE)
PORTFOLIOS_WITH_D256_K32_59FE = (CANDIDATE_D256_K32_59FE, CANDIDATE_HIGHD_RAG_22E9, CANDIDATE_HIGHD_SEARCH_BE66)
PORTFOLIOS_WITH_HIGHD_RAG_22E9 = (CANDIDATE_HIGHD_RAG_22E9, CANDIDATE_HIGHD_SEARCH_BE66)
PORTFOLIOS_WITH_HIGHD_SEARCH_BE66 = (CANDIDATE_HIGHD_SEARCH_BE66, CANDIDATE_D128_K48_DD2B)
PORTFOLIOS_WITH_D128_K48_DD2B = (CANDIDATE_D128_K48_DD2B,)
PORTFOLIOS_WITH_RECT_D128_K20_S12WARP4 = (CANDIDATE_RECT_D128_K20_S12WARP4, CANDIDATE_Q128_K10_ROWLD_1BED)
PORTFOLIOS_WITH_Q128_K10_ROWLD_1BED = (CANDIDATE_Q128_K10_ROWLD_1BED,)
PORTFOLIOS_WITH_Q32_TAIL143 = (CANDIDATE_D128_K48_DD2B, CANDIDATE_RECT_D128_K20_S12WARP4, CANDIDATE_Q128_K10_ROWLD_1BED)
ROUTE_ENTRYPOINT = ''.join([format(MODULE, ''), ':launch_from_contract_inputs'])
BENCHMARK_ENTRYPOINT = ''.join([format(MODULE, ''), ':benchmark_knn_build_dispatch_v11_common_d_seed_portfolio_a4ec_v1'])
BASE_ENTRYPOINT = base_current.ROUTE_ENTRYPOINT
A4EC_BASE_ENTRYPOINT = ''.join([format(MODULE, ''), ':launch_from_contract_inputs_a4ec_baseline'])
FA04_BASE_ENTRYPOINT = ''.join([format(MODULE, ''), ':launch_from_contract_inputs_fa04_baseline'])
F328_BASE_ENTRYPOINT = ''.join([format(MODULE, ''), ':launch_from_contract_inputs_f328_baseline'])
MIXED_BASE_ENTRYPOINT = ''.join([format(MODULE, ''), ':launch_from_contract_inputs_mixed_baseline'])
D64_REPAIR_BASE_ENTRYPOINT = ''.join([format(MODULE, ''), ':launch_from_contract_inputs_d64_repair_baseline'])
D64_M128_BASE_ENTRYPOINT = ''.join([format(MODULE, ''), ':launch_from_contract_inputs_d64_m128_baseline'])
D64_Q4096_C271_BASE_ENTRYPOINT = ''.join([format(MODULE, ''), ':launch_from_contract_inputs_c271_baseline'])
C271_7DC5_BASE_ENTRYPOINT = ''.join([format(MODULE, ''), ':launch_from_contract_inputs_c271_7dc5_baseline'])
PRE_Q128_K10_WARPMERGE_BASE_ENTRYPOINT = ''.join([format(MODULE, ''), ':launch_from_contract_inputs_pre_q128_k10_warpmerge_baseline'])
BASE_2498_ENTRYPOINT = ''.join([format(MODULE, ''), ':launch_from_contract_inputs_2498_baseline'])
PRE_Q1_M524_N128_BASE_ENTRYPOINT = ''.join([format(MODULE, ''), ':launch_from_contract_inputs_pre_q1_m524_n128_baseline'])
PRE_Q32_EXACT_BASE_ENTRYPOINT = ''.join([format(MODULE, ''), ':launch_from_contract_inputs_pre_q32exact_baseline'])
PRE_Q32TAIL_BASE_ENTRYPOINT = ''.join([format(MODULE, ''), ':launch_from_contract_inputs_pre_q32tail_baseline'])
PRE_EXPANDED_K32_Q48_BASE_ENTRYPOINT = ''.join([format(MODULE, ''), ':launch_from_contract_inputs_pre_expanded_k32_q48_baseline'])
PRE_LOWK_C3D2_BASE_ENTRYPOINT = ''.join([format(MODULE, ''), ':launch_from_contract_inputs_pre_lowk_c3d2_baseline'])
D665_LOWK_BASE_ENTRYPOINT = ''.join([format(MODULE, ''), ':launch_from_contract_inputs_d665_lowk_baseline'])
PRE_D64_TAIL_017A_BASE_ENTRYPOINT = ''.join([format(MODULE, ''), ':launch_from_contract_inputs_pre_d64_tail_017a_baseline'])
PRE_D256_Q4_E2DF_BASE_ENTRYPOINT = ''.join([format(MODULE, ''), ':launch_from_contract_inputs_pre_d256_q4_e2df_baseline'])
PRE_D256_Q128_59FE_BASE_ENTRYPOINT = ''.join([format(MODULE, ''), ':launch_from_contract_inputs_pre_d256_q128_59fe_baseline'])
PRE_D256_K32_59FE_BASE_ENTRYPOINT = ''.join([format(MODULE, ''), ':launch_from_contract_inputs_pre_d256_k32_59fe_baseline'])
PRE_HIGHD_RAG_22E9_BASE_ENTRYPOINT = ''.join([format(MODULE, ''), ':launch_from_contract_inputs_pre_highd_rag_22e9_baseline'])
PRE_HIGHD_SEARCH_BE66_BASE_ENTRYPOINT = ''.join([format(MODULE, ''), ':launch_from_contract_inputs_pre_highd_search_be66_baseline'])
PRE_D128_K48_DD2B_BASE_ENTRYPOINT = ''.join([format(MODULE, ''), ':launch_from_contract_inputs_pre_d128_k48_dd2b_baseline'])
PRE_RECT_D128_K20_S12WARP4_BASE_ENTRYPOINT = ''.join([format(MODULE, ''), ':launch_from_contract_inputs_pre_rect_d128_k20_s12warp4_baseline'])
PRE_Q128_K10_ROWLD_1BED_BASE_ENTRYPOINT = ''.join([format(MODULE, ''), ':launch_from_contract_inputs_pre_q128_k10_rowld_1bed_baseline'])
D256_A4EC_ENTRYPOINT = 'loom.examples.weave.knn_build_common_d256_q1024_56f3_v1:launch_from_contract_inputs'
D256_ENTRYPOINT = 'loom.examples.weave.knn_build_common_d_56f3_build_d256_q1024_v1:launch_from_contract_inputs'
D768_SEARCH_ENTRYPOINT = 'loom.examples.weave.knn_build_common_d_eeff_search_d768_v1:launch_from_contract_inputs'
SEARCH_D256_ENTRYPOINT = 'loom.examples.weave.knn_build_common_d_5e7f_search_d256_v1:launch_from_contract_inputs'
D768_FAST_ENTRYPOINT = 'loom.examples.weave.knn_build_common_d768_build_eeff_m64split_v1:launch_from_contract_inputs'
HIGHD_ENTRYPOINT = 'loom.examples.weave.knn_build_common_d_56f3_build_highd_v1:launch_from_contract_inputs'
D768_RAG_ENTRYPOINT = 'loom.examples.weave.knn_build_non128_frontier_4be7_d768fused_v1:launch_from_contract_inputs'
HIGHD_RAG_ENTRYPOINT = 'loom.examples.weave.knn_build_common_d_5e7f_rag_highd_v1:launch_from_contract_inputs'
RAG_D64D256_ENTRYPOINT = 'loom.examples.weave.knn_build_common_d_5e7f_rag_d64_d256_v1:launch_from_contract_inputs'
RAG_D64_REPAIR_ENTRYPOINT = 'loom.examples.weave.knn_build_common_d_5e7f_rag_d64_repair_v1:launch_from_contract_inputs'
RAG_D64_M128_ENTRYPOINT = 'loom.examples.weave.knn_build_common_d_1438_rag_d64_m128_v1:launch_from_contract_inputs'
D64_Q4096_C271_ENTRYPOINT = 'loom.examples.weave.knn_build_d64_q4096_c271_twostage_v1:launch_from_contract_inputs'
LARGE_SQUARE_K32_EFE4_ENTRYPOINT = 'loom.examples.weave.knn_build_large_square_k32_efe4_prodcache_v1:launch_from_contract_inputs'
K13_K48_WRAPPER_MODULE = 'loom.examples.weave.knn_build_k13_k48_floor_repair_7dc5_v1'
K13_K48_WRAPPER_ENTRYPOINT = ''.join([format(K13_K48_WRAPPER_MODULE, ''), ':launch_from_contract_inputs'])
K13_ENTRYPOINT = seed_k13.ROUTE_Q4096_K13_UNORDERED
K48_ENTRYPOINT = seed_k48.ROUTE_K48_WARPSELECT
Q128_DUALM_ENTRYPOINT = 'loom.examples.weave.knn_build_rag_stream_k32_q128_dualm_a162_v1:launch_from_contract_inputs'
Q128_S72R2_ENTRYPOINT = 'loom.examples.weave.knn_build_rag_stream_k32_q128_s72r2_a162_v1:launch_from_contract_inputs'
Q128_K10_WARPMERGE_ENTRYPOINT = q128_k10_warpmerge.ROUTE_ENTRYPOINT
Q128_K10_ROWLD_1BED_MODULE = 'loom.examples.weave.knn_build_rag_stream_k10_q128_1bed_rowld_v1'
Q128_K10_ROWLD_1BED_ENTRYPOINT = ''.join([format(Q128_K10_ROWLD_1BED_MODULE, ''), ':launch_from_contract_inputs'])
Q16_M250_ENTRYPOINT = 'loom.examples.weave.knn_build_d128_rag_q16m250_df0f_v1:launch_from_contract_inputs'
RAG_STREAM_K10_34DA_ENTRYPOINT = 'loom.examples.weave.knn_build_rag_stream_k10_warpmerge_34da_v1:launch_from_contract_inputs'
Q1_M524_N128_ENTRYPOINT = 'loom.examples.weave.knn_build_ragonline_mbucket_ea43_q1m524_n128_v1:launch_from_contract_inputs'
Q32_EXACT_ENTRYPOINT = q32exact.ROUTE_Q32_ROWLD2EXACT_ENTRYPOINT
Q32TAIL_MODULE = 'loom.examples.weave.knn_build_rag_microbucket_k32_0cb5_q31tail_v1'
Q32TAIL_ENTRYPOINT = ''.join([format(Q32TAIL_MODULE, ''), ':launch_from_contract_inputs'])
Q32TAIL143_LOW_MODULE = 'loom.examples.weave.knn_build_rag_microbucket_k32_5317_q32tail143low_v1'
Q32TAIL143_LOW_ENTRYPOINT = ''.join([format(Q32TAIL143_LOW_MODULE, ''), ':launch_from_contract_inputs'])
Q32TAIL143_HIGH_MODULE = 'loom.examples.weave.knn_build_rag_microbucket_k32_314c_q32tail143_v1'
Q32TAIL143_HIGH_ENTRYPOINT = ''.join([format(Q32TAIL143_HIGH_MODULE, ''), ':launch_from_contract_inputs'])
Q31TAIL_V2_MODULE = 'loom.examples.weave.knn_build_rag_microbucket_k32_0cb5_q31tail_v2'
Q31TAIL_V2_ENTRYPOINT = ''.join([format(Q31TAIL_V2_MODULE, ''), ':launch_from_contract_inputs'])
Q33TILE_MODULE = 'loom.examples.weave.knn_build_rag_microbucket_k32_c489_q33tile_v1'
Q33TILE_ENTRYPOINT = ''.join([format(Q33TILE_MODULE, ''), ':launch_from_contract_inputs'])
Q48_K12_MODULE = 'loom.examples.weave.knn_build_rag_microbucket_k12_2f22_q48exact_v1'
Q48_K12_ENTRYPOINT = ''.join([format(Q48_K12_MODULE, ''), ':launch_from_contract_inputs'])
Q32_LOWK_C3D2_MODULE = 'loom.examples.weave.knn_build_rag_microbucket_q32_lowk_c3d2_v1'
Q32_LOWK_C3D2_ENTRYPOINT = ''.join([format(Q32_LOWK_C3D2_MODULE, ''), ':launch_from_contract_inputs'])
Q32_K31_C3D2_MODULE = 'loom.examples.weave.knn_build_rag_microbucket_q32_k31_c3d2_v1'
Q32_K31_C3D2_ENTRYPOINT = ''.join([format(Q32_K31_C3D2_MODULE, ''), ':launch_from_contract_inputs'])
D64_TAIL_017A_ENTRYPOINT = d64_tail_017a.ROUTE_ENTRYPOINT
D256_Q4_E2DF_ENTRYPOINT = d256_q4_e2df.ROUTE_ENTRYPOINT
D256_Q128_59FE_ENTRYPOINT = d256_q128_59fe.ROUTE_ENTRYPOINT
D256_K32_59FE_ENTRYPOINT = d256_k32_59fe.ROUTE_ENTRYPOINT
HIGHD_RAG_22E9_ENTRYPOINT = highd_rag_22e9.ROUTE_ENTRYPOINT
HIGHD_SEARCH_BE66_ENTRYPOINT = highd_search_be66.ROUTE_ENTRYPOINT
D128_K48_DD2B_ENTRYPOINT = d128_k48_dd2b.ROUTE_ENTRYPOINT
RECT_D128_K20_S12WARP4_ENTRYPOINT = rect_d128_k20_s12warp4.ROUTE_ENTRYPOINT
SEED_K13_ID = '7dc5_2c1c_q4096_k13_unordered_s4'
SEED_K48_ID = '7dc5_d03c_k48_s4_warpselect'
SEED_K13_A162_ID = 'a162_2c1c_q4096_k13_unordered_s4'
SEED_Q128_DUALM_ID = 'rag_stream_k32_q128_dualm_a162_v1_rowld_s72_warp1'
SEED_Q128_S72R2_ID = 'a162_rag_stream_q128_k32_s72r2_v1'
SEED_Q128_K10_WARPMERGE_ID = q128_k10_warpmerge.SEED_ID
SEED_Q128_K10_ROWLD_1BED_ID = 'rag_stream_k10_q128_rowld_1bed_v1_s74'
SEED_Q16_M250_ID = 'df0f_bdd2_q16_m250_k32_s288'
SEED_LARGE_SQUARE_K32_EFE4_ID = 'large_square_k32_efe4_prodcache_v1'
SEED_RAG_STREAM_K10_34DA_ID = 'rag_stream_k10_warpmerge_34da_v1_s72_r4'
SEED_Q1_M524_N128_ID = 'ea43_q1m524_n128_s148_g4_m64n128'
SEED_Q32_EXACT_ID = q32exact.SEED_K32_Q32_ROWLD2EXACT_F653_V1_ID
SEED_Q32TAIL_ID = 'rag_microbucket_k32_0cb5_q31tail_v1'
SEED_Q32TAIL143_LOW_ID = 'rag_microbucket_k32_5317_q32tail143low_v1'
SEED_Q32TAIL143_HIGH_ID = 'rag_microbucket_k32_314c_q32tail143_v1'
SEED_Q31TAIL_V2_ID = 'rag_microbucket_k32_0cb5_q31tail_v2'
SEED_Q33TILE_ID = 'rag_microbucket_k32_c489_q33tile_v1'
SEED_Q48_K12_ID = 'rag_microbucket_k12_2f22_q48exact_v1'
SEED_Q32_LOWK_C3D2_ID = 'rag_microbucket_q32_lowk_c3d2_v1'
SEED_Q32_K31_C3D2_ID = 'rag_microbucket_q32_k31_c3d2_v1'
SEED_D64_TAIL_017A_ID = d64_tail_017a.CANDIDATE_ID
SEED_D256_Q4_E2DF_ID = d256_q4_e2df.CANDIDATE_ID
SEED_D256_Q128_59FE_ID = d256_q128_59fe.CANDIDATE_ID
SEED_D256_K32_59FE_ID = d256_k32_59fe.CANDIDATE_ID
SEED_HIGHD_RAG_22E9_ID = highd_rag_22e9.CANDIDATE_ID
SEED_HIGHD_SEARCH_BE66_ID = highd_search_be66.CANDIDATE_ID
SEED_D128_K48_DD2B_ID = d128_k48_dd2b.CANDIDATE_ID
SEED_RECT_D128_K20_S12WARP4_ID = rect_d128_k20_s12warp4.SEED_ID
Q32_EXACT_GUARD_ID = '12ac_f653_q32rowld2exact_stage1_exact_guard'
Q32_EXACT_GUARD_CONDITION = 'exact BF16 non-build B=1 Q=32 M=100000 D=128 K=32'
Q32_EXACT_PRODUCER_TOPOLOGY = 'f653_rowld2_ROW_16x256B_two_compute_warp_exact_stage1_s141'
Q32TAIL143_LOW_GUARD_ID = 'd6b5_q32_m99999_split143_exact_guard'
Q32TAIL143_HIGH_GUARD_ID = 'd6b5_q32_m100001_split143_exact_guard'
Q31TAIL_V2_GUARD_ID = 'c3d2_0cb5_q31tail_v2_exact_guard'
Q33TILE_GUARD_ID = 'c3d2_c489_q33_q40_exact_guard'
Q48_K12_GUARD_ID = '2f22_q48_m75000_k12_exact_guard'
Q32_LOWK_C3D2_GUARD_ID = 'c3d2_q32_m100000_lowk_k20_exact_guard'
Q32_K31_C3D2_GUARD_ID = 'eaf7_q32_m100000_k31_exact_guard'
D64_TAIL_017A_GUARD_ID = '017a_v12_d64_long_m_tail_exact_guard'
D256_Q4_E2DF_GUARD_ID = 'e2df_v12_d256_q4_k10_long_m_exact_guard'
D256_Q128_59FE_GUARD_ID = '59fe_v12_d256_q128_k10_exact_guard'
D256_K32_59FE_GUARD_ID = '59fe_v12_d256_k32_tail_exact_guard'
HIGHD_RAG_22E9_GUARD_ID = '22e9_v12_highd_rag_exact_guard'
HIGHD_SEARCH_BE66_GUARD_ID = 'be66_v12_highd_search_exact_guard'
D128_K48_DD2B_GUARD_ID = 'dd2b_v12_d128_q16_m100000_k48_exact_guard'
RECT_D128_K20_S12WARP4_GUARD_ID = '7768_rect_d128_k20_q1536_s12warp4_exact_guard'
Q128_K10_ROWLD_1BED_GUARD_ID = '1bed_rowld_rag_stream_k10_q128_s74_exact_guard'
BUILD_D256 = d256_build.BUILD_D256_Q1024
BUILD_D768 = highd_build.BUILD_D768
BUILD_D1024 = highd_build.BUILD_D1024
BUILD_D4096 = highd_build.BUILD_D4096
D64_Q4096 = d64_q4096_c271.TARGET_SHAPE
D768_SEARCH = d768_search.SEARCH_D768
SEARCH_D256 = search_d256.SEARCH_D256
D768_RAG = d768_rag.D768_SHAPE
RAG_D1024 = highd_rag.RAG_D1024
RAG_D4096 = highd_rag.RAG_D4096
RAG_D64 = rag_d64d256.RAG_D64
RAG_D256 = rag_d64d256.RAG_D256
BUILD_K13 = 'build_k_sweep_qm4096_k13'
BUILD_K48_Q2048 = 'build_over32_stress_qm2048_k48'
BUILD_K48_Q4096 = 'build_over32_stress_qm4096_k48'
RAG_Q128_M100000_K32 = 'rag_stream_largek_b1_q128_m100000_d128_k32'
RAG_Q128_M131071_K32 = 'rag_stream_largek_b1_q128_m131071_d128_k32'
RAG_Q128_M100000_K10 = q128_k10_warpmerge.TARGET_SHAPE
RAG_Q16_M250000_K32 = 'rag_microbatch_largek_b1_q16_m250000_d128_k32'
BUILD_LARGE_SQUARE_K32 = large_square_k32_efe4.TARGET_SHAPES[0]
RAG_STREAM_K10 = RAG_Q128_M100000_K10
RAG_Q1_M524287_K10 = q1_m524_n128.ONLINE_M524K_SHAPE
RAG_Q32_M100000_K32 = q32exact.Q32_K32_SHAPE
EXPANDED_Q31_M100000_K32 = 'expanded_guard_boundary_q31_m100000_d128_k32'
EXPANDED_Q33_M100000_K32 = 'expanded_guard_boundary_q33_m100000_d128_k32'
EXPANDED_Q32_M99999_K32 = 'expanded_tail_q32_m99999_d128_k32'
EXPANDED_Q32_M100001_K32 = 'expanded_tail_q32_m100001_d128_k32'
EXPANDED_Q40_M100000_K32 = 'expanded_heldout_q40_m100000_d128_k32'
EXPANDED_Q32_M100000_K20 = 'expanded_guard_overlap_q32_m100000_d128_k20'
EXPANDED_Q32_M100000_K31 = 'expanded_guard_miss_q32_m100000_d128_k31'
EXPANDED_Q48_M75000_K12 = 'expanded_random_q48_m75000_d128_k12'
EXPANDED_Q32_GUARD_BOUNDARY_8_SHAPES = (EXPANDED_Q31_M100000_K32, EXPANDED_Q33_M100000_K32, EXPANDED_Q32_M99999_K32, EXPANDED_Q32_M100001_K32, EXPANDED_Q40_M100000_K32, EXPANDED_Q32_M100000_K20, EXPANDED_Q32_M100000_K31, EXPANDED_Q48_M75000_K12)
EXPANDED_Q32TAIL_CONSUMED_SHAPES = (EXPANDED_Q32_M99999_K32, EXPANDED_Q32_M100001_K32)
EXPANDED_Q32_GUARD_BOUNDARY_8_BY_LABEL = {EXPANDED_Q31_M100000_K32: {'label': EXPANDED_Q31_M100000_K32, 'params': {'B': 1, 'Q': 31, 'M': 100000, 'D': 128, 'K': 32, 'dtype': 'bfloat16', 'seed': 626331, 'build': False, 'check_correctness': True, 'correctness_query_sample': 31, 'recall_min': 0.999, 'benchmark': True, 'time_flashlib': True}}, EXPANDED_Q33_M100000_K32: {'label': EXPANDED_Q33_M100000_K32, 'params': {'B': 1, 'Q': 33, 'M': 100000, 'D': 128, 'K': 32, 'dtype': 'bfloat16', 'seed': 626333, 'build': False, 'check_correctness': True, 'correctness_query_sample': 33, 'recall_min': 0.999, 'benchmark': True, 'time_flashlib': True}}, EXPANDED_Q32_M99999_K32: {'label': EXPANDED_Q32_M99999_K32, 'params': {'B': 1, 'Q': 32, 'M': 99999, 'D': 128, 'K': 32, 'dtype': 'bfloat16', 'seed': 626999, 'build': False, 'check_correctness': True, 'correctness_query_sample': 32, 'recall_min': 0.999, 'benchmark': True, 'time_flashlib': True}}, EXPANDED_Q32_M100001_K32: {'label': EXPANDED_Q32_M100001_K32, 'params': {'B': 1, 'Q': 32, 'M': 100001, 'D': 128, 'K': 32, 'dtype': 'bfloat16', 'seed': 627001, 'build': False, 'check_correctness': True, 'correctness_query_sample': 32, 'recall_min': 0.999, 'benchmark': True, 'time_flashlib': True}}, EXPANDED_Q40_M100000_K32: {'label': EXPANDED_Q40_M100000_K32, 'params': {'B': 1, 'Q': 40, 'M': 100000, 'D': 128, 'K': 32, 'dtype': 'bfloat16', 'seed': 626440, 'build': False, 'check_correctness': True, 'correctness_query_sample': 40, 'recall_min': 0.999, 'benchmark': True, 'time_flashlib': True}}, EXPANDED_Q32_M100000_K20: {'label': EXPANDED_Q32_M100000_K20, 'params': {'B': 1, 'Q': 32, 'M': 100000, 'D': 128, 'K': 20, 'dtype': 'bfloat16', 'seed': 626320, 'build': False, 'check_correctness': True, 'correctness_query_sample': 32, 'recall_min': 0.999, 'benchmark': True, 'time_flashlib': True}}, EXPANDED_Q32_M100000_K31: {'label': EXPANDED_Q32_M100000_K31, 'params': {'B': 1, 'Q': 32, 'M': 100000, 'D': 128, 'K': 31, 'dtype': 'bfloat16', 'seed': 6263310, 'build': False, 'check_correctness': True, 'correctness_query_sample': 32, 'recall_min': 0.999, 'benchmark': True, 'time_flashlib': True}}, EXPANDED_Q48_M75000_K12: {'label': EXPANDED_Q48_M75000_K12, 'params': {'B': 1, 'Q': 48, 'M': 75000, 'D': 128, 'K': 12, 'dtype': 'bfloat16', 'seed': 626812, 'build': False, 'check_correctness': True, 'correctness_query_sample': 48, 'recall_min': 0.999, 'benchmark': True, 'time_flashlib': True}}}
FLOOR_REPAIR_SEED_SHAPES = (BUILD_K13, BUILD_K48_Q2048, BUILD_K48_Q4096, RAG_Q128_M100000_K32, RAG_Q128_M131071_K32, RAG_Q16_M250000_K32)
MIXED_CONSUMED_SEED_SHAPES = (BUILD_D256, BUILD_D768, BUILD_D1024, BUILD_D4096, D768_SEARCH, D768_RAG, RAG_D1024, RAG_D4096, SEARCH_D256, RAG_D256)
D64_M128_CONSUMED_SEED_SHAPES = (BUILD_D256, BUILD_D768, BUILD_D1024, BUILD_D4096, D768_SEARCH, D768_RAG, RAG_D1024, RAG_D4096, RAG_D64, SEARCH_D256, RAG_D256)
C271_CONSUMED_SEED_SHAPES = (*D64_M128_CONSUMED_SEED_SHAPES, D64_Q4096)
C271_7DC5_CONSUMED_SEED_SHAPES = (*C271_CONSUMED_SEED_SHAPES, BUILD_K13, BUILD_K48_Q2048, BUILD_K48_Q4096)
PRE_34DA_CONSUMED_SEED_SHAPES = (*C271_CONSUMED_SEED_SHAPES, *FLOOR_REPAIR_SEED_SHAPES, BUILD_LARGE_SQUARE_K32)
PRE_EXPANDED_K32_Q48_CONSUMED_SEED_SHAPES = (*PRE_34DA_CONSUMED_SEED_SHAPES, RAG_STREAM_K10, RAG_Q1_M524287_K10, RAG_Q32_M100000_K32, *EXPANDED_Q32TAIL_CONSUMED_SHAPES)
EXPANDED_K32_Q31_Q33_Q40_Q48_CONSUMED_SHAPES = (EXPANDED_Q31_M100000_K32, EXPANDED_Q33_M100000_K32, EXPANDED_Q40_M100000_K32, EXPANDED_Q48_M75000_K12)
PRE_LOWK_C3D2_CONSUMED_SEED_SHAPES = (*PRE_EXPANDED_K32_Q48_CONSUMED_SEED_SHAPES, *EXPANDED_K32_Q31_Q33_Q40_Q48_CONSUMED_SHAPES)
EXPANDED_Q32_LOWK_C3D2_CONSUMED_SHAPES = (EXPANDED_Q32_M100000_K20,)
EXPANDED_Q32_K31_C3D2_CONSUMED_SHAPES = (EXPANDED_Q32_M100000_K31,)
RAG_ONLINE_D64_Q1_M262143_K10 = d64_tail_017a.RAG_ONLINE_D64_Q1_M262
RAG_MICRO_D64_Q4_M100000_K10 = d64_tail_017a.RAG_MICRO_D64_Q4_M100
D64_TAIL_017A_CONSUMED_SHAPES = d64_tail_017a.TARGET_SHAPES
PRE_D64_TAIL_CONSUMED_SEED_SHAPES = (*PRE_LOWK_C3D2_CONSUMED_SEED_SHAPES, *EXPANDED_Q32_LOWK_C3D2_CONSUMED_SHAPES, *EXPANDED_Q32_K31_C3D2_CONSUMED_SHAPES)
RAG_MICRO_D256_Q4_M100000_K10 = d256_q4_e2df.RAG_MICRO_D256_Q4_M100
D256_Q4_E2DF_CONSUMED_SHAPES = d256_q4_e2df.TARGET_SHAPES
PRE_D256_Q4_E2DF_CONSUMED_SEED_SHAPES = (*PRE_D64_TAIL_CONSUMED_SEED_SHAPES, *D64_TAIL_017A_CONSUMED_SHAPES)
PRE_D256_Q128_59FE_CONSUMED_SEED_SHAPES = (*PRE_D256_Q4_E2DF_CONSUMED_SEED_SHAPES, *D256_Q4_E2DF_CONSUMED_SHAPES)
RAG_STREAM_D256_Q128_M100000_K10 = d256_q128_59fe.RAG_STREAM_D256_Q128_M100
D256_Q128_59FE_CONSUMED_SHAPES = d256_q128_59fe.TARGET_SHAPES
PRE_D256_K32_59FE_CONSUMED_SEED_SHAPES = (*PRE_D256_Q128_59FE_CONSUMED_SEED_SHAPES, *D256_Q128_59FE_CONSUMED_SHAPES)
RAG_MICRO_D256_Q8_M100000_K32 = d256_k32_59fe.RAG_MICRO_D256_Q8_M100_K32
RAG_STREAM_D256_Q128_M100000_K32 = d256_k32_59fe.RAG_STREAM_D256_Q128_M100_K32
D256_K32_59FE_CONSUMED_SHAPES = d256_k32_59fe.TARGET_SHAPES
HIGHD_RAG_D768_Q8_M100000_K10 = highd_rag_22e9.RAG_D768
HIGHD_RAG_D1024_Q4_M100000_K10 = highd_rag_22e9.RAG_D1024
HIGHD_RAG_D4096_Q1_M65536_K10 = highd_rag_22e9.RAG_D4096
HIGHD_RAG_22E9_CONSUMED_SHAPES = highd_rag_22e9.TARGET_SHAPES
HIGHD_SEARCH_D1024_Q256_M8192_K10 = highd_search_be66.SEARCH_D1024
HIGHD_SEARCH_D4096_Q128_M4096_K10 = highd_search_be66.SEARCH_D4096
HIGHD_SEARCH_BE66_CONSUMED_SHAPES = highd_search_be66.TARGET_SHAPES
V12_D128_K48_OVER32 = d128_k48_dd2b.TARGET_SHAPE
D128_K48_DD2B_CONSUMED_SHAPES = d128_k48_dd2b.TARGET_SHAPES
RECT_D128_K20_Q1536 = rect_d128_k20_s12warp4.TARGET_SHAPE
RECT_D128_K20_S12WARP4_CONSUMED_SHAPES = rect_d128_k20_s12warp4.TARGET_SHAPES
PRE_HIGHD_RAG_22E9_CONSUMED_SEED_SHAPES = (*PRE_D256_K32_59FE_CONSUMED_SEED_SHAPES, *D256_K32_59FE_CONSUMED_SHAPES)
PRE_HIGHD_SEARCH_BE66_CONSUMED_SEED_SHAPES = (*PRE_HIGHD_RAG_22E9_CONSUMED_SEED_SHAPES, *HIGHD_RAG_22E9_CONSUMED_SHAPES)
PRE_D128_K48_DD2B_CONSUMED_SEED_SHAPES = (*PRE_HIGHD_SEARCH_BE66_CONSUMED_SEED_SHAPES, *HIGHD_SEARCH_BE66_CONSUMED_SHAPES)
PRE_RECT_D128_K20_S12WARP4_CONSUMED_SEED_SHAPES = (*PRE_D128_K48_DD2B_CONSUMED_SEED_SHAPES, *D128_K48_DD2B_CONSUMED_SHAPES)
CONSUMED_SEED_SHAPES = (*PRE_RECT_D128_K20_S12WARP4_CONSUMED_SEED_SHAPES, *RECT_D128_K20_S12WARP4_CONSUMED_SHAPES)
FOCUS_COMMON_D_SHAPES = base_current.FOCUS_SHAPES
eval_mod = base_current.eval_mod
ir = _decode_capture(_json_loads('{"__ir__": "knn_build_non128_frontier_7231_stage1_d768", "arg_keys": ["tmap_query", "tmap_database", "query_sq", "database_sq", "partial_dists", "partial_indices", "B", "Q", "M", "K", "num_q_tiles", "db_tiles_per_split", "split_count", "total_work"], "cluster_dims": [1, 1, 1], "computed_smem_bytes": 50432, "constants": [["BLOCK_Q", 128], ["BLOCK_M", 64], ["K_TILE", 128], ["TOP_K_MAX", 10], ["FEATURE_CHUNKS", 6]], "cta_group": 1, "threads": 192}'))
SOURCE_TASKS = {**base_current.SOURCE_TASKS, 'common_d256_q1024_56f3_v1': 'weave-evolve-knn-build-165c D256 Q1024 build seed, retained as a4ec baseline route', 'common_d_56f3_build_d256_q1024_v1': 'weave-evolve-knn-build-6164 faster D256 Q1024 build seed', 'common_d_eeff_search_d768_v1': 'weave-evolve-knn-build-cda9 D768 rectangular search seed', 'common_d768_build_eeff_m64split_v1': 'weave-evolve-knn-build-34d8 fastest D768 build seed', 'common_d_56f3_build_highd_v1': 'weave-evolve-knn-build-f1d9 high-D build seed', 'non128_frontier_4be7_d768fused_v1': 'generalize-auto-tuning-knn-build-447d D768 RAG guard replay', 'common_d_5e7f_rag_highd_v1': 'weave-evolve-knn-build-4cf7 high-D RAG D1024/D4096 seed', 'common_d_5e7f_search_d256_v1': 'weave-evolve-knn-build-8dbc D256 rectangular search seed', 'common_d_5e7f_rag_d64_d256_v1': 'weave-evolve-knn-build-ba22 D64/D256 RAG seed; D64 remains below the 1.20x floor', 'common_d_5e7f_rag_d64_repair_v1': 'weave-evolve-knn-build-0474 repaired D64 RAG seed', 'common_d_1438_rag_d64_m128_v1': 'weave-evolve-knn-build-631e D64 RAG M128 backfill seed', 'd64_q4096_c271_twostage_v1': 'weave-evolve-knn-build-8f70-q4096-c271-refresh D64 Q4096 exact seed', 'knn_build_k13_k48_floor_repair_7dc5_v1': 'weave-evolve-knn-build-4bd4 exact K13/K48 floor-repair wrapper', SEED_LARGE_SQUARE_K32_EFE4_ID: 'weave-evolve-knn-build-fac0 EFE4 large-square K32 producer-cache seed', SEED_K13_ID: 'weave-evolve-knn-build-4bd4 inherited 2c1c Q4096 K13 unordered split4 seed', SEED_K48_ID: 'weave-evolve-knn-build-4bd4 inherited d03c K48 split4 warp-select seed', SEED_K13_A162_ID: 'weave-evolve-knn-build-3b00 promoted K13 source; 7bef best timing evidence', SEED_Q128_DUALM_ID: 'weave-evolve-knn-build-5698 Q128 K32 dual-M rowld/warp1 seed', SEED_Q128_S72R2_ID: 'weave-evolve-knn-build-681b Q128 K32 split72 rows2 seed', SEED_Q128_K10_WARPMERGE_ID: 'weave-evolve-knn-build-4ce8 Q128/M100000/D128/K10 split74 warp-merge seed', SEED_Q128_K10_ROWLD_1BED_ID: 'weave-evolve-knn-build-41bb exact Q128/M100000/D128/K10 split74 row-load seed', SEED_Q16_M250_ID: 'weave-evolve-knn-build-cb99 Q16/M250000 K32 split288 seed', SEED_RAG_STREAM_K10_34DA_ID: 'weave-evolve-knn-build-34da RAG stream K10 split72 warp-row merge seed', SEED_Q1_M524_N128_ID: 'weave-evolve-knn-build-32b9 Q1/M524287/D128/K10 M64/N128 seed', SEED_Q32_EXACT_ID: 'weave-evolve-knn-build-12ac exact Q32/M100000/D128/K32 f653 rowld2 stage1 seed', SEED_Q32TAIL_ID: 'weave-evolve-knn-build-6866 Q32/M99999 and Q32/M100001 K32 tail seed', SEED_Q32TAIL143_LOW_ID: 'weave-evolve-knn-build-7eed consumed 5317 split143 Q32/M99999 K32 tail seed', SEED_Q32TAIL143_HIGH_ID: 'weave-evolve-knn-build-7eed consumed 314c split143 Q32/M100001 K32 tail seed', SEED_Q31TAIL_V2_ID: 'weave-evolve-knn-build-0cb5 Q31 exact rowld2 seed', SEED_Q33TILE_ID: 'weave-evolve-knn-build-c489 Q33/Q40 expanded K32 seed', SEED_Q48_K12_ID: 'weave-evolve-knn-build-2f22 Q48/M75000/K12 exact low-K seed', SEED_Q32_LOWK_C3D2_ID: 'weave-evolve-knn-build-4cc4 Q32/M100000 low-K K20 seed', SEED_Q32_K31_C3D2_ID: 'weave-evolve-knn-build-eaf7 Q32/M100000 K31 exact seed; clears expanded K31 floor', SEED_D64_TAIL_017A_ID: 'weave-evolve-knn-build-6e8c exact v12 D64 long-M tail seed', SEED_D256_Q4_E2DF_ID: 'weave-evolve-knn-build-fd28 exact v12 D256 Q4/K10 long-M seed', SEED_D256_Q128_59FE_ID: 'weave-evolve-knn-build-93f5 exact v12 D256 Q128/K10 long-M seed', SEED_D256_K32_59FE_ID: 'weave-evolve-knn-build-9203 exact v12 D256 K32 long-M seed', SEED_HIGHD_RAG_22E9_ID: 'weave-evolve-knn-build-7902 exact v12 high-D RAG seed', SEED_HIGHD_SEARCH_BE66_ID: 'weave-evolve-knn-build-ad73 exact v12 high-D rectangular search seed', SEED_D128_K48_DD2B_ID: 'weave-evolve-knn-build-dd2b exact v12 D128 Q16/M100000 K48 seed', SEED_RECT_D128_K20_S12WARP4_ID: 'weave-evolve-knn-build-0e29 exact rectangular D128/K20/Q1536 split12/warp4 seed'}
PRODUCTION_ROUTE_MODULES = {**base_current.PRODUCTION_ROUTE_MODULES, 'common_d_56f3_build_d256_q1024_v1': D256_ENTRYPOINT, 'common_d_eeff_search_d768_v1': D768_SEARCH_ENTRYPOINT, 'common_d768_build_eeff_m64split_v1': D768_FAST_ENTRYPOINT, 'common_d_56f3_build_highd_v1': HIGHD_ENTRYPOINT, 'non128_frontier_4be7_d768fused_v1': D768_RAG_ENTRYPOINT, 'common_d_5e7f_rag_highd_v1': HIGHD_RAG_ENTRYPOINT, 'common_d_5e7f_search_d256_v1': SEARCH_D256_ENTRYPOINT, 'common_d_5e7f_rag_d64_d256_v1': RAG_D64D256_ENTRYPOINT, 'common_d_5e7f_rag_d64_repair_v1': RAG_D64_REPAIR_ENTRYPOINT, 'common_d_1438_rag_d64_m128_v1': RAG_D64_M128_ENTRYPOINT, 'd64_q4096_c271_twostage_v1': D64_Q4096_C271_ENTRYPOINT, SEED_LARGE_SQUARE_K32_EFE4_ID: LARGE_SQUARE_K32_EFE4_ENTRYPOINT, 'knn_build_k13_k48_floor_repair_7dc5_v1': K13_K48_WRAPPER_ENTRYPOINT, SEED_K13_ID: K13_ENTRYPOINT, SEED_K48_ID: K48_ENTRYPOINT, SEED_K13_A162_ID: K13_ENTRYPOINT, SEED_Q128_DUALM_ID: Q128_DUALM_ENTRYPOINT, SEED_Q128_S72R2_ID: Q128_S72R2_ENTRYPOINT, SEED_Q128_K10_WARPMERGE_ID: Q128_K10_WARPMERGE_ENTRYPOINT, SEED_Q128_K10_ROWLD_1BED_ID: Q128_K10_ROWLD_1BED_ENTRYPOINT, SEED_Q16_M250_ID: Q16_M250_ENTRYPOINT, SEED_RAG_STREAM_K10_34DA_ID: RAG_STREAM_K10_34DA_ENTRYPOINT, SEED_Q1_M524_N128_ID: Q1_M524_N128_ENTRYPOINT, SEED_Q32_EXACT_ID: Q32_EXACT_ENTRYPOINT, SEED_Q32TAIL_ID: Q32TAIL_ENTRYPOINT, SEED_Q32TAIL143_LOW_ID: Q32TAIL143_LOW_ENTRYPOINT, SEED_Q32TAIL143_HIGH_ID: Q32TAIL143_HIGH_ENTRYPOINT, SEED_Q31TAIL_V2_ID: Q31TAIL_V2_ENTRYPOINT, SEED_Q33TILE_ID: Q33TILE_ENTRYPOINT, SEED_Q48_K12_ID: Q48_K12_ENTRYPOINT, SEED_Q32_LOWK_C3D2_ID: Q32_LOWK_C3D2_ENTRYPOINT, SEED_Q32_K31_C3D2_ID: Q32_K31_C3D2_ENTRYPOINT, SEED_D64_TAIL_017A_ID: D64_TAIL_017A_ENTRYPOINT, SEED_D256_Q4_E2DF_ID: D256_Q4_E2DF_ENTRYPOINT, SEED_D256_Q128_59FE_ID: D256_Q128_59FE_ENTRYPOINT, SEED_D256_K32_59FE_ID: D256_K32_59FE_ENTRYPOINT, SEED_HIGHD_RAG_22E9_ID: HIGHD_RAG_22E9_ENTRYPOINT, SEED_HIGHD_SEARCH_BE66_ID: HIGHD_SEARCH_BE66_ENTRYPOINT, SEED_D128_K48_DD2B_ID: D128_K48_DD2B_ENTRYPOINT, SEED_RECT_D128_K20_S12WARP4_ID: RECT_D128_K20_S12WARP4_ENTRYPOINT, CANDIDATE_FA04_BASE: FA04_BASE_ENTRYPOINT, CANDIDATE_F328_BASE: F328_BASE_ENTRYPOINT, CANDIDATE_MIXED: MIXED_BASE_ENTRYPOINT, CANDIDATE_D64_REPAIR: D64_REPAIR_BASE_ENTRYPOINT, CANDIDATE_D64_M128: D64_M128_BASE_ENTRYPOINT, CANDIDATE_D64_Q4096_C271: D64_Q4096_C271_BASE_ENTRYPOINT, CANDIDATE_C271_7DC5_K13_K48: C271_7DC5_BASE_ENTRYPOINT, CANDIDATE_2498_BASELINE: BASE_2498_ENTRYPOINT, CANDIDATE_PRE_Q1_M524_N128: PRE_Q1_M524_N128_BASE_ENTRYPOINT, CANDIDATE_PRE_Q32_EXACT: PRE_Q32_EXACT_BASE_ENTRYPOINT, CANDIDATE_PRE_Q32TAIL: PRE_Q32TAIL_BASE_ENTRYPOINT, CANDIDATE_PRE_EXPANDED_K32_Q48: PRE_EXPANDED_K32_Q48_BASE_ENTRYPOINT, CANDIDATE_PRE_LOWK_C3D2: PRE_LOWK_C3D2_BASE_ENTRYPOINT, CANDIDATE_D665_LOWK_BASELINE: D665_LOWK_BASE_ENTRYPOINT, CANDIDATE_PRE_D64_TAIL_017A: PRE_D64_TAIL_017A_BASE_ENTRYPOINT, CANDIDATE_PRE_D256_Q4_E2DF: PRE_D256_Q4_E2DF_BASE_ENTRYPOINT, CANDIDATE_D256_Q128_59FE: PRE_D256_K32_59FE_BASE_ENTRYPOINT, CANDIDATE_D256_K32_59FE: PRE_HIGHD_RAG_22E9_BASE_ENTRYPOINT, CANDIDATE_HIGHD_RAG_22E9: PRE_HIGHD_SEARCH_BE66_BASE_ENTRYPOINT, CANDIDATE_HIGHD_SEARCH_BE66: PRE_D128_K48_DD2B_BASE_ENTRYPOINT, CANDIDATE_D128_K48_DD2B: PRE_RECT_D128_K20_S12WARP4_BASE_ENTRYPOINT, CANDIDATE_RECT_D128_K20_S12WARP4: ROUTE_ENTRYPOINT, CANDIDATE_Q128_K10_ROWLD_1BED: ROUTE_ENTRYPOINT, CANDIDATE_Q128_K10_WARPMERGE: ROUTE_ENTRYPOINT, CANDIDATE_FLOOR_SEEDS_Q128_5698: ROUTE_ENTRYPOINT, CANDIDATE_FLOOR_SEEDS_Q128_681B: ROUTE_ENTRYPOINT, CANDIDATE_FLOOR_SEEDS_Q128_MIXED: ROUTE_ENTRYPOINT, CANDIDATE_SELECTED_SYNTHESIS: ROUTE_ENTRYPOINT}
D64_REPAIR_0474_TIMING = {'seed_id': 'common_d_5e7f_rag_d64_repair_v1', 'kernel_ms': 0.049792, 'flashlib_ms': 0.065376, 'ratio_vs_flashlib': 1.312982005141388, 'tflops': 2.056555269922879, 'timing_backend': 'cupti', 'source_payload': 'design_doc/active/weave_evolve_knn_build_round_166_5e7f_d64repair.md'}
SEED_TIMING_ROWS = {BUILD_D256: {'seed_id': 'common_d_56f3_build_d256_q1024_v1', 'kernel_ms': 0.028256, 'flashlib_ms': 0.074304, 'ratio_vs_flashlib': 2.629671574178935, 'tflops': 19.000244620611554, 'timing_backend': 'cupti', 'source_payload': 'design_doc/active/weave_evolve_knn_build_round_161_56f3_d256q1024.md'}, BUILD_D768: {'seed_id': 'common_d768_build_eeff_m64split_v1', 'kernel_ms': 0.030432, 'flashlib_ms': 0.108385, 'ratio_vs_flashlib': 3.5615470557308093, 'tflops': 52.92497160883281, 'timing_backend': 'cupti', 'source_payload': 'artifacts/weave_evolve/knn_build_common_d768_build_eeff_m64split_v1/common_d768_build_eeff_m64split_v1.json'}, BUILD_D1024: {'seed_id': 'common_d_56f3_build_highd_v1', 'kernel_ms': 0.037856, 'flashlib_ms': 0.122848, 'ratio_vs_flashlib': 3.2451394759087067, 'tflops': 14.181923922231615, 'timing_backend': 'cupti', 'source_payload': 'design_doc/active/weave_evolve_knn_build_round_161_56f3.md'}, BUILD_D4096: {'seed_id': 'common_d_56f3_build_highd_v1', 'kernel_ms': 0.058336, 'flashlib_ms': 0.33136, 'ratio_vs_flashlib': 5.6801974766867795, 'tflops': 36.81232254525507, 'timing_backend': 'cupti', 'source_payload': 'design_doc/active/weave_evolve_knn_build_round_161_56f3.md'}, D768_SEARCH: {'seed_id': 'common_d_eeff_search_d768_v1', 'kernel_ms': 0.061664, 'flashlib_ms': 0.163041, 'ratio_vs_flashlib': 2.644022444213804, 'tflops': 104.4766953814219, 'timing_backend': 'cupti', 'source_payload': 'design_doc/active/weave_evolve_knn_build_round_161_eeff.md'}, D768_RAG: {'seed_id': 'non128_frontier_4be7_d768fused_v1', 'kernel_ms': 0.097568, 'flashlib_ms': 0.171969, 'ratio_vs_flashlib': 1.762555346015087, 'tflops': 12.594293210888814, 'timing_backend': 'cupti', 'source_payload': 'artifacts/generalize_auto_tuning/knn_build_447d_v11_common_d_4be7_replay/full100_dispatch_v11_common_d_4be7_447d_v1.json'}, RAG_D1024: {'seed_id': 'common_d_5e7f_rag_highd_v1', 'kernel_ms': 0.075233, 'flashlib_ms': 0.1469445, 'ratio_vs_flashlib': 1.9531920832613345, 'tflops': 10.888838674517832, 'timing_backend': 'cupti', 'source_payload': 'design_doc/active/weave_evolve_knn_build_round_165_5e7f.md'}, RAG_D4096: {'seed_id': 'common_d_5e7f_rag_highd_v1', 'kernel_ms': 0.160865, 'flashlib_ms': 0.309859, 'ratio_vs_flashlib': 1.926205203120629, 'tflops': 6.6748007583999005, 'timing_backend': 'cupti', 'source_payload': 'design_doc/active/weave_evolve_knn_build_round_165_5e7f.md'}, SEARCH_D256: {'seed_id': 'common_d_5e7f_search_d256_v1', 'kernel_ms': 0.19437, 'flashlib_ms': 0.466597, 'ratio_vs_flashlib': 2.400560786129547, 'tflops': 88.38745271389618, 'timing_backend': 'cupti', 'source_payload': 'design_doc/active/weave_evolve_knn_build_round_165_5e7f_search_d256.md'}, RAG_D64: {'seed_id': 'common_d_1438_rag_d64_m128_v1', 'kernel_ms': 0.036288, 'flashlib_ms': 0.068641, 'ratio_vs_flashlib': 1.8915619488536153, 'tflops': 2.821869488536155, 'timing_backend': 'cupti', 'source_payload': 'design_doc/active/weave_evolve_knn_build_round_166_1438_d64_m128.md'}, RAG_D256: {'seed_id': 'common_d_5e7f_rag_d64_d256_v1', 'kernel_ms': 0.060353, 'flashlib_ms': 0.081472, 'ratio_vs_flashlib': 1.3499246102099316, 'tflops': 5.396514610821611, 'timing_backend': 'cupti', 'source_payload': 'design_doc/active/weave_evolve_knn_build_round_165_5e7f_d64d256.md'}, RAG_ONLINE_D64_Q1_M262143_K10: {'seed_id': SEED_D64_TAIL_017A_ID, 'kernel_ms': 0.059648, 'flashlib_ms': 0.080672, 'ratio_vs_flashlib': 1.3524678111587982, 'tflops': 0.5625386266094421, 'timing_backend': 'cupti', 'source_payload': 'artifacts/weave_evolve/knn_build_v12_d64_tail_017a_v1/v12_d64_tail_017a_2row_cupti.json', 'source_task': 'weave-evolve-knn-build-6e8c'}, RAG_MICRO_D64_Q4_M100000_K10: {'seed_id': SEED_D64_TAIL_017A_ID, 'kernel_ms': 0.033217, 'flashlib_ms': 0.067745, 'ratio_vs_flashlib': 2.0394677424210492, 'tflops': 1.5413794141554022, 'timing_backend': 'cupti', 'source_payload': 'artifacts/weave_evolve/knn_build_v12_d64_tail_017a_v1/v12_d64_tail_017a_2row_cupti.json', 'source_task': 'weave-evolve-knn-build-6e8c'}, RAG_MICRO_D256_Q4_M100000_K10: {'seed_id': SEED_D256_Q4_E2DF_ID, 'kernel_ms': 0.056352, 'flashlib_ms': 0.084225, 'ratio_vs_flashlib': 1.494626641467916, 'tflops': 3.6342986939239066, 'timing_backend': 'cupti', 'source_payload': 'artifacts/weave_evolve/knn_build_v12_d256_k10_longm_e2df_v1/d256_q4_cupti.json', 'source_task': 'weave-evolve-knn-build-fd28'}, RAG_STREAM_D256_Q128_M100000_K10: {'seed_id': SEED_D256_Q128_59FE_ID, 'kernel_ms': 0.068289, 'flashlib_ms': 0.156577, 'ratio_vs_flashlib': 2.2928582934293953, 'tflops': 95.96860402114541, 'timing_backend': 'cupti', 'source_payload': 'artifacts/weave_evolve/knn_build_v12_d256_q128_k10_longm_59fe_v1/d256_q128_cupti.json', 'source_task': 'weave-evolve-knn-build-93f5'}, RAG_MICRO_D256_Q8_M100000_K32: {'seed_id': SEED_D256_K32_59FE_ID, 'kernel_ms': 0.100768, 'flashlib_ms': 0.122464, 'ratio_vs_flashlib': 1.2153064464909495, 'tflops': 4.0647824706255955, 'timing_backend': 'cupti', 'source_payload': 'artifacts/weave_evolve/knn_build_v12_d256_k32_tail_59fe_v1/d256_k32_tail_59fe_v1.json', 'source_task': 'weave-evolve-knn-build-9203'}, RAG_STREAM_D256_Q128_M100000_K32: {'seed_id': SEED_D256_K32_59FE_ID, 'kernel_ms': 0.2519695, 'flashlib_ms': 0.30637, 'ratio_vs_flashlib': 1.2159011308908418, 'tflops': 26.009497181206456, 'timing_backend': 'cupti', 'source_payload': 'artifacts/weave_evolve/knn_build_v12_d256_k32_tail_59fe_v1/d256_k32_tail_59fe_v1.json', 'source_task': 'weave-evolve-knn-build-9203'}, HIGHD_RAG_D768_Q8_M100000_K10: {'seed_id': SEED_HIGHD_RAG_22E9_ID, 'kernel_ms': 0.092737, 'flashlib_ms': 0.161858, 'ratio_vs_flashlib': 1.7453443609346864, 'tflops': 13.250374715593562, 'timing_backend': 'cupti', 'source_payload': 'artifacts/weave_evolve/knn_build_v12_highd_rag_22e9_v1/v12_highd_rag_22e9_v1.json', 'source_task': 'weave-evolve-knn-build-7902'}, HIGHD_RAG_D1024_Q4_M100000_K10: {'seed_id': SEED_HIGHD_RAG_22E9_ID, 'kernel_ms': 0.113793, 'flashlib_ms': 0.170305, 'ratio_vs_flashlib': 1.4966210575342949, 'tflops': 7.199036847609255, 'timing_backend': 'cupti', 'source_payload': 'artifacts/weave_evolve/knn_build_v12_highd_rag_22e9_v1/v12_highd_rag_22e9_v1.json', 'source_task': 'weave-evolve-knn-build-7902'}, HIGHD_RAG_D4096_Q1_M65536_K10: {'seed_id': SEED_HIGHD_RAG_22E9_ID, 'kernel_ms': 0.300963, 'flashlib_ms': 0.377251, 'ratio_vs_flashlib': 1.2534796636131353, 'tflops': 1.783843568810784, 'timing_backend': 'cupti', 'source_payload': 'artifacts/weave_evolve/knn_build_v12_highd_rag_22e9_v1/v12_highd_rag_22e9_v1.json', 'source_task': 'weave-evolve-knn-build-7902'}, HIGHD_SEARCH_D1024_Q256_M8192_K10: {'seed_id': SEED_HIGHD_SEARCH_BE66_ID, 'kernel_ms': 0.036609, 'flashlib_ms': 0.120065, 'ratio_vs_flashlib': 3.279658007593761, 'tflops': 117.31998404763856, 'timing_backend': 'cupti', 'source_payload': 'artifacts/weave_evolve/knn_build_v12_highd_search_be66_v1/v12_highd_search_be66_v1.json', 'source_task': 'weave-evolve-knn-build-ad73'}, HIGHD_SEARCH_D4096_Q128_M4096_K10: {'seed_id': SEED_HIGHD_SEARCH_BE66_ID, 'kernel_ms': 0.055072, 'flashlib_ms': 0.325762, 'ratio_vs_flashlib': 5.915201917489831, 'tflops': 77.98822080185937, 'timing_backend': 'cupti', 'source_payload': 'artifacts/weave_evolve/knn_build_v12_highd_search_be66_v1/v12_highd_search_be66_v1.json', 'source_task': 'weave-evolve-knn-build-ad73'}, V12_D128_K48_OVER32: {'seed_id': SEED_D128_K48_DD2B_ID, 'kernel_ms': 0.105408, 'flashlib_ms': 0.196066, 'ratio_vs_flashlib': 1.860067547055252, 'tflops': 3.885853066180935, 'timing_backend': 'cupti', 'source_payload': 'artifacts/weave_evolve/knn_build_v12_d128_q16_k48_dd2b_v1/v12_d128_q16_k48_dd2b_v1.json', 'source_task': 'weave-evolve-knn-build-dd2b'}, RECT_D128_K20_Q1536: {'seed_id': SEED_RECT_D128_K20_S12WARP4_ID, 'kernel_ms': 0.446723, 'flashlib_ms': 0.769382, 'ratio_vs_flashlib': 1.7222798020249686, 'tflops': 57.686315179652716, 'timing_backend': 'cupti', 'source_payload': 'artifacts/generalize_auto_tuning/knn_build_7768_rect_d128_k20_q1536_s12warp4/rect_d128_k20_q1536_s12warp4_7768_v1.json', 'source_task': 'weave-evolve-knn-build-0e29'}, D64_Q4096: {'seed_id': 'd64_q4096_c271_twostage_v1', 'kernel_ms': 0.106913, 'flashlib_ms': 0.145249, 'ratio_vs_flashlib': 1.3585719229654019, 'tflops': 20.086272464527234, 'timing_backend': 'cupti', 'source_payload': 'design_doc/active/weave_evolve_knn_build_round_169_8f70_q4096_c271_refresh.md'}, BUILD_K13: {'seed_id': SEED_K13_ID, 'kernel_ms': 0.124261, 'flashlib_ms': 0.164709, 'ratio_vs_flashlib': 1.3255084056944657, 'tflops': 34.56408121614988, 'timing_backend': 'cupti', 'source_payload': 'artifacts/weave_evolve/knn_build_7dc5_k13_k48_floor_repair_round170/7dc5_k13_k48_exact3_dispatch_7dc5_k13_k48_v1.json'}, BUILD_K48_Q2048: {'seed_id': SEED_K48_ID, 'kernel_ms': 0.170214, 'flashlib_ms': 0.3967495, 'ratio_vs_flashlib': 2.3308864135735012, 'tflops': 6.308187481640758, 'timing_backend': 'cupti', 'source_payload': 'artifacts/weave_evolve/knn_build_7dc5_k13_k48_floor_repair_round170/7dc5_k13_k48_exact3_dispatch_7dc5_k13_k48_v1.json'}, BUILD_K48_Q4096: {'seed_id': SEED_K48_ID, 'kernel_ms': 0.294186, 'flashlib_ms': 0.527506, 'ratio_vs_flashlib': 1.7931036827041396, 'tflops': 14.599495883556662, 'timing_backend': 'cupti', 'source_payload': 'artifacts/weave_evolve/knn_build_7dc5_k13_k48_floor_repair_round170/7dc5_k13_k48_exact3_dispatch_7dc5_k13_k48_v1.json'}, BUILD_LARGE_SQUARE_K32: {'seed_id': SEED_LARGE_SQUARE_K32_EFE4_ID, 'kernel_ms': 0.38906, 'flashlib_ms': 0.622566, 'ratio_vs_flashlib': 1.6001799208348324, 'tflops': 44.157377227162904, 'timing_backend': 'cupti', 'source_payload': 'artifacts/weave_evolve/knn_build_large_square_k32_fac0_round171/large_square_k32_efe4_prodcache_fac0_target.json'}, RAG_STREAM_K10: {'seed_id': SEED_RAG_STREAM_K10_34DA_ID, 'kernel_ms': 0.109793, 'flashlib_ms': 0.133857, 'ratio_vs_flashlib': 1.2191760859071161, 'tflops': 29.84525425118177, 'timing_backend': 'cupti', 'source_payload': 'artifacts/weave_evolve/knn_build_34da_rag_stream_k10_warpmerge_round172/rag_stream_k10_warpmerge_34da_v1.json', 'source_task': 'weave-evolve-knn-build-34da'}, RAG_Q1_M524287_K10: {'seed_id': SEED_Q1_M524_N128_ID, 'kernel_ms': 0.095489, 'flashlib_ms': 0.115842, 'ratio_vs_flashlib': 1.2131449695776477, 'tflops': 1.4055804542931645, 'timing_backend': 'cupti', 'source_payload': 'artifacts/weave_evolve/knn_build_ea43_q1m524_n128/ea43_q1m524_n128_candidate_n128_1row_cupti.json', 'source_task': 'weave-evolve-knn-build-32b9'}, RAG_Q32_M100000_K32: {'seed_id': SEED_Q32_EXACT_ID, 'kernel_ms': 0.125057, 'flashlib_ms': 0.159393, 'ratio_vs_flashlib': 1.274570795717153, 'tflops': 6.550612920508248, 'timing_backend': 'cupti', 'source_payload': 'artifacts/weave_evolve/knn_build_q32rowld2exact_f653_v1/q32rowld2exact_s141_f653_v1_cupti.json', 'source_task': 'weave-evolve-knn-build-12ac'}, EXPANDED_Q32_M99999_K32: {'seed_id': SEED_Q32TAIL143_LOW_ID, 'kernel_ms': 0.129665, 'flashlib_ms': 0.159297, 'ratio_vs_flashlib': 1.2285273589634829, 'tflops': 6.317755816912814, 'timing_backend': 'cupti', 'source_payload': 'artifacts/weave_evolve/knn_build_dispatch_d6b5_q32tail143_v1/d6b5_q32tail143_dispatch_8row_cupti.json', 'source_task': 'weave-evolve-knn-build-7eed'}, EXPANDED_Q32_M100001_K32: {'seed_id': SEED_Q32TAIL143_HIGH_ID, 'kernel_ms': 0.131841, 'flashlib_ms': 0.159041, 'ratio_vs_flashlib': 1.2063091147670297, 'tflops': 6.213607239022761, 'timing_backend': 'cupti', 'source_payload': 'artifacts/weave_evolve/knn_build_dispatch_d6b5_q32tail143_v1/d6b5_q32tail143_dispatch_8row_cupti.json', 'source_task': 'weave-evolve-knn-build-7eed'}, EXPANDED_Q31_M100000_K32: {'seed_id': SEED_Q31TAIL_V2_ID, 'kernel_ms': 0.128545, 'flashlib_ms': 0.156065, 'ratio_vs_flashlib': 1.2140884515150339, 'tflops': 6.173713485549808, 'timing_backend': 'cupti', 'source_payload': 'artifacts/weave_evolve/knn_build_0cb5_q31tail_v2/0cb5_q31tail_v2_1row_s153_cupti.json', 'source_task': 'weave-evolve-knn-build-0cb5'}, EXPANDED_Q33_M100000_K32: {'seed_id': SEED_Q33TILE_ID, 'kernel_ms': 0.142273, 'flashlib_ms': 0.218273, 'ratio_vs_flashlib': 1.5341842795189529, 'tflops': 5.937879991284361, 'timing_backend': 'cupti', 'source_payload': 'artifacts/weave_evolve/knn_build_c489_q33tile_v1/c489_q33tile_5row_s144_g12_cupti.json', 'source_task': 'weave-evolve-knn-build-c489'}, EXPANDED_Q40_M100000_K32: {'seed_id': SEED_Q33TILE_ID, 'kernel_ms': 0.132225, 'flashlib_ms': 0.214561, 'ratio_vs_flashlib': 1.6226961618453393, 'tflops': 7.744375118169786, 'timing_backend': 'cupti', 'source_payload': 'artifacts/weave_evolve/knn_build_c489_q33tile_v1/c489_q33tile_5row_s144_g12_cupti.json', 'source_task': 'weave-evolve-knn-build-c489'}, EXPANDED_Q48_M75000_K12: {'seed_id': SEED_Q48_K12_ID, 'kernel_ms': 0.045216, 'flashlib_ms': 0.103297, 'ratio_vs_flashlib': 2.284523177636235, 'tflops': 20.382165605095544, 'timing_backend': 'cupti', 'source_payload': 'artifacts/weave_evolve/knn_build_rag_microbucket_k12_2f22_q48exact_v1/2f22_q48exact_1row_s148_cupti.json', 'source_task': 'weave-evolve-knn-build-2f22'}, EXPANDED_Q32_M100000_K20: {'seed_id': SEED_Q32_LOWK_C3D2_ID, 'kernel_ms': 0.083521, 'flashlib_ms': 0.129696, 'ratio_vs_flashlib': 1.552854970606195, 'tflops': 9.808311682091928, 'timing_backend': 'cupti', 'source_payload': 'artifacts/weave_evolve/knn_build_q32_lowk_c3d2_v1/q32_lowk_c3d2_2row_s152_cupti.json', 'source_task': 'weave-evolve-knn-build-4cc4'}, EXPANDED_Q32_M100000_K31: {'seed_id': SEED_Q32_K31_C3D2_ID, 'kernel_ms': 0.124705, 'flashlib_ms': 0.150465, 'ratio_vs_flashlib': 1.206567499298344, 'tflops': 6.569103083276533, 'timing_backend': 'cupti', 'source_payload': 'artifacts/weave_evolve/knn_build_q32_k31_c3d2_v1/q32_k31_c3d2_1row_s152_cupti.json', 'source_task': 'weave-evolve-knn-build-eaf7'}}
D665_Q32_LOWK_C3D2_TIMING_ROWS = {EXPANDED_Q32_M100000_K20: SEED_TIMING_ROWS[EXPANDED_Q32_M100000_K20], EXPANDED_Q32_M100000_K31: {'seed_id': SEED_Q32_LOWK_C3D2_ID, 'kernel_ms': 0.131041, 'flashlib_ms': 0.149538, 'ratio_vs_flashlib': 1.14115429522058, 'tflops': 6.251478544882899, 'timing_backend': 'cupti', 'source_payload': 'artifacts/weave_evolve/knn_build_q32_lowk_c3d2_v1/q32_lowk_c3d2_2row_s152_cupti.json', 'source_task': 'weave-evolve-knn-build-4cc4'}}
EXPANDED_FALLBACK_TIMING_ROWS = {EXPANDED_Q31_M100000_K32: {'kernel_ms': 14.349643499999999, 'flashlib_ms': 0.155745, 'ratio_vs_flashlib': 0.010853579742242377, 'timing_backend': 'cupti', 'source_payload': 'artifacts/generalize_auto_tuning/knn_build_0cb5_q32exact_dispatcher_consumption/expanded_q32_guard_boundary_8_dispatch_v11_common_d_seed_portfolio_a4ec_v1.json'}, EXPANDED_Q33_M100000_K32: {'kernel_ms': 11.84969, 'flashlib_ms': 0.218306, 'ratio_vs_flashlib': 0.01842292920743074, 'timing_backend': 'cupti', 'source_payload': 'artifacts/generalize_auto_tuning/knn_build_0cb5_q32exact_dispatcher_consumption/expanded_q32_guard_boundary_8_dispatch_v11_common_d_seed_portfolio_a4ec_v1.json'}, EXPANDED_Q32_M99999_K32: {'kernel_ms': 13.201157, 'flashlib_ms': 0.159297, 'ratio_vs_flashlib': 0.012066896863661268, 'timing_backend': 'cupti', 'source_payload': 'artifacts/generalize_auto_tuning/knn_build_0cb5_q32exact_dispatcher_consumption/expanded_q32_guard_boundary_8_dispatch_v11_common_d_seed_portfolio_a4ec_v1.json'}, EXPANDED_Q32_M100001_K32: {'kernel_ms': 13.2865505, 'flashlib_ms': 0.158625, 'ratio_vs_flashlib': 0.011938764692912579, 'timing_backend': 'cupti', 'source_payload': 'artifacts/generalize_auto_tuning/knn_build_0cb5_q32exact_dispatcher_consumption/expanded_q32_guard_boundary_8_dispatch_v11_common_d_seed_portfolio_a4ec_v1.json'}, EXPANDED_Q40_M100000_K32: {'kernel_ms': 11.875965, 'flashlib_ms': 0.216546, 'ratio_vs_flashlib': 0.01823397088152415, 'timing_backend': 'cupti', 'source_payload': 'artifacts/generalize_auto_tuning/knn_build_0cb5_q32exact_dispatcher_consumption/expanded_q32_guard_boundary_8_dispatch_v11_common_d_seed_portfolio_a4ec_v1.json'}, EXPANDED_Q32_M100000_K20: {'kernel_ms': 13.248295, 'flashlib_ms': 0.129665, 'ratio_vs_flashlib': 0.00978729715786069, 'timing_backend': 'cupti', 'source_payload': 'artifacts/generalize_auto_tuning/knn_build_0cb5_q32exact_dispatcher_consumption/expanded_q32_guard_boundary_8_dispatch_v11_common_d_seed_portfolio_a4ec_v1.json'}, EXPANDED_Q32_M100000_K31: {'kernel_ms': 13.315449000000001, 'flashlib_ms': 0.150689, 'ratio_vs_flashlib': 0.011316854579969476, 'timing_backend': 'cupti', 'source_payload': 'artifacts/generalize_auto_tuning/knn_build_0cb5_q32exact_dispatcher_consumption/expanded_q32_guard_boundary_8_dispatch_v11_common_d_seed_portfolio_a4ec_v1.json'}, EXPANDED_Q48_M75000_K12: {'kernel_ms': 9.683852000000002, 'flashlib_ms': 0.102688, 'ratio_vs_flashlib': 0.010604044754091655, 'timing_backend': 'cupti', 'source_payload': 'artifacts/generalize_auto_tuning/knn_build_0cb5_q32exact_dispatcher_consumption/expanded_q32_guard_boundary_8_dispatch_v11_common_d_seed_portfolio_a4ec_v1.json'}}
K13_A162_SOURCE_TIMING = {'seed_id': SEED_K13_A162_ID, 'kernel_ms': 0.124577, 'flashlib_ms': 0.164546, 'ratio_vs_flashlib': 1.3208377148269745, 'tflops': 34.476406527689704, 'timing_backend': 'cupti', 'source_payload': 'artifacts/weave_evolve/knn_build_a162_q4096_k13_floor_repair_round170/a162_q4096_k13_exact1_dispatch_a162_q4096_k13_v1.json', 'source_task': 'weave-evolve-knn-build-3b00'}
K13_A162_BEST_TIMING = {'seed_id': SEED_K13_A162_ID, 'kernel_ms': 0.124449, 'flashlib_ms': 0.164225, 'ratio_vs_flashlib': 1.319616871168109, 'tflops': 34.51186667630917, 'timing_backend': 'cupti', 'source_payload': 'weave-evolve-knn-build-7bef/artifacts/weave_evolve/knn_build_a162_q4096_k13_floor_repair_round170/a162_q4096_k13_exact1_dispatch_a162_q4096_k13_v1.json', 'source_task': 'weave-evolve-knn-build-7bef'}
Q128_DUALM_TIMING_ROWS = {RAG_Q128_M100000_K32: {'seed_id': SEED_Q128_DUALM_ID, 'kernel_ms': 0.221122, 'flashlib_ms': 0.28253, 'ratio_vs_flashlib': 1.2777109468980925, 'tflops': 14.818968714103525, 'timing_backend': 'cupti', 'source_payload': 'artifacts/weave_evolve/knn_build_a162_rag_stream_q128_dualm_round170/q128_dualm_a162_cupti.json', 'source_task': 'weave-evolve-knn-build-5698'}, RAG_Q128_M131071_K32: {'seed_id': SEED_Q128_DUALM_ID, 'kernel_ms': 0.265059, 'flashlib_ms': 0.340835, 'ratio_vs_flashlib': 1.285883520272845, 'tflops': 16.203692491105755, 'timing_backend': 'cupti', 'source_payload': 'artifacts/weave_evolve/knn_build_a162_rag_stream_q128_dualm_round170/q128_dualm_a162_cupti.json', 'source_task': 'weave-evolve-knn-build-5698'}}
Q128_S72R2_TIMING_ROWS = {RAG_Q128_M100000_K32: {'seed_id': SEED_Q128_S72R2_ID, 'kernel_ms': 0.216995, 'flashlib_ms': 0.282532, 'ratio_vs_flashlib': 1.3020207838890299, 'tflops': 15.100808774395723, 'timing_backend': 'cupti', 'source_payload': 'artifacts/weave_evolve/knn_build_a162_rag_stream_q128_s72r2_round170/a162_rag_stream_q128_s72r2_exact2.json', 'source_task': 'weave-evolve-knn-build-681b'}, RAG_Q128_M131071_K32: {'seed_id': SEED_Q128_S72R2_ID, 'kernel_ms': 0.269891, 'flashlib_ms': 0.339204, 'ratio_vs_flashlib': 1.2568184933917768, 'tflops': 15.913589293455507, 'timing_backend': 'cupti', 'source_payload': 'artifacts/weave_evolve/knn_build_a162_rag_stream_q128_s72r2_round170/a162_rag_stream_q128_s72r2_exact2.json', 'source_task': 'weave-evolve-knn-build-681b'}}
Q128_K10_WARPMERGE_TIMING = {'seed_id': SEED_Q128_K10_WARPMERGE_ID, 'kernel_ms': 0.109633, 'baseline_ms': 0.115521, 'flashlib_ms': 0.13389, 'ratio_vs_flashlib': 1.221256373537165, 'speedup_vs_direct_split72': 1.053706456997437, 'tflops': 29.888810850747493, 'baseline_tflops': 28.36540542412202, 'timing_backend': 'cupti', 'source_payload': 'artifacts/weave_evolve/knn_build_df0f_d128_rag_q128_k10_s74_warpmerge_round173/df0f_q128_k10_s74_warpmerge_v1.json', 'source_task': 'weave-evolve-knn-build-4ce8'}
Q128_K10_ROWLD_1BED_TIMING = {'seed_id': SEED_Q128_K10_ROWLD_1BED_ID, 'kernel_ms': 0.081311, 'baseline_34da_ms': 0.109184, 'flashlib_ms': 0.134144, 'ratio_vs_flashlib': 1.6497644845100912, 'speedup_vs_34da': 1.3427949477930416, 'tflops': 40.2995904613152, 'timing_backend': 'cupti', 'source_payload': 'artifacts/weave_evolve/knn_build_1bed_rag_stream_k10_q128_rowld/rag_stream_k10_q128_s74_rowld_1bed_v1.json', 'source_task': 'weave-evolve-knn-build-41bb'}
Q16_M250_TIMING = {'seed_id': SEED_Q16_M250_ID, 'kernel_ms': 0.160128, 'flashlib_ms': 0.19824, 'ratio_vs_flashlib': 1.2380095923261392, 'tflops': 6.394884092725819, 'timing_backend': 'cupti', 'source_payload': 'artifacts/weave_evolve/knn_build_df0f_d128_rag_q16m250_s288_round170/df0f_q16m250_exact1_dispatch_df0f_q16m250_s288_v1.json', 'source_task': 'weave-evolve-knn-build-cb99'}
ALT_F1D9_D768_TIMING = {'seed_id': 'common_d_56f3_build_highd_v1', 'kernel_ms': 0.036609, 'flashlib_ms': 0.108768, 'ratio_vs_flashlib': 2.9710726870441695, 'tflops': 43.99499401786446, 'timing_backend': 'cupti', 'source_payload': 'design_doc/active/weave_evolve_knn_build_round_161_56f3.md'}
HISTORICAL_BASELINE_ROWS = {BUILD_D256: {'kernel_ms': 0.185761, 'ratio_vs_flashlib': 0.3791538589908538}, BUILD_D768: {'kernel_ms': 3.461614, 'ratio_vs_flashlib': 0.03124582925768153}, BUILD_D1024: {'kernel_ms': 1.20314, 'ratio_vs_flashlib': 0.10178699070764831}, BUILD_D4096: {'kernel_ms': 4.6607225, 'ratio_vs_flashlib': 0.07148784335475883}, D768_SEARCH: {'kernel_ms': 14.019321999999999, 'ratio_vs_flashlib': 0.011618250868337286}, D768_RAG: {'kernel_ms': 0.097568, 'ratio_vs_flashlib': 1.762555346015087}, RAG_D1024: {'kernel_ms': 26.347535, 'ratio_vs_flashlib': 0.0056949919603484726}, RAG_D4096: {'kernel_ms': 69.00885, 'ratio_vs_flashlib': 0.0044159408539629335}, SEARCH_D256: {'kernel_ms': 3.645093, 'ratio_vs_flashlib': 0.12795421131916251}, RAG_D64: {'kernel_ms': 5.0076345, 'ratio_vs_flashlib': 0.013247172891711644}, RAG_D256: {'kernel_ms': 4.999218, 'ratio_vs_flashlib': 0.0159963018216049}}
CANDIDATE_DISPATCHERS = ({'id': CANDIDATE_BASE, 'entrypoint': BASE_ENTRYPOINT, 'consumed_seeds': (), 'guard_plan': ('current v11 common-D fallback dispatcher', 'coverage-only high-D generic fallback rows'), 'expected_shape_wins': (), 'fallback': BASE_ENTRYPOINT, 'rejected_reason': 'baseline only'}, {'id': CANDIDATE_A4EC_BASE, 'entrypoint': A4EC_BASE_ENTRYPOINT, 'consumed_seeds': ('common_d256_q1024_56f3_v1', 'common_d768_build_eeff_m64split_v1', 'common_d_56f3_build_highd_v1', 'non128_frontier_4be7_d768fused_v1'), 'guard_plan': ('a4ec D256 exact build guard', '34d8 D768 exact build guard', 'f1d9 D1024/D4096 exact build guards', '4be7 D768 RAG exact guard', 'then base common-D dispatcher'), 'expected_shape_wins': (BUILD_D256, BUILD_D768, BUILD_D1024, BUILD_D4096, D768_RAG), 'fallback': BASE_ENTRYPOINT, 'rejected_reason': 'same-branch pre-replay baseline; lacks cda9 D768 search and uses slower 165c D256 build seed'}, {'id': CANDIDATE_F1D9_BUILD, 'entrypoint': ''.join([format(MODULE, ''), ':launch_from_contract_inputs_f1d9_build_policy']), 'consumed_seeds': ('common_d_56f3_build_d256_q1024_v1', 'common_d_eeff_search_d768_v1', 'common_d_56f3_build_highd_v1', 'non128_frontier_4be7_d768fused_v1'), 'guard_plan': ('6164 D256 exact build guard', 'cda9 D768 search exact guard', 'f1d9 D768/D1024/D4096 exact build guards', '4be7 D768 RAG exact guard', 'then base common-D dispatcher'), 'expected_shape_wins': (BUILD_D256, D768_SEARCH, BUILD_D768, BUILD_D1024, BUILD_D4096, D768_RAG), 'fallback': BASE_ENTRYPOINT, 'rejected_reason': 'D768 build loses historical exact-shape seed delta to 34d8'}, {'id': CANDIDATE_FA04_BASE, 'entrypoint': FA04_BASE_ENTRYPOINT, 'consumed_seeds': ('common_d_56f3_build_d256_q1024_v1', 'common_d_eeff_search_d768_v1', 'common_d768_build_eeff_m64split_v1', 'common_d_56f3_build_highd_v1', 'non128_frontier_4be7_d768fused_v1'), 'guard_plan': ('6164 D256 exact build guard', '34d8 D768 exact build guard', 'f1d9 D1024/D4096 exact build guards', 'cda9 D768 rectangular search exact guard', '4be7 D768 RAG exact guard', 'then base common-D dispatcher'), 'expected_shape_wins': (BUILD_D256, BUILD_D768, BUILD_D1024, BUILD_D4096, D768_SEARCH, D768_RAG), 'fallback': BASE_ENTRYPOINT, 'rejected_reason': 'same-session baseline for the 4cf7 high-D RAG consumption; lacks D1024/D4096 RAG guards'}, {'id': CANDIDATE_F328_BASE, 'entrypoint': F328_BASE_ENTRYPOINT, 'consumed_seeds': ('common_d_56f3_build_d256_q1024_v1', 'common_d_eeff_search_d768_v1', 'common_d768_build_eeff_m64split_v1', 'common_d_56f3_build_highd_v1', 'non128_frontier_4be7_d768fused_v1', 'common_d_5e7f_rag_highd_v1'), 'guard_plan': ('6164 D256 exact build guard', '34d8 D768 exact build guard', 'f1d9 D1024/D4096 exact build guards', 'cda9 D768 rectangular search exact guard', '4be7 D768 RAG exact guard', '4cf7 D1024/D4096 high-D RAG exact guards', 'then base common-D dispatcher'), 'expected_shape_wins': (BUILD_D256, BUILD_D768, BUILD_D1024, BUILD_D4096, D768_SEARCH, D768_RAG, RAG_D1024, RAG_D4096), 'fallback': BASE_ENTRYPOINT, 'rejected_reason': 'same-session f328 baseline for the 8dbc+ba22 dispatcher synthesis; lacks D256 search and D64/D256 RAG guards'}, {'id': CANDIDATE_MIXED, 'entrypoint': MIXED_BASE_ENTRYPOINT, 'consumed_seeds': ('common_d_56f3_build_d256_q1024_v1', 'common_d_eeff_search_d768_v1', 'common_d768_build_eeff_m64split_v1', 'common_d_56f3_build_highd_v1', 'non128_frontier_4be7_d768fused_v1', 'common_d_5e7f_rag_highd_v1', 'common_d_5e7f_search_d256_v1', 'common_d_5e7f_rag_d64_d256_v1'), 'guard_plan': ('6164 D256 exact build guard', '34d8 D768 exact build guard', 'f1d9 D1024/D4096 exact build guards', 'cda9 D768 rectangular search exact guard', '4be7 D768 RAG exact guard', '4cf7 D1024/D4096 high-D RAG exact guards', '8dbc D256 rectangular search exact guard', 'ba22 D256 RAG exact guard', 'D64 RAG remains on the base route and open for bucket-kernel repair', 'then base common-D dispatcher'), 'expected_shape_wins': MIXED_CONSUMED_SEED_SHAPES, 'fallback': BASE_ENTRYPOINT, 'rejected_reason': 'same-session 1438 baseline for the 0474 D64 RAG repair consumption; lacks D64 RAG repair guard'}, {'id': CANDIDATE_D64_REPAIR, 'entrypoint': D64_REPAIR_BASE_ENTRYPOINT, 'consumed_seeds': ('common_d_56f3_build_d256_q1024_v1', 'common_d_eeff_search_d768_v1', 'common_d768_build_eeff_m64split_v1', 'common_d_56f3_build_highd_v1', 'non128_frontier_4be7_d768fused_v1', 'common_d_5e7f_rag_highd_v1', 'common_d_5e7f_search_d256_v1', 'common_d_5e7f_rag_d64_d256_v1', 'common_d_5e7f_rag_d64_repair_v1'), 'guard_plan': ('6164 D256 exact build guard', '34d8 D768 exact build guard', 'f1d9 D1024/D4096 exact build guards', 'cda9 D768 rectangular search exact guard', '4be7 D768 RAG exact guard', '4cf7 D1024/D4096 high-D RAG exact guards', '0474 D64 RAG repair exact guard', '8dbc D256 rectangular search exact guard', 'ba22 D256 RAG exact guard', 'then base common-D dispatcher'), 'expected_shape_wins': D64_M128_CONSUMED_SEED_SHAPES, 'fallback': BASE_ENTRYPOINT, 'rejected_reason': 'same-session 2ccc baseline for the 631e M128 D64 backfill consumption'}, {'id': CANDIDATE_D64_M128, 'entrypoint': D64_M128_BASE_ENTRYPOINT, 'consumed_seeds': ('common_d_56f3_build_d256_q1024_v1', 'common_d_eeff_search_d768_v1', 'common_d768_build_eeff_m64split_v1', 'common_d_56f3_build_highd_v1', 'non128_frontier_4be7_d768fused_v1', 'common_d_5e7f_rag_highd_v1', 'common_d_5e7f_search_d256_v1', 'common_d_5e7f_rag_d64_d256_v1', 'common_d_1438_rag_d64_m128_v1'), 'guard_plan': ('6164 D256 exact build guard', '34d8 D768 exact build guard', 'f1d9 D1024/D4096 exact build guards', 'cda9 D768 rectangular search exact guard', '4be7 D768 RAG exact guard', '4cf7 D1024/D4096 high-D RAG exact guards', '631e D64 RAG M128 exact guard', '8dbc D256 rectangular search exact guard', 'ba22 D256 RAG exact guard', 'then base common-D dispatcher'), 'expected_shape_wins': D64_M128_CONSUMED_SEED_SHAPES, 'fallback': BASE_ENTRYPOINT, 'rejected_reason': 'same-session baseline for the c271 D64 Q4096 consumption; lacks the c271 exact guard'}, {'id': CANDIDATE_D64_Q4096_C271, 'entrypoint': D64_Q4096_C271_BASE_ENTRYPOINT, 'consumed_seeds': ('common_d_56f3_build_d256_q1024_v1', 'common_d_eeff_search_d768_v1', 'common_d768_build_eeff_m64split_v1', 'common_d_56f3_build_highd_v1', 'non128_frontier_4be7_d768fused_v1', 'common_d_5e7f_rag_highd_v1', 'common_d_5e7f_search_d256_v1', 'common_d_5e7f_rag_d64_d256_v1', 'common_d_1438_rag_d64_m128_v1', 'd64_q4096_c271_twostage_v1'), 'guard_plan': ('6164 D256 exact build guard', '34d8 D768 exact build guard', 'f1d9 D1024/D4096 exact build guards', 'cda9 D768 rectangular search exact guard', '4be7 D768 RAG exact guard', '4cf7 D1024/D4096 high-D RAG exact guards', '631e D64 RAG M128 exact guard', '8dbc D256 rectangular search exact guard', 'ba22 D256 RAG exact guard', 'c271 D64 Q4096 build exact guard', 'then base common-D dispatcher'), 'expected_shape_wins': C271_CONSUMED_SEED_SHAPES, 'fallback': BASE_ENTRYPOINT, 'rejected_reason': 'same-session current-trunk baseline for the c271 plus K13/K48 replay; lacks K13/K48 guards'}, {'id': CANDIDATE_C271_7DC5_K13_K48, 'entrypoint': C271_7DC5_BASE_ENTRYPOINT, 'consumed_seeds': ('common_d_56f3_build_d256_q1024_v1', 'common_d_eeff_search_d768_v1', 'common_d768_build_eeff_m64split_v1', 'common_d_56f3_build_highd_v1', 'non128_frontier_4be7_d768fused_v1', 'common_d_5e7f_rag_highd_v1', 'common_d_5e7f_search_d256_v1', 'common_d_5e7f_rag_d64_d256_v1', 'common_d_1438_rag_d64_m128_v1', 'd64_q4096_c271_twostage_v1', 'knn_build_k13_k48_floor_repair_7dc5_v1'), 'guard_plan': ('6164 D256 exact build guard', '34d8 D768 exact build guard', 'f1d9 D1024/D4096 exact build guards', 'cda9 D768 rectangular search exact guard', '4be7 D768 RAG exact guard', '4cf7 D1024/D4096 high-D RAG exact guards', '631e D64 RAG M128 exact guard', '8dbc D256 rectangular search exact guard', 'ba22 D256 RAG exact guard', 'c271 D64 Q4096 build exact guard', '7dc5 Q4096 K13 exact build guard', '7dc5 Q2048/Q4096 K48 exact build guards', 'then base common-D dispatcher'), 'expected_shape_wins': C271_7DC5_CONSUMED_SEED_SHAPES, 'fallback': BASE_ENTRYPOINT, 'rejected_reason': 'same-session baseline for the merged Q128/Q16 floor-clearing synthesis; lacks Q128/Q16 guards'}, {'id': CANDIDATE_2498_BASELINE, 'entrypoint': BASE_2498_ENTRYPOINT, 'consumed_seeds': ('common_d_56f3_build_d256_q1024_v1', 'common_d_eeff_search_d768_v1', 'common_d768_build_eeff_m64split_v1', 'common_d_56f3_build_highd_v1', 'non128_frontier_4be7_d768fused_v1', 'common_d_5e7f_rag_highd_v1', 'common_d_5e7f_search_d256_v1', 'common_d_5e7f_rag_d64_d256_v1', 'common_d_1438_rag_d64_m128_v1', 'd64_q4096_c271_twostage_v1', 'knn_build_k13_k48_floor_repair_7dc5_v1', SEED_Q128_S72R2_ID, SEED_Q128_DUALM_ID, SEED_Q16_M250_ID, SEED_LARGE_SQUARE_K32_EFE4_ID), 'guard_plan': ('6164 D256 exact build guard', '34d8 D768 exact build guard', 'f1d9 D1024/D4096 exact build guards', 'cda9 D768 rectangular search exact guard', '4be7 D768 RAG exact guard', '4cf7 D1024/D4096 high-D RAG exact guards', '631e D64 RAG M128 exact guard', '8dbc D256 rectangular search exact guard', 'ba22 D256 RAG exact guard', 'c271 D64 Q4096 build exact guard', '7dc5 Q4096 K13 exact build guard', '7dc5 Q2048/Q4096 K48 exact build guards', '681b Q128 K32 exact guard for M=100000', '5698 Q128 K32 exact guard for M=131071', 'cb99 Q16/M250000 K32 exact guard', 'fac0 EFE4 exact large-square D128 K32 guard', 'then base common-D dispatcher'), 'expected_shape_wins': PRE_34DA_CONSUMED_SEED_SHAPES, 'fallback': BASE_ENTRYPOINT, 'rejected_reason': 'same-session 2498 baseline for the 34da RAG stream K10 consumption; lacks 34da guard'}, {'id': CANDIDATE_PRE_Q1_M524_N128, 'entrypoint': PRE_Q1_M524_N128_BASE_ENTRYPOINT, 'consumed_seeds': ('common_d_56f3_build_d256_q1024_v1', 'common_d_eeff_search_d768_v1', 'common_d768_build_eeff_m64split_v1', 'common_d_56f3_build_highd_v1', 'non128_frontier_4be7_d768fused_v1', 'common_d_5e7f_rag_highd_v1', 'common_d_5e7f_search_d256_v1', 'common_d_5e7f_rag_d64_d256_v1', 'common_d_1438_rag_d64_m128_v1', 'd64_q4096_c271_twostage_v1', 'knn_build_k13_k48_floor_repair_7dc5_v1', SEED_Q128_S72R2_ID, SEED_Q128_DUALM_ID, SEED_Q16_M250_ID, SEED_LARGE_SQUARE_K32_EFE4_ID, SEED_RAG_STREAM_K10_34DA_ID), 'guard_plan': ('6164 D256 exact build guard', '34d8 D768 exact build guard', 'f1d9 D1024/D4096 exact build guards', 'cda9 D768 rectangular search exact guard', '4be7 D768 RAG exact guard', '4cf7 D1024/D4096 high-D RAG exact guards', '631e D64 RAG M128 exact guard', '8dbc D256 rectangular search exact guard', 'ba22 D256 RAG exact guard', 'c271 D64 Q4096 build exact guard', '7dc5 Q4096 K13 exact build guard', '7dc5 Q2048/Q4096 K48 exact build guards', '681b Q128 K32 exact guard for M=100000', '5698 Q128 K32 exact guard for M=131071', 'cb99 Q16/M250000 K32 exact guard', 'fac0 EFE4 exact large-square D128 K32 guard', '34da Q128/M100000/K10 RAG stream warp-row merge exact guard', 'then base common-D dispatcher'), 'expected_shape_wins': (*PRE_34DA_CONSUMED_SEED_SHAPES, RAG_STREAM_K10), 'fallback': BASE_ENTRYPOINT, 'rejected_reason': 'same-session ea43 baseline for Q1/M524287 N128 seed consumption; lacks 32b9 guard'}, {'id': CANDIDATE_PRE_Q32_EXACT, 'entrypoint': PRE_Q32_EXACT_BASE_ENTRYPOINT, 'consumed_seeds': ('common_d_56f3_build_d256_q1024_v1', 'common_d_eeff_search_d768_v1', 'common_d768_build_eeff_m64split_v1', 'common_d_56f3_build_highd_v1', 'non128_frontier_4be7_d768fused_v1', 'common_d_5e7f_rag_highd_v1', 'common_d_5e7f_search_d256_v1', 'common_d_5e7f_rag_d64_d256_v1', 'common_d_1438_rag_d64_m128_v1', 'd64_q4096_c271_twostage_v1', 'knn_build_k13_k48_floor_repair_7dc5_v1', SEED_Q128_S72R2_ID, SEED_Q128_DUALM_ID, SEED_Q16_M250_ID, SEED_LARGE_SQUARE_K32_EFE4_ID, SEED_RAG_STREAM_K10_34DA_ID, SEED_Q1_M524_N128_ID), 'guard_plan': ('6164 D256 exact build guard', '34d8 D768 exact build guard', 'f1d9 D1024/D4096 exact build guards', 'cda9 D768 rectangular search exact guard', '4be7 D768 RAG exact guard', '4cf7 D1024/D4096 high-D RAG exact guards', '631e D64 RAG M128 exact guard', '8dbc D256 rectangular search exact guard', 'ba22 D256 RAG exact guard', 'c271 D64 Q4096 build exact guard', '7dc5 Q4096 K13 exact build guard', '7dc5 Q2048/Q4096 K48 exact build guards', '681b Q128 K32 exact guard for M=100000', '5698 Q128 K32 exact guard for M=131071', 'cb99 Q16/M250000 K32 exact guard', 'fac0 EFE4 exact large-square D128 K32 guard', '34da Q128/M100000/K10 RAG stream warp-row merge exact guard', '32b9 Q1/M524287/D128/K10 M64/N128 exact guard', 'then base common-D dispatcher'), 'expected_shape_wins': (*PRE_34DA_CONSUMED_SEED_SHAPES, RAG_STREAM_K10, RAG_Q1_M524287_K10), 'fallback': BASE_ENTRYPOINT, 'rejected_reason': 'same-session baseline for Q32 exact guard consumption; lacks 12ac f653 q32rowld2exact guard'}, {'id': CANDIDATE_PRE_Q32TAIL, 'entrypoint': PRE_Q32TAIL_BASE_ENTRYPOINT, 'consumed_seeds': ('common_d_56f3_build_d256_q1024_v1', 'common_d_eeff_search_d768_v1', 'common_d768_build_eeff_m64split_v1', 'common_d_56f3_build_highd_v1', 'non128_frontier_4be7_d768fused_v1', 'common_d_5e7f_rag_highd_v1', 'common_d_5e7f_search_d256_v1', 'common_d_5e7f_rag_d64_d256_v1', 'common_d_1438_rag_d64_m128_v1', 'd64_q4096_c271_twostage_v1', 'knn_build_k13_k48_floor_repair_7dc5_v1', SEED_Q128_S72R2_ID, SEED_Q128_DUALM_ID, SEED_Q16_M250_ID, SEED_LARGE_SQUARE_K32_EFE4_ID, SEED_RAG_STREAM_K10_34DA_ID, SEED_Q1_M524_N128_ID, SEED_Q32_EXACT_ID), 'guard_plan': ('6164 D256 exact build guard', '34d8 D768 exact build guard', 'f1d9 D1024/D4096 exact build guards', 'cda9 D768 rectangular search exact guard', '4be7 D768 RAG exact guard', '4cf7 D1024/D4096 high-D RAG exact guards', '631e D64 RAG M128 exact guard', '8dbc D256 rectangular search exact guard', 'ba22 D256 RAG exact guard', 'c271 D64 Q4096 build exact guard', '7dc5 Q4096 K13 exact build guard', '7dc5 Q2048/Q4096 K48 exact build guards', '681b Q128 K32 exact guard for M=100000', '5698 Q128 K32 exact guard for M=131071', 'cb99 Q16/M250000 K32 exact guard', 'fac0 EFE4 exact large-square D128 K32 guard', '34da Q128/M100000/K10 RAG stream warp-row merge exact guard', '32b9 Q1/M524287/D128/K10 M64/N128 exact guard', '12ac Q32/M100000/D128/K32 f653 rowld2 exact-stage1 guard', 'then base common-D dispatcher'), 'expected_shape_wins': (*PRE_34DA_CONSUMED_SEED_SHAPES, RAG_STREAM_K10, RAG_Q1_M524287_K10, RAG_Q32_M100000_K32), 'fallback': BASE_ENTRYPOINT, 'rejected_reason': 'same-session baseline for Q32-tail consumption; lacks 6866 Q32 M-tail guard'}, {'id': CANDIDATE_PRE_EXPANDED_K32_Q48, 'entrypoint': PRE_Q32TAIL_BASE_ENTRYPOINT, 'consumed_seeds': ('common_d_56f3_build_d256_q1024_v1', 'common_d_eeff_search_d768_v1', 'common_d768_build_eeff_m64split_v1', 'common_d_56f3_build_highd_v1', 'non128_frontier_4be7_d768fused_v1', 'common_d_5e7f_rag_highd_v1', 'common_d_5e7f_search_d256_v1', 'common_d_5e7f_rag_d64_d256_v1', 'common_d_1438_rag_d64_m128_v1', 'd64_q4096_c271_twostage_v1', 'knn_build_k13_k48_floor_repair_7dc5_v1', SEED_Q128_S72R2_ID, SEED_Q128_DUALM_ID, SEED_Q16_M250_ID, SEED_LARGE_SQUARE_K32_EFE4_ID, SEED_RAG_STREAM_K10_34DA_ID, SEED_Q1_M524_N128_ID, SEED_Q32_EXACT_ID, SEED_Q32TAIL_ID), 'guard_plan': ('6164 D256 exact build guard', '34d8 D768 exact build guard', 'f1d9 D1024/D4096 exact build guards', 'cda9 D768 rectangular search exact guard', '4be7 D768 RAG exact guard', '4cf7 D1024/D4096 high-D RAG exact guards', '631e D64 RAG M128 exact guard', '8dbc D256 rectangular search exact guard', 'ba22 D256 RAG exact guard', 'c271 D64 Q4096 build exact guard', '7dc5 Q4096 K13 exact build guard', '7dc5 Q2048/Q4096 K48 exact build guards', '681b Q128 K32 exact guard for M=100000', '5698 Q128 K32 exact guard for M=131071', 'cb99 Q16/M250000 K32 exact guard', 'fac0 EFE4 exact large-square D128 K32 guard', '34da Q128/M100000/K10 RAG stream warp-row merge exact guard', '32b9 Q1/M524287/D128/K10 M64/N128 exact guard', '12ac Q32/M100000/D128/K32 f653 rowld2 exact-stage1 guard', '6866 Q32/M99999 and Q32/M100001 D128/K32 tail guard', 'then base common-D dispatcher'), 'expected_shape_wins': PRE_EXPANDED_K32_Q48_CONSUMED_SEED_SHAPES, 'fallback': BASE_ENTRYPOINT, 'rejected_reason': 'same-session baseline for expanded Q31/Q33/Q40/Q48 consumption; lacks those exact expanded guards'}, {'id': CANDIDATE_PRE_LOWK_C3D2, 'entrypoint': PRE_LOWK_C3D2_BASE_ENTRYPOINT, 'consumed_seeds': ('common_d_56f3_build_d256_q1024_v1', 'common_d_eeff_search_d768_v1', 'common_d768_build_eeff_m64split_v1', 'common_d_56f3_build_highd_v1', 'non128_frontier_4be7_d768fused_v1', 'common_d_5e7f_rag_highd_v1', 'common_d_5e7f_search_d256_v1', 'common_d_5e7f_rag_d64_d256_v1', 'common_d_1438_rag_d64_m128_v1', 'd64_q4096_c271_twostage_v1', 'knn_build_k13_k48_floor_repair_7dc5_v1', SEED_Q128_S72R2_ID, SEED_Q128_DUALM_ID, SEED_Q16_M250_ID, SEED_LARGE_SQUARE_K32_EFE4_ID, SEED_RAG_STREAM_K10_34DA_ID, SEED_Q1_M524_N128_ID, SEED_Q32_EXACT_ID, SEED_Q32TAIL_ID, SEED_Q31TAIL_V2_ID, SEED_Q33TILE_ID, SEED_Q48_K12_ID), 'guard_plan': ('6164 D256 exact build guard', '34d8 D768 exact build guard', 'f1d9 D1024/D4096 exact build guards', 'cda9 D768 rectangular search exact guard', '4be7 D768 RAG exact guard', '4cf7 D1024/D4096 high-D RAG exact guards', '631e D64 RAG M128 exact guard', '8dbc D256 rectangular search exact guard', 'ba22 D256 RAG exact guard', 'c271 D64 Q4096 build exact guard', '7dc5 Q4096 K13 exact guard', '7dc5 Q2048/Q4096 K48 exact guards', '681b Q128 K32 exact guard for M=100000', '5698 Q128 K32 exact guard for M=131071', 'cb99 Q16/M250000 K32 exact guard', 'fac0 EFE4 exact large-square D128 K32 guard', '34da Q128/M100000/K10 RAG stream warp-row merge exact guard', '32b9 Q1/M524287/D128/K10 M64/N128 exact guard', '12ac Q32/M100000/D128/K32 f653 rowld2 exact-stage1 guard', '6866 Q32/M99999 and Q32/M100001 D128/K32 tail guard', 'c3d2 0cb5 Q31/M100000/D128/K32 exact guard', 'c3d2 c489 Q33/Q40 M100000/D128/K32 exact guard', '2f22 Q48/M75000/D128/K12 exact guard', 'then base common-D dispatcher'), 'expected_shape_wins': PRE_LOWK_C3D2_CONSUMED_SEED_SHAPES, 'fallback': BASE_ENTRYPOINT, 'rejected_reason': 'same-session baseline for low-K c3d2 consumption; lacks K20/K31 exact guards'}, {'id': CANDIDATE_SELECTED_SYNTHESIS, 'entrypoint': ROUTE_ENTRYPOINT, 'consumed_seeds': ('common_d_56f3_build_d256_q1024_v1', 'common_d_eeff_search_d768_v1', 'common_d768_build_eeff_m64split_v1', 'common_d_56f3_build_highd_v1', 'non128_frontier_4be7_d768fused_v1', 'common_d_5e7f_rag_highd_v1', 'common_d_5e7f_search_d256_v1', 'common_d_5e7f_rag_d64_d256_v1', 'common_d_1438_rag_d64_m128_v1', 'd64_q4096_c271_twostage_v1', 'knn_build_k13_k48_floor_repair_7dc5_v1', SEED_Q128_S72R2_ID, SEED_Q128_DUALM_ID, SEED_Q16_M250_ID, SEED_LARGE_SQUARE_K32_EFE4_ID, SEED_RAG_STREAM_K10_34DA_ID, SEED_Q1_M524_N128_ID, SEED_Q32_EXACT_ID, SEED_Q32TAIL_ID, SEED_Q31TAIL_V2_ID, SEED_Q33TILE_ID, SEED_Q48_K12_ID, SEED_Q32_LOWK_C3D2_ID, SEED_Q32_K31_C3D2_ID, SEED_D64_TAIL_017A_ID, SEED_D256_Q4_E2DF_ID), 'guard_plan': ('6164 D256 exact build guard', '34d8 D768 exact build guard', 'f1d9 D1024/D4096 exact build guards', 'cda9 D768 rectangular search exact guard', '4be7 D768 RAG exact guard', '4cf7 D1024/D4096 high-D RAG exact guards', '631e D64 RAG M128 exact guard', '8dbc D256 rectangular search exact guard', 'ba22 D256 RAG exact guard', 'c271 D64 Q4096 build exact guard', '7dc5 Q4096 K13 exact guard', '7dc5 Q2048/Q4096 K48 exact guards', '681b Q128 K32 exact guard for M=100000', '5698 Q128 K32 exact guard for M=131071', 'cb99 Q16/M250000 K32 exact guard', 'fac0 EFE4 exact large-square D128 K32 guard', '34da Q128/M100000/K10 RAG stream warp-row merge exact guard', '32b9 Q1/M524287/D128/K10 M64/N128 exact guard', '12ac Q32/M100000/D128/K32 f653 rowld2 exact-stage1 guard', '6866 Q32/M99999 and Q32/M100001 D128/K32 tail guard', 'c3d2 0cb5 Q31/M100000/D128/K32 exact guard', 'c3d2 c489 Q33/Q40 M100000/D128/K32 exact guard', '2f22 Q48/M75000/D128/K12 exact guard', '4cc4 c3d2 Q32/M100000/D128 low-K exact guard for K20', 'eaf7 c3d2 Q32/M100000/D128/K31 exact guard', '017a v12 D64 Q1/M262143 and Q4/M100000 exact tail guards', 'e2df v12 D256 Q4/M100000/K10 exact long-M guard', 'then base common-D dispatcher'), 'expected_shape_wins': PRE_D256_Q128_59FE_CONSUMED_SEED_SHAPES, 'fallback': BASE_ENTRYPOINT, 'rejected_reason': 'same-session baseline for D256 Q128/K10 59fe consumption; lacks the 59fe exact guard'}, {'id': CANDIDATE_D256_Q128_59FE, 'entrypoint': PRE_D256_K32_59FE_BASE_ENTRYPOINT, 'consumed_seeds': ('common_d_56f3_build_d256_q1024_v1', 'common_d_eeff_search_d768_v1', 'common_d768_build_eeff_m64split_v1', 'common_d_56f3_build_highd_v1', 'non128_frontier_4be7_d768fused_v1', 'common_d_5e7f_rag_highd_v1', 'common_d_5e7f_search_d256_v1', 'common_d_5e7f_rag_d64_d256_v1', 'common_d_1438_rag_d64_m128_v1', 'd64_q4096_c271_twostage_v1', 'knn_build_k13_k48_floor_repair_7dc5_v1', SEED_Q128_S72R2_ID, SEED_Q128_DUALM_ID, SEED_Q16_M250_ID, SEED_LARGE_SQUARE_K32_EFE4_ID, SEED_RAG_STREAM_K10_34DA_ID, SEED_Q1_M524_N128_ID, SEED_Q32_EXACT_ID, SEED_Q32TAIL_ID, SEED_Q31TAIL_V2_ID, SEED_Q33TILE_ID, SEED_Q48_K12_ID, SEED_Q32_LOWK_C3D2_ID, SEED_Q32_K31_C3D2_ID, SEED_D64_TAIL_017A_ID, SEED_D256_Q4_E2DF_ID, SEED_D256_Q128_59FE_ID), 'guard_plan': ('4ce7 guard portfolio through D64 tail and D256 Q4 e2df exact guards', '59fe v12 D256 Q128/M100000/K10 exact long-M guard', 'then base common-D dispatcher'), 'expected_shape_wins': PRE_D256_K32_59FE_CONSUMED_SEED_SHAPES, 'fallback': BASE_ENTRYPOINT, 'rejected_reason': 'same-session baseline for D256/K32 9203 consumption; lacks the exact K32 guard'}, {'id': CANDIDATE_D256_K32_59FE, 'entrypoint': PRE_HIGHD_RAG_22E9_BASE_ENTRYPOINT, 'consumed_seeds': ('common_d_56f3_build_d256_q1024_v1', 'common_d_eeff_search_d768_v1', 'common_d768_build_eeff_m64split_v1', 'common_d_56f3_build_highd_v1', 'non128_frontier_4be7_d768fused_v1', 'common_d_5e7f_rag_highd_v1', 'common_d_5e7f_search_d256_v1', 'common_d_5e7f_rag_d64_d256_v1', 'common_d_1438_rag_d64_m128_v1', 'd64_q4096_c271_twostage_v1', 'knn_build_k13_k48_floor_repair_7dc5_v1', SEED_Q128_S72R2_ID, SEED_Q128_DUALM_ID, SEED_Q16_M250_ID, SEED_LARGE_SQUARE_K32_EFE4_ID, SEED_RAG_STREAM_K10_34DA_ID, SEED_Q1_M524_N128_ID, SEED_Q32_EXACT_ID, SEED_Q32TAIL_ID, SEED_Q31TAIL_V2_ID, SEED_Q33TILE_ID, SEED_Q48_K12_ID, SEED_Q32_LOWK_C3D2_ID, SEED_Q32_K31_C3D2_ID, SEED_D64_TAIL_017A_ID, SEED_D256_Q4_E2DF_ID, SEED_D256_Q128_59FE_ID, SEED_D256_K32_59FE_ID), 'guard_plan': ('22e9 guard portfolio through D64 tail, D256 Q4, and D256 Q128/K10 exact guards', '9203 v12 D256 Q8/Q128 M100000 K32 exact long-M guards', 'then base common-D dispatcher'), 'expected_shape_wins': PRE_HIGHD_RAG_22E9_CONSUMED_SEED_SHAPES, 'fallback': BASE_ENTRYPOINT, 'rejected_reason': 'same-session be66 baseline for 7902 high-D RAG consumption; lacks the exact high-D RAG guard'}, {'id': CANDIDATE_HIGHD_RAG_22E9, 'entrypoint': ROUTE_ENTRYPOINT, 'consumed_seeds': ('common_d_56f3_build_d256_q1024_v1', 'common_d_eeff_search_d768_v1', 'common_d768_build_eeff_m64split_v1', 'common_d_56f3_build_highd_v1', 'non128_frontier_4be7_d768fused_v1', 'common_d_5e7f_rag_highd_v1', 'common_d_5e7f_search_d256_v1', 'common_d_5e7f_rag_d64_d256_v1', 'common_d_1438_rag_d64_m128_v1', 'd64_q4096_c271_twostage_v1', 'knn_build_k13_k48_floor_repair_7dc5_v1', SEED_Q128_S72R2_ID, SEED_Q128_DUALM_ID, SEED_Q16_M250_ID, SEED_LARGE_SQUARE_K32_EFE4_ID, SEED_RAG_STREAM_K10_34DA_ID, SEED_Q1_M524_N128_ID, SEED_Q32_EXACT_ID, SEED_Q32TAIL_ID, SEED_Q31TAIL_V2_ID, SEED_Q33TILE_ID, SEED_Q48_K12_ID, SEED_Q32_LOWK_C3D2_ID, SEED_Q32_K31_C3D2_ID, SEED_D64_TAIL_017A_ID, SEED_D256_Q4_E2DF_ID, SEED_D256_Q128_59FE_ID, SEED_D256_K32_59FE_ID, SEED_HIGHD_RAG_22E9_ID), 'guard_plan': ('be66 guard portfolio through D64, D256 Q4/Q128 K10, and D256 K32 exact guards', '7902 exact BF16 high-D RAG guards for D768/Q8/M100000, D1024/Q4/M100000, and D4096/Q1/M65536', 'then base common-D dispatcher'), 'expected_shape_wins': PRE_HIGHD_SEARCH_BE66_CONSUMED_SEED_SHAPES, 'fallback': BASE_ENTRYPOINT, 'rejected_reason': 'same-session baseline for ad73 high-D search consumption; lacks the exact rectangular-search guard'}, {'id': CANDIDATE_HIGHD_SEARCH_BE66, 'entrypoint': PRE_D128_K48_DD2B_BASE_ENTRYPOINT, 'consumed_seeds': ('common_d_56f3_build_d256_q1024_v1', 'common_d_eeff_search_d768_v1', 'common_d768_build_eeff_m64split_v1', 'common_d_56f3_build_highd_v1', 'non128_frontier_4be7_d768fused_v1', 'common_d_5e7f_rag_highd_v1', 'common_d_5e7f_search_d256_v1', 'common_d_5e7f_rag_d64_d256_v1', 'common_d_1438_rag_d64_m128_v1', 'd64_q4096_c271_twostage_v1', 'knn_build_k13_k48_floor_repair_7dc5_v1', SEED_Q128_S72R2_ID, SEED_Q128_DUALM_ID, SEED_Q16_M250_ID, SEED_LARGE_SQUARE_K32_EFE4_ID, SEED_RAG_STREAM_K10_34DA_ID, SEED_Q1_M524_N128_ID, SEED_Q32_EXACT_ID, SEED_Q32TAIL_ID, SEED_Q31TAIL_V2_ID, SEED_Q33TILE_ID, SEED_Q48_K12_ID, SEED_Q32_LOWK_C3D2_ID, SEED_Q32_K31_C3D2_ID, SEED_D64_TAIL_017A_ID, SEED_D256_Q4_E2DF_ID, SEED_D256_Q128_59FE_ID, SEED_D256_K32_59FE_ID, SEED_HIGHD_RAG_22E9_ID, SEED_HIGHD_SEARCH_BE66_ID), 'guard_plan': ('0d3d guard portfolio through D64, D256 Q4/Q128/K32, and high-D RAG exact guards', 'ad73 exact BF16 high-D rectangular-search guards for D1024/Q256/M8192 and D4096/Q128/M4096', 'then base common-D dispatcher'), 'expected_shape_wins': PRE_D128_K48_DD2B_CONSUMED_SEED_SHAPES, 'fallback': BASE_ENTRYPOINT, 'rejected_reason': 'same-session baseline for dd2b D128/Q16/K48 consumption; lacks the exact K48 guard'}, {'id': CANDIDATE_D128_K48_DD2B, 'entrypoint': PRE_RECT_D128_K20_S12WARP4_BASE_ENTRYPOINT, 'consumed_seeds': ('common_d_56f3_build_d256_q1024_v1', 'common_d_eeff_search_d768_v1', 'common_d768_build_eeff_m64split_v1', 'common_d_56f3_build_highd_v1', 'non128_frontier_4be7_d768fused_v1', 'common_d_5e7f_rag_highd_v1', 'common_d_5e7f_search_d256_v1', 'common_d_5e7f_rag_d64_d256_v1', 'common_d_1438_rag_d64_m128_v1', 'd64_q4096_c271_twostage_v1', 'knn_build_k13_k48_floor_repair_7dc5_v1', SEED_Q128_S72R2_ID, SEED_Q128_DUALM_ID, SEED_Q16_M250_ID, SEED_LARGE_SQUARE_K32_EFE4_ID, SEED_RAG_STREAM_K10_34DA_ID, SEED_Q1_M524_N128_ID, SEED_Q32_EXACT_ID, SEED_Q32TAIL_ID, SEED_Q31TAIL_V2_ID, SEED_Q33TILE_ID, SEED_Q48_K12_ID, SEED_Q32_LOWK_C3D2_ID, SEED_Q32_K31_C3D2_ID, SEED_D64_TAIL_017A_ID, SEED_D256_Q4_E2DF_ID, SEED_D256_Q128_59FE_ID, SEED_D256_K32_59FE_ID, SEED_HIGHD_RAG_22E9_ID, SEED_HIGHD_SEARCH_BE66_ID, SEED_D128_K48_DD2B_ID), 'guard_plan': ('2614 guard portfolio through D64, D256, high-D RAG, and high-D rectangular-search exact guards', 'dd2b exact BF16 non-build B=1 Q=16 M=100000 D=128 K=48 guard', 'then base common-D dispatcher'), 'expected_shape_wins': PRE_RECT_D128_K20_S12WARP4_CONSUMED_SEED_SHAPES, 'fallback': BASE_ENTRYPOINT, 'rejected_reason': 'same-session baseline for 7768 rectangular consumption; lacks the Q1536 D128/K20 split12/warp4 guard'}, {'id': CANDIDATE_RECT_D128_K20_S12WARP4, 'entrypoint': ROUTE_ENTRYPOINT, 'consumed_seeds': ('common_d_56f3_build_d256_q1024_v1', 'common_d_eeff_search_d768_v1', 'common_d768_build_eeff_m64split_v1', 'common_d_56f3_build_highd_v1', 'non128_frontier_4be7_d768fused_v1', 'common_d_5e7f_rag_highd_v1', 'common_d_5e7f_search_d256_v1', 'common_d_5e7f_rag_d64_d256_v1', 'common_d_1438_rag_d64_m128_v1', 'd64_q4096_c271_twostage_v1', 'knn_build_k13_k48_floor_repair_7dc5_v1', SEED_Q128_S72R2_ID, SEED_Q128_DUALM_ID, SEED_Q16_M250_ID, SEED_LARGE_SQUARE_K32_EFE4_ID, SEED_RAG_STREAM_K10_34DA_ID, SEED_Q1_M524_N128_ID, SEED_Q32_EXACT_ID, SEED_Q32TAIL_ID, SEED_Q31TAIL_V2_ID, SEED_Q33TILE_ID, SEED_Q48_K12_ID, SEED_Q32_LOWK_C3D2_ID, SEED_Q32_K31_C3D2_ID, SEED_D64_TAIL_017A_ID, SEED_D256_Q4_E2DF_ID, SEED_D256_Q128_59FE_ID, SEED_D256_K32_59FE_ID, SEED_HIGHD_RAG_22E9_ID, SEED_HIGHD_SEARCH_BE66_ID, SEED_D128_K48_DD2B_ID, SEED_RECT_D128_K20_S12WARP4_ID), 'guard_plan': ('2614 guard portfolio through D64, D256, high-D RAG, high-D rectangular-search, and D128/K48 guards', '7768 exact BF16 non-build B=1 Q=1536 M=65536 D=128 K=20 split12/warp4 guard', 'then base common-D dispatcher'), 'expected_shape_wins': CONSUMED_SEED_SHAPES, 'fallback': BASE_ENTRYPOINT, 'rejected_reason': None})

def _select_contract_shapes(shape_labels):
    labels = tuple(shape_labels)
    selected = []
    remaining = []
    for label in labels:
        if label in EXPANDED_Q32_GUARD_BOUNDARY_8_BY_LABEL:
            selected.append(EXPANDED_Q32_GUARD_BOUNDARY_8_BY_LABEL[label])
        else:
            remaining.append(label)
    if remaining:
        selected.extend(base_current._select_contract_shapes(tuple(remaining)))
    return selected

def _trace_inputs_for_shape(shape: dict[str, Any]) -> dict[str, Any]:
    if str(shape['label']) not in EXPANDED_Q32_GUARD_BOUNDARY_8_BY_LABEL:
        return base_current._trace_inputs_for_shape(shape)
    params = dict(shape.get('params', {}))
    params['label'] = shape['label']
    return params

def _inputs_for_label(label: str) -> dict[str, Any]:
    return _trace_inputs_for_shape(_select_contract_shapes((label,))[0])

def _normalize_route_row(row: dict[str, Any]) -> dict[str, Any]:
    return base_current._normalize_route_row(row)

def _highd_label(inputs: dict[str, Any]) -> str | None:
    return highd_build._target_label_for_inputs(inputs)

def _eligible_d768_rag(inputs: dict[str, Any]) -> bool:
    return d768_rag._target_label_for_inputs(inputs) == D768_RAG

def _eligible_d256_6164(inputs: dict[str, Any]) -> bool:
    return d256_build._target_label_for_inputs(inputs) == BUILD_D256

def _eligible_d768_search(inputs: dict[str, Any]) -> bool:
    return d768_search._target_label_for_inputs(inputs) == D768_SEARCH

def _highd_rag_label(inputs: dict[str, Any]) -> str | None:
    return highd_rag._target_label_for_inputs(inputs)

def _search_d256_label(inputs: dict[str, Any]) -> str | None:
    return search_d256._target_label_for_inputs(inputs)

def _rag_d256_label(inputs: dict[str, Any]) -> str | None:
    label = rag_d64d256._target_label_for_inputs(inputs)
    return label if label == RAG_D256 else None

def _rag_d64_repair_label(inputs: dict[str, Any]) -> str | None:
    return rag_d64_repair._target_label_for_inputs(inputs)

def _rag_d64_m128_label(inputs: dict[str, Any]) -> str | None:
    return rag_d64_m128._target_label_for_inputs(inputs)

def _eligible_d64_q4096_c271(inputs: dict[str, Any]) -> bool:
    return d64_q4096_c271._eligible_exact_q4096_d64(inputs)

def _d64_tail_017a_label(inputs: dict[str, Any]) -> str | None:
    return d64_tail_017a._target_label_for_inputs(inputs)

def _d256_q4_e2df_label(inputs: dict[str, Any]) -> str | None:
    return d256_q4_e2df._target_label_for_inputs(inputs)

def _d256_q128_59fe_label(inputs: dict[str, Any]) -> str | None:
    return d256_q128_59fe._target_label_for_inputs(inputs)

def _d256_k32_59fe_label(inputs: dict[str, Any]) -> str | None:
    return d256_k32_59fe._target_label_for_inputs(inputs)

def _highd_rag_22e9_label(inputs: dict[str, Any]) -> str | None:
    return highd_rag_22e9._target_label_for_inputs(inputs)

def _highd_search_be66_label(inputs: dict[str, Any]) -> str | None:
    return highd_search_be66._target_label_for_inputs(inputs)

def _d128_k48_dd2b_label(inputs: dict[str, Any]) -> str | None:
    return V12_D128_K48_OVER32 if d128_k48_dd2b._eligible_k48(inputs) else None

def _q128_dualm():
    from . import knn_build_rag_stream_k32_q128_dualm_a162_v1 as mod
    return mod

def _q128_s72r2():
    from . import knn_build_rag_stream_k32_q128_s72r2_a162_v1 as mod
    return mod

def _q16_m250():
    from . import knn_build_d128_rag_q16m250_df0f_v1 as mod
    return mod

def _rag_stream_k10_34da():
    from . import knn_build_rag_stream_k10_warpmerge_34da_v1 as mod
    return mod

def _eligible_q1_m524_n128(inputs: dict[str, Any]) -> bool:
    return q1_m524_n128._eligible_q1_m524_n128(inputs)

def _eligible_q32_exact(inputs: dict[str, Any]) -> bool:
    return q32exact._eligible_q32_rowld2exact(inputs)

def _q32tail():
    from . import knn_build_rag_microbucket_k32_0cb5_q31tail_v1 as mod
    return mod

def _eligible_q32tail(inputs: dict[str, Any]) -> bool:
    return bool(_q32tail()._eligible_q32tail_exact(inputs))

def _q32tail143_low():
    from . import knn_build_rag_microbucket_k32_5317_q32tail143low_v1 as mod
    return mod

def _q32tail143_high():
    from . import knn_build_rag_microbucket_k32_314c_q32tail143_v1 as mod
    return mod

def _eligible_q32tail143(inputs: dict[str, Any]) -> bool:
    return bool(_q32tail143_low()._eligible_q32tail143low(inputs) or _q32tail143_high()._eligible_q32tail143(inputs))

def _q32tail143_module_for_inputs(inputs: dict[str, Any]):
    if _q32tail143_low()._eligible_q32tail143low(inputs):
        return _q32tail143_low()
    if _q32tail143_high()._eligible_q32tail143(inputs):
        return _q32tail143_high()
    raise ValueError('Q32 split143 tail route requested for non-target inputs')

def _q32tail143_route_metadata(inputs: dict[str, Any]) -> tuple[str, str, str]:
    if _q32tail143_low()._eligible_q32tail143low(inputs):
        return (SEED_Q32TAIL143_LOW_ID, Q32TAIL143_LOW_ENTRYPOINT, Q32TAIL143_LOW_GUARD_ID)
    if _q32tail143_high()._eligible_q32tail143(inputs):
        return (SEED_Q32TAIL143_HIGH_ID, Q32TAIL143_HIGH_ENTRYPOINT, Q32TAIL143_HIGH_GUARD_ID)
    raise ValueError('Q32 split143 tail metadata requested for non-target inputs')

def _q31tail_v2():
    from . import knn_build_rag_microbucket_k32_0cb5_q31tail_v2 as mod
    return mod

def _eligible_q31tail_v2(inputs: dict[str, Any]) -> bool:
    return bool(_q31tail_v2()._eligible_q31tail_v2(inputs))

def _q33tile():
    from . import knn_build_rag_microbucket_k32_c489_q33tile_v1 as mod
    return mod

def _eligible_q33tile(inputs: dict[str, Any]) -> bool:
    return bool(_q33tile()._eligible_q33tile(inputs))

def _q48_k12():
    from . import knn_build_rag_microbucket_k12_2f22_q48exact_v1 as mod
    return mod

def _eligible_q48_k12(inputs: dict[str, Any]) -> bool:
    return bool(_q48_k12()._eligible_q48_k12(inputs))

def _q32_lowk_c3d2():
    from . import knn_build_rag_microbucket_q32_lowk_c3d2_v1 as mod
    return mod

def _eligible_q32_lowk_c3d2(inputs: dict[str, Any]) -> bool:
    return bool(_q32_lowk_c3d2()._eligible_q32_lowk(inputs))

def _q32_k31_c3d2():
    from . import knn_build_rag_microbucket_q32_k31_c3d2_v1 as mod
    return mod

def _eligible_q32_k31_c3d2(inputs: dict[str, Any]) -> bool:
    return bool(_q32_k31_c3d2()._eligible_q32_k31(inputs))

def _uses_floor_seed_portfolio(portfolio_id: str) -> bool:
    return portfolio_id in FLOOR_SEED_PORTFOLIOS

def _eligible_q128_floor_seed(inputs: dict[str, Any]) -> bool:
    return bool(_q128_s72r2()._eligible_q128_stream_k32(inputs))

def _eligible_q16_floor_seed(inputs: dict[str, Any]) -> bool:
    selected, _label = _q16_m250()._selected_seed(inputs)
    return selected == SEED_Q16_M250_ID

def _eligible_large_square_k32_efe4(inputs: dict[str, Any]) -> bool:
    return large_square_k32_efe4._eligible_q8192_k32(inputs)

def _eligible_q128_k10_warpmerge(inputs: dict[str, Any]) -> bool:
    return q128_k10_warpmerge._eligible_split74_warpmerge(inputs)

def _q128_k10_rowld_1bed():
    from . import knn_build_rag_stream_k10_q128_1bed_rowld_v1 as mod
    return mod

def _eligible_q128_k10_rowld_1bed(inputs: dict[str, Any]) -> bool:
    return _q128_k10_rowld_1bed()._eligible_q128_m100000_k10(inputs)

def _eligible_rag_stream_k10_34da(inputs: dict[str, Any]) -> bool:
    return bool(_rag_stream_k10_34da()._eligible_q128_m100000_k10(inputs))

def _q128_module_for_portfolio(inputs: dict[str, Any], portfolio_id: str):
    if portfolio_id == CANDIDATE_FLOOR_SEEDS_Q128_5698:
        return _q128_dualm()
    if portfolio_id == CANDIDATE_FLOOR_SEEDS_Q128_681B:
        return _q128_s72r2()
    if int(inputs.get('M', -1)) == 131071:
        return _q128_dualm()
    return _q128_s72r2()

def _k13_k48_seed_label(inputs: dict[str, Any]) -> tuple[str | None, str | None]:
    if seed_k13._eligible_midk_q4096(inputs):
        return (SEED_K13_ID, BUILD_K13)
    if seed_k48._eligible_k48(inputs):
        label = str(inputs.get('label') or (BUILD_K48_Q2048 if int(inputs.get('Q', -1)) == 2048 else BUILD_K48_Q4096))
        return (SEED_K48_ID, label)
    return (None, None)

def _route_k13_k48(inputs: dict[str, Any]) -> str:
    selected_seed, _label = _k13_k48_seed_label(inputs)
    if selected_seed == SEED_K13_ID:
        return seed_k13.route_for_contract_inputs(inputs)
    if selected_seed == SEED_K48_ID:
        return seed_k48.route_for_contract_inputs(inputs)
    raise ValueError('K13/K48 route requested for non-target inputs')

def _route_for_policy(inputs: dict[str, Any], *, portfolio_id: str, force_fallback: bool=False) -> str:
    if force_fallback:
        return base_current.route_for_contract_inputs(inputs, force_fallback=True)
    if portfolio_id in PORTFOLIOS_WITH_Q128_K10_ROWLD_1BED:
        if _eligible_q128_k10_rowld_1bed(inputs):
            return _q128_k10_rowld_1bed().route_for_contract_inputs(inputs)
        return _route_for_policy(inputs, portfolio_id=CANDIDATE_RECT_D128_K20_S12WARP4)
    if portfolio_id in PORTFOLIOS_WITH_RECT_D128_K20_S12WARP4:
        if rect_d128_k20_s12warp4._eligible_rect_d128_k20_q1536(inputs):
            return rect_d128_k20_s12warp4.route_for_contract_inputs(inputs)
        return _route_for_policy(inputs, portfolio_id=CANDIDATE_D128_K48_DD2B)
    if portfolio_id in PORTFOLIOS_WITH_Q32_TAIL143 and _eligible_q32tail143(inputs):
        return _q32tail143_module_for_inputs(inputs).route_for_contract_inputs(inputs)
    if portfolio_id in PORTFOLIOS_WITH_D128_K48_DD2B:
        if _d128_k48_dd2b_label(inputs) is not None:
            return d128_k48_dd2b.route_for_contract_inputs(inputs)
        return _route_for_policy(inputs, portfolio_id=CANDIDATE_HIGHD_SEARCH_BE66)
    if portfolio_id in PORTFOLIOS_WITH_HIGHD_SEARCH_BE66:
        if _highd_search_be66_label(inputs) is not None:
            return highd_search_be66.route_for_contract_inputs(inputs)
        return _route_for_policy(inputs, portfolio_id=CANDIDATE_HIGHD_RAG_22E9)
    if portfolio_id in PORTFOLIOS_WITH_HIGHD_RAG_22E9:
        if _highd_rag_22e9_label(inputs) is not None:
            return highd_rag_22e9.route_for_contract_inputs(inputs)
        return _route_for_policy(inputs, portfolio_id=CANDIDATE_D256_K32_59FE)
    if portfolio_id in PORTFOLIOS_WITH_D256_K32_59FE:
        if _d256_k32_59fe_label(inputs) is not None:
            return d256_k32_59fe.route_for_contract_inputs(inputs)
        return _route_for_policy(inputs, portfolio_id=CANDIDATE_D256_Q128_59FE)
    if portfolio_id in PORTFOLIOS_WITH_D256_Q128_59FE:
        if _d256_q128_59fe_label(inputs) is not None:
            return d256_q128_59fe.route_for_contract_inputs(inputs)
        return _route_for_policy(inputs, portfolio_id=CANDIDATE_SELECTED_SYNTHESIS)
    if portfolio_id == CANDIDATE_A4EC_BASE:
        if d256_build_a4ec._eligible_d256_q1024(inputs):
            return d256_build_a4ec.route_for_contract_inputs(inputs)
        if d768_build_fast._eligible_d768_build(inputs):
            return d768_build_fast.route_for_contract_inputs(inputs)
        label = _highd_label(inputs)
        if label is not None:
            return highd_build.route_for_contract_inputs(inputs)
        if _eligible_d768_rag(inputs):
            return d768_rag.route_for_contract_inputs(inputs)
        return base_current.route_for_contract_inputs(inputs)
    if _eligible_d256_6164(inputs):
        return d256_build.route_for_contract_inputs(inputs)
    if portfolio_id != CANDIDATE_F1D9_BUILD and d768_build_fast._eligible_d768_build(inputs):
        return d768_build_fast.route_for_contract_inputs(inputs)
    label = _highd_label(inputs)
    if label is not None:
        return highd_build.route_for_contract_inputs(inputs)
    if _eligible_d768_search(inputs):
        return d768_search.route_for_contract_inputs(inputs)
    if _eligible_d768_rag(inputs):
        return d768_rag.route_for_contract_inputs(inputs)
    if portfolio_id in PORTFOLIOS_WITH_HIGHD_RAG and _highd_rag_label(inputs) is not None:
        return highd_rag.route_for_contract_inputs(inputs)
    if portfolio_id in PORTFOLIOS_WITH_SEARCH_D256 and _search_d256_label(inputs) is not None:
        return search_d256.route_for_contract_inputs(inputs)
    if portfolio_id in PORTFOLIOS_WITH_RAG_D64_M128 and _rag_d64_m128_label(inputs) is not None:
        return rag_d64_m128.route_for_contract_inputs(inputs)
    if portfolio_id == CANDIDATE_D64_REPAIR and _rag_d64_repair_label(inputs) is not None:
        return rag_d64_repair.route_for_contract_inputs(inputs)
    if portfolio_id in PORTFOLIOS_WITH_SEARCH_D256 and _rag_d256_label(inputs) is not None:
        return rag_d64d256.route_for_contract_inputs(inputs)
    if portfolio_id in PORTFOLIOS_WITH_D256_Q4_E2DF and _d256_q4_e2df_label(inputs) is not None:
        return d256_q4_e2df.route_for_contract_inputs(inputs)
    if portfolio_id in PORTFOLIOS_WITH_D64_Q4096_C271 and _eligible_d64_q4096_c271(inputs):
        return d64_q4096_c271.route_name_for_inputs(inputs)
    if portfolio_id in PORTFOLIOS_WITH_D64_TAIL_017A and _d64_tail_017a_label(inputs) is not None:
        return d64_tail_017a.route_for_contract_inputs(inputs)
    if portfolio_id in PORTFOLIOS_WITH_K13_K48 and _k13_k48_seed_label(inputs)[0] is not None:
        return _route_k13_k48(inputs)
    if _uses_floor_seed_portfolio(portfolio_id) and _eligible_q128_floor_seed(inputs):
        return _q128_module_for_portfolio(inputs, portfolio_id).route_for_contract_inputs(inputs)
    if portfolio_id in PORTFOLIOS_WITH_Q128_K10_WARPMERGE and _eligible_q128_k10_warpmerge(inputs):
        return q128_k10_warpmerge.route_for_contract_inputs(inputs)
    if _uses_floor_seed_portfolio(portfolio_id) and _eligible_q16_floor_seed(inputs):
        return _q16_m250().route_for_contract_inputs(inputs)
    if portfolio_id in PORTFOLIOS_WITH_LARGE_SQUARE_K32_EFE4 and _eligible_large_square_k32_efe4(inputs):
        return large_square_k32_efe4.route_for_contract_inputs(inputs)
    if portfolio_id in PORTFOLIOS_WITH_RAG_STREAM_K10_34DA and _eligible_rag_stream_k10_34da(inputs):
        return _rag_stream_k10_34da().route_for_contract_inputs(inputs)
    if portfolio_id in PORTFOLIOS_WITH_Q1_M524_N128 and _eligible_q1_m524_n128(inputs):
        return q1_m524_n128.route_for_contract_inputs(inputs)
    if portfolio_id in PORTFOLIOS_WITH_Q32_EXACT and _eligible_q32_exact(inputs):
        return q32exact.route_for_contract_inputs(inputs)
    if portfolio_id in PORTFOLIOS_WITH_Q32_TAIL and _eligible_q32tail(inputs):
        return _q32tail().route_for_contract_inputs(inputs)
    if portfolio_id in PORTFOLIOS_WITH_EXPANDED_K32_Q31_Q33_Q40 and _eligible_q31tail_v2(inputs):
        return _q31tail_v2().route_for_contract_inputs(inputs)
    if portfolio_id in PORTFOLIOS_WITH_EXPANDED_K32_Q31_Q33_Q40 and _eligible_q33tile(inputs):
        return _q33tile().route_for_contract_inputs(inputs)
    if portfolio_id in PORTFOLIOS_WITH_Q48_K12 and _eligible_q48_k12(inputs):
        return _q48_k12().route_for_contract_inputs(inputs)
    if portfolio_id in PORTFOLIOS_WITH_Q32_K31_C3D2 and _eligible_q32_k31_c3d2(inputs):
        return _q32_k31_c3d2().route_for_contract_inputs(inputs)
    if portfolio_id in PORTFOLIOS_WITH_Q32_LOWK_C3D2 and _eligible_q32_lowk_c3d2(inputs):
        return _q32_lowk_c3d2().route_for_contract_inputs(inputs)
    return base_current.route_for_contract_inputs(inputs)

def route_for_contract_inputs(inputs: dict[str, Any], *, force_fallback: bool=False, portfolio_id: str=CANDIDATE_DEFAULT) -> str:
    return _route_for_policy(inputs, portfolio_id=portfolio_id, force_fallback=force_fallback)

def route_for_contract_inputs_f1d9_build_policy(inputs: dict[str, Any], *, force_fallback: bool=False) -> str:
    return route_for_contract_inputs(inputs, force_fallback=force_fallback, portfolio_id=CANDIDATE_F1D9_BUILD)

def route_for_contract_inputs_a4ec_baseline(inputs: dict[str, Any], *, force_fallback: bool=False) -> str:
    return route_for_contract_inputs(inputs, force_fallback=force_fallback, portfolio_id=CANDIDATE_A4EC_BASE)

def route_for_contract_inputs_fa04_baseline(inputs: dict[str, Any], *, force_fallback: bool=False) -> str:
    return route_for_contract_inputs(inputs, force_fallback=force_fallback, portfolio_id=CANDIDATE_FA04_BASE)

def route_for_contract_inputs_f328_baseline(inputs: dict[str, Any], *, force_fallback: bool=False) -> str:
    return route_for_contract_inputs(inputs, force_fallback=force_fallback, portfolio_id=CANDIDATE_F328_BASE)

def launch_from_contract_inputs(inputs: dict[str, Any], *, force_fallback: bool=False, portfolio_id: str=CANDIDATE_DEFAULT) -> None:
    if force_fallback:
        base_current.launch_from_contract_inputs(inputs, force_fallback=True)
        return
    if portfolio_id in PORTFOLIOS_WITH_Q128_K10_ROWLD_1BED:
        if _eligible_q128_k10_rowld_1bed(inputs):
            _q128_k10_rowld_1bed().launch_from_contract_inputs(inputs)
            return
        launch_from_contract_inputs(inputs, portfolio_id=CANDIDATE_RECT_D128_K20_S12WARP4)
        return
    if portfolio_id in PORTFOLIOS_WITH_RECT_D128_K20_S12WARP4:
        if rect_d128_k20_s12warp4._eligible_rect_d128_k20_q1536(inputs):
            rect_d128_k20_s12warp4.launch_from_contract_inputs(inputs)
            return
        launch_from_contract_inputs(inputs, portfolio_id=CANDIDATE_D128_K48_DD2B)
        return
    if portfolio_id in PORTFOLIOS_WITH_Q32_TAIL143 and _eligible_q32tail143(inputs):
        _q32tail143_module_for_inputs(inputs).launch_from_contract_inputs(inputs)
        return
    if portfolio_id in PORTFOLIOS_WITH_D128_K48_DD2B:
        if _d128_k48_dd2b_label(inputs) is not None:
            d128_k48_dd2b.launch_from_contract_inputs(inputs)
            return
        launch_from_contract_inputs(inputs, portfolio_id=CANDIDATE_HIGHD_SEARCH_BE66)
        return
    if portfolio_id in PORTFOLIOS_WITH_HIGHD_SEARCH_BE66:
        if _highd_search_be66_label(inputs) is not None:
            highd_search_be66.launch_from_contract_inputs(inputs)
            return
        launch_from_contract_inputs(inputs, portfolio_id=CANDIDATE_HIGHD_RAG_22E9)
        return
    if portfolio_id in PORTFOLIOS_WITH_HIGHD_RAG_22E9:
        if _highd_rag_22e9_label(inputs) is not None:
            highd_rag_22e9.launch_from_contract_inputs(inputs)
            return
        launch_from_contract_inputs(inputs, portfolio_id=CANDIDATE_D256_K32_59FE)
        return
    if portfolio_id in PORTFOLIOS_WITH_D256_K32_59FE:
        if _d256_k32_59fe_label(inputs) is not None:
            d256_k32_59fe.launch_from_contract_inputs(inputs)
            return
        launch_from_contract_inputs(inputs, portfolio_id=CANDIDATE_D256_Q128_59FE)
        return
    if portfolio_id in PORTFOLIOS_WITH_D256_Q128_59FE:
        if _d256_q128_59fe_label(inputs) is not None:
            d256_q128_59fe.launch_from_contract_inputs(inputs)
            return
        launch_from_contract_inputs(inputs, portfolio_id=CANDIDATE_SELECTED_SYNTHESIS)
        return
    if portfolio_id == CANDIDATE_A4EC_BASE:
        if d256_build_a4ec._eligible_d256_q1024(inputs):
            d256_build_a4ec.launch_from_contract_inputs(inputs)
            return
        if d768_build_fast._eligible_d768_build(inputs):
            d768_build_fast.launch_from_contract_inputs(inputs)
            return
        if _highd_label(inputs) is not None:
            highd_build.launch_from_contract_inputs(inputs)
            return
        if _eligible_d768_rag(inputs):
            d768_rag.launch_from_contract_inputs(inputs)
            return
        base_current.launch_from_contract_inputs(inputs)
        return
    if _eligible_d256_6164(inputs):
        d256_build.launch_from_contract_inputs(inputs)
        return
    if portfolio_id in PORTFOLIOS_WITH_D768_FAST and d768_build_fast._eligible_d768_build(inputs):
        d768_build_fast.launch_from_contract_inputs(inputs)
        return
    if _highd_label(inputs) is not None:
        highd_build.launch_from_contract_inputs(inputs)
        return
    if _eligible_d768_search(inputs):
        d768_search.launch_from_contract_inputs(inputs)
        return
    if _eligible_d768_rag(inputs):
        d768_rag.launch_from_contract_inputs(inputs)
        return
    if portfolio_id in PORTFOLIOS_WITH_HIGHD_RAG and _highd_rag_label(inputs) is not None:
        highd_rag.launch_from_contract_inputs(inputs)
        return
    if portfolio_id in PORTFOLIOS_WITH_SEARCH_D256 and _search_d256_label(inputs) is not None:
        search_d256.launch_from_contract_inputs(inputs)
        return
    if portfolio_id in PORTFOLIOS_WITH_RAG_D64_M128 and _rag_d64_m128_label(inputs) is not None:
        rag_d64_m128.launch_from_contract_inputs(inputs)
        return
    if portfolio_id == CANDIDATE_D64_REPAIR and _rag_d64_repair_label(inputs) is not None:
        rag_d64_repair.launch_from_contract_inputs(inputs)
        return
    if portfolio_id in PORTFOLIOS_WITH_SEARCH_D256 and _rag_d256_label(inputs) is not None:
        rag_d64d256.launch_from_contract_inputs(inputs)
        return
    if portfolio_id in PORTFOLIOS_WITH_D256_Q4_E2DF and _d256_q4_e2df_label(inputs) is not None:
        d256_q4_e2df.launch_from_contract_inputs(inputs)
        return
    if portfolio_id in PORTFOLIOS_WITH_D64_Q4096_C271 and _eligible_d64_q4096_c271(inputs):
        d64_q4096_c271.launch_from_contract_inputs(inputs)
        return
    if portfolio_id in PORTFOLIOS_WITH_D64_TAIL_017A and _d64_tail_017a_label(inputs) is not None:
        d64_tail_017a.launch_from_contract_inputs(inputs)
        return
    if portfolio_id in PORTFOLIOS_WITH_K13_K48 and _k13_k48_seed_label(inputs)[0] is not None:
        selected_seed, _label = _k13_k48_seed_label(inputs)
        if selected_seed == SEED_K13_ID:
            seed_k13.launch_from_contract_inputs(inputs)
        else:
            seed_k48.launch_from_contract_inputs(inputs)
        return
    if _uses_floor_seed_portfolio(portfolio_id) and _eligible_q128_floor_seed(inputs):
        _q128_module_for_portfolio(inputs, portfolio_id).launch_from_contract_inputs(inputs)
        return
    if portfolio_id in PORTFOLIOS_WITH_Q128_K10_WARPMERGE and _eligible_q128_k10_warpmerge(inputs):
        q128_k10_warpmerge.launch_from_contract_inputs(inputs)
        return
    if _uses_floor_seed_portfolio(portfolio_id) and _eligible_q16_floor_seed(inputs):
        _q16_m250().launch_from_contract_inputs(inputs)
        return
    if portfolio_id in PORTFOLIOS_WITH_LARGE_SQUARE_K32_EFE4 and _eligible_large_square_k32_efe4(inputs):
        large_square_k32_efe4.launch_from_contract_inputs(inputs)
        return
    if portfolio_id in PORTFOLIOS_WITH_RAG_STREAM_K10_34DA and _eligible_rag_stream_k10_34da(inputs):
        _rag_stream_k10_34da().launch_from_contract_inputs(inputs)
        return
    if portfolio_id in PORTFOLIOS_WITH_Q1_M524_N128 and _eligible_q1_m524_n128(inputs):
        q1_m524_n128.launch_from_contract_inputs(inputs)
        return
    if portfolio_id in PORTFOLIOS_WITH_Q32_EXACT and _eligible_q32_exact(inputs):
        q32exact.launch_from_contract_inputs(inputs)
        return
    if portfolio_id in PORTFOLIOS_WITH_Q32_TAIL and _eligible_q32tail(inputs):
        _q32tail().launch_from_contract_inputs(inputs)
        return
    if portfolio_id in PORTFOLIOS_WITH_EXPANDED_K32_Q31_Q33_Q40 and _eligible_q31tail_v2(inputs):
        _q31tail_v2().launch_from_contract_inputs(inputs)
        return
    if portfolio_id in PORTFOLIOS_WITH_EXPANDED_K32_Q31_Q33_Q40 and _eligible_q33tile(inputs):
        _q33tile().launch_from_contract_inputs(inputs)
        return
    if portfolio_id in PORTFOLIOS_WITH_Q48_K12 and _eligible_q48_k12(inputs):
        _q48_k12().launch_from_contract_inputs(inputs)
        return
    if portfolio_id in PORTFOLIOS_WITH_Q32_K31_C3D2 and _eligible_q32_k31_c3d2(inputs):
        _q32_k31_c3d2().launch_from_contract_inputs(inputs)
        return
    if portfolio_id in PORTFOLIOS_WITH_Q32_LOWK_C3D2 and _eligible_q32_lowk_c3d2(inputs):
        _q32_lowk_c3d2().launch_from_contract_inputs(inputs)
        return
    base_current.launch_from_contract_inputs(inputs)

def launch_from_contract_inputs_f1d9_build_policy(inputs: dict[str, Any], *, force_fallback: bool=False) -> None:
    launch_from_contract_inputs(inputs, force_fallback=force_fallback, portfolio_id=CANDIDATE_F1D9_BUILD)

def launch_from_contract_inputs_a4ec_baseline(inputs: dict[str, Any], *, force_fallback: bool=False) -> None:
    launch_from_contract_inputs(inputs, force_fallback=force_fallback, portfolio_id=CANDIDATE_A4EC_BASE)

def launch_from_contract_inputs_fa04_baseline(inputs: dict[str, Any], *, force_fallback: bool=False) -> None:
    launch_from_contract_inputs(inputs, force_fallback=force_fallback, portfolio_id=CANDIDATE_FA04_BASE)

def launch_from_contract_inputs_f328_baseline(inputs: dict[str, Any], *, force_fallback: bool=False) -> None:
    launch_from_contract_inputs(inputs, force_fallback=force_fallback, portfolio_id=CANDIDATE_F328_BASE)

def route_for_contract_inputs_mixed_baseline(inputs: dict[str, Any], *, force_fallback: bool=False) -> str:
    return route_for_contract_inputs(inputs, force_fallback=force_fallback, portfolio_id=CANDIDATE_MIXED)

def launch_from_contract_inputs_mixed_baseline(inputs: dict[str, Any], *, force_fallback: bool=False) -> None:
    launch_from_contract_inputs(inputs, force_fallback=force_fallback, portfolio_id=CANDIDATE_MIXED)

def route_for_contract_inputs_d64_repair_baseline(inputs: dict[str, Any], *, force_fallback: bool=False) -> str:
    return route_for_contract_inputs(inputs, force_fallback=force_fallback, portfolio_id=CANDIDATE_D64_REPAIR)

def launch_from_contract_inputs_d64_repair_baseline(inputs: dict[str, Any], *, force_fallback: bool=False) -> None:
    launch_from_contract_inputs(inputs, force_fallback=force_fallback, portfolio_id=CANDIDATE_D64_REPAIR)

def route_for_contract_inputs_d64_m128_baseline(inputs: dict[str, Any], *, force_fallback: bool=False) -> str:
    return route_for_contract_inputs(inputs, force_fallback=force_fallback, portfolio_id=CANDIDATE_D64_M128)

def launch_from_contract_inputs_d64_m128_baseline(inputs: dict[str, Any], *, force_fallback: bool=False) -> None:
    launch_from_contract_inputs(inputs, force_fallback=force_fallback, portfolio_id=CANDIDATE_D64_M128)

def route_for_contract_inputs_c271_baseline(inputs: dict[str, Any], *, force_fallback: bool=False) -> str:
    return route_for_contract_inputs(inputs, force_fallback=force_fallback, portfolio_id=CANDIDATE_D64_Q4096_C271)

def launch_from_contract_inputs_c271_baseline(inputs: dict[str, Any], *, force_fallback: bool=False) -> None:
    launch_from_contract_inputs(inputs, force_fallback=force_fallback, portfolio_id=CANDIDATE_D64_Q4096_C271)

def route_for_contract_inputs_c271_7dc5_baseline(inputs: dict[str, Any], *, force_fallback: bool=False) -> str:
    return route_for_contract_inputs(inputs, force_fallback=force_fallback, portfolio_id=CANDIDATE_C271_7DC5_K13_K48)

def launch_from_contract_inputs_c271_7dc5_baseline(inputs: dict[str, Any], *, force_fallback: bool=False) -> None:
    launch_from_contract_inputs(inputs, force_fallback=force_fallback, portfolio_id=CANDIDATE_C271_7DC5_K13_K48)

def route_for_contract_inputs_pre_q128_k10_warpmerge_baseline(inputs: dict[str, Any], *, force_fallback: bool=False) -> str:
    return route_for_contract_inputs(inputs, force_fallback=force_fallback, portfolio_id=CANDIDATE_PRE_Q128_K10_WARPMERGE)

def launch_from_contract_inputs_pre_q128_k10_warpmerge_baseline(inputs: dict[str, Any], *, force_fallback: bool=False) -> None:
    launch_from_contract_inputs(inputs, force_fallback=force_fallback, portfolio_id=CANDIDATE_PRE_Q128_K10_WARPMERGE)

def route_for_contract_inputs_2498_baseline(inputs: dict[str, Any], *, force_fallback: bool=False) -> str:
    return route_for_contract_inputs(inputs, force_fallback=force_fallback, portfolio_id=CANDIDATE_2498_BASELINE)

def launch_from_contract_inputs_2498_baseline(inputs: dict[str, Any], *, force_fallback: bool=False) -> None:
    launch_from_contract_inputs(inputs, force_fallback=force_fallback, portfolio_id=CANDIDATE_2498_BASELINE)

def route_for_contract_inputs_pre_q1_m524_n128_baseline(inputs: dict[str, Any], *, force_fallback: bool=False) -> str:
    return route_for_contract_inputs(inputs, force_fallback=force_fallback, portfolio_id=CANDIDATE_PRE_Q1_M524_N128)

def launch_from_contract_inputs_pre_q1_m524_n128_baseline(inputs: dict[str, Any], *, force_fallback: bool=False) -> None:
    launch_from_contract_inputs(inputs, force_fallback=force_fallback, portfolio_id=CANDIDATE_PRE_Q1_M524_N128)

def route_for_contract_inputs_pre_q32exact_baseline(inputs: dict[str, Any], *, force_fallback: bool=False) -> str:
    return route_for_contract_inputs(inputs, force_fallback=force_fallback, portfolio_id=CANDIDATE_PRE_Q32_EXACT)

def launch_from_contract_inputs_pre_q32exact_baseline(inputs: dict[str, Any], *, force_fallback: bool=False) -> None:
    launch_from_contract_inputs(inputs, force_fallback=force_fallback, portfolio_id=CANDIDATE_PRE_Q32_EXACT)

def route_for_contract_inputs_pre_q32tail_baseline(inputs: dict[str, Any], *, force_fallback: bool=False) -> str:
    return route_for_contract_inputs(inputs, force_fallback=force_fallback, portfolio_id=CANDIDATE_PRE_Q32TAIL)

def launch_from_contract_inputs_pre_q32tail_baseline(inputs: dict[str, Any], *, force_fallback: bool=False) -> None:
    launch_from_contract_inputs(inputs, force_fallback=force_fallback, portfolio_id=CANDIDATE_PRE_Q32TAIL)

def route_for_contract_inputs_pre_expanded_k32_q48_baseline(inputs: dict[str, Any], *, force_fallback: bool=False) -> str:
    return route_for_contract_inputs(inputs, force_fallback=force_fallback, portfolio_id=CANDIDATE_PRE_EXPANDED_K32_Q48)

def launch_from_contract_inputs_pre_expanded_k32_q48_baseline(inputs: dict[str, Any], *, force_fallback: bool=False) -> None:
    launch_from_contract_inputs(inputs, force_fallback=force_fallback, portfolio_id=CANDIDATE_PRE_EXPANDED_K32_Q48)

def route_for_contract_inputs_pre_lowk_c3d2_baseline(inputs: dict[str, Any], *, force_fallback: bool=False) -> str:
    return route_for_contract_inputs(inputs, force_fallback=force_fallback, portfolio_id=CANDIDATE_PRE_LOWK_C3D2)

def launch_from_contract_inputs_pre_lowk_c3d2_baseline(inputs: dict[str, Any], *, force_fallback: bool=False) -> None:
    launch_from_contract_inputs(inputs, force_fallback=force_fallback, portfolio_id=CANDIDATE_PRE_LOWK_C3D2)

def route_for_contract_inputs_d665_lowk_baseline(inputs: dict[str, Any], *, force_fallback: bool=False) -> str:
    return route_for_contract_inputs(inputs, force_fallback=force_fallback, portfolio_id=CANDIDATE_D665_LOWK_BASELINE)

def launch_from_contract_inputs_d665_lowk_baseline(inputs: dict[str, Any], *, force_fallback: bool=False) -> None:
    launch_from_contract_inputs(inputs, force_fallback=force_fallback, portfolio_id=CANDIDATE_D665_LOWK_BASELINE)

def route_for_contract_inputs_pre_d64_tail_017a_baseline(inputs: dict[str, Any], *, force_fallback: bool=False) -> str:
    return route_for_contract_inputs(inputs, force_fallback=force_fallback, portfolio_id=CANDIDATE_PRE_D64_TAIL_017A)

def launch_from_contract_inputs_pre_d64_tail_017a_baseline(inputs: dict[str, Any], *, force_fallback: bool=False) -> None:
    launch_from_contract_inputs(inputs, force_fallback=force_fallback, portfolio_id=CANDIDATE_PRE_D64_TAIL_017A)

def route_for_contract_inputs_pre_d256_q4_e2df_baseline(inputs: dict[str, Any], *, force_fallback: bool=False) -> str:
    return route_for_contract_inputs(inputs, force_fallback=force_fallback, portfolio_id=CANDIDATE_PRE_D256_Q4_E2DF)

def launch_from_contract_inputs_pre_d256_q4_e2df_baseline(inputs: dict[str, Any], *, force_fallback: bool=False) -> None:
    launch_from_contract_inputs(inputs, force_fallback=force_fallback, portfolio_id=CANDIDATE_PRE_D256_Q4_E2DF)

def route_for_contract_inputs_pre_d256_q128_59fe_baseline(inputs: dict[str, Any], *, force_fallback: bool=False) -> str:
    return route_for_contract_inputs(inputs, force_fallback=force_fallback, portfolio_id=CANDIDATE_SELECTED_SYNTHESIS)

def launch_from_contract_inputs_pre_d256_q128_59fe_baseline(inputs: dict[str, Any], *, force_fallback: bool=False) -> None:
    launch_from_contract_inputs(inputs, force_fallback=force_fallback, portfolio_id=CANDIDATE_SELECTED_SYNTHESIS)

def route_for_contract_inputs_pre_d256_k32_59fe_baseline(inputs: dict[str, Any], *, force_fallback: bool=False) -> str:
    return route_for_contract_inputs(inputs, force_fallback=force_fallback, portfolio_id=CANDIDATE_D256_Q128_59FE)

def launch_from_contract_inputs_pre_d256_k32_59fe_baseline(inputs: dict[str, Any], *, force_fallback: bool=False) -> None:
    launch_from_contract_inputs(inputs, force_fallback=force_fallback, portfolio_id=CANDIDATE_D256_Q128_59FE)

def route_for_contract_inputs_pre_highd_rag_22e9_baseline(inputs: dict[str, Any], *, force_fallback: bool=False) -> str:
    return route_for_contract_inputs(inputs, force_fallback=force_fallback, portfolio_id=CANDIDATE_D256_K32_59FE)

def launch_from_contract_inputs_pre_highd_rag_22e9_baseline(inputs: dict[str, Any], *, force_fallback: bool=False) -> None:
    launch_from_contract_inputs(inputs, force_fallback=force_fallback, portfolio_id=CANDIDATE_D256_K32_59FE)

def route_for_contract_inputs_pre_highd_search_be66_baseline(inputs: dict[str, Any], *, force_fallback: bool=False) -> str:
    return route_for_contract_inputs(inputs, force_fallback=force_fallback, portfolio_id=CANDIDATE_HIGHD_RAG_22E9)

def launch_from_contract_inputs_pre_highd_search_be66_baseline(inputs: dict[str, Any], *, force_fallback: bool=False) -> None:
    launch_from_contract_inputs(inputs, force_fallback=force_fallback, portfolio_id=CANDIDATE_HIGHD_RAG_22E9)

def route_for_contract_inputs_pre_d128_k48_dd2b_baseline(inputs: dict[str, Any], *, force_fallback: bool=False) -> str:
    return route_for_contract_inputs(inputs, force_fallback=force_fallback, portfolio_id=CANDIDATE_HIGHD_SEARCH_BE66)

def launch_from_contract_inputs_pre_d128_k48_dd2b_baseline(inputs: dict[str, Any], *, force_fallback: bool=False) -> None:
    launch_from_contract_inputs(inputs, force_fallback=force_fallback, portfolio_id=CANDIDATE_HIGHD_SEARCH_BE66)

def route_for_contract_inputs_pre_rect_d128_k20_s12warp4_baseline(inputs: dict[str, Any], *, force_fallback: bool=False) -> str:
    return route_for_contract_inputs(inputs, force_fallback=force_fallback, portfolio_id=CANDIDATE_D128_K48_DD2B)

def launch_from_contract_inputs_pre_rect_d128_k20_s12warp4_baseline(inputs: dict[str, Any], *, force_fallback: bool=False) -> None:
    launch_from_contract_inputs(inputs, force_fallback=force_fallback, portfolio_id=CANDIDATE_D128_K48_DD2B)

def route_for_contract_inputs_pre_q128_k10_rowld_1bed_baseline(inputs: dict[str, Any], *, force_fallback: bool=False) -> str:
    return route_for_contract_inputs(inputs, force_fallback=force_fallback, portfolio_id=CANDIDATE_RECT_D128_K20_S12WARP4)

def launch_from_contract_inputs_pre_q128_k10_rowld_1bed_baseline(inputs: dict[str, Any], *, force_fallback: bool=False) -> None:
    launch_from_contract_inputs(inputs, force_fallback=force_fallback, portfolio_id=CANDIDATE_RECT_D128_K20_S12WARP4)

def candidate(inputs: dict[str, Any]) -> None:
    launch_from_contract_inputs(inputs)

def candidate_f1d9_build_policy(inputs: dict[str, Any]) -> None:
    launch_from_contract_inputs_f1d9_build_policy(inputs)

def candidate_base_current(inputs: dict[str, Any]) -> None:
    base_current.launch_from_contract_inputs(inputs)

def candidate_pre_d256_k32_59fe_baseline(inputs: dict[str, Any]) -> None:
    launch_from_contract_inputs_pre_d256_k32_59fe_baseline(inputs)

def candidate_pre_highd_rag_22e9_baseline(inputs: dict[str, Any]) -> None:
    launch_from_contract_inputs_pre_highd_rag_22e9_baseline(inputs)

def candidate_pre_highd_search_be66_baseline(inputs: dict[str, Any]) -> None:
    launch_from_contract_inputs_pre_highd_search_be66_baseline(inputs)

def candidate_pre_d128_k48_dd2b_baseline(inputs: dict[str, Any]) -> None:
    launch_from_contract_inputs_pre_d128_k48_dd2b_baseline(inputs)

def candidate_pre_rect_d128_k20_s12warp4_baseline(inputs: dict[str, Any]) -> None:
    launch_from_contract_inputs_pre_rect_d128_k20_s12warp4_baseline(inputs)

def candidate_pre_q128_k10_rowld_1bed_baseline(inputs: dict[str, Any]) -> None:
    launch_from_contract_inputs_pre_q128_k10_rowld_1bed_baseline(inputs)

def candidate_a4ec_baseline(inputs: dict[str, Any]) -> None:
    launch_from_contract_inputs_a4ec_baseline(inputs)

def candidate_fa04_baseline(inputs: dict[str, Any]) -> None:
    launch_from_contract_inputs_fa04_baseline(inputs)

def candidate_f328_baseline(inputs: dict[str, Any]) -> None:
    launch_from_contract_inputs_f328_baseline(inputs)

def candidate_mixed_baseline(inputs: dict[str, Any]) -> None:
    launch_from_contract_inputs_mixed_baseline(inputs)

def candidate_d64_repair_baseline(inputs: dict[str, Any]) -> None:
    launch_from_contract_inputs_d64_repair_baseline(inputs)

def candidate_d64_m128_baseline(inputs: dict[str, Any]) -> None:
    launch_from_contract_inputs_d64_m128_baseline(inputs)

def candidate_c271_baseline(inputs: dict[str, Any]) -> None:
    launch_from_contract_inputs_c271_baseline(inputs)

def candidate_c271_7dc5_baseline(inputs: dict[str, Any]) -> None:
    launch_from_contract_inputs_c271_7dc5_baseline(inputs)

def candidate_pre_q128_k10_warpmerge_baseline(inputs: dict[str, Any]) -> None:
    launch_from_contract_inputs_pre_q128_k10_warpmerge_baseline(inputs)

def candidate_2498_baseline(inputs: dict[str, Any]) -> None:
    launch_from_contract_inputs_2498_baseline(inputs)

def candidate_pre_q1_m524_n128_baseline(inputs: dict[str, Any]) -> None:
    launch_from_contract_inputs_pre_q1_m524_n128_baseline(inputs)

def candidate_pre_q32exact_baseline(inputs: dict[str, Any]) -> None:
    launch_from_contract_inputs_pre_q32exact_baseline(inputs)

def candidate_pre_q32tail_baseline(inputs: dict[str, Any]) -> None:
    launch_from_contract_inputs_pre_q32tail_baseline(inputs)

def candidate_pre_expanded_k32_q48_baseline(inputs: dict[str, Any]) -> None:
    launch_from_contract_inputs_pre_expanded_k32_q48_baseline(inputs)

def candidate_pre_lowk_c3d2_baseline(inputs: dict[str, Any]) -> None:
    launch_from_contract_inputs_pre_lowk_c3d2_baseline(inputs)

def candidate_d665_lowk_baseline(inputs: dict[str, Any]) -> None:
    launch_from_contract_inputs_d665_lowk_baseline(inputs)

def candidate_pre_d64_tail_017a_baseline(inputs: dict[str, Any]) -> None:
    launch_from_contract_inputs_pre_d64_tail_017a_baseline(inputs)

def candidate_pre_d256_q4_e2df_baseline(inputs: dict[str, Any]) -> None:
    launch_from_contract_inputs_pre_d256_q4_e2df_baseline(inputs)

def candidate_pre_d256_q128_59fe_baseline(inputs: dict[str, Any]) -> None:
    launch_from_contract_inputs_pre_d256_q128_59fe_baseline(inputs)

def candidate_force_fallback(inputs: dict[str, Any]) -> None:
    launch_from_contract_inputs(inputs, force_fallback=True)

def evaluate_contract(*, shapes=None, correctness: bool=True, benchmark: bool=True, kernel_fn: Callable[[dict[str, Any]], Any]=candidate) -> dict[str, Any]:
    return eval_mod.evaluate(kernel_fn, shapes=shapes, correctness=correctness, benchmark=benchmark)

def compile_and_launch_knn_build(*, shape_labels=CONSUMED_SEED_SHAPES, benchmark: bool=False) -> dict[str, Any]:
    selected = None if shape_labels is None else _select_contract_shapes(shape_labels)
    report = evaluate_contract(shapes=selected, correctness=True, benchmark=benchmark)
    passed = bool(report.get('summary', {}).get('all_correct', False))
    report['passed'] = passed
    report['all_pass'] = passed
    return report

def _base_trace_record(inputs: dict[str, Any], *, force_fallback: bool=False) -> dict[str, Any]:
    label = str(inputs.get('label'))
    if label in EXPANDED_Q32_GUARD_BOUNDARY_8_BY_LABEL:
        timing = EXPANDED_FALLBACK_TIMING_ROWS[label]
        route = base_current.route_for_contract_inputs(inputs, force_fallback=force_fallback)
        return _normalize_route_row({'shape_key': label, 'selected_route': route, 'selected_entrypoint': BASE_ENTRYPOINT, 'selected_seed': None, 'expected_seed': None, 'route_kind': 'general', 'route_source': 'broad-dispatcher', 'guard_id': 'base_common_d_v11_guard_stack', 'guard_condition': 'current v11 common-D fallback dispatcher for expanded Q32-neighborhood row', 'classification': 'fallback-slow', 'base_dispatcher_route': route, 'dispatcher_kernel_ms': timing['kernel_ms'], 'speedup_vs_external_baseline': timing['ratio_vs_flashlib'], 'external_baseline_ms': timing['flashlib_ms'], 'external_baseline_ref': 'historical_same_shape'})
    row = dict(base_current.route_trace_for_contract_shapes((label,), force_fallback=force_fallback)[0])
    row.setdefault('selected_seed', row.get('consumed_seed'))
    row.setdefault('expected_seed', row.get('selected_seed'))
    row.setdefault('selected_entrypoint', BASE_ENTRYPOINT)
    row.setdefault('route_kind', 'general')
    row.setdefault('route_source', 'broad-dispatcher' if not row.get('selected_seed') else 'shape-specific-seed')
    row.setdefault('guard_id', 'base_common_d_v11_guard_stack')
    row.setdefault('guard_condition', 'current v11 common-D fallback dispatcher')
    row.setdefault('classification', 'route-ok')
    if label in HIGHD_SEARCH_BE66_CONSUMED_SHAPES and (not force_fallback):
        row['guard_id'] = 'inherited_397b_or_guard_miss'
        row['guard_condition'] = 'synthesized guard miss; delegate to baseline 7c3a Weave policy'
        row['classification'] = 'benchmark-path-mismatch'
        row['failure_exception_type'] = 'ValueError'
        row['failure_message'] = ''.join(['knn_build_evolve_7bfc_v1 expects D=128, got ', format(int(inputs.get('D', -1)), '')])
        row['speedup_vs_external_baseline'] = 0.0
        row['external_baseline_ref'] = 'not_available'
    if label == V12_D128_K48_OVER32 and (not force_fallback):
        row['guard_id'] = 'inherited_397b_or_guard_miss'
        row['guard_condition'] = 'synthesized guard miss; delegate to baseline 7c3a Weave policy'
        row['classification'] = 'benchmark-path-mismatch'
        row['failure_exception_type'] = 'ValueError'
        row['failure_message'] = 'knn_build_evolve_7bfc_v1 supports K <= 32, got 48'
        row['speedup_vs_external_baseline'] = 0.0
        row['external_baseline_ref'] = 'not_available'
    row['base_dispatcher_route'] = base_current.route_for_contract_inputs(inputs, force_fallback=force_fallback)
    return _normalize_route_row(row)

def _specialized_trace_record(inputs: dict[str, Any], *, portfolio_id: str) -> dict[str, Any] | None:
    label = str(inputs.get('label'))
    base_row = _base_trace_record(inputs)
    d256_q4_e2df_label = _d256_q4_e2df_label(inputs)
    d256_q128_59fe_label = _d256_q128_59fe_label(inputs)
    d256_k32_59fe_label = _d256_k32_59fe_label(inputs)
    highd_rag_22e9_label = _highd_rag_22e9_label(inputs)
    highd_search_be66_label = _highd_search_be66_label(inputs)
    d128_k48_dd2b_label = _d128_k48_dd2b_label(inputs)
    d64_tail_017a_label = _d64_tail_017a_label(inputs)
    rect_d128_k20_s12warp4_eligible = rect_d128_k20_s12warp4._eligible_rect_d128_k20_q1536(inputs)
    if portfolio_id in PORTFOLIOS_WITH_Q128_K10_ROWLD_1BED and (not _eligible_q128_k10_rowld_1bed(inputs)):
        return _specialized_trace_record(inputs, portfolio_id=CANDIDATE_RECT_D128_K20_S12WARP4)
    elif portfolio_id in PORTFOLIOS_WITH_Q128_K10_ROWLD_1BED and _eligible_q128_k10_rowld_1bed(inputs):
        replaced_policy = CANDIDATE_RECT_D128_K20_S12WARP4
    elif portfolio_id in PORTFOLIOS_WITH_RECT_D128_K20_S12WARP4 and (not rect_d128_k20_s12warp4_eligible):
        return _specialized_trace_record(inputs, portfolio_id=CANDIDATE_D128_K48_DD2B)
    elif portfolio_id in PORTFOLIOS_WITH_RECT_D128_K20_S12WARP4 and rect_d128_k20_s12warp4_eligible:
        replaced_policy = CANDIDATE_D128_K48_DD2B
    elif portfolio_id in PORTFOLIOS_WITH_Q32_TAIL143 and _eligible_q32tail143(inputs):
        replaced_policy = CANDIDATE_HIGHD_SEARCH_BE66
    elif portfolio_id in PORTFOLIOS_WITH_D128_K48_DD2B and d128_k48_dd2b_label is None:
        return _specialized_trace_record(inputs, portfolio_id=CANDIDATE_HIGHD_SEARCH_BE66)
    elif portfolio_id in PORTFOLIOS_WITH_D128_K48_DD2B and d128_k48_dd2b_label is not None:
        replaced_policy = CANDIDATE_HIGHD_SEARCH_BE66
    elif portfolio_id in PORTFOLIOS_WITH_HIGHD_SEARCH_BE66 and highd_search_be66_label is None:
        return _specialized_trace_record(inputs, portfolio_id=CANDIDATE_HIGHD_RAG_22E9)
    elif portfolio_id in PORTFOLIOS_WITH_HIGHD_SEARCH_BE66 and highd_search_be66_label is not None:
        replaced_policy = CANDIDATE_HIGHD_RAG_22E9
    elif portfolio_id in PORTFOLIOS_WITH_HIGHD_RAG_22E9 and highd_rag_22e9_label is None:
        return _specialized_trace_record(inputs, portfolio_id=CANDIDATE_D256_K32_59FE)
    elif portfolio_id in PORTFOLIOS_WITH_HIGHD_RAG_22E9 and highd_rag_22e9_label is not None:
        replaced_policy = CANDIDATE_D256_K32_59FE
    elif portfolio_id in PORTFOLIOS_WITH_D256_K32_59FE and d256_k32_59fe_label is None:
        return _specialized_trace_record(inputs, portfolio_id=CANDIDATE_D256_Q128_59FE)
    elif portfolio_id in PORTFOLIOS_WITH_D256_K32_59FE and d256_k32_59fe_label is not None:
        replaced_policy = CANDIDATE_D256_Q128_59FE
    elif portfolio_id in PORTFOLIOS_WITH_D256_Q128_59FE and d256_q128_59fe_label is None:
        return _specialized_trace_record(inputs, portfolio_id=CANDIDATE_SELECTED_SYNTHESIS)
    elif portfolio_id in PORTFOLIOS_WITH_D256_Q128_59FE and d256_q128_59fe_label is not None:
        replaced_policy = CANDIDATE_SELECTED_SYNTHESIS
    elif portfolio_id == CANDIDATE_SELECTED_SYNTHESIS and d256_q4_e2df_label is not None:
        replaced_policy = CANDIDATE_PRE_D256_Q4_E2DF
    elif portfolio_id in PORTFOLIOS_WITH_D64_TAIL_017A and d64_tail_017a_label is not None:
        replaced_policy = CANDIDATE_PRE_D64_TAIL_017A
    elif portfolio_id in (CANDIDATE_PRE_D256_Q4_E2DF, CANDIDATE_SELECTED_SYNTHESIS) and _eligible_q32_k31_c3d2(inputs):
        replaced_policy = CANDIDATE_D665_LOWK_BASELINE
    elif portfolio_id in (CANDIDATE_PRE_D256_Q4_E2DF, CANDIDATE_SELECTED_SYNTHESIS) and _eligible_q32_lowk_c3d2(inputs):
        replaced_policy = CANDIDATE_PRE_LOWK_C3D2
    elif portfolio_id in (CANDIDATE_PRE_LOWK_C3D2, CANDIDATE_PRE_D64_TAIL_017A, CANDIDATE_PRE_D256_Q4_E2DF, CANDIDATE_SELECTED_SYNTHESIS) and (_eligible_q31tail_v2(inputs) or _eligible_q33tile(inputs) or _eligible_q48_k12(inputs)):
        replaced_policy = CANDIDATE_PRE_EXPANDED_K32_Q48
    elif portfolio_id in (CANDIDATE_PRE_EXPANDED_K32_Q48, CANDIDATE_PRE_LOWK_C3D2, CANDIDATE_PRE_D64_TAIL_017A, CANDIDATE_PRE_D256_Q4_E2DF, CANDIDATE_SELECTED_SYNTHESIS) and _eligible_q32tail(inputs):
        replaced_policy = CANDIDATE_PRE_Q32TAIL
    elif portfolio_id in (CANDIDATE_PRE_Q32TAIL, CANDIDATE_PRE_EXPANDED_K32_Q48, CANDIDATE_PRE_LOWK_C3D2, CANDIDATE_PRE_D64_TAIL_017A, CANDIDATE_PRE_D256_Q4_E2DF, CANDIDATE_SELECTED_SYNTHESIS) and _eligible_q32_exact(inputs):
        replaced_policy = CANDIDATE_PRE_Q32_EXACT
    elif portfolio_id in (CANDIDATE_PRE_Q32_EXACT, CANDIDATE_PRE_Q32TAIL, CANDIDATE_PRE_EXPANDED_K32_Q48, CANDIDATE_PRE_LOWK_C3D2, CANDIDATE_PRE_D64_TAIL_017A, CANDIDATE_PRE_D256_Q4_E2DF, CANDIDATE_SELECTED_SYNTHESIS) and _eligible_q1_m524_n128(inputs):
        replaced_policy = CANDIDATE_PRE_Q1_M524_N128
    elif portfolio_id in (CANDIDATE_PRE_Q1_M524_N128, CANDIDATE_PRE_Q32_EXACT, CANDIDATE_PRE_Q32TAIL, CANDIDATE_PRE_EXPANDED_K32_Q48, CANDIDATE_PRE_LOWK_C3D2, CANDIDATE_PRE_D64_TAIL_017A, CANDIDATE_PRE_D256_Q4_E2DF, CANDIDATE_SELECTED_SYNTHESIS) and _eligible_rag_stream_k10_34da(inputs):
        replaced_policy = CANDIDATE_2498_BASELINE
    elif _uses_floor_seed_portfolio(portfolio_id):
        replaced_policy = CANDIDATE_C271_7DC5_K13_K48
    elif portfolio_id == CANDIDATE_C271_7DC5_K13_K48:
        replaced_policy = CANDIDATE_D64_Q4096_C271
    elif portfolio_id == CANDIDATE_D64_Q4096_C271:
        replaced_policy = CANDIDATE_D64_M128
    elif portfolio_id == CANDIDATE_D64_M128:
        replaced_policy = CANDIDATE_D64_REPAIR
    elif portfolio_id == CANDIDATE_D64_REPAIR:
        replaced_policy = CANDIDATE_MIXED
    elif portfolio_id == CANDIDATE_MIXED:
        replaced_policy = CANDIDATE_F328_BASE
    elif portfolio_id == CANDIDATE_F328_BASE:
        replaced_policy = CANDIDATE_FA04_BASE
    else:
        replaced_policy = CANDIDATE_A4EC_BASE
    replaced_route = _route_for_policy(inputs, portfolio_id=replaced_policy)
    if portfolio_id == CANDIDATE_A4EC_BASE and d256_build_a4ec._eligible_d256_q1024(inputs):
        seed_id = 'common_d256_q1024_56f3_v1'
        return _normalize_route_row({'shape_key': label, 'selected_route': d256_build_a4ec.route_for_contract_inputs(inputs), 'selected_entrypoint': D256_A4EC_ENTRYPOINT, 'selected_seed': seed_id, 'expected_seed': seed_id, 'route_kind': 'specialized', 'route_source': 'shape-specific-seed', 'guard_id': 'a4ec_common_d256_q1024_build_exact_guard', 'guard_condition': 'exact BF16 build B=1 Q=M=1024 D=256 K=10', 'classification': 'route-ok', 'replaced_route': base_row.get('selected_route'), 'base_dispatcher_route': base_row.get('selected_route'), 'shape_specific_kernel_ms': 0.053984})
    if _eligible_d256_6164(inputs):
        seed_id = 'common_d_56f3_build_d256_q1024_v1'
        return _normalize_route_row({'shape_key': label, 'selected_route': d256_build.route_for_contract_inputs(inputs), 'selected_entrypoint': D256_ENTRYPOINT, 'selected_seed': seed_id, 'expected_seed': seed_id, 'route_kind': 'specialized', 'route_source': 'shape-specific-seed', 'guard_id': 'fa04_6164_common_d256_q1024_build_exact_guard', 'guard_condition': 'exact BF16 build B=1 Q=M=1024 D=256 K=10', 'classification': 'seed-consumed', 'replaced_route': replaced_route, 'base_dispatcher_route': base_row.get('selected_route'), 'shape_specific_kernel_ms': SEED_TIMING_ROWS[label]['kernel_ms']})
    if portfolio_id in PORTFOLIOS_WITH_D768_FAST and d768_build_fast._eligible_d768_build(inputs):
        seed_id = 'common_d768_build_eeff_m64split_v1'
        return _normalize_route_row({'shape_key': label, 'selected_route': d768_build_fast.route_for_contract_inputs(inputs), 'selected_entrypoint': D768_FAST_ENTRYPOINT, 'selected_seed': seed_id, 'expected_seed': seed_id, 'route_kind': 'specialized', 'route_source': 'shape-specific-seed', 'guard_id': 'a4ec_34d8_common_d768_build_exact_guard', 'guard_condition': 'exact BF16 build B=1 Q=M=1024 D=768 K=10', 'classification': 'seed-consumed', 'replaced_route': replaced_route, 'base_dispatcher_route': base_row.get('selected_route'), 'shape_specific_kernel_ms': SEED_TIMING_ROWS[label]['kernel_ms']})
    highd_label = _highd_label(inputs)
    if highd_label is not None:
        seed_id = 'common_d_56f3_build_highd_v1'
        spec = highd_build.SHAPE_SPECS[highd_label]
        timing = ALT_F1D9_D768_TIMING if highd_label == BUILD_D768 else SEED_TIMING_ROWS[highd_label]
        return _normalize_route_row({'shape_key': highd_label, 'selected_route': highd_build.route_for_contract_inputs(inputs), 'selected_entrypoint': HIGHD_ENTRYPOINT, 'selected_seed': seed_id, 'expected_seed': seed_id, 'route_kind': 'specialized', 'route_source': 'shape-specific-seed', 'guard_id': 'a4ec_f1d9_common_d_highd_build_exact_guard', 'guard_condition': ''.join(['exact BF16 build B=', format(spec['B'], ''), ' Q=', format(spec['Q'], ''), ' M=', format(spec['M'], ''), ' D=', format(spec['D'], ''), ' K=', format(spec['K'], '')]), 'classification': 'seed-consumed', 'feature_chunks': spec['feature_chunks'], 'split_count': spec['split_count'], 'replaced_route': replaced_route, 'base_dispatcher_route': base_row.get('selected_route'), 'shape_specific_kernel_ms': timing['kernel_ms']})
    if _eligible_d768_search(inputs) and portfolio_id != CANDIDATE_A4EC_BASE:
        seed_id = 'common_d_eeff_search_d768_v1'
        return _normalize_route_row({'shape_key': D768_SEARCH, 'selected_route': d768_search.route_for_contract_inputs(inputs), 'selected_entrypoint': D768_SEARCH_ENTRYPOINT, 'selected_seed': seed_id, 'expected_seed': seed_id, 'route_kind': 'specialized', 'route_source': 'shape-specific-seed', 'guard_id': 'fa04_cda9_common_d_search_d768_exact_guard', 'guard_condition': 'exact BF16 non-build B=1 Q=512 M=8192 D=768 K=10', 'classification': 'seed-consumed', 'feature_chunks': d768_search.FEATURE_CHUNKS, 'split_count': d768_search._split_count(), 'group_count': d768_search._group_count(), 'replaced_route': replaced_route, 'base_dispatcher_route': base_row.get('selected_route'), 'shape_specific_kernel_ms': SEED_TIMING_ROWS[D768_SEARCH]['kernel_ms']})
    if _eligible_d768_rag(inputs):
        seed_id = 'non128_frontier_4be7_d768fused_v1'
        spec = d768_rag.SHAPE_SPECS[D768_RAG]
        return _normalize_route_row({'shape_key': D768_RAG, 'selected_route': d768_rag.route_for_contract_inputs(inputs), 'selected_entrypoint': D768_RAG_ENTRYPOINT, 'selected_seed': seed_id, 'expected_seed': seed_id, 'route_kind': 'specialized', 'route_source': 'shape-specific-seed', 'guard_id': 'a4ec_447d_4be7_d768_rag_exact_guard', 'guard_condition': ''.join(['exact BF16 non-build B=', format(spec['B'], ''), ' Q=', format(spec['Q'], ''), ' M=', format(spec['M'], ''), ' D=', format(spec['D'], ''), ' K=', format(spec['K'], '')]), 'classification': 'seed-consumed', 'replaced_route': replaced_route, 'base_dispatcher_route': base_row.get('selected_route'), 'shape_specific_kernel_ms': SEED_TIMING_ROWS[D768_RAG]['kernel_ms']})
    highd_rag_label = _highd_rag_label(inputs)
    if portfolio_id in PORTFOLIOS_WITH_HIGHD_RAG and highd_rag_label is not None:
        seed_id = 'common_d_5e7f_rag_highd_v1'
        spec = highd_rag.SHAPE_SPECS[highd_rag_label]
        return _normalize_route_row({'shape_key': highd_rag_label, 'selected_route': highd_rag.route_for_contract_inputs(inputs), 'selected_entrypoint': HIGHD_RAG_ENTRYPOINT, 'selected_seed': seed_id, 'expected_seed': seed_id, 'route_kind': 'specialized', 'route_source': 'shape-specific-seed', 'guard_id': '4cf7_common_d_5e7f_highd_rag_exact_guard', 'guard_condition': ''.join(['exact BF16 non-build B=', format(spec['B'], ''), ' Q=', format(spec['Q'], ''), ' M=', format(spec['M'], ''), ' D=', format(spec['D'], ''), ' K=', format(spec['K'], '')]), 'classification': 'seed-consumed', 'feature_chunks': spec['feature_chunks'], 'split_count': highd_rag._split_count_for_label(highd_rag_label), 'group_count': highd_rag._group_count_for_label(highd_rag_label), 'producer_topology': 'M64_N64_tcgen05_tma', 'merge_topology': 'fused_group_split_merge', 'replaced_route': replaced_route, 'base_dispatcher_route': base_row.get('selected_route'), 'shape_specific_kernel_ms': SEED_TIMING_ROWS[highd_rag_label]['kernel_ms']})
    search_d256_label = _search_d256_label(inputs)
    if portfolio_id in PORTFOLIOS_WITH_SEARCH_D256 and search_d256_label is not None:
        seed_id = 'common_d_5e7f_search_d256_v1'
        return _normalize_route_row({'shape_key': search_d256_label, 'selected_route': search_d256.route_for_contract_inputs(inputs), 'selected_entrypoint': SEARCH_D256_ENTRYPOINT, 'selected_seed': seed_id, 'expected_seed': seed_id, 'route_kind': 'specialized', 'route_source': 'shape-specific-seed', 'guard_id': '8dbc_common_d_5e7f_search_d256_exact_guard', 'guard_condition': 'exact BF16 non-build B=1 Q=1024 M=32768 D=256 K=10', 'classification': 'seed-consumed', 'feature_chunks': search_d256.FEATURE_CHUNKS, 'split_count': search_d256._split_count(), 'group_count': search_d256._group_count(), 'producer_topology': 'chunked128_d256_tcgen05_tma', 'merge_topology': 'fused_group_split_merge', 'replaced_route': replaced_route, 'base_dispatcher_route': base_row.get('selected_route'), 'shape_specific_kernel_ms': SEED_TIMING_ROWS[search_d256_label]['kernel_ms']})
    rag_d64_m128_label = _rag_d64_m128_label(inputs)
    if portfolio_id in PORTFOLIOS_WITH_RAG_D64_M128 and rag_d64_m128_label is not None:
        seed_id = 'common_d_1438_rag_d64_m128_v1'
        timing = SEED_TIMING_ROWS[rag_d64_m128_label]
        return _normalize_route_row({'shape_key': rag_d64_m128_label, 'selected_route': rag_d64_m128.route_for_contract_inputs(inputs), 'selected_entrypoint': RAG_D64_M128_ENTRYPOINT, 'selected_seed': seed_id, 'expected_seed': seed_id, 'route_kind': 'specialized', 'route_source': 'shape-specific-seed', 'guard_id': '631e_common_d_1438_rag_d64_m128_exact_guard', 'guard_condition': 'exact BF16 non-build B=1 Q=16 M=50000 D=64 K=10', 'classification': 'seed-consumed', 'split_count': rag_d64_m128._split_count(), 'group_count': rag_d64_m128._group_count(), 'producer_topology': 'D64_M128_tcgen05_smem', 'merge_topology': 'fused_group_split_merge', 'replaced_route': replaced_route, 'base_dispatcher_route': base_row.get('selected_route'), 'shape_specific_kernel_ms': timing['kernel_ms'], 'speedup_vs_external_baseline': timing['ratio_vs_flashlib'], 'external_baseline_ms': timing['flashlib_ms'], 'external_baseline_ref': 'historical_same_shape'})
    rag_d64_repair_label = _rag_d64_repair_label(inputs)
    if portfolio_id == CANDIDATE_D64_REPAIR and rag_d64_repair_label is not None:
        seed_id = 'common_d_5e7f_rag_d64_repair_v1'
        spec = rag_d64_repair.SHAPE_SPECS[rag_d64_repair_label]
        timing = D64_REPAIR_0474_TIMING
        return _normalize_route_row({'shape_key': rag_d64_repair_label, 'selected_route': rag_d64_repair.route_for_contract_inputs(inputs), 'selected_entrypoint': RAG_D64_REPAIR_ENTRYPOINT, 'selected_seed': seed_id, 'expected_seed': seed_id, 'route_kind': 'specialized', 'route_source': 'shape-specific-seed', 'guard_id': '0474_common_d_5e7f_rag_d64_repair_exact_guard', 'guard_condition': ''.join(['exact BF16 non-build B=', format(spec['B'], ''), ' Q=', format(spec['Q'], ''), ' M=', format(spec['M'], ''), ' D=', format(spec['D'], ''), ' K=', format(spec['K'], '')]), 'classification': 'seed-consumed', 'split_count': rag_d64_repair._split_count_for_label(rag_d64_repair_label), 'group_count': rag_d64_repair._group_count_for_label(rag_d64_repair_label), 'producer_topology': 'D64_M64_N64_K64_tcgen05_tma', 'merge_topology': 'fused_group_split_merge', 'replaced_route': replaced_route, 'base_dispatcher_route': base_row.get('selected_route'), 'shape_specific_kernel_ms': timing['kernel_ms'], 'speedup_vs_external_baseline': timing['ratio_vs_flashlib'], 'external_baseline_ms': timing['flashlib_ms'], 'external_baseline_ref': 'historical_same_shape'})
    rag_d64d256_label = _rag_d256_label(inputs)
    if portfolio_id in PORTFOLIOS_WITH_SEARCH_D256 and rag_d64d256_label is not None:
        seed_id = 'common_d_5e7f_rag_d64_d256_v1'
        spec = rag_d64d256.SHAPE_SPECS[rag_d64d256_label]
        tma_dim = rag_d64d256._feature_chunks_for_label(rag_d64d256_label) * rag_d64d256.K_TILE
        timing = SEED_TIMING_ROWS[rag_d64d256_label]
        return _normalize_route_row({'shape_key': rag_d64d256_label, 'selected_route': rag_d64d256.route_for_contract_inputs(inputs), 'selected_entrypoint': RAG_D64D256_ENTRYPOINT, 'selected_seed': seed_id, 'expected_seed': seed_id, 'route_kind': 'specialized', 'route_source': 'shape-specific-seed', 'guard_id': 'ba22_common_d_5e7f_rag_d64_d256_exact_guard', 'guard_condition': ''.join(['exact BF16 non-build B=', format(spec['B'], ''), ' Q=', format(spec['Q'], ''), ' M=', format(spec['M'], ''), ' D=', format(spec['D'], ''), ' K=', format(spec['K'], '')]), 'classification': 'kernel-slow' if float(timing['ratio_vs_flashlib']) < SPEEDUP_FLOOR else 'seed-consumed', 'feature_chunks': spec['feature_chunks'], 'split_count': rag_d64d256._split_count_for_label(rag_d64d256_label), 'group_count': rag_d64d256._group_count_for_label(rag_d64d256_label), 'producer_topology': 'D64_M64_tcgen05_tma' if rag_d64d256._uses_d64_exact(rag_d64d256_label) else 'M64_N64_tcgen05_tma', 'preprocess_stage': None if rag_d64d256._uses_d64_exact(rag_d64d256_label) else ''.join(['d', format(int(spec['D']), ''), '_weave_pad_to_d', format(tma_dim, '')]) if int(spec['D']) != tma_dim else None, 'merge_topology': 'fused_group_split_merge', 'replaced_route': replaced_route, 'base_dispatcher_route': base_row.get('selected_route'), 'shape_specific_kernel_ms': timing['kernel_ms'], 'speedup_vs_external_baseline': timing['ratio_vs_flashlib'], 'external_baseline_ms': timing['flashlib_ms'], 'external_baseline_ref': 'historical_same_shape'})
    if portfolio_id in PORTFOLIOS_WITH_D256_Q4_E2DF and d256_q4_e2df_label is not None:
        spec = d256_q4_e2df.SHAPE_SPECS[d256_q4_e2df_label]
        timing = SEED_TIMING_ROWS[d256_q4_e2df_label]
        return _normalize_route_row({'shape_key': d256_q4_e2df_label, 'selected_route': d256_q4_e2df.route_for_contract_inputs(inputs), 'selected_entrypoint': D256_Q4_E2DF_ENTRYPOINT, 'selected_seed': SEED_D256_Q4_E2DF_ID, 'expected_seed': SEED_D256_Q4_E2DF_ID, 'route_kind': 'specialized', 'route_source': 'shape-specific-seed', 'guard_id': D256_Q4_E2DF_GUARD_ID, 'guard_condition': ''.join(['exact BF16 non-build B=', format(spec['B'], ''), ' Q=', format(spec['Q'], ''), ' M=', format(spec['M'], ''), ' D=', format(spec['D'], ''), ' K=', format(spec['K'], '')]), 'classification': 'seed-consumed' if float(timing['ratio_vs_flashlib']) >= SPEEDUP_FLOOR else 'kernel-slow', 'feature_chunks': spec['feature_chunks'], 'split_count': d256_q4_e2df._split_count_for_label(d256_q4_e2df_label), 'group_count': d256_q4_e2df._group_count_for_label(d256_q4_e2df_label), 'producer_topology': 'M64_N64_D256_tcgen05/TMA_chunked', 'merge_topology': 'fused_group_split_merge', 'replaced_route': replaced_route, 'base_dispatcher_route': base_row.get('selected_route'), 'shape_specific_kernel_ms': timing['kernel_ms'], 'speedup_vs_external_baseline': timing['ratio_vs_flashlib'], 'external_baseline_ms': timing['flashlib_ms'], 'external_baseline_ref': 'historical_same_shape'})
    if portfolio_id in PORTFOLIOS_WITH_D256_Q128_59FE and d256_q128_59fe_label is not None:
        spec = d256_q128_59fe.SHAPE_SPECS[d256_q128_59fe_label]
        timing = SEED_TIMING_ROWS[d256_q128_59fe_label]
        return _normalize_route_row({'shape_key': d256_q128_59fe_label, 'selected_route': d256_q128_59fe.route_for_contract_inputs(inputs), 'selected_entrypoint': D256_Q128_59FE_ENTRYPOINT, 'selected_seed': SEED_D256_Q128_59FE_ID, 'expected_seed': SEED_D256_Q128_59FE_ID, 'route_kind': 'specialized', 'route_source': 'shape-specific-seed', 'guard_id': D256_Q128_59FE_GUARD_ID, 'guard_condition': ''.join(['exact BF16 non-build B=', format(spec['B'], ''), ' Q=', format(spec['Q'], ''), ' M=', format(spec['M'], ''), ' D=', format(spec['D'], ''), ' K=', format(spec['K'], '')]), 'classification': 'seed-consumed' if float(timing['ratio_vs_flashlib']) >= SPEEDUP_FLOOR else 'kernel-slow', 'feature_chunks': spec['feature_chunks'], 'split_count': d256_q128_59fe._split_count_for_label(d256_q128_59fe_label), 'group_count': d256_q128_59fe._group_count_for_label(d256_q128_59fe_label), 'producer_topology': 'M128_N64_D256_tcgen05/TMA_chunked', 'merge_topology': 'fused_group_split_merge', 'replaced_route': replaced_route, 'base_dispatcher_route': base_row.get('selected_route'), 'shape_specific_kernel_ms': timing['kernel_ms'], 'speedup_vs_external_baseline': timing['ratio_vs_flashlib'], 'external_baseline_ms': timing['flashlib_ms'], 'external_baseline_ref': 'historical_same_shape'})
    if portfolio_id in PORTFOLIOS_WITH_D256_K32_59FE and d256_k32_59fe_label is not None:
        spec = d256_k32_59fe.SHAPE_SPECS[d256_k32_59fe_label]
        timing = SEED_TIMING_ROWS[d256_k32_59fe_label]
        return _normalize_route_row({'shape_key': d256_k32_59fe_label, 'selected_route': d256_k32_59fe.route_for_contract_inputs(inputs), 'selected_entrypoint': D256_K32_59FE_ENTRYPOINT, 'selected_seed': SEED_D256_K32_59FE_ID, 'expected_seed': SEED_D256_K32_59FE_ID, 'route_kind': 'specialized', 'route_source': 'shape-specific-seed', 'guard_id': D256_K32_59FE_GUARD_ID, 'guard_condition': 'exact BF16 non-build B=1 M=100000 D=256 K=32 Q in {8,128}', 'classification': 'seed-consumed' if float(timing['ratio_vs_flashlib']) >= SPEEDUP_FLOOR else 'kernel-slow', 'feature_chunks': d256_k32_59fe.D256_FEATURE_CHUNKS, 'split_count': d256_k32_59fe._split_count_for_label(d256_k32_59fe_label), 'rows_per_merge_cta': d256_k32_59fe.D256_WARP_MERGE_ROWS_PER_CTA, 'producer_topology': 'ROW_16x256B_M64N64_D256_tcgen05_TMA_two_chunk', 'merge_topology': 'warp_row_split_list_merge', 'replaced_route': replaced_route, 'base_dispatcher_route': base_row.get('selected_route'), 'shape_specific_kernel_ms': timing['kernel_ms'], 'speedup_vs_external_baseline': timing['ratio_vs_flashlib'], 'external_baseline_ms': timing['flashlib_ms'], 'external_baseline_ref': 'historical_same_shape', 'K': spec['K'], 'Q': spec['Q'], 'M': spec['M'], 'D': spec['D'], 'B': spec['B'], 'build': spec['build']})
    if portfolio_id in PORTFOLIOS_WITH_D128_K48_DD2B and d128_k48_dd2b_label is not None:
        timing = SEED_TIMING_ROWS[d128_k48_dd2b_label]
        ratio = timing['ratio_vs_flashlib']
        return _normalize_route_row({'shape_key': d128_k48_dd2b_label, 'selected_route': d128_k48_dd2b.route_for_contract_inputs(inputs), 'selected_entrypoint': D128_K48_DD2B_ENTRYPOINT, 'selected_seed': SEED_D128_K48_DD2B_ID, 'expected_seed': SEED_D128_K48_DD2B_ID, 'route_kind': 'specialized', 'route_source': 'shape-specific-seed', 'guard_id': D128_K48_DD2B_GUARD_ID, 'guard_condition': 'exact BF16 non-build B=1 Q=16 M=100000 D=128 K=48', 'classification': 'seed-consumed' if ratio >= SPEEDUP_FLOOR else 'kernel-slow', 'split_count': d128_k48_dd2b.K48_SPLIT_COUNT, 'rows_per_merge_cta': d128_k48_dd2b.K48_ROWS_PER_CTA, 'producer_topology': 'ROW_16x256B two-compute-warp K48 stage', 'merge_topology': 'K48 warp-row split-list merge', 'replaced_route': replaced_route, 'base_dispatcher_route': base_row.get('selected_route'), 'shape_specific_kernel_ms': timing['kernel_ms'], 'speedup_vs_external_baseline': ratio, 'external_baseline_ms': timing['flashlib_ms'], 'external_baseline_ref': 'historical_same_shape', 'K': 48, 'Q': 16, 'M': 100000, 'D': 128, 'B': 1, 'build': False})
    if portfolio_id in PORTFOLIOS_WITH_RECT_D128_K20_S12WARP4 and rect_d128_k20_s12warp4_eligible:
        timing = SEED_TIMING_ROWS[RECT_D128_K20_Q1536]
        ratio = timing['ratio_vs_flashlib']
        return _normalize_route_row({'shape_key': RECT_D128_K20_Q1536, 'selected_route': rect_d128_k20_s12warp4.route_for_contract_inputs(inputs), 'selected_entrypoint': RECT_D128_K20_S12WARP4_ENTRYPOINT, 'selected_seed': SEED_RECT_D128_K20_S12WARP4_ID, 'expected_seed': SEED_RECT_D128_K20_S12WARP4_ID, 'route_kind': 'specialized', 'route_source': 'shape-specific-seed', 'guard_id': RECT_D128_K20_S12WARP4_GUARD_ID, 'guard_condition': 'exact BF16 non-build B=1 Q=1536 M=65536 D=128 K=20', 'classification': 'seed-consumed' if ratio >= SPEEDUP_FLOOR else 'kernel-slow', 'split_count': rect_d128_k20_s12warp4.DEFAULT_SPLIT_COUNT, 'rows_per_merge_cta': 4, 'producer_topology': '7768 split12 unordered tcgen05/TMA K20 stage1', 'merge_topology': 'warp4 repeated-min K20 merge', 'replaced_route': replaced_route, 'parent_v11_route': replaced_route, 'base_dispatcher_route': base_row.get('selected_route'), 'shape_specific_kernel_ms': timing['kernel_ms'], 'speedup_vs_external_baseline': ratio, 'external_baseline_ms': timing['flashlib_ms'], 'external_baseline_ref': 'historical_same_shape', 'K': 20, 'Q': 1536, 'M': 65536, 'D': 128, 'B': 1, 'build': False})
    if portfolio_id in PORTFOLIOS_WITH_D64_Q4096_C271 and _eligible_d64_q4096_c271(inputs):
        seed_id = 'd64_q4096_c271_twostage_v1'
        timing = SEED_TIMING_ROWS[D64_Q4096]
        return _normalize_route_row({'shape_key': D64_Q4096, 'selected_route': d64_q4096_c271.route_name_for_inputs(inputs), 'selected_entrypoint': D64_Q4096_C271_ENTRYPOINT, 'selected_seed': seed_id, 'expected_seed': seed_id, 'route_kind': 'specialized', 'route_source': 'shape-specific-seed', 'guard_id': 'c271_d64_q4096_build_exact_guard', 'guard_condition': 'exact BF16 build B=1 Q=M=4096 D=64 K=10', 'classification': 'seed-consumed', 'split_count': d64_q4096_c271.STAGE1_SPLIT4, 'producer_topology': 'D64_Q4096_tcgen05_tma_unordered_split4', 'merge_topology': 'exact_k10_rowbase_cached_split4_merge', 'replaced_route': replaced_route, 'base_dispatcher_route': base_row.get('selected_route'), 'shape_specific_kernel_ms': timing['kernel_ms'], 'speedup_vs_external_baseline': timing['ratio_vs_flashlib'], 'external_baseline_ms': timing['flashlib_ms'], 'external_baseline_ref': 'historical_same_shape'})
    if portfolio_id in PORTFOLIOS_WITH_HIGHD_SEARCH_BE66 and highd_search_be66_label is not None:
        spec = highd_search_be66.SHAPE_SPECS[highd_search_be66_label]
        timing = SEED_TIMING_ROWS[highd_search_be66_label]
        ratio = timing['ratio_vs_flashlib']
        return _normalize_route_row({'shape_key': highd_search_be66_label, 'selected_route': highd_search_be66.route_for_contract_inputs(inputs), 'selected_entrypoint': HIGHD_SEARCH_BE66_ENTRYPOINT, 'selected_seed': SEED_HIGHD_SEARCH_BE66_ID, 'expected_seed': SEED_HIGHD_SEARCH_BE66_ID, 'route_kind': 'specialized', 'route_source': 'shape-specific-seed', 'guard_id': HIGHD_SEARCH_BE66_GUARD_ID, 'guard_condition': 'exact BF16 non-build v12 high-D rectangular search B=1 K=10 for D/Q/M in {(1024,256,8192), (4096,128,4096)}', 'classification': 'seed-consumed' if ratio >= SPEEDUP_FLOOR else 'kernel-slow', 'feature_chunks': spec['feature_chunks'], 'split_count': highd_search_be66._split_count_for_label(highd_search_be66_label), 'group_count': highd_search_be66._group_count_for_label(highd_search_be66_label), 'producer_topology': 'M128_N64_tcgen05_tma_highd_chunked', 'merge_topology': 'fused_group_split_merge', 'replaced_route': replaced_route, 'base_dispatcher_route': base_row.get('selected_route'), 'shape_specific_kernel_ms': timing['kernel_ms'], 'speedup_vs_external_baseline': ratio, 'external_baseline_ms': timing['flashlib_ms'], 'external_baseline_ref': 'historical_same_shape', 'K': spec['K'], 'Q': spec['Q'], 'M': spec['M'], 'D': spec['D'], 'B': spec['B'], 'build': spec['build']})
    if portfolio_id in PORTFOLIOS_WITH_HIGHD_RAG_22E9 and highd_rag_22e9_label is not None:
        spec = highd_rag_22e9.SHAPE_SPECS[highd_rag_22e9_label]
        timing = SEED_TIMING_ROWS[highd_rag_22e9_label]
        ratio = timing['ratio_vs_flashlib']
        return _normalize_route_row({'shape_key': highd_rag_22e9_label, 'selected_route': highd_rag_22e9.route_for_contract_inputs(inputs), 'selected_entrypoint': HIGHD_RAG_22E9_ENTRYPOINT, 'selected_seed': SEED_HIGHD_RAG_22E9_ID, 'expected_seed': SEED_HIGHD_RAG_22E9_ID, 'route_kind': 'specialized', 'route_source': 'shape-specific-seed', 'guard_id': HIGHD_RAG_22E9_GUARD_ID, 'guard_condition': 'exact BF16 non-build v12 high-D RAG B=1 K=10 for D/Q/M in {(768,8,100000), (1024,4,100000), (4096,1,65536)}', 'classification': 'seed-consumed' if ratio >= SPEEDUP_FLOOR else 'kernel-slow', 'feature_chunks': spec['feature_chunks'], 'split_count': highd_rag_22e9._split_count_for_label(highd_rag_22e9_label), 'group_count': highd_rag_22e9._group_count_for_label(highd_rag_22e9_label), 'producer_topology': 'M64_N64_tcgen05_tma_highd_chunked', 'merge_topology': 'fused_group_split_merge', 'replaced_route': replaced_route, 'base_dispatcher_route': base_row.get('selected_route'), 'shape_specific_kernel_ms': timing['kernel_ms'], 'speedup_vs_external_baseline': ratio, 'external_baseline_ms': timing['flashlib_ms'], 'external_baseline_ref': 'historical_same_shape'})
    if portfolio_id in PORTFOLIOS_WITH_D64_TAIL_017A and d64_tail_017a_label is not None:
        timing = SEED_TIMING_ROWS[d64_tail_017a_label]
        spec = d64_tail_017a.SHAPE_SPECS[d64_tail_017a_label]
        return _normalize_route_row({'shape_key': d64_tail_017a_label, 'selected_route': d64_tail_017a.route_for_contract_inputs(inputs), 'selected_entrypoint': D64_TAIL_017A_ENTRYPOINT, 'selected_seed': SEED_D64_TAIL_017A_ID, 'expected_seed': SEED_D64_TAIL_017A_ID, 'route_kind': 'specialized', 'route_source': 'shape-specific-seed', 'guard_id': D64_TAIL_017A_GUARD_ID, 'guard_condition': ''.join(['exact BF16 non-build B=', format(spec['B'], ''), ' Q=', format(spec['Q'], ''), ' M=', format(spec['M'], ''), ' D=', format(spec['D'], ''), ' K=', format(spec['K'], '')]), 'classification': 'seed-consumed' if float(timing['ratio_vs_flashlib']) >= SPEEDUP_FLOOR else 'kernel-slow', 'split_count': d64_tail_017a._split_count_for_label(d64_tail_017a_label), 'group_count': d64_tail_017a._group_count_for_label(d64_tail_017a_label), 'producer_topology': 'D64_M64_tcgen05_tma_tail', 'merge_topology': 'fused_group_split_merge', 'replaced_route': replaced_route, 'base_dispatcher_route': base_row.get('selected_route'), 'shape_specific_kernel_ms': timing['kernel_ms'], 'speedup_vs_external_baseline': timing['ratio_vs_flashlib'], 'external_baseline_ms': timing['flashlib_ms'], 'external_baseline_ref': 'historical_same_shape'})
    k13_k48_seed, k13_k48_label = _k13_k48_seed_label(inputs)
    if portfolio_id in PORTFOLIOS_WITH_K13_K48 and k13_k48_seed is not None:
        is_k13 = k13_k48_seed == SEED_K13_ID
        route = _route_k13_k48(inputs)
        timing = SEED_TIMING_ROWS[str(k13_k48_label)]
        return _normalize_route_row({'shape_key': k13_k48_label, 'selected_route': route, 'selected_entrypoint': K13_ENTRYPOINT if is_k13 else K48_ENTRYPOINT, 'selected_seed': k13_k48_seed, 'expected_seed': k13_k48_seed, 'route_kind': 'specialized', 'route_source': 'shape-specific-seed', 'guard_id': '7dc5_2c1c_q4096_k13_exact_guard' if is_k13 else '7dc5_d03c_k48_exact_guard', 'guard_condition': 'exact BF16 build B=1 Q=M=4096 D=128 K=13 unordered split4' if is_k13 else 'exact BF16 build B=1 Q=M in {2048,4096} D=128 K=48 split4 warp-select', 'classification': 'seed-consumed', 'split_count': seed_k13.Q4096_K13_SPLIT_COUNT if is_k13 else seed_k48.K48_SPLITS, 'producer_topology': '2c1c_unordered_split4_tcgen05_tma' if is_k13 else 'd03c_k48_split4_tcgen05_tma', 'merge_topology': 'unordered_split4_merge' if is_k13 else 'warp_select_split4_merge', 'replaced_route': replaced_route, 'base_dispatcher_route': base_row.get('selected_route'), 'shape_specific_kernel_ms': timing['kernel_ms'], 'speedup_vs_external_baseline': timing['ratio_vs_flashlib'], 'external_baseline_ms': timing['flashlib_ms'], 'external_baseline_ref': 'historical_same_shape'})
    if _uses_floor_seed_portfolio(portfolio_id) and _eligible_q128_floor_seed(inputs):
        q128_mod = _q128_module_for_portfolio(inputs, portfolio_id)
        if q128_mod is _q128_dualm():
            seed_id = SEED_Q128_DUALM_ID
            timing = Q128_DUALM_TIMING_ROWS[label]
            guard_id = 'a162_q128_dualm_k32_rowld_s72_warp1_exact_guard'
            rows_per_cta = 1
        else:
            seed_id = SEED_Q128_S72R2_ID
            timing = Q128_S72R2_TIMING_ROWS[label]
            guard_id = 'a162_q128_stream_k32_s72r2_exact_guard'
            rows_per_cta = 2
        return _normalize_route_row({'shape_key': label, 'selected_route': q128_mod.route_for_contract_inputs(inputs), 'selected_entrypoint': q128_mod.ROUTE_ENTRYPOINT, 'selected_seed': seed_id, 'expected_seed': seed_id, 'route_kind': 'specialized', 'route_source': 'shape-specific-seed', 'guard_id': guard_id, 'guard_condition': 'exact BF16 non-build B=1 Q=128 M in {100000,131071} D=128 K=32', 'classification': 'seed-consumed', 'matched_label': label, 'split_count': 72, 'rows_per_cta': rows_per_cta, 'producer_topology': 'Q128_K32_rowld_tcgen05_tma', 'merge_topology': ''.join(['warp-row split-list merge rows_per_cta=', format(rows_per_cta, '')]), 'replaced_route': replaced_route, 'base_dispatcher_route': base_row.get('selected_route'), 'shape_specific_kernel_ms': timing['kernel_ms'], 'speedup_vs_external_baseline': timing['ratio_vs_flashlib'], 'external_baseline_ms': timing['flashlib_ms'], 'external_baseline_ref': 'historical_same_shape'})
    if portfolio_id in PORTFOLIOS_WITH_Q128_K10_WARPMERGE and _eligible_q128_k10_warpmerge(inputs):
        timing = Q128_K10_WARPMERGE_TIMING
        return _normalize_route_row({'shape_key': RAG_Q128_M100000_K10, 'selected_route': q128_k10_warpmerge.route_for_contract_inputs(inputs), 'selected_entrypoint': Q128_K10_WARPMERGE_ENTRYPOINT, 'selected_seed': SEED_Q128_K10_WARPMERGE_ID, 'expected_seed': SEED_Q128_K10_WARPMERGE_ID, 'route_kind': 'specialized', 'route_source': 'shape-specific-seed', 'guard_id': '4ce8_df0f_q128_m100000_k10_s74_warpmerge_exact_guard', 'guard_condition': 'exact BF16 RAG B=1 Q=128 M=100000 D=128 K=10 split74 warp-merge', 'classification': 'seed-consumed', 'matched_label': RAG_Q128_M100000_K10, 'split_count': q128_k10_warpmerge.SPLIT_COUNT, 'rows_per_cta': q128_k10_warpmerge.ROWS_PER_MERGE_CTA, 'producer_topology': 'Q128_K10_split74_tcgen05_tma', 'merge_topology': 'one_warp_per_query_row_three_splits_per_lane', 'replaced_route': replaced_route, 'base_dispatcher_route': base_row.get('selected_route'), 'shape_specific_kernel_ms': timing['kernel_ms'], 'speedup_vs_external_baseline': timing['ratio_vs_flashlib'], 'external_baseline_ms': timing['flashlib_ms'], 'external_baseline_ref': 'historical_same_shape'})
    if portfolio_id in PORTFOLIOS_WITH_Q128_K10_ROWLD_1BED and _eligible_q128_k10_rowld_1bed(inputs):
        timing = Q128_K10_ROWLD_1BED_TIMING
        return _normalize_route_row({'shape_key': RAG_Q128_M100000_K10, 'selected_route': _q128_k10_rowld_1bed().route_for_contract_inputs(inputs), 'selected_entrypoint': Q128_K10_ROWLD_1BED_ENTRYPOINT, 'selected_seed': SEED_Q128_K10_ROWLD_1BED_ID, 'expected_seed': SEED_Q128_K10_ROWLD_1BED_ID, 'route_kind': 'specialized', 'route_source': 'shape-specific-seed', 'guard_id': Q128_K10_ROWLD_1BED_GUARD_ID, 'guard_condition': 'exact BF16 non-build B=1 Q=128 M=100000 D=128 K=10 split74 row-load', 'classification': 'seed-consumed', 'matched_label': RAG_Q128_M100000_K10, 'split_count': _q128_k10_rowld_1bed().DEFAULT_SPLIT_COUNT, 'rows_per_merge_cta': _q128_k10_rowld_1bed().ROWS_PER_MERGE_CTA, 'producer_topology': 'ROW_16x256B_q128_k10_rowld_tcgen05_tma_stage1', 'merge_topology': 'warp-row split-list merge', 'replaced_route': replaced_route, 'base_dispatcher_route': base_row.get('selected_route'), 'shape_specific_kernel_ms': timing['kernel_ms'], 'speedup_vs_external_baseline': timing['ratio_vs_flashlib'], 'external_baseline_ms': timing['flashlib_ms'], 'external_baseline_ref': 'historical_same_shape'})
    if _uses_floor_seed_portfolio(portfolio_id) and _eligible_q16_floor_seed(inputs):
        timing = Q16_M250_TIMING
        return _normalize_route_row({'shape_key': RAG_Q16_M250000_K32, 'selected_route': _q16_m250().route_for_contract_inputs(inputs), 'selected_entrypoint': Q16_M250_ENTRYPOINT, 'selected_seed': SEED_Q16_M250_ID, 'expected_seed': SEED_Q16_M250_ID, 'route_kind': 'specialized', 'route_source': 'shape-specific-seed', 'guard_id': 'df0f_q16_m250_k32_exact_guard', 'guard_condition': 'exact BF16 RAG B=1 Q=16 M=250000 D=128 K=32 split288', 'classification': 'seed-consumed', 'matched_label': RAG_Q16_M250000_K32, 'split_count': 288, 'producer_topology': 'Q16_M250000_K32_rowld1_2warp_tcgen05_tma', 'merge_topology': 'rows4 warp split-list merge', 'replaced_route': replaced_route, 'base_dispatcher_route': base_row.get('selected_route'), 'shape_specific_kernel_ms': timing['kernel_ms'], 'speedup_vs_external_baseline': timing['ratio_vs_flashlib'], 'external_baseline_ms': timing['flashlib_ms'], 'external_baseline_ref': 'historical_same_shape'})
    if portfolio_id in PORTFOLIOS_WITH_LARGE_SQUARE_K32_EFE4 and _eligible_large_square_k32_efe4(inputs):
        timing = SEED_TIMING_ROWS[BUILD_LARGE_SQUARE_K32]
        return _normalize_route_row({'shape_key': BUILD_LARGE_SQUARE_K32, 'selected_route': large_square_k32_efe4.route_for_contract_inputs(inputs), 'selected_entrypoint': LARGE_SQUARE_K32_EFE4_ENTRYPOINT, 'selected_seed': SEED_LARGE_SQUARE_K32_EFE4_ID, 'expected_seed': SEED_LARGE_SQUARE_K32_EFE4_ID, 'route_kind': 'specialized', 'route_source': 'shape-specific-seed', 'guard_id': 'fac0_large_square_k32_efe4_prodcache_exact_guard', 'guard_condition': 'exact BF16 build B=1 Q=M=8192 D=128 K=32 split2 producer-cache', 'classification': 'seed-consumed', 'split_count': large_square_k32_efe4.SPLIT_COUNT, 'producer_topology': 'EFE4_split2_tcgen05_tma_chunkworst_prodcache', 'merge_topology': 'split2_warp8_k32_merge', 'replaced_route': replaced_route, 'base_dispatcher_route': base_row.get('selected_route'), 'shape_specific_kernel_ms': timing['kernel_ms'], 'speedup_vs_external_baseline': timing['ratio_vs_flashlib'], 'external_baseline_ms': timing['flashlib_ms'], 'external_baseline_ref': 'historical_same_shape'})
    if portfolio_id in PORTFOLIOS_WITH_RAG_STREAM_K10_34DA and _eligible_rag_stream_k10_34da(inputs):
        timing = SEED_TIMING_ROWS[RAG_STREAM_K10]
        mod = _rag_stream_k10_34da()
        return _normalize_route_row({'shape_key': RAG_STREAM_K10, 'selected_route': mod.route_for_contract_inputs(inputs), 'selected_entrypoint': RAG_STREAM_K10_34DA_ENTRYPOINT, 'selected_seed': SEED_RAG_STREAM_K10_34DA_ID, 'expected_seed': SEED_RAG_STREAM_K10_34DA_ID, 'route_kind': 'specialized', 'route_source': 'shape-specific-seed', 'guard_id': '34da_rag_stream_k10_s72_warpmerge_exact_guard', 'guard_condition': 'exact BF16 non-build B=1 Q=128 M=100000 D=128 K=10', 'classification': 'seed-consumed', 'split_count': mod.SPLIT_COUNT, 'rows_per_merge_cta': mod.ROWS_PER_CTA, 'producer_topology': 'split72_tcgen05_tma_k10_stage1', 'merge_topology': 'warp-row split-list merge', 'replaced_route': replaced_route, 'base_dispatcher_route': base_row.get('selected_route'), 'shape_specific_kernel_ms': timing['kernel_ms'], 'speedup_vs_external_baseline': timing['ratio_vs_flashlib'], 'external_baseline_ms': timing['flashlib_ms'], 'external_baseline_ref': 'historical_same_shape'})
    if portfolio_id in PORTFOLIOS_WITH_Q1_M524_N128 and _eligible_q1_m524_n128(inputs):
        timing = SEED_TIMING_ROWS[RAG_Q1_M524287_K10]
        return _normalize_route_row({'shape_key': RAG_Q1_M524287_K10, 'selected_route': q1_m524_n128.route_for_contract_inputs(inputs), 'selected_entrypoint': Q1_M524_N128_ENTRYPOINT, 'selected_seed': SEED_Q1_M524_N128_ID, 'expected_seed': SEED_Q1_M524_N128_ID, 'route_kind': 'specialized', 'route_source': 'shape-specific-seed', 'guard_id': '32b9_ea43_q1_m524287_n128_exact_guard', 'guard_condition': 'exact BF16 non-build B=1 Q=1 M=524287 D=128 K=10 split148 group4', 'classification': 'seed-consumed', 'split_count': q1_m524_n128.Q1_N128_SPLIT, 'group_count': q1_m524_n128.Q1_N128_GROUPS, 'producer_topology': 'Q1_M524287_M64_N128_tcgen05_tma', 'merge_topology': 's148_g4_fused_sorted_stream_merge', 'replaced_route': replaced_route, 'base_dispatcher_route': base_row.get('selected_route'), 'shape_specific_kernel_ms': timing['kernel_ms'], 'speedup_vs_external_baseline': timing['ratio_vs_flashlib'], 'external_baseline_ms': timing['flashlib_ms'], 'external_baseline_ref': 'historical_same_shape'})
    if portfolio_id in PORTFOLIOS_WITH_Q32_EXACT and _eligible_q32_exact(inputs):
        timing = SEED_TIMING_ROWS[RAG_Q32_M100000_K32]
        return _normalize_route_row({'shape_key': RAG_Q32_M100000_K32, 'selected_route': q32exact.route_for_contract_inputs(inputs), 'selected_entrypoint': Q32_EXACT_ENTRYPOINT, 'selected_seed': SEED_Q32_EXACT_ID, 'expected_seed': SEED_Q32_EXACT_ID, 'route_kind': 'specialized', 'route_source': 'shape-specific-seed', 'guard_id': Q32_EXACT_GUARD_ID, 'guard_condition': 'exact BF16 non-build B=1 Q=32 M=100000 D=128 K=32', 'classification': 'seed-consumed', 'split_count': q32exact.K32_Q32_SPLIT_COUNT, 'rows_per_merge_cta': q32exact.K32_ROWS4_ROWS_PER_CTA, 'producer_topology': Q32_EXACT_PRODUCER_TOPOLOGY, 'merge_topology': 'rows4 warp-row split-list merge', 'replaced_route': replaced_route, 'base_dispatcher_route': base_row.get('selected_route'), 'shape_specific_kernel_ms': timing['kernel_ms'], 'speedup_vs_external_baseline': timing['ratio_vs_flashlib'], 'external_baseline_ms': timing['flashlib_ms'], 'external_baseline_ref': 'historical_same_shape'})
    if portfolio_id in PORTFOLIOS_WITH_Q32_TAIL143 and _eligible_q32tail143(inputs):
        timing = SEED_TIMING_ROWS[label]
        selected_seed, selected_entrypoint, guard_id = _q32tail143_route_metadata(inputs)
        mod = _q32tail143_module_for_inputs(inputs)
        split_count = mod.K32_Q32TAIL143_LOW_SPLIT_COUNT if selected_seed == SEED_Q32TAIL143_LOW_ID else mod.K32_Q32TAIL143_SPLIT_COUNT
        return _normalize_route_row({'shape_key': label, 'selected_route': mod.route_for_contract_inputs(inputs), 'selected_entrypoint': selected_entrypoint, 'selected_seed': selected_seed, 'expected_seed': selected_seed, 'route_kind': 'specialized', 'route_source': 'shape-specific-seed', 'guard_id': guard_id, 'guard_condition': ''.join(['exact BF16 non-build B=1 Q=32 D=128 K=32 M=', format(int(inputs.get('M', -1)), '')]), 'classification': 'seed-consumed' if float(timing['ratio_vs_flashlib']) >= SPEEDUP_FLOOR else 'kernel-slow', 'split_count': split_count, 'rows_per_merge_cta': mod.q32exact.rows4.K32_ROWS4_ROWS_PER_CTA, 'producer_topology': 'f590_rowld2_ROW_16x256B_two_compute_warp_exact_stage1_split143', 'merge_topology': 'rows4 warp-row split-list merge', 'replaced_route': replaced_route, 'parent_v11_route': replaced_route, 'base_dispatcher_route': base_row.get('selected_route'), 'shape_specific_kernel_ms': timing['kernel_ms'], 'speedup_vs_external_baseline': timing['ratio_vs_flashlib'], 'external_baseline_ms': timing['flashlib_ms'], 'external_baseline_ref': 'historical_same_shape'})
    if portfolio_id in PORTFOLIOS_WITH_Q32_TAIL and _eligible_q32tail(inputs):
        timing = SEED_TIMING_ROWS[label]
        return _normalize_route_row({'shape_key': label, 'selected_route': _q32tail().route_for_contract_inputs(inputs), 'selected_entrypoint': Q32TAIL_ENTRYPOINT, 'selected_seed': SEED_Q32TAIL_ID, 'expected_seed': SEED_Q32TAIL_ID, 'route_kind': 'specialized', 'route_source': 'shape-specific-seed', 'guard_id': '6866_q32tail_m99999_m100001_k32_exact_guard', 'guard_condition': 'exact BF16 non-build B=1 Q=32 M in {99999,100001} D=128 K=32', 'classification': 'seed-consumed', 'split_count': _q32tail().K32_Q32TAIL_EXACT_SPLIT_COUNT, 'rows_per_merge_cta': _q32tail().q32exact.rows4.K32_ROWS4_ROWS_PER_CTA, 'producer_topology': 'f590_rowld2_ROW_16x256B_two_compute_warp_exact_stage1', 'merge_topology': 'rows4 warp-row split-list merge', 'replaced_route': replaced_route, 'base_dispatcher_route': base_row.get('selected_route'), 'shape_specific_kernel_ms': timing['kernel_ms'], 'speedup_vs_external_baseline': timing['ratio_vs_flashlib'], 'external_baseline_ms': timing['flashlib_ms'], 'external_baseline_ref': 'historical_same_shape'})
    if portfolio_id in PORTFOLIOS_WITH_EXPANDED_K32_Q31_Q33_Q40 and _eligible_q31tail_v2(inputs):
        timing = SEED_TIMING_ROWS[EXPANDED_Q31_M100000_K32]
        mod = _q31tail_v2()
        return _normalize_route_row({'shape_key': EXPANDED_Q31_M100000_K32, 'selected_route': mod.route_for_contract_inputs(inputs), 'selected_entrypoint': Q31TAIL_V2_ENTRYPOINT, 'selected_seed': SEED_Q31TAIL_V2_ID, 'expected_seed': SEED_Q31TAIL_V2_ID, 'route_kind': 'specialized', 'route_source': 'shape-specific-seed', 'guard_id': Q31TAIL_V2_GUARD_ID, 'guard_condition': 'exact BF16 non-build B=1 Q=31 M=100000 D=128 K=32', 'classification': 'seed-consumed', 'split_count': mod.K32_Q31_EXACT_SPLIT_COUNT, 'rows_per_merge_cta': mod.q32exact.rows4.K32_ROWS4_ROWS_PER_CTA, 'producer_topology': '0cb5_v2_q31_exact_rowld2_ROW_16x256B_tcgen05_tma', 'merge_topology': 'rows4 warp-row split-list merge', 'replaced_route': replaced_route, 'base_dispatcher_route': base_row.get('selected_route'), 'shape_specific_kernel_ms': timing['kernel_ms'], 'speedup_vs_external_baseline': timing['ratio_vs_flashlib'], 'external_baseline_ms': timing['flashlib_ms'], 'external_baseline_ref': 'historical_same_shape'})
    if portfolio_id in PORTFOLIOS_WITH_EXPANDED_K32_Q31_Q33_Q40 and _eligible_q33tile(inputs):
        timing = SEED_TIMING_ROWS[label]
        mod = _q33tile()
        return _normalize_route_row({'shape_key': label, 'selected_route': mod.route_for_contract_inputs(inputs), 'selected_entrypoint': Q33TILE_ENTRYPOINT, 'selected_seed': SEED_Q33TILE_ID, 'expected_seed': SEED_Q33TILE_ID, 'route_kind': 'specialized', 'route_source': 'shape-specific-seed', 'guard_id': Q33TILE_GUARD_ID, 'guard_condition': 'exact BF16 non-build B=1 Q in {33,40} M=100000 D=128 K=32', 'classification': 'seed-consumed', 'split_count': mod.K32_Q33TILE_SPLIT_COUNT, 'group_count': mod.K32_Q33TILE_GROUP_COUNT, 'producer_topology': 'c489_e5db_m64_rowld_ROW_16x256B_tcgen05_tma', 'merge_topology': 'e5db fused split merge', 'replaced_route': replaced_route, 'base_dispatcher_route': base_row.get('selected_route'), 'shape_specific_kernel_ms': timing['kernel_ms'], 'speedup_vs_external_baseline': timing['ratio_vs_flashlib'], 'external_baseline_ms': timing['flashlib_ms'], 'external_baseline_ref': 'historical_same_shape'})
    if portfolio_id in PORTFOLIOS_WITH_Q48_K12 and _eligible_q48_k12(inputs):
        timing = SEED_TIMING_ROWS[EXPANDED_Q48_M75000_K12]
        mod = _q48_k12()
        return _normalize_route_row({'shape_key': EXPANDED_Q48_M75000_K12, 'selected_route': mod.route_for_contract_inputs(inputs), 'selected_entrypoint': Q48_K12_ENTRYPOINT, 'selected_seed': SEED_Q48_K12_ID, 'expected_seed': SEED_Q48_K12_ID, 'route_kind': 'specialized', 'route_source': 'shape-specific-seed', 'guard_id': Q48_K12_GUARD_ID, 'guard_condition': 'exact BF16 non-build B=1 Q=48 M=75000 D=128 K=12', 'classification': 'seed-consumed', 'split_count': mod.Q48_K12_SPLIT_COUNT, 'rows_per_merge_cta': mod.Q48_K12_ROWS_PER_MERGE_CTA, 'producer_topology': '2f22_m64n64_ROW_16x256B_tcgen05_tma_k12', 'merge_topology': 'four-row split-list merge', 'replaced_route': replaced_route, 'base_dispatcher_route': base_row.get('selected_route'), 'shape_specific_kernel_ms': timing['kernel_ms'], 'speedup_vs_external_baseline': timing['ratio_vs_flashlib'], 'external_baseline_ms': timing['flashlib_ms'], 'external_baseline_ref': 'historical_same_shape'})
    if portfolio_id in PORTFOLIOS_WITH_Q32_K31_C3D2 and _eligible_q32_k31_c3d2(inputs):
        timing = SEED_TIMING_ROWS[EXPANDED_Q32_M100000_K31]
        mod = _q32_k31_c3d2()
        return _normalize_route_row({'shape_key': EXPANDED_Q32_M100000_K31, 'selected_route': mod.route_for_contract_inputs(inputs), 'selected_entrypoint': Q32_K31_C3D2_ENTRYPOINT, 'selected_seed': SEED_Q32_K31_C3D2_ID, 'expected_seed': SEED_Q32_K31_C3D2_ID, 'route_kind': 'specialized', 'route_source': 'shape-specific-seed', 'guard_id': Q32_K31_C3D2_GUARD_ID, 'guard_condition': 'exact BF16 non-build B=1 Q=32 M=100000 D=128 K=31', 'classification': 'seed-consumed', 'split_count': mod.Q32_K31_SPLIT_COUNT, 'rows_per_merge_cta': mod.Q32_K31_ROWS_PER_MERGE_CTA, 'partial_top_k': 31, 'output_top_k': 31, 'producer_topology': 'eaf7_q32_k31_ROW_16x256B_two_compute_warp_tcgen05_tma', 'merge_topology': 'four-row K31 split-list merge', 'replaced_route': replaced_route, 'base_dispatcher_route': base_row.get('selected_route'), 'shape_specific_kernel_ms': timing['kernel_ms'], 'speedup_vs_external_baseline': timing['ratio_vs_flashlib'], 'external_baseline_ms': timing['flashlib_ms'], 'external_baseline_ref': 'historical_same_shape'})
    if portfolio_id in PORTFOLIOS_WITH_Q32_LOWK_C3D2 and _eligible_q32_lowk_c3d2(inputs):
        timing = D665_Q32_LOWK_C3D2_TIMING_ROWS.get(label, SEED_TIMING_ROWS[label])
        mod = _q32_lowk_c3d2()
        ratio = timing['ratio_vs_flashlib']
        return _normalize_route_row({'shape_key': label, 'selected_route': mod.route_for_contract_inputs(inputs), 'selected_entrypoint': Q32_LOWK_C3D2_ENTRYPOINT, 'selected_seed': SEED_Q32_LOWK_C3D2_ID, 'expected_seed': SEED_Q32_LOWK_C3D2_ID, 'route_kind': 'specialized', 'route_source': 'shape-specific-seed', 'guard_id': Q32_LOWK_C3D2_GUARD_ID, 'guard_condition': 'exact BF16 non-build B=1 Q=32 M=100000 D=128 K=20', 'classification': 'seed-consumed' if ratio >= SPEEDUP_FLOOR else 'kernel-slow', 'split_count': mod.Q32_LOWK_SPLIT_COUNT, 'partial_top_k': int(inputs.get('K', -1)), 'output_top_k': int(inputs.get('K', -1)), 'producer_topology': 'c3d2_e5db_lowk_ROW_16x256B_tcgen05_tma', 'merge_topology': 'four-row low-K split-list merge', 'replaced_route': replaced_route, 'base_dispatcher_route': base_row.get('selected_route'), 'shape_specific_kernel_ms': timing['kernel_ms'], 'speedup_vs_external_baseline': ratio, 'external_baseline_ms': timing['flashlib_ms'], 'external_baseline_ref': 'historical_same_shape'})
    return None

def _route_trace_record(inputs: dict[str, Any], *, force_fallback: bool=False, portfolio_id: str=CANDIDATE_DEFAULT) -> dict[str, Any]:
    if force_fallback:
        row = _base_trace_record(inputs, force_fallback=True)
        label = str(inputs.get('label'))
        if label in CONSUMED_SEED_SHAPES:
            row['expected_seed'] = _expected_seed_for_label(label, portfolio_id=portfolio_id)
            row['guard_id'] = ''.join(['forced_fallback_a4ec_', format(label, '')])
            row['guard_condition'] = 'forced fallback disables a4ec specialized common-D seed guards'
            row['classification'] = 'guard-miss'
        return _normalize_route_row(row)
    specialized = _specialized_trace_record(inputs, portfolio_id=portfolio_id)
    if specialized is not None:
        return specialized
    return _base_trace_record(inputs)

def route_trace_for_contract_shapes(shape_labels=None, *, force_fallback: bool=False, portfolio_id: str=CANDIDATE_DEFAULT) -> list[dict[str, Any]]:
    selected = eval_mod.CANONICAL_SHAPES if shape_labels is None else _select_contract_shapes(shape_labels)
    return [_route_trace_record(_trace_inputs_for_shape(shape), force_fallback=force_fallback, portfolio_id=portfolio_id) for shape in selected]

def _expected_seed_for_label(label: str, *, portfolio_id: str=CANDIDATE_DEFAULT) -> str | None:
    if portfolio_id in PORTFOLIOS_WITH_Q128_K10_ROWLD_1BED:
        if label == RAG_Q128_M100000_K10:
            return SEED_Q128_K10_ROWLD_1BED_ID
        return _expected_seed_for_label(label, portfolio_id=CANDIDATE_RECT_D128_K20_S12WARP4)
    if label == EXPANDED_Q32_M99999_K32 and portfolio_id in PORTFOLIOS_WITH_Q32_TAIL143:
        return SEED_Q32TAIL143_LOW_ID
    if label == EXPANDED_Q32_M100001_K32 and portfolio_id in PORTFOLIOS_WITH_Q32_TAIL143:
        return SEED_Q32TAIL143_HIGH_ID
    if portfolio_id in PORTFOLIOS_WITH_RECT_D128_K20_S12WARP4:
        if label in RECT_D128_K20_S12WARP4_CONSUMED_SHAPES:
            return SEED_RECT_D128_K20_S12WARP4_ID
        return _expected_seed_for_label(label, portfolio_id=CANDIDATE_D128_K48_DD2B)
    if portfolio_id in PORTFOLIOS_WITH_D128_K48_DD2B:
        if label in D128_K48_DD2B_CONSUMED_SHAPES:
            return SEED_D128_K48_DD2B_ID
        return _expected_seed_for_label(label, portfolio_id=CANDIDATE_HIGHD_SEARCH_BE66)
    if portfolio_id in PORTFOLIOS_WITH_HIGHD_SEARCH_BE66:
        if label in HIGHD_SEARCH_BE66_CONSUMED_SHAPES:
            return SEED_HIGHD_SEARCH_BE66_ID
        return _expected_seed_for_label(label, portfolio_id=CANDIDATE_HIGHD_RAG_22E9)
    if portfolio_id in PORTFOLIOS_WITH_HIGHD_RAG_22E9:
        if label in HIGHD_RAG_22E9_CONSUMED_SHAPES:
            return SEED_HIGHD_RAG_22E9_ID
        return _expected_seed_for_label(label, portfolio_id=CANDIDATE_D256_K32_59FE)
    if portfolio_id in PORTFOLIOS_WITH_D256_K32_59FE:
        if label in D256_K32_59FE_CONSUMED_SHAPES:
            return SEED_D256_K32_59FE_ID
        return _expected_seed_for_label(label, portfolio_id=CANDIDATE_D256_Q128_59FE)
    if portfolio_id in PORTFOLIOS_WITH_D256_Q128_59FE:
        if label in D256_Q128_59FE_CONSUMED_SHAPES:
            return SEED_D256_Q128_59FE_ID
        return _expected_seed_for_label(label, portfolio_id=CANDIDATE_SELECTED_SYNTHESIS)
    if label == BUILD_D256:
        if portfolio_id == CANDIDATE_A4EC_BASE:
            return 'common_d256_q1024_56f3_v1'
        return 'common_d_56f3_build_d256_q1024_v1'
    if label == BUILD_D768:
        return 'common_d_56f3_build_highd_v1' if portfolio_id == CANDIDATE_F1D9_BUILD else 'common_d768_build_eeff_m64split_v1'
    if label in (BUILD_D1024, BUILD_D4096):
        return 'common_d_56f3_build_highd_v1'
    if label == D768_SEARCH and portfolio_id != CANDIDATE_A4EC_BASE:
        return 'common_d_eeff_search_d768_v1'
    if label == D768_RAG:
        return 'non128_frontier_4be7_d768fused_v1'
    if label in (RAG_D1024, RAG_D4096) and portfolio_id in PORTFOLIOS_WITH_HIGHD_RAG:
        return 'common_d_5e7f_rag_highd_v1'
    if label == SEARCH_D256 and portfolio_id in PORTFOLIOS_WITH_SEARCH_D256:
        return 'common_d_5e7f_search_d256_v1'
    if label == RAG_D64 and portfolio_id in PORTFOLIOS_WITH_RAG_D64_M128:
        return 'common_d_1438_rag_d64_m128_v1'
    if label == RAG_D64 and portfolio_id == CANDIDATE_D64_REPAIR:
        return 'common_d_5e7f_rag_d64_repair_v1'
    if label == RAG_D256 and portfolio_id in PORTFOLIOS_WITH_SEARCH_D256:
        return 'common_d_5e7f_rag_d64_d256_v1'
    if label in D256_Q4_E2DF_CONSUMED_SHAPES and portfolio_id in PORTFOLIOS_WITH_D256_Q4_E2DF:
        return SEED_D256_Q4_E2DF_ID
    if label in D256_K32_59FE_CONSUMED_SHAPES and portfolio_id in PORTFOLIOS_WITH_D256_K32_59FE:
        return SEED_D256_K32_59FE_ID
    if label == D64_Q4096 and portfolio_id in PORTFOLIOS_WITH_D64_Q4096_C271:
        return 'd64_q4096_c271_twostage_v1'
    if label in D64_TAIL_017A_CONSUMED_SHAPES and portfolio_id in PORTFOLIOS_WITH_D64_TAIL_017A:
        return SEED_D64_TAIL_017A_ID
    if label == BUILD_K13 and portfolio_id in PORTFOLIOS_WITH_K13_K48:
        return SEED_K13_ID
    if label in (BUILD_K48_Q2048, BUILD_K48_Q4096) and portfolio_id in PORTFOLIOS_WITH_K13_K48:
        return SEED_K48_ID
    if label in (RAG_Q128_M100000_K32, RAG_Q128_M131071_K32) and _uses_floor_seed_portfolio(portfolio_id):
        if portfolio_id == CANDIDATE_FLOOR_SEEDS_Q128_5698:
            return SEED_Q128_DUALM_ID
        if portfolio_id == CANDIDATE_FLOOR_SEEDS_Q128_681B:
            return SEED_Q128_S72R2_ID
        if label == RAG_Q128_M131071_K32:
            return SEED_Q128_DUALM_ID
        return SEED_Q128_S72R2_ID
    if label == RAG_Q128_M100000_K10 and portfolio_id in PORTFOLIOS_WITH_Q128_K10_WARPMERGE:
        return SEED_Q128_K10_WARPMERGE_ID
    if label == RAG_Q16_M250000_K32 and _uses_floor_seed_portfolio(portfolio_id):
        return SEED_Q16_M250_ID
    if label == BUILD_LARGE_SQUARE_K32 and portfolio_id in PORTFOLIOS_WITH_LARGE_SQUARE_K32_EFE4:
        return SEED_LARGE_SQUARE_K32_EFE4_ID
    if label == RAG_STREAM_K10 and portfolio_id in PORTFOLIOS_WITH_RAG_STREAM_K10_34DA:
        return SEED_RAG_STREAM_K10_34DA_ID
    if label == RAG_Q1_M524287_K10 and portfolio_id in PORTFOLIOS_WITH_Q1_M524_N128:
        return SEED_Q1_M524_N128_ID
    if label == RAG_Q32_M100000_K32 and portfolio_id in PORTFOLIOS_WITH_Q32_EXACT:
        return SEED_Q32_EXACT_ID
    if label in EXPANDED_Q32TAIL_CONSUMED_SHAPES and portfolio_id in PORTFOLIOS_WITH_Q32_TAIL:
        return SEED_Q32TAIL_ID
    if label == EXPANDED_Q31_M100000_K32 and portfolio_id in PORTFOLIOS_WITH_EXPANDED_K32_Q31_Q33_Q40:
        return SEED_Q31TAIL_V2_ID
    if label in (EXPANDED_Q33_M100000_K32, EXPANDED_Q40_M100000_K32) and portfolio_id in PORTFOLIOS_WITH_EXPANDED_K32_Q31_Q33_Q40:
        return SEED_Q33TILE_ID
    if label == EXPANDED_Q48_M75000_K12 and portfolio_id in PORTFOLIOS_WITH_Q48_K12:
        return SEED_Q48_K12_ID
    if label in EXPANDED_Q32_LOWK_C3D2_CONSUMED_SHAPES and portfolio_id in PORTFOLIOS_WITH_Q32_LOWK_C3D2:
        return SEED_Q32_LOWK_C3D2_ID
    if label in EXPANDED_Q32_K31_C3D2_CONSUMED_SHAPES and portfolio_id in PORTFOLIOS_WITH_Q32_K31_C3D2:
        return SEED_Q32_K31_C3D2_ID
    if label in EXPANDED_Q32_K31_C3D2_CONSUMED_SHAPES and portfolio_id in PORTFOLIOS_WITH_Q32_LOWK_C3D2:
        return SEED_Q32_LOWK_C3D2_ID
    return None

def _set_bench_backend(use_cupti: bool):
    previous = eval_mod.CONTRACT.bench.get('use_cupti', True)
    eval_mod.CONTRACT.bench['use_cupti'] = bool(use_cupti)
    return previous

def _run_with_timing_backend(*, use_cupti: bool, shape_labels=None, kernel_fn: Callable[[dict[str, Any]], Any]=candidate) -> dict[str, Any]:
    previous = _set_bench_backend(use_cupti)
    try:
        shapes = None if shape_labels is None else _select_contract_shapes(shape_labels)
        return evaluate_contract(shapes=shapes, correctness=True, benchmark=True, kernel_fn=kernel_fn)
    finally:
        eval_mod.CONTRACT.bench['use_cupti'] = previous

def _annotate_route_trace(route_trace: list[dict[str, Any]], candidate_report: dict[str, Any], baseline_report: dict[str, Any]) -> list[dict[str, Any]]:
    annotated = []
    for row in route_trace:
        label = str(row.get('shape_key'))
        out = dict(row)
        candidate_row = candidate_report.get('per_shape', {}).get(label, {})
        baseline_row = baseline_report.get('per_shape', {}).get(label, {})
        candidate_ms = candidate_row.get('kernel_ms')
        baseline_ms = baseline_row.get('kernel_ms')
        ratio = candidate_row.get('ratio_vs_flashlib')
        relative = baseline_ms / candidate_ms if candidate_ms and baseline_ms else None
        for key in ('B', 'Q', 'M', 'D', 'K', 'build', 'dtype'):
            if key in candidate_row:
                out[key] = candidate_row[key]
        out['dispatcher_kernel_ms'] = candidate_ms
        out['baseline_dispatcher_kernel_ms'] = baseline_ms
        out['external_baseline_ms'] = candidate_row.get('flashlib_ms')
        out['flashlib_ms'] = candidate_row.get('flashlib_ms')
        out['external_baseline_ref'] = 'same_session' if candidate_row.get('flashlib_ms') is not None else 'not_available'
        out['speedup_vs_external_baseline'] = ratio
        out['relative_speedup_vs_baseline'] = relative
        out['timing_backend'] = candidate_row.get('timing_backend') or baseline_row.get('timing_backend')
        benchmark_unmeasured = candidate_row.get('benchmark_skipped_reason') is not None or (candidate_row.get('measurement_comparable') is False and candidate_ms is None)
        if candidate_row.get('passed') is False:
            failures = candidate_row.get('correctness_failures') or candidate_row.get('diagnostics') or ()
            first_failure = failures[0] if isinstance(failures, list) and failures else {}
            if isinstance(first_failure, dict):
                out['failure_kind'] = first_failure.get('failure_kind')
                out['failure_message'] = first_failure.get('message')
                out['failure_exception_type'] = first_failure.get('exception_type')
            out['classification'] = 'benchmark-path-mismatch'
        elif benchmark_unmeasured:
            out['classification'] = 'unmeasured'
            out['unmeasured_reason'] = candidate_row.get('benchmark_skipped_reason') or candidate_row.get('measurement_invalid_reason')
        elif out.get('route_kind') == 'specialized':
            if out.get('expected_seed') is not None and out.get('selected_seed') != out.get('expected_seed'):
                out['classification'] = 'guard-miss'
            elif isinstance(ratio, (float, int)) and float(ratio) < SPEEDUP_FLOOR:
                out['classification'] = 'kernel-slow'
            elif label in CONSUMED_SEED_SHAPES and out.get('selected_seed') == out.get('expected_seed'):
                out['classification'] = 'seed-consumed'
            else:
                out['classification'] = 'route-ok'
        elif isinstance(ratio, (float, int)) and float(ratio) < SPEEDUP_FLOOR:
            out['classification'] = 'fallback-slow'
        annotated.append(_normalize_route_row(out))
    return annotated

def _candidate_floor_record(*, label: str, candidate_id: str, selected_seed: str, guard_id: str, selected_route: str, selected_entrypoint: str, timing: dict[str, Any]) -> dict[str, Any]:
    return {'candidate_id': candidate_id, 'candidate_kernel_ms': timing['kernel_ms'], 'flashlib_ms': timing['flashlib_ms'], 'speedup_vs_flashlib': timing['ratio_vs_flashlib'], 'selected_route': selected_route, 'selected_seed': selected_seed, 'selected_entrypoint': selected_entrypoint, 'guard_id': guard_id, 'timing_backend': timing['timing_backend'], 'baseline_ref_scope': 'historical_exact_shape', 'source_payload': timing['source_payload'], 'shape_key': label}

def _historical_seed_candidates_for_label(label: str) -> list[dict[str, Any]]:
    inputs = _inputs_for_label(label)
    if label == BUILD_K13:
        return [_candidate_floor_record(label=label, candidate_id='historical_a162_k13', selected_seed=SEED_K13_A162_ID, guard_id='a162_2c1c_q4096_k13_exact_guard', selected_route=seed_k13.route_for_contract_inputs(inputs), selected_entrypoint=K13_ENTRYPOINT, timing=K13_A162_SOURCE_TIMING), _candidate_floor_record(label=label, candidate_id='historical_a162_k13_best_timing', selected_seed=SEED_K13_A162_ID, guard_id='a162_2c1c_q4096_k13_exact_guard', selected_route=seed_k13.route_for_contract_inputs(inputs), selected_entrypoint=K13_ENTRYPOINT, timing=K13_A162_BEST_TIMING)]
    if label in (RAG_Q128_M100000_K32, RAG_Q128_M131071_K32):
        return [_candidate_floor_record(label=label, candidate_id=CANDIDATE_FLOOR_SEEDS_Q128_5698, selected_seed=SEED_Q128_DUALM_ID, guard_id='a162_q128_dualm_k32_rowld_s72_warp1_exact_guard', selected_route=_q128_dualm().route_for_contract_inputs(inputs), selected_entrypoint=Q128_DUALM_ENTRYPOINT, timing=Q128_DUALM_TIMING_ROWS[label]), _candidate_floor_record(label=label, candidate_id=CANDIDATE_FLOOR_SEEDS_Q128_681B, selected_seed=SEED_Q128_S72R2_ID, guard_id='a162_q128_stream_k32_s72r2_exact_guard', selected_route=_q128_s72r2().route_for_contract_inputs(inputs), selected_entrypoint=Q128_S72R2_ENTRYPOINT, timing=Q128_S72R2_TIMING_ROWS[label])]
    if label == RAG_Q128_M100000_K10:
        return [_candidate_floor_record(label=label, candidate_id='weave-evolve-knn-build-41bb_exact_q128_k10_rowld_1bed', selected_seed=SEED_Q128_K10_ROWLD_1BED_ID, guard_id=Q128_K10_ROWLD_1BED_GUARD_ID, selected_route=_q128_k10_rowld_1bed().route_for_contract_inputs(inputs), selected_entrypoint=Q128_K10_ROWLD_1BED_ENTRYPOINT, timing=Q128_K10_ROWLD_1BED_TIMING), _candidate_floor_record(label=label, candidate_id='weave-evolve-knn-build-4ce8_exact_q128_k10_warpmerge', selected_seed=SEED_Q128_K10_WARPMERGE_ID, guard_id='4ce8_df0f_q128_m100000_k10_s74_warpmerge_exact_guard', selected_route=q128_k10_warpmerge.ROUTE_WARPMERGE, selected_entrypoint=Q128_K10_WARPMERGE_ENTRYPOINT, timing=Q128_K10_WARPMERGE_TIMING)]
    if label == RAG_Q1_M524287_K10:
        inputs = _inputs_for_label(label)
        return [_candidate_floor_record(label=label, candidate_id='weave-evolve-knn-build-32b9_exact_q1_m524_n128', selected_seed=SEED_Q1_M524_N128_ID, guard_id='32b9_ea43_q1_m524287_n128_exact_guard', selected_route=q1_m524_n128.route_for_contract_inputs(inputs), selected_entrypoint=Q1_M524_N128_ENTRYPOINT, timing=SEED_TIMING_ROWS[label])]
    if label in D64_TAIL_017A_CONSUMED_SHAPES:
        return [_candidate_floor_record(label=label, candidate_id=SEED_D64_TAIL_017A_ID, selected_seed=SEED_D64_TAIL_017A_ID, guard_id=D64_TAIL_017A_GUARD_ID, selected_route=d64_tail_017a.route_for_contract_inputs(inputs), selected_entrypoint=D64_TAIL_017A_ENTRYPOINT, timing=SEED_TIMING_ROWS[label])]
    if label in D256_Q4_E2DF_CONSUMED_SHAPES:
        return [_candidate_floor_record(label=label, candidate_id=SEED_D256_Q4_E2DF_ID, selected_seed=SEED_D256_Q4_E2DF_ID, guard_id=D256_Q4_E2DF_GUARD_ID, selected_route=d256_q4_e2df.route_for_contract_inputs(inputs), selected_entrypoint=D256_Q4_E2DF_ENTRYPOINT, timing=SEED_TIMING_ROWS[label])]
    if label in D256_Q128_59FE_CONSUMED_SHAPES:
        return [_candidate_floor_record(label=label, candidate_id=SEED_D256_Q128_59FE_ID, selected_seed=SEED_D256_Q128_59FE_ID, guard_id=D256_Q128_59FE_GUARD_ID, selected_route=d256_q128_59fe.route_for_contract_inputs(inputs), selected_entrypoint=D256_Q128_59FE_ENTRYPOINT, timing=SEED_TIMING_ROWS[label])]
    if label in D256_K32_59FE_CONSUMED_SHAPES:
        return [_candidate_floor_record(label=label, candidate_id=SEED_D256_K32_59FE_ID, selected_seed=SEED_D256_K32_59FE_ID, guard_id=D256_K32_59FE_GUARD_ID, selected_route=d256_k32_59fe.route_for_contract_inputs(inputs), selected_entrypoint=D256_K32_59FE_ENTRYPOINT, timing=SEED_TIMING_ROWS[label])]
    if label in HIGHD_RAG_22E9_CONSUMED_SHAPES:
        return [_candidate_floor_record(label=label, candidate_id=SEED_HIGHD_RAG_22E9_ID, selected_seed=SEED_HIGHD_RAG_22E9_ID, guard_id=HIGHD_RAG_22E9_GUARD_ID, selected_route=highd_rag_22e9.route_for_contract_inputs(inputs), selected_entrypoint=HIGHD_RAG_22E9_ENTRYPOINT, timing=SEED_TIMING_ROWS[label])]
    if label in D128_K48_DD2B_CONSUMED_SHAPES:
        return [_candidate_floor_record(label=label, candidate_id=SEED_D128_K48_DD2B_ID, selected_seed=SEED_D128_K48_DD2B_ID, guard_id=D128_K48_DD2B_GUARD_ID, selected_route=d128_k48_dd2b.route_for_contract_inputs(inputs), selected_entrypoint=D128_K48_DD2B_ENTRYPOINT, timing=SEED_TIMING_ROWS[label])]
    if label in RECT_D128_K20_S12WARP4_CONSUMED_SHAPES:
        return [_candidate_floor_record(label=label, candidate_id=SEED_RECT_D128_K20_S12WARP4_ID, selected_seed=SEED_RECT_D128_K20_S12WARP4_ID, guard_id=RECT_D128_K20_S12WARP4_GUARD_ID, selected_route=rect_d128_k20_s12warp4.route_for_contract_inputs(inputs), selected_entrypoint=RECT_D128_K20_S12WARP4_ENTRYPOINT, timing=SEED_TIMING_ROWS[label])]
    if label == RAG_Q32_M100000_K32:
        return [_candidate_floor_record(label=label, candidate_id='weave-evolve-knn-build-12ac_exact_q32_k32_f653_rowld2', selected_seed=SEED_Q32_EXACT_ID, guard_id=Q32_EXACT_GUARD_ID, selected_route=q32exact.route_for_contract_inputs(inputs), selected_entrypoint=Q32_EXACT_ENTRYPOINT, timing=SEED_TIMING_ROWS[label])]
    if label == EXPANDED_Q31_M100000_K32:
        return [_candidate_floor_record(label=label, candidate_id='weave-evolve-knn-build-0cb5_exact_q31_k32_v2', selected_seed=SEED_Q31TAIL_V2_ID, guard_id=Q31TAIL_V2_GUARD_ID, selected_route=_q31tail_v2().route_for_contract_inputs(inputs), selected_entrypoint=Q31TAIL_V2_ENTRYPOINT, timing=SEED_TIMING_ROWS[label])]
    if label in (EXPANDED_Q33_M100000_K32, EXPANDED_Q40_M100000_K32):
        return [_candidate_floor_record(label=label, candidate_id='weave-evolve-knn-build-c489_exact_q33_q40_k32', selected_seed=SEED_Q33TILE_ID, guard_id=Q33TILE_GUARD_ID, selected_route=_q33tile().route_for_contract_inputs(inputs), selected_entrypoint=Q33TILE_ENTRYPOINT, timing=SEED_TIMING_ROWS[label])]
    if label == EXPANDED_Q48_M75000_K12:
        return [_candidate_floor_record(label=label, candidate_id='weave-evolve-knn-build-2f22_exact_q48_k12', selected_seed=SEED_Q48_K12_ID, guard_id=Q48_K12_GUARD_ID, selected_route=_q48_k12().route_for_contract_inputs(inputs), selected_entrypoint=Q48_K12_ENTRYPOINT, timing=SEED_TIMING_ROWS[label])]
    if label in EXPANDED_Q32_LOWK_C3D2_CONSUMED_SHAPES:
        return [_candidate_floor_record(label=label, candidate_id='weave-evolve-knn-build-4cc4_q32_lowk_c3d2', selected_seed=SEED_Q32_LOWK_C3D2_ID, guard_id=Q32_LOWK_C3D2_GUARD_ID, selected_route=_q32_lowk_c3d2().route_for_contract_inputs(inputs), selected_entrypoint=Q32_LOWK_C3D2_ENTRYPOINT, timing=SEED_TIMING_ROWS[label])]
    if label in EXPANDED_Q32_K31_C3D2_CONSUMED_SHAPES:
        return [_candidate_floor_record(label=label, candidate_id='weave-evolve-knn-build-eaf7_q32_k31_c3d2', selected_seed=SEED_Q32_K31_C3D2_ID, guard_id=Q32_K31_C3D2_GUARD_ID, selected_route=_q32_k31_c3d2().route_for_contract_inputs(inputs), selected_entrypoint=Q32_K31_C3D2_ENTRYPOINT, timing=SEED_TIMING_ROWS[label])]
    return []

def _selected_historical_timing(label: str, portfolio_id: str) -> dict[str, Any] | None:
    if portfolio_id in PORTFOLIOS_WITH_Q128_K10_ROWLD_1BED:
        if label == RAG_Q128_M100000_K10:
            return Q128_K10_ROWLD_1BED_TIMING
        return _selected_historical_timing(label, CANDIDATE_RECT_D128_K20_S12WARP4)
    if portfolio_id in PORTFOLIOS_WITH_RECT_D128_K20_S12WARP4:
        if label in RECT_D128_K20_S12WARP4_CONSUMED_SHAPES:
            return SEED_TIMING_ROWS[label]
        return _selected_historical_timing(label, CANDIDATE_D128_K48_DD2B)
    if portfolio_id in PORTFOLIOS_WITH_D128_K48_DD2B:
        if label in D128_K48_DD2B_CONSUMED_SHAPES:
            return SEED_TIMING_ROWS[label]
        return _selected_historical_timing(label, CANDIDATE_HIGHD_SEARCH_BE66)
    if portfolio_id in PORTFOLIOS_WITH_HIGHD_SEARCH_BE66:
        if label in HIGHD_SEARCH_BE66_CONSUMED_SHAPES:
            return SEED_TIMING_ROWS[label]
        return _selected_historical_timing(label, CANDIDATE_HIGHD_RAG_22E9)
    if portfolio_id in PORTFOLIOS_WITH_HIGHD_RAG_22E9:
        if label in HIGHD_RAG_22E9_CONSUMED_SHAPES:
            return SEED_TIMING_ROWS[label]
        return _selected_historical_timing(label, CANDIDATE_D256_K32_59FE)
    if portfolio_id in PORTFOLIOS_WITH_D256_K32_59FE:
        if label in D256_K32_59FE_CONSUMED_SHAPES:
            return SEED_TIMING_ROWS[label]
        return _selected_historical_timing(label, CANDIDATE_D256_Q128_59FE)
    if portfolio_id in PORTFOLIOS_WITH_D256_Q128_59FE:
        if label in D256_Q128_59FE_CONSUMED_SHAPES:
            return SEED_TIMING_ROWS[label]
        return _selected_historical_timing(label, CANDIDATE_SELECTED_SYNTHESIS)
    if label == RAG_Q128_M100000_K32 and _uses_floor_seed_portfolio(portfolio_id):
        if portfolio_id == CANDIDATE_FLOOR_SEEDS_Q128_5698:
            return Q128_DUALM_TIMING_ROWS[label]
        return Q128_S72R2_TIMING_ROWS[label]
    if label == RAG_Q128_M131071_K32 and _uses_floor_seed_portfolio(portfolio_id):
        if portfolio_id == CANDIDATE_FLOOR_SEEDS_Q128_681B:
            return Q128_S72R2_TIMING_ROWS[label]
        return Q128_DUALM_TIMING_ROWS[label]
    if label == RAG_Q16_M250000_K32 and _uses_floor_seed_portfolio(portfolio_id):
        return Q16_M250_TIMING
    if label == RAG_Q128_M100000_K10 and portfolio_id in PORTFOLIOS_WITH_Q128_K10_WARPMERGE:
        return Q128_K10_WARPMERGE_TIMING
    if label in EXPANDED_Q32_LOWK_C3D2_CONSUMED_SHAPES and portfolio_id in PORTFOLIOS_WITH_Q32_LOWK_C3D2:
        return D665_Q32_LOWK_C3D2_TIMING_ROWS[label]
    if label in EXPANDED_Q32_K31_C3D2_CONSUMED_SHAPES and portfolio_id in PORTFOLIOS_WITH_Q32_K31_C3D2:
        return SEED_TIMING_ROWS[label]
    if label in EXPANDED_Q32_K31_C3D2_CONSUMED_SHAPES and portfolio_id in PORTFOLIOS_WITH_Q32_LOWK_C3D2:
        return D665_Q32_LOWK_C3D2_TIMING_ROWS[label]
    return SEED_TIMING_ROWS.get(label)

def _seed_delta_matrix(candidate_report: dict[str, Any], baseline_report: dict[str, Any]) -> list[dict[str, Any]]:
    rows = []
    for label in CONSUMED_SEED_SHAPES:
        cand = candidate_report.get('per_shape', {}).get(label, {})
        base = baseline_report.get('per_shape', {}).get(label, {})
        candidate_ms = cand.get('kernel_ms')
        baseline_ms = base.get('kernel_ms')
        historical = _selected_historical_timing(label, CANDIDATE_DEFAULT) or {}
        alt_rows = [{'candidate_id': CANDIDATE_DEFAULT, 'metric_delta': candidate_ms - baseline_ms if candidate_ms and baseline_ms else None, 'candidate_kernel_ms': candidate_ms, 'flashlib_ms': cand.get('flashlib_ms'), 'speedup_vs_flashlib': cand.get('ratio_vs_flashlib'), 'floor_status': 'pass' if isinstance(cand.get('ratio_vs_flashlib'), (float, int)) and float(cand['ratio_vs_flashlib']) >= SPEEDUP_FLOOR else 'fail' if isinstance(cand.get('ratio_vs_flashlib'), (float, int)) else 'unknown', 'selected_route': _route_for_policy(_inputs_for_label(label), portfolio_id=CANDIDATE_DEFAULT), 'selected_seed': _expected_seed_for_label(label, portfolio_id=CANDIDATE_DEFAULT), 'guard_id': _route_trace_record(_inputs_for_label(label), portfolio_id=CANDIDATE_DEFAULT).get('guard_id'), 'timing_backend': cand.get('timing_backend'), 'baseline_ref_scope': 'same_session_full_dispatch'}]
        for row in _historical_seed_candidates_for_label(label):
            row = dict(row)
            kernel_ms = row.get('candidate_kernel_ms')
            row['metric_delta'] = kernel_ms - baseline_ms if kernel_ms is not None and baseline_ms is not None else None
            alt_rows.append(row)
        if label == BUILD_D768:
            alt_rows.append({'candidate_id': CANDIDATE_F1D9_BUILD, 'metric_delta': ALT_F1D9_D768_TIMING['kernel_ms'] - baseline_ms if baseline_ms else None, 'candidate_kernel_ms': ALT_F1D9_D768_TIMING['kernel_ms'], 'timing_backend': ALT_F1D9_D768_TIMING['timing_backend'], 'baseline_ref_scope': 'historical_exact_shape'})
        rows.append({'shape_key': label, 'baseline_route': _route_for_policy(_inputs_for_label(label), portfolio_id=CANDIDATE_D128_K48_DD2B), 'selected_seed': _expected_seed_for_label(label), 'candidate_deltas': alt_rows, 'historical_seed_kernel_ms': historical.get('kernel_ms'), 'historical_seed_speedup_vs_flashlib': historical.get('ratio_vs_flashlib'), 'historical_seed_payload': historical.get('source_payload')})
    return rows

def _flashlib_parity_ledger(route_trace: list[dict[str, Any]]) -> dict[str, Any]:
    below_1x = []
    below_floor = []
    for row in route_trace:
        ratio = row.get('speedup_vs_external_baseline')
        if not isinstance(ratio, (float, int)):
            continue
        record = {'shape_key': row.get('shape_key'), 'selected_route': row.get('selected_route'), 'selected_seed': row.get('selected_seed'), 'route_kind': row.get('route_kind'), 'speedup_vs_external_baseline': ratio, 'classification': row.get('classification')}
        if float(ratio) < 1.0:
            below_1x.append(record)
        if float(ratio) < SPEEDUP_FLOOR:
            below_floor.append(record)
    return {'baseline_ref_scope': 'same_session', 'baseline_payload': None, 'speedup_floor': SPEEDUP_FLOOR, 'rows_below_1x': below_1x, 'rows_below_floor': below_floor, 'omitted_reason': None}

def _benchmark_payload(candidate_report: dict[str, Any], baseline_report: dict[str, Any], *, use_cupti: bool, shape_labels) -> dict[str, Any]:
    route_trace = _annotate_route_trace(route_trace_for_contract_shapes(shape_labels), candidate_report, baseline_report)
    candidate_metric = candidate_report.get('summary', {}).get('primary_mean')
    baseline_metric = baseline_report.get('summary', {}).get('primary_mean')
    ledger = _flashlib_parity_ledger(route_trace)
    timing_backend = 'cupti' if use_cupti else 'cuda_event'
    classification_blockers = [{'shape_key': row.get('shape_key'), 'selected_route': row.get('selected_route'), 'selected_seed': row.get('selected_seed'), 'route_kind': row.get('route_kind'), 'classification': row.get('classification'), 'failure_message': row.get('failure_message'), 'unmeasured_reason': row.get('unmeasured_reason'), 'speedup_vs_external_baseline': row.get('speedup_vs_external_baseline')} for row in route_trace if row.get('classification') in {'benchmark-path-mismatch', 'fallback-slow', 'kernel-slow', 'unmeasured'}]
    return {'candidate_id': CANDIDATE_DEFAULT, 'tflops': candidate_metric, 'baseline_tflops': baseline_metric, 'metric_delta': candidate_metric - baseline_metric if candidate_metric and baseline_metric else None, 'all_correct': candidate_report.get('summary', {}).get('all_correct'), 'baseline_all_correct': baseline_report.get('summary', {}).get('all_correct'), 'performance_comparable': candidate_report.get('summary', {}).get('performance_comparable'), 'baseline_performance_comparable': baseline_report.get('summary', {}).get('performance_comparable'), 'invalid_performance_reason': candidate_report.get('summary', {}).get('invalid_performance_reason'), 'baseline_invalid_performance_reason': baseline_report.get('summary', {}).get('invalid_performance_reason'), 'measured_entrypoint': BENCHMARK_ENTRYPOINT, 'baseline_entrypoint': PRE_Q128_K10_ROWLD_1BED_BASE_ENTRYPOINT, 'measured_shape_labels': 'all_canonical' if shape_labels is None else tuple(shape_labels), 'selected_route_labels': CONSUMED_SEED_SHAPES, 'consumed_seed_labels': CONSUMED_SEED_SHAPES, 'seed_delta_matrix': _seed_delta_matrix(candidate_report, baseline_report), 'candidate_dispatchers': CANDIDATE_DISPATCHERS, 'selected_candidate_dispatcher': CANDIDATE_DEFAULT, 'route_modules': PRODUCTION_ROUTE_MODULES, 'route_trace': route_trace, 'forced_fallback_route_trace': route_trace_for_contract_shapes(shape_labels, force_fallback=True), 'route_trace_included': True, 'contract_summary': candidate_report.get('summary', {}), 'baseline_contract_summary': baseline_report.get('summary', {}), 'contract_performance': candidate_report.get('performance', {}), 'baseline_contract_performance': baseline_report.get('performance', {}), 'timing_backend': timing_backend, 'timing_backends': [timing_backend], 'timing_backend_requested': timing_backend, 'use_cupti': use_cupti, 'performance_coverage': 'partial' if ledger['rows_below_floor'] or classification_blockers else 'pass', 'coverage_only_routes': [row['shape_key'] for row in route_trace if row.get('route_kind') in ('coverage-only', 'fallback') or row.get('classification') == 'fallback-slow'], 'hot_bucket_blockers': ledger['rows_below_floor'] + classification_blockers, 'flashlib_parity_ledger': ledger, 'report': candidate_report, 'baseline_report': baseline_report}

def benchmark_knn_build_dispatch_v11_common_d_seed_portfolio_a4ec_v1(*, use_cupti: bool=True, shape_labels=None, baseline_report: dict[str, Any] | None=None) -> dict[str, Any]:
    candidate_report = _run_with_timing_backend(use_cupti=use_cupti, shape_labels=shape_labels, kernel_fn=candidate)
    if baseline_report is None:
        baseline_report = _run_with_timing_backend(use_cupti=use_cupti, shape_labels=shape_labels, kernel_fn=candidate_pre_q128_k10_rowld_1bed_baseline)
    return _benchmark_payload(candidate_report, baseline_report, use_cupti=use_cupti, shape_labels=shape_labels)

def _denominator_label(shape_labels) -> str:
    if shape_labels is None:
        return ''.join(['full', format(len(eval_mod.CANONICAL_SHAPES), '')])
    if tuple(shape_labels) == EXPANDED_Q32_GUARD_BOUNDARY_8_SHAPES:
        return 'expanded_q32_guard_boundary_8'
    return ''.join(['shape', format(len(tuple(shape_labels)), '')])

def write_trace_artifacts(artifact_dir: str | Path, *, shape_labels=None) -> dict[str, str]:
    out_dir = Path(artifact_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    denom = _denominator_label(shape_labels)
    route_trace = route_trace_for_contract_shapes(shape_labels)
    forced_trace = route_trace_for_contract_shapes(shape_labels, force_fallback=True)
    route_trace_path = out_dir / ''.join([format(denom, ''), '_route_trace_v11_common_d_seed_portfolio_a4ec_v1.json'])
    forced_trace_path = out_dir / ''.join([format(denom, ''), '_forced_fallback_trace_v11_common_d_seed_portfolio_a4ec_v1.json'])
    route_trace_path.write_text(json.dumps(route_trace, indent=2, sort_keys=True) + '\n', encoding='utf-8')
    forced_trace_path.write_text(json.dumps(forced_trace, indent=2, sort_keys=True) + '\n', encoding='utf-8')
    return {'route_trace': str(route_trace_path), 'forced_fallback_trace': str(forced_trace_path)}

def write_benchmark_artifacts(artifact_dir: str | Path, *, use_cupti: bool=True, shape_labels=None) -> dict[str, str]:
    payload = benchmark_knn_build_dispatch_v11_common_d_seed_portfolio_a4ec_v1(use_cupti=use_cupti, shape_labels=shape_labels)
    out_dir = Path(artifact_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    denom = _denominator_label(shape_labels)
    candidate_path = out_dir / ''.join([format(denom, ''), '_dispatch_v11_common_d_seed_portfolio_a4ec_v1.json'])
    baseline_path = out_dir / ''.join([format(denom, ''), '_same_session_baseline_common_d_v11_for_seed_portfolio_a4ec_v1.json'])
    route_trace_path = out_dir / ''.join([format(denom, ''), '_route_trace_v11_common_d_seed_portfolio_a4ec_v1.json'])
    forced_trace_path = out_dir / ''.join([format(denom, ''), '_forced_fallback_trace_v11_common_d_seed_portfolio_a4ec_v1.json'])
    seed_matrix_path = out_dir / ''.join([format(denom, ''), '_seed_delta_matrix_v11_common_d_seed_portfolio_a4ec_v1.json'])
    candidate_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + '\n', encoding='utf-8')
    baseline_path.write_text(json.dumps({'candidate_id': CANDIDATE_RECT_D128_K20_S12WARP4, 'baseline_for_candidate_id': CANDIDATE_Q128_K10_ROWLD_1BED, 'measured_entrypoint': payload['baseline_entrypoint'], 'measured_shape_labels': payload['measured_shape_labels'], 'timing_backend': payload['timing_backend'], 'timing_backends': payload['timing_backends'], 'timing_backend_requested': payload['timing_backend_requested'], 'tflops': payload['baseline_tflops'], 'all_correct': payload['baseline_all_correct'], 'performance_comparable': payload['baseline_performance_comparable'], 'contract_summary': payload['baseline_contract_summary'], 'contract_performance': payload['baseline_contract_performance'], 'route_trace': route_trace_for_contract_shapes(shape_labels, portfolio_id=CANDIDATE_RECT_D128_K20_S12WARP4), 'route_trace_included': True, 'report': payload['baseline_report']}, indent=2, sort_keys=True) + '\n', encoding='utf-8')
    route_trace_path.write_text(json.dumps(payload['route_trace'], indent=2, sort_keys=True) + '\n', encoding='utf-8')
    forced_trace_path.write_text(json.dumps(payload['forced_fallback_route_trace'], indent=2, sort_keys=True) + '\n', encoding='utf-8')
    seed_matrix_path.write_text(json.dumps(payload['seed_delta_matrix'], indent=2, sort_keys=True) + '\n', encoding='utf-8')
    return {'candidate_payload': str(candidate_path), 'same_session_baseline_payload': str(baseline_path), 'route_trace': str(route_trace_path), 'forced_fallback_trace': str(forced_trace_path), 'seed_delta_matrix': str(seed_matrix_path)}
