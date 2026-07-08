"""Round-f8eb seed-bank dispatcher synthesis for exact BF16 kNN.

Minimum target architecture: sm_100a for the tcgen05/TMEM seed routes. This
dispatcher-synthesis wrapper starts from the current exported 6bc6 Weave
dispatcher and adds guarded seed-bank profiles for b08d or 49f8 Q128 routing,
the a7f3 exact high-Q route, the b72b exact ANN route, 5161's repaired
extended-K sweep route for the Q4096/M32768/K48 blind spot, and bbd5's exact
Q4096/M20000 K1/K2 low-K repair. It does not retune any seed schedule. Narrow
synthesis profiles keep the repaired extended-K and low-D routes while limiting
b08d to explicitly measured Q128 sub-buckets.
"""
from __future__ import annotations
from json import loads as _json_loads
from .._dispatch_runtime import _capture_cuTensorMapEncodeTiled, _decode_capture, _import_dispatch_module, _ir_proxy
from typing import Any
from . import knn_search_7ce1_plus_ann_split7_dispatch_0615_b72b_v1 as ann_b72b
from . import knn_search_dispatch0610_extk_sweep_176c_r3_lowkseed_v1 as extk_bbd5
from . import knn_search_dispatch0610_extk_sweep_176c_r2_v1 as extk_5161
from . import knn_search_highq_q4096_m65536_k10_fulltile_0615_a7f3_v1 as highq_a7f3
from . import knn_search_lowd_non128_tile_reduce_0615_7d36_v1 as lowd_7d36
from . import knn_search_q128_fulltile_midbucket_dispatch0610_49f8_v1 as q128_49f8
from . import knn_search_q128_midtail_current_dispatch0610_b08d_v1 as q128_b08d
from . import knn_search_r55_095b_q4096k64_rowflag_fusedcert_direct_dispatch_0615_6bc6_v1 as current_exported
THREADS = current_exported.THREADS
MERGE_THREADS = current_exported.MERGE_THREADS
BLOCK_Q = current_exported.BLOCK_Q
BLOCK_M = current_exported.BLOCK_M
D_STATIC = current_exported.D_STATIC
K_MAX = current_exported.K_MAX
SPLIT_M = current_exported.SPLIT_M
ir = _decode_capture(_json_loads('{"__ir__": "knn_search_lowq_row16_mma_partial_dispatch0610_r18_d14a_vec16_bguard_v1", "arg_keys": ["queries", "database", "partial_distances", "partial_indices", "B", "Q", "M", "split_m", "num_q_tiles", "total_m_tiles"], "cluster_dims": [1, 1, 1], "computed_smem_bytes": 96000, "constants": [["K_MAX_", 10]], "cta_group": 1, "threads": 512}'))
current_ir = _decode_capture(_json_loads('{"__ir__": "knn_search_lowq_row16_mma_partial_dispatch0610_r18_d14a_vec16_bguard_v1", "arg_keys": ["queries", "database", "partial_distances", "partial_indices", "B", "Q", "M", "split_m", "num_q_tiles", "total_m_tiles"], "cluster_dims": [1, 1, 1], "computed_smem_bytes": 96000, "constants": [["K_MAX_", 10]], "cta_group": 1, "threads": 512}'))
q128_b08d_ir = _decode_capture(_json_loads('{"__ir__": "knn_search_mma_split_partial_v1", "arg_keys": ["queries", "database", "partial_distances", "partial_indices", "B", "Q", "M", "split_m", "num_q_tiles", "total_m_tiles", "tiles_per_split"], "cluster_dims": [1, 1, 1], "computed_smem_bytes": 108800, "constants": [["K_MAX_", 10], ["EXPOSE_COL_COHORTS", 0], ["FULL_M_TILES", 0]], "cta_group": 1, "threads": 640}'))
q128_49f8_ir = _decode_capture(_json_loads('{"__ir__": "knn_search_mma_split_partial_v1", "arg_keys": ["queries", "database", "partial_distances", "partial_indices", "B", "Q", "M", "split_m", "num_q_tiles", "total_m_tiles", "tiles_per_split"], "cluster_dims": [1, 1, 1], "computed_smem_bytes": 108800, "constants": [["K_MAX_", 10], ["EXPOSE_COL_COHORTS", 0], ["FULL_M_TILES", 0]], "cta_group": 1, "threads": 640}'))
highq_a7f3_ir = _decode_capture(_json_loads('{"__ir__": "knn_search_mma_split_partial_v1", "arg_keys": ["queries", "database", "partial_distances", "partial_indices", "B", "Q", "M", "split_m", "num_q_tiles", "total_m_tiles", "tiles_per_split"], "cluster_dims": [1, 1, 1], "computed_smem_bytes": 108800, "constants": [["EXPOSE_COL_COHORTS", 0], ["FULL_M_TILES", 1]], "cta_group": 1, "threads": 640}'))
ann_b72b_ir = _decode_capture(_json_loads('{"__ir__": "knn_search_mma_split_partial_v1", "arg_keys": ["queries", "database", "partial_distances", "partial_indices", "B", "Q", "M", "split_m", "num_q_tiles", "total_m_tiles", "tiles_per_split"], "cluster_dims": [1, 1, 1], "computed_smem_bytes": 108800, "constants": [["K_MAX_", 10], ["EXPOSE_COL_COHORTS", 0], ["FULL_M_TILES", 0]], "cta_group": 1, "threads": 640}'))
extk_bbd5_ir = _decode_capture(_json_loads('{"__ir__": "knn_search_k1_top1_margin_qfull_partial_0614_r93_v1", "arg_keys": ["queries", "database", "partial_distances", "partial_indices", "B", "Q", "M", "split_m", "num_q_tiles", "total_m_tiles", "tiles_per_split"], "cluster_dims": [1, 1, 1], "computed_smem_bytes": 108800, "constants": [], "cta_group": 1, "threads": 640}'))
extk_5161_ir = _decode_capture(_json_loads('{"__ir__": "knn_search_k48_q4096split128_m32768_k48scratch_partial_0614_ddbc_q4096k48_v2", "arg_keys": ["queries", "database", "partial_distances", "partial_indices", "B", "Q", "M", "split_m", "num_q_tiles", "total_m_tiles", "tiles_per_split"], "cluster_dims": [1, 1, 1], "computed_smem_bytes": 165120, "constants": [["K_MAX_", 64]], "cta_group": 1, "threads": 512}'))
lowd_7d36_ir = _decode_capture(_json_loads('{"__ir__": "knn_search_lowd_non128_tile_reduce_partial_0615_7d36_v1", "arg_keys": ["queries", "database", "partial_distances", "partial_indices", "B", "Q", "M", "K", "num_m_tiles"], "cluster_dims": [1, 1, 1], "computed_smem_bytes": 10240, "constants": [["D_", 64], ["K_MAX_", 10], ["BLOCK_M_", 1280], ["NUM_ROW_WORKERS_", 128], ["SUBWARP_WIDTH_", 2], ["SUBWARPS_PER_WARP_", 16], ["CHUNKS_", 4], ["LOCAL_LIST_CAP_", 10]], "cta_group": 1, "threads": 256}'))
ROUTE_CURRENT_EXPORTED_6BC6 = 'current_exported_6bc6_dispatcher'
ROUTE_Q128_B08D = q128_b08d.ROUTE_Q128_MIDTAIL_E2EB
ROUTE_Q128_49F8 = 'round49f8_q128_fulltile_midbucket_candidate'
ROUTE_HIGHQ_A7F3 = highq_a7f3.ROUTE_A7F3_Q4096_M65536_K10_FULLTILE
ROUTE_ANN_B72B = ann_b72b.ROUTE_ANN_SPLIT7
ROUTE_EXTK_BBD5_LOWK = 'roundbbd5_q4096_lowk_repaired_extendedk_sweep'
ROUTE_EXTK_5161_SELECTED = 'round5161_selected_extendedk_sweep'
ROUTE_LOWD_7D36 = lowd_7d36.ROUTE_LOWD_NON128_TILE_REDUCE
PROFILE_CURRENT = 'current_exported_6bc6'
PROFILE_EXTK_REPAIR = 'f8eb_current_6bc6_plus_5161_extk_repair'
PROFILE_EXACT_REPAIR = 'f8eb_a7f3_b72b_5161_7d36_no_q128'
PROFILE_B08D_TAIL_EXTK = 'f8eb_q128_tail_b08d_291f_b72b_5161_extk'
PROFILE_B08D_CORE = 'f8eb_b08d_291f_b72b_no_extk'
PROFILE_HIGHQ_ANN_EXTK = 'f8eb_291f_b72b_5161_extk_no_q128'
PROFILE_B08D_M65536_EXTK = 'f8eb_b08d_m65536_291f_b72b_5161_extk'
PROFILE_B08D_EXTK = 'f8eb_b08d_291f_b72b_5161_extk'
PROFILE_B08D_BBD5_EXTK = 'f8eb_b08d_291f_b72b_bbd5_extk'
PROFILE_49F8_EXTK = 'f8eb_49f8_291f_b72b_5161_extk'
PROFILE_ALL = PROFILE_B08D_EXTK
CANDIDATE_PROFILES: dict[str, tuple[str, ...]] = {PROFILE_CURRENT: (), PROFILE_EXTK_REPAIR: ('extk_5161', 'lowd_7d36'), PROFILE_EXACT_REPAIR: ('highq_a7f3', 'ann_b72b', 'extk_5161', 'lowd_7d36'), PROFILE_B08D_TAIL_EXTK: ('highq_a7f3', 'ann_b72b', 'extk_5161', 'lowd_7d36', 'q128_b08d_tail'), PROFILE_B08D_CORE: ('q128_b08d', 'highq_a7f3', 'ann_b72b'), PROFILE_HIGHQ_ANN_EXTK: ('highq_a7f3', 'ann_b72b', 'extk_5161', 'lowd_7d36'), PROFILE_B08D_M65536_EXTK: ('highq_a7f3', 'ann_b72b', 'extk_5161', 'lowd_7d36', 'q128_b08d_m65536'), PROFILE_B08D_EXTK: ('q128_b08d', 'highq_a7f3', 'ann_b72b', 'extk_5161', 'lowd_7d36'), PROFILE_B08D_BBD5_EXTK: ('q128_b08d', 'highq_a7f3', 'ann_b72b', 'extk_bbd5', 'extk_5161', 'lowd_7d36'), PROFILE_49F8_EXTK: ('q128_49f8', 'highq_a7f3', 'ann_b72b', 'extk_5161', 'lowd_7d36')}
_Q128_B08D_ENTRY: dict[str, str] = {'shape_key': 'f8eb_b08d_q128_midtail_e2eb', 'guard': q128_b08d.SHAPE_DISPATCH_REGISTRY[0]['guard'], 'route': ROUTE_Q128_B08D, 'entrypoint': 'loom.examples.weave.knn_search_q128_midtail_current_dispatch0610_b08d_v1:launch_for_eval', 'source_round_doc': 'design_doc/active/weave_evolve_knn_search_round_1_b08d.md'}
_Q128_B08D_M65536_ENTRY: dict[str, str] = {'shape_key': 'f8eb_b08d_q128_exact_m65536_e2eb', 'guard': 'B == 1 and Q == 128 and M == 65536 and D == 128 and K <= 10 and tcgen05', 'route': ROUTE_Q128_B08D, 'entrypoint': 'loom.examples.weave.knn_search_q128_midtail_current_dispatch0610_b08d_v1:launch_for_eval', 'source_round_doc': 'design_doc/active/weave_evolve_knn_search_round_1_b08d.md'}
_Q128_B08D_TAIL_ENTRY: dict[str, str] = {'shape_key': 'f8eb_b08d_q128_tail_only_e2eb', 'guard': 'B == 1 and Q == 128 and M in {65535,65537} and D == 128 and K <= 10 and tcgen05', 'route': ROUTE_Q128_B08D, 'entrypoint': 'loom.examples.weave.knn_search_q128_midtail_current_dispatch0610_b08d_v1:launch_for_eval', 'source_round_doc': 'design_doc/active/weave_evolve_knn_search_round_1_b08d.md'}
_Q128_49F8_ENTRY: dict[str, str] = {'shape_key': 'f8eb_49f8_q128_fulltile_midbucket', 'guard': q128_49f8.SHAPE_DISPATCH_REGISTRY[0]['guard'], 'route': ROUTE_Q128_49F8, 'entrypoint': 'loom.examples.weave.knn_search_q128_fulltile_midbucket_dispatch0610_49f8_v1:launch_for_eval', 'source_round_doc': 'design_doc/active/weave_evolve_knn_search_round_1_49f8.md'}
_HIGHQ_A7F3_ENTRY: dict[str, str] = {'shape_key': 'f8eb_a7f3_blind_highq_q4096_m65536_k10', 'guard': highq_a7f3.SHAPE_DISPATCH_REGISTRY[0]['guard'], 'route': ROUTE_HIGHQ_A7F3, 'entrypoint': 'loom.examples.weave.knn_search_highq_q4096_m65536_k10_fulltile_0615_a7f3_v1:launch_for_eval', 'source_round_doc': 'design_doc/active/weave_evolve_knn_search_round_103_a7f3.md'}
_ANN_B72B_ENTRY: dict[str, str] = {'shape_key': 'f8eb_b72b_ann_q10000_m100000_split7', 'guard': ann_b72b.SHAPE_DISPATCH_REGISTRY[0]['guard'], 'route': ROUTE_ANN_B72B, 'entrypoint': 'loom.examples.weave.knn_search_7ce1_plus_ann_split7_dispatch_0615_b72b_v1:launch_for_eval', 'source_round_doc': 'design_doc/active/weave_evolve_knn_search_round_1_b72b_ann_split7.md'}
_EXTK_BBD5_ENTRY: dict[str, str] = {'shape_key': 'f8eb_bbd5_q4096_m20000_lowk_repaired_extendedk_sweep', 'guard': 'B == 1 and Q == 4096 and M == 20000 and D == 128 and K in {1,2}; consume weave-evolve-knn-search-bbd5 low-K repair before inherited K<=10 route', 'route': ROUTE_EXTK_BBD5_LOWK, 'entrypoint': 'loom.examples.weave.knn_search_dispatch0610_extk_sweep_176c_r3_lowkseed_v1:launch_for_eval', 'source_round_doc': 'design_doc/active/weave_evolve_knn_search_round_3_176c_extksweep_lowkseed.md'}
_EXTK_5161_ENTRY: dict[str, str] = {'shape_key': 'f8eb_5161_extendedk_sweep_exact_rows', 'guard': 'B == 1 and D == 128 and not force_fallback and ((Q == 128 and M == 131072 and K in {11,12,16,20,30,48,64}) or (Q == 4096 and M == 20000 and K == 64) or (Q == 4096 and M == 32768 and K == 48) or (K == 64 and (Q,M) in {(64,131072),(128,65536),(128,262144),(512,65536),(4096,32768)})); exact rows from weave-evolve-knn-search-5161', 'route': ROUTE_EXTK_5161_SELECTED, 'entrypoint': 'loom.examples.weave.knn_search_dispatch0610_extk_sweep_176c_r2_v1:launch_for_eval', 'source_round_doc': 'design_doc/active/weave_evolve_knn_search_round_2_176c_extksweep.md'}
_LOWD_7D36_ENTRY: dict[str, str] = {'shape_key': 'f8eb_7d36_lowd_non128_exact_rows', 'guard': lowd_7d36.SHAPE_DISPATCH_REGISTRY[0]['guard'], 'route': ROUTE_LOWD_7D36, 'entrypoint': 'loom.examples.weave.knn_search_lowd_non128_tile_reduce_0615_7d36_v1:launch_for_eval', 'source_round_doc': 'design_doc/active/weave_evolve_knn_search_round_20_7d36_lowd_non128.md'}
SHAPE_DISPATCH_REGISTRY: tuple[dict[str, str], ...] = (_HIGHQ_A7F3_ENTRY, _ANN_B72B_ENTRY, _EXTK_BBD5_ENTRY, _EXTK_5161_ENTRY, _LOWD_7D36_ENTRY, _Q128_B08D_M65536_ENTRY, _Q128_B08D_TAIL_ENTRY, _Q128_B08D_ENTRY, _Q128_49F8_ENTRY, *current_exported.SHAPE_DISPATCH_REGISTRY)

def _profile_overlays(profile: str) -> tuple[str, ...]:
    try:
        return CANDIDATE_PROFILES[profile]
    except KeyError as exc:
        raise ValueError(''.join(['unknown f8eb dispatcher profile: ', format(profile, '')])) from exc

def _forced_fallback(inputs: dict[str, Any]) -> bool:
    parent_forced = current_exported._forced_fallback(inputs) if hasattr(current_exported, '_forced_fallback') else False
    return bool(inputs.get('force_fallback', False)) or bool(parent_forced)

def _use_highq_a7f3(inputs: dict[str, Any]) -> bool:
    return not _forced_fallback(inputs) and highq_a7f3._use_a7f3_highq_fulltile(inputs)

def _use_ann_b72b(inputs: dict[str, Any]) -> bool:
    return not _forced_fallback(inputs) and ann_b72b._use_ann_split7(inputs)

def _use_extk_bbd5_lowk(inputs: dict[str, Any]) -> bool:
    return not _forced_fallback(inputs) and (extk_bbd5._use_q4096_k1_lowk(inputs) or extk_bbd5._use_q4096_k2_lowk(inputs))

def _use_extk_5161(inputs: dict[str, Any]) -> bool:
    if _forced_fallback(inputs):
        return False
    if int(inputs.get('B', 1)) != 1 or int(inputs['D']) != D_STATIC:
        return False
    q_rows = int(inputs['Q'])
    m_rows = int(inputs['M'])
    k = int(inputs['K'])
    if q_rows == 128 and m_rows == 131072 and (k in {11, 12, 16, 20, 30, 48, 64}):
        return True
    if q_rows == 4096 and m_rows == 20000 and (k == 64):
        return True
    if q_rows == 4096 and m_rows == 32768 and (k == 48):
        return True
    return k == 64 and (q_rows, m_rows) in {(64, 131072), (128, 65536), (128, 262144), (512, 65536), (4096, 32768)}

def _use_lowd_7d36(inputs: dict[str, Any]) -> bool:
    return not _forced_fallback(inputs) and lowd_7d36._use_lowd_non128_tile_reduce(inputs)

def _use_q128_b08d(inputs: dict[str, Any]) -> bool:
    return not _forced_fallback(inputs) and q128_b08d._use_q128_midtail_e2eb(inputs)

def _use_q128_b08d_m65536(inputs: dict[str, Any]) -> bool:
    return _use_q128_b08d(inputs) and int(inputs['M']) == 65536

def _use_q128_b08d_tail(inputs: dict[str, Any]) -> bool:
    return _use_q128_b08d(inputs) and int(inputs['M']) in {65535, 65537}

def _use_q128_49f8(inputs: dict[str, Any]) -> bool:
    return not _forced_fallback(inputs) and q128_49f8._use_q128_fulltile_midbucket(inputs)

def _base_route(inputs: dict[str, Any]) -> str:
    return current_exported.selected_route(inputs)

def selected_route_for_profile(inputs: dict[str, Any], profile: str) -> str:
    overlays = _profile_overlays(profile)
    if profile == PROFILE_CURRENT or _forced_fallback(inputs):
        return _base_route(inputs)
    if 'highq_a7f3' in overlays and _use_highq_a7f3(inputs):
        return ROUTE_HIGHQ_A7F3
    if 'ann_b72b' in overlays and _use_ann_b72b(inputs):
        return ROUTE_ANN_B72B
    if 'extk_bbd5' in overlays and _use_extk_bbd5_lowk(inputs):
        return extk_bbd5.selected_route(inputs)
    if 'extk_5161' in overlays and _use_extk_5161(inputs):
        return extk_5161.selected_route(inputs)
    if 'lowd_7d36' in overlays and _use_lowd_7d36(inputs):
        return lowd_7d36.selected_route(inputs)
    if 'q128_b08d_m65536' in overlays and _use_q128_b08d_m65536(inputs):
        return ROUTE_Q128_B08D
    if 'q128_b08d_tail' in overlays and _use_q128_b08d_tail(inputs):
        return ROUTE_Q128_B08D
    if 'q128_49f8' in overlays and _use_q128_49f8(inputs):
        return q128_49f8.selected_route(inputs)
    if 'q128_b08d' in overlays and _use_q128_b08d(inputs):
        return ROUTE_Q128_B08D
    return _base_route(inputs)

def selected_route(inputs: dict[str, Any]) -> str:
    return selected_route_for_profile(inputs, PROFILE_ALL)

def selected_route_name(inputs: dict[str, Any]) -> str:
    return selected_route(inputs)

def _guard_order(profile: str) -> list[str]:
    overlays = _profile_overlays(profile)
    order = ['force_fallback_metadata']
    if 'highq_a7f3' in overlays:
        order.append(str(_HIGHQ_A7F3_ENTRY['shape_key']))
    if 'ann_b72b' in overlays:
        order.append(str(_ANN_B72B_ENTRY['shape_key']))
    if 'extk_bbd5' in overlays:
        order.append(str(_EXTK_BBD5_ENTRY['shape_key']))
    if 'extk_5161' in overlays:
        order.append(str(_EXTK_5161_ENTRY['shape_key']))
    if 'lowd_7d36' in overlays:
        order.append(str(_LOWD_7D36_ENTRY['shape_key']))
    if 'q128_b08d_m65536' in overlays:
        order.append(str(_Q128_B08D_M65536_ENTRY['shape_key']))
    if 'q128_b08d_tail' in overlays:
        order.append(str(_Q128_B08D_TAIL_ENTRY['shape_key']))
    if 'q128_49f8' in overlays:
        order.append(str(_Q128_49F8_ENTRY['shape_key']))
    if 'q128_b08d' in overlays:
        order.append(str(_Q128_B08D_ENTRY['shape_key']))
    order.extend((str(entry['shape_key']) for entry in current_exported.SHAPE_DISPATCH_REGISTRY))
    return order

def _base_info(inputs: dict[str, Any]) -> dict[str, Any]:
    info = dict(current_exported.route_info(inputs))
    info.setdefault('route', _base_route(inputs))
    info.setdefault('selected_route', info['route'])
    info.setdefault('production_policy', 'weave_only')
    info.setdefault('external_fallback', None)
    return info

def _specialized_info(*, profile: str, route: str, inputs: dict[str, Any], entry: dict[str, str], coverage_class: str, source_task: str) -> dict[str, Any]:
    parent_info = _base_info(inputs)
    parent_route = parent_info.get('route') or parent_info.get('selected_route')
    return {'profile': profile, 'route': route, 'selected_route': route, 'selected_entrypoint': entry['entrypoint'], 'parent_route': parent_route, 'replaced_route': parent_route, 'route_kind': 'specialized', 'coverage_class': coverage_class, 'classification': 'seed_bank_overlay', 'coverage_only': False, 'production_policy': 'weave_only', 'external_fallback': None, 'guard_order': _guard_order(profile), 'forced_fallback': False, 'selected_guard': entry['guard'], 'fallback': ROUTE_CURRENT_EXPORTED_6BC6, 'missing_weave_route': False, 'source_task': source_task, 'source_round_doc': entry['source_round_doc']}

def _extk_5161_info(inputs: dict[str, Any], profile: str) -> dict[str, Any]:
    child = dict(extk_5161.route_info(inputs))
    route = str(child.get('route') or child.get('selected_route') or extk_5161.selected_route(inputs))
    parent_info = _base_info(inputs)
    parent_route = parent_info.get('route') or parent_info.get('selected_route')
    return {**child, 'profile': profile, 'route': route, 'selected_route': route, 'selected_entrypoint': child.get('selected_entrypoint') or _EXTK_5161_ENTRY['entrypoint'], 'parent_route': parent_route, 'replaced_route': parent_route, 'route_kind': 'specialized', 'coverage_class': child.get('coverage_class') or 'performance_route_5161_extended_k_sweep', 'classification': 'seed_bank_overlay', 'coverage_only': False, 'production_policy': 'weave_only', 'external_fallback': None, 'guard_order': _guard_order(profile), 'forced_fallback': False, 'selected_guard': child.get('selected_guard') or _EXTK_5161_ENTRY['guard'], 'fallback': ROUTE_CURRENT_EXPORTED_6BC6, 'missing_weave_route': False, 'source_task': 'weave-evolve-knn-search-5161', 'source_round_doc': _EXTK_5161_ENTRY['source_round_doc'], 'selected_5161_route': route}

def _extk_bbd5_info(inputs: dict[str, Any], profile: str) -> dict[str, Any]:
    child = dict(extk_bbd5.route_info(inputs))
    route = str(child.get('route') or child.get('selected_route') or extk_bbd5.selected_route(inputs))
    parent_route = selected_route_for_profile(inputs, PROFILE_B08D_EXTK)
    return {**child, 'profile': profile, 'route': route, 'selected_route': route, 'selected_entrypoint': child.get('selected_entrypoint') or _EXTK_BBD5_ENTRY['entrypoint'], 'parent_route': parent_route, 'replaced_route': parent_route, 'route_kind': 'specialized', 'coverage_class': child.get('coverage_class') or 'performance_route_bbd5_q4096_lowk', 'classification': 'seed_bank_overlay', 'coverage_only': False, 'production_policy': 'weave_only', 'external_fallback': None, 'guard_order': _guard_order(profile), 'forced_fallback': False, 'selected_guard': child.get('selected_guard') or _EXTK_BBD5_ENTRY['guard'], 'fallback': ROUTE_CURRENT_EXPORTED_6BC6, 'missing_weave_route': False, 'source_task': 'weave-evolve-knn-search-bbd5', 'source_round_doc': _EXTK_BBD5_ENTRY['source_round_doc'], 'selected_bbd5_route': route}

def _lowd_7d36_info(inputs: dict[str, Any], profile: str) -> dict[str, Any]:
    child = dict(lowd_7d36.route_info(inputs))
    route = str(child.get('route') or child.get('selected_route') or lowd_7d36.selected_route(inputs))
    parent_info = _base_info(inputs)
    parent_route = parent_info.get('route') or parent_info.get('selected_route')
    return {**child, 'profile': profile, 'route': route, 'selected_route': route, 'selected_entrypoint': child.get('selected_entrypoint') or _LOWD_7D36_ENTRY['entrypoint'], 'parent_route': parent_route, 'replaced_route': parent_route, 'route_kind': 'specialized', 'coverage_class': child.get('coverage_class') or 'performance_route_lowd_non128_tile_reduce', 'classification': 'seed_bank_overlay', 'coverage_only': False, 'production_policy': 'weave_only', 'external_fallback': None, 'guard_order': _guard_order(profile), 'forced_fallback': False, 'selected_guard': child.get('selected_guard') or _LOWD_7D36_ENTRY['guard'], 'fallback': ROUTE_CURRENT_EXPORTED_6BC6, 'missing_weave_route': False, 'source_task': 'weave-evolve-knn-search-7d36', 'source_round_doc': _LOWD_7D36_ENTRY['source_round_doc']}

def route_info_for_profile(inputs: dict[str, Any], profile: str) -> dict[str, Any]:
    route = selected_route_for_profile(inputs, profile)
    overlays = _profile_overlays(profile)
    if profile == PROFILE_CURRENT:
        info = _base_info(inputs)
        info['profile'] = profile
        info['guard_order'] = _guard_order(profile)
        return info
    if _forced_fallback(inputs):
        info = _base_info(inputs)
        info['profile'] = profile
        info['guard_order'] = _guard_order(profile)
        info['forced_fallback'] = True
        info['selected_guard'] = 'force_fallback metadata'
        info.setdefault('route_kind', 'fallback')
        return info
    if route == ROUTE_HIGHQ_A7F3:
        return _specialized_info(profile=profile, route=route, inputs=inputs, entry=_HIGHQ_A7F3_ENTRY, coverage_class='performance_route_a7f3_blind_highq_q4096_m65536_k10', source_task='weave-evolve-knn-search-291f')
    if route == ROUTE_ANN_B72B:
        return _specialized_info(profile=profile, route=route, inputs=inputs, entry=_ANN_B72B_ENTRY, coverage_class='performance_route_b72b_ann_q10000_m100000_k10', source_task='weave-evolve-knn-search-b72b')
    if 'extk_bbd5' in overlays and _use_extk_bbd5_lowk(inputs):
        return _extk_bbd5_info(inputs, profile)
    if 'extk_5161' in overlays and _use_extk_5161(inputs):
        return _extk_5161_info(inputs, profile)
    if 'lowd_7d36' in overlays and _use_lowd_7d36(inputs):
        return _lowd_7d36_info(inputs, profile)
    if route == ROUTE_Q128_B08D:
        if 'q128_b08d_m65536' in overlays and _use_q128_b08d_m65536(inputs):
            entry = _Q128_B08D_M65536_ENTRY
            coverage_class = 'q128_exact_m65536_b08d'
        elif 'q128_b08d_tail' in overlays and _use_q128_b08d_tail(inputs):
            entry = _Q128_B08D_TAIL_ENTRY
            coverage_class = 'q128_tail_only_b08d'
        else:
            entry = _Q128_B08D_ENTRY
            coverage_class = 'q128_small_mid_tail_b08d'
        return _specialized_info(profile=profile, route=route, inputs=inputs, entry=entry, coverage_class=coverage_class, source_task='weave-evolve-knn-search-b08d')
    if 'q128_fulltile_midbucket' in route:
        return _specialized_info(profile=profile, route=route, inputs=inputs, entry=_Q128_49F8_ENTRY, coverage_class='q128_fulltile_midbucket_49f8', source_task='weave-evolve-knn-search-49f8')
    info = _base_info(inputs)
    info['profile'] = profile
    info['route'] = route
    info['selected_route'] = route
    info['guard_order'] = _guard_order(profile)
    info['forced_fallback'] = bool(info.get('forced_fallback', False))
    info.setdefault('route_kind', 'general')
    info.setdefault('coverage_only', False)
    info.setdefault('fallback', None)
    return info

def route_info(inputs: dict[str, Any]) -> dict[str, Any]:
    return route_info_for_profile(inputs, PROFILE_ALL)

def route_trace_entry(label: str, inputs: dict[str, Any], profile: str=PROFILE_ALL) -> dict[str, Any]:
    return {'label': label, 'B': int(inputs['B']), 'Q': int(inputs['Q']), 'M': int(inputs['M']), 'D': int(inputs['D']), 'K': int(inputs['K']), 'self_search': bool(inputs.get('self_search', False)), **route_info_for_profile(inputs, profile)}

def launch_for_profile(inputs: dict[str, Any], profile: str) -> dict[str, Any]:
    overlays = _profile_overlays(profile)
    route = selected_route_for_profile(inputs, profile)
    if profile == PROFILE_CURRENT or _forced_fallback(inputs):
        return current_exported.launch_for_eval(inputs)
    if route == ROUTE_HIGHQ_A7F3:
        return highq_a7f3.launch_for_eval(inputs)
    if route == ROUTE_ANN_B72B:
        return ann_b72b.launch_for_eval(inputs)
    if 'extk_bbd5' in overlays and _use_extk_bbd5_lowk(inputs):
        return extk_bbd5.launch_for_eval(inputs)
    if 'extk_5161' in overlays and _use_extk_5161(inputs):
        return extk_5161.launch_for_eval(inputs)
    if 'lowd_7d36' in overlays and _use_lowd_7d36(inputs):
        return lowd_7d36.launch_for_eval(inputs)
    if 'q128_b08d_m65536' in overlays and _use_q128_b08d_m65536(inputs):
        return q128_b08d.launch_for_eval(inputs)
    if 'q128_b08d_tail' in overlays and _use_q128_b08d_tail(inputs):
        return q128_b08d.launch_for_eval(inputs)
    if route == ROUTE_Q128_B08D:
        return q128_b08d.launch_for_eval(inputs)
    if 'q128_fulltile_midbucket' in route:
        return q128_49f8.launch_for_eval(inputs)
    return current_exported.launch_for_eval(inputs)

def launch_current_for_eval(inputs: dict[str, Any]) -> dict[str, Any]:
    return launch_for_profile(inputs, PROFILE_CURRENT)

def launch_extk_repair_for_eval(inputs: dict[str, Any]) -> dict[str, Any]:
    return launch_for_profile(inputs, PROFILE_EXTK_REPAIR)

def launch_exact_repair_for_eval(inputs: dict[str, Any]) -> dict[str, Any]:
    return launch_for_profile(inputs, PROFILE_EXACT_REPAIR)

def launch_b08d_tail_extk_for_eval(inputs: dict[str, Any]) -> dict[str, Any]:
    return launch_for_profile(inputs, PROFILE_B08D_TAIL_EXTK)

def launch_b08d_core_for_eval(inputs: dict[str, Any]) -> dict[str, Any]:
    return launch_for_profile(inputs, PROFILE_B08D_CORE)

def launch_highq_ann_extk_for_eval(inputs: dict[str, Any]) -> dict[str, Any]:
    return launch_for_profile(inputs, PROFILE_HIGHQ_ANN_EXTK)

def launch_b08d_m65536_extk_for_eval(inputs: dict[str, Any]) -> dict[str, Any]:
    return launch_for_profile(inputs, PROFILE_B08D_M65536_EXTK)

def launch_b08d_extk_for_eval(inputs: dict[str, Any]) -> dict[str, Any]:
    return launch_for_profile(inputs, PROFILE_B08D_EXTK)

def launch_b08d_bbd5_extk_for_eval(inputs: dict[str, Any]) -> dict[str, Any]:
    return launch_for_profile(inputs, PROFILE_B08D_BBD5_EXTK)

def launch_49f8_extk_for_eval(inputs: dict[str, Any]) -> dict[str, Any]:
    return launch_for_profile(inputs, PROFILE_49F8_EXTK)

def launch_for_eval(inputs: dict[str, Any]) -> dict[str, Any]:
    return launch_b08d_extk_for_eval(inputs)
