"""Exact-D128 no-padding Split-K Flash-KMeans priority seed.

Minimum architecture: sm_100a.  This D=128-only seed reuses the validated
BLOCK_N=64 G1/R4 tcgen05/TMEM producer and four-lane reducer, but exposes every
256-centroid tile as a Split-K work item for the five priority K=4096/8192
rows.  It must not be used on sm_120a/sm_121a, where ptxas rejects tcgen05.
"""
from __future__ import annotations
from json import loads as _json_loads
from .._dispatch_runtime import _capture_cuTensorMapEncodeTiled, _decode_capture, _import_dispatch_module, _ir_proxy
from typing import Any
from . import flash_kmeans_assign_highd_splitk_blockn64_g1r4_streamdep_r63_v1 as _g1r4
ROUTE_ID = 'd128_splitk_priority_575c_v1'
SEED_ID = 'd128-no-padding-splitk-priority-575c-v1'
TARGET_D = 128
MIN_K_TILES = 16
_g1r4.SUPPORTED_DIMS.add(TARGET_D)

def _use_d128_priority_splitk(*, dim: int, num_n_tiles: int, k_tiles: int) -> bool:
    return dim == TARGET_D and num_n_tiles <= _g1r4.MAX_POINT_TILES and (k_tiles >= MIN_K_TILES)
_g1r4._use_blockn64_splitk = _use_d128_priority_splitk
ir = _decode_capture(_json_loads('{"__ir__": "flash_kmeans_assign_highd_splitk_partial_blockn64_g1r4_streamdep_r63_v1", "arg_keys": ["x_tmap", "c_tmap", "c_sq", "partial_scores", "partial_indices", "B", "N", "D", "K", "num_n_tiles", "K_tiles", "K_slices"], "cluster_dims": [1, 1, 1], "computed_smem_bytes": 43008, "constants": [], "cta_group": 1, "threads": 192}'))

def launch_for_eval(inputs: dict[str, Any]) -> dict[str, Any]:
    """Launch only the exact D128, no-padding priority Split-K schedule."""
    if int(inputs['B']) != 1:
        raise ValueError(''.join([format(ROUTE_ID, ''), ' requires B=1']))
    if int(inputs['D']) != TARGET_D:
        raise ValueError(''.join([format(ROUTE_ID, ''), ' requires D=', format(TARGET_D, ''), ', got ', format(inputs['D'], '')]))
    if int(inputs['N']) not in (512, 1024, 2048):
        raise ValueError(''.join([format(ROUTE_ID, ''), ' requires N in {512, 1024, 2048}']))
    if int(inputs['K']) not in (4096, 8192):
        raise ValueError(''.join([format(ROUTE_ID, ''), ' requires K in {4096, 8192}']))
    outputs = _g1r4.launch_for_eval(inputs)
    trace = dict(outputs.get('route_trace', {}))
    trace.update({'selected_route': ROUTE_ID, 'selected_entrypoint': 'loom.examples.weave.flash_kmeans_assign_d128_splitk_priority_575c_v1:launch_for_eval', 'selected_seed': SEED_ID, 'guard_id': 'guard_d128_no_padding_splitk_priority_575c_v1', 'guard_condition': 'B == 1 and D == 128 and N in [512,1024,2048] and K in [4096,8192]', 'reason': 'exact-D128 priority Split-K: one 256-K tile per G1 producer work item plus four-lane reducer'})
    return {'cluster_ids': outputs['cluster_ids'], 'selected_route': ROUTE_ID, 'route_trace': trace}
