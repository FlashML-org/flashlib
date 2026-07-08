"""Exact-D352 Split-K Flash-KMeans staging seed.

Minimum architecture: sm_100a.  This D352-only seed reuses the validated
tcgen05/TMEM D32/K256 producer-reducer ABI with D=352 (eleven D32 chunks),
avoiding the incumbent D352-to-D384 packing kernels.  It is unsupported on
sm_120a/sm_121a because those targets reject tcgen05 instructions.
"""
from __future__ import annotations
from json import loads as _json_loads
from .._dispatch_runtime import _capture_cuTensorMapEncodeTiled, _decode_capture, _import_dispatch_module, _ir_proxy
from typing import Any
from . import flash_kmeans_assign_d480_splitk_producer_reducer_d32k256_v1 as _abi
from . import flash_kmeans_assign_gap_pad_v1 as _gap
BLOCK_N = _abi.BLOCK_N
MMA_BLOCK_K = _abi.MMA_BLOCK_K
FEAT_D = 352
MAX_POINT_TILES = _abi.MAX_POINT_TILES
ROUTE_ID = 'd352_exactd_splitk_c95c_v2'
SEED_ID = 'd352-exactd-splitk-c95c-v2'
ir = _decode_capture(_json_loads('{"__ir__": "flash_kmeans_assign_d480_splitk_partial_d32k256_v1", "arg_keys": ["x_tmap", "c_tmap", "c_sq", "partial_scores", "partial_indices", "B", "N", "D", "K", "num_n_tiles", "K_tiles", "K_slices"], "cluster_dims": [1, 1, 1], "computed_smem_bytes": 22528, "constants": [], "cta_group": 1, "threads": 192}'))

def launch_for_eval(inputs: dict[str, Any]) -> dict[str, Any]:
    bsz, n, d, k = (int(inputs[x]) for x in ('B', 'N', 'D', 'K'))
    dtype = str(inputs.get('dtype', getattr(inputs['x'], 'dtype', 'bfloat16'))).replace('torch.', '')
    if dtype not in {'bfloat16', 'bf16'} or d != FEAT_D or n % BLOCK_N or k % MMA_BLOCK_K:
        raise ValueError('D352 exact seed requires BF16 D=352, N%64=0, and K%256=0')
    ntiles, ktiles = (n // BLOCK_N, k // MMA_BLOCK_K)
    if ntiles > MAX_POINT_TILES or ktiles < 4:
        return _gap.launch_for_eval(inputs)
    _abi._launch_blockn64_splitk(inputs, bsz=bsz, n_points=n, dim=d, n_clusters=k, num_n_tiles=ntiles, k_tiles=ktiles, k_slices=ktiles)
    total_work = bsz * ntiles * ktiles
    trace = _abi._route_trace(inputs, total_work=total_work, grid=(min(total_work, _abi.SPLITK_GRID_CAP), 1, 1), k_slices=ktiles)
    trace.update({'selected_route': ROUTE_ID, 'selected_entrypoint': ''.join([format(__name__, ''), ':launch_for_eval']), 'selected_seed': SEED_ID, 'guard_id': 'guard_d352_exactd_splitk_c95c_v2', 'guard_condition': 'D == 352 and N/64 <= 32 and K/256 >= 4', 'reason': 'exact D352 D32-chunk producer removes D384 packing'})
    return {'cluster_ids': inputs['out'], 'selected_route': ROUTE_ID, 'route_trace': trace}
