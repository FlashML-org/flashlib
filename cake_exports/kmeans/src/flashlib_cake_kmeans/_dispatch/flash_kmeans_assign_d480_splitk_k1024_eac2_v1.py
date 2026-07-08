"""D480 K1024 Split-K activation experiment for Flash-KMeans assignment.

Minimum architecture: sm_100a.  This D480-only staging seed reuses the
validated tcgen05/TMEM D32x15 producer and Weave reducer, but exposes K=1024
rows with at most 32 N=64 tiles to the producer/reducer ABI.  The parent keeps
those rows on serial CTA ownership; this variant tests whether four K=256
partials amortize the reducer without changing dispatcher policy.
"""
from __future__ import annotations
from json import loads as _json_loads
from .._dispatch_runtime import _capture_cuTensorMapEncodeTiled, _decode_capture, _import_dispatch_module, _ir_proxy
from typing import Any
from . import flash_kmeans_assign_d480_splitk_producer_reducer_d32k256_v1 as _parent
BLOCK_N = _parent.BLOCK_N
MMA_BLOCK_K = _parent.MMA_BLOCK_K
FEAT_D = 480
ROUTE_ID = 'd480_splitk_k1024_eac2_v1'
SEED_ID = 'd480-splitk-k1024-eac2-v1'
ir = _decode_capture(_json_loads('{"__ir__": "flash_kmeans_assign_d480_splitk_partial_d32k256_v1", "arg_keys": ["x_tmap", "c_tmap", "c_sq", "partial_scores", "partial_indices", "B", "N", "D", "K", "num_n_tiles", "K_tiles", "K_slices"], "cluster_dims": [1, 1, 1], "computed_smem_bytes": 22528, "constants": [], "cta_group": 1, "threads": 192}'))

def launch_for_eval(inputs: dict[str, Any]) -> dict[str, Any]:
    """Run Split-K at K1024+ when the parent producer grid remains bounded."""
    bsz = int(inputs['B'])
    n_points = int(inputs['N'])
    dim = int(inputs['D'])
    n_clusters = int(inputs['K'])
    _parent._validate_shape(inputs, bsz=bsz, n_points=n_points, dim=dim, n_clusters=n_clusters)
    num_n_tiles = n_points // BLOCK_N
    k_tiles = n_clusters // MMA_BLOCK_K
    if not _use_k1024_splitk(dim=dim, num_n_tiles=num_n_tiles, k_tiles=k_tiles):
        return _parent.launch_for_eval(inputs)
    k_slices = k_tiles // _parent.SPLITK_GROUP_K_TILES
    _parent._launch_blockn64_splitk(inputs, bsz=bsz, n_points=n_points, dim=dim, n_clusters=n_clusters, num_n_tiles=num_n_tiles, k_tiles=k_tiles, k_slices=k_slices)
    total_work = bsz * num_n_tiles * k_slices
    trace = _parent._route_trace(inputs, total_work=total_work, grid=(min(total_work, _parent.SPLITK_GRID_CAP), 1, 1), k_slices=k_slices)
    trace.update({'selected_route': ROUTE_ID, 'selected_entrypoint': ''.join([format(__name__, ''), ':launch_for_eval']), 'selected_seed': SEED_ID, 'guard_id': 'guard_d480_splitk_k1024_eac2_v1', 'guard_condition': 'dtype == bfloat16 and D == 480 and N % 64 == 0 and K % 256 == 0 and N/64 <= 32 and K/256 >= 4', 'reason': 'D480 K1024+ bounded-point-tile rows use the tcgen05 partial/reducer path'})
    return {'cluster_ids': inputs['out'], 'selected_route': ROUTE_ID, 'route_trace': trace}

def _use_k1024_splitk(*, dim: int, num_n_tiles: int, k_tiles: int) -> bool:
    return dim == FEAT_D and num_n_tiles <= _parent.MAX_POINT_TILES and (k_tiles >= 4)
