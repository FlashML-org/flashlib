"""Flash-KMeans Euclidean assignment guarded micro-D seed for D=16/32.

Minimum architecture: sm_100a. This bucket seed combines two Weave-only
tcgen05/TMEM paths: direct SMEM staging for small K=512 rows where global
scratch packing is overhead-dominant, and the existing 6cd2 pack+TMA path for
large/high-K rows where centroid reuse makes TMA staging faster. It is not
intended for sm_120a/sm_121a where ptxas rejects tcgen05 instructions.
"""
from __future__ import annotations
from json import loads as _json_loads
from .._dispatch_runtime import _capture_cuTensorMapEncodeTiled, _decode_capture, _import_dispatch_module, _ir_proxy
from typing import Any
from . import flash_kmeans_assign_microdim_6cd2_v1 as _pack
from . import flash_kmeans_assign_microdim_direct_9c0d_v1 as _direct
BLOCK_N = _direct.BLOCK_N
BLOCK_K = _direct.BLOCK_K
SUPPORTED_D = _direct.SUPPORTED_D
BF16_DTYPE_NAMES = _direct.BF16_DTYPE_NAMES
ROUTE_ID = 'microdim_hybrid_9c0d_v1'
SEED_ID = 'microdim-hybrid-9c0d-v1'
DIRECT_CHILD_ROUTE_ID = _direct.ROUTE_ID
PACK_CHILD_ROUTE_ID = 'microdim_pack_6cd2_v1'
ir = _decode_capture(_json_loads('{"__ir__": "flash_kmeans_assign_microdim_direct_9c0d_v1", "arg_keys": ["x", "centroids", "x_sq", "c_sq", "out", "B", "N", "D", "K", "num_n_tiles", "K_tiles"], "cluster_dims": [1, 1, 1], "computed_smem_bytes": 51200, "constants": [], "cta_group": 1, "threads": 192}'))

def launch_for_eval(inputs: dict[str, Any]) -> dict[str, Any]:
    bsz, n_points, dim, n_clusters, dtype_name = _shape_fields(inputs)
    _validate_supported_shape(N=n_points, D=dim, K=n_clusters, dtype=dtype_name)
    if _use_direct_path(N=n_points, K=n_clusters):
        child_outputs = _direct.launch_for_eval(inputs)
        child_route = DIRECT_CHILD_ROUTE_ID
        child_seed = _direct.SEED_ID
    else:
        child_outputs = _pack.launch_for_eval(inputs)
        child_route = PACK_CHILD_ROUTE_ID
        child_seed = 'microdim-pack-6cd2-v1'
    normalized = _normalize_outputs(child_outputs, inputs)
    trace = _route_trace(inputs, child_route=child_route, child_seed=child_seed, direct_path=child_route == DIRECT_CHILD_ROUTE_ID, total_tiles=bsz * (n_points // BLOCK_N))
    normalized['selected_route'] = ROUTE_ID
    normalized['route_trace'] = trace
    return normalized

def _use_direct_path(*, N: int, K: int) -> bool:
    return K == 512 and N <= 2432

def _validate_supported_shape(*, N: int, D: int, K: int, dtype: Any) -> None:
    dtype_name = str(dtype).replace('torch.', '')
    if dtype_name not in BF16_DTYPE_NAMES:
        raise ValueError(''.join(['flash_kmeans_assign_microdim_hybrid_9c0d_v1 requires bfloat16 input, got ', format(dtype, '')]))
    if D not in SUPPORTED_D:
        raise ValueError(''.join(['flash_kmeans_assign_microdim_hybrid_9c0d_v1 requires D in ', format(sorted(SUPPORTED_D), ''), ', got ', format(D, '')]))
    if N % BLOCK_N != 0:
        raise ValueError(''.join(['N must be divisible by BLOCK_N=', format(BLOCK_N, ''), ', got ', format(N, '')]))
    if K % BLOCK_K != 0:
        raise ValueError(''.join(['K must be divisible by BLOCK_K=', format(BLOCK_K, ''), ', got ', format(K, '')]))

def _normalize_outputs(outputs: Any, inputs: dict[str, Any]) -> dict[str, Any]:
    if outputs is None:
        return {'cluster_ids': inputs['out']}
    if hasattr(outputs, 'shape'):
        return {'cluster_ids': outputs}
    if isinstance(outputs, dict):
        normalized = dict(outputs)
        if 'cluster_ids' not in normalized and 'out' in normalized:
            normalized['cluster_ids'] = normalized['out']
        if 'cluster_ids' in normalized:
            return normalized
    raise TypeError("flash_kmeans_assign microdim hybrid route must return cluster_ids or write inputs['out']")

def _route_trace(inputs: dict[str, Any], *, child_route: str, child_seed: str, direct_path: bool, total_tiles: int) -> dict[str, Any]:
    return {'shape_key': _shape_key(inputs), 'selected_route': ROUTE_ID, 'selected_entrypoint': 'loom.examples.weave.flash_kmeans_assign_microdim_hybrid_9c0d_v1:launch_for_eval', 'selected_seed': SEED_ID, 'expected_seed': None, 'route_kind': 'specialized', 'route_source': 'shape-specific-seed', 'guard_id': 'guard_microdim_hybrid_9c0d_v1', 'guard_condition': 'dtype == bfloat16 and D in [16, 32] and N % 128 == 0 and K % 256 == 0', 'classification': 'route-ok', 'dispatcher_kernel_ms': None, 'shape_specific_kernel_ms': None, 'relative_speedup_vs_baseline': None, 'child_route': child_route, 'child_seed': child_seed, 'direct_staging': direct_path, 'total_tiles': total_tiles, 'reason': 'K=512 short rows use direct SMEM staging; larger rows retain the 6cd2 pack+TMA path for centroid reuse'}

def _shape_fields(inputs: dict[str, Any]) -> tuple[int, int, int, int, str]:
    bsz = int(inputs['B'])
    n_points = int(inputs['N'])
    dim = int(inputs['D'])
    n_clusters = int(inputs['K'])
    dtype_name = _dtype_name(inputs)
    return (bsz, n_points, dim, n_clusters, dtype_name)

def _shape_key(inputs: dict[str, Any]) -> str:
    label = inputs.get('label')
    if label:
        return str(label)
    return ''.join(['b', format(int(inputs['B']), ''), '_n', format(int(inputs['N']), ''), '_k', format(int(inputs['K']), ''), '_d', format(int(inputs['D']), '')])

def _dtype_name(inputs: dict[str, Any]) -> str:
    dtype = inputs.get('dtype')
    if dtype is not None:
        return str(dtype).replace('torch.', '')
    x = inputs.get('x')
    if x is not None and hasattr(x, 'dtype'):
        return str(x.dtype).replace('torch.', '')
    return 'bfloat16'
