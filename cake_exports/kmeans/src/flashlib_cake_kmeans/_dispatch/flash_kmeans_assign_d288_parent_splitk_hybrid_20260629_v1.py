"""Flash-KMeans Euclidean assignment D288 parent/Split-K hybrid seed.

Minimum architecture: sm_100a. This production-dispatchable seed composes the
validated exact-D288 tcgen05/TMEM parent for ``K <= 2048`` with the validated
CTA-per-(N tile, K tile) Split-K producer plus its in-timed reduction for
``K >= 4096``. It is not intended for sm_120a/sm_121a where ptxas rejects
tcgen05 instructions.
"""
from __future__ import annotations
from json import loads as _json_loads
from .._dispatch_runtime import _capture_cuTensorMapEncodeTiled, _decode_capture, _import_dispatch_module, _ir_proxy
from typing import Any
from . import flash_kmeans_assign_d288_exactd_a532_v1 as _parent
from . import flash_kmeans_assign_d288_splitk_cta_0438_v1 as _splitk
BLOCK_N = _parent.BLOCK_N
BLOCK_K = _parent.BLOCK_K
SUPPORTED_D = {288}
BF16_DTYPE_NAMES = _splitk.BF16_DTYPE_NAMES
ROUTE_ID = 'd288_parent_splitk_hybrid_20260629_v1'
SEED_ID = 'd288-parent-splitk-hybrid-20260629-v1'
PARENT_CHILD_ROUTE_ID = 'd288_exactd_a532_v1'
SPLITK_CHILD_ROUTE_ID = _splitk.ROUTE_ID
ir = _decode_capture(_json_loads('{"__ir__": "flash_kmeans_assign_d288_exactd_a532_v1", "arg_keys": ["x_tmap", "c_tmap", "x_sq", "c_sq", "out", "B", "N", "D", "K", "num_n_tiles", "K_tiles"], "cluster_dims": [1, 1, 1], "computed_smem_bytes": 26624, "constants": [], "cta_group": 1, "threads": 192}'))

def launch_for_eval(inputs: dict[str, Any]) -> dict[str, Any]:
    bsz, n_points, dim, n_clusters, dtype_name = _shape_fields(inputs)
    _validate_supported_shape(N=n_points, D=dim, K=n_clusters, dtype=dtype_name)
    if n_clusters <= 2048:
        child_outputs = _parent.launch_for_eval(inputs)
        child_route = PARENT_CHILD_ROUTE_ID
        child_seed = _parent.__name__.rsplit('.', maxsplit=1)[-1]
        selected_path = 'parent'
    elif n_clusters >= 4096:
        child_outputs = _splitk.launch_for_eval(inputs)
        child_route = SPLITK_CHILD_ROUTE_ID
        child_seed = _splitk.SEED_ID
        selected_path = 'splitk'
    else:
        raise ValueError(''.join(['flash_kmeans_assign_d288_parent_splitk_hybrid_20260629_v1 supports K <= 2048 through the parent or K >= 4096 through Split-K, got K=', format(n_clusters, '')]))
    normalized = _normalize_outputs(child_outputs, inputs)
    normalized['selected_route'] = ROUTE_ID
    normalized['route_trace'] = _route_trace(inputs, child_route=child_route, child_seed=child_seed, selected_path=selected_path, total_tiles=bsz * (n_points // BLOCK_N))
    return normalized

def _validate_supported_shape(*, N: int, D: int, K: int, dtype: Any) -> None:
    dtype_name = str(dtype).replace('torch.', '')
    if dtype_name not in BF16_DTYPE_NAMES:
        raise ValueError(''.join(['flash_kmeans_assign_d288_parent_splitk_hybrid_20260629_v1 requires bfloat16 input, got ', format(dtype, '')]))
    if D not in SUPPORTED_D:
        raise ValueError(''.join(['flash_kmeans_assign_d288_parent_splitk_hybrid_20260629_v1 requires D == 288, got ', format(D, '')]))
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
    raise TypeError("D288 parent/Split-K hybrid route must return cluster_ids or write inputs['out']")

def _route_trace(inputs: dict[str, Any], *, child_route: str, child_seed: str, selected_path: str, total_tiles: int) -> dict[str, Any]:
    return {'shape_key': _shape_key(inputs), 'selected_route': ROUTE_ID, 'selected_entrypoint': 'loom.examples.weave.flash_kmeans_assign_d288_parent_splitk_hybrid_20260629_v1:launch_for_eval', 'selected_seed': SEED_ID, 'expected_seed': None, 'route_kind': 'specialized', 'route_source': 'shape-specific-seed', 'guard_id': 'guard_d288_parent_splitk_hybrid_20260629_v1', 'guard_condition': 'dtype == bfloat16 and D == 288 and N % 128 == 0 and K % 256 == 0; parent child when K <= 2048, Split-K child with in-timed reduction when K >= 4096', 'classification': 'route-ok', 'dispatcher_kernel_ms': None, 'shape_specific_kernel_ms': None, 'relative_speedup_vs_baseline': None, 'child_route': child_route, 'child_seed': child_seed, 'selected_path': selected_path, 'total_tiles': total_tiles, 'reason': 'K-guarded D288 composition of exact-D parent and real CTA Split-K reduction paths'}

def _shape_fields(inputs: dict[str, Any]) -> tuple[int, int, int, int, str]:
    return (int(inputs['B']), int(inputs['N']), int(inputs['D']), int(inputs['K']), _dtype_name(inputs))

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
