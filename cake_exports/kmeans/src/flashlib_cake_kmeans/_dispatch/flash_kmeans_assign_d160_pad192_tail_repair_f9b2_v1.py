"""Flash-KMeans Euclidean assignment D144/D160/D176 pad-to-D192 seed.

Minimum architecture: sm_100a. This candidate zero-pads non-128D BF16 inputs
with a Weave pack kernel, then routes assignment to the existing D192
tcgen05/TMEM split-D seed. It is not intended for sm_120a/sm_121a where ptxas
rejects tcgen05 instructions.
"""
from __future__ import annotations
from json import loads as _json_loads
from .._dispatch_runtime import _capture_cuTensorMapEncodeTiled, _decode_capture, _import_dispatch_module, _ir_proxy
from functools import lru_cache
from typing import Any
from . import flash_kmeans_assign_cleanroom_tcgen05_d192_splitd_v1 as _d192
from .flash_kmeans_assign_stream import stream_cache_key
BLOCK_N = _d192.BLOCK_N
BLOCK_K = _d192.BLOCK_K
FEAT_D_PAD = 192
FEAT_D_PAD_VECS = FEAT_D_PAD // 8
SUPPORTED_D = {144, 160, 176}
PACK_THREADS = 256
PACK_GRID_CAP = 4096
_SCRATCH_CACHE: dict[tuple[int, ...], tuple[Any, Any]] = {}
_ROUTE_INPUT_CACHE: dict[tuple[int, ...], dict[str, Any]] = {}
flash_kmeans_assign_d160_pad192_pack_f9b2_v1 = _decode_capture(_json_loads('{"__ir__": "flash_kmeans_assign_d160_pad192_pack_f9b2_v1", "arg_keys": ["x", "centroids", "x_pad", "c_pad", "B", "N", "D", "K", "total_x_pad", "total_c_pad"], "cluster_dims": [1, 1, 1], "computed_smem_bytes": 0, "constants": [], "cta_group": 1, "threads": 256}'))
ir = _decode_capture(_json_loads('{"__ir__": "flash_kmeans_assign_d160_pad192_pack_f9b2_v1", "arg_keys": ["x", "centroids", "x_pad", "c_pad", "B", "N", "D", "K", "total_x_pad", "total_c_pad"], "cluster_dims": [1, 1, 1], "computed_smem_bytes": 0, "constants": [], "cta_group": 1, "threads": 256}'))
pack_ir = _decode_capture(_json_loads('{"__ir__": "flash_kmeans_assign_d160_pad192_pack_f9b2_v1", "arg_keys": ["x", "centroids", "x_pad", "c_pad", "B", "N", "D", "K", "total_x_pad", "total_c_pad"], "cluster_dims": [1, 1, 1], "computed_smem_bytes": 0, "constants": [], "cta_group": 1, "threads": 256}'))

def _cuda_include_dirs() -> list[str]:
    from .._dispatch_runtime import _cuda_include_dirs as _common_cuda_include_dirs
    return _common_cuda_include_dirs()

def _compiled_pack_kernel() -> tuple[bytes, str, int, int]:
    return _decode_capture(_json_loads('{"__tuple__": [{"__kernel_source__": "dispatch_kernel_0003"}, "kernel_flash_kmeans_assign_d160_pad192_pack_f9b2_v1", 0, 256]}'))

def launch_for_eval(inputs: dict[str, Any]) -> dict[str, Any]:
    bsz = int(inputs['B'])
    n_points = int(inputs['N'])
    dim = int(inputs['D'])
    n_clusters = int(inputs['K'])
    dtype = str(inputs.get('dtype', getattr(inputs['x'], 'dtype', 'bfloat16'))).replace('torch.', '')
    if dtype not in {'bfloat16', 'bf16'}:
        raise ValueError(''.join(['flash_kmeans_assign_d160_pad192_tail_repair_f9b2_v1 requires bfloat16 input, got ', format(dtype, '')]))
    if dim not in SUPPORTED_D:
        raise ValueError(''.join(['flash_kmeans_assign_d160_pad192_tail_repair_f9b2_v1 requires D in ', format(sorted(SUPPORTED_D), ''), ', got ', format(dim, '')]))
    if n_points % BLOCK_N != 0:
        raise ValueError(''.join(['N must be divisible by BLOCK_N=', format(BLOCK_N, ''), ', got ', format(n_points, '')]))
    if n_clusters % BLOCK_K != 0:
        raise ValueError(''.join(['K must be divisible by BLOCK_K=', format(BLOCK_K, ''), ', got ', format(n_clusters, '')]))
    x_pad, c_pad = _scratch_buffers(inputs)
    _launch_pack(inputs, x_pad, c_pad)
    d192_inputs = _route_inputs(inputs, x_pad, c_pad)
    outputs = _d192.launch_for_eval(d192_inputs)
    normalized = _normalize_outputs(outputs, inputs)
    route_kind = _route_kind(n_points=n_points, n_clusters=n_clusters)
    selected_route = ''.join(['d', format(dim, ''), '_pad192_', format(route_kind, ''), '_repeated_mma_f9b2_v1'])
    normalized['selected_route'] = selected_route
    normalized['route_trace'] = {'shape_key': _shape_key(inputs), 'selected_route': selected_route, 'selected_entrypoint': ''.join([format(__name__, ''), ':launch_for_eval']), 'selected_seed': ''.join(['d', format(dim, ''), '-pad192-', format(route_kind, ''), '-repeated-mma-f9b2-v1']), 'route_kind': 'specialized', 'route_source': 'shape-specific-seed', 'guard_id': ''.join(['guard_d', format(dim, ''), '_pad192_tail_repair_f9b2_v1']), 'guard_condition': ''.join(['dtype == bfloat16 and D == ', format(dim, ''), ' and N % ', format(BLOCK_N, ''), ' == 0 and K % ', format(BLOCK_K, ''), ' == 0']), 'classification': 'seed-probe', 'dispatcher_kernel_ms': None, 'residual_contract_regions': ['zero_pad_to_d192_pack'], 'reason': 'D144/D160/D176 tail-safe route pads to D192 before the tcgen05 score producer'}
    return normalized

def _scratch_buffers(inputs: dict[str, Any]) -> tuple[Any, Any]:
    import torch
    bsz = int(inputs['B'])
    n_points = int(inputs['N'])
    n_clusters = int(inputs['K'])
    dim = int(inputs['D'])
    key = stream_cache_key(inputs, int(inputs['x'].data_ptr()), int(inputs['centroids'].data_ptr()), bsz, n_points, n_clusters, dim, FEAT_D_PAD)
    cached = _SCRATCH_CACHE.get(key)
    if cached is not None:
        return cached
    x_pad = torch.empty((bsz, n_points, FEAT_D_PAD), dtype=inputs['x'].dtype, device=inputs['x'].device)
    c_pad = torch.empty((bsz, n_clusters, FEAT_D_PAD), dtype=inputs['centroids'].dtype, device=inputs['centroids'].device)
    _SCRATCH_CACHE[key] = (x_pad, c_pad)
    return (x_pad, c_pad)

def _launch_pack(inputs: dict[str, Any], x_pad: Any, c_pad: Any) -> None:
    from .._dispatch_runtime import CUDAKernel
    bsz = int(inputs['B'])
    n_points = int(inputs['N'])
    dim = int(inputs['D'])
    n_clusters = int(inputs['K'])
    total_x_pad = bsz * n_points * FEAT_D_PAD
    total_c_pad = bsz * n_clusters * FEAT_D_PAD
    work_items = max(total_x_pad, total_c_pad)
    grid_x = min((work_items + PACK_THREADS - 1) // PACK_THREADS, PACK_GRID_CAP)
    cubin, kernel_name, smem_bytes, threads = _compiled_pack_kernel()
    args = [inputs['x'], inputs['centroids'], x_pad, c_pad, bsz, n_points, dim, n_clusters, total_x_pad, total_c_pad]
    with CUDAKernel(cubin, kernel_name) as kernel:
        kernel.launch(grid=(grid_x, 1, 1), block=(threads, 1, 1), args=args, shared_mem=smem_bytes)

def _route_inputs(inputs: dict[str, Any], x_pad: Any, c_pad: Any) -> dict[str, Any]:
    key = stream_cache_key(inputs, int(x_pad.data_ptr()), int(c_pad.data_ptr()), int(inputs['x_sq'].data_ptr()), int(inputs['c_sq'].data_ptr()), int(inputs['out'].data_ptr()), int(inputs['B']), int(inputs['N']), int(inputs['K']), FEAT_D_PAD)
    cached = _ROUTE_INPUT_CACHE.get(key)
    if cached is not None:
        return cached
    route_inputs = {'label': inputs.get('label'), 'B': int(inputs['B']), 'N': int(inputs['N']), 'D': FEAT_D_PAD, 'K': int(inputs['K']), 'dtype': inputs.get('dtype', 'bfloat16'), 'x': x_pad, 'centroids': c_pad, 'x_sq': inputs['x_sq'], 'c_sq': inputs['c_sq'], 'out': inputs['out'], 'original_D': int(inputs['D'])}
    _ROUTE_INPUT_CACHE[key] = route_inputs
    return route_inputs

def _route_kind(*, n_points: int, n_clusters: int) -> str:
    if _d192._use_single_tile_path(n_points=n_points, n_clusters=n_clusters):
        return 'single'
    return 'paired'

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
    raise TypeError("flash_kmeans_assign D160 pad192 route must return cluster_ids or write inputs['out']")

def _shape_key(inputs: dict[str, Any]) -> str:
    label = inputs.get('label')
    if label:
        return str(label)
    return ''.join(['b', format(int(inputs['B']), ''), '_n', format(int(inputs['N']), ''), '_k', format(int(inputs['K']), ''), '_d', format(int(inputs['D']), '')])

def compile_and_launch_flash_kmeans_assign_pad192(B: int=1, N: int=2048, K: int=2048, D: int=160, *, benchmark: bool=False) -> dict[str, Any]:
    import torch
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA GPU required')
    torch.manual_seed(16005)
    x = torch.randn((B, N, D), dtype=torch.bfloat16, device='cuda').contiguous()
    centroids = torch.randn((B, K, D), dtype=torch.bfloat16, device='cuda').contiguous()
    x_sq = (x.float() ** 2).sum(-1).contiguous()
    c_sq = (centroids.float() ** 2).sum(-1).contiguous()
    out = torch.empty((B, N), dtype=torch.int32, device='cuda')
    inputs = {'label': ''.join(['manual_b', format(B, ''), '_n', format(N, ''), '_k', format(K, ''), '_d', format(D, '')]), 'B': B, 'N': N, 'D': D, 'K': K, 'dtype': 'bfloat16', 'x': x, 'centroids': centroids, 'x_sq': x_sq, 'c_sq': c_sq, 'out': out}
    launch_for_eval(inputs)
    ref_dist = x_sq.unsqueeze(-1) + c_sq.unsqueeze(1) - 2.0 * torch.einsum('bnd,bkd->bnk', x.float(), centroids.float())
    ref = ref_dist.clamp_min(0.0).argmin(dim=-1).to(torch.int32)
    result: dict[str, Any] = {'passed': bool(torch.equal(out, ref))}
    if benchmark:
        from .._dispatch_runtime import evaluate
        result['contract_eval'] = evaluate(launch_for_eval, shapes=[{'label': ''.join(['manual_b', format(B, ''), '_n', format(N, ''), '_k', format(K, ''), '_d', format(D, '')]), 'params': {'B': B, 'N': N, 'D': D, 'K': K, 'dtype': 'bfloat16', 'seed': 16005}}])
    return result
