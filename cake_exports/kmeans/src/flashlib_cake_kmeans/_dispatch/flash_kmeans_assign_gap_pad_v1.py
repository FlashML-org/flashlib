"""Flash-KMeans assignment gap-D pad-to-supported seed adapter.

Minimum architecture: sm_100a. This generated-variant route zero-pads legal
between-bucket and selected micro-D BF16 dimensions to the next existing Weave seed bucket, then
delegates to the unmodified tcgen05/TMEM seed for that padded dimension. It is
not intended for sm_120a/sm_121a where ptxas rejects tcgen05 instructions.
"""
from __future__ import annotations
from json import loads as _json_loads
from .._dispatch_runtime import _capture_cuTensorMapEncodeTiled, _decode_capture, _import_dispatch_module, _ir_proxy
from functools import lru_cache
from typing import Any
from . import flash_kmeans_assign_cleanroom_tcgen05_non128d_splitd_v1 as _non128d
from . import flash_kmeans_assign_cleanroom_tcgen05_v10 as _single
from . import flash_kmeans_assign_cleanroom_tcgen05_v15 as _paired
from . import flash_kmeans_assign_d64_direct_1p2gap_9f2a_v1 as _d64
from . import flash_kmeans_assign_highd_splitd_6fcf_v1 as _highd_splitd
from . import flash_kmeans_assign_highd_splitk_8de8_v1 as _highd_splitk
from .flash_kmeans_assign_stream import stream_cache_key
BLOCK_N = _single.BLOCK_N
BLOCK_K = _single.BLOCK_K
SUPPORTED_PAD: dict[int, int] = {16: 64, 32: 64, 48: 64, 112: 128, 224: 256, 288: 320, 352: 384, 416: 448, 480: 512}
SUPPORTED_D = set(SUPPORTED_PAD)
SEED_ID = 'gap-pad-to-supported-seed-v1'
ROUTE_ID = 'gap_pad_to_supported_seed_v1'
PACK_THREADS = 256
PACK_GRID_CAP = 4096
_SCRATCH_CACHE: dict[tuple[int, ...], tuple[Any, Any]] = {}
_ROUTE_INPUT_CACHE: dict[tuple[int, ...], dict[str, Any]] = {}
flash_kmeans_assign_gap_pad_pack_v1 = _decode_capture(_json_loads('{"__ir__": "flash_kmeans_assign_gap_pad_pack_v1", "arg_keys": ["x", "centroids", "x_pad", "c_pad", "B", "N", "D", "K", "D_PAD", "total_x_pad", "total_c_pad"], "cluster_dims": [1, 1, 1], "computed_smem_bytes": 0, "constants": [], "cta_group": 1, "threads": 256}'))
ir = _decode_capture(_json_loads('{"__ir__": "flash_kmeans_assign_gap_pad_pack_v1", "arg_keys": ["x", "centroids", "x_pad", "c_pad", "B", "N", "D", "K", "D_PAD", "total_x_pad", "total_c_pad"], "cluster_dims": [1, 1, 1], "computed_smem_bytes": 0, "constants": [], "cta_group": 1, "threads": 256}'))
pack_ir = _decode_capture(_json_loads('{"__ir__": "flash_kmeans_assign_gap_pad_pack_v1", "arg_keys": ["x", "centroids", "x_pad", "c_pad", "B", "N", "D", "K", "D_PAD", "total_x_pad", "total_c_pad"], "cluster_dims": [1, 1, 1], "computed_smem_bytes": 0, "constants": [], "cta_group": 1, "threads": 256}'))

def _cuda_include_dirs() -> list[str]:
    from .._dispatch_runtime import _cuda_include_dirs as _common_cuda_include_dirs
    return _common_cuda_include_dirs()

def _compiled_pack_kernel() -> tuple[bytes, str, int, int]:
    return _decode_capture(_json_loads('{"__tuple__": [{"__kernel_source__": "dispatch_kernel_0011"}, "kernel_flash_kmeans_assign_gap_pad_pack_v1", 0, 256]}'))

def launch_for_eval(inputs: dict[str, Any]) -> dict[str, Any]:
    bsz = int(inputs['B'])
    n_points = int(inputs['N'])
    dim = int(inputs['D'])
    n_clusters = int(inputs['K'])
    dtype = str(inputs.get('dtype', getattr(inputs['x'], 'dtype', 'bfloat16'))).replace('torch.', '')
    if dtype not in {'bfloat16', 'bf16'}:
        raise ValueError(''.join(['flash_kmeans_assign_gap_pad_v1 requires bfloat16 input, got ', format(dtype, '')]))
    if dim not in SUPPORTED_D:
        raise ValueError(''.join(['flash_kmeans_assign_gap_pad_v1 requires D in ', format(sorted(SUPPORTED_D), ''), ', got ', format(dim, '')]))
    if n_points % BLOCK_N != 0:
        raise ValueError(''.join(['N must be divisible by BLOCK_N=', format(BLOCK_N, ''), ', got ', format(n_points, '')]))
    if n_clusters % BLOCK_K != 0:
        raise ValueError(''.join(['K must be divisible by BLOCK_K=', format(BLOCK_K, ''), ', got ', format(n_clusters, '')]))
    d_pad = SUPPORTED_PAD[dim]
    x_pad, c_pad = _scratch_buffers(inputs, d_pad)
    _launch_pack(inputs, x_pad, c_pad, d_pad)
    route_inputs = _route_inputs(inputs, x_pad, c_pad, d_pad)
    route_fn, route_id, route_seed, route_entrypoint = _route_for_padded_dim(d_pad=d_pad, n_points=n_points, n_clusters=n_clusters)
    outputs = route_fn(route_inputs)
    normalized = _normalize_outputs(outputs, inputs)
    normalized['selected_route'] = ''.join(['d', format(dim, ''), '_pad', format(d_pad, ''), '_', format(route_id, '')])
    normalized['route_trace'] = {'shape_key': _shape_key(inputs), 'selected_route': normalized['selected_route'], 'selected_entrypoint': ''.join([format(__name__, ''), ':launch_for_eval']), 'selected_seed': ''.join(['d', format(dim, ''), '-pad', format(d_pad, ''), '-', format(route_seed, '')]), 'route_kind': 'specialized', 'route_source': 'generated-variant', 'guard_id': ''.join(['guard_d', format(dim, ''), '_pad', format(d_pad, ''), '_gap_pad_v1']), 'guard_condition': ''.join(['dtype == bfloat16 and D == ', format(dim, ''), ' and N % ', format(BLOCK_N, ''), ' == 0 and K % ', format(BLOCK_K, ''), ' == 0']), 'classification': 'route-ok', 'dispatcher_kernel_ms': None, 'residual_contract_regions': [''.join(['zero_pad_to_d', format(d_pad, ''), '_pack'])], 'reason': ''.join(['D', format(dim, ''), ' gap route pads to D', format(d_pad, ''), ' and delegates to ', format(route_entrypoint, '')])}
    return normalized

def _scratch_buffers(inputs: dict[str, Any], d_pad: int) -> tuple[Any, Any]:
    import torch
    bsz = int(inputs['B'])
    n_points = int(inputs['N'])
    n_clusters = int(inputs['K'])
    dim = int(inputs['D'])
    key = stream_cache_key(inputs, int(inputs['x'].data_ptr()), int(inputs['centroids'].data_ptr()), bsz, n_points, n_clusters, dim, d_pad)
    cached = _SCRATCH_CACHE.get(key)
    if cached is not None:
        return cached
    x_pad = torch.empty((bsz, n_points, d_pad), dtype=inputs['x'].dtype, device=inputs['x'].device)
    c_pad = torch.empty((bsz, n_clusters, d_pad), dtype=inputs['centroids'].dtype, device=inputs['centroids'].device)
    _SCRATCH_CACHE[key] = (x_pad, c_pad)
    return (x_pad, c_pad)

def _launch_pack(inputs: dict[str, Any], x_pad: Any, c_pad: Any, d_pad: int) -> None:
    from .._dispatch_runtime import CUDAKernel
    bsz = int(inputs['B'])
    n_points = int(inputs['N'])
    dim = int(inputs['D'])
    n_clusters = int(inputs['K'])
    total_x_pad = bsz * n_points * d_pad
    total_c_pad = bsz * n_clusters * d_pad
    work_items = max(total_x_pad, total_c_pad)
    grid_x = min((work_items + PACK_THREADS - 1) // PACK_THREADS, PACK_GRID_CAP)
    cubin, kernel_name, smem_bytes, threads = _compiled_pack_kernel()
    args = [inputs['x'], inputs['centroids'], x_pad, c_pad, bsz, n_points, dim, n_clusters, d_pad, total_x_pad, total_c_pad]
    with CUDAKernel(cubin, kernel_name) as kernel:
        kernel.launch(grid=(grid_x, 1, 1), block=(threads, 1, 1), args=args, shared_mem=smem_bytes)

def _route_inputs(inputs: dict[str, Any], x_pad: Any, c_pad: Any, d_pad: int) -> dict[str, Any]:
    key = stream_cache_key(inputs, int(x_pad.data_ptr()), int(c_pad.data_ptr()), int(inputs['x_sq'].data_ptr()), int(inputs['c_sq'].data_ptr()), int(inputs['out'].data_ptr()), int(inputs['B']), int(inputs['N']), int(inputs['K']), d_pad)
    cached = _ROUTE_INPUT_CACHE.get(key)
    if cached is not None:
        return cached
    route_inputs = {'label': inputs.get('label'), 'B': int(inputs['B']), 'N': int(inputs['N']), 'D': d_pad, 'K': int(inputs['K']), 'dtype': inputs.get('dtype', 'bfloat16'), 'x': x_pad, 'centroids': c_pad, 'x_sq': inputs['x_sq'], 'c_sq': inputs['c_sq'], 'out': inputs['out'], 'original_D': int(inputs['D'])}
    _ROUTE_INPUT_CACHE[key] = route_inputs
    return route_inputs

def _route_for_padded_dim(*, d_pad: int, n_points: int, n_clusters: int):
    num_n_tiles = n_points // BLOCK_N
    k_tiles = n_clusters // BLOCK_K
    if d_pad == 64:
        return (_d64.launch_for_eval, 'd64_direct_single64_1p2gap_9f2a_v1', 'd64-direct-single64-1p2gap-9f2a-v1', 'loom.examples.weave.flash_kmeans_assign_d64_direct_1p2gap_9f2a_v1:launch_for_eval')
    if d_pad == 128:
        if num_n_tiles <= 8 and k_tiles <= 2:
            return (_single.launch_for_eval, 'small_grid_single_tile_v10', 'small-grid-single-tile-v10', 'loom.examples.weave.flash_kmeans_assign_cleanroom_tcgen05_v10:launch_for_eval')
        if num_n_tiles % 2 == 0:
            return (_paired.launch_for_eval, 'paired_large_v15', 'paired-large-v15', 'loom.examples.weave.flash_kmeans_assign_cleanroom_tcgen05_v15:launch_for_eval')
        return (_single.launch_for_eval, 'aligned_weave_v10_fallback', 'small-grid-single-tile-v10', 'loom.examples.weave.flash_kmeans_assign_cleanroom_tcgen05_v10:launch_for_eval')
    if d_pad == 256:
        return (_non128d.launch_for_eval, 'd256_single_repeated_mma_v1', 'd256-single-repeated-mma-v1', 'loom.examples.weave.flash_kmeans_assign_cleanroom_tcgen05_non128d_splitd_v1:launch_for_eval')
    if d_pad in _highd_splitd.SUPPORTED_DIMS:
        if _use_highd_splitk(d_pad=d_pad, num_n_tiles=num_n_tiles, k_tiles=k_tiles):
            return (_highd_splitk.launch_for_eval, 'highd_splitk_8de8_v1', _highd_splitk.SEED_ID, 'loom.examples.weave.flash_kmeans_assign_highd_splitk_8de8_v1:launch_for_eval')
        return (_highd_splitd.launch_for_eval, 'highd_splitd_6fcf_v1', _highd_splitd.SEED_ID, 'loom.examples.weave.flash_kmeans_assign_highd_splitd_6fcf_v1:launch_for_eval')
    raise ValueError(''.join(['no padded route for D_PAD=', format(d_pad, '')]))

def _use_highd_splitk(*, d_pad: int, num_n_tiles: int, k_tiles: int) -> bool:
    if num_n_tiles > 16:
        return False
    return k_tiles >= 32 or (k_tiles >= _highd_splitk.SPLITK_MIN_K_TILES and d_pad >= 448)

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
    raise TypeError("flash_kmeans_assign gap pad route must return cluster_ids or write inputs['out']")

def _shape_key(inputs: dict[str, Any]) -> str:
    label = inputs.get('label')
    if label:
        return str(label)
    return ''.join(['b', format(int(inputs['B']), ''), '_n', format(int(inputs['N']), ''), '_k', format(int(inputs['K']), ''), '_d', format(int(inputs['D']), '')])

def compile_and_launch_flash_kmeans_assign_gap_pad(B: int=2, N: int=2048, K: int=512, D: int=112, *, benchmark: bool=False) -> dict[str, Any]:
    import torch
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA GPU required')
    torch.manual_seed(11205)
    x = torch.randn((B, N, D), dtype=torch.bfloat16, device='cuda').contiguous()
    centroids = torch.randn((B, K, D), dtype=torch.bfloat16, device='cuda').contiguous()
    x_sq = (x.float() ** 2).sum(-1).contiguous()
    c_sq = (centroids.float() ** 2).sum(-1).contiguous()
    out = torch.empty((B, N), dtype=torch.int32, device='cuda')
    inputs = {'label': ''.join(['manual_b', format(B, ''), '_n', format(N, ''), '_k', format(K, ''), '_d', format(D, '')]), 'B': B, 'N': N, 'D': D, 'K': K, 'dtype': 'bfloat16', 'x': x, 'centroids': centroids, 'x_sq': x_sq, 'c_sq': c_sq, 'out': out}
    launch_for_eval(inputs)
    ref_dist = x_sq.unsqueeze(-1) + c_sq.unsqueeze(1) - 2.0 * torch.einsum('bnd,bkd->bnk', x.float(), centroids.float())
    ref = ref_dist.clamp_min(0.0).argmin(dim=-1).to(torch.int32)
    result: dict[str, Any] = {'passed': bool(torch.equal(out, ref)), 'route_trace': inputs.get('_flash_kmeans_assign_dispatch_route')}
    if benchmark:
        from .._dispatch_runtime import evaluate
        result['contract_eval'] = evaluate(launch_for_eval, shapes=[{'label': ''.join(['manual_b', format(B, ''), '_n', format(N, ''), '_k', format(K, ''), '_d', format(D, '')]), 'params': {'B': B, 'N': N, 'D': D, 'K': K, 'dtype': 'bfloat16', 'seed': 11205}}])
    return result
