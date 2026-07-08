"""Flash-KMeans Euclidean assignment direct D64 seed.

Minimum architecture: sm_100a. This candidate uses Blackwell tcgen05/TMEM
(``lm.mma``) for exact D=64 rows. It reuses the single 64-wide feature tile
path from the micro-D seed but TMA-loads the original tensors directly instead
of launching a BF16 pad/copy sidecar. It is not intended for sm_120a/sm_121a
where ptxas rejects tcgen05 instructions.
"""
from __future__ import annotations
from json import loads as _json_loads
from .._dispatch_runtime import _capture_cuTensorMapEncodeTiled, _decode_capture, _import_dispatch_module, _ir_proxy
from functools import lru_cache
from typing import Any
BLOCK_N = 128
BLOCK_K = 256
SCORE_CHUNK_K = 128
CSQ_VEC = 4
CSQ_STAGE_VEC = 4
FEAT_D_PAD = 64
SUPPORTED_D = {64}
NUM_COMPUTE_WARPS = 4
X_TILE_BYTES = BLOCK_N * FEAT_D_PAD * 2
C_TILE_BYTES = BLOCK_K * FEAT_D_PAD * 2
CSQ_TILE_BYTES = BLOCK_K * 4
_TMAP_CACHE: dict[tuple[int, int, int, int, int, int], Any] = {}
ROUTE_ID = 'd64_direct_single64_1p2gap_9f2a_v1'
SEED_ID = 'd64-direct-single64-1p2gap-9f2a-v1'
flash_kmeans_assign_d64_direct_1p2gap_9f2a_v1 = _decode_capture(_json_loads('{"__ir__": "flash_kmeans_assign_d64_direct_1p2gap_9f2a_v1", "arg_keys": ["x_tmap", "c_tmap", "x_sq", "c_sq", "out", "B", "N", "D", "K", "num_n_tiles", "K_tiles"], "cluster_dims": [1, 1, 1], "computed_smem_bytes": 51200, "constants": [], "cta_group": 1, "threads": 192}'))
ir = _decode_capture(_json_loads('{"__ir__": "flash_kmeans_assign_d64_direct_1p2gap_9f2a_v1", "arg_keys": ["x_tmap", "c_tmap", "x_sq", "c_sq", "out", "B", "N", "D", "K", "num_n_tiles", "K_tiles"], "cluster_dims": [1, 1, 1], "computed_smem_bytes": 51200, "constants": [], "cta_group": 1, "threads": 192}'))

def _cuda_include_dirs() -> list[str]:
    from .._dispatch_runtime import _cuda_include_dirs as _common_cuda_include_dirs
    return _common_cuda_include_dirs()

def _create_direct_tensor_map_3d(data_ptr: int, global_height: int, shared_height: int):
    import torch
    from .._dispatch_runtime import create_tensor_map_3d
    device_index = torch.cuda.current_device()
    key = (device_index, int(data_ptr), int(global_height), int(shared_height), FEAT_D_PAD, FEAT_D_PAD)
    cached = _TMAP_CACHE.get(key)
    if cached is not None:
        return cached
    cached = create_tensor_map_3d(data_ptr, global_height, shared_height, FEAT_D_PAD, FEAT_D_PAD).to(device=torch.device('cuda', device_index))
    _TMAP_CACHE[key] = cached
    return cached

def _make_tmaps(inputs: dict[str, Any]) -> tuple[Any, Any]:
    cache_key = (int(inputs['x'].data_ptr()), int(inputs['centroids'].data_ptr()), int(inputs['B']), int(inputs['N']), int(inputs['K']), FEAT_D_PAD)
    cached = inputs.get('_flash_kmeans_assign_d64_direct_1p2gap_9f2a_v1_tmaps')
    if cached is not None and cached[0] == cache_key:
        return (cached[1], cached[2])
    bsz = int(inputs['B'])
    n_points = int(inputs['N'])
    n_clusters = int(inputs['K'])
    tmap_x = _create_direct_tensor_map_3d(inputs['x'].data_ptr(), bsz * n_points, BLOCK_N)
    tmap_c = _create_direct_tensor_map_3d(inputs['centroids'].data_ptr(), bsz * n_clusters, BLOCK_K)
    inputs['_flash_kmeans_assign_d64_direct_1p2gap_9f2a_v1_tmaps'] = (cache_key, tmap_x, tmap_c)
    return (tmap_x, tmap_c)

def _compiled_kernel() -> tuple[bytes, str, int, int]:
    return _decode_capture(_json_loads('{"__tuple__": [{"__kernel_source__": "dispatch_kernel_0339"}, "kernel_flash_kmeans_assign_d64_direct_1p2gap_9f2a_v1", 51200, 192]}'))

def launch_for_eval(inputs: dict[str, Any]) -> Any:
    from .._dispatch_runtime import CUDAKernel
    from .._dispatch_runtime import pack_kernel_args
    bsz = int(inputs['B'])
    n_points = int(inputs['N'])
    dim = int(inputs['D'])
    n_clusters = int(inputs['K'])
    dtype = str(inputs.get('dtype', getattr(inputs['x'], 'dtype', 'bfloat16'))).replace('torch.', '')
    if dtype not in {'bfloat16', 'bf16'}:
        raise ValueError(''.join(['flash_kmeans_assign_d64_direct_1p2gap_9f2a_v1 requires bfloat16 input, got ', format(dtype, '')]))
    if dim not in SUPPORTED_D:
        raise ValueError(''.join(['flash_kmeans_assign_d64_direct_1p2gap_9f2a_v1 requires D=64, got ', format(dim, '')]))
    if n_points % BLOCK_N != 0:
        raise ValueError(''.join(['N must be divisible by BLOCK_N=', format(BLOCK_N, ''), ', got ', format(n_points, '')]))
    if n_clusters % BLOCK_K != 0:
        raise ValueError(''.join(['K must be divisible by BLOCK_K=', format(BLOCK_K, ''), ', got ', format(n_clusters, '')]))
    tmap_x, tmap_c = _make_tmaps(inputs)
    cubin, kernel_name, smem_bytes, threads = _compiled_kernel()
    num_n_tiles = n_points // BLOCK_N
    k_tiles = n_clusters // BLOCK_K
    grid = (bsz * num_n_tiles, 1, 1)
    block = (threads, 1, 1)
    args = pack_kernel_args(ir, x_tmap=tmap_x, c_tmap=tmap_c, x_sq=inputs['x_sq'], c_sq=inputs['c_sq'], out=inputs['out'], B=bsz, N=n_points, D=dim, K=n_clusters, num_n_tiles=num_n_tiles, K_tiles=k_tiles)
    with CUDAKernel(cubin, kernel_name) as kernel:
        kernel.launch(grid=grid, block=block, args=args, shared_mem=smem_bytes)
    return {'cluster_ids': inputs['out'], 'selected_route': ROUTE_ID, 'route_trace': _route_trace(inputs, grid=grid)}

def compile_and_launch_flash_kmeans_assign_d64_direct(B: int=4, N: int=1024, K: int=512, D: int=64, *, benchmark: bool=False) -> dict[str, Any]:
    import torch
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA GPU required')
    torch.manual_seed(6401)
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
        result['contract_eval'] = evaluate(launch_for_eval, shapes=[{'label': ''.join(['manual_b', format(B, ''), '_n', format(N, ''), '_k', format(K, ''), '_d', format(D, '')]), 'params': {'B': B, 'N': N, 'D': D, 'K': K, 'dtype': 'bfloat16', 'seed': 6401}}])
    return result

def _route_trace(inputs: dict[str, Any], *, grid: tuple[int, int, int]) -> dict[str, Any]:
    return {'shape_key': _shape_key(inputs), 'selected_route': ROUTE_ID, 'selected_entrypoint': 'loom.examples.weave.flash_kmeans_assign_d64_direct_1p2gap_9f2a_v1:launch_for_eval', 'selected_seed': SEED_ID, 'route_kind': 'specialized', 'route_source': 'shape-specific-seed', 'guard_id': 'guard_d64_direct_single64_1p2gap_9f2a_v1', 'guard_condition': 'dtype == bfloat16 and D == 64 and N % 128 == 0 and K % 256 == 0', 'classification': 'seed-probe', 'dispatcher_kernel_ms': None, 'reason': 'direct D64 seed uses one 64-feature tcgen05 tile without BF16 pack/pad sidecar', 'launch_grid': grid}

def _shape_key(inputs: dict[str, Any]) -> str:
    label = inputs.get('label')
    if label:
        return str(label)
    return ''.join(['b', format(int(inputs['B']), ''), '_n', format(int(inputs['N']), ''), '_k', format(int(inputs['K']), ''), '_d', format(int(inputs['D']), '')])
