"""Clean-room Flash-KMeans Euclidean assignment D192 repeated-MMA paired-tile candidate.

Minimum architecture: sm_100a. This candidate dispatches between a D-specific repeated-MMA single point-tile path and a D-specific repeated-MMA paired point-tile tcgen05/TMEM path. It is not intended for
sm_120a/sm_121a where ptxas rejects tcgen05 instructions.
"""
from __future__ import annotations
from json import loads as _json_loads
from .._dispatch_runtime import _capture_cuTensorMapEncodeTiled, _decode_capture, _import_dispatch_module, _ir_proxy
from functools import lru_cache
from typing import Any
from . import flash_kmeans_assign_cleanroom_tcgen05_d192_single_v1 as _single
BLOCK_N = 128
BLOCK_K = 256
SCORE_CHUNK_K = 128
CSQ_VEC = 16
CSQ_STAGE_VEC = 4
FEAT_D = 192
NUM_COMPUTE_WARPS = 4
X_TILE_BYTES = BLOCK_N * FEAT_D * 2
C_TILE_BYTES = BLOCK_K * FEAT_D * 2
X_FEAT_CHUNK_BYTES = BLOCK_N * 64 * 2
C_FEAT_CHUNK_BYTES = BLOCK_K * 64 * 2
CSQ_TILE_BYTES = BLOCK_K * 4
X1_OFFSET = X_TILE_BYTES
C_OFFSET = 2 * X_TILE_BYTES
CSQ_OFFSET = 2 * X_TILE_BYTES + C_TILE_BYTES
PAIRED_GRID_CAP = 2516
flash_kmeans_assign_cleanroom_tcgen05_d192_splitd_v1 = _decode_capture(_json_loads('{"__ir__": "flash_kmeans_assign_cleanroom_tcgen05_d192_splitd_v1", "arg_keys": ["x_tmap", "c_tmap", "x_sq", "c_sq", "out", "B", "N", "D", "K", "num_n_tiles", "K_tiles"], "cluster_dims": [1, 1, 1], "computed_smem_bytes": 198656, "constants": [], "cta_group": 1, "threads": 192}'))

def _cuda_include_dirs() -> list[str]:
    from .._dispatch_runtime import _cuda_include_dirs as _common_cuda_include_dirs
    return _common_cuda_include_dirs()

def _make_tmaps(inputs: dict[str, Any]) -> tuple[Any, Any]:
    cache_key = (int(inputs['x'].data_ptr()), int(inputs['centroids'].data_ptr()), int(inputs['B']), int(inputs['N']), int(inputs['K']), int(inputs['D']))
    cached = inputs.get('_flash_kmeans_assign_cleanroom_d192_splitd_v1_tmaps')
    if cached is not None and cached[0] == cache_key:
        return (cached[1], cached[2])
    from .._dispatch_runtime import create_tensor_map_3d
    bsz = int(inputs['B'])
    n_points = int(inputs['N'])
    n_clusters = int(inputs['K'])
    dim = int(inputs['D'])
    tmap_x = create_tensor_map_3d(inputs['x'].data_ptr(), bsz * n_points, BLOCK_N, dim, dim)
    tmap_c = create_tensor_map_3d(inputs['centroids'].data_ptr(), bsz * n_clusters, BLOCK_K, dim, dim)
    inputs['_flash_kmeans_assign_cleanroom_d192_splitd_v1_tmaps'] = (cache_key, tmap_x, tmap_c)
    return (tmap_x, tmap_c)

def _compiled_kernel() -> tuple[bytes, str, int, int]:
    return _decode_capture(_json_loads('{"__tuple__": [{"__kernel_source__": "dispatch_kernel_0005"}, "kernel_flash_kmeans_assign_cleanroom_tcgen05_d192_splitd_v1", 198656, 192]}'))
ir = _decode_capture(_json_loads('{"__ir__": "flash_kmeans_assign_cleanroom_tcgen05_d192_splitd_v1", "arg_keys": ["x_tmap", "c_tmap", "x_sq", "c_sq", "out", "B", "N", "D", "K", "num_n_tiles", "K_tiles"], "cluster_dims": [1, 1, 1], "computed_smem_bytes": 198656, "constants": [], "cta_group": 1, "threads": 192}'))

def _use_single_tile_path(*, n_points: int, n_clusters: int) -> bool:
    num_n_tiles = n_points // BLOCK_N
    k_tiles = n_clusters // BLOCK_K
    return num_n_tiles % 2 != 0 or (num_n_tiles <= 8 and k_tiles <= 2)

def launch_for_eval(inputs: dict[str, Any]) -> Any:
    from .._dispatch_runtime import CUDAKernel
    from .._dispatch_runtime import pack_kernel_args
    bsz = int(inputs['B'])
    n_points = int(inputs['N'])
    dim = int(inputs['D'])
    n_clusters = int(inputs['K'])
    if _use_single_tile_path(n_points=n_points, n_clusters=n_clusters):
        return _single.launch_for_eval(inputs)
    if dim != FEAT_D:
        raise ValueError(''.join(['flash_kmeans_assign_cleanroom_tcgen05_d192_splitd_v1 requires D=', format(FEAT_D, ''), ', got ', format(dim, '')]))
    if n_points % BLOCK_N != 0:
        raise ValueError(''.join(['N must be divisible by BLOCK_N=', format(BLOCK_N, ''), ', got ', format(n_points, '')]))
    if n_clusters % BLOCK_K != 0:
        raise ValueError(''.join(['K must be divisible by BLOCK_K=', format(BLOCK_K, ''), ', got ', format(n_clusters, '')]))
    num_n_tiles = n_points // BLOCK_N
    if num_n_tiles % 2 != 0:
        raise ValueError('flash_kmeans_assign_cleanroom_tcgen05_d192_splitd_v1 requires an even number of BLOCK_N point tiles')
    tmap_x, tmap_c = _make_tmaps(inputs)
    cubin, kernel_name, smem_bytes, threads = _compiled_kernel()
    k_tiles = n_clusters // BLOCK_K
    total_tiles = bsz * (num_n_tiles // 2)
    grid = (min(total_tiles, PAIRED_GRID_CAP), 1, 1)
    block = (threads, 1, 1)
    args = pack_kernel_args(ir, x_tmap=tmap_x, c_tmap=tmap_c, x_sq=inputs['x_sq'], c_sq=inputs['c_sq'], out=inputs['out'], B=bsz, N=n_points, D=dim, K=n_clusters, num_n_tiles=num_n_tiles, K_tiles=k_tiles)
    with CUDAKernel(cubin, kernel_name) as kernel:
        kernel.launch(grid=grid, block=block, args=args, shared_mem=smem_bytes)
    return {'cluster_ids': inputs['out']}

def compile_and_launch_flash_kmeans_assign_cleanroom(B: int=2, N: int=1024, K: int=512, D: int=192, *, benchmark: bool=False) -> dict[str, Any]:
    import torch
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA GPU required')
    torch.manual_seed(5101)
    x = torch.randn((B, N, D), dtype=torch.bfloat16, device='cuda').contiguous()
    centroids = torch.randn((B, K, D), dtype=torch.bfloat16, device='cuda').contiguous()
    x_sq = (x.float() ** 2).sum(-1).contiguous()
    c_sq = (centroids.float() ** 2).sum(-1).contiguous()
    out = torch.empty((B, N), dtype=torch.int32, device='cuda')
    inputs = {'B': B, 'N': N, 'D': D, 'K': K, 'x': x, 'centroids': centroids, 'x_sq': x_sq, 'c_sq': c_sq, 'out': out}
    launch_for_eval(inputs)
    ref_dist = x_sq.unsqueeze(-1) + c_sq.unsqueeze(1) - 2.0 * torch.einsum('bnd,bkd->bnk', x.float(), centroids.float())
    ref = ref_dist.clamp_min(0.0).argmin(dim=-1).to(torch.int32)
    passed = bool(torch.equal(out, ref))
    result: dict[str, Any] = {'passed': passed}
    if benchmark:
        from .._dispatch_runtime import evaluate
        result['contract_eval'] = evaluate(launch_for_eval, shapes=[{'label': ''.join(['manual_b', format(B, ''), '_n', format(N, ''), '_k', format(K, ''), '_d', format(D, '')]), 'params': {'B': B, 'N': N, 'D': D, 'K': K, 'dtype': 'bfloat16', 'seed': 5101}}])
    return result
