"""Flash-KMeans Euclidean assignment high-D split-D candidate.

Minimum architecture: sm_100a. This candidate uses Blackwell tcgen05/TMEM
(``lm.mma``) for D in {320, 384, 448, 512}. It slices the feature dimension
into 64-wide TMA/MMA groups so high-D shapes stay within shared memory. It is
not intended for sm_120a/sm_121a where ptxas rejects tcgen05 instructions.
"""
from __future__ import annotations
from json import loads as _json_loads
from .._dispatch_runtime import _capture_cuTensorMapEncodeTiled, _decode_capture, _import_dispatch_module, _ir_proxy
from functools import lru_cache
from typing import Any
BLOCK_N = 128
BLOCK_K = 256
FEAT_CHUNK = 64
SCORE_CHUNK_K = 128
CSQ_VEC = 4
CSQ_STAGE_VEC = 4
NUM_COMPUTE_WARPS = 4
X_TILE_BYTES = BLOCK_N * FEAT_CHUNK * 2
C_TILE_BYTES = BLOCK_K * FEAT_CHUNK * 2
CSQ_TILE_BYTES = BLOCK_K * 4
PAIRED_GRID_CAP = 2516
SUPPORTED_DIMS = {320, 384, 448, 512}
BF16_DTYPE_NAMES = {'bfloat16', 'bf16', 'torch.bfloat16'}
ROUTE_ID = 'highd_splitd_single_tile_6fcf_v1'
SEED_ID = 'highd-splitd-single-tile-6fcf-v1'
flash_kmeans_assign_highd_splitd_6fcf_v1 = _decode_capture(_json_loads('{"__ir__": "flash_kmeans_assign_highd_splitd_6fcf_v1", "arg_keys": ["x_tmap", "c_tmap", "x_sq", "c_sq", "out", "B", "N", "D", "K", "num_n_tiles", "K_tiles"], "cluster_dims": [1, 1, 1], "computed_smem_bytes": 51200, "constants": [], "cta_group": 1, "threads": 192}'))

def _cuda_include_dirs() -> list[str]:
    from .._dispatch_runtime import _cuda_include_dirs as _common_cuda_include_dirs
    return _common_cuda_include_dirs()

def _make_tmaps(inputs: dict[str, Any]) -> tuple[Any, Any]:
    cache_key = (int(inputs['x'].data_ptr()), int(inputs['centroids'].data_ptr()), int(inputs['B']), int(inputs['N']), int(inputs['K']), int(inputs['D']))
    cached = inputs.get('_flash_kmeans_assign_highd_splitd_6fcf_v1_tmaps')
    if cached is not None and cached[0] == cache_key:
        return (cached[1], cached[2])
    from .._dispatch_runtime import create_tensor_map_3d
    bsz = int(inputs['B'])
    n_points = int(inputs['N'])
    n_clusters = int(inputs['K'])
    dim = int(inputs['D'])
    tmap_x = create_tensor_map_3d(inputs['x'].data_ptr(), bsz * n_points, BLOCK_N, dim, FEAT_CHUNK)
    tmap_c = create_tensor_map_3d(inputs['centroids'].data_ptr(), bsz * n_clusters, BLOCK_K, dim, FEAT_CHUNK)
    inputs['_flash_kmeans_assign_highd_splitd_6fcf_v1_tmaps'] = (cache_key, tmap_x, tmap_c)
    return (tmap_x, tmap_c)

def _compiled_kernel() -> tuple[bytes, str, int, int]:
    return _decode_capture(_json_loads('{"__tuple__": [{"__kernel_source__": "dispatch_kernel_0006"}, "kernel_flash_kmeans_assign_highd_splitd_6fcf_v1", 51200, 192]}'))
ir = _decode_capture(_json_loads('{"__ir__": "flash_kmeans_assign_highd_splitd_6fcf_v1", "arg_keys": ["x_tmap", "c_tmap", "x_sq", "c_sq", "out", "B", "N", "D", "K", "num_n_tiles", "K_tiles"], "cluster_dims": [1, 1, 1], "computed_smem_bytes": 51200, "constants": [], "cta_group": 1, "threads": 192}'))

def launch_for_eval(inputs: dict[str, Any]) -> dict[str, Any]:
    from .._dispatch_runtime import CUDAKernel
    from .._dispatch_runtime import pack_kernel_args
    bsz = int(inputs['B'])
    n_points = int(inputs['N'])
    dim = int(inputs['D'])
    n_clusters = int(inputs['K'])
    dtype_name = _dtype_name(inputs)
    if dtype_name not in BF16_DTYPE_NAMES:
        raise ValueError(''.join(['flash_kmeans_assign_highd_splitd_6fcf_v1 requires bfloat16 input, got ', format(dtype_name, '')]))
    if dim not in SUPPORTED_DIMS:
        raise ValueError(''.join(['flash_kmeans_assign_highd_splitd_6fcf_v1 supports D=', format(sorted(SUPPORTED_DIMS), ''), ', got ', format(dim, '')]))
    if n_points % BLOCK_N != 0:
        raise ValueError(''.join(['N must be divisible by BLOCK_N=', format(BLOCK_N, ''), ', got ', format(n_points, '')]))
    if n_clusters % BLOCK_K != 0:
        raise ValueError(''.join(['K must be divisible by BLOCK_K=', format(BLOCK_K, ''), ', got ', format(n_clusters, '')]))
    tmap_x, tmap_c = _make_tmaps(inputs)
    cubin, kernel_name, smem_bytes, threads = _compiled_kernel()
    num_n_tiles = n_points // BLOCK_N
    k_tiles = n_clusters // BLOCK_K
    total_tiles = bsz * num_n_tiles
    grid = (min(total_tiles, PAIRED_GRID_CAP), 1, 1)
    block = (threads, 1, 1)
    args = pack_kernel_args(ir, x_tmap=tmap_x, c_tmap=tmap_c, x_sq=inputs['x_sq'], c_sq=inputs['c_sq'], out=inputs['out'], B=bsz, N=n_points, D=dim, K=n_clusters, num_n_tiles=num_n_tiles, K_tiles=k_tiles)
    with CUDAKernel(cubin, kernel_name) as kernel:
        kernel.launch(grid=grid, block=block, args=args, shared_mem=smem_bytes)
    trace = _route_trace(inputs, total_tiles=total_tiles, grid=grid)
    return {'cluster_ids': inputs['out'], 'selected_route': ROUTE_ID, 'route_trace': trace}

def compile_and_launch_flash_kmeans_assign_highd_splitd(B: int=1, N: int=1024, K: int=512, D: int=512, *, benchmark: bool=False) -> dict[str, Any]:
    import torch
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA GPU required')
    torch.manual_seed(51201)
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
        result['contract_eval'] = evaluate(launch_for_eval, shapes=[{'label': ''.join(['manual_b', format(B, ''), '_n', format(N, ''), '_k', format(K, ''), '_d', format(D, '')]), 'params': {'B': B, 'N': N, 'D': D, 'K': K, 'dtype': 'bfloat16', 'seed': 51201}}])
    return result

def _route_trace(inputs: dict[str, Any], *, total_tiles: int, grid: tuple[int, int, int]) -> dict[str, Any]:
    return {'shape_key': _shape_key(inputs), 'selected_route': ROUTE_ID, 'selected_entrypoint': 'loom.examples.weave.flash_kmeans_assign_highd_splitd_6fcf_v1:launch_for_eval', 'selected_seed': SEED_ID, 'expected_seed': None, 'route_kind': 'specialized', 'route_source': 'shape-specific-seed', 'guard_id': 'guard_highd_splitd_6fcf_v1', 'guard_condition': 'dtype == bfloat16 and D in [320, 384, 448, 512] and N % 128 == 0 and K % 256 == 0', 'classification': 'route-ok', 'dispatcher_kernel_ms': None, 'shape_specific_kernel_ms': None, 'relative_speedup_vs_baseline': None, 'total_tiles': total_tiles, 'launch_grid': grid, 'reason': 'high-D split-D seed accumulates 64-feature tcgen05 slices into one argmin path'}

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
