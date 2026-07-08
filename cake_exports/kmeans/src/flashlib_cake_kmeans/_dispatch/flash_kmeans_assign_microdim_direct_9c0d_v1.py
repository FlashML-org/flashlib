"""Flash-KMeans Euclidean assignment direct micro-D seed for D=16/32.

Minimum architecture: sm_100a. This candidate uses Blackwell tcgen05/TMEM
(``lm.mma``) for D=16/32 by staging logical BF16 rows directly into a
64-wide padded K-major shared-memory tile. It is not intended for
sm_120a/sm_121a where ptxas rejects tcgen05 instructions.
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
FEAT_D_PAD_VECS = FEAT_D_PAD // 8
STAGE_VEC = 8
SUPPORTED_D = {16, 32}
NUM_COMPUTE_WARPS = 4
X_TILE_BYTES = BLOCK_N * FEAT_D_PAD * 2
C_TILE_BYTES = BLOCK_K * FEAT_D_PAD * 2
CSQ_TILE_BYTES = BLOCK_K * 4
ROUTE_ID = 'microdim_direct_staged_9c0d_v1'
SEED_ID = 'microdim-direct-staged-9c0d-v1'
BF16_DTYPE_NAMES = {'bfloat16', 'bf16', 'torch.bfloat16'}
flash_kmeans_assign_microdim_direct_9c0d_v1 = _decode_capture(_json_loads('{"__ir__": "flash_kmeans_assign_microdim_direct_9c0d_v1", "arg_keys": ["x", "centroids", "x_sq", "c_sq", "out", "B", "N", "D", "K", "num_n_tiles", "K_tiles"], "cluster_dims": [1, 1, 1], "computed_smem_bytes": 51200, "constants": [], "cta_group": 1, "threads": 192}'))
ir = _decode_capture(_json_loads('{"__ir__": "flash_kmeans_assign_microdim_direct_9c0d_v1", "arg_keys": ["x", "centroids", "x_sq", "c_sq", "out", "B", "N", "D", "K", "num_n_tiles", "K_tiles"], "cluster_dims": [1, 1, 1], "computed_smem_bytes": 51200, "constants": [], "cta_group": 1, "threads": 192}'))

def _cuda_include_dirs() -> list[str]:
    from .._dispatch_runtime import _cuda_include_dirs as _common_cuda_include_dirs
    return _common_cuda_include_dirs()

def _compiled_kernel() -> tuple[bytes, str, int, int]:
    return _decode_capture(_json_loads('{"__tuple__": [{"__kernel_source__": "dispatch_kernel_0332"}, "kernel_flash_kmeans_assign_microdim_direct_9c0d_v1", 51200, 192]}'))

def launch_for_eval(inputs: dict[str, Any]) -> dict[str, Any]:
    from .._dispatch_runtime import CUDAKernel
    bsz = int(inputs['B'])
    n_points = int(inputs['N'])
    dim = int(inputs['D'])
    n_clusters = int(inputs['K'])
    dtype_name = _dtype_name(inputs)
    if dtype_name not in BF16_DTYPE_NAMES:
        raise ValueError(''.join(['flash_kmeans_assign_microdim_direct_9c0d_v1 requires bfloat16 input, got ', format(dtype_name, '')]))
    if dim not in SUPPORTED_D:
        raise ValueError(''.join(['flash_kmeans_assign_microdim_direct_9c0d_v1 requires D in ', format(sorted(SUPPORTED_D), ''), ', got ', format(dim, '')]))
    if n_points % BLOCK_N != 0:
        raise ValueError(''.join(['N must be divisible by BLOCK_N=', format(BLOCK_N, ''), ', got ', format(n_points, '')]))
    if n_clusters % BLOCK_K != 0:
        raise ValueError(''.join(['K must be divisible by BLOCK_K=', format(BLOCK_K, ''), ', got ', format(n_clusters, '')]))
    cubin, kernel_name, smem_bytes, threads = _compiled_kernel()
    num_n_tiles = n_points // BLOCK_N
    k_tiles = n_clusters // BLOCK_K
    grid = (bsz * num_n_tiles, 1, 1)
    block = (threads, 1, 1)
    args = [inputs['x'], inputs['centroids'], inputs['x_sq'], inputs['c_sq'], inputs['out'], bsz, n_points, dim, n_clusters, num_n_tiles, k_tiles]
    with CUDAKernel(cubin, kernel_name) as kernel:
        kernel.launch(grid=grid, block=block, args=args, shared_mem=smem_bytes)
    trace = _route_trace(inputs, total_tiles=bsz * num_n_tiles, grid=grid)
    return {'cluster_ids': inputs['out'], 'selected_route': ROUTE_ID, 'route_trace': trace}

def compile_and_launch_flash_kmeans_assign_microdim_direct(B: int=4, N: int=1024, K: int=512, D: int=16, *, benchmark: bool=False) -> dict[str, Any]:
    import torch
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA GPU required')
    torch.manual_seed(1601)
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
        result['contract_eval'] = evaluate(launch_for_eval, shapes=[{'label': ''.join(['manual_b', format(B, ''), '_n', format(N, ''), '_k', format(K, ''), '_d', format(D, '')]), 'params': {'B': B, 'N': N, 'D': D, 'K': K, 'dtype': 'bfloat16', 'seed': 1601}}])
    return result

def _route_trace(inputs: dict[str, Any], *, total_tiles: int, grid: tuple[int, int, int]) -> dict[str, Any]:
    return {'shape_key': _shape_key(inputs), 'selected_route': ROUTE_ID, 'selected_entrypoint': 'loom.examples.weave.flash_kmeans_assign_microdim_direct_9c0d_v1:launch_for_eval', 'selected_seed': SEED_ID, 'expected_seed': None, 'route_kind': 'specialized', 'route_source': 'shape-specific-seed', 'guard_id': 'guard_microdim_direct_staged_9c0d_v1', 'guard_condition': 'dtype == bfloat16 and D in [16, 32] and N % 128 == 0 and K % 256 == 0', 'classification': 'route-ok', 'dispatcher_kernel_ms': None, 'shape_specific_kernel_ms': None, 'relative_speedup_vs_baseline': None, 'total_tiles': total_tiles, 'launch_grid': grid, 'reason': 'micro-D direct SMEM staging feeds one padded tcgen05 score tile without global scratch packing'}

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
