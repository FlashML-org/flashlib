"""Flash-KMeans Euclidean assignment low-D seed for D=48/64/80/96/112.

Minimum architecture: sm_100a. This candidate uses Blackwell tcgen05/TMEM
(``lm.mma``) with a Weave BF16 pack/zero-fill producer from logical
D=48/64/80/96/112 to a 128-wide MMA K tile. It is not intended for
sm_120a/sm_121a where ptxas rejects tcgen05 instructions.
"""
from __future__ import annotations
from json import loads as _json_loads
from .._dispatch_runtime import _capture_cuTensorMapEncodeTiled, _decode_capture, _import_dispatch_module, _ir_proxy
from functools import lru_cache
from typing import Any
from .flash_kmeans_assign_microdim_6cd2_v1 import ir as d64_ir
from .flash_kmeans_assign_stream import stream_cache_key
BLOCK_N = 128
BLOCK_K = 256
SCORE_CHUNK_K = 128
CSQ_VEC = 4
CSQ_STAGE_VEC = 4
FEAT_D_PAD = 128
FEAT_D_PAD_VECS = FEAT_D_PAD // 8
FEAT_D_DIRECT = 64
SUPPORTED_D = {48, 64, 80, 96, 112}
NUM_COMPUTE_WARPS = 4
X_TILE_BYTES = BLOCK_N * FEAT_D_PAD * 2
C_TILE_BYTES = BLOCK_K * FEAT_D_PAD * 2
CSQ_TILE_BYTES = BLOCK_K * 4
PACK_THREADS = 256
PACK_GRID_CAP = 4096
_TMAP_CACHE: dict[tuple[int, int, int, int, int, int], Any] = {}
_D64_TMAP_CACHE: dict[tuple[int, int, int, int, int, int], Any] = {}
flash_kmeans_assign_lowdim_pack_e50c_v1 = _decode_capture(_json_loads('{"__ir__": "flash_kmeans_assign_lowdim_pack_e50c_v1", "arg_keys": ["x", "centroids", "x_pad", "c_pad", "B", "N", "D", "K", "total_x_pad", "total_c_pad"], "cluster_dims": [1, 1, 1], "computed_smem_bytes": 0, "constants": [], "cta_group": 1, "threads": 256}'))
flash_kmeans_assign_lowdim_e50c_v1 = _decode_capture(_json_loads('{"__ir__": "flash_kmeans_assign_lowdim_e50c_v1", "arg_keys": ["x_tmap", "c_tmap", "x_sq", "c_sq", "out", "B", "N", "D", "K", "num_n_tiles", "K_tiles"], "cluster_dims": [1, 1, 1], "computed_smem_bytes": 100352, "constants": [], "cta_group": 1, "threads": 192}'))
ir = _decode_capture(_json_loads('{"__ir__": "flash_kmeans_assign_lowdim_e50c_v1", "arg_keys": ["x_tmap", "c_tmap", "x_sq", "c_sq", "out", "B", "N", "D", "K", "num_n_tiles", "K_tiles"], "cluster_dims": [1, 1, 1], "computed_smem_bytes": 100352, "constants": [], "cta_group": 1, "threads": 192}'))
pack_ir = _decode_capture(_json_loads('{"__ir__": "flash_kmeans_assign_lowdim_pack_e50c_v1", "arg_keys": ["x", "centroids", "x_pad", "c_pad", "B", "N", "D", "K", "total_x_pad", "total_c_pad"], "cluster_dims": [1, 1, 1], "computed_smem_bytes": 0, "constants": [], "cta_group": 1, "threads": 256}'))

def _cuda_include_dirs() -> list[str]:
    from .._dispatch_runtime import _cuda_include_dirs as _common_cuda_include_dirs
    return _common_cuda_include_dirs()

def _create_padded_tensor_map_3d(data_ptr: int, global_height: int, shared_height: int):
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

def _make_tmaps(inputs: dict[str, Any], x_pad: Any, c_pad: Any) -> tuple[Any, Any]:
    cache_key = (int(x_pad.data_ptr()), int(c_pad.data_ptr()), int(inputs['B']), int(inputs['N']), int(inputs['K']), FEAT_D_PAD)
    cached = inputs.get('_flash_kmeans_assign_lowdim_e50c_v1_tmaps')
    if cached is not None and cached[0] == cache_key:
        return (cached[1], cached[2])
    bsz = int(inputs['B'])
    n_points = int(inputs['N'])
    n_clusters = int(inputs['K'])
    tmap_x = _create_padded_tensor_map_3d(x_pad.data_ptr(), bsz * n_points, BLOCK_N)
    tmap_c = _create_padded_tensor_map_3d(c_pad.data_ptr(), bsz * n_clusters, BLOCK_K)
    inputs['_flash_kmeans_assign_lowdim_e50c_v1_tmaps'] = (cache_key, tmap_x, tmap_c)
    return (tmap_x, tmap_c)

def _create_d64_tensor_map_3d(data_ptr: int, global_height: int, shared_height: int):
    import torch
    from .._dispatch_runtime import create_tensor_map_3d
    device_index = torch.cuda.current_device()
    key = (device_index, int(data_ptr), int(global_height), int(shared_height), FEAT_D_DIRECT, FEAT_D_DIRECT)
    cached = _D64_TMAP_CACHE.get(key)
    if cached is not None:
        return cached
    cached = create_tensor_map_3d(data_ptr, global_height, shared_height, FEAT_D_DIRECT, FEAT_D_DIRECT).to(device=torch.device('cuda', device_index))
    _D64_TMAP_CACHE[key] = cached
    return cached

def _make_d64_tmaps(inputs: dict[str, Any]) -> tuple[Any, Any]:
    cache_key = (int(inputs['x'].data_ptr()), int(inputs['centroids'].data_ptr()), int(inputs['B']), int(inputs['N']), int(inputs['K']), FEAT_D_DIRECT)
    cached = inputs.get('_flash_kmeans_assign_lowdim_e50c_v1_d64_tmaps')
    if cached is not None and cached[0] == cache_key:
        return (cached[1], cached[2])
    bsz = int(inputs['B'])
    n_points = int(inputs['N'])
    n_clusters = int(inputs['K'])
    tmap_x = _create_d64_tensor_map_3d(inputs['x'].data_ptr(), bsz * n_points, BLOCK_N)
    tmap_c = _create_d64_tensor_map_3d(inputs['centroids'].data_ptr(), bsz * n_clusters, BLOCK_K)
    inputs['_flash_kmeans_assign_lowdim_e50c_v1_d64_tmaps'] = (cache_key, tmap_x, tmap_c)
    return (tmap_x, tmap_c)

def _scratch_buffers(inputs: dict[str, Any]) -> tuple[Any, Any]:
    import torch
    bsz = int(inputs['B'])
    n_points = int(inputs['N'])
    n_clusters = int(inputs['K'])
    cache_key = stream_cache_key(inputs, int(inputs['x'].data_ptr()), int(inputs['centroids'].data_ptr()), bsz, n_points, n_clusters, int(inputs['D']), FEAT_D_PAD)
    cache = inputs.setdefault('_flash_kmeans_assign_lowdim_e50c_v1_scratch', {})
    cached = cache.get(cache_key)
    if cached is not None:
        return cached
    x_pad = torch.empty((bsz, n_points, FEAT_D_PAD), dtype=inputs['x'].dtype, device=inputs['x'].device)
    c_pad = torch.empty((bsz, n_clusters, FEAT_D_PAD), dtype=inputs['centroids'].dtype, device=inputs['centroids'].device)
    cache[cache_key] = (x_pad, c_pad)
    inputs.pop('_flash_kmeans_assign_lowdim_e50c_v1_tmaps', None)
    return (x_pad, c_pad)

def _compiled_pack_kernel() -> tuple[bytes, str, int, int]:
    return _decode_capture(_json_loads('{"__tuple__": [{"__kernel_source__": "dispatch_kernel_0335"}, "kernel_flash_kmeans_assign_lowdim_pack_e50c_v1", 0, 256]}'))

def _compiled_d64_kernel() -> tuple[bytes, str, int, int]:
    return _decode_capture(_json_loads('{"__tuple__": [{"__kernel_source__": "dispatch_kernel_0336"}, "kernel_flash_kmeans_assign_microdim_6cd2_v1", 51200, 192]}'))

def _launch_d64_direct(inputs: dict[str, Any]) -> None:
    from .._dispatch_runtime import CUDAKernel
    from .._dispatch_runtime import pack_kernel_args
    bsz = int(inputs['B'])
    n_points = int(inputs['N'])
    dim = int(inputs['D'])
    n_clusters = int(inputs['K'])
    tmap_x, tmap_c = _make_d64_tmaps(inputs)
    cubin, kernel_name, smem_bytes, threads = _compiled_d64_kernel()
    num_n_tiles = n_points // BLOCK_N
    k_tiles = n_clusters // BLOCK_K
    args = pack_kernel_args(d64_ir, x_tmap=tmap_x, c_tmap=tmap_c, x_sq=inputs['x_sq'], c_sq=inputs['c_sq'], out=inputs['out'], B=bsz, N=n_points, D=dim, K=n_clusters, num_n_tiles=num_n_tiles, K_tiles=k_tiles)
    with CUDAKernel(cubin, kernel_name) as kernel:
        kernel.launch(grid=(bsz * num_n_tiles, 1, 1), block=(threads, 1, 1), args=args, shared_mem=smem_bytes)

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

def _compiled_kernel() -> tuple[bytes, str, int, int]:
    return _decode_capture(_json_loads('{"__tuple__": [{"__kernel_source__": "dispatch_kernel_0337"}, "kernel_flash_kmeans_assign_lowdim_e50c_v1", 100352, 192]}'))

def launch_for_eval(inputs: dict[str, Any]) -> Any:
    from .._dispatch_runtime import CUDAKernel
    from .._dispatch_runtime import pack_kernel_args
    bsz = int(inputs['B'])
    n_points = int(inputs['N'])
    dim = int(inputs['D'])
    n_clusters = int(inputs['K'])
    dtype = str(inputs.get('dtype', getattr(inputs['x'], 'dtype', 'bfloat16'))).replace('torch.', '')
    if dtype not in {'bfloat16', 'bf16'}:
        raise ValueError(''.join(['flash_kmeans_assign_lowdim_e50c_v1 requires bfloat16 input, got ', format(dtype, '')]))
    if dim not in SUPPORTED_D:
        raise ValueError(''.join(['flash_kmeans_assign_lowdim_e50c_v1 requires D in ', format(sorted(SUPPORTED_D), ''), ', got ', format(dim, '')]))
    if n_points % BLOCK_N != 0:
        raise ValueError(''.join(['N must be divisible by BLOCK_N=', format(BLOCK_N, ''), ', got ', format(n_points, '')]))
    if n_clusters % BLOCK_K != 0:
        raise ValueError(''.join(['K must be divisible by BLOCK_K=', format(BLOCK_K, ''), ', got ', format(n_clusters, '')]))
    if dim == FEAT_D_DIRECT:
        _launch_d64_direct(inputs)
        return {'cluster_ids': inputs['out']}
    x_pad, c_pad = _scratch_buffers(inputs)
    _launch_pack(inputs, x_pad, c_pad)
    tmap_x, tmap_c = _make_tmaps(inputs, x_pad, c_pad)
    cubin, kernel_name, smem_bytes, threads = _compiled_kernel()
    num_n_tiles = n_points // BLOCK_N
    k_tiles = n_clusters // BLOCK_K
    grid = (bsz * num_n_tiles, 1, 1)
    block = (threads, 1, 1)
    args = pack_kernel_args(ir, x_tmap=tmap_x, c_tmap=tmap_c, x_sq=inputs['x_sq'], c_sq=inputs['c_sq'], out=inputs['out'], B=bsz, N=n_points, D=dim, K=n_clusters, num_n_tiles=num_n_tiles, K_tiles=k_tiles)
    with CUDAKernel(cubin, kernel_name) as kernel:
        kernel.launch(grid=grid, block=block, args=args, shared_mem=smem_bytes)
    return {'cluster_ids': inputs['out']}

def compile_and_launch_flash_kmeans_assign_lowdim(B: int=4, N: int=1024, K: int=512, D: int=64, *, benchmark: bool=False) -> dict[str, Any]:
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
