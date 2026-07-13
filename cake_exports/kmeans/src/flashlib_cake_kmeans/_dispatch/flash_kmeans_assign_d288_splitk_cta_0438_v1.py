"""Flash-KMeans Euclidean assignment D288 split-K candidate.

Minimum architecture: sm_100a. This candidate uses Blackwell tcgen05/TMEM
(``lm.mma``) for exact D=288. High-K rows split the cluster
tile loop across CTAs, write one partial argmin per K tile, then reduce those
partials with a second Weave kernel. Non-splitK rows reuse the validated D288
parent. It is not intended for sm_120a/sm_121a where ptxas rejects
tcgen05 instructions.
"""
from __future__ import annotations
from json import loads as _json_loads
from .._dispatch_runtime import _capture_cuTensorMapEncodeTiled, _decode_capture, _import_dispatch_module, _ir_proxy
import os
from functools import lru_cache
from typing import Any
from . import flash_kmeans_assign_d288_exactd_a532_v1 as _direct
from .flash_kmeans_assign_stream import stream_cache_key
BLOCK_N = _direct.BLOCK_N
BLOCK_K = _direct.BLOCK_K
CHUNK_D = _direct.CHUNK_D
D_CHUNKS = _direct.D_CHUNKS
SCORE_CHUNK_K = _direct.SCORE_CHUNK_K
CSQ_VEC = _direct.CSQ_VEC
CSQ_STAGE_VEC = _direct.CSQ_STAGE_VEC
NUM_COMPUTE_WARPS = _direct.NUM_COMPUTE_WARPS
X_TILE_BYTES = _direct.X_TILE_BYTES
C_TILE_BYTES = _direct.C_TILE_BYTES
CSQ_TILE_BYTES = _direct.CSQ_TILE_BYTES
SUPPORTED_DIMS = {288}
BF16_DTYPE_NAMES = {'bfloat16', 'torch.bfloat16'}
SPLITK_GRID_CAP = 4096
REDUCE_THREADS = 128
SPLITK_MIN_K_TILES = 8
ROUTE_ID = 'd288_splitk_cta_0438_v1'
SEED_ID = 'd288-splitk-cta-0438-v1'
flash_kmeans_assign_d288_splitk_cta_0438_v1_partial = _decode_capture(_json_loads('{"__ir__": "flash_kmeans_assign_d288_splitk_cta_0438_v1_partial", "arg_keys": ["x_tmap", "c_tmap", "c_sq", "partial_scores", "partial_indices", "B", "N", "D", "K", "num_n_tiles", "K_tiles"], "cluster_dims": [1, 1, 1], "computed_smem_bytes": 26624, "constants": [], "cta_group": 1, "threads": 192}'))
flash_kmeans_assign_d288_splitk_cta_0438_v1_reduce = _decode_capture(_json_loads('{"__ir__": "flash_kmeans_assign_d288_splitk_cta_0438_v1_reduce", "arg_keys": ["partial_scores", "partial_indices", "out", "B", "N", "K", "num_n_tiles", "K_tiles"], "cluster_dims": [1, 1, 1], "computed_smem_bytes": 0, "constants": [], "cta_group": 1, "threads": 128}'))
partial_ir = _decode_capture(_json_loads('{"__ir__": "flash_kmeans_assign_d288_splitk_cta_0438_v1_partial", "arg_keys": ["x_tmap", "c_tmap", "c_sq", "partial_scores", "partial_indices", "B", "N", "D", "K", "num_n_tiles", "K_tiles"], "cluster_dims": [1, 1, 1], "computed_smem_bytes": 26624, "constants": [], "cta_group": 1, "threads": 192}'))
reduce_ir = _decode_capture(_json_loads('{"__ir__": "flash_kmeans_assign_d288_splitk_cta_0438_v1_reduce", "arg_keys": ["partial_scores", "partial_indices", "out", "B", "N", "K", "num_n_tiles", "K_tiles"], "cluster_dims": [1, 1, 1], "computed_smem_bytes": 0, "constants": [], "cta_group": 1, "threads": 128}'))

def _verify_export_ir() -> Any:
    if os.environ.get('LOOM_FLASH_KMEANS_D288_SPLITK_CTA_0438_V1_VERIFY_KERNEL') == 'reduce':
        return reduce_ir
    return partial_ir
ir = _decode_capture(_json_loads('{"__ir__": "flash_kmeans_assign_d288_splitk_cta_0438_v1_partial", "arg_keys": ["x_tmap", "c_tmap", "c_sq", "partial_scores", "partial_indices", "B", "N", "D", "K", "num_n_tiles", "K_tiles"], "cluster_dims": [1, 1, 1], "computed_smem_bytes": 26624, "constants": [], "cta_group": 1, "threads": 192}'))

def _cuda_include_dirs() -> list[str]:
    return _direct._cuda_include_dirs()

def _compiled_partial_kernel() -> tuple[bytes, str, int, int]:
    return _decode_capture(_json_loads('{"__tuple__": [{"__kernel_source__": "dispatch_kernel_0027"}, "kernel_flash_kmeans_assign_d288_splitk_cta_0438_v1_partial", 26624, 192]}'))

def _compiled_reduce_kernel() -> tuple[bytes, str, int, int]:
    return _decode_capture(_json_loads('{"__tuple__": [{"__kernel_source__": "dispatch_kernel_0028"}, "kernel_flash_kmeans_assign_d288_splitk_cta_0438_v1_reduce", 0, 128]}'))

def _compile_ir(ir_obj: Any) -> tuple[bytes, str, int, int]:
    from .._dispatch_runtime import generate_kernel
    from .._dispatch_runtime import compile_cuda
    smem_bytes = ir_obj.computed_smem_bytes
    source = generate_kernel(ir_obj, validate=False, smem_bytes=smem_bytes)
    cubin = compile_cuda(source, options=['--use_fast_math'], include_dirs=_cuda_include_dirs())
    return (cubin, ''.join(['kernel_', format(ir_obj.symbol, '')]), int(smem_bytes), int(ir_obj.threads))

def launch_for_eval(inputs: dict[str, Any]) -> dict[str, Any]:
    bsz = int(inputs['B'])
    n_points = int(inputs['N'])
    dim = int(inputs['D'])
    n_clusters = int(inputs['K'])
    _validate_shape(inputs, bsz=bsz, n_points=n_points, dim=dim, n_clusters=n_clusters)
    num_n_tiles = n_points // BLOCK_N
    k_tiles = n_clusters // BLOCK_K
    if not _use_splitk(dim=dim, num_n_tiles=num_n_tiles, k_tiles=k_tiles):
        outputs = _direct.launch_for_eval(inputs)
        trace = dict(outputs.get('route_trace', {}))
        trace['selected_route'] = 'd288_exactd_a532_v1'
        outputs['route_trace'] = trace
        return outputs
    _launch_splitk(inputs, bsz=bsz, n_points=n_points, dim=dim, n_clusters=n_clusters, num_n_tiles=num_n_tiles, k_tiles=k_tiles)
    total_work = bsz * num_n_tiles * k_tiles
    trace = _route_trace(inputs, total_work=total_work, grid=(min(total_work, SPLITK_GRID_CAP), 1, 1))
    return {'cluster_ids': inputs['out'], 'selected_route': ROUTE_ID, 'route_trace': trace}

def _validate_shape(inputs: dict[str, Any], *, bsz: int, n_points: int, dim: int, n_clusters: int) -> None:
    del bsz
    dtype_name = _dtype_name(inputs)
    if dtype_name not in BF16_DTYPE_NAMES:
        raise ValueError(''.join(['flash_kmeans_assign_d288_splitk_cta_0438_v1 requires bfloat16 input, got ', format(dtype_name, '')]))
    if dim not in SUPPORTED_DIMS:
        raise ValueError(''.join(['flash_kmeans_assign_d288_splitk_cta_0438_v1 supports D=', format(sorted(SUPPORTED_DIMS), ''), ', got ', format(dim, '')]))
    if n_points % BLOCK_N != 0:
        raise ValueError(''.join(['N must be divisible by BLOCK_N=', format(BLOCK_N, ''), ', got ', format(n_points, '')]))
    if n_clusters % BLOCK_K != 0:
        raise ValueError(''.join(['K must be divisible by BLOCK_K=', format(BLOCK_K, ''), ', got ', format(n_clusters, '')]))

def _use_splitk(*, dim: int, num_n_tiles: int, k_tiles: int) -> bool:
    if num_n_tiles > 16:
        return False
    return k_tiles >= SPLITK_MIN_K_TILES

def _launch_splitk(inputs: dict[str, Any], *, bsz: int, n_points: int, dim: int, n_clusters: int, num_n_tiles: int, k_tiles: int) -> None:
    from .._dispatch_runtime import CUDAKernel
    from .._dispatch_runtime import pack_kernel_args
    total_work = bsz * num_n_tiles * k_tiles
    partial_scores, partial_indices = _partial_buffers(inputs, total_work)
    tmap_x, tmap_c = _direct._make_tmaps(inputs)
    cubin, kernel_name, smem_bytes, threads = _compiled_partial_kernel()
    partial_args = pack_kernel_args(flash_kmeans_assign_d288_splitk_cta_0438_v1_partial, x_tmap=tmap_x, c_tmap=tmap_c, c_sq=inputs['c_sq'], partial_scores=partial_scores, partial_indices=partial_indices, B=bsz, N=n_points, D=dim, K=n_clusters, num_n_tiles=num_n_tiles, K_tiles=k_tiles)
    with CUDAKernel(cubin, kernel_name) as kernel:
        kernel.launch(grid=(min(total_work, SPLITK_GRID_CAP), 1, 1), block=(threads, 1, 1), args=partial_args, shared_mem=smem_bytes)
    cubin, kernel_name, smem_bytes, threads = _compiled_reduce_kernel()
    reduce_args = pack_kernel_args(reduce_ir, partial_scores=partial_scores, partial_indices=partial_indices, out=inputs['out'], B=bsz, N=n_points, K=n_clusters, num_n_tiles=num_n_tiles, K_tiles=k_tiles)
    with CUDAKernel(cubin, kernel_name) as kernel:
        kernel.launch(grid=(min(bsz * num_n_tiles, SPLITK_GRID_CAP), 1, 1), block=(threads, 1, 1), args=reduce_args, shared_mem=smem_bytes)

def _partial_buffers(inputs: dict[str, Any], total_work: int) -> tuple[Any, Any]:
    import torch
    cache_key = stream_cache_key(inputs, int(inputs['x'].data_ptr()), int(inputs['centroids'].data_ptr()), int(inputs['B']), int(inputs['N']), int(inputs['K']), int(inputs['D']), int(total_work))
    cache = inputs.setdefault('_flash_kmeans_assign_d288_splitk_cta_0438_v1_partials', {})
    cached = cache.get(cache_key)
    if cached is not None:
        return cached
    scores = torch.empty((total_work, BLOCK_N), dtype=torch.float32, device=inputs['x'].device)
    indices = torch.empty((total_work, BLOCK_N), dtype=torch.int32, device=inputs['x'].device)
    cache[cache_key] = (scores, indices)
    return (scores, indices)

def _route_trace(inputs: dict[str, Any], *, total_work: int, grid: tuple[int, int, int]) -> dict[str, Any]:
    return {'shape_key': _shape_key(inputs), 'selected_route': ROUTE_ID, 'selected_entrypoint': 'loom.examples.weave.flash_kmeans_assign_d288_splitk_cta_0438_v1:launch_for_eval', 'selected_seed': SEED_ID, 'expected_seed': None, 'route_kind': 'specialized', 'route_source': 'shape-specific-seed', 'guard_id': 'guard_d288_splitk_cta_0438_v1', 'guard_condition': 'dtype == bfloat16 and D == 288 and N % 128 == 0 and K % 256 == 0 and num_n_tiles <= 16 and K_tiles >= 8', 'classification': 'route-ok', 'dispatcher_kernel_ms': None, 'shape_specific_kernel_ms': None, 'relative_speedup_vs_baseline': None, 'total_tiles': total_work, 'launch_grid': grid, 'reason': 'D288 split-K seed exposes K tiles as CTA work then reduces partial argmins'}

def compile_and_launch_flash_kmeans_assign_d288_splitk_cta_0438_v1(B: int=1, N: int=512, K: int=8192, D: int=288, *, benchmark: bool=False) -> dict[str, Any]:
    import torch
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA GPU required')
    torch.manual_seed(51204)
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
        result['contract_eval'] = evaluate(launch_for_eval, shapes=[{'label': ''.join(['manual_b', format(B, ''), '_n', format(N, ''), '_k', format(K, ''), '_d', format(D, '')]), 'params': {'B': B, 'N': N, 'D': D, 'K': K, 'dtype': 'bfloat16', 'seed': 51204}}])
    return result

def _shape_key(inputs: dict[str, Any]) -> str:
    return ''.join(['B=', format(int(inputs['B']), ''), ',N=', format(int(inputs['N']), ''), ',K=', format(int(inputs['K']), ''), ',D=', format(int(inputs['D']), '')])

def _dtype_name(inputs: dict[str, Any]) -> str:
    return str(inputs.get('dtype', inputs['x'].dtype)).replace('torch.', '')
