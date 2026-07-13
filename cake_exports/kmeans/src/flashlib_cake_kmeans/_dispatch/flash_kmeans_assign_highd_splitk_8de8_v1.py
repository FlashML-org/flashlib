"""Flash-KMeans Euclidean assignment high-D split-K candidate.

Minimum architecture: sm_100a. This candidate uses Blackwell tcgen05/TMEM
(``lm.mma``) for D in {320, 384, 448, 512}. High-K rows split the cluster
tile loop across CTAs, write one partial argmin per K tile, then reduce those
partials with a second Weave kernel. Non-splitK rows reuse the validated 6fcf
split-D seed. It is not intended for sm_120a/sm_121a where ptxas rejects
tcgen05 instructions.
"""
from __future__ import annotations
from json import loads as _json_loads
from .._dispatch_runtime import _capture_cuTensorMapEncodeTiled, _decode_capture, _import_dispatch_module, _ir_proxy
import os
from functools import lru_cache
from typing import Any
from . import flash_kmeans_assign_highd_splitd_6fcf_v1 as _splitd
from .flash_kmeans_assign_stream import stream_cache_key
BLOCK_N = _splitd.BLOCK_N
BLOCK_K = _splitd.BLOCK_K
FEAT_CHUNK = _splitd.FEAT_CHUNK
SCORE_CHUNK_K = _splitd.SCORE_CHUNK_K
CSQ_VEC = _splitd.CSQ_VEC
CSQ_STAGE_VEC = _splitd.CSQ_STAGE_VEC
NUM_COMPUTE_WARPS = _splitd.NUM_COMPUTE_WARPS
X_TILE_BYTES = _splitd.X_TILE_BYTES
C_TILE_BYTES = _splitd.C_TILE_BYTES
CSQ_TILE_BYTES = _splitd.CSQ_TILE_BYTES
SUPPORTED_DIMS = set(_splitd.SUPPORTED_DIMS)
BF16_DTYPE_NAMES = set(_splitd.BF16_DTYPE_NAMES)
SPLITK_GRID_CAP = 4096
REDUCE_THREADS = 128
SPLITK_MIN_K_TILES = 16
ROUTE_ID = 'highd_splitk_8de8_v1'
SEED_ID = 'highd-splitk-8de8-v1'
flash_kmeans_assign_highd_splitk_partial_8de8_v1 = _decode_capture(_json_loads('{"__ir__": "flash_kmeans_assign_highd_splitk_partial_8de8_v1", "arg_keys": ["x_tmap", "c_tmap", "c_sq", "partial_scores", "partial_indices", "B", "N", "D", "K", "num_n_tiles", "K_tiles"], "cluster_dims": [1, 1, 1], "computed_smem_bytes": 51200, "constants": [], "cta_group": 1, "threads": 192}'))
flash_kmeans_assign_highd_splitk_reduce_8de8_v1 = _decode_capture(_json_loads('{"__ir__": "flash_kmeans_assign_highd_splitk_reduce_8de8_v1", "arg_keys": ["partial_scores", "partial_indices", "out", "B", "N", "K", "num_n_tiles", "K_tiles"], "cluster_dims": [1, 1, 1], "computed_smem_bytes": 0, "constants": [], "cta_group": 1, "threads": 128}'))
partial_ir = _decode_capture(_json_loads('{"__ir__": "flash_kmeans_assign_highd_splitk_partial_8de8_v1", "arg_keys": ["x_tmap", "c_tmap", "c_sq", "partial_scores", "partial_indices", "B", "N", "D", "K", "num_n_tiles", "K_tiles"], "cluster_dims": [1, 1, 1], "computed_smem_bytes": 51200, "constants": [], "cta_group": 1, "threads": 192}'))
reduce_ir = _decode_capture(_json_loads('{"__ir__": "flash_kmeans_assign_highd_splitk_reduce_8de8_v1", "arg_keys": ["partial_scores", "partial_indices", "out", "B", "N", "K", "num_n_tiles", "K_tiles"], "cluster_dims": [1, 1, 1], "computed_smem_bytes": 0, "constants": [], "cta_group": 1, "threads": 128}'))

def _verify_export_ir() -> Any:
    if os.environ.get('LOOM_FLASH_KMEANS_HIGHD_SPLITK_VERIFY_KERNEL') == 'reduce':
        return reduce_ir
    return partial_ir
ir = _decode_capture(_json_loads('{"__ir__": "flash_kmeans_assign_highd_splitk_partial_8de8_v1", "arg_keys": ["x_tmap", "c_tmap", "c_sq", "partial_scores", "partial_indices", "B", "N", "D", "K", "num_n_tiles", "K_tiles"], "cluster_dims": [1, 1, 1], "computed_smem_bytes": 51200, "constants": [], "cta_group": 1, "threads": 192}'))

def _cuda_include_dirs() -> list[str]:
    return _splitd._cuda_include_dirs()

def _compiled_partial_kernel() -> tuple[bytes, str, int, int]:
    return _decode_capture(_json_loads('{"__tuple__": [{"__kernel_source__": "dispatch_kernel_0031"}, "kernel_flash_kmeans_assign_highd_splitk_partial_8de8_v1", 51200, 192]}'))

def _compiled_reduce_kernel() -> tuple[bytes, str, int, int]:
    return _decode_capture(_json_loads('{"__tuple__": [{"__kernel_source__": "dispatch_kernel_0032"}, "kernel_flash_kmeans_assign_highd_splitk_reduce_8de8_v1", 0, 128]}'))

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
        outputs = _splitd.launch_for_eval(inputs)
        trace = dict(outputs.get('route_trace', {}))
        trace['selected_route'] = _splitd.ROUTE_ID
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
        raise ValueError(''.join(['flash_kmeans_assign_highd_splitk_8de8_v1 requires bfloat16 input, got ', format(dtype_name, '')]))
    if dim not in SUPPORTED_DIMS:
        raise ValueError(''.join(['flash_kmeans_assign_highd_splitk_8de8_v1 supports D=', format(sorted(SUPPORTED_DIMS), ''), ', got ', format(dim, '')]))
    if n_points % BLOCK_N != 0:
        raise ValueError(''.join(['N must be divisible by BLOCK_N=', format(BLOCK_N, ''), ', got ', format(n_points, '')]))
    if n_clusters % BLOCK_K != 0:
        raise ValueError(''.join(['K must be divisible by BLOCK_K=', format(BLOCK_K, ''), ', got ', format(n_clusters, '')]))

def _use_splitk(*, dim: int, num_n_tiles: int, k_tiles: int) -> bool:
    if num_n_tiles > 16:
        return False
    return k_tiles >= 32 or (k_tiles >= SPLITK_MIN_K_TILES and dim >= 448)

def _launch_splitk(inputs: dict[str, Any], *, bsz: int, n_points: int, dim: int, n_clusters: int, num_n_tiles: int, k_tiles: int) -> None:
    from .._dispatch_runtime import CUDAKernel
    from .._dispatch_runtime import pack_kernel_args
    total_work = bsz * num_n_tiles * k_tiles
    partial_scores, partial_indices = _partial_buffers(inputs, total_work)
    tmap_x, tmap_c = _splitd._make_tmaps(inputs)
    cubin, kernel_name, smem_bytes, threads = _compiled_partial_kernel()
    partial_args = pack_kernel_args(partial_ir, x_tmap=tmap_x, c_tmap=tmap_c, c_sq=inputs['c_sq'], partial_scores=partial_scores, partial_indices=partial_indices, B=bsz, N=n_points, D=dim, K=n_clusters, num_n_tiles=num_n_tiles, K_tiles=k_tiles)
    with CUDAKernel(cubin, kernel_name) as kernel:
        kernel.launch(grid=(min(total_work, SPLITK_GRID_CAP), 1, 1), block=(threads, 1, 1), args=partial_args, shared_mem=smem_bytes)
    cubin, kernel_name, smem_bytes, threads = _compiled_reduce_kernel()
    reduce_args = [partial_scores, partial_indices, inputs['out'], bsz, n_points, n_clusters, num_n_tiles, k_tiles]
    with CUDAKernel(cubin, kernel_name) as kernel:
        kernel.launch(grid=(min(bsz * num_n_tiles, SPLITK_GRID_CAP), 1, 1), block=(threads, 1, 1), args=reduce_args, shared_mem=smem_bytes)

def _partial_buffers(inputs: dict[str, Any], total_work: int) -> tuple[Any, Any]:
    import torch
    cache_key = stream_cache_key(inputs, int(inputs['x'].data_ptr()), int(inputs['centroids'].data_ptr()), int(inputs['B']), int(inputs['N']), int(inputs['K']), int(inputs['D']), int(total_work))
    cache = inputs.setdefault('_flash_kmeans_assign_highd_splitk_8de8_v1_partials', {})
    cached = cache.get(cache_key)
    if cached is not None:
        return cached
    scores = torch.empty((total_work, BLOCK_N), dtype=torch.float32, device=inputs['x'].device)
    indices = torch.empty((total_work, BLOCK_N), dtype=torch.int32, device=inputs['x'].device)
    cache[cache_key] = (scores, indices)
    return (scores, indices)

def _route_trace(inputs: dict[str, Any], *, total_work: int, grid: tuple[int, int, int]) -> dict[str, Any]:
    return {'shape_key': _shape_key(inputs), 'selected_route': ROUTE_ID, 'selected_entrypoint': 'loom.examples.weave.flash_kmeans_assign_highd_splitk_8de8_v1:launch_for_eval', 'selected_seed': SEED_ID, 'expected_seed': None, 'route_kind': 'specialized', 'route_source': 'shape-specific-seed', 'guard_id': 'guard_highd_splitk_8de8_v1', 'guard_condition': 'dtype == bfloat16 and D in [320, 384, 448, 512] and N % 128 == 0 and K % 256 == 0 and num_n_tiles <= 16 and (K_tiles >= 32 or (K_tiles >= 16 and D >= 448))', 'classification': 'route-ok', 'dispatcher_kernel_ms': None, 'shape_specific_kernel_ms': None, 'relative_speedup_vs_baseline': None, 'total_tiles': total_work, 'launch_grid': grid, 'reason': 'high-D split-K seed exposes K tiles as CTA work then reduces partial argmins'}

def compile_and_launch_flash_kmeans_assign_highd_splitk(B: int=1, N: int=512, K: int=8192, D: int=512, *, benchmark: bool=False) -> dict[str, Any]:
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
    return _splitd._shape_key(inputs)

def _dtype_name(inputs: dict[str, Any]) -> str:
    return _splitd._dtype_name(inputs)
