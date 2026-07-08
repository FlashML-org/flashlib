"""Flash-KMeans high-D Split-K BLOCK_N=64 G1/R4 streamdep round-63 candidate.

Minimum architecture: sm_100a. This additive child keeps the M=64 ROW_16x256B
partial geometry but reduces the grouped producer from two MMA K tiles per
work item to one. The target is to double producer work feed for no-padding
D448/D512 mid-K rows while preserving the four-lane reducer and same-stream
producer-to-reducer ordering. It is not intended for sm_120a/sm_121a where
ptxas rejects tcgen05 instructions.
"""
from __future__ import annotations
from json import loads as _json_loads
from .._dispatch_runtime import _capture_cuTensorMapEncodeTiled, _decode_capture, _import_dispatch_module, _ir_proxy
import os
from functools import lru_cache
from typing import Any
from . import flash_kmeans_assign_highd_splitk_8de8_v1 as _base
from .flash_kmeans_assign_stream import stream_cache_key
from .._dispatch_runtime import pack_kernel_args
BLOCK_N = 64
MMA_BLOCK_K = _base.BLOCK_K
FEAT_CHUNK = _base.FEAT_CHUNK
ROW16_LOAD_COUNT = 64
ROW16_LOGICAL_CHUNK_K = 128
CSQ_STAGE_VEC = _base.CSQ_STAGE_VEC
X_TILE_BYTES = BLOCK_N * FEAT_CHUNK * 2
C_TILE_BYTES = MMA_BLOCK_K * FEAT_CHUNK * 2
CSQ_TILE_BYTES = MMA_BLOCK_K * 4
SUPPORTED_DIMS = set(_base.SUPPORTED_DIMS)
BF16_DTYPE_NAMES = set(_base.BF16_DTYPE_NAMES)
SPLITK_GRID_CAP = _base.SPLITK_GRID_CAP
SPLITK_MIN_K_TILES = _base.SPLITK_MIN_K_TILES
SPLITK_GROUP_K_TILES = 1
SPLITK_TILE_K = MMA_BLOCK_K * SPLITK_GROUP_K_TILES
MAX_POINT_TILES = 32
NUM_COMPUTE_WARPS = 4
REDUCE_LANES_PER_ROW = 4
REDUCE_THREADS = BLOCK_N * REDUCE_LANES_PER_ROW
ROUTE_ID = 'highd_splitk_blockn64_g1r4_streamdep_r63_v1'
SEED_ID = 'highd-splitk-blockn64-g1r4-streamdep-r63-v1'
VERIFY_ENV = 'LOOM_FLASH_KMEANS_HIGHD_SPLITK_BLOCKN64_G1R4_STREAMDEP_R63_VERIFY_KERNEL'
flash_kmeans_assign_highd_splitk_partial_blockn64_g1r4_streamdep_r63_v1 = _decode_capture(_json_loads('{"__ir__": "flash_kmeans_assign_highd_splitk_partial_blockn64_g1r4_streamdep_r63_v1", "arg_keys": ["x_tmap", "c_tmap", "c_sq", "partial_scores", "partial_indices", "B", "N", "D", "K", "num_n_tiles", "K_tiles", "K_slices"], "cluster_dims": [1, 1, 1], "computed_smem_bytes": 43008, "constants": [], "cta_group": 1, "threads": 192}'))
flash_kmeans_assign_highd_splitk_reduce_blockn64_g1r4_streamdep_r63_v1 = _decode_capture(_json_loads('{"__ir__": "flash_kmeans_assign_highd_splitk_reduce_blockn64_g1r4_streamdep_r63_v1", "arg_keys": ["partial_scores", "partial_indices", "out", "B", "N", "K", "num_n_tiles", "K_slices"], "cluster_dims": [1, 1, 1], "computed_smem_bytes": 0, "constants": [], "cta_group": 1, "threads": 256}'))
partial_ir = _decode_capture(_json_loads('{"__ir__": "flash_kmeans_assign_highd_splitk_partial_blockn64_g1r4_streamdep_r63_v1", "arg_keys": ["x_tmap", "c_tmap", "c_sq", "partial_scores", "partial_indices", "B", "N", "D", "K", "num_n_tiles", "K_tiles", "K_slices"], "cluster_dims": [1, 1, 1], "computed_smem_bytes": 43008, "constants": [], "cta_group": 1, "threads": 192}'))
reduce_ir = _decode_capture(_json_loads('{"__ir__": "flash_kmeans_assign_highd_splitk_reduce_blockn64_g1r4_streamdep_r63_v1", "arg_keys": ["partial_scores", "partial_indices", "out", "B", "N", "K", "num_n_tiles", "K_slices"], "cluster_dims": [1, 1, 1], "computed_smem_bytes": 0, "constants": [], "cta_group": 1, "threads": 256}'))

def _verify_export_ir() -> Any:
    if os.environ.get(VERIFY_ENV) == 'reduce':
        return reduce_ir
    return partial_ir
ir = _decode_capture(_json_loads('{"__ir__": "flash_kmeans_assign_highd_splitk_partial_blockn64_g1r4_streamdep_r63_v1", "arg_keys": ["x_tmap", "c_tmap", "c_sq", "partial_scores", "partial_indices", "B", "N", "D", "K", "num_n_tiles", "K_tiles", "K_slices"], "cluster_dims": [1, 1, 1], "computed_smem_bytes": 43008, "constants": [], "cta_group": 1, "threads": 192}'))

def _compiled_partial_kernel() -> tuple[bytes, str, int, int]:
    return _decode_capture(_json_loads('{"__tuple__": [{"__kernel_source__": "dispatch_kernel_0328"}, "kernel_flash_kmeans_assign_highd_splitk_partial_blockn64_g1r4_streamdep_r63_v1", 43008, 192]}'))

def _compiled_reduce_kernel() -> tuple[bytes, str, int, int]:
    return _decode_capture(_json_loads('{"__tuple__": [{"__kernel_source__": "dispatch_kernel_0329"}, "kernel_flash_kmeans_assign_highd_splitk_reduce_blockn64_g1r4_streamdep_r63_v1", 0, 256]}'))

@lru_cache(maxsize=1)
def _loaded_partial_kernel() -> tuple[Any, int, int]:
    from .._dispatch_runtime import CUDAKernel
    cubin, kernel_name, smem_bytes, threads = _compiled_partial_kernel()
    return (CUDAKernel(cubin, kernel_name), smem_bytes, threads)

@lru_cache(maxsize=1)
def _loaded_reduce_kernel() -> tuple[Any, int, int]:
    from .._dispatch_runtime import CUDAKernel
    cubin, kernel_name, smem_bytes, threads = _compiled_reduce_kernel()
    return (CUDAKernel(cubin, kernel_name), smem_bytes, threads)

def launch_for_eval(inputs: dict[str, Any]) -> dict[str, Any]:
    bsz = int(inputs['B'])
    n_points = int(inputs['N'])
    dim = int(inputs['D'])
    n_clusters = int(inputs['K'])
    _validate_shape(inputs, bsz=bsz, n_points=n_points, dim=dim, n_clusters=n_clusters)
    num_n_tiles = n_points // BLOCK_N
    k_tiles = n_clusters // MMA_BLOCK_K
    if not _use_blockn64_splitk(dim=dim, num_n_tiles=num_n_tiles, k_tiles=k_tiles):
        outputs = _base.launch_for_eval(inputs)
        trace = dict(outputs.get('route_trace', {}))
        trace['selected_route'] = _base.ROUTE_ID
        trace['fallback_from'] = ROUTE_ID
        trace['fallback_reason'] = 'shape does not satisfy BLOCK_N=64 grouped Split-K constraints'
        outputs['route_trace'] = trace
        return outputs
    k_slices = k_tiles // SPLITK_GROUP_K_TILES
    _launch_blockn64_splitk(inputs, bsz=bsz, n_points=n_points, dim=dim, n_clusters=n_clusters, num_n_tiles=num_n_tiles, k_tiles=k_tiles, k_slices=k_slices)
    total_work = bsz * num_n_tiles * k_slices
    trace = _route_trace(inputs, total_work=total_work, grid=(min(total_work, SPLITK_GRID_CAP), 1, 1), k_slices=k_slices)
    return {'cluster_ids': inputs['out'], 'selected_route': ROUTE_ID, 'route_trace': trace}

def _validate_shape(inputs: dict[str, Any], *, bsz: int, n_points: int, dim: int, n_clusters: int) -> None:
    del bsz
    dtype_name = _base._dtype_name(inputs)
    if dtype_name not in BF16_DTYPE_NAMES:
        raise ValueError(''.join([format(ROUTE_ID, ''), ' requires bfloat16 input, got ', format(dtype_name, '')]))
    if dim not in SUPPORTED_DIMS:
        raise ValueError(''.join([format(ROUTE_ID, ''), ' supports D=', format(sorted(SUPPORTED_DIMS), ''), ', got ', format(dim, '')]))
    if n_points % BLOCK_N != 0:
        raise ValueError(''.join(['N must be divisible by BLOCK_N=', format(BLOCK_N, ''), ', got ', format(n_points, '')]))
    if n_clusters % MMA_BLOCK_K != 0:
        raise ValueError(''.join(['K must be divisible by MMA_BLOCK_K=', format(MMA_BLOCK_K, ''), ', got ', format(n_clusters, '')]))

def _use_blockn64_splitk(*, dim: int, num_n_tiles: int, k_tiles: int) -> bool:
    if num_n_tiles > MAX_POINT_TILES:
        return False
    if k_tiles % SPLITK_GROUP_K_TILES != 0:
        return False
    return k_tiles >= 32 or (k_tiles >= SPLITK_MIN_K_TILES and dim >= 448)

def _launch_blockn64_splitk(inputs: dict[str, Any], *, bsz: int, n_points: int, dim: int, n_clusters: int, num_n_tiles: int, k_tiles: int, k_slices: int) -> None:
    import torch
    total_work = bsz * num_n_tiles * k_slices
    partial_scores, partial_indices = _partial_buffers(inputs, total_work)
    tmap_x, tmap_c = _make_tmaps(inputs)
    stream = torch.cuda.current_stream()
    partial_kernel, smem_bytes, threads = _loaded_partial_kernel()
    partial_args = pack_kernel_args(partial_ir, x_tmap=tmap_x, c_tmap=tmap_c, c_sq=inputs['c_sq'], partial_scores=partial_scores, partial_indices=partial_indices, B=bsz, N=n_points, D=dim, K=n_clusters, num_n_tiles=num_n_tiles, K_tiles=k_tiles, K_slices=k_slices)
    partial_kernel.launch(grid=(min(total_work, SPLITK_GRID_CAP), 1, 1), block=(threads, 1, 1), args=partial_args, shared_mem=smem_bytes, stream=stream)
    reduce_kernel, smem_bytes, threads = _loaded_reduce_kernel()
    reduce_args = [partial_scores, partial_indices, inputs['out'], bsz, n_points, n_clusters, num_n_tiles, k_slices]
    reduce_kernel.launch(grid=(min(bsz * num_n_tiles, SPLITK_GRID_CAP), 1, 1), block=(threads, 1, 1), args=reduce_args, shared_mem=smem_bytes, stream=stream)

def _make_tmaps(inputs: dict[str, Any]) -> tuple[Any, Any]:
    cache_key = (int(inputs['x'].data_ptr()), int(inputs['centroids'].data_ptr()), int(inputs['B']), int(inputs['N']), int(inputs['K']), int(inputs['D']), int(BLOCK_N))
    cached = inputs.get('_flash_kmeans_assign_highd_splitk_blockn64_g1r4_streamdep_r63_v1_tmaps')
    if cached is not None and cached[0] == cache_key:
        return (cached[1], cached[2])
    from .._dispatch_runtime import create_tensor_map_3d
    bsz = int(inputs['B'])
    n_points = int(inputs['N'])
    n_clusters = int(inputs['K'])
    dim = int(inputs['D'])
    tmap_x = create_tensor_map_3d(inputs['x'].data_ptr(), bsz * n_points, BLOCK_N, dim, FEAT_CHUNK)
    tmap_c = create_tensor_map_3d(inputs['centroids'].data_ptr(), bsz * n_clusters, MMA_BLOCK_K, dim, FEAT_CHUNK)
    inputs['_flash_kmeans_assign_highd_splitk_blockn64_g1r4_streamdep_r63_v1_tmaps'] = (cache_key, tmap_x, tmap_c)
    return (tmap_x, tmap_c)

def _partial_buffers(inputs: dict[str, Any], total_work: int) -> tuple[Any, Any]:
    import torch
    cache_key = stream_cache_key(inputs, int(inputs['x'].data_ptr()), int(inputs['centroids'].data_ptr()), int(inputs['B']), int(inputs['N']), int(inputs['K']), int(inputs['D']), int(total_work), int(BLOCK_N), int(SPLITK_GROUP_K_TILES))
    cache = inputs.setdefault('_flash_kmeans_assign_highd_splitk_blockn64_g1r4_streamdep_r63_v1_partials', {})
    cached = cache.get(cache_key)
    if cached is not None:
        return cached
    scores = torch.empty((total_work, BLOCK_N), dtype=torch.float32, device=inputs['x'].device)
    indices = torch.empty((total_work, BLOCK_N), dtype=torch.int32, device=inputs['x'].device)
    cache[cache_key] = (scores, indices)
    return (scores, indices)

def _route_trace(inputs: dict[str, Any], *, total_work: int, grid: tuple[int, int, int], k_slices: int) -> dict[str, Any]:
    return {'shape_key': _base._shape_key(inputs), 'selected_route': ROUTE_ID, 'selected_entrypoint': 'loom.examples.weave.flash_kmeans_assign_highd_splitk_blockn64_g1r4_streamdep_r63_v1:launch_for_eval', 'selected_seed': SEED_ID, 'expected_seed': None, 'route_kind': 'specialized', 'route_source': 'shape-specific-seed', 'guard_id': 'guard_highd_splitk_blockn64_g1r4_streamdep_r63_v1', 'guard_condition': 'dtype == bfloat16 and D in [320, 384, 448, 512] and N % 64 == 0 and K % 256 == 0 and num_n_tiles <= 32 and (K_tiles >= 32 or (K_tiles >= 16 and D >= 448))', 'classification': 'route-ok', 'dispatcher_kernel_ms': None, 'shape_specific_kernel_ms': None, 'relative_speedup_vs_baseline': None, 'total_tiles': total_work, 'launch_grid': grid, 'tile_shape': {'BLOCK_N': BLOCK_N, 'BLOCK_K': SPLITK_TILE_K, 'mma_block_k': MMA_BLOCK_K, 'split_k_slices': k_slices, 'grouped_mma_k_tiles': SPLITK_GROUP_K_TILES, 'cluster_grouping': 1, 'tmem_layout': 'ROW_16x256B', 'producer_to_reducer_sync': 'same_stream_ordering', 'cuda_module_cache': 'persistent_per_process', 'producer_work_feed': '2x_g2_for_same_N_K'}, 'reason': 'BLOCK_N=64 G1/R4 tile-search variant uses ROW_16x256B score drains, single-MMA producer slices, and four-lane Split-K reducer'}

def compile_and_launch_flash_kmeans_assign_highd_splitk_blockn64_g1r4(B: int=1, N: int=512, K: int=8192, D: int=512, *, benchmark: bool=False) -> dict[str, Any]:
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
