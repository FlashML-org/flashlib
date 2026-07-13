"""Flash-KMeans high-D Split-K G2/R4 stream-dependency candidate.

Minimum architecture: sm_100a. This candidate reuses the b5a6 BLOCK_N=64
G2/R4 tcgen05/TMEM partial producer and four-lane Split-K reducer unchanged,
but queues the reducer after the producer on the same CUDA stream instead of
waiting on the host between launches. It also keeps the loaded CUDA modules
cached across contract iterations so the secondary launch can be queued with
less inter-kernel idle time. It is not intended for sm_120a/sm_121a where
ptxas rejects tcgen05 instructions.
"""
from __future__ import annotations
from json import loads as _json_loads
from .._dispatch_runtime import _capture_cuTensorMapEncodeTiled, _decode_capture, _import_dispatch_module, _ir_proxy
import os
from functools import lru_cache
from typing import Any
from . import flash_kmeans_assign_highd_splitk_blockn64_g2r4_b5a6_v1 as _g2r4
from .._dispatch_runtime import pack_kernel_args
BLOCK_N = _g2r4.BLOCK_N
MMA_BLOCK_K = _g2r4.MMA_BLOCK_K
SPLITK_GROUP_K_TILES = _g2r4.SPLITK_GROUP_K_TILES
SPLITK_TILE_K = _g2r4.SPLITK_TILE_K
SPLITK_GRID_CAP = _g2r4.SPLITK_GRID_CAP
ROUTE_ID = 'highd_splitk_blockn64_g2r4_streamdep_4f2c_v1'
SEED_ID = 'highd-splitk-blockn64-g2r4-streamdep-4f2c-v1'
partial_ir = _decode_capture(_json_loads('{"__ir__": "flash_kmeans_assign_highd_splitk_partial_blockn64_g2r4_b5a6_v1", "arg_keys": ["x_tmap", "c_tmap", "c_sq", "partial_scores", "partial_indices", "B", "N", "D", "K", "num_n_tiles", "K_tiles", "K_slices"], "cluster_dims": [1, 1, 1], "computed_smem_bytes": 43008, "constants": [], "cta_group": 1, "threads": 192}'))
reduce_ir = _decode_capture(_json_loads('{"__ir__": "flash_kmeans_assign_highd_splitk_reduce_blockn64_g2r4_b5a6_v1", "arg_keys": ["partial_scores", "partial_indices", "out", "B", "N", "K", "num_n_tiles", "K_slices"], "cluster_dims": [1, 1, 1], "computed_smem_bytes": 0, "constants": [], "cta_group": 1, "threads": 256}'))

def _verify_export_ir() -> Any:
    if os.environ.get('LOOM_FLASH_KMEANS_HIGHD_SPLITK_BLOCKN64_G2R4_STREAMDEP_4F2C_VERIFY_KERNEL') == 'reduce':
        return reduce_ir
    return partial_ir
ir = _decode_capture(_json_loads('{"__ir__": "flash_kmeans_assign_highd_splitk_partial_blockn64_g2r4_b5a6_v1", "arg_keys": ["x_tmap", "c_tmap", "c_sq", "partial_scores", "partial_indices", "B", "N", "D", "K", "num_n_tiles", "K_tiles", "K_slices"], "cluster_dims": [1, 1, 1], "computed_smem_bytes": 43008, "constants": [], "cta_group": 1, "threads": 192}'))

@lru_cache(maxsize=1)
def _loaded_partial_kernel() -> tuple[Any, int, int]:
    from .._dispatch_runtime import CUDAKernel
    cubin, kernel_name, smem_bytes, threads = _g2r4._compiled_partial_kernel()
    return (CUDAKernel(cubin, kernel_name), smem_bytes, threads)

@lru_cache(maxsize=1)
def _loaded_reduce_kernel() -> tuple[Any, int, int]:
    from .._dispatch_runtime import CUDAKernel
    cubin, kernel_name, smem_bytes, threads = _g2r4._compiled_reduce_kernel()
    return (CUDAKernel(cubin, kernel_name), smem_bytes, threads)

def launch_for_eval(inputs: dict[str, Any]) -> dict[str, Any]:
    bsz = int(inputs['B'])
    n_points = int(inputs['N'])
    dim = int(inputs['D'])
    n_clusters = int(inputs['K'])
    _g2r4._validate_shape(inputs, bsz=bsz, n_points=n_points, dim=dim, n_clusters=n_clusters)
    num_n_tiles = n_points // BLOCK_N
    k_tiles = n_clusters // MMA_BLOCK_K
    if not _g2r4._use_blockn64_splitk(dim=dim, num_n_tiles=num_n_tiles, k_tiles=k_tiles):
        outputs = _g2r4.launch_for_eval(inputs)
        trace = dict(outputs.get('route_trace', {}))
        trace['selected_route'] = trace.get('selected_route', _g2r4.ROUTE_ID)
        trace['fallback_from'] = ROUTE_ID
        trace['fallback_reason'] = 'shape does not satisfy BLOCK_N=64 grouped Split-K constraints'
        outputs['route_trace'] = trace
        return outputs
    k_slices = k_tiles // SPLITK_GROUP_K_TILES
    total_work = bsz * num_n_tiles * k_slices
    _launch_stream_ordered_splitk(inputs, bsz=bsz, n_points=n_points, dim=dim, n_clusters=n_clusters, num_n_tiles=num_n_tiles, k_tiles=k_tiles, k_slices=k_slices, total_work=total_work)
    trace = _route_trace(inputs, total_work=total_work, grid=(min(total_work, SPLITK_GRID_CAP), 1, 1), k_slices=k_slices)
    return {'cluster_ids': inputs['out'], 'selected_route': ROUTE_ID, 'route_trace': trace}

def _launch_stream_ordered_splitk(inputs: dict[str, Any], *, bsz: int, n_points: int, dim: int, n_clusters: int, num_n_tiles: int, k_tiles: int, k_slices: int, total_work: int) -> None:
    import torch
    partial_scores, partial_indices = _g2r4._partial_buffers(inputs, total_work)
    tmap_x, tmap_c = _g2r4._make_tmaps(inputs)
    stream = torch.cuda.current_stream()
    partial_kernel, smem_bytes, threads = _loaded_partial_kernel()
    partial_args = pack_kernel_args(partial_ir, x_tmap=tmap_x, c_tmap=tmap_c, c_sq=inputs['c_sq'], partial_scores=partial_scores, partial_indices=partial_indices, B=bsz, N=n_points, D=dim, K=n_clusters, num_n_tiles=num_n_tiles, K_tiles=k_tiles, K_slices=k_slices)
    partial_kernel.launch(grid=(min(total_work, SPLITK_GRID_CAP), 1, 1), block=(threads, 1, 1), args=partial_args, shared_mem=smem_bytes, stream=stream)
    reduce_kernel, smem_bytes, threads = _loaded_reduce_kernel()
    reduce_args = [partial_scores, partial_indices, inputs['out'], bsz, n_points, n_clusters, num_n_tiles, k_slices]
    reduce_kernel.launch(grid=(min(bsz * num_n_tiles, SPLITK_GRID_CAP), 1, 1), block=(threads, 1, 1), args=reduce_args, shared_mem=smem_bytes, stream=stream)

def _route_trace(inputs: dict[str, Any], *, total_work: int, grid: tuple[int, int, int], k_slices: int) -> dict[str, Any]:
    trace = _g2r4._route_trace(inputs, total_work=total_work, grid=grid, k_slices=k_slices)
    trace.update({'selected_route': ROUTE_ID, 'selected_entrypoint': 'loom.examples.weave.flash_kmeans_assign_highd_splitk_blockn64_g2r4_streamdep_4f2c_v1:launch_for_eval', 'selected_seed': SEED_ID, 'guard_id': 'guard_highd_splitk_blockn64_g2r4_streamdep_4f2c_v1', 'guard_condition': 'dtype == bfloat16 and D in [320, 384, 448, 512] and N % 64 == 0 and K % 256 == 0 and num_n_tiles <= 32 and K_tiles % 2 == 0 and (K_tiles >= 32 or (K_tiles >= 16 and D >= 448))', 'tile_shape': {'BLOCK_N': BLOCK_N, 'BLOCK_K': SPLITK_TILE_K, 'mma_block_k': MMA_BLOCK_K, 'split_k_slices': k_slices, 'grouped_mma_k_tiles': SPLITK_GROUP_K_TILES, 'cluster_grouping': 1, 'tmem_layout': 'ROW_16x256B', 'producer_to_reducer_sync': 'same_stream_ordering', 'cuda_module_cache': 'persistent_per_process'}, 'reason': 'BLOCK_N=64 G2/R4 tile-search variant reuses b5a6 IR but queues producer and reducer on one stream without an intermediate host synchronize'})
    return trace

def compile_and_launch_flash_kmeans_assign_highd_splitk_blockn64(B: int=1, N: int=512, K: int=8192, D: int=448, *, benchmark: bool=False) -> dict[str, Any]:
    import torch
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA GPU required')
    torch.manual_seed(44802)
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
        result['contract_eval'] = evaluate(launch_for_eval, shapes=[{'label': ''.join(['manual_b', format(B, ''), '_n', format(N, ''), '_k', format(K, ''), '_d', format(D, '')]), 'params': {'B': B, 'N': N, 'D': D, 'K': K, 'dtype': 'bfloat16', 'seed': 44802}}])
    return result
