"""Flash-KMeans high-D paired owner-reduce round-39 candidate.

Minimum architecture: sm_100a. This additive exact-bucket seed keeps the
Blackwell tcgen05/TMEM packed-key producer on the contract-visible path for
B=1,N=2048,K=4096 high-D rows. D448 keeps the validated R1 reducer ownership
but fully unrolls its eight-slice packed-key scan after a reduce2 row-group
probe underperformed. D512 delegates to the validated gridcap160 packed-partial
route. It is not intended for sm_120a/sm_121a where ptxas rejects tcgen05
instructions.
"""
from __future__ import annotations
from json import loads as _json_loads
from .._dispatch_runtime import _capture_cuTensorMapEncodeTiled, _decode_capture, _import_dispatch_module, _ir_proxy
import os
from functools import lru_cache
from typing import Any
from . import flash_kmeans_assign_highd_paired_packedpartial_gridcap160_0194_v1 as _gridcap160
from . import flash_kmeans_assign_highd_paired_packedpartial_r1_r1c1_v1 as _r1
from . import flash_kmeans_assign_highd_paired_packedpartial_r2_7b3c_v1 as _r2
from . import flash_kmeans_assign_highd_splitk_8de8_v1 as _base
from . import flash_kmeans_assign_highd_splitk_blockn64_g2r4_b5a6_v1 as _g2r4
from .._dispatch_runtime import pack_kernel_args
BLOCK_N = _g2r4.BLOCK_N
MMA_BLOCK_K = _g2r4.MMA_BLOCK_K
SPLITK_GROUP_K_TILES = _g2r4.SPLITK_GROUP_K_TILES
SPLITK_TILE_K = _g2r4.SPLITK_TILE_K
SPLITK_GRID_CAP = _g2r4.SPLITK_GRID_CAP
BF16_DTYPE_NAMES = set(_base.BF16_DTYPE_NAMES)
PRODUCER_GRID_CAP_D448 = 128
PAIRED_K_SLICES = 8
REDUCE_ROW_GROUPS_D448 = 2
REDUCE_ROWS_PER_CTA_D448 = BLOCK_N // REDUCE_ROW_GROUPS_D448
REDUCE_THREADS_D448 = 32
REDUCE_THREADS_R1_D448 = BLOCK_N
U32_MASK = 4294967295
ROUTE_ID = 'highd_paired_ownerreduce_r39_v1'
SEED_ID = 'highd-paired-ownerreduce-r39-v1'
flash_kmeans_assign_highd_paired_ownerreduce_r39_reduce1_unroll_v1 = _decode_capture(_json_loads('{"__ir__": "flash_kmeans_assign_highd_paired_ownerreduce_r39_reduce1_unroll_v1", "arg_keys": ["partial_keys", "out", "B", "N", "K", "num_n_tiles", "K_slices"], "cluster_dims": [1, 1, 1], "computed_smem_bytes": 0, "constants": [], "cta_group": 1, "threads": 64}'))
flash_kmeans_assign_highd_paired_ownerreduce_r39_reduce2_v1 = _decode_capture(_json_loads('{"__ir__": "flash_kmeans_assign_highd_paired_ownerreduce_r39_reduce2_v1", "arg_keys": ["partial_keys", "out", "B", "N", "K", "num_n_tiles", "K_slices"], "cluster_dims": [1, 1, 1], "computed_smem_bytes": 0, "constants": [], "cta_group": 1, "threads": 32}'))
reduce1_ir = _decode_capture(_json_loads('{"__ir__": "flash_kmeans_assign_highd_paired_ownerreduce_r39_reduce1_unroll_v1", "arg_keys": ["partial_keys", "out", "B", "N", "K", "num_n_tiles", "K_slices"], "cluster_dims": [1, 1, 1], "computed_smem_bytes": 0, "constants": [], "cta_group": 1, "threads": 64}'))
reduce2_ir = _decode_capture(_json_loads('{"__ir__": "flash_kmeans_assign_highd_paired_ownerreduce_r39_reduce2_v1", "arg_keys": ["partial_keys", "out", "B", "N", "K", "num_n_tiles", "K_slices"], "cluster_dims": [1, 1, 1], "computed_smem_bytes": 0, "constants": [], "cta_group": 1, "threads": 32}'))
reduce_ir = _decode_capture(_json_loads('{"__ir__": "flash_kmeans_assign_highd_paired_ownerreduce_r39_reduce1_unroll_v1", "arg_keys": ["partial_keys", "out", "B", "N", "K", "num_n_tiles", "K_slices"], "cluster_dims": [1, 1, 1], "computed_smem_bytes": 0, "constants": [], "cta_group": 1, "threads": 64}'))

def _verify_export_ir() -> Any:
    verify_kernel = os.environ.get('LOOM_FLASH_KMEANS_HIGHD_PAIRED_OWNERREDUCE_R39_VERIFY_KERNEL')
    if verify_kernel == 'd448_reduce1':
        return reduce1_ir
    if verify_kernel == 'd448_reduce':
        return reduce2_ir
    if verify_kernel == 'd512_partial':
        return _r2.partial_ir
    if verify_kernel == 'd512_reduce':
        return _r2.reduce_ir
    return _r1.partial_ir
ir = _decode_capture(_json_loads('{"__ir__": "flash_kmeans_assign_highd_paired_packedpartial_producer_r1c1_v1", "arg_keys": ["x_tmap", "c_tmap", "c_sq", "partial_keys", "B", "N", "D", "K", "num_n_tiles", "K_tiles", "K_slices"], "cluster_dims": [1, 1, 1], "computed_smem_bytes": 43008, "constants": [], "cta_group": 1, "threads": 192}'))

@lru_cache(maxsize=1)
def _loaded_reduce2_kernel() -> tuple[Any, int, int]:
    from .._dispatch_runtime import CUDAKernel
    cubin, kernel_name, smem_bytes, threads = _base._compile_ir(reduce2_ir)
    return (CUDAKernel(cubin, kernel_name), smem_bytes, threads)

@lru_cache(maxsize=1)
def _loaded_reduce1_kernel() -> tuple[Any, int, int]:
    from .._dispatch_runtime import CUDAKernel
    cubin, kernel_name, smem_bytes, threads = _base._compile_ir(reduce1_ir)
    return (CUDAKernel(cubin, kernel_name), smem_bytes, threads)

def launch_for_eval(inputs: dict[str, Any]) -> dict[str, Any]:
    bsz = int(inputs['B'])
    n_points = int(inputs['N'])
    dim = int(inputs['D'])
    n_clusters = int(inputs['K'])
    _validate_shape(inputs, bsz=bsz, n_points=n_points, dim=dim, n_clusters=n_clusters)
    if _use_d448_custom(bsz=bsz, n_points=n_points, dim=dim, n_clusters=n_clusters):
        num_n_tiles = n_points // BLOCK_N
        k_tiles = n_clusters // MMA_BLOCK_K
        k_slices = k_tiles // SPLITK_GROUP_K_TILES
        if k_slices != PAIRED_K_SLICES:
            raise ValueError(''.join([format(ROUTE_ID, ''), ' requires K_slices=', format(PAIRED_K_SLICES, ''), ', got ', format(k_slices, '')]))
        total_work = bsz * num_n_tiles * k_slices
        producer_grid, reducer_grid = _launch_d448_reduce1(inputs, bsz=bsz, n_points=n_points, dim=dim, n_clusters=n_clusters, num_n_tiles=num_n_tiles, k_tiles=k_tiles, k_slices=k_slices, total_work=total_work)
        trace = _route_trace_d448(inputs, total_work=total_work, producer_grid=producer_grid, reducer_grid=reducer_grid, k_slices=k_slices)
        return {'cluster_ids': inputs['out'], 'selected_route': ROUTE_ID, 'route_trace': trace}
    outputs = _gridcap160.launch_for_eval(inputs)
    trace = _wrap_gridcap160_trace(inputs, child_outputs=outputs)
    return {'cluster_ids': outputs['cluster_ids'], 'selected_route': ROUTE_ID, 'route_trace': trace}

def _use_d448_custom(*, bsz: int, n_points: int, dim: int, n_clusters: int) -> bool:
    return bsz == 1 and n_points == 2048 and (n_clusters == 4096) and (dim == 448)

def _validate_shape(inputs: dict[str, Any], *, bsz: int, n_points: int, dim: int, n_clusters: int) -> None:
    dtype_name = _base._dtype_name(inputs)
    if dtype_name not in BF16_DTYPE_NAMES:
        raise ValueError(''.join([format(ROUTE_ID, ''), ' requires bfloat16 input, got ', format(dtype_name, '')]))
    if (bsz, n_points, n_clusters, dim) not in {(1, 2048, 4096, 448), (1, 2048, 4096, 512)}:
        raise ValueError(''.join([format(ROUTE_ID, ''), ' is exact-shape only for paired D448/D512 rows']))

def _launch_d448_reduce2(inputs: dict[str, Any], *, bsz: int, n_points: int, dim: int, n_clusters: int, num_n_tiles: int, k_tiles: int, k_slices: int, total_work: int) -> tuple[tuple[int, int, int], tuple[int, int, int]]:
    import torch
    partial_keys = _r1._partial_key_buffer(inputs, total_work)
    tmap_x, tmap_c = _g2r4._make_tmaps(inputs)
    stream = torch.cuda.current_stream()
    partial_kernel, smem_bytes, threads = _r1._loaded_partial_key_kernel()
    partial_args = pack_kernel_args(_r1.partial_ir, x_tmap=tmap_x, c_tmap=tmap_c, c_sq=inputs['c_sq'], partial_keys=partial_keys, B=bsz, N=n_points, D=dim, K=n_clusters, num_n_tiles=num_n_tiles, K_tiles=k_tiles, K_slices=k_slices)
    producer_grid = (min(total_work, PRODUCER_GRID_CAP_D448), 1, 1)
    partial_kernel.launch(grid=producer_grid, block=(threads, 1, 1), args=partial_args, shared_mem=smem_bytes, stream=stream)
    reduce_kernel, smem_bytes, threads = _loaded_reduce2_kernel()
    reduce_args = [partial_keys, inputs['out'], bsz, n_points, n_clusters, num_n_tiles, k_slices]
    reducer_work = bsz * num_n_tiles * REDUCE_ROW_GROUPS_D448
    reducer_grid = (min(reducer_work, SPLITK_GRID_CAP), 1, 1)
    reduce_kernel.launch(grid=reducer_grid, block=(threads, 1, 1), args=reduce_args, shared_mem=smem_bytes, stream=stream)
    return (producer_grid, reducer_grid)

def _launch_d448_reduce1(inputs: dict[str, Any], *, bsz: int, n_points: int, dim: int, n_clusters: int, num_n_tiles: int, k_tiles: int, k_slices: int, total_work: int) -> tuple[tuple[int, int, int], tuple[int, int, int]]:
    import torch
    partial_keys = _r1._partial_key_buffer(inputs, total_work)
    tmap_x, tmap_c = _g2r4._make_tmaps(inputs)
    stream = torch.cuda.current_stream()
    partial_kernel, smem_bytes, threads = _r1._loaded_partial_key_kernel()
    partial_args = pack_kernel_args(_r1.partial_ir, x_tmap=tmap_x, c_tmap=tmap_c, c_sq=inputs['c_sq'], partial_keys=partial_keys, B=bsz, N=n_points, D=dim, K=n_clusters, num_n_tiles=num_n_tiles, K_tiles=k_tiles, K_slices=k_slices)
    producer_grid = (min(total_work, PRODUCER_GRID_CAP_D448), 1, 1)
    partial_kernel.launch(grid=producer_grid, block=(threads, 1, 1), args=partial_args, shared_mem=smem_bytes, stream=stream)
    reduce_kernel, smem_bytes, threads = _loaded_reduce1_kernel()
    reduce_args = [partial_keys, inputs['out'], bsz, n_points, n_clusters, num_n_tiles, k_slices]
    reducer_grid = (min(bsz * num_n_tiles, SPLITK_GRID_CAP), 1, 1)
    reduce_kernel.launch(grid=reducer_grid, block=(threads, 1, 1), args=reduce_args, shared_mem=smem_bytes, stream=stream)
    return (producer_grid, reducer_grid)

def _route_trace_d448(inputs: dict[str, Any], *, total_work: int, producer_grid: tuple[int, int, int], reducer_grid: tuple[int, int, int], k_slices: int) -> dict[str, Any]:
    return {'shape_key': _base._shape_key(inputs), 'selected_route': ROUTE_ID, 'selected_entrypoint': 'loom.examples.weave.flash_kmeans_assign_highd_paired_ownerreduce_r39_v1:launch_for_eval', 'selected_seed': SEED_ID, 'expected_seed': None, 'route_kind': 'shape-specific-seed', 'route_source': 'shape-specific-seed', 'guard_id': 'guard_highd_paired_ownerreduce_r39_v1', 'guard_condition': 'dtype == bfloat16 and B == 1 and N == 2048 and K == 4096 and D in [448,512]; D448 uses unrolled R1 reducer ownership, D512 delegates to gridcap160', 'classification': 'route-ok', 'dispatcher_kernel_ms': None, 'shape_specific_kernel_ms': None, 'relative_speedup_vs_baseline': None, 'total_tiles': total_work, 'launch_grid': producer_grid, 'reduce_grid': reducer_grid, 'paired_packedpartial_route': True, 'selected_reducer_variant': 'r39_reduce1_unroll8', 'tile_shape': {'BLOCK_N': BLOCK_N, 'BLOCK_K': SPLITK_TILE_K, 'mma_block_k': MMA_BLOCK_K, 'split_k_slices': k_slices, 'grouped_mma_k_tiles': SPLITK_GROUP_K_TILES, 'producer_work_items': total_work, 'producer_grid_cap': PRODUCER_GRID_CAP_D448, 'producer_grid_work_items_per_cta': total_work // min(total_work, PRODUCER_GRID_CAP_D448), 'producer_to_consumer': 'u64_partial_key_buffer_plus_reduce1_unroll8', 'partial_key': 'ordered_f32_score_high32_inverse_cluster_index_low32', 'reducer_lanes_per_row': 1, 'reducer_threads': REDUCE_THREADS_R1_D448, 'producer_to_reducer_sync': 'same_stream_ordering', 'reducer_scan_unroll': PAIRED_K_SLICES}, 'reason': 'D448 keeps the R1 tcgen05 packed-key producer and row ownership but fully unrolls the eight-slice packed-key scan after the reduce2 row-group probe underfed the seed'}

def _wrap_gridcap160_trace(inputs: dict[str, Any], *, child_outputs: dict[str, Any]) -> dict[str, Any]:
    child_trace = dict(child_outputs.get('route_trace', {}))
    trace = dict(child_trace)
    trace.update({'shape_key': _base._shape_key(inputs), 'selected_route': ROUTE_ID, 'selected_entrypoint': 'loom.examples.weave.flash_kmeans_assign_highd_paired_ownerreduce_r39_v1:launch_for_eval', 'selected_seed': SEED_ID, 'expected_seed': None, 'route_kind': 'shape-specific-seed-composite', 'route_source': 'shape-specific-seed', 'guard_id': 'guard_highd_paired_ownerreduce_r39_v1', 'guard_condition': 'dtype == bfloat16 and B == 1 and N == 2048 and K == 4096 and D in [448,512]; D448 uses unrolled R1 reducer ownership, D512 delegates to gridcap160', 'classification': 'route-ok', 'child_route': child_outputs.get('selected_route') or child_trace.get('selected_route'), 'child_entrypoint': child_trace.get('selected_entrypoint'), 'child_guard_id': child_trace.get('guard_id'), 'child_tile_shape': child_trace.get('tile_shape'), 'selected_reducer_variant': 'gridcap160_child', 'paired_packedpartial_route': True, 'reason': 'D512 keeps the validated gridcap160 producer cap and R2 reducer child'})
    return trace

def compile_and_launch_flash_kmeans_assign_highd_paired_ownerreduce_r39(B: int=1, N: int=2048, K: int=4096, D: int=448, *, benchmark: bool=False) -> dict[str, Any]:
    import torch
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA GPU required')
    seed = 44801 if D == 448 else 51202
    torch.manual_seed(seed)
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
        result['contract_eval'] = evaluate(launch_for_eval, shapes=[{'label': ''.join(['manual_b', format(B, ''), '_n', format(N, ''), '_k', format(K, ''), '_d', format(D, '')]), 'params': {'B': B, 'N': N, 'D': D, 'K': K, 'dtype': 'bfloat16', 'seed': seed}}])
    return result
