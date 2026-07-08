"""Flash-KMeans high-D paired D448 X-reuse dual-TMEM round-47 candidate.

Minimum architecture: sm_100a. This additive exact-bucket seed keeps the
Blackwell tcgen05/TMEM packed-key producer on the contract-visible path for
B=1,N=2048,K=4096,D=448, but reorders the paired local-K producer so each
64-wide X feature tile is loaded once and consumed by two 256-column TMEM
accumulators. The D448 reducer remains the round-39 unrolled R1 reducer, and
D512 delegates to the round-39 route. It is not intended for sm_120a/sm_121a
where ptxas rejects tcgen05 instructions.
"""
from __future__ import annotations
from json import loads as _json_loads
from .._dispatch_runtime import _capture_cuTensorMapEncodeTiled, _decode_capture, _import_dispatch_module, _ir_proxy
import os
from functools import lru_cache
from typing import Any
from . import flash_kmeans_assign_highd_paired_ownerreduce_r39_v1 as _r39
from . import flash_kmeans_assign_highd_paired_packedpartial_r1_r1c1_v1 as _r1
from . import flash_kmeans_assign_highd_splitk_8de8_v1 as _base
from . import flash_kmeans_assign_highd_splitk_blockn64_g2r4_b5a6_v1 as _g2r4
from .._dispatch_runtime import pack_kernel_args
BLOCK_N = _r39.BLOCK_N
MMA_BLOCK_K = _r39.MMA_BLOCK_K
FEAT_CHUNK = _g2r4.FEAT_CHUNK
ROW16_LOAD_COUNT = _g2r4.ROW16_LOAD_COUNT
ROW16_LOGICAL_CHUNK_K = _g2r4.ROW16_LOGICAL_CHUNK_K
CSQ_STAGE_VEC = _g2r4.CSQ_STAGE_VEC
X_TILE_BYTES = _g2r4.X_TILE_BYTES
C_TILE_BYTES = _g2r4.C_TILE_BYTES
CSQ_TILE_BYTES = _g2r4.CSQ_TILE_BYTES
SPLITK_GROUP_K_TILES = _r39.SPLITK_GROUP_K_TILES
C_PAIR_BYTES = SPLITK_GROUP_K_TILES * C_TILE_BYTES
SPLITK_TILE_K = _r39.SPLITK_TILE_K
SPLITK_GRID_CAP = _r39.SPLITK_GRID_CAP
BF16_DTYPE_NAMES = set(_base.BF16_DTYPE_NAMES)
PRODUCER_GRID_CAP_D448 = _r39.PRODUCER_GRID_CAP_D448
PAIRED_K_SLICES = _r39.PAIRED_K_SLICES
D448_FEATURE_TILES = 448 // FEAT_CHUNK
NUM_COMPUTE_WARPS = _g2r4.NUM_COMPUTE_WARPS
U32_MASK = _r39.U32_MASK
ROUTE_ID = 'highd_paired_xreuse_dualtmem_r47_v1'
SEED_ID = 'highd-paired-xreuse-dualtmem-r47-v1'
flash_kmeans_assign_highd_paired_xreuse_dualtmem_producer_r47_v1 = _decode_capture(_json_loads('{"__ir__": "flash_kmeans_assign_highd_paired_xreuse_dualtmem_producer_r47_v1", "arg_keys": ["x_tmap", "c_tmap", "c_sq", "partial_keys", "B", "N", "D", "K", "num_n_tiles", "K_tiles", "K_slices"], "cluster_dims": [1, 1, 1], "computed_smem_bytes": 75776, "constants": [], "cta_group": 1, "threads": 192}'))
partial_ir = _decode_capture(_json_loads('{"__ir__": "flash_kmeans_assign_highd_paired_xreuse_dualtmem_producer_r47_v1", "arg_keys": ["x_tmap", "c_tmap", "c_sq", "partial_keys", "B", "N", "D", "K", "num_n_tiles", "K_tiles", "K_slices"], "cluster_dims": [1, 1, 1], "computed_smem_bytes": 75776, "constants": [], "cta_group": 1, "threads": 192}'))
reduce_ir = _decode_capture(_json_loads('{"__ir__": "flash_kmeans_assign_highd_paired_ownerreduce_r39_reduce1_unroll_v1", "arg_keys": ["partial_keys", "out", "B", "N", "K", "num_n_tiles", "K_slices"], "cluster_dims": [1, 1, 1], "computed_smem_bytes": 0, "constants": [], "cta_group": 1, "threads": 64}'))

def _verify_export_ir() -> Any:
    verify_kernel = os.environ.get('LOOM_FLASH_KMEANS_HIGHD_PAIRED_XREUSE_DUALTMEM_R47_VERIFY_KERNEL')
    if verify_kernel == 'd448_reduce':
        return reduce_ir
    if verify_kernel == 'd512_partial':
        return _r39._r2.partial_ir
    if verify_kernel == 'd512_reduce':
        return _r39._r2.reduce_ir
    return partial_ir
ir = _decode_capture(_json_loads('{"__ir__": "flash_kmeans_assign_highd_paired_xreuse_dualtmem_producer_r47_v1", "arg_keys": ["x_tmap", "c_tmap", "c_sq", "partial_keys", "B", "N", "D", "K", "num_n_tiles", "K_tiles", "K_slices"], "cluster_dims": [1, 1, 1], "computed_smem_bytes": 75776, "constants": [], "cta_group": 1, "threads": 192}'))

@lru_cache(maxsize=1)
def _loaded_partial_key_kernel() -> tuple[Any, int, int]:
    from .._dispatch_runtime import CUDAKernel
    cubin, kernel_name, smem_bytes, threads = _base._compile_ir(partial_ir)
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
        producer_grid, reducer_grid = _launch_d448(inputs, bsz=bsz, n_points=n_points, dim=dim, n_clusters=n_clusters, num_n_tiles=num_n_tiles, k_tiles=k_tiles, k_slices=k_slices, total_work=total_work)
        trace = _route_trace_d448(inputs, total_work=total_work, producer_grid=producer_grid, reducer_grid=reducer_grid, k_slices=k_slices)
        return {'cluster_ids': inputs['out'], 'selected_route': ROUTE_ID, 'route_trace': trace}
    outputs = _r39.launch_for_eval(inputs)
    trace = _wrap_r39_trace(inputs, child_outputs=outputs)
    return {'cluster_ids': outputs['cluster_ids'], 'selected_route': ROUTE_ID, 'route_trace': trace}

def _use_d448_custom(*, bsz: int, n_points: int, dim: int, n_clusters: int) -> bool:
    return bsz == 1 and n_points == 2048 and (n_clusters == 4096) and (dim == 448)

def _validate_shape(inputs: dict[str, Any], *, bsz: int, n_points: int, dim: int, n_clusters: int) -> None:
    dtype_name = _base._dtype_name(inputs)
    if dtype_name not in BF16_DTYPE_NAMES:
        raise ValueError(''.join([format(ROUTE_ID, ''), ' requires bfloat16 input, got ', format(dtype_name, '')]))
    if (bsz, n_points, n_clusters, dim) not in {(1, 2048, 4096, 448), (1, 2048, 4096, 512)}:
        raise ValueError(''.join([format(ROUTE_ID, ''), ' is exact-shape only for paired D448/D512 rows']))

def _launch_d448(inputs: dict[str, Any], *, bsz: int, n_points: int, dim: int, n_clusters: int, num_n_tiles: int, k_tiles: int, k_slices: int, total_work: int) -> tuple[tuple[int, int, int], tuple[int, int, int]]:
    import torch
    partial_keys = _r1._partial_key_buffer(inputs, total_work)
    tmap_x, tmap_c = _g2r4._make_tmaps(inputs)
    stream = torch.cuda.current_stream()
    partial_kernel, smem_bytes, threads = _loaded_partial_key_kernel()
    partial_args = pack_kernel_args(partial_ir, x_tmap=tmap_x, c_tmap=tmap_c, c_sq=inputs['c_sq'], partial_keys=partial_keys, B=bsz, N=n_points, D=dim, K=n_clusters, num_n_tiles=num_n_tiles, K_tiles=k_tiles, K_slices=k_slices)
    producer_grid = (min(total_work, PRODUCER_GRID_CAP_D448), 1, 1)
    partial_kernel.launch(grid=producer_grid, block=(threads, 1, 1), args=partial_args, shared_mem=smem_bytes, stream=stream)
    reduce_kernel, smem_bytes, threads = _r39._loaded_reduce1_kernel()
    reduce_args = [partial_keys, inputs['out'], bsz, n_points, n_clusters, num_n_tiles, k_slices]
    reducer_grid = (min(bsz * num_n_tiles, SPLITK_GRID_CAP), 1, 1)
    reduce_kernel.launch(grid=reducer_grid, block=(threads, 1, 1), args=reduce_args, shared_mem=smem_bytes, stream=stream)
    return (producer_grid, reducer_grid)

def _route_trace_d448(inputs: dict[str, Any], *, total_work: int, producer_grid: tuple[int, int, int], reducer_grid: tuple[int, int, int], k_slices: int) -> dict[str, Any]:
    return {'shape_key': _base._shape_key(inputs), 'selected_route': ROUTE_ID, 'selected_entrypoint': 'loom.examples.weave.flash_kmeans_assign_highd_paired_xreuse_dualtmem_r47_v1:launch_for_eval', 'selected_seed': SEED_ID, 'expected_seed': None, 'route_kind': 'shape-specific-seed', 'route_source': 'shape-specific-seed', 'guard_id': 'guard_highd_paired_xreuse_dualtmem_r47_v1', 'guard_condition': 'dtype == bfloat16 and B == 1 and N == 2048 and K == 4096 and D in [448,512]; D448 uses X-reuse dual-TMEM producer, D512 delegates to R39', 'classification': 'route-ok', 'dispatcher_kernel_ms': None, 'shape_specific_kernel_ms': None, 'relative_speedup_vs_baseline': None, 'total_tiles': total_work, 'launch_grid': producer_grid, 'reduce_grid': reducer_grid, 'paired_packedpartial_route': True, 'selected_reducer_variant': 'r39_reduce1_unroll8', 'tile_shape': {'BLOCK_N': BLOCK_N, 'BLOCK_K': SPLITK_TILE_K, 'mma_block_k': MMA_BLOCK_K, 'split_k_slices': k_slices, 'feature_tiles': D448_FEATURE_TILES, 'feature_loop_bound': 'compile_time_7', 'feature_loop_unroll': D448_FEATURE_TILES, 'grouped_mma_k_tiles': SPLITK_GROUP_K_TILES, 'x_tma_loads_per_work_item': D448_FEATURE_TILES, 'score_tmem_regions': 2, 'score_tmem_cols': [0, MMA_BLOCK_K], 'producer_work_items': total_work, 'producer_grid_cap': PRODUCER_GRID_CAP_D448, 'producer_grid_work_items_per_cta': total_work // min(total_work, PRODUCER_GRID_CAP_D448), 'producer_to_consumer': 'dual_tmem_xreuse_u64_partial_key_buffer_plus_reduce1_unroll8', 'partial_key': 'ordered_f32_score_high32_inverse_cluster_index_low32', 'reducer_lanes_per_row': 1, 'reducer_threads': _r39.REDUCE_THREADS_R1_D448, 'producer_to_reducer_sync': 'same_stream_ordering', 'reducer_scan_unroll': PAIRED_K_SLICES}, 'reason': 'D448 keeps the R39 packed-key producer/reducer ABI but loads each X feature tile once and accumulates the two paired local-K halves into separate TMEM regions before draining'}

def _wrap_r39_trace(inputs: dict[str, Any], *, child_outputs: dict[str, Any]) -> dict[str, Any]:
    child_trace = dict(child_outputs.get('route_trace', {}))
    trace = dict(child_trace)
    trace.update({'shape_key': _base._shape_key(inputs), 'selected_route': ROUTE_ID, 'selected_entrypoint': 'loom.examples.weave.flash_kmeans_assign_highd_paired_xreuse_dualtmem_r47_v1:launch_for_eval', 'selected_seed': SEED_ID, 'expected_seed': None, 'route_kind': 'shape-specific-seed-composite', 'route_source': 'shape-specific-seed', 'guard_id': 'guard_highd_paired_xreuse_dualtmem_r47_v1', 'guard_condition': 'dtype == bfloat16 and B == 1 and N == 2048 and K == 4096 and D in [448,512]; D448 uses X-reuse dual-TMEM producer, D512 delegates to R39', 'classification': 'route-ok', 'child_route': child_outputs.get('selected_route') or child_trace.get('selected_route'), 'child_entrypoint': child_trace.get('selected_entrypoint'), 'child_guard_id': child_trace.get('guard_id'), 'child_tile_shape': child_trace.get('tile_shape'), 'selected_reducer_variant': 'r39_d512_child', 'paired_packedpartial_route': True, 'reason': 'D512 delegates unchanged to the round-39 gridcap160 child route'})
    return trace

def compile_and_launch_flash_kmeans_assign_highd_paired_xreuse_dualtmem_r47(B: int=1, N: int=2048, K: int=4096, D: int=448, *, benchmark: bool=False) -> dict[str, Any]:
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
