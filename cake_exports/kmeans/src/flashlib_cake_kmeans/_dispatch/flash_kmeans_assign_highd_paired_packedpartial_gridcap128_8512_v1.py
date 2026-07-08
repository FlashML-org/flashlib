"""Flash-KMeans high-D paired packed-partial grid-cap candidate.

Minimum architecture: sm_100a. This additive bucket-kernel candidate keeps the
Blackwell tcgen05/TMEM G2 packed-key distance producer on the contract-visible
path for paired B=1,N=2048,K=4096 high-D rows. It preserves the validated R1
reducer for D=448 and R2 reducer for D=512, but caps producer launch work at
128 CTAs so each producer CTA owns two fixed K-slice work items. Huge-K focus
rows delegate to the validated 4f2c stream-ordered child. It is not intended
for sm_120a/sm_121a where ptxas rejects tcgen05 instructions.
"""
from __future__ import annotations
from json import loads as _json_loads
from .._dispatch_runtime import _capture_cuTensorMapEncodeTiled, _decode_capture, _import_dispatch_module, _ir_proxy
import os
from typing import Any
from . import flash_kmeans_assign_highd_paired_packedpartial_r1_r1c1_v1 as _r1
from . import flash_kmeans_assign_highd_paired_packedpartial_r2_7b3c_v1 as _r2
from . import flash_kmeans_assign_highd_splitk_8de8_v1 as _base
from . import flash_kmeans_assign_highd_splitk_blockn64_g2r4_b5a6_v1 as _g2r4
from . import flash_kmeans_assign_highd_splitk_blockn64_g2r4_streamdep_4f2c_v1 as _streamdep
from .._dispatch_runtime import pack_kernel_args
BLOCK_N = _g2r4.BLOCK_N
MMA_BLOCK_K = _g2r4.MMA_BLOCK_K
SPLITK_GROUP_K_TILES = _g2r4.SPLITK_GROUP_K_TILES
SPLITK_TILE_K = _g2r4.SPLITK_TILE_K
SPLITK_GRID_CAP = _g2r4.SPLITK_GRID_CAP
PRODUCER_GRID_CAP = 128
SUPPORTED_DIMS = set(_base.SUPPORTED_DIMS)
BF16_DTYPE_NAMES = set(_base.BF16_DTYPE_NAMES)
PAIRED_PACKEDPARTIAL_DIMS = {448, 512}
ROUTE_ID = 'highd_paired_packedpartial_gridcap128_8512_v1'
SEED_ID = 'highd-paired-packedpartial-gridcap128-8512-v1'

def _verify_export_ir() -> Any:
    verify_kernel = os.environ.get('LOOM_FLASH_KMEANS_HIGHD_PAIRED_PACKEDPARTIAL_GRIDCAP128_8512_VERIFY_KERNEL')
    if verify_kernel == 'r1_reduce':
        return _r1.reduce_ir
    if verify_kernel == 'r2_partial':
        return _r2.partial_ir
    if verify_kernel == 'r2_reduce':
        return _r2.reduce_ir
    if verify_kernel == 'streamdep_partial':
        return _streamdep.partial_ir
    if verify_kernel == 'streamdep_reduce':
        return _streamdep.reduce_ir
    return _r1.partial_ir
ir = _decode_capture(_json_loads('{"__ir__": "flash_kmeans_assign_highd_paired_packedpartial_producer_r1c1_v1", "arg_keys": ["x_tmap", "c_tmap", "c_sq", "partial_keys", "B", "N", "D", "K", "num_n_tiles", "K_tiles", "K_slices"], "cluster_dims": [1, 1, 1], "computed_smem_bytes": 43008, "constants": [], "cta_group": 1, "threads": 192}'))

def launch_for_eval(inputs: dict[str, Any]) -> dict[str, Any]:
    bsz = int(inputs['B'])
    n_points = int(inputs['N'])
    dim = int(inputs['D'])
    n_clusters = int(inputs['K'])
    _validate_shape(inputs, bsz=bsz, n_points=n_points, dim=dim, n_clusters=n_clusters)
    if _use_paired_gridcap(bsz=bsz, n_points=n_points, dim=dim, n_clusters=n_clusters):
        child = _r1 if dim == 448 else _r2
        reducer = 'r1_single_lane' if dim == 448 else 'r2_two_lane'
        _launch_paired_gridcap(inputs, child=child, bsz=bsz, n_points=n_points, dim=dim, n_clusters=n_clusters)
        trace = _route_trace_paired(inputs, child=child, reducer=reducer)
        return {'cluster_ids': inputs['out'], 'selected_route': ROUTE_ID, 'route_trace': trace}
    outputs = _streamdep.launch_for_eval(inputs)
    trace = _wrap_child_trace(inputs, child_outputs=outputs)
    return {'cluster_ids': outputs['cluster_ids'], 'selected_route': ROUTE_ID, 'route_trace': trace}

def _use_paired_gridcap(*, bsz: int, n_points: int, dim: int, n_clusters: int) -> bool:
    return bsz == 1 and n_points == 2048 and (n_clusters == 4096) and (dim in PAIRED_PACKEDPARTIAL_DIMS)

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

def _launch_paired_gridcap(inputs: dict[str, Any], *, child: Any, bsz: int, n_points: int, dim: int, n_clusters: int) -> None:
    import torch
    num_n_tiles = n_points // BLOCK_N
    k_tiles = n_clusters // MMA_BLOCK_K
    k_slices = k_tiles // SPLITK_GROUP_K_TILES
    total_work = bsz * num_n_tiles * k_slices
    partial_keys = child._partial_key_buffer(inputs, total_work)
    tmap_x, tmap_c = _g2r4._make_tmaps(inputs)
    stream = torch.cuda.current_stream()
    partial_kernel, smem_bytes, threads = child._loaded_partial_key_kernel()
    partial_args = pack_kernel_args(child.partial_ir, x_tmap=tmap_x, c_tmap=tmap_c, c_sq=inputs['c_sq'], partial_keys=partial_keys, B=bsz, N=n_points, D=dim, K=n_clusters, num_n_tiles=num_n_tiles, K_tiles=k_tiles, K_slices=k_slices)
    partial_kernel.launch(grid=(min(total_work, PRODUCER_GRID_CAP), 1, 1), block=(threads, 1, 1), args=partial_args, shared_mem=smem_bytes, stream=stream)
    reduce_kernel, smem_bytes, threads = child._loaded_reduce_key_kernel()
    reduce_args = [partial_keys, inputs['out'], bsz, n_points, n_clusters, num_n_tiles, k_slices]
    reduce_kernel.launch(grid=(min(bsz * num_n_tiles, SPLITK_GRID_CAP), 1, 1), block=(threads, 1, 1), args=reduce_args, shared_mem=smem_bytes, stream=stream)

def _route_trace_paired(inputs: dict[str, Any], *, child: Any, reducer: str) -> dict[str, Any]:
    n_points = int(inputs['N'])
    n_clusters = int(inputs['K'])
    num_n_tiles = n_points // BLOCK_N
    k_tiles = n_clusters // MMA_BLOCK_K
    k_slices = k_tiles // SPLITK_GROUP_K_TILES
    total_work = int(inputs['B']) * num_n_tiles * k_slices
    child_entrypoint = 'loom.examples.weave.flash_kmeans_assign_highd_paired_packedpartial_r1_r1c1_v1:launch_for_eval' if child is _r1 else 'loom.examples.weave.flash_kmeans_assign_highd_paired_packedpartial_r2_7b3c_v1:launch_for_eval'
    child_route = _r1.ROUTE_ID if child is _r1 else _r2.ROUTE_ID
    child_guard = 'guard_highd_paired_packedpartial_r1_r1c1_v1' if child is _r1 else 'guard_highd_paired_packedpartial_r2_7b3c_v1'
    return {'shape_key': _base._shape_key(inputs), 'selected_route': ROUTE_ID, 'selected_entrypoint': 'loom.examples.weave.flash_kmeans_assign_highd_paired_packedpartial_gridcap128_8512_v1:launch_for_eval', 'selected_seed': SEED_ID, 'expected_seed': None, 'route_kind': 'shape-specific-seed-composite', 'route_source': 'shape-specific-seed', 'guard_id': 'guard_highd_paired_packedpartial_gridcap128_8512_v1', 'guard_condition': 'dtype == bfloat16 and B == 1 and N == 2048 and K == 4096 and D in [448,512]; D=448 uses packed partial R1, D=512 uses packed partial R2, producer grid capped at 128 CTAs', 'classification': 'route-ok', 'child_route': child_route, 'child_entrypoint': child_entrypoint, 'child_guard_id': child_guard, 'dispatcher_kernel_ms': None, 'shape_specific_kernel_ms': None, 'relative_speedup_vs_baseline': None, 'total_tiles': total_work, 'launch_grid': (min(total_work, PRODUCER_GRID_CAP), 1, 1), 'paired_packedpartial_route': True, 'selected_reducer_variant': reducer, 'tile_shape': {'BLOCK_N': BLOCK_N, 'BLOCK_K': SPLITK_TILE_K, 'mma_block_k': MMA_BLOCK_K, 'split_k_slices': k_slices, 'grouped_mma_k_tiles': SPLITK_GROUP_K_TILES, 'producer_work_items': total_work, 'producer_grid_cap': PRODUCER_GRID_CAP, 'producer_grid_work_items_per_cta': total_work // min(total_work, PRODUCER_GRID_CAP), 'producer_to_consumer': 'u64_partial_key_buffer_plus_shape_guarded_r1_or_r2_reduce', 'partial_key': 'ordered_f32_score_high32_inverse_cluster_index_low32', 'reducer_lanes_per_row': _r1.REDUCE_LANES_PER_ROW if child is _r1 else _r2.REDUCE_LANES_PER_ROW, 'producer_to_reducer_sync': 'same_stream_ordering'}, 'reason': 'paired high-D row keeps the G2 tcgen05 packed-key producer but caps producer CTAs at 128 to reduce launch wave scheduling overhead while preserving the same reducer ABI'}

def _wrap_child_trace(inputs: dict[str, Any], *, child_outputs: dict[str, Any]) -> dict[str, Any]:
    child_trace = dict(child_outputs.get('route_trace', {}))
    child_route = child_outputs.get('selected_route') or child_trace.get('selected_route')
    trace = dict(child_trace)
    trace.update({'shape_key': _base._shape_key(inputs), 'selected_route': ROUTE_ID, 'selected_entrypoint': 'loom.examples.weave.flash_kmeans_assign_highd_paired_packedpartial_gridcap128_8512_v1:launch_for_eval', 'selected_seed': SEED_ID, 'expected_seed': None, 'route_kind': 'shape-specific-seed-composite', 'route_source': 'shape-specific-seed', 'guard_id': 'guard_highd_paired_packedpartial_gridcap128_8512_v1', 'guard_condition': 'dtype == bfloat16 and D in [384,448,512] and N % 64 == 0 and K % 256 == 0; paired B=1,N=2048,K=4096,D in [448,512] uses 128-grid-cap packed partial keys, other high-D rows use highd_splitk_blockn64_g2r4_streamdep_4f2c_v1', 'classification': 'route-ok', 'child_route': child_route, 'child_entrypoint': child_trace.get('selected_entrypoint'), 'child_guard_id': child_trace.get('guard_id'), 'child_tile_shape': child_trace.get('tile_shape'), 'dispatcher_kernel_ms': None, 'shape_specific_kernel_ms': None, 'relative_speedup_vs_baseline': None, 'paired_packedpartial_route': False, 'reason': 'non-paired high-D focus row keeps the stream-ordered G2/R4 Split-K child'})
    return trace

def compile_and_launch_flash_kmeans_assign_highd_paired_packedpartial(B: int=1, N: int=2048, K: int=4096, D: int=512, *, benchmark: bool=False) -> dict[str, Any]:
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
