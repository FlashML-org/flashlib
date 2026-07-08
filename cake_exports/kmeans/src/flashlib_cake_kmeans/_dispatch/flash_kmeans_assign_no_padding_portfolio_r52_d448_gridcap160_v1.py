"""Flash-KMeans no-padding high-D Split-K portfolio round-52 candidate.

Minimum architecture: sm_100a. This additive bucket-kernel variant keeps the
round-51 no-padding high-D denominator but changes the D448 paired route to
launch the round-47 dual-TMEM X-reuse producer with a 160-CTA grid cap before
the round-39 R1 packed-key reducer. D512 paired and non-D448 rows delegate to
the round-51 portfolio children. It is not intended for sm_120a/sm_121a where
ptxas rejects tcgen05 instructions.
"""
from __future__ import annotations
from json import loads as _json_loads
from .._dispatch_runtime import _capture_cuTensorMapEncodeTiled, _decode_capture, _import_dispatch_module, _ir_proxy
import os
from typing import Any
from . import flash_kmeans_assign_highd_paired_xreuse_dualtmem_r47_v1 as _r47
from . import flash_kmeans_assign_highd_splitk_blockn64_g2r4_b5a6_v1 as _g2r4
from . import flash_kmeans_assign_no_padding_portfolio_r51_v1 as _r51
from .._dispatch_runtime import pack_kernel_args
ROUTE_ID = 'flash_kmeans_assign_no_padding_portfolio_r52_d448_gridcap160_v1'
SEED_ID = 'no-padding-highd-portfolio-r52-d448-gridcap160-v1'
VERIFY_ENV = 'LOOM_FLASH_KMEANS_NO_PADDING_PORTFOLIO_R52_D448_GRIDCAP160_VERIFY_KERNEL'
PRODUCER_GRID_CAP_D448 = 160
NO_PADDING_HIGHD_SHAPES = _r51.NO_PADDING_HIGHD_SHAPES
NO_PADDING_HIGHD_EVAL_SHAPES = _r51.NO_PADDING_HIGHD_EVAL_SHAPES

def _verify_export_ir() -> Any:
    verify_kernel = os.environ.get(VERIFY_ENV)
    if verify_kernel == 'd448_reduce':
        return _r47.reduce_ir
    if verify_kernel == 'd512_partial':
        return _r47._r39._r2.partial_ir
    if verify_kernel == 'd512_reduce':
        return _r47._r39._r2.reduce_ir
    if verify_kernel == 'hugek_partial':
        return _r51._streamdep.partial_ir
    if verify_kernel == 'hugek_reduce':
        return _r51._streamdep.reduce_ir
    return _r47.partial_ir
ir = _decode_capture(_json_loads('{"__ir__": "flash_kmeans_assign_highd_paired_xreuse_dualtmem_producer_r47_v1", "arg_keys": ["x_tmap", "c_tmap", "c_sq", "partial_keys", "B", "N", "D", "K", "num_n_tiles", "K_tiles", "K_slices"], "cluster_dims": [1, 1, 1], "computed_smem_bytes": 75776, "constants": [], "cta_group": 1, "threads": 192}'))

def launch_for_eval(inputs: dict[str, Any]) -> dict[str, Any]:
    bsz = int(inputs['B'])
    n_points = int(inputs['N'])
    dim = int(inputs['D'])
    n_clusters = int(inputs['K'])
    _r51._validate_shape(inputs, bsz=bsz, n_points=n_points, dim=dim, n_clusters=n_clusters)
    if _r51._is_d448_paired(bsz=bsz, n_points=n_points, dim=dim, n_clusters=n_clusters):
        num_n_tiles = n_points // _r47.BLOCK_N
        k_tiles = n_clusters // _r47.MMA_BLOCK_K
        k_slices = k_tiles // _r47.SPLITK_GROUP_K_TILES
        if k_slices != _r47.PAIRED_K_SLICES:
            raise ValueError(''.join([format(ROUTE_ID, ''), ' requires K_slices=', format(_r47.PAIRED_K_SLICES, ''), ', got ', format(k_slices, '')]))
        total_work = bsz * num_n_tiles * k_slices
        producer_grid, reducer_grid = _launch_d448_gridcap160(inputs, bsz=bsz, n_points=n_points, dim=dim, n_clusters=n_clusters, num_n_tiles=num_n_tiles, k_tiles=k_tiles, k_slices=k_slices, total_work=total_work)
        trace = _route_trace_d448_gridcap160(inputs, total_work=total_work, producer_grid=producer_grid, reducer_grid=reducer_grid, k_slices=k_slices)
        return {'cluster_ids': inputs['out'], 'selected_route': ROUTE_ID, 'route_trace': trace}
    outputs = _r51.launch_for_eval(inputs)
    trace = _wrap_r51_child_trace(inputs, child_outputs=outputs, reason='non-D448 rows delegate to the round-51 no-padding portfolio')
    return {'cluster_ids': outputs['cluster_ids'], 'selected_route': ROUTE_ID, 'route_trace': trace}

def _launch_d448_gridcap160(inputs: dict[str, Any], *, bsz: int, n_points: int, dim: int, n_clusters: int, num_n_tiles: int, k_tiles: int, k_slices: int, total_work: int) -> tuple[tuple[int, int, int], tuple[int, int, int]]:
    import torch
    partial_keys = _r47._r1._partial_key_buffer(inputs, total_work)
    tmap_x, tmap_c = _g2r4._make_tmaps(inputs)
    stream = torch.cuda.current_stream()
    partial_kernel, smem_bytes, threads = _r47._loaded_partial_key_kernel()
    partial_args = pack_kernel_args(_r47.partial_ir, x_tmap=tmap_x, c_tmap=tmap_c, c_sq=inputs['c_sq'], partial_keys=partial_keys, B=bsz, N=n_points, D=dim, K=n_clusters, num_n_tiles=num_n_tiles, K_tiles=k_tiles, K_slices=k_slices)
    producer_grid = (min(total_work, PRODUCER_GRID_CAP_D448), 1, 1)
    partial_kernel.launch(grid=producer_grid, block=(threads, 1, 1), args=partial_args, shared_mem=smem_bytes, stream=stream)
    reduce_kernel, smem_bytes, threads = _r47._r39._loaded_reduce1_kernel()
    reduce_args = [partial_keys, inputs['out'], bsz, n_points, n_clusters, num_n_tiles, k_slices]
    reducer_grid = (min(bsz * num_n_tiles, _r47.SPLITK_GRID_CAP), 1, 1)
    reduce_kernel.launch(grid=reducer_grid, block=(threads, 1, 1), args=reduce_args, shared_mem=smem_bytes, stream=stream)
    return (producer_grid, reducer_grid)

def _route_trace_d448_gridcap160(inputs: dict[str, Any], *, total_work: int, producer_grid: tuple[int, int, int], reducer_grid: tuple[int, int, int], k_slices: int) -> dict[str, Any]:
    trace = _r47._route_trace_d448(inputs, total_work=total_work, producer_grid=producer_grid, reducer_grid=reducer_grid, k_slices=k_slices)
    trace.update({'selected_route': ROUTE_ID, 'selected_entrypoint': 'loom.examples.weave.flash_kmeans_assign_no_padding_portfolio_r52_d448_gridcap160_v1:launch_for_eval', 'selected_seed': SEED_ID, 'route_source': 'shape-specific-seed-portfolio', 'guard_id': 'guard_no_padding_highd_splitk_portfolio_r52_d448_gridcap160_v1', 'guard_condition': 'exact no-padding high-D bucket; D448 paired uses R47 producer gridcap160, other rows delegate to round-51', 'portfolio_selected_child': 'r47_dualtmem_xreuse_gridcap160_plus_r39_reduce1', 'reason': 'D448 paired row keeps the round-47 dual-TMEM X-reuse producer/reducer ABI but raises the producer launch cap from 128 to 160 CTAs to reduce persistent work underfeed'})
    tile_shape = dict(trace.get('tile_shape', {}))
    tile_shape.update({'producer_grid_cap': PRODUCER_GRID_CAP_D448, 'producer_grid_work_items_per_cta_floor': total_work // PRODUCER_GRID_CAP_D448, 'producer_grid_remainder_work_items': total_work % PRODUCER_GRID_CAP_D448})
    trace['tile_shape'] = tile_shape
    return trace

def _wrap_r51_child_trace(inputs: dict[str, Any], *, child_outputs: dict[str, Any], reason: str) -> dict[str, Any]:
    child_trace = dict(child_outputs.get('route_trace', {}))
    trace = dict(child_trace)
    trace.update({'shape_key': _r51._base._shape_key(inputs), 'selected_route': ROUTE_ID, 'selected_entrypoint': 'loom.examples.weave.flash_kmeans_assign_no_padding_portfolio_r52_d448_gridcap160_v1:launch_for_eval', 'selected_seed': SEED_ID, 'route_source': 'shape-specific-seed-portfolio', 'guard_id': 'guard_no_padding_highd_splitk_portfolio_r52_d448_gridcap160_v1', 'portfolio_selected_child': child_trace.get('portfolio_selected_child'), 'child_route': child_outputs.get('selected_route') or child_trace.get('selected_route'), 'child_entrypoint': child_trace.get('selected_entrypoint'), 'child_guard_id': child_trace.get('guard_id'), 'reason': reason})
    return trace

def compile_and_launch_flash_kmeans_assign_no_padding_portfolio_r52_d448_gridcap160(B: int=1, N: int=2048, K: int=4096, D: int=448, *, benchmark: bool=False) -> dict[str, Any]:
    import torch
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA GPU required')
    seed = {(1, 768, 8192, 384): 38404, (1, 512, 8192, 448): 44802, (1, 2048, 4096, 448): 44801, (1, 512, 8192, 512): 51204, (1, 2048, 4096, 512): 51202}[B, N, K, D]
    torch.manual_seed(seed)
    x = torch.randn((B, N, D), dtype=torch.bfloat16, device='cuda').contiguous()
    centroids = torch.randn((B, K, D), dtype=torch.bfloat16, device='cuda').contiguous()
    x_sq = (x.float() ** 2).sum(-1).contiguous()
    c_sq = (centroids.float() ** 2).sum(-1).contiguous()
    out = torch.empty((B, N), dtype=torch.int32, device='cuda')
    inputs = {'label': NO_PADDING_HIGHD_SHAPES[B, N, K, D], 'B': B, 'N': N, 'D': D, 'K': K, 'dtype': 'bfloat16', 'x': x, 'centroids': centroids, 'x_sq': x_sq, 'c_sq': c_sq, 'out': out}
    launch_for_eval(inputs)
    ref_dist = x_sq.unsqueeze(-1) + c_sq.unsqueeze(1) - 2.0 * torch.einsum('bnd,bkd->bnk', x.float(), centroids.float())
    ref = ref_dist.clamp_min(0.0).argmin(dim=-1).to(torch.int32)
    result: dict[str, Any] = {'passed': bool(torch.equal(out, ref))}
    if benchmark:
        from .._dispatch_runtime import evaluate
        result['contract_eval'] = evaluate(launch_for_eval, shapes=NO_PADDING_HIGHD_EVAL_SHAPES, correctness=True, benchmark=True, flashlib_baseline=True, benchmark_warmup_ms=100, benchmark_ms=1000, time_triton_baseline=False)
    return result
