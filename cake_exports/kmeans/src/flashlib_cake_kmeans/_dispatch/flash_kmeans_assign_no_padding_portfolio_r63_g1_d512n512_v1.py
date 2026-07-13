"""Flash-KMeans no-padding high-D round-63 G1/R4 D512 N512 portfolio.

Minimum architecture: sm_100a. This additive bucket-kernel variant keeps the
round-52 no-padding portfolio for established rows, but routes only the
D512 K=4096 N512 mid-K row through the round-63 G1/R4 streamdep child to
increase producer work feed. It is not intended for sm_120a/sm_121a where
ptxas rejects tcgen05 instructions.
"""
from __future__ import annotations
from json import loads as _json_loads
from .._dispatch_runtime import _capture_cuTensorMapEncodeTiled, _decode_capture, _import_dispatch_module, _ir_proxy
import os
from typing import Any
from . import flash_kmeans_assign_highd_splitk_blockn64_g1r4_streamdep_r63_v1 as _g1
from . import flash_kmeans_assign_no_padding_portfolio_r52_d448_gridcap160_v1 as _r52
ROUTE_ID = 'flash_kmeans_assign_no_padding_portfolio_r63_g1_d512n512_v1'
SEED_ID = 'no-padding-highd-portfolio-r63-g1-d512n512-v1'
VERIFY_ENV = 'LOOM_FLASH_KMEANS_NO_PADDING_PORTFOLIO_R63_G1_D512N512_VERIFY_KERNEL'
NO_PADDING_HIGHD_SHAPES = _r52.NO_PADDING_HIGHD_SHAPES
NO_PADDING_HIGHD_EVAL_SHAPES = _r52.NO_PADDING_HIGHD_EVAL_SHAPES

def _verify_export_ir() -> Any:
    verify_kernel = os.environ.get(VERIFY_ENV)
    if verify_kernel == 'g1_reduce':
        return _g1.reduce_ir
    if verify_kernel == 'r52':
        return _r52.ir
    return _g1.partial_ir
ir = _decode_capture(_json_loads('{"__ir__": "flash_kmeans_assign_highd_splitk_partial_blockn64_g1r4_streamdep_r63_v1", "arg_keys": ["x_tmap", "c_tmap", "c_sq", "partial_scores", "partial_indices", "B", "N", "D", "K", "num_n_tiles", "K_tiles", "K_slices"], "cluster_dims": [1, 1, 1], "computed_smem_bytes": 43008, "constants": [], "cta_group": 1, "threads": 192}'))

def launch_for_eval(inputs: dict[str, Any]) -> dict[str, Any]:
    bsz = int(inputs['B'])
    n_points = int(inputs['N'])
    dim = int(inputs['D'])
    n_clusters = int(inputs['K'])
    _r52._r51._validate_shape(inputs, bsz=bsz, n_points=n_points, dim=dim, n_clusters=n_clusters)
    if _is_g1_d512n512_target(bsz=bsz, n_points=n_points, dim=dim, n_clusters=n_clusters):
        outputs = _g1.launch_for_eval(inputs)
        trace = _wrap_child_trace(inputs, child_outputs=outputs, selected_child='r63_blockn64_g1r4_streamdep_child', reason='D512 N512 mid-K no-padding row uses the round-63 G1/R4 streamdep child to double producer work feed versus G2/R4')
        return {'cluster_ids': outputs['cluster_ids'], 'selected_route': ROUTE_ID, 'route_trace': trace}
    outputs = _r52.launch_for_eval(inputs)
    trace = _wrap_child_trace(inputs, child_outputs=outputs, selected_child=outputs.get('route_trace', {}).get('portfolio_selected_child', 'r52_portfolio_child'), reason='non-target rows delegate unchanged to the round-52 no-padding portfolio')
    return {'cluster_ids': outputs['cluster_ids'], 'selected_route': ROUTE_ID, 'route_trace': trace}

def _is_g1_d512n512_target(*, bsz: int, n_points: int, dim: int, n_clusters: int) -> bool:
    return bsz == 1 and n_points == 512 and (dim == 512) and (n_clusters == 4096)

def _wrap_child_trace(inputs: dict[str, Any], *, child_outputs: dict[str, Any], selected_child: str, reason: str) -> dict[str, Any]:
    child_trace = dict(child_outputs.get('route_trace', {}))
    trace = dict(child_trace)
    trace.update({'shape_key': _r52._r51._base._shape_key(inputs), 'selected_route': ROUTE_ID, 'selected_entrypoint': 'loom.examples.weave.flash_kmeans_assign_no_padding_portfolio_r63_g1_d512n512_v1:launch_for_eval', 'selected_seed': SEED_ID, 'route_source': 'shape-specific-seed-portfolio', 'guard_id': 'guard_no_padding_highd_splitk_portfolio_r63_g1_d512n512_v1', 'guard_condition': 'exact no-padding high-D bucket; only D512 K4096 N512 uses G1/R4 streamdep child, other rows delegate to R52', 'portfolio_selected_child': selected_child, 'child_route': child_outputs.get('selected_route') or child_trace.get('selected_route'), 'child_entrypoint': child_trace.get('selected_entrypoint'), 'child_guard_id': child_trace.get('guard_id'), 'child_tile_shape': child_trace.get('tile_shape'), 'reason': reason})
    return trace

def compile_and_launch_flash_kmeans_assign_no_padding_portfolio_r63_g1_d512n512(B: int=1, N: int=512, K: int=4096, D: int=512, *, benchmark: bool=False) -> dict[str, Any]:
    import torch
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA GPU required')
    seed = {(1, 512, 8192, 320): 32004, (1, 1024, 8192, 320): 32005, (1, 2048, 8192, 320): 32006, (1, 512, 8192, 384): 38403, (1, 768, 8192, 384): 38404, (1, 1024, 8192, 384): 38405, (1, 512, 4096, 448): 44803, (1, 1024, 4096, 448): 44805, (1, 512, 8192, 448): 44802, (1, 1024, 8192, 448): 44806, (1, 2048, 4096, 448): 44801, (1, 512, 4096, 512): 51203, (1, 1024, 4096, 512): 51205, (1, 512, 8192, 512): 51204, (1, 1024, 8192, 512): 51206, (1, 2048, 4096, 512): 51202}[B, N, K, D]
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
