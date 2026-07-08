"""Flash-KMeans no-padding high-D Split-K portfolio round-51 candidate.

Minimum architecture: sm_100a. This additive bucket-kernel composition keeps
Blackwell tcgen05/TMEM children on the contract-visible path for the no-padding
high-D Split-K rows from the round-50 handoff plus expanded no-padding heldout
rows. It routes D448 paired to the round-47 dual-TMEM X-reuse producer plus
round-39 R1 reducer, D512 paired to the recorded round-39 incumbent, and the
other high-D rows directly through the 4f2c streamdep child. It is not intended
for sm_120a/sm_121a where ptxas rejects tcgen05 instructions.
"""
from __future__ import annotations
from json import loads as _json_loads
from .._dispatch_runtime import _capture_cuTensorMapEncodeTiled, _decode_capture, _import_dispatch_module, _ir_proxy
import os
from typing import Any
from . import flash_kmeans_assign_highd_paired_ownerreduce_r39_v1 as _r39
from . import flash_kmeans_assign_highd_paired_xreuse_dualtmem_r47_v1 as _r47
from . import flash_kmeans_assign_highd_splitk_8de8_v1 as _base
from . import flash_kmeans_assign_highd_splitk_blockn64_g2r4_streamdep_4f2c_v1 as _streamdep
ROUTE_ID = 'flash_kmeans_assign_no_padding_portfolio_r51_v1'
SEED_ID = 'no-padding-highd-portfolio-r51-v1'
VERIFY_ENV = 'LOOM_FLASH_KMEANS_NO_PADDING_PORTFOLIO_R51_VERIFY_KERNEL'
NO_PADDING_HIGHD_SHAPES = {(1, 512, 8192, 320): 'd320_hugek_b1_n512_k8192_d320', (1, 1024, 8192, 320): 'd320_hugek_b1_n1024_k8192_d320', (1, 2048, 8192, 320): 'd320_hugek_b1_n2048_k8192_d320', (1, 512, 8192, 384): 'd384_hugek_b1_n512_k8192_d384', (1, 768, 8192, 384): 'd384_hugek_b1_n768_k8192_d384', (1, 1024, 8192, 384): 'd384_hugek_b1_n1024_k8192_d384', (1, 512, 4096, 448): 'd448_midk_b1_n512_k4096_d448', (1, 1024, 4096, 448): 'd448_midk_b1_n1024_k4096_d448', (1, 512, 8192, 448): 'd448_hugek_b1_n512_k8192_d448', (1, 1024, 8192, 448): 'd448_hugek_b1_n1024_k8192_d448', (1, 2048, 4096, 448): 'd448_paired_b1_n2048_k4096_d448', (1, 512, 4096, 512): 'd512_midk_b1_n512_k4096_d512', (1, 1024, 4096, 512): 'd512_midk_b1_n1024_k4096_d512', (1, 512, 8192, 512): 'd512_hugek_b1_n512_k8192_d512', (1, 1024, 8192, 512): 'd512_hugek_b1_n1024_k8192_d512', (1, 2048, 4096, 512): 'd512_paired_b1_n2048_k4096_d512'}
NO_PADDING_HIGHD_EVAL_SHAPES = [{'label': 'd320_hugek_b1_n512_k8192_d320', 'params': {'B': 1, 'N': 512, 'D': 320, 'K': 8192, 'dtype': 'bfloat16', 'seed': 32004}}, {'label': 'd320_hugek_b1_n1024_k8192_d320', 'params': {'B': 1, 'N': 1024, 'D': 320, 'K': 8192, 'dtype': 'bfloat16', 'seed': 32005}}, {'label': 'd320_hugek_b1_n2048_k8192_d320', 'params': {'B': 1, 'N': 2048, 'D': 320, 'K': 8192, 'dtype': 'bfloat16', 'seed': 32006}}, {'label': 'd384_hugek_b1_n512_k8192_d384', 'params': {'B': 1, 'N': 512, 'D': 384, 'K': 8192, 'dtype': 'bfloat16', 'seed': 38403}}, {'label': 'd384_hugek_b1_n768_k8192_d384', 'params': {'B': 1, 'N': 768, 'D': 384, 'K': 8192, 'dtype': 'bfloat16', 'seed': 38404}}, {'label': 'd384_hugek_b1_n1024_k8192_d384', 'params': {'B': 1, 'N': 1024, 'D': 384, 'K': 8192, 'dtype': 'bfloat16', 'seed': 38405}}, {'label': 'd448_midk_b1_n512_k4096_d448', 'params': {'B': 1, 'N': 512, 'D': 448, 'K': 4096, 'dtype': 'bfloat16', 'seed': 44803}}, {'label': 'd448_midk_b1_n1024_k4096_d448', 'params': {'B': 1, 'N': 1024, 'D': 448, 'K': 4096, 'dtype': 'bfloat16', 'seed': 44805}}, {'label': 'd448_hugek_b1_n512_k8192_d448', 'params': {'B': 1, 'N': 512, 'D': 448, 'K': 8192, 'dtype': 'bfloat16', 'seed': 44802}}, {'label': 'd448_hugek_b1_n1024_k8192_d448', 'params': {'B': 1, 'N': 1024, 'D': 448, 'K': 8192, 'dtype': 'bfloat16', 'seed': 44806}}, {'label': 'd448_paired_b1_n2048_k4096_d448', 'params': {'B': 1, 'N': 2048, 'D': 448, 'K': 4096, 'dtype': 'bfloat16', 'seed': 44801}}, {'label': 'd512_midk_b1_n512_k4096_d512', 'params': {'B': 1, 'N': 512, 'D': 512, 'K': 4096, 'dtype': 'bfloat16', 'seed': 51203}}, {'label': 'd512_midk_b1_n1024_k4096_d512', 'params': {'B': 1, 'N': 1024, 'D': 512, 'K': 4096, 'dtype': 'bfloat16', 'seed': 51205}}, {'label': 'd512_hugek_b1_n512_k8192_d512', 'params': {'B': 1, 'N': 512, 'D': 512, 'K': 8192, 'dtype': 'bfloat16', 'seed': 51204}}, {'label': 'd512_hugek_b1_n1024_k8192_d512', 'params': {'B': 1, 'N': 1024, 'D': 512, 'K': 8192, 'dtype': 'bfloat16', 'seed': 51206}}, {'label': 'd512_paired_b1_n2048_k4096_d512', 'params': {'B': 1, 'N': 2048, 'D': 512, 'K': 4096, 'dtype': 'bfloat16', 'seed': 51202}}]

def _verify_export_ir() -> Any:
    verify_kernel = os.environ.get(VERIFY_ENV)
    if verify_kernel == 'd448_reduce':
        return _r47.reduce_ir
    if verify_kernel == 'd512_partial':
        return _r39._r2.partial_ir
    if verify_kernel == 'd512_reduce':
        return _r39._r2.reduce_ir
    if verify_kernel == 'hugek_partial':
        return _streamdep.partial_ir
    if verify_kernel == 'hugek_reduce':
        return _streamdep.reduce_ir
    return _r47.partial_ir
ir = _decode_capture(_json_loads('{"__ir__": "flash_kmeans_assign_highd_paired_xreuse_dualtmem_producer_r47_v1", "arg_keys": ["x_tmap", "c_tmap", "c_sq", "partial_keys", "B", "N", "D", "K", "num_n_tiles", "K_tiles", "K_slices"], "cluster_dims": [1, 1, 1], "computed_smem_bytes": 75776, "constants": [], "cta_group": 1, "threads": 192}'))

def launch_for_eval(inputs: dict[str, Any]) -> dict[str, Any]:
    bsz = int(inputs['B'])
    n_points = int(inputs['N'])
    dim = int(inputs['D'])
    n_clusters = int(inputs['K'])
    _validate_shape(inputs, bsz=bsz, n_points=n_points, dim=dim, n_clusters=n_clusters)
    if _is_d448_paired(bsz=bsz, n_points=n_points, dim=dim, n_clusters=n_clusters):
        outputs = _r47.launch_for_eval(inputs)
        trace = _wrap_child_trace(inputs, child_outputs=outputs, route_kind='shape-specific-seed-composition', selected_child='r47_dualtmem_xreuse_plus_r39_reduce1', reason='D448 paired row uses the round-47 dual-TMEM X-reuse producer and the round-39 unrolled R1 reducer')
        return {'cluster_ids': outputs['cluster_ids'], 'selected_route': ROUTE_ID, 'route_trace': trace}
    if _is_d512_paired(bsz=bsz, n_points=n_points, dim=dim, n_clusters=n_clusters):
        outputs = _r39.launch_for_eval(inputs)
        trace = _wrap_child_trace(inputs, child_outputs=outputs, route_kind='shape-specific-seed-composition', selected_child='r39_ownerreduce_incumbent', reason='D512 paired row keeps the recorded round-39 incumbent route')
        return {'cluster_ids': outputs['cluster_ids'], 'selected_route': ROUTE_ID, 'route_trace': trace}
    outputs = _streamdep.launch_for_eval(inputs)
    trace = _wrap_child_trace(inputs, child_outputs=outputs, route_kind='shape-specific-seed-composition', selected_child='4f2c_streamdep_child', reason='huge-K rows use the same-process fastest direct stream-dependency high-D Split-K child')
    return {'cluster_ids': outputs['cluster_ids'], 'selected_route': ROUTE_ID, 'route_trace': trace}

def _validate_shape(inputs: dict[str, Any], *, bsz: int, n_points: int, dim: int, n_clusters: int) -> None:
    dtype_name = _base._dtype_name(inputs)
    if dtype_name not in _base.BF16_DTYPE_NAMES:
        raise ValueError(''.join([format(ROUTE_ID, ''), ' requires bfloat16 input, got ', format(dtype_name, '')]))
    if (bsz, n_points, n_clusters, dim) not in NO_PADDING_HIGHD_SHAPES:
        raise ValueError(''.join([format(ROUTE_ID, ''), ' is exact-shape only for the no-padding high-D bucket, got B=', format(bsz, ''), ', N=', format(n_points, ''), ', K=', format(n_clusters, ''), ', D=', format(dim, '')]))

def _is_d448_paired(*, bsz: int, n_points: int, dim: int, n_clusters: int) -> bool:
    return bsz == 1 and n_points == 2048 and (n_clusters == 4096) and (dim == 448)

def _is_d512_paired(*, bsz: int, n_points: int, dim: int, n_clusters: int) -> bool:
    return bsz == 1 and n_points == 2048 and (n_clusters == 4096) and (dim == 512)

def _wrap_child_trace(inputs: dict[str, Any], *, child_outputs: dict[str, Any], route_kind: str, selected_child: str, reason: str) -> dict[str, Any]:
    child_trace = dict(child_outputs.get('route_trace', {}))
    child_route = child_outputs.get('selected_route') or child_trace.get('selected_route')
    trace = dict(child_trace)
    trace.update({'shape_key': _base._shape_key(inputs), 'selected_route': ROUTE_ID, 'selected_entrypoint': 'loom.examples.weave.flash_kmeans_assign_no_padding_portfolio_r51_v1:launch_for_eval', 'selected_seed': SEED_ID, 'route_kind': route_kind, 'route_source': 'shape-specific-seed-portfolio', 'guard_id': 'guard_no_padding_highd_splitk_portfolio_r51_v1', 'guard_condition': 'exact no-padding high-D bucket: D320/D384 hugeK, D448/D512 midK/hugeK/paired, all N/K aligned without padding', 'classification': 'route-ok', 'child_route': child_route, 'child_entrypoint': child_trace.get('selected_entrypoint'), 'child_guard_id': child_trace.get('guard_id'), 'child_tile_shape': child_trace.get('tile_shape'), 'portfolio_selected_child': selected_child, 'dispatcher_kernel_ms': None, 'shape_specific_kernel_ms': None, 'relative_speedup_vs_baseline': None, 'reason': reason})
    return trace

def compile_and_launch_flash_kmeans_assign_no_padding_portfolio_r51(B: int=1, N: int=2048, K: int=4096, D: int=448, *, benchmark: bool=False) -> dict[str, Any]:
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
