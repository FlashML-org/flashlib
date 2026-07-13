"""Flash-KMeans Euclidean assignment D192/D256 repeated-MMA bucket seed.

Minimum architecture: sm_100a. This wrapper routes contract evaluation to
D-specific Weave tcgen05/TMEM seeds only; it does not call external runtime
fallbacks. The seed modules are not intended for sm_120a/sm_121a where ptxas
rejects tcgen05 instructions. D160 is listed as a blocked sibling because its
32-wide BF16 tail currently hangs when lowered as K=32 tcgen05 MMA.
"""
from __future__ import annotations
from json import loads as _json_loads
from .._dispatch_runtime import _capture_cuTensorMapEncodeTiled, _decode_capture, _import_dispatch_module, _ir_proxy
from typing import Any
from . import flash_kmeans_assign_cleanroom_tcgen05_d192_single_v1 as _d192_single
from . import flash_kmeans_assign_cleanroom_tcgen05_d192_splitd_v1 as _d192
from . import flash_kmeans_assign_cleanroom_tcgen05_d256_single_v1 as _d256
BLOCK_N = _d192.BLOCK_N
BLOCK_K = _d192.BLOCK_K
SUPPORTED_DIMS = (192, 256)
BUCKET_SHAPES: list[dict[str, Any]] = [{'label': 'd192_small_b2_n1024_k512_d192', 'params': {'B': 2, 'N': 1024, 'D': 192, 'K': 512, 'dtype': 'bfloat16', 'seed': 19201}}, {'label': 'd192_paired_b2_n2048_k2048_d192', 'params': {'B': 2, 'N': 2048, 'D': 192, 'K': 2048, 'dtype': 'bfloat16', 'seed': 19202}}, {'label': 'd192_fallback_b1_n2176_k1024_d192', 'params': {'B': 1, 'N': 2176, 'D': 192, 'K': 1024, 'dtype': 'bfloat16', 'seed': 19203}}, {'label': 'd192_large_b4_n32768_k1024_d192', 'params': {'B': 4, 'N': 32768, 'D': 192, 'K': 1024, 'dtype': 'bfloat16', 'seed': 19204}}, {'label': 'd256_small_b1_n1024_k512_d256', 'params': {'B': 1, 'N': 1024, 'D': 256, 'K': 512, 'dtype': 'bfloat16', 'seed': 25601}}, {'label': 'd256_paired_b1_n4096_k4096_d256', 'params': {'B': 1, 'N': 4096, 'D': 256, 'K': 4096, 'dtype': 'bfloat16', 'seed': 25602}}, {'label': 'd256_fallback_b2_n2432_k2048_d256', 'params': {'B': 2, 'N': 2432, 'D': 256, 'K': 2048, 'dtype': 'bfloat16', 'seed': 25603}}, {'label': 'd256_hugek_b1_n512_k8192_d256', 'params': {'B': 1, 'N': 512, 'D': 256, 'K': 8192, 'dtype': 'bfloat16', 'seed': 25604}}]
D160_BLOCKED_SHAPES: list[dict[str, Any]] = [{'label': 'd160_fallback_b2_n2432_k1024_d160', 'params': {'B': 2, 'N': 2432, 'D': 160, 'K': 1024, 'dtype': 'bfloat16', 'seed': 16002}, 'blocker': 'D160 requires a BF16 32-wide feature tail; the current K=32 tcgen05 path hangs at runtime.'}, {'label': 'd160_paired_b1_n2048_k2048_d160', 'params': {'B': 1, 'N': 2048, 'D': 160, 'K': 2048, 'dtype': 'bfloat16', 'seed': 16001}, 'blocker': 'D160 requires a BF16 32-wide feature tail; the current K=32 tcgen05 path hangs at runtime.'}]

def launch_for_eval(inputs: dict[str, Any]) -> dict[str, Any]:
    dim = int(inputs['D'])
    module = _module_for_dim(dim)
    route_kind = _route_kind(inputs)
    outputs = module.launch_for_eval(inputs)
    normalized = _normalize_outputs(outputs, inputs)
    selected_route = ''.join(['d', format(dim, ''), '_', format(route_kind, ''), '_repeated_mma_v1'])
    selected_entrypoint = _selected_entrypoint(dim, route_kind)
    normalized['selected_route'] = selected_route
    normalized['route_trace'] = {'shape_key': _shape_key(inputs), 'selected_route': selected_route, 'selected_entrypoint': selected_entrypoint, 'selected_seed': ''.join(['d', format(dim, ''), '-', format(route_kind, ''), '-repeated-mma-v1']), 'route_kind': 'specialized', 'route_source': 'shape-specific-seed', 'guard_id': ''.join(['guard_d', format(dim, ''), '_', format(route_kind, ''), '_repeated_mma_v1']), 'guard_condition': ''.join(['dtype == bfloat16 and D == ', format(dim, ''), ' and N % ', format(BLOCK_N, ''), ' == 0 and K % ', format(BLOCK_K, ''), ' == 0']), 'classification': 'seed-probe', 'dispatcher_kernel_ms': None, 'reason': 'non-128D bucket seed wrapper routes to D-specific Weave tcgen05 candidate'}
    return normalized

def _module_for_dim(dim: int):
    if dim == 192:
        return _d192
    if dim == 256:
        return _d256
    raise ValueError(''.join(['non128d split-D seed supports D in ', format(SUPPORTED_DIMS, ''), ', got ', format(dim, '')]))

def _selected_entrypoint(dim: int, route_kind: str) -> str:
    if dim == 192 and route_kind == 'single':
        return ''.join([format(_d192_single.__name__, ''), ':launch_for_eval'])
    if dim == 192:
        return ''.join([format(_d192.__name__, ''), ':launch_for_eval'])
    if dim == 256:
        return ''.join([format(_d256.__name__, ''), ':launch_for_eval'])
    raise ValueError(''.join(['non128d split-D seed supports D in ', format(SUPPORTED_DIMS, ''), ', got ', format(dim, '')]))

def _route_kind(inputs: dict[str, Any]) -> str:
    if int(inputs['D']) == 256:
        return 'single'
    n_tiles = int(inputs['N']) // BLOCK_N
    k_tiles = int(inputs['K']) // BLOCK_K
    if n_tiles % 2 != 0 or (n_tiles <= 8 and k_tiles <= 2):
        return 'single'
    return 'paired'

def _normalize_outputs(outputs: Any, inputs: dict[str, Any]) -> dict[str, Any]:
    if outputs is None:
        return {'cluster_ids': inputs['out']}
    if hasattr(outputs, 'shape'):
        return {'cluster_ids': outputs}
    if isinstance(outputs, dict):
        normalized = dict(outputs)
        if 'cluster_ids' not in normalized and 'out' in normalized:
            normalized['cluster_ids'] = normalized['out']
        if 'cluster_ids' in normalized:
            return normalized
    raise TypeError("flash_kmeans_assign non128d split-D route must return cluster_ids or write inputs['out']")

def _shape_key(inputs: dict[str, Any]) -> str:
    label = inputs.get('label')
    if label:
        return str(label)
    return ''.join(['b', format(int(inputs['B']), ''), '_n', format(int(inputs['N']), ''), '_k', format(int(inputs['K']), ''), '_d', format(int(inputs['D']), '')])

def compile_and_launch_flash_kmeans_assign_cleanroom(B: int=1, N: int=1024, K: int=512, D: int=192, *, benchmark: bool=False) -> dict[str, Any]:
    import torch
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA GPU required')
    torch.manual_seed(5101)
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
        result['contract_eval'] = evaluate(launch_for_eval, shapes=[{'label': ''.join(['manual_b', format(B, ''), '_n', format(N, ''), '_k', format(K, ''), '_d', format(D, '')]), 'params': {'B': B, 'N': N, 'D': D, 'K': K, 'dtype': 'bfloat16', 'seed': 5101}}])
    return result
