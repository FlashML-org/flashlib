"""Exact D1024/Q8/M65536/K10 tcgen05 kNN with idle-warps removed.

Minimum target architecture: sm_100a.  This exact-shape B300 candidate keeps
the incumbent's direct-stride tcgen05 producer and constant-148 merge ABI, but
uses the 16 warps that own the four 32-column TMEM cohorts.  The incumbent's
extra four work-role warps do not stage, reduce, or consume contract-visible
data; removing them reduces CTA barrier participation without changing the
producer-to-partial-list or partial-list-to-output path.
"""
from __future__ import annotations
from json import loads as _json_loads
from .._dispatch_runtime import _capture_cuTensorMapEncodeTiled, _decode_capture, _import_dispatch_module, _ir_proxy
from .._dispatch_runtime import dc as dc
from typing import Any
from .._dispatch_runtime import evaluate
from . import knn_search_target0629_d1024_q8_m65536_k10_root_q8stage_v1 as parent
TARGET_LABEL = parent.TARGET_LABEL
TARGET_ROUTE = 'target0630_d1024_q8_m65536_k10_floor14_b3fc_tcgen05_16warp_const148merge'
TARGET_SPLIT_M = parent.TARGET_SPLIT_M
TARGET_Q = parent.TARGET_Q
TARGET_M = parent.TARGET_M
K_MAX = parent.K_MAX
THREADS = 512
SMEM_BYTES = parent.HIGH_MMA_SMEM_BYTES
TARGET_SHAPES: list[dict[str, Any]] = [{'label': TARGET_LABEL, 'params': {'B': 1, 'Q': TARGET_Q, 'M': TARGET_M, 'D': parent.HIGH_D_MAX, 'K': K_MAX, 'dtype': 'bfloat16', 'seed': 612110, 'self_search': False, 'min_recall': 0.999}}]
_parent_role = _decode_capture(_json_loads('{"__ir__": "loom.examples.weave.knn_search_target0630_d1024_q8_m65536_k10_floor14_b3fc_v1:_parent_role"}'))
_work_role = _decode_capture(_json_loads('{"__ir__": "loom.examples.weave.knn_search_target0630_d1024_q8_m65536_k10_floor14_b3fc_v1:_work_role"}'))
_warps = _decode_capture(_json_loads('{"__ir__": "loom.examples.weave.knn_search_target0630_d1024_q8_m65536_k10_floor14_b3fc_v1:_warps"}'))
partial_ir = _decode_capture(_json_loads('{"__ir__": "knn_search_d1024_q8_tcgen05_partial_16warp_b3fc_v1", "arg_keys": ["queries", "database", "partial_distances", "partial_indices", "B", "Q", "M", "split_m", "num_q_tiles", "total_m_tiles", "active_split_m"], "cluster_dims": [1, 1, 1], "computed_smem_bytes": 143104, "constants": [["K_MAX_", 10], ["D_ORIG_", 1024], ["NUM_D_PASSES_", 8], ["Q_NORM_PARTS_", 64]], "cta_group": 1, "threads": 512}'))
merge_ir = _decode_capture(_json_loads('{"__ir__": "knn_search_mma_split_merge_q128_const148_v1", "arg_keys": ["partial_distances", "partial_indices", "out_distances", "out_indices", "B", "Q", "K", "split_m", "num_q_tiles"], "cluster_dims": [1, 1, 1], "computed_smem_bytes": 0, "constants": [["K_MAX_", 10]], "cta_group": 1, "threads": 32}'))
ir = _decode_capture(_json_loads('{"__ir__": "knn_search_d1024_q8_tcgen05_partial_16warp_b3fc_v1", "arg_keys": ["queries", "database", "partial_distances", "partial_indices", "B", "Q", "M", "split_m", "num_q_tiles", "total_m_tiles", "active_split_m"], "cluster_dims": [1, 1, 1], "computed_smem_bytes": 143104, "constants": [["K_MAX_", 10], ["D_ORIG_", 1024], ["NUM_D_PASSES_", 8], ["Q_NORM_PARTS_", 64]], "cta_group": 1, "threads": 512}'))
_KERNELS: dict[str, Any] = {}

def selected_route(inputs: dict[str, Any]) -> str:
    if int(inputs['B']) == 1 and int(inputs['Q']) == TARGET_Q and (int(inputs['M']) == TARGET_M) and (int(inputs['D']) == parent.HIGH_D_MAX) and (int(inputs['K']) == K_MAX) and (not bool(inputs.get('self_search', False))) and (not bool(inputs.get('force_fallback', False))) and parent.mma._tcgen05_capable_arch():
        return TARGET_ROUTE
    return 'unsupported_shape'

def route_info(inputs: dict[str, Any]) -> dict[str, Any]:
    route = selected_route(inputs)
    return {'route': route, 'selected_route': route, 'route_kind': 'specialized' if route == TARGET_ROUTE else 'unsupported', 'route_source': 'bucket-specific-seed', 'coverage_only': False, 'production_policy': 'weave_only', 'arch_requirement': 'sm_100a', 'padding_tag': 'none', 'uses_materialized_padding': False, 'uses_kernel_padding': False, 'padding_overhead_timed': False, 'padded_D': parent.HIGH_D_MAX, 'workspace_reuse': 'producer partial scratch cache'}

def _compile() -> dict[str, Any]:
    return _decode_capture(_json_loads('{"partial": {"__kernel__": "dispatch_kernel_0575"}, "shared_mem": 143104}'))

def launch_for_eval(inputs: dict[str, Any]) -> dict[str, Any]:
    if selected_route(inputs) != TARGET_ROUTE:
        raise ValueError('16warp seed supports only B=1,Q=8,M=65536,D=1024,K=10, non-self search')
    kernels = _KERNELS.get('d1024')
    if kernels is None:
        kernels = _compile()
        _KERNELS['d1024'] = kernels
    partial_dist, partial_idx = parent._partial_scratch(inputs, TARGET_SPLIT_M, 1)
    kernels['partial'].launch(grid=(TARGET_SPLIT_M, 1, 1), block=(THREADS, 1, 1), args=[inputs['queries'], inputs['database'], partial_dist, partial_idx, 1, TARGET_Q, TARGET_M, TARGET_SPLIT_M, 1, TARGET_M // parent.BLOCK_M, TARGET_M // parent.BLOCK_M // TARGET_SPLIT_M + 1], shared_mem=int(kernels['shared_mem']))
    parent._target_merge_kernel().launch(grid=(TARGET_Q, 1, 1), block=(parent.MERGE_THREADS, 1, 1), args=[partial_dist, partial_idx, inputs['out_distances'], inputs['out_indices'], 1, TARGET_Q, K_MAX, TARGET_SPLIT_M, 1], shared_mem=parent.MERGE_SMEM_BYTES)
    return {'distances': inputs['out_distances'], 'indices': inputs['out_indices']}

def knn_search_compile_and_launch_target0630(*, benchmark: bool=True) -> dict[str, Any]:
    return evaluate(launch_for_eval, shapes=TARGET_SHAPES, benchmark=benchmark)
