"""Fixed148 Q8 tcgen05 kNN with 128 active producers and identity slots.

Minimum target architecture: sm_100a.  This exact-shape bucket candidate keeps
the physical 148-list partial ABI consumed by the constant-148 merge, but only
the first 128 producer CTAs scan database tiles.  The remaining physical slots
write the producer's INF/-1 top10 identity lists, so they participate in the
contract-visible merge without changing keyed scratch identity.
"""
from __future__ import annotations
from json import loads as _json_loads
from .._dispatch_runtime import _capture_cuTensorMapEncodeTiled, _decode_capture, _import_dispatch_module, _ir_proxy
from typing import Any
from .._dispatch_runtime import evaluate
from . import knn_search_target0629_d1024_q8_m65536_k10_root_q8stage_v1 as incumbent
D_ORIG = 1024
K_MAX = 10
TARGET_LABEL = 'target0627_d1024_q8_m65536_k10'
PHYSICAL_SPLITS = 148
ACTIVE_SPLITS = 128
ROUTE = 'q8_fixed148_active128_tcgen05_identity_slots'
ENTRYPOINT = 'loom.examples.weave.knn_search_q8_fixed148_compaction_52ab_v1:launch_for_eval'
TARGET_SHAPES: list[dict[str, Any]] = [{'label': TARGET_LABEL, 'params': {'B': 1, 'Q': 8, 'M': 65536, 'D': D_ORIG, 'K': K_MAX, 'dtype': 'bfloat16', 'seed': 612110, 'self_search': False, 'min_recall': 0.999}}]
partial_ir = _decode_capture(_json_loads('{"__ir__": "knn_search_dynamic_d768d1024_q32q16_tcgen05_partial_0618_9286_v1", "arg_keys": ["queries", "database", "partial_distances", "partial_indices", "B", "Q", "M", "split_m", "num_q_tiles", "total_m_tiles", "active_split_m"], "cluster_dims": [1, 1, 1], "computed_smem_bytes": 143104, "constants": [["K_MAX_", 10], ["D_ORIG_", 1024], ["NUM_D_PASSES_", 8], ["Q_NORM_PARTS_", 64]], "cta_group": 1, "threads": 640}'))
merge_ir = _decode_capture(_json_loads('{"__ir__": "knn_search_mma_split_merge_q128_const148_v1", "arg_keys": ["partial_distances", "partial_indices", "out_distances", "out_indices", "B", "Q", "K", "split_m", "num_q_tiles"], "cluster_dims": [1, 1, 1], "computed_smem_bytes": 0, "constants": [["K_MAX_", 10]], "cta_group": 1, "threads": 32}'))
ir = _decode_capture(_json_loads('{"__ir__": "knn_search_dynamic_d768d1024_q32q16_tcgen05_partial_0618_9286_v1", "arg_keys": ["queries", "database", "partial_distances", "partial_indices", "B", "Q", "M", "split_m", "num_q_tiles", "total_m_tiles", "active_split_m"], "cluster_dims": [1, 1, 1], "computed_smem_bytes": 143104, "constants": [["K_MAX_", 10], ["D_ORIG_", 1024], ["NUM_D_PASSES_", 8], ["Q_NORM_PARTS_", 64]], "cta_group": 1, "threads": 640}'))

def _matches(inputs: dict[str, Any]) -> bool:
    return int(inputs['B']) == 1 and int(inputs['Q']) == 8 and (int(inputs['M']) == 65536) and (int(inputs['D']) == D_ORIG) and (int(inputs['K']) == K_MAX) and (not bool(inputs.get('self_search', False))) and (not bool(inputs.get('force_fallback', False))) and incumbent.mma._tcgen05_capable_arch()

def selected_route(inputs: dict[str, Any]) -> str:
    return ROUTE if _matches(inputs) else 'unsupported_shape'

def route_info(inputs: dict[str, Any]) -> dict[str, Any]:
    if not _matches(inputs):
        return {'route': 'unsupported_shape', 'selected_route': 'unsupported_shape', 'route_kind': 'unsupported'}
    return {'route': ROUTE, 'selected_route': ROUTE, 'selected_entrypoint': ENTRYPOINT, 'route_kind': 'specialized', 'route_source': 'q8-fixed148-compaction-52ab', 'coverage_class': TARGET_LABEL, 'classification': 'seed-produced', 'coverage_only': False, 'production_policy': 'weave_only', 'external_fallback': None, 'guard_id': TARGET_LABEL, 'fallback': None, 'missing_weave_route': False, 'arch_requirement': 'sm_100a', 'padding_tag': 'none', 'uses_materialized_padding': False, 'uses_kernel_padding': False, 'padding_overhead_timed': False, 'padded_D': D_ORIG, 'workspace_reuse': 'keyed fixed148 producer partial scratch and cubin cache', 'physical_partial_lists': PHYSICAL_SPLITS, 'active_producer_lists': ACTIVE_SPLITS, 'identity_partial_lists': PHYSICAL_SPLITS - ACTIVE_SPLITS}

def launch_for_eval(inputs: dict[str, Any]) -> dict[str, Any]:
    if not _matches(inputs):
        raise ValueError('active128 fixed148 route supports only B=1,Q=8,M=65536,D=1024,K=10, non-self search')
    kernels = incumbent._DIRECT_MMA_KERNELS.get(D_ORIG)
    if kernels is None:
        kernels = incumbent._compile_direct_mma_kernels(D_ORIG)
        incumbent._DIRECT_MMA_KERNELS[D_ORIG] = kernels
    partial_dist, partial_idx = incumbent._partial_scratch(inputs, PHYSICAL_SPLITS, 1)
    kernels['partial'].launch(grid=(PHYSICAL_SPLITS, 1, 1), block=(incumbent.THREADS, 1, 1), args=[inputs['queries'], inputs['database'], partial_dist, partial_idx, 1, 8, 65536, PHYSICAL_SPLITS, 1, 65536 // incumbent.BLOCK_M, ACTIVE_SPLITS], shared_mem=int(kernels['shared_mem']))
    incumbent._target_merge_kernel().launch(grid=(8, 1, 1), block=(incumbent.MERGE_THREADS, 1, 1), args=[partial_dist, partial_idx, inputs['out_distances'], inputs['out_indices'], 1, 8, K_MAX, PHYSICAL_SPLITS, 1], shared_mem=incumbent.MERGE_SMEM_BYTES)
    return {'distances': inputs['out_distances'], 'indices': inputs['out_indices']}

def compile_and_launch(*, benchmark: bool=True) -> dict[str, Any]:
    return evaluate(launch_for_eval, shapes=TARGET_SHAPES, benchmark=benchmark)
