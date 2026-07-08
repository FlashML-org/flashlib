"""Exact D256/Q24/M49152/K10 split-M=128 tcgen05 kNN seed.

Minimum target architecture: sm_100a.  This additive exact-shape variant keeps
the D256/K10 tcgen05 producer and contract-visible Weave top-10 merge, while
using 128 producer CTAs with exactly three database tiles per CTA, eliminating
the uneven tail work of the prior 148-way split while reducing merge fan-in.
"""
from __future__ import annotations
from json import loads as _json_loads
from .._dispatch_runtime import _capture_cuTensorMapEncodeTiled, _decode_capture, _import_dispatch_module, _ir_proxy
import math
from typing import Any
from . import knn_search_target0630_d256_q24_m49152_k10_5992_v1 as seed
THREADS = seed.THREADS
MERGE_THREADS = seed.MERGE_THREADS
BLOCK_Q = seed.BLOCK_Q
BLOCK_M = seed.BLOCK_M
K_MAX = seed.K_MAX
POST_MMA_COL_COHORTS = seed.POST_MMA_COL_COHORTS
MMA_SMEM_BYTES = seed.MMA_SMEM_BYTES
MERGE_SMEM_BYTES = seed.MERGE_SMEM_BYTES
SPLIT_M = 128
partial_ir = _decode_capture(_json_loads('{"__ir__": "knn_search_mma_split_partial_v1", "arg_keys": ["queries", "database", "partial_distances", "partial_indices", "B", "Q", "M", "split_m", "num_q_tiles", "total_m_tiles", "tiles_per_split"], "cluster_dims": [1, 1, 1], "computed_smem_bytes": 143104, "constants": [["K_MAX_", 10], ["EXPOSE_COL_COHORTS", 1]], "cta_group": 1, "threads": 256}'))
merge_ir = _decode_capture(_json_loads('{"__ir__": "knn_search_mma_split_merge_v1", "arg_keys": ["partial_distances", "partial_indices", "out_distances", "out_indices", "B", "Q", "K", "split_m", "num_q_tiles"], "cluster_dims": [1, 1, 1], "computed_smem_bytes": 0, "constants": [["K_MAX_", 10]], "cta_group": 1, "threads": 32}'))
ir = _decode_capture(_json_loads('{"__ir__": "knn_search_mma_split_partial_v1", "arg_keys": ["queries", "database", "partial_distances", "partial_indices", "B", "Q", "M", "split_m", "num_q_tiles", "total_m_tiles", "tiles_per_split"], "cluster_dims": [1, 1, 1], "computed_smem_bytes": 143104, "constants": [["K_MAX_", 10], ["EXPOSE_COL_COHORTS", 1]], "cta_group": 1, "threads": 256}'))
TARGET_LABEL = 'coverage_random_d256_q24_m49152_k10'
TARGET_SHAPES = seed.TARGET_SHAPES
ROUTE = 'target0630_d256_q24_m49152_k10_tcgen05_split128_d256specializedseed_v2'
ENTRYPOINT = 'loom.examples.weave.knn_search_target0630_d256_q24_m49152_k10_d256specializedseed_v2:launch_for_eval'
SELECTED_SEED = 'weave-evolve-knn-search-d256specializedseed-v2'
SHAPE_DISPATCH_REGISTRY = ({'shape_key': TARGET_LABEL, 'label': TARGET_LABEL, 'shape': (1, 24, 49152, 256, 10, False), 'guard': 'B == 1 and Q == 24 and M == 49152 and D == 256 and K == 10 and not self_search and not force_fallback and arch in {sm_100a,sm_103a}', 'route': ROUTE, 'entrypoint': ENTRYPOINT, 'selected_seed': SELECTED_SEED, 'source_round_doc': 'design_doc/active/weave_evolve_knn_search_round_239_d256specializedseed.md', 'coverage_class': 'bucket_seed_d256_q24_m49152_k10_tcgen05_split128', 'arch_requirement': 'sm_100a', 'split_m': SPLIT_M, 'padding_tag': 'none', 'workspace_reuse': 'cached partial scratch keyed by exact shape/device/dtype'},)

def _shape_key(inputs: dict[str, Any]) -> tuple[int, int, int, int, int, bool]:
    return (int(inputs.get('B', 1)), int(inputs['Q']), int(inputs['M']), int(inputs['D']), int(inputs['K']), bool(inputs.get('self_search', False)))

def _use_target(inputs: dict[str, Any]) -> bool:
    return not bool(inputs.get('force_fallback', False)) and seed.d256_k10.parent._tcgen05_capable_arch() and (_shape_key(inputs) == SHAPE_DISPATCH_REGISTRY[0]['shape'])

def _launch_target(inputs: dict[str, Any]) -> dict[str, Any]:
    kernels = seed._ensure_kernels()
    bsz, q_rows, m_rows, k = (int(inputs['B']), int(inputs['Q']), int(inputs['M']), int(inputs['K']))
    num_q_tiles = math.ceil(q_rows / BLOCK_Q)
    total_m_tiles = math.ceil(m_rows / BLOCK_M)
    split_m = min(SPLIT_M, total_m_tiles)
    tiles_per_split = math.ceil(total_m_tiles / split_m)
    partial_list_count = split_m * POST_MMA_COL_COHORTS
    partial_dist, partial_idx = seed._scratch(inputs, partial_list_count, num_q_tiles)
    kernels['partial'].launch(grid=(bsz * num_q_tiles * split_m, 1, 1), block=(THREADS, 1, 1), args=[inputs['queries'], inputs['database'], partial_dist, partial_idx, bsz, q_rows, m_rows, split_m, num_q_tiles, total_m_tiles, tiles_per_split], shared_mem=MMA_SMEM_BYTES)
    kernels['merge'].launch(grid=(bsz * q_rows, 1, 1), block=(MERGE_THREADS, 1, 1), args=[partial_dist, partial_idx, inputs['out_distances'], inputs['out_indices'], bsz, q_rows, k, partial_list_count, num_q_tiles], shared_mem=MERGE_SMEM_BYTES)
    return {'distances': inputs['out_distances'], 'indices': inputs['out_indices']}

def selected_route(inputs: dict[str, Any]) -> str:
    return ROUTE if _use_target(inputs) else seed.parent.selected_route(inputs)

def selected_route_name(inputs: dict[str, Any]) -> str:
    return selected_route(inputs)

def route_info(inputs: dict[str, Any]) -> dict[str, Any]:
    if not _use_target(inputs):
        return dict(seed.parent.route_info(inputs))
    entry = SHAPE_DISPATCH_REGISTRY[0]
    return {'route': ROUTE, 'selected_route': ROUTE, 'selected_entrypoint': ENTRYPOINT, 'route_kind': 'specialized', 'route_source': 'shape-specific-seed', 'classification': 'seed-produced', 'coverage_only': False, 'production_policy': 'weave_only', 'external_fallback': None, 'guard_id': entry['shape_key'], 'selected_guard': entry['guard'], 'guard_condition': entry['guard'], 'selected_seed': SELECTED_SEED, 'source_round_doc': entry['source_round_doc'], 'arch_requirement': 'sm_100a', 'padding_tag': 'none', 'uses_materialized_padding': False, 'uses_kernel_padding': False, 'padding_overhead_timed': False, 'workspace_reuse': entry['workspace_reuse']}

def launch_for_eval(inputs: dict[str, Any]) -> dict[str, Any]:
    return _launch_target(inputs) if _use_target(inputs) else seed.parent.launch_for_eval(inputs)

def knn_search_compile_and_launch_target0630_d256_q24_m49152_k10(*, benchmark: bool=True) -> dict[str, Any]:
    from .._dispatch_runtime import evaluate
    result = evaluate(launch_for_eval, shapes=TARGET_SHAPES, benchmark=benchmark)
    result['passed'] = bool(result.get('summary', {}).get('all_correct'))
    print(result)
    return result
