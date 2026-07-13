"""Profile-guided exact D4096/Q4/M32768/K10 tcgen05 kNN candidate.

Minimum target architecture: sm_100a.  For the exact B=1/Q=4/M=32768/D=4096/
K=10 route this keeps two M128 tiles owned by each split CTA in separate TMEM
and SMEM-B slots.  Q is staged once per D128 pass and the first tile's MMA can
run while the second tile's B stage is being populated.  The exact top-10 merge
and the contract-visible distance/index output ABI remain unchanged.
"""
from __future__ import annotations
from json import loads as _json_loads
from .._dispatch_runtime import _capture_cuTensorMapEncodeTiled, _decode_capture, _import_dispatch_module, _ir_proxy
import math
from typing import Any
from .._dispatch_runtime import evaluate, select_named_shapes
from . import knn_search_target0627_d4096_q4_m32768_k10_split128_r221_1900_v1 as parent
THREADS = parent.THREADS
MERGE_THREADS = parent.MERGE_THREADS
BLOCK_Q = parent.BLOCK_Q
BLOCK_M = parent.BLOCK_M
D_ORIG = parent.D_ORIG
K_MAX = parent.K_MAX
SPLIT_M = parent.SPLIT_M
seed = parent.parent.seed
SMEM_A_BYTES = seed.SMEM_A_BYTES
SMEM_B0_OFFSET = SMEM_A_BYTES
SMEM_B1_OFFSET = SMEM_B0_OFFSET + seed.SMEM_B_BYTES
SMEM_Q_NORM_PART_OFFSET = SMEM_B1_OFFSET + seed.SMEM_B_BYTES
SMEM_DB_NORM_PART_OFFSET = SMEM_Q_NORM_PART_OFFSET + seed.SMEM_Q_NORM_PART_BYTES
SMEM_DB_NORM0_OFFSET = SMEM_DB_NORM_PART_OFFSET + seed.SMEM_DB_NORM_PART_BYTES
SMEM_DB_NORM1_OFFSET = SMEM_DB_NORM0_OFFSET + seed.SMEM_DB_NORM_BYTES
SMEM_LOCAL_D_OFFSET = SMEM_DB_NORM1_OFFSET + seed.SMEM_DB_NORM_BYTES
SMEM_LOCAL_I_OFFSET = SMEM_LOCAL_D_OFFSET + seed.SMEM_LOCAL_D_BYTES
SMEM_POOL_BYTES = SMEM_LOCAL_I_OFFSET + seed.SMEM_LOCAL_I_BYTES + 256
SMEM_BYTES = SMEM_POOL_BYTES + parent.mma.WEAVE_SMEM_SYSTEM_BYTES
ROUTE = 'profiled_target0630_d4096_q4_m32768_k10_qreuse_tcgen05'
ENTRYPOINT = 'loom.examples.weave.knn_search_target0630_d4096_q4_m32768_k10_profiled_weave_evolve_v1:launch_for_eval'
TARGET_LABELS = ('target0627_d4096_q4_m32768_k10',)
TARGET_SHAPES = _decode_capture(_json_loads('[{"__dict_items__": [["label", "target0627_d4096_q4_m32768_k10"], ["params", {"__dict_items__": [["B", 1], ["Q", 4], ["M", 32768], ["D", 4096], ["K", 10], ["dtype", "bfloat16"], ["seed", 612114], ["self_search", false], ["min_recall", 0.999]]}]]}]'))
_stage_issue_pair = _ir_proxy('loom.examples.weave.knn_search_target0630_d4096_q4_m32768_k10_profiled_weave_evolve_v1:_stage_issue_pair', 256)
knn_search_target0630_d4096_q4_m32768_k10_partial_qreuse_v1 = _decode_capture(_json_loads('{"__ir__": "knn_search_target0630_d4096_q4_m32768_k10_partial_qreuse_v1", "arg_keys": ["queries", "database", "partial_distances", "partial_indices", "B", "Q", "M", "split_m", "num_q_tiles", "total_m_tiles"], "cluster_dims": [1, 1, 1], "computed_smem_bytes": 192768, "constants": [["K_MAX_", 10]], "cta_group": 1, "threads": 512}'))
partial_ir = _decode_capture(_json_loads('{"__ir__": "knn_search_target0630_d4096_q4_m32768_k10_partial_qreuse_v1", "arg_keys": ["queries", "database", "partial_distances", "partial_indices", "B", "Q", "M", "split_m", "num_q_tiles", "total_m_tiles"], "cluster_dims": [1, 1, 1], "computed_smem_bytes": 192768, "constants": [["K_MAX_", 10]], "cta_group": 1, "threads": 512}'))
merge_ir = _decode_capture(_json_loads('{"__ir__": "knn_search_target0627_d4096_q4_m32768_k10_merge128_r221_v1", "arg_keys": ["partial_distances", "partial_indices", "out_distances", "out_indices", "B", "Q", "K", "split_m", "num_q_tiles"], "cluster_dims": [1, 1, 1], "computed_smem_bytes": 0, "constants": [["K_MAX_", 10]], "cta_group": 1, "threads": 32}'))
ir = _decode_capture(_json_loads('{"__ir__": "knn_search_target0630_d4096_q4_m32768_k10_partial_qreuse_v1", "arg_keys": ["queries", "database", "partial_distances", "partial_indices", "B", "Q", "M", "split_m", "num_q_tiles", "total_m_tiles"], "cluster_dims": [1, 1, 1], "computed_smem_bytes": 192768, "constants": [["K_MAX_", 10]], "cta_group": 1, "threads": 512}'))
_KERNELS: dict[str, Any] = {}

def _active(inputs: dict[str, Any]) -> bool:
    return (int(inputs.get('B', 1)), int(inputs['Q']), int(inputs['M']), int(inputs['D']), int(inputs['K']), bool(inputs.get('self_search', False))) == (1, 4, 32768, 4096, 10, False) and (not bool(inputs.get('force_fallback', False))) and seed._tcgen05_capable_arch()

def selected_route(inputs: dict[str, Any]) -> str:
    return ROUTE if _active(inputs) else parent.selected_route(inputs)

def selected_route_name(inputs: dict[str, Any]) -> str:
    return selected_route(inputs)

def route_info(inputs: dict[str, Any]) -> dict[str, Any]:
    if not _active(inputs):
        return parent.route_info(inputs)
    return {'route': ROUTE, 'selected_route': ROUTE, 'selected_entrypoint': ENTRYPOINT, 'route_kind': 'specialized', 'route_source': 'shape-specific-seed', 'coverage_class': 'bucket_seed_target0627_d4096_q4_m32768_k10', 'production_policy': 'weave_only', 'external_fallback': None, 'guard_id': 'target0627_d4096_q4_m32768_k10_profiled_qreuse', 'guard_condition': 'B==1,Q==4,M==32768,D==4096,K==10,nonself,sm100a_or_sm103a', 'selected_seed': 'weave-evolve-knn-search-profiled-qreuse', 'padding_tag': 'none', 'uses_materialized_padding': False, 'uses_kernel_padding': False, 'padding_overhead_timed': False, 'padded_D': D_ORIG, 'workspace_reuse': True}

def _compile_kernels() -> dict[str, Any]:
    return _decode_capture(_json_loads('{"merge": {"__kernel__": "dispatch_kernel_0145"}, "partial": {"__kernel__": "dispatch_kernel_0144"}}'))

def _launch_exact(inputs: dict[str, Any]) -> dict[str, Any]:
    import torch
    if not _KERNELS:
        _KERNELS.update(_compile_kernels())
    num_q_tiles = math.ceil(int(inputs['Q']) / BLOCK_Q)
    total_m_tiles = math.ceil(int(inputs['M']) / BLOCK_M)
    partial_dist, partial_idx = seed._scratch(inputs, SPLIT_M, num_q_tiles)
    _KERNELS['partial'].launch(grid=(int(inputs['B']) * num_q_tiles * SPLIT_M, 1, 1), block=(THREADS, 1, 1), args=[inputs['queries'], inputs['database'], partial_dist, partial_idx, int(inputs['B']), int(inputs['Q']), int(inputs['M']), SPLIT_M, num_q_tiles, total_m_tiles], shared_mem=SMEM_BYTES)
    _KERNELS['merge'].launch(grid=(int(inputs['B']) * int(inputs['Q']), 1, 1), block=(MERGE_THREADS, 1, 1), args=[partial_dist, partial_idx, inputs['out_distances'], inputs['out_indices'], int(inputs['B']), int(inputs['Q']), int(inputs['K']), SPLIT_M, num_q_tiles], shared_mem=parent.mma.MERGE_SMEM_BYTES)
    return {'distances': inputs['out_distances'], 'indices': inputs['out_indices']}

def launch_for_eval(inputs: dict[str, Any]) -> dict[str, Any]:
    return _launch_exact(inputs) if _active(inputs) else parent.launch_for_eval(inputs)

def compile_and_launch(*, benchmark: bool=True) -> dict[str, Any]:
    result = evaluate(launch_for_eval, shapes=TARGET_SHAPES, benchmark=benchmark)
    result['passed'] = bool(result.get('summary', {}).get('all_correct'))
    print(result)
    return result
