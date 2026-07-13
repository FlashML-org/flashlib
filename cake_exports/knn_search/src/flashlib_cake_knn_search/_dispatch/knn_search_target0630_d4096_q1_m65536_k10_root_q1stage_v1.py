"""Exact D4096/Q1/M65536/K10 tcgen05 kNN seed with Q1 query staging.

Minimum target architecture: sm_100a.  The tcgen05 producer and Weave merge
remain on the contract path; this Q1 specialization leaves unused tcgen05
rows unwritten instead of materializing zero query rows for every M tile.
"""
from __future__ import annotations
from json import loads as _json_loads
from .._dispatch_runtime import _capture_cuTensorMapEncodeTiled, _decode_capture, _import_dispatch_module, _ir_proxy
import math
from typing import Any
from .._dispatch_runtime import evaluate
from . import knn_search_d4096_q4q8_m8192m16384_k10_0623_5ff7_v1 as p
TARGET_LABEL = 'target0627_d4096_q1_m65536_k10'
ROUTE = 'root_target0630_d4096_q1_m65536_k10_q1stage_tcgen05'
ENTRYPOINT = 'loom.examples.weave.knn_search_target0630_d4096_q1_m65536_k10_root_q1stage_v1:launch_for_eval'
TARGET_SHAPES: list[dict[str, Any]] = [{'label': TARGET_LABEL, 'params': {'B': 1, 'Q': 1, 'M': 65536, 'D': 4096, 'K': 10, 'dtype': 'bfloat16', 'seed': 612113, 'self_search': False, 'min_recall': 1.0}}]
THREADS = p.THREADS
BLOCK_Q = p.BLOCK_Q
BLOCK_M = p.BLOCK_M
D_STAGE = p.D_STAGE
D_ORIG = p.D_ORIG
NUM_D_PASSES = p.NUM_D_PASSES
K_MAX = p.K_MAX
VEC = p.VEC
PACK_WORDS = p.PACK_WORDS
PARTIAL_STRIDE_Q = p.PARTIAL_STRIDE_Q
LOCAL_LISTS_PER_ROW = p.LOCAL_LISTS_PER_ROW
SMEM_BYTES = p.SMEM_BYTES
_KERNELS: dict[str, Any] = {}
_accumulate_q0_norm = _ir_proxy('loom.examples.weave.knn_search_target0630_d4096_q1_m65536_k10_root_q1stage_v1:_accumulate_q0_norm', 256)
_stage_q0_pass = _ir_proxy('loom.examples.weave.knn_search_target0630_d4096_q1_m65536_k10_root_q1stage_v1:_stage_q0_pass', 256)
knn_search_d4096_q1_m65536_k10_partial_q1stage_v1 = _decode_capture(_json_loads('{"__ir__": "knn_search_d4096_q1_m65536_k10_partial_q1stage_v1", "arg_keys": ["queries", "database", "partial_distances", "partial_indices", "B", "Q", "M", "split_m", "num_q_tiles", "total_m_tiles"], "cluster_dims": [1, 1, 1], "computed_smem_bytes": 159488, "constants": [["K_MAX_", 10]], "cta_group": 1, "threads": 512}'))
partial_ir = _decode_capture(_json_loads('{"__ir__": "knn_search_d4096_q1_m65536_k10_partial_q1stage_v1", "arg_keys": ["queries", "database", "partial_distances", "partial_indices", "B", "Q", "M", "split_m", "num_q_tiles", "total_m_tiles"], "cluster_dims": [1, 1, 1], "computed_smem_bytes": 159488, "constants": [["K_MAX_", 10]], "cta_group": 1, "threads": 512}'))
merge_ir = _decode_capture(_json_loads('{"__ir__": "knn_search_d4096_q4q8_m8192m16384_k10_merge_0623_5ff7_v1", "arg_keys": ["partial_distances", "partial_indices", "out_distances", "out_indices", "B", "Q", "K", "split_m", "num_q_tiles"], "cluster_dims": [1, 1, 1], "computed_smem_bytes": 0, "constants": [["K_MAX_", 10]], "cta_group": 1, "threads": 32}'))
ir = _decode_capture(_json_loads('{"__ir__": "knn_search_d4096_q1_m65536_k10_partial_q1stage_v1", "arg_keys": ["queries", "database", "partial_distances", "partial_indices", "B", "Q", "M", "split_m", "num_q_tiles", "total_m_tiles"], "cluster_dims": [1, 1, 1], "computed_smem_bytes": 159488, "constants": [["K_MAX_", 10]], "cta_group": 1, "threads": 512}'))

def _matches(inputs: dict[str, Any]) -> bool:
    return int(inputs['B']) == 1 and int(inputs['Q']) == 1 and (int(inputs['M']) == 65536) and (int(inputs['D']) == D_ORIG) and (int(inputs['K']) == K_MAX) and (not bool(inputs.get('self_search', False))) and (not bool(inputs.get('force_fallback', False))) and p._tcgen05_capable_arch()

def _compile_kernels() -> dict[str, Any]:
    return _decode_capture(_json_loads('{"merge": {"__kernel__": "dispatch_kernel_0126"}, "partial": {"__kernel__": "dispatch_kernel_0273"}}'))

def selected_route(inputs: dict[str, Any]) -> str:
    return ROUTE if _matches(inputs) else 'unsupported_shape'

def route_info(inputs: dict[str, Any]) -> dict[str, Any]:
    return {'route': selected_route(inputs), 'selected_route': selected_route(inputs), 'selected_entrypoint': ENTRYPOINT, 'route_kind': 'specialized', 'route_source': 'bucket-specific-seed', 'production_policy': 'weave_only', 'padding_tag': 'none', 'uses_materialized_padding': False, 'uses_kernel_padding': False, 'padding_overhead_timed': False, 'padded_D': D_ORIG, 'workspace_reuse': 'partial scratch cache'}

def launch_for_eval(inputs: dict[str, Any]) -> dict[str, Any]:
    if not _matches(inputs):
        raise ValueError('seed supports only B=1,Q=1,M=65536,D=4096,K=10, non-self search')
    if not _KERNELS:
        _KERNELS.update(_compile_kernels())
    split_m = min(p.SPLIT_M, math.ceil(int(inputs['M']) / BLOCK_M))
    partial_dist, partial_idx = p._scratch(inputs, split_m, 1)
    _KERNELS['partial'].launch(grid=(split_m, 1, 1), block=(THREADS, 1, 1), args=[inputs['queries'], inputs['database'], partial_dist, partial_idx, 1, 1, int(inputs['M']), split_m, 1, math.ceil(int(inputs['M']) / BLOCK_M)], shared_mem=SMEM_BYTES)
    _KERNELS['merge'].launch(grid=(1, 1, 1), block=(p.MERGE_THREADS, 1, 1), args=[partial_dist, partial_idx, inputs['out_distances'], inputs['out_indices'], 1, 1, K_MAX, split_m, 1], shared_mem=p.mma.MERGE_SMEM_BYTES)
    return {'distances': inputs['out_distances'], 'indices': inputs['out_indices']}

def compile_and_launch(*, benchmark: bool=True) -> dict[str, Any]:
    return evaluate(launch_for_eval, shapes=TARGET_SHAPES, benchmark=benchmark)
