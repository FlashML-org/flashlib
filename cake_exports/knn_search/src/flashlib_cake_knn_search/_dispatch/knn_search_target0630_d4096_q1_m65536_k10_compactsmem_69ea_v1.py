"""Exact D4096/Q1/M65536/K10 kNN seed with compact Q1 shared-memory state.

Minimum target architecture: sm_100a.  The tcgen05 database-dot producer,
TMEM readback, partial top-10, and Weave merge all remain on the eval path.
Unlike the general Q64 seed, this Q1 route allocates norm and local-list shared
memory only for the one contract-visible query row.
"""
from __future__ import annotations
from json import loads as _json_loads
from .._dispatch_runtime import _capture_cuTensorMapEncodeTiled, _decode_capture, _import_dispatch_module, _ir_proxy
import math
from typing import Any
from .._dispatch_runtime import evaluate
from . import knn_search_target0630_d4096_q1_m65536_k10_root_q1stage_v1 as parent
TARGET_LABEL = parent.TARGET_LABEL
TARGET_SHAPES = parent.TARGET_SHAPES
ROUTE = 'target0630_d4096_q1_m65536_k10_compactsmem_tcgen05'
ENTRYPOINT = 'loom.examples.weave.knn_search_target0630_d4096_q1_m65536_k10_compactsmem_69ea_v1:launch_for_eval'
SPLIT_M = 296
MERGE_SLOTS = _decode_capture(_json_loads('10'))
THREADS = parent.THREADS
BLOCK_M = parent.BLOCK_M
D_STAGE = parent.D_STAGE
D_ORIG = parent.D_ORIG
NUM_D_PASSES = parent.NUM_D_PASSES
K_MAX = parent.K_MAX
VEC = parent.VEC
PACK_WORDS = parent.PACK_WORDS
Q_NORM_PARTS = parent.p.Q_NORM_PARTS
DB_NORM_PARTS = parent.p.DB_NORM_PARTS
SMEM_A_BYTES = parent.p.SMEM_A_BYTES
SMEM_B_BYTES = parent.p.SMEM_B_BYTES
SMEM_Q_NORM_PART_BYTES = Q_NORM_PARTS * 4
SMEM_DB_NORM_PART_BYTES = parent.p.SMEM_DB_NORM_PART_BYTES
SMEM_DB_NORM_BYTES = parent.p.SMEM_DB_NORM_BYTES
SMEM_LOCAL_D_BYTES = 8 * K_MAX * 4
SMEM_LOCAL_I_BYTES = 8 * K_MAX * 4
SMEM_B_OFFSET = SMEM_A_BYTES
SMEM_Q_NORM_PART_OFFSET = SMEM_B_OFFSET + SMEM_B_BYTES
SMEM_DB_NORM_PART_OFFSET = SMEM_Q_NORM_PART_OFFSET + SMEM_Q_NORM_PART_BYTES
SMEM_DB_NORM_OFFSET = SMEM_DB_NORM_PART_OFFSET + SMEM_DB_NORM_PART_BYTES
SMEM_LOCAL_D_OFFSET = SMEM_DB_NORM_OFFSET + SMEM_DB_NORM_BYTES
SMEM_LOCAL_I_OFFSET = SMEM_LOCAL_D_OFFSET + SMEM_LOCAL_D_BYTES
SMEM_POOL_BYTES = SMEM_LOCAL_I_OFFSET + SMEM_LOCAL_I_BYTES + 256
SMEM_BYTES = SMEM_POOL_BYTES + parent.p.mma.WEAVE_SMEM_SYSTEM_BYTES
_KERNELS: dict[str, Any] = {}
_accumulate_q0_norm_compact = _ir_proxy('loom.examples.weave.knn_search_target0630_d4096_q1_m65536_k10_compactsmem_69ea_v1:_accumulate_q0_norm_compact', 256)
knn_search_d4096_q1_m65536_k10_partial_compactsmem_69ea_v1 = _decode_capture(_json_loads('{"__ir__": "knn_search_d4096_q1_m65536_k10_partial_compactsmem_69ea_v1", "arg_keys": ["queries", "database", "partial_distances", "partial_indices", "B", "Q", "M", "split_m", "num_q_tiles", "total_m_tiles"], "cluster_dims": [1, 1, 1], "computed_smem_bytes": 54656, "constants": [["K_MAX_", 10]], "cta_group": 1, "threads": 512}'))
partial_ir = _decode_capture(_json_loads('{"__ir__": "knn_search_d4096_q1_m65536_k10_partial_compactsmem_69ea_v1", "arg_keys": ["queries", "database", "partial_distances", "partial_indices", "B", "Q", "M", "split_m", "num_q_tiles", "total_m_tiles"], "cluster_dims": [1, 1, 1], "computed_smem_bytes": 54656, "constants": [["K_MAX_", 10]], "cta_group": 1, "threads": 512}'))
merge_ir = _decode_capture(_json_loads('{"__ir__": "knn_search_mma_split_merge_v1", "arg_keys": ["partial_distances", "partial_indices", "out_distances", "out_indices", "B", "Q", "K", "split_m", "num_q_tiles"], "cluster_dims": [1, 1, 1], "computed_smem_bytes": 0, "constants": [["K_MAX_", 10], ["MERGE_SLOTS_", 5]], "cta_group": 1, "threads": 32}'))
ir = _decode_capture(_json_loads('{"__ir__": "knn_search_d4096_q1_m65536_k10_partial_compactsmem_69ea_v1", "arg_keys": ["queries", "database", "partial_distances", "partial_indices", "B", "Q", "M", "split_m", "num_q_tiles", "total_m_tiles"], "cluster_dims": [1, 1, 1], "computed_smem_bytes": 54656, "constants": [["K_MAX_", 10]], "cta_group": 1, "threads": 512}'))

def _matches(inputs: dict[str, Any]) -> bool:
    return parent._matches(inputs)

def _compile_kernels() -> dict[str, Any]:
    return _decode_capture(_json_loads('{"merge": {"__kernel__": "dispatch_kernel_0143"}, "partial": {"__kernel__": "dispatch_kernel_0272"}}'))

def selected_route(inputs: dict[str, Any]) -> str:
    return ROUTE if _matches(inputs) else 'unsupported_shape'

def route_info(inputs: dict[str, Any]) -> dict[str, Any]:
    return {'route': selected_route(inputs), 'selected_route': selected_route(inputs), 'selected_entrypoint': ENTRYPOINT, 'route_kind': 'specialized', 'route_source': 'bucket-specific-seed', 'production_policy': 'weave_only', 'padding_tag': 'none', 'uses_materialized_padding': False, 'uses_kernel_padding': False, 'padding_overhead_timed': False, 'padded_D': D_ORIG, 'workspace_reuse': 'partial scratch cache'}

def launch_for_eval(inputs: dict[str, Any]) -> dict[str, Any]:
    if not _matches(inputs):
        raise ValueError('seed supports only B=1,Q=1,M=65536,D=4096,K=10, non-self search')
    if not _KERNELS:
        _KERNELS.update(_compile_kernels())
    total_m_tiles = math.ceil(int(inputs['M']) / BLOCK_M)
    split_m = min(SPLIT_M, total_m_tiles)
    partial_dist, partial_idx = parent.p._scratch(inputs, split_m, 1)
    _KERNELS['partial'].launch(grid=(split_m, 1, 1), block=(THREADS, 1, 1), args=[inputs['queries'], inputs['database'], partial_dist, partial_idx, 1, 1, int(inputs['M']), split_m, 1, total_m_tiles], shared_mem=SMEM_BYTES)
    _KERNELS['merge'].launch(grid=(1, 1, 1), block=(parent.p.MERGE_THREADS, 1, 1), args=[partial_dist, partial_idx, inputs['out_distances'], inputs['out_indices'], 1, 1, K_MAX, split_m, 1], shared_mem=parent.p.mma.MERGE_SMEM_BYTES)
    return {'distances': inputs['out_distances'], 'indices': inputs['out_indices']}

def compile_and_launch(*, benchmark: bool=True) -> dict[str, Any]:
    return evaluate(launch_for_eval, shapes=TARGET_SHAPES, benchmark=benchmark)
