"""Restored exact D4096/Q1/M65536/K10 tcgen05 seed.

Minimum target architecture: sm_100a.  This additive seed restores the
validated 296-way qcache-pipeline producer/merge schedule after the 128-way
split-M experiment regressed the exact contract bucket.  The eval path remains
fully Weave: tcgen05 MMA -> TMEM readback -> partial top-10 -> Weave merge.
"""
from __future__ import annotations
from json import loads as _json_loads
from .._dispatch_runtime import _capture_cuTensorMapEncodeTiled, _decode_capture, _import_dispatch_module, _ir_proxy
from typing import Any
from .._dispatch_runtime import evaluate
from . import knn_search_target0630_d4096_q1_m65536_k10_qcache_pipe_69ea_v1 as base
TARGET_LABEL = base.TARGET_LABEL
TARGET_SHAPES = base.TARGET_SHAPES
ROUTE = 'target0630_d4096_q1_m65536_k10_restore296_tcgen05'
ENTRYPOINT = 'loom.examples.weave.knn_search_target0630_d4096_q1_m65536_k10_restore296_92e6_v1:launch_for_eval'
SPLIT_M = base.SPLIT_M
partial_ir = _decode_capture(_json_loads('{"__ir__": "knn_search_d4096_q1_m65536_k10_partial_qcache_pipe_69ea_v1", "arg_keys": ["queries", "database", "partial_distances", "partial_indices", "B", "Q", "M", "split_m", "num_q_tiles", "total_m_tiles"], "cluster_dims": [1, 1, 1], "computed_smem_bytes": 62848, "constants": [["K_MAX_", 10]], "cta_group": 1, "threads": 512}'))
merge_ir = _decode_capture(_json_loads('{"__ir__": "knn_search_mma_split_merge_v1", "arg_keys": ["partial_distances", "partial_indices", "out_distances", "out_indices", "B", "Q", "K", "split_m", "num_q_tiles"], "cluster_dims": [1, 1, 1], "computed_smem_bytes": 0, "constants": [["K_MAX_", 10], ["MERGE_SLOTS_", 5]], "cta_group": 1, "threads": 32}'))
ir = _decode_capture(_json_loads('{"__ir__": "knn_search_d4096_q1_m65536_k10_partial_qcache_pipe_69ea_v1", "arg_keys": ["queries", "database", "partial_distances", "partial_indices", "B", "Q", "M", "split_m", "num_q_tiles", "total_m_tiles"], "cluster_dims": [1, 1, 1], "computed_smem_bytes": 62848, "constants": [["K_MAX_", 10]], "cta_group": 1, "threads": 512}'))

def _matches(inputs: dict[str, Any]) -> bool:
    return base._matches(inputs)

def selected_route(inputs: dict[str, Any]) -> str:
    return ROUTE if _matches(inputs) else 'unsupported_shape'

def route_info(inputs: dict[str, Any]) -> dict[str, Any]:
    info = dict(base.route_info(inputs))
    info.update({'route': selected_route(inputs), 'selected_route': selected_route(inputs), 'selected_entrypoint': ENTRYPOINT, 'route_source': 'bucket-specific-regression-restore'})
    return info

def launch_for_eval(inputs: dict[str, Any]) -> dict[str, Any]:
    if not _matches(inputs):
        raise ValueError('seed supports only B=1,Q=1,M=65536,D=4096,K=10, non-self search')
    return base.launch_for_eval(inputs)

def compile_and_launch(*, benchmark: bool=True) -> dict[str, Any]:
    return evaluate(launch_for_eval, shapes=TARGET_SHAPES, benchmark=benchmark)
