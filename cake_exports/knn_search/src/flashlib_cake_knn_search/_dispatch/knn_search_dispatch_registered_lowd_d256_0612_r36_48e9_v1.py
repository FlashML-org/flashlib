"""Round-36 registered kNN dispatcher with low-D/D256 application routes.

Minimum target architecture: sm_80 for inherited scalar/low-D routes and
sm_100a for the tcgen05 routes. This additive dispatcher preserves the current
registered D128/K64 dispatcher, and routes only the expanded low-D plus
``B=1,Q=128,M=131072,D=256,K in {10,64}`` application labels through the
round-35 D256 K10/K64 composite route.
"""
from __future__ import annotations
from json import loads as _json_loads
from .._dispatch_runtime import _capture_cuTensorMapEncodeTiled, _decode_capture, _import_dispatch_module, _ir_proxy
from typing import Any
from . import knn_search_k64_q4096split79_twotile_oddevensort_fastmerge_0612_r34_11c1_v1 as registered_parent
from . import knn_search_lowd_d256_over48e9_0612_r35_k10k64mma_v1 as lowd_d256
ir = _decode_capture(_json_loads('{"__ir__": "knn_search_mma_split_partial_v1", "arg_keys": ["queries", "database", "partial_distances", "partial_indices", "B", "Q", "M", "split_m", "num_q_tiles", "total_m_tiles", "tiles_per_split"], "cluster_dims": [1, 1, 1], "computed_smem_bytes": 143104, "constants": [["K_MAX_", 64], ["EXPOSE_COL_COHORTS", 1]], "cta_group": 1, "threads": 256}'))
partial_ir = _decode_capture(_json_loads('{"__ir__": "knn_search_mma_split_partial_v1", "arg_keys": ["queries", "database", "partial_distances", "partial_indices", "B", "Q", "M", "split_m", "num_q_tiles", "total_m_tiles", "tiles_per_split"], "cluster_dims": [1, 1, 1], "computed_smem_bytes": 143104, "constants": [["K_MAX_", 64], ["EXPOSE_COL_COHORTS", 1]], "cta_group": 1, "threads": 256}'))
merge_ir = _decode_capture(_json_loads('{"__ir__": "knn_search_mma_split_merge_v1", "arg_keys": ["partial_distances", "partial_indices", "out_distances", "out_indices", "B", "Q", "K", "split_m", "num_q_tiles"], "cluster_dims": [1, 1, 1], "computed_smem_bytes": 0, "constants": [["K_MAX_", 64]], "cta_group": 1, "threads": 32}'))
registered_ir = _decode_capture(_json_loads('{"__ir__": "knn_search_k64_q4096split79_twotile_oddevensort_partial_0612_r34_11c1_v1", "arg_keys": ["queries", "database", "partial_distances", "partial_indices", "B", "Q", "M", "split_m", "num_q_tiles", "total_m_tiles", "tiles_per_split"], "cluster_dims": [1, 1, 1], "computed_smem_bytes": 165120, "constants": [["K_MAX_", 64]], "cta_group": 1, "threads": 512}'))
LOWD_D256_OVER48E9_SHAPES = lowd_d256.LOWD_D256_OVER48E9_SHAPES
LOWD_D256_PRESERVE_SHAPES = lowd_d256.LOWD_D256_PRESERVE_SHAPES
REGISTERED_Q4096_K64_PRESERVE_SHAPES = [shape for shape in registered_parent.K64_Q4096_SPLIT79_ODDEVENSORT_FASTMERGE_SHAPES if shape['label'] == 'ksweep_q4096_m20000_d128_k64']
DISPATCH_R36_SHAPES = [*LOWD_D256_OVER48E9_SHAPES, *REGISTERED_Q4096_K64_PRESERVE_SHAPES]
_LOWD_D256_KEYS = {(int(shape['params']['B']), int(shape['params']['Q']), int(shape['params']['M']), int(shape['params']['D']), int(shape['params']['K'])) for shape in LOWD_D256_OVER48E9_SHAPES}

def _shape_key(inputs: dict[str, Any]) -> tuple[int, int, int, int, int]:
    return (int(inputs['B']), int(inputs['Q']), int(inputs['M']), int(inputs['D']), int(inputs['K']))

def _use_lowd_d256_application(inputs: dict[str, Any]) -> bool:
    return _shape_key(inputs) in _LOWD_D256_KEYS

def selected_route(inputs: dict[str, Any]) -> str:
    if _use_lowd_d256_application(inputs):
        return 'r35_lowd_d256_k10k64_composite'
    return 'registered_parent_q4096split79'

def launch_for_eval(inputs: dict[str, Any]) -> dict[str, Any]:
    if _use_lowd_d256_application(inputs):
        return lowd_d256.launch_for_eval(inputs)
    return registered_parent.launch_for_eval(inputs)

def _select_contract_shapes(shape_labels: str | tuple[str, ...] | list[str] | None):
    return registered_parent._select_contract_shapes(shape_labels)

def knn_search_compile_and_launch_mma_split(*, benchmark: bool=True, shape_labels: str | tuple[str, ...] | list[str] | None=None) -> dict[str, Any]:
    from .._dispatch_runtime import evaluate
    result = evaluate(launch_for_eval, shapes=_select_contract_shapes(shape_labels), benchmark=benchmark)
    result['passed'] = bool(result.get('summary', {}).get('all_correct'))
    print(result)
    return result

def knn_search_compile_and_launch_registered_lowd_d256(*, benchmark: bool=True, shapes: list[dict[str, Any]] | None=None) -> dict[str, Any]:
    from .._dispatch_runtime import evaluate
    result = evaluate(launch_for_eval, shapes=DISPATCH_R36_SHAPES if shapes is None else shapes, benchmark=benchmark)
    result['passed'] = bool(result.get('summary', {}).get('all_correct'))
    print(result)
    return result
