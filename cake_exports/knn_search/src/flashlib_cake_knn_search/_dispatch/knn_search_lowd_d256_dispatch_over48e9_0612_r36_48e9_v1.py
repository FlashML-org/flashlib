"""Round-36 dispatcher over the validated low-D/D256/48e9 kNN routes.

Minimum target architecture: sm_100a for the registered D256 tcgen05 routes.
Low-D and non-tcgen05 fallback routes inherit the round-34/round-33 minimum
targets. This additive dispatcher consumes the round-35 D256 K10-capacity
shape kernel and the round-34 D256 K64-capacity shape kernel, with all other
coverage delegated to the round-34 application wrapper.
"""
from __future__ import annotations
from json import loads as _json_loads
from .._dispatch_runtime import _capture_cuTensorMapEncodeTiled, _decode_capture, _import_dispatch_module, _ir_proxy
from typing import Any
from . import knn_search_lowd_d256_mma64_over48e9_0612_r34_48e9_v1 as d256_k64
from . import knn_search_lowd_d256_mma64k10_over48e9_0612_r35_48e9_v1 as d256_k10
THREADS = d256_k64.THREADS
BLOCK_Q = d256_k64.BLOCK_Q
BLOCK_M = d256_k64.BLOCK_M
D_STATIC = d256_k64.D_STATIC
MERGE_THREADS = d256_k64.MERGE_THREADS
MMA_SMEM_BYTES = d256_k64.MMA_SMEM_BYTES
MERGE_SMEM_BYTES = d256_k64.MERGE_SMEM_BYTES
D256_K10_MAX = d256_k10.D256_K10_MAX
D256_K10_SPLIT_M = d256_k10.D256_K10_SPLIT_M
D256_K64_MAX = d256_k64.K_MAX
D256_K64_SPLIT_M = d256_k64.D256_MMA_SPLIT_M
partial_ir = _decode_capture(_json_loads('{"__ir__": "knn_search_mma_split_partial_v1", "arg_keys": ["queries", "database", "partial_distances", "partial_indices", "B", "Q", "M", "split_m", "num_q_tiles", "total_m_tiles", "tiles_per_split"], "cluster_dims": [1, 1, 1], "computed_smem_bytes": 143104, "constants": [["K_MAX_", 10], ["EXPOSE_COL_COHORTS", 1]], "cta_group": 1, "threads": 256}'))
merge_ir = _decode_capture(_json_loads('{"__ir__": "knn_search_mma_split_merge_v1", "arg_keys": ["partial_distances", "partial_indices", "out_distances", "out_indices", "B", "Q", "K", "split_m", "num_q_tiles"], "cluster_dims": [1, 1, 1], "computed_smem_bytes": 0, "constants": [["K_MAX_", 10]], "cta_group": 1, "threads": 32}'))
ir = _decode_capture(_json_loads('{"__ir__": "knn_search_mma_split_partial_v1", "arg_keys": ["queries", "database", "partial_distances", "partial_indices", "B", "Q", "M", "split_m", "num_q_tiles", "total_m_tiles", "tiles_per_split"], "cluster_dims": [1, 1, 1], "computed_smem_bytes": 143104, "constants": [["K_MAX_", 10], ["EXPOSE_COL_COHORTS", 1]], "cta_group": 1, "threads": 256}'))
k64_partial_ir = _decode_capture(_json_loads('{"__ir__": "knn_search_mma_split_partial_v1", "arg_keys": ["queries", "database", "partial_distances", "partial_indices", "B", "Q", "M", "split_m", "num_q_tiles", "total_m_tiles", "tiles_per_split"], "cluster_dims": [1, 1, 1], "computed_smem_bytes": 143104, "constants": [["K_MAX_", 64], ["EXPOSE_COL_COHORTS", 1]], "cta_group": 1, "threads": 256}'))
k64_merge_ir = _decode_capture(_json_loads('{"__ir__": "knn_search_mma_split_merge_v1", "arg_keys": ["partial_distances", "partial_indices", "out_distances", "out_indices", "B", "Q", "K", "split_m", "num_q_tiles"], "cluster_dims": [1, 1, 1], "computed_smem_bytes": 0, "constants": [["K_MAX_", 64]], "cta_group": 1, "threads": 32}'))
LOWD_D256_OVER48E9_SHAPES = d256_k10.LOWD_D256_OVER48E9_SHAPES
LOWD_D256_PRESERVE_SHAPES = d256_k10.LOWD_D256_PRESERVE_SHAPES
D256_DISPATCH_SHAPES = [shape for shape in LOWD_D256_OVER48E9_SHAPES if shape['label'] in {'glm5_rag_q128_m131072_d256_k10', 'glm5_rag_q128_m131072_d256_k64'}]
SHAPE_DISPATCH_REGISTRY: tuple[dict[str, str], ...] = ({'shape_key': 'd256_q128_m131072_k10', 'guard': 'B == 1 and Q == 128 and M == 131072 and D == 256 and K == 10 and tcgen05', 'route': 'round35_d256_k10_capacity_tcgen05'}, {'shape_key': 'd256_q128_m131072_k64', 'guard': 'B == 1 and Q == 128 and M == 131072 and D == 256 and K == 64 and tcgen05', 'route': 'round34_d256_k64_capacity_tcgen05'}, {'shape_key': 'inherited_lowd_48e9', 'guard': 'otherwise', 'route': 'round34_application_wrapper'})

def _is_d256_contract_shape(inputs: dict[str, Any]) -> bool:
    return int(inputs['B']) == 1 and int(inputs['Q']) == 128 and (int(inputs['M']) == 131072) and (int(inputs['D']) == D_STATIC)

def _use_registered_k10(inputs: dict[str, Any]) -> bool:
    return d256_k64._tcgen05_capable_arch() and _is_d256_contract_shape(inputs) and (int(inputs['K']) == D256_K10_MAX)

def _use_registered_k64(inputs: dict[str, Any]) -> bool:
    return d256_k64._tcgen05_capable_arch() and _is_d256_contract_shape(inputs) and (int(inputs['K']) == D256_K64_MAX)

def selected_route(inputs: dict[str, Any]) -> str:
    if _use_registered_k10(inputs):
        return 'round35_d256_k10_capacity_tcgen05'
    if _use_registered_k64(inputs):
        return 'round34_d256_k64_capacity_tcgen05'
    return 'round34_application_wrapper'

def launch_for_eval(inputs: dict[str, Any]) -> dict[str, Any]:
    route = selected_route(inputs)
    if route == 'round35_d256_k10_capacity_tcgen05':
        return d256_k10._launch_d256_k10_mma(inputs)
    if route == 'round34_d256_k64_capacity_tcgen05':
        return d256_k64._launch_d256_mma(inputs)
    return d256_k64.launch_for_eval(inputs)

def _select_contract_shapes(shape_labels: str | tuple[str, ...] | list[str] | None):
    return d256_k64._select_contract_shapes(shape_labels)

def knn_search_compile_and_launch_mma_split(*, benchmark: bool=True, shape_labels: str | tuple[str, ...] | list[str] | None=None) -> dict[str, Any]:
    from .._dispatch_runtime import evaluate
    result = evaluate(launch_for_eval, shapes=_select_contract_shapes(shape_labels), benchmark=benchmark)
    result['passed'] = bool(result.get('summary', {}).get('all_correct'))
    print(result)
    return result

def knn_search_compile_and_launch_lowd_d256_over48e9(*, benchmark: bool=True, shapes: list[dict[str, Any]] | None=None) -> dict[str, Any]:
    from .._dispatch_runtime import evaluate
    result = evaluate(launch_for_eval, shapes=LOWD_D256_OVER48E9_SHAPES if shapes is None else shapes, benchmark=benchmark)
    result['passed'] = bool(result.get('summary', {}).get('all_correct'))
    print(result)
    return result

def knn_search_compile_and_launch_lowd_d256_preserve(*, benchmark: bool=True) -> dict[str, Any]:
    from .._dispatch_runtime import evaluate
    result = evaluate(launch_for_eval, shapes=LOWD_D256_PRESERVE_SHAPES, benchmark=benchmark)
    result['passed'] = bool(result.get('summary', {}).get('all_correct'))
    print(result)
    return result
