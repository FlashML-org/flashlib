"""Round-73 repaired extended-K routes over the current registered kNN search.

Minimum target architecture: sm_100a for the tcgen05 extended-K routes. This
additive wrapper preserves the round-25 registered dispatcher for all existing
buckets, but routes exact ``B=1,Q=128,M>=131072,D=128,K=13..20`` through the
round-70 late-index stable merge and ``K in {31,32}`` through the round-68
static late-index merge.
"""
from __future__ import annotations
from json import loads as _json_loads
from .._dispatch_runtime import _capture_cuTensorMapEncodeTiled, _decode_capture, _import_dispatch_module, _ir_proxy
from typing import Any
from . import knn_search_k20k30_lateidx_dispatch0610_r70_8386_v1 as k20_lateidx
from . import knn_search_k31k32_staticmerge_dispatch0610_r68_8386_v1 as k31_staticmerge
from . import knn_search_k64_q4096split80_twotile_registered_0612_r25_4e2c_v1 as parent
ir = _decode_capture(_json_loads('{"__ir__": "knn_search_k64_q4096split80_twotile_partial_0612_r25_4e2c_v1", "arg_keys": ["queries", "database", "partial_distances", "partial_indices", "B", "Q", "M", "split_m", "num_q_tiles", "total_m_tiles", "tiles_per_split"], "cluster_dims": [1, 1, 1], "computed_smem_bytes": 165120, "constants": [["K_MAX_", 64]], "cta_group": 1, "threads": 512}'))
partial_ir = _decode_capture(_json_loads('{"__ir__": "knn_search_k64_q4096split80_twotile_partial_0612_r25_4e2c_v1", "arg_keys": ["queries", "database", "partial_distances", "partial_indices", "B", "Q", "M", "split_m", "num_q_tiles", "total_m_tiles", "tiles_per_split"], "cluster_dims": [1, 1, 1], "computed_smem_bytes": 165120, "constants": [["K_MAX_", 64]], "cta_group": 1, "threads": 512}'))
merge_ir = _decode_capture(_json_loads('{"__ir__": "knn_search_k64_q4096split80_merge10_0612_r22_4e2c_v1", "arg_keys": ["partial_distances", "partial_indices", "out_distances", "out_indices", "B", "Q", "K", "split_m", "num_q_tiles"], "cluster_dims": [1, 1, 1], "computed_smem_bytes": 0, "constants": [["K_MAX_", 64]], "cta_group": 1, "threads": 32}'))
k20_partial_ir = _decode_capture(_json_loads('{"__ir__": "knn_search_mma_split_partial_v1", "arg_keys": ["queries", "database", "partial_distances", "partial_indices", "B", "Q", "M", "split_m", "num_q_tiles", "total_m_tiles", "tiles_per_split"], "cluster_dims": [1, 1, 1], "computed_smem_bytes": 165120, "constants": [["K_MAX_", 20], ["EXPOSE_COL_COHORTS", 0]], "cta_group": 1, "threads": 512}'))
k20_merge_ir = _decode_capture(_json_loads('{"__ir__": "knn_search_k20k30_q128_split148_lateidx_merge_r70_v1", "arg_keys": ["partial_distances", "partial_indices", "out_distances", "out_indices", "B", "Q", "K", "split_m", "num_q_tiles"], "cluster_dims": [1, 1, 1], "computed_smem_bytes": 0, "constants": [["K_MAX_", 20]], "cta_group": 1, "threads": 32}'))
k31_partial_ir = _decode_capture(_json_loads('{"__ir__": "knn_search_mma_split_partial_v1", "arg_keys": ["queries", "database", "partial_distances", "partial_indices", "B", "Q", "M", "split_m", "num_q_tiles", "total_m_tiles", "tiles_per_split"], "cluster_dims": [1, 1, 1], "computed_smem_bytes": 165120, "constants": [["K_MAX_", 32], ["EXPOSE_COL_COHORTS", 0]], "cta_group": 1, "threads": 512}'))
k31_merge_ir = _decode_capture(_json_loads('{"__ir__": "knn_search_k31k32_q128_split148_static_lateidx_merge_r68_v1", "arg_keys": ["partial_distances", "partial_indices", "out_distances", "out_indices", "B", "Q", "K", "split_m", "num_q_tiles"], "cluster_dims": [1, 1, 1], "computed_smem_bytes": 0, "constants": [["K_MAX_", 32]], "cta_group": 1, "threads": 32}'))
K20_K31_R25_REGISTERED_SHAPES: list[dict[str, Any]] = [{'label': 'r73_q128_k20_lateidx', 'params': {'B': 1, 'Q': 128, 'M': 131072, 'D': 128, 'K': 20, 'dtype': 'bfloat16', 'seed': 610308, 'self_search': False, 'min_recall': 0.999}}, {'label': 'r73_q128_k31_staticmerge', 'params': {'B': 1, 'Q': 128, 'M': 131072, 'D': 128, 'K': 31, 'dtype': 'bfloat16', 'seed': 610314, 'self_search': False, 'min_recall': 0.999}}, {'label': 'r73_q128_k32_staticmerge', 'params': {'B': 1, 'Q': 128, 'M': 131072, 'D': 128, 'K': 32, 'dtype': 'bfloat16', 'seed': 610315, 'self_search': False, 'min_recall': 0.999}}, {'label': 'r73_q4096_k64_twotile', 'params': {'B': 1, 'Q': 4096, 'M': 20000, 'D': 128, 'K': 64, 'dtype': 'bfloat16', 'seed': 610313, 'self_search': False, 'min_recall': 0.999}}]

def _use_k13_k20_lateidx(inputs: dict[str, Any]) -> bool:
    return k20_lateidx._bucket_for_k(int(inputs['K'])) is not None and k20_lateidx._use_bucket_mma(inputs)

def _use_k31_k32_staticmerge(inputs: dict[str, Any]) -> bool:
    return k31_staticmerge._bucket_for_k(int(inputs['K'])) is not None and k31_staticmerge._use_staticmerge_mma(inputs)

def launch_for_eval(inputs: dict[str, Any]) -> dict[str, Any]:
    if _use_k13_k20_lateidx(inputs):
        return k20_lateidx.launch_for_eval(inputs)
    if _use_k31_k32_staticmerge(inputs):
        return k31_staticmerge.launch_for_eval(inputs)
    return parent.launch_for_eval(inputs)

def _select_contract_shapes(shape_labels: str | tuple[str, ...] | list[str] | None):
    return parent._select_contract_shapes(shape_labels)

def knn_search_compile_and_launch_mma_split(*, benchmark: bool=True, shape_labels: str | tuple[str, ...] | list[str] | None=None) -> dict[str, Any]:
    from .._dispatch_runtime import evaluate
    result = evaluate(launch_for_eval, shapes=_select_contract_shapes(shape_labels), benchmark=benchmark)
    result['passed'] = bool(result.get('summary', {}).get('all_correct'))
    print(result)
    return result

def knn_search_compile_and_launch_k20_k31_r25_registered(*, benchmark: bool=True, shapes: list[dict[str, Any]] | None=None) -> dict[str, Any]:
    from .._dispatch_runtime import evaluate
    result = evaluate(launch_for_eval, shapes=K20_K31_R25_REGISTERED_SHAPES if shapes is None else shapes, benchmark=benchmark)
    result['passed'] = bool(result.get('summary', {}).get('all_correct'))
    print(result)
    return result
