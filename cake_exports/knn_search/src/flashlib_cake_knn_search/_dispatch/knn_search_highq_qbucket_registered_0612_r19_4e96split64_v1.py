"""Round-19 current-registry high-Q Q-bucket route for BF16 kNN search.

Minimum target architecture: sm_100a for the inherited tcgen05 MMA path. This
additive wrapper preserves the current registered round-73 dispatcher for all
routes except ``256 <= Q <= 2048, 16384 <= M <= 65536, D=128, K<=10``, which
uses the source-clean round-19 Q-bucket split policy.
"""
from __future__ import annotations
from json import loads as _json_loads
from .._dispatch_runtime import _capture_cuTensorMapEncodeTiled, _decode_capture, _import_dispatch_module, _ir_proxy
from typing import Any
from . import knn_search_highq_midm_qbucket_0611_r19_4e96_v1 as highq_qbucket
from . import knn_search_k20_k31_r25_registered_dispatch0610_r73_8386_v1 as parent
ir = _decode_capture(_json_loads('{"__ir__": "knn_search_mma_split_partial_v1", "arg_keys": ["queries", "database", "partial_distances", "partial_indices", "B", "Q", "M", "split_m", "num_q_tiles", "total_m_tiles", "tiles_per_split"], "cluster_dims": [1, 1, 1], "computed_smem_bytes": 108800, "constants": [["K_MAX_", 10], ["EXPOSE_COL_COHORTS", 0], ["FULL_M_TILES", 0]], "cta_group": 1, "threads": 640}'))
partial_ir = _decode_capture(_json_loads('{"__ir__": "knn_search_k64_q4096split80_twotile_partial_0612_r25_4e2c_v1", "arg_keys": ["queries", "database", "partial_distances", "partial_indices", "B", "Q", "M", "split_m", "num_q_tiles", "total_m_tiles", "tiles_per_split"], "cluster_dims": [1, 1, 1], "computed_smem_bytes": 165120, "constants": [["K_MAX_", 64]], "cta_group": 1, "threads": 512}'))
merge_ir = _decode_capture(_json_loads('{"__ir__": "knn_search_k64_q4096split80_merge10_0612_r22_4e2c_v1", "arg_keys": ["partial_distances", "partial_indices", "out_distances", "out_indices", "B", "Q", "K", "split_m", "num_q_tiles"], "cluster_dims": [1, 1, 1], "computed_smem_bytes": 0, "constants": [["K_MAX_", 64]], "cta_group": 1, "threads": 32}'))
k20_partial_ir = _decode_capture(_json_loads('{"__ir__": "knn_search_mma_split_partial_v1", "arg_keys": ["queries", "database", "partial_distances", "partial_indices", "B", "Q", "M", "split_m", "num_q_tiles", "total_m_tiles", "tiles_per_split"], "cluster_dims": [1, 1, 1], "computed_smem_bytes": 165120, "constants": [["K_MAX_", 20], ["EXPOSE_COL_COHORTS", 0]], "cta_group": 1, "threads": 512}'))
k20_merge_ir = _decode_capture(_json_loads('{"__ir__": "knn_search_k20k30_q128_split148_lateidx_merge_r70_v1", "arg_keys": ["partial_distances", "partial_indices", "out_distances", "out_indices", "B", "Q", "K", "split_m", "num_q_tiles"], "cluster_dims": [1, 1, 1], "computed_smem_bytes": 0, "constants": [["K_MAX_", 20]], "cta_group": 1, "threads": 32}'))
k31_partial_ir = _decode_capture(_json_loads('{"__ir__": "knn_search_mma_split_partial_v1", "arg_keys": ["queries", "database", "partial_distances", "partial_indices", "B", "Q", "M", "split_m", "num_q_tiles", "total_m_tiles", "tiles_per_split"], "cluster_dims": [1, 1, 1], "computed_smem_bytes": 165120, "constants": [["K_MAX_", 32], ["EXPOSE_COL_COHORTS", 0]], "cta_group": 1, "threads": 512}'))
k31_merge_ir = _decode_capture(_json_loads('{"__ir__": "knn_search_k31k32_q128_split148_static_lateidx_merge_r68_v1", "arg_keys": ["partial_distances", "partial_indices", "out_distances", "out_indices", "B", "Q", "K", "split_m", "num_q_tiles"], "cluster_dims": [1, 1, 1], "computed_smem_bytes": 0, "constants": [["K_MAX_", 32]], "cta_group": 1, "threads": 32}'))
HIGHQ_QBUCKET_REGISTERED_SHAPES: list[dict[str, Any]] = [{'label': 'dispatch_q256_m65536_d128_k10', 'params': {'B': 1, 'Q': 256, 'M': 65536, 'D': 128, 'K': 10, 'dtype': 'bfloat16', 'seed': 610206, 'self_search': False, 'min_recall': 0.999}}, {'label': 'dispatch_q512_m65536_d128_k10', 'params': {'B': 1, 'Q': 512, 'M': 65536, 'D': 128, 'K': 10, 'dtype': 'bfloat16', 'seed': 610207, 'self_search': False, 'min_recall': 0.999}}, {'label': 'dispatch_q1024_m65536_d128_k10', 'params': {'B': 1, 'Q': 1024, 'M': 65536, 'D': 128, 'K': 10, 'dtype': 'bfloat16', 'seed': 610208, 'self_search': False, 'min_recall': 0.999}}, {'label': 'dispatch_q2048_m65536_d128_k10', 'params': {'B': 1, 'Q': 2048, 'M': 65536, 'D': 128, 'K': 10, 'dtype': 'bfloat16', 'seed': 610209, 'self_search': False, 'min_recall': 0.999}}]

def _use_highq_qbucket(inputs: dict[str, Any]) -> bool:
    return highq_qbucket._use_highq_midm_qbucket_policy(inputs)

def launch_for_eval(inputs: dict[str, Any]) -> dict[str, Any]:
    if _use_highq_qbucket(inputs):
        return highq_qbucket._launch_highq_midm_qbucket_mma(inputs)
    return parent.launch_for_eval(inputs)

def _select_contract_shapes(shape_labels: str | tuple[str, ...] | list[str] | None):
    return parent._select_contract_shapes(shape_labels)

def knn_search_compile_and_launch_mma_split(*, benchmark: bool=True, shape_labels: str | tuple[str, ...] | list[str] | None=None) -> dict[str, Any]:
    from .._dispatch_runtime import evaluate
    result = evaluate(launch_for_eval, shapes=_select_contract_shapes(shape_labels), benchmark=benchmark)
    result['passed'] = bool(result.get('summary', {}).get('all_correct'))
    print(result)
    return result

def knn_search_compile_and_launch_highq_qbucket_registered(*, benchmark: bool=True, shapes: list[dict[str, Any]] | None=None) -> dict[str, Any]:
    from .._dispatch_runtime import evaluate
    result = evaluate(launch_for_eval, shapes=HIGHQ_QBUCKET_REGISTERED_SHAPES if shapes is None else shapes, benchmark=benchmark)
    result['passed'] = bool(result.get('summary', {}).get('all_correct'))
    print(result)
    return result
