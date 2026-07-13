"""Round-37 D128/Q128 dispatcher over the r36 low-D/D256 kNN routes.

Minimum target architecture: sm_100a for the D128/Q128 and D256 tcgen05
routes. This additive dispatcher preserves the round-36 low-D/D256/Q4096
preservation wrapper and adds the source-clean Q128/D128/K<=10 split-policy
route for neighboring m-bucket and rag-hot contract labels.
"""
from __future__ import annotations
from json import loads as _json_loads
from .._dispatch_runtime import _capture_cuTensorMapEncodeTiled, _decode_capture, _import_dispatch_module, _ir_proxy
from typing import Any
from .._dispatch_runtime import select_named_shapes
from . import knn_search_lowd_d256_dispatch_over48e9_0612_r36_48e9_v1 as parent
from . import knn_search_q128_split_policy_f295_v1 as d128_q128
THREADS = d128_q128.THREADS
BLOCK_Q = d128_q128.BLOCK_Q
BLOCK_M = d128_q128.BLOCK_M
D_STATIC = d128_q128.D_STATIC
K_MAX = d128_q128.K_MAX
ir = _decode_capture(_json_loads('{"__ir__": "knn_search_mma_split_partial_v1", "arg_keys": ["queries", "database", "partial_distances", "partial_indices", "B", "Q", "M", "split_m", "num_q_tiles", "total_m_tiles", "tiles_per_split"], "cluster_dims": [1, 1, 1], "computed_smem_bytes": 108800, "constants": [["K_MAX_", 10], ["EXPOSE_COL_COHORTS", 0], ["FULL_M_TILES", 0]], "cta_group": 1, "threads": 640}'))
partial_ir = _decode_capture(_json_loads('{"__ir__": "knn_search_mma_split_partial_v1", "arg_keys": ["queries", "database", "partial_distances", "partial_indices", "B", "Q", "M", "split_m", "num_q_tiles", "total_m_tiles", "tiles_per_split"], "cluster_dims": [1, 1, 1], "computed_smem_bytes": 108800, "constants": [["K_MAX_", 10], ["EXPOSE_COL_COHORTS", 0], ["FULL_M_TILES", 0]], "cta_group": 1, "threads": 640}'))
parent_ir = _decode_capture(_json_loads('{"__ir__": "knn_search_mma_split_partial_v1", "arg_keys": ["queries", "database", "partial_distances", "partial_indices", "B", "Q", "M", "split_m", "num_q_tiles", "total_m_tiles", "tiles_per_split"], "cluster_dims": [1, 1, 1], "computed_smem_bytes": 143104, "constants": [["K_MAX_", 10], ["EXPOSE_COL_COHORTS", 1]], "cta_group": 1, "threads": 256}'))
d256_partial_ir = _decode_capture(_json_loads('{"__ir__": "knn_search_mma_split_partial_v1", "arg_keys": ["queries", "database", "partial_distances", "partial_indices", "B", "Q", "M", "split_m", "num_q_tiles", "total_m_tiles", "tiles_per_split"], "cluster_dims": [1, 1, 1], "computed_smem_bytes": 143104, "constants": [["K_MAX_", 10], ["EXPOSE_COL_COHORTS", 1]], "cta_group": 1, "threads": 256}'))
d256_merge_ir = _decode_capture(_json_loads('{"__ir__": "knn_search_mma_split_merge_v1", "arg_keys": ["partial_distances", "partial_indices", "out_distances", "out_indices", "B", "Q", "K", "split_m", "num_q_tiles"], "cluster_dims": [1, 1, 1], "computed_smem_bytes": 0, "constants": [["K_MAX_", 10]], "cta_group": 1, "threads": 32}'))
k64_partial_ir = _decode_capture(_json_loads('{"__ir__": "knn_search_mma_split_partial_v1", "arg_keys": ["queries", "database", "partial_distances", "partial_indices", "B", "Q", "M", "split_m", "num_q_tiles", "total_m_tiles", "tiles_per_split"], "cluster_dims": [1, 1, 1], "computed_smem_bytes": 143104, "constants": [["K_MAX_", 64], ["EXPOSE_COL_COHORTS", 1]], "cta_group": 1, "threads": 256}'))
k64_merge_ir = _decode_capture(_json_loads('{"__ir__": "knn_search_mma_split_merge_v1", "arg_keys": ["partial_distances", "partial_indices", "out_distances", "out_indices", "B", "Q", "K", "split_m", "num_q_tiles"], "cluster_dims": [1, 1, 1], "computed_smem_bytes": 0, "constants": [["K_MAX_", 64]], "cta_group": 1, "threads": 32}'))
D128_Q128_SHAPE_LABELS: tuple[str, ...] = ('dispatch_q128_m8192_d128_k10', 'dispatch_q128_m16384_d128_k10', 'dispatch_q128_m32768_d128_k10', 'dispatch_q128_m65536_d128_k10', 'rag_q128_m131072_d128_k10', 'rag_batch_q128_m131072_d128_k10', 'dispatch_q128_m262144_d128_k10')
D128_Q128_DISPATCH_SHAPES = _decode_capture(_json_loads('[{"__dict_items__": [["label", "dispatch_q128_m8192_d128_k10"], ["params", {"__dict_items__": [["B", 1], ["Q", 128], ["M", 8192], ["D", 128], ["K", 10], ["dtype", "bfloat16"], ["seed", 610201], ["self_search", false], ["min_recall", 0.999]]}]]}, {"__dict_items__": [["label", "dispatch_q128_m16384_d128_k10"], ["params", {"__dict_items__": [["B", 1], ["Q", 128], ["M", 16384], ["D", 128], ["K", 10], ["dtype", "bfloat16"], ["seed", 610202], ["self_search", false], ["min_recall", 0.999]]}]]}, {"__dict_items__": [["label", "dispatch_q128_m32768_d128_k10"], ["params", {"__dict_items__": [["B", 1], ["Q", 128], ["M", 32768], ["D", 128], ["K", 10], ["dtype", "bfloat16"], ["seed", 610203], ["self_search", false], ["min_recall", 0.999]]}]]}, {"__dict_items__": [["label", "dispatch_q128_m65536_d128_k10"], ["params", {"__dict_items__": [["B", 1], ["Q", 128], ["M", 65536], ["D", 128], ["K", 10], ["dtype", "bfloat16"], ["seed", 610204], ["self_search", false], ["min_recall", 0.999]]}]]}, {"__dict_items__": [["label", "rag_q128_m131072_d128_k10"], ["params", {"__dict_items__": [["B", 1], ["Q", 128], ["M", 131072], ["D", 128], ["K", 10], ["dtype", "bfloat16"], ["seed", 2], ["self_search", false], ["check_correctness", true], ["benchmark", true], ["min_recall", 0.999]]}]]}, {"__dict_items__": [["label", "rag_batch_q128_m131072_d128_k10"], ["params", {"__dict_items__": [["B", 1], ["Q", 128], ["M", 131072], ["D", 128], ["K", 10], ["dtype", "bfloat16"], ["seed", 610109], ["self_search", false], ["min_recall", 0.999]]}]]}, {"__dict_items__": [["label", "dispatch_q128_m262144_d128_k10"], ["params", {"__dict_items__": [["B", 1], ["Q", 128], ["M", 262144], ["D", 128], ["K", 10], ["dtype", "bfloat16"], ["seed", 610205], ["self_search", false], ["min_recall", 0.999]]}]]}]'))
LOWD_D256_OVER48E9_SHAPES = parent.LOWD_D256_OVER48E9_SHAPES
LOWD_D256_PRESERVE_SHAPES = parent.LOWD_D256_PRESERVE_SHAPES
ROUND37_DISPATCH_SHAPES = [*D128_Q128_DISPATCH_SHAPES, *LOWD_D256_OVER48E9_SHAPES]
ROUND37_PRESERVE_SHAPES = [*ROUND37_DISPATCH_SHAPES, LOWD_D256_PRESERVE_SHAPES[-1]]
SHAPE_DISPATCH_REGISTRY: tuple[dict[str, str], ...] = ({'shape_key': 'd128_q128_mbucket_k10', 'guard': 'B == 1 and Q == 128 and D == 128 and 8192 <= M and K <= 10 and tcgen05', 'route': 'round_f295_d128_q128_split_policy'}, *parent.SHAPE_DISPATCH_REGISTRY)

def _tcgen05_capable_arch() -> bool:
    return d128_q128._incumbent._tcgen05_capable_arch()

def _use_d128_q128_policy(inputs: dict[str, Any]) -> bool:
    return _tcgen05_capable_arch() and d128_q128._use_q128_split_dispatch(inputs)

def selected_route(inputs: dict[str, Any]) -> str:
    if _use_d128_q128_policy(inputs):
        return 'round_f295_d128_q128_split_policy'
    return parent.selected_route(inputs)

def launch_for_eval(inputs: dict[str, Any]) -> dict[str, Any]:
    if _use_d128_q128_policy(inputs):
        return d128_q128.launch_for_eval(inputs)
    return parent.launch_for_eval(inputs)

def _select_contract_shapes(shape_labels: str | tuple[str, ...] | list[str] | None):
    if shape_labels is None:
        return None
    return select_named_shapes(shape_labels)

def knn_search_compile_and_launch_d128_q128_dispatch(*, benchmark: bool=True, shapes: list[dict[str, Any]] | None=None) -> dict[str, Any]:
    from .._dispatch_runtime import evaluate
    result = evaluate(launch_for_eval, shapes=D128_Q128_DISPATCH_SHAPES if shapes is None else shapes, benchmark=benchmark)
    result['passed'] = bool(result.get('summary', {}).get('all_correct'))
    print(result)
    return result

def knn_search_compile_and_launch_round37_dispatch(*, benchmark: bool=True, shapes: list[dict[str, Any]] | None=None) -> dict[str, Any]:
    from .._dispatch_runtime import evaluate
    result = evaluate(launch_for_eval, shapes=ROUND37_DISPATCH_SHAPES if shapes is None else shapes, benchmark=benchmark)
    result['passed'] = bool(result.get('summary', {}).get('all_correct'))
    print(result)
    return result

def knn_search_compile_and_launch_round37_preserve(*, benchmark: bool=True) -> dict[str, Any]:
    from .._dispatch_runtime import evaluate
    result = evaluate(launch_for_eval, shapes=ROUND37_PRESERVE_SHAPES, benchmark=benchmark)
    result['passed'] = bool(result.get('summary', {}).get('all_correct'))
    print(result)
    return result
