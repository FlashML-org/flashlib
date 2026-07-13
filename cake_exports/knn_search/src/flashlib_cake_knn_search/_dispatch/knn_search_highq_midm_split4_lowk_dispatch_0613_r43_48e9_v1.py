"""Round-43 high-Q mid-M D128 split-4/low-K dispatcher for BF16 kNN.

Minimum target architecture: sm_100a for the tcgen05 high-Q MMA routes. This
additive dispatcher recreates the missing round-42 split-4 high-Q handoff under
a new namespace, routes Q4096/M20000/K in {1,2} through the existing round-41
K-specialized split-4 merge, routes the remaining Q4096/M16K..32K/K<=10 labels
through the split-4 tie-stable stream merge, and preserves the round-40 parent
for all other contract shapes.
"""
from __future__ import annotations
from json import loads as _json_loads
from .._dispatch_runtime import _capture_cuTensorMapEncodeTiled, _decode_capture, _import_dispatch_module, _ir_proxy
from typing import Any
from .._dispatch_runtime import select_named_shapes
from . import knn_search_highq_midm_over_r39_0613_r40_48e9_v1 as parent
from . import knn_search_highq_qbucket_registered_0612_r19_4e96split64_v1 as highq_qbucket
from . import knn_search_q4096_lowk_merge_0613_r41_48e9_v1 as q4096_lowk
from . import knn_search_q4096_split4_tiestable_0612_r15_4e2c_v1 as q4096_split4
THREADS = parent.THREADS
BLOCK_Q = parent.BLOCK_Q
BLOCK_M = parent.BLOCK_M
D_STATIC = parent.D_STATIC
K_MAX = parent.K_MAX
ir = _decode_capture(_json_loads('{"__ir__": "knn_search_mma_split_partial_v1", "arg_keys": ["queries", "database", "partial_distances", "partial_indices", "B", "Q", "M", "split_m", "num_q_tiles", "total_m_tiles", "tiles_per_split"], "cluster_dims": [1, 1, 1], "computed_smem_bytes": 108800, "constants": [["K_MAX_", 10], ["EXPOSE_COL_COHORTS", 0], ["FULL_M_TILES", 0]], "cta_group": 1, "threads": 640}'))
partial_ir = _decode_capture(_json_loads('{"__ir__": "knn_search_mma_split_partial_v1", "arg_keys": ["queries", "database", "partial_distances", "partial_indices", "B", "Q", "M", "split_m", "num_q_tiles", "total_m_tiles", "tiles_per_split"], "cluster_dims": [1, 1, 1], "computed_smem_bytes": 108800, "constants": [["K_MAX_", 10], ["EXPOSE_COL_COHORTS", 0], ["FULL_M_TILES", 0]], "cta_group": 1, "threads": 640}'))
q4096_ir = _decode_capture(_json_loads('{"__ir__": "knn_search_mma_split_partial_v1", "arg_keys": ["queries", "database", "partial_distances", "partial_indices", "B", "Q", "M", "split_m", "num_q_tiles", "total_m_tiles", "tiles_per_split"], "cluster_dims": [1, 1, 1], "computed_smem_bytes": 108800, "constants": [["K_MAX_", 10], ["EXPOSE_COL_COHORTS", 0], ["FULL_M_TILES", 0]], "cta_group": 1, "threads": 640}'))
q4096_lowk_merge_ir = _decode_capture(_json_loads('{"__ir__": "knn_search_mma_split_merge_q4096_lowk_v1", "arg_keys": ["partial_distances", "partial_indices", "out_distances", "out_indices", "B", "Q", "K", "split_m", "num_q_tiles"], "cluster_dims": [1, 1, 1], "computed_smem_bytes": 0, "constants": [["K_OUT_MAX_", 2]], "cta_group": 1, "threads": 32}'))
parent_ir = _decode_capture(_json_loads('{"__ir__": "knn_search_mma_split_partial_v1", "arg_keys": ["queries", "database", "partial_distances", "partial_indices", "B", "Q", "M", "split_m", "num_q_tiles", "total_m_tiles", "tiles_per_split"], "cluster_dims": [1, 1, 1], "computed_smem_bytes": 108800, "constants": [["K_MAX_", 10], ["EXPOSE_COL_COHORTS", 0], ["FULL_M_TILES", 0]], "cta_group": 1, "threads": 640}'))
d256_partial_ir = _decode_capture(_json_loads('{"__ir__": "knn_search_mma_split_partial_v1", "arg_keys": ["queries", "database", "partial_distances", "partial_indices", "B", "Q", "M", "split_m", "num_q_tiles", "total_m_tiles", "tiles_per_split"], "cluster_dims": [1, 1, 1], "computed_smem_bytes": 143104, "constants": [["K_MAX_", 10], ["EXPOSE_COL_COHORTS", 1]], "cta_group": 1, "threads": 256}'))
d256_merge_ir = _decode_capture(_json_loads('{"__ir__": "knn_search_mma_split_merge_v1", "arg_keys": ["partial_distances", "partial_indices", "out_distances", "out_indices", "B", "Q", "K", "split_m", "num_q_tiles"], "cluster_dims": [1, 1, 1], "computed_smem_bytes": 0, "constants": [["K_MAX_", 10]], "cta_group": 1, "threads": 32}'))
k64_partial_ir = _decode_capture(_json_loads('{"__ir__": "knn_search_mma_split_partial_v1", "arg_keys": ["queries", "database", "partial_distances", "partial_indices", "B", "Q", "M", "split_m", "num_q_tiles", "total_m_tiles", "tiles_per_split"], "cluster_dims": [1, 1, 1], "computed_smem_bytes": 143104, "constants": [["K_MAX_", 64], ["EXPOSE_COL_COHORTS", 1]], "cta_group": 1, "threads": 256}'))
k64_merge_ir = _decode_capture(_json_loads('{"__ir__": "knn_search_mma_split_merge_v1", "arg_keys": ["partial_distances", "partial_indices", "out_distances", "out_indices", "B", "Q", "K", "split_m", "num_q_tiles"], "cluster_dims": [1, 1, 1], "computed_smem_bytes": 0, "constants": [["K_MAX_", 64]], "cta_group": 1, "threads": 32}'))
HIGHQ_QBUCKET_LABELS: tuple[str, ...] = parent.HIGHQ_QBUCKET_LABELS
Q4096_SPLIT4_LABELS: tuple[str, ...] = ('rag_q4096_m20000_d128_k10', 'rag_batch_q4096_m20000_d128_k10', 'dispatch_q4096_m16384_d128_k10', 'dispatch_q4096_m32768_d128_k10')
Q4096_LOWK_LABELS: tuple[str, ...] = q4096_lowk.Q4096_LOWK_LABELS
HIGHQ_MIDM_SPLIT4_LOWK_LABELS: tuple[str, ...] = (*HIGHQ_QBUCKET_LABELS, *Q4096_SPLIT4_LABELS, *Q4096_LOWK_LABELS)
HIGHQ_QBUCKET_SHAPES = parent.HIGHQ_QBUCKET_SHAPES
Q4096_SPLIT4_SHAPES = _decode_capture(_json_loads('[{"__dict_items__": [["label", "rag_q4096_m20000_d128_k10"], ["params", {"__dict_items__": [["B", 1], ["Q", 4096], ["M", 20000], ["D", 128], ["K", 10], ["dtype", "bfloat16"], ["seed", 3], ["self_search", false], ["check_correctness", true], ["benchmark", true], ["min_recall", 0.999]]}]]}, {"__dict_items__": [["label", "rag_batch_q4096_m20000_d128_k10"], ["params", {"__dict_items__": [["B", 1], ["Q", 4096], ["M", 20000], ["D", 128], ["K", 10], ["dtype", "bfloat16"], ["seed", 610110], ["self_search", false], ["min_recall", 0.999]]}]]}, {"__dict_items__": [["label", "dispatch_q4096_m16384_d128_k10"], ["params", {"__dict_items__": [["B", 1], ["Q", 4096], ["M", 16384], ["D", 128], ["K", 10], ["dtype", "bfloat16"], ["seed", 610210], ["self_search", false], ["min_recall", 0.999]]}]]}, {"__dict_items__": [["label", "dispatch_q4096_m32768_d128_k10"], ["params", {"__dict_items__": [["B", 1], ["Q", 4096], ["M", 32768], ["D", 128], ["K", 10], ["dtype", "bfloat16"], ["seed", 610211], ["self_search", false], ["min_recall", 0.999]]}]]}]'))
Q4096_LOWK_SHAPES = _decode_capture(_json_loads('[{"__dict_items__": [["label", "ksweep_q4096_m20000_d128_k1"], ["params", {"__dict_items__": [["B", 1], ["Q", 4096], ["M", 20000], ["D", 128], ["K", 1], ["dtype", "bfloat16"], ["seed", 610310], ["self_search", false], ["min_recall", 1.0]]}]]}, {"__dict_items__": [["label", "ksweep_q4096_m20000_d128_k2"], ["params", {"__dict_items__": [["B", 1], ["Q", 4096], ["M", 20000], ["D", 128], ["K", 2], ["dtype", "bfloat16"], ["seed", 610311], ["self_search", false], ["min_recall", 1.0]]}]]}]'))
HIGHQ_MIDM_SPLIT4_LOWK_SHAPES = _decode_capture(_json_loads('[{"__dict_items__": [["label", "dispatch_q256_m65536_d128_k10"], ["params", {"__dict_items__": [["B", 1], ["Q", 256], ["M", 65536], ["D", 128], ["K", 10], ["dtype", "bfloat16"], ["seed", 610206], ["self_search", false], ["min_recall", 0.999]]}]]}, {"__dict_items__": [["label", "dispatch_q512_m65536_d128_k10"], ["params", {"__dict_items__": [["B", 1], ["Q", 512], ["M", 65536], ["D", 128], ["K", 10], ["dtype", "bfloat16"], ["seed", 610207], ["self_search", false], ["min_recall", 0.999]]}]]}, {"__dict_items__": [["label", "dispatch_q1024_m65536_d128_k10"], ["params", {"__dict_items__": [["B", 1], ["Q", 1024], ["M", 65536], ["D", 128], ["K", 10], ["dtype", "bfloat16"], ["seed", 610208], ["self_search", false], ["min_recall", 0.999]]}]]}, {"__dict_items__": [["label", "dispatch_q2048_m65536_d128_k10"], ["params", {"__dict_items__": [["B", 1], ["Q", 2048], ["M", 65536], ["D", 128], ["K", 10], ["dtype", "bfloat16"], ["seed", 610209], ["self_search", false], ["min_recall", 0.999]]}]]}, {"__dict_items__": [["label", "rag_q4096_m20000_d128_k10"], ["params", {"__dict_items__": [["B", 1], ["Q", 4096], ["M", 20000], ["D", 128], ["K", 10], ["dtype", "bfloat16"], ["seed", 3], ["self_search", false], ["check_correctness", true], ["benchmark", true], ["min_recall", 0.999]]}]]}, {"__dict_items__": [["label", "rag_batch_q4096_m20000_d128_k10"], ["params", {"__dict_items__": [["B", 1], ["Q", 4096], ["M", 20000], ["D", 128], ["K", 10], ["dtype", "bfloat16"], ["seed", 610110], ["self_search", false], ["min_recall", 0.999]]}]]}, {"__dict_items__": [["label", "dispatch_q4096_m16384_d128_k10"], ["params", {"__dict_items__": [["B", 1], ["Q", 4096], ["M", 16384], ["D", 128], ["K", 10], ["dtype", "bfloat16"], ["seed", 610210], ["self_search", false], ["min_recall", 0.999]]}]]}, {"__dict_items__": [["label", "dispatch_q4096_m32768_d128_k10"], ["params", {"__dict_items__": [["B", 1], ["Q", 4096], ["M", 32768], ["D", 128], ["K", 10], ["dtype", "bfloat16"], ["seed", 610211], ["self_search", false], ["min_recall", 0.999]]}]]}, {"__dict_items__": [["label", "ksweep_q4096_m20000_d128_k1"], ["params", {"__dict_items__": [["B", 1], ["Q", 4096], ["M", 20000], ["D", 128], ["K", 1], ["dtype", "bfloat16"], ["seed", 610310], ["self_search", false], ["min_recall", 1.0]]}]]}, {"__dict_items__": [["label", "ksweep_q4096_m20000_d128_k2"], ["params", {"__dict_items__": [["B", 1], ["Q", 4096], ["M", 20000], ["D", 128], ["K", 2], ["dtype", "bfloat16"], ["seed", 610311], ["self_search", false], ["min_recall", 1.0]]}]]}]'))
ROUND43_DISPATCHER_SHAPES = [*HIGHQ_MIDM_SPLIT4_LOWK_SHAPES, *parent.ROUND40_DISPATCHER_SHAPES]
ROUND43_PRESERVE_SHAPES = [*HIGHQ_MIDM_SPLIT4_LOWK_SHAPES, *parent.ROUND40_PRESERVE_SHAPES]
SHAPE_DISPATCH_REGISTRY: tuple[dict[str, str], ...] = ({'shape_key': 'd128_highq_midm_qbucket_k10', 'guard': 'B == 1 and 256 <= Q <= 2048 and 16384 <= M <= 65536 and D == 128 and K <= 10 and tcgen05', 'route': 'round43_highq_midm_qbucket'}, {'shape_key': 'd128_q4096_m20000_lowk_split4_kmerge', 'guard': 'B == 1 and Q == 4096 and M == 20000 and D == 128 and K in {1,2} and tcgen05', 'route': 'round43_q4096_lowk_split4_kmerge'}, {'shape_key': 'd128_q4096_midm_lowk_split4', 'guard': 'B == 1 and Q == 4096 and M in {16384,20000,32768} and D == 128 and 1 <= K <= 10 and tcgen05', 'route': 'round43_q4096_split4_tiestable'}, *parent.SHAPE_DISPATCH_REGISTRY)

def _use_highq_qbucket(inputs: dict[str, Any]) -> bool:
    return highq_qbucket._use_highq_qbucket(inputs)

def _use_q4096_lowk(inputs: dict[str, Any]) -> bool:
    return q4096_lowk._use_q4096_lowk(inputs)

def _use_q4096_split4(inputs: dict[str, Any]) -> bool:
    return int(inputs['B']) == 1 and int(inputs['M']) in {16384, 20000, 32768} and q4096_split4._use_q4096_split4_tiestable(inputs)

def selected_route(inputs: dict[str, Any]) -> str:
    if _use_highq_qbucket(inputs):
        return 'round43_highq_midm_qbucket'
    if _use_q4096_lowk(inputs):
        return 'round43_q4096_lowk_split4_kmerge'
    if _use_q4096_split4(inputs):
        return 'round43_q4096_split4_tiestable'
    return parent.selected_route(inputs)

def launch_for_eval(inputs: dict[str, Any]) -> dict[str, Any]:
    if _use_highq_qbucket(inputs):
        return highq_qbucket.launch_for_eval(inputs)
    if _use_q4096_lowk(inputs):
        return q4096_lowk._launch_q4096_lowk(inputs)
    if _use_q4096_split4(inputs):
        return q4096_split4.launch_for_eval(inputs)
    return parent.launch_for_eval(inputs)

def _select_contract_shapes(shape_labels: str | tuple[str, ...] | list[str] | None):
    if shape_labels is None:
        return None
    return select_named_shapes(shape_labels)

def knn_search_compile_and_launch_highq_midm_split4_lowk(*, benchmark: bool=True, shapes: list[dict[str, Any]] | None=None) -> dict[str, Any]:
    from .._dispatch_runtime import evaluate
    result = evaluate(launch_for_eval, shapes=HIGHQ_MIDM_SPLIT4_LOWK_SHAPES if shapes is None else shapes, benchmark=benchmark)
    result['passed'] = bool(result.get('summary', {}).get('all_correct'))
    print(result)
    return result

def knn_search_compile_and_launch_round43_dispatch(*, benchmark: bool=True, shapes: list[dict[str, Any]] | None=None) -> dict[str, Any]:
    from .._dispatch_runtime import evaluate
    result = evaluate(launch_for_eval, shapes=ROUND43_DISPATCHER_SHAPES if shapes is None else shapes, benchmark=benchmark)
    result['passed'] = bool(result.get('summary', {}).get('all_correct'))
    print(result)
    return result

def knn_search_compile_and_launch_round43_preserve(*, benchmark: bool=True) -> dict[str, Any]:
    from .._dispatch_runtime import evaluate
    result = evaluate(launch_for_eval, shapes=ROUND43_PRESERVE_SHAPES, benchmark=benchmark)
    result['passed'] = bool(result.get('summary', {}).get('all_correct'))
    print(result)
    return result
