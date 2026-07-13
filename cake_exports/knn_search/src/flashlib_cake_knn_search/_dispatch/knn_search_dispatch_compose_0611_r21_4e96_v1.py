"""Round-21 composed dispatch wrapper for exact BF16 squared-L2 kNN.

Minimum target architecture: sm_80 for the Q1 tail route and sm_100a for the
inherited tcgen05 MMA routes. This additive candidate preserves the round-20
K13-K30 Q128 guarded bucket path, while composing in the existing source-clean
Q1 large-M tail route and high-Q mid-M Q-bucket split policy.
"""
from __future__ import annotations
from json import loads as _json_loads
from .._dispatch_runtime import _capture_cuTensorMapEncodeTiled, _decode_capture, _import_dispatch_module, _ir_proxy
from typing import Any
from . import knn_search_highq_midm_qbucket_0611_r19_4e96_v1 as highq_qbucket
from . import knn_search_k20k30_mma_q128register_0611_r20_4e96_v1 as parent
from . import knn_search_q1_irregular_m_tail_v1 as q1_tail
K_MAX = q1_tail.K_MAX
D_STATIC = q1_tail.D_STATIC
ir = _decode_capture(_json_loads('{"__ir__": "knn_search_mma_split_partial_v1", "arg_keys": ["queries", "database", "partial_distances", "partial_indices", "B", "Q", "M", "split_m", "num_q_tiles", "total_m_tiles", "tiles_per_split"], "cluster_dims": [1, 1, 1], "computed_smem_bytes": 165120, "constants": [["K_MAX_", 20], ["EXPOSE_COL_COHORTS", 0]], "cta_group": 1, "threads": 512}'))
Q1_TAIL_SHAPES: list[dict[str, Any]] = [{'label': 'rag_online_q1_m100000_d128_k10', 'params': {'B': 1, 'Q': 1, 'M': 100000, 'D': 128, 'K': 10, 'dtype': 'bfloat16', 'seed': 610101, 'self_search': False, 'min_recall': 1.0}}, {'label': 'rag_online_q1_m131072_d128_k10', 'params': {'B': 1, 'Q': 1, 'M': 131072, 'D': 128, 'K': 10, 'dtype': 'bfloat16', 'seed': 610102, 'self_search': False, 'min_recall': 1.0}}]
HIGHQ_MIDM_SHAPES: list[dict[str, Any]] = [{'label': 'dispatch_q256_m65536_d128_k10', 'params': {'B': 1, 'Q': 256, 'M': 65536, 'D': 128, 'K': 10, 'dtype': 'bfloat16', 'seed': 610206, 'self_search': False, 'min_recall': 0.999}}, {'label': 'dispatch_q512_m65536_d128_k10', 'params': {'B': 1, 'Q': 512, 'M': 65536, 'D': 128, 'K': 10, 'dtype': 'bfloat16', 'seed': 610207, 'self_search': False, 'min_recall': 0.999}}, {'label': 'dispatch_q1024_m65536_d128_k10', 'params': {'B': 1, 'Q': 1024, 'M': 65536, 'D': 128, 'K': 10, 'dtype': 'bfloat16', 'seed': 610208, 'self_search': False, 'min_recall': 0.999}}, {'label': 'dispatch_q2048_m65536_d128_k10', 'params': {'B': 1, 'Q': 2048, 'M': 65536, 'D': 128, 'K': 10, 'dtype': 'bfloat16', 'seed': 610209, 'self_search': False, 'min_recall': 0.999}}, {'label': 'dispatch_q4096_m16384_d128_k10', 'params': {'B': 1, 'Q': 4096, 'M': 16384, 'D': 128, 'K': 10, 'dtype': 'bfloat16', 'seed': 610210, 'self_search': False, 'min_recall': 0.999}}, {'label': 'dispatch_q4096_m32768_d128_k10', 'params': {'B': 1, 'Q': 4096, 'M': 32768, 'D': 128, 'K': 10, 'dtype': 'bfloat16', 'seed': 610211, 'self_search': False, 'min_recall': 0.999}}]

def _use_q1_tail_kernel(inputs: dict[str, Any]) -> bool:
    return int(inputs['Q']) == 1 and int(inputs['M']) >= 65536 and (int(inputs['D']) == D_STATIC) and (int(inputs['K']) <= K_MAX)

def launch_for_eval(inputs: dict[str, Any]) -> dict[str, Any]:
    if _use_q1_tail_kernel(inputs):
        return q1_tail.launch_for_eval(inputs)
    if highq_qbucket._use_highq_midm_qbucket_policy(inputs):
        return highq_qbucket._launch_highq_midm_qbucket_mma(inputs)
    return parent.launch_for_eval(inputs)

def _select_contract_shapes(shape_labels: str | tuple[str, ...] | list[str] | None):
    return parent._select_contract_shapes(shape_labels)

def knn_search_compile_and_launch_dispatch_compose(*, benchmark: bool=True, shapes: list[dict[str, Any]] | None=None) -> dict[str, Any]:
    from .._dispatch_runtime import evaluate
    result = evaluate(launch_for_eval, shapes=shapes, benchmark=benchmark)
    result['passed'] = bool(result.get('summary', {}).get('all_correct'))
    print(result)
    return result
