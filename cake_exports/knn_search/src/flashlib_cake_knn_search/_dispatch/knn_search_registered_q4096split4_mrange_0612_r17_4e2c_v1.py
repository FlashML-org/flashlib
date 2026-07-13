"""Round-17 registered kNN dispatcher with Q4096 split-4 M-range routing.

Minimum target architecture: sm_100a for the inherited tcgen05 MMA paths. This
wrapper preserves the current registered dispatcher for all non-Q4096 split-4
M-range shapes, and routes ``Q=4096,D=128,K<=10,16384<=M<=65536`` through the
round-14 split-4 tcgen05 partial producer and tie-stable stream merge.
"""
from __future__ import annotations
from json import loads as _json_loads
from .._dispatch_runtime import _capture_cuTensorMapEncodeTiled, _decode_capture, _import_dispatch_module, _ir_proxy
from typing import Any
from . import knn_search_mma_split_v1 as mma
from . import knn_search_q4096_split4_0611_r14_4e2c_v1 as q4096_split4
from . import knn_search_registered_q4096split4_tiestable_0612_r16_4e2c_v1 as parent
ir = _decode_capture(_json_loads('{"__ir__": "knn_search_mma_split_partial_v1", "arg_keys": ["queries", "database", "partial_distances", "partial_indices", "B", "Q", "M", "split_m", "num_q_tiles", "total_m_tiles", "tiles_per_split"], "cluster_dims": [1, 1, 1], "computed_smem_bytes": 165120, "constants": [["K_MAX_", 20], ["EXPOSE_COL_COHORTS", 0]], "cta_group": 1, "threads": 512}'))
Q4096_ROWS = 4096
Q4096_SPLIT4_M_MIN = 16384
Q4096_SPLIT4_M_MAX = 65536
Q4096_SPLIT4_MRANGE_SHAPES: list[dict[str, Any]] = [{'label': 'dispatch_q4096_m16384_d128_k10', 'params': {'B': 1, 'Q': 4096, 'M': 16384, 'D': 128, 'K': 10, 'dtype': 'bfloat16', 'seed': 610210, 'self_search': False, 'min_recall': 0.999}}, {'label': 'rag_q4096_m20000_d128_k10', 'params': {'B': 1, 'Q': 4096, 'M': 20000, 'D': 128, 'K': 10, 'dtype': 'bfloat16', 'seed': 3, 'self_search': False, 'min_recall': 0.999}}, {'label': 'heldout_q4096_m24576_d128_k10', 'params': {'B': 1, 'Q': 4096, 'M': 24576, 'D': 128, 'K': 10, 'dtype': 'bfloat16', 'seed': 610214, 'self_search': False, 'min_recall': 0.999}}, {'label': 'dispatch_q4096_m32768_d128_k10', 'params': {'B': 1, 'Q': 4096, 'M': 32768, 'D': 128, 'K': 10, 'dtype': 'bfloat16', 'seed': 610211, 'self_search': False, 'min_recall': 0.999}}, {'label': 'heldout_q4096_m40960_d128_k10', 'params': {'B': 1, 'Q': 4096, 'M': 40960, 'D': 128, 'K': 10, 'dtype': 'bfloat16', 'seed': 610215, 'self_search': False, 'min_recall': 0.999}}, {'label': 'sweep_q4096_m49152_d128_k10', 'params': {'B': 1, 'Q': 4096, 'M': 49152, 'D': 128, 'K': 10, 'dtype': 'bfloat16', 'seed': 610212, 'self_search': False, 'min_recall': 0.999}}, {'label': 'heldout_q4096_m57344_d128_k10', 'params': {'B': 1, 'Q': 4096, 'M': 57344, 'D': 128, 'K': 10, 'dtype': 'bfloat16', 'seed': 610216, 'self_search': False, 'min_recall': 0.999}}, {'label': 'dispatch_q4096_m65536_d128_k10', 'params': {'B': 1, 'Q': 4096, 'M': 65536, 'D': 128, 'K': 10, 'dtype': 'bfloat16', 'seed': 610213, 'self_search': False, 'min_recall': 0.999}}]

def _use_q4096_split4_mrange(inputs: dict[str, Any]) -> bool:
    m_rows = int(inputs['M'])
    return int(inputs['Q']) == Q4096_ROWS and Q4096_SPLIT4_M_MIN <= m_rows <= Q4096_SPLIT4_M_MAX and (int(inputs['D']) == mma.D_STATIC) and (int(inputs['K']) <= mma.K_MAX) and mma._tcgen05_capable_arch()

def launch_for_eval(inputs: dict[str, Any]) -> dict[str, Any]:
    if _use_q4096_split4_mrange(inputs):
        return q4096_split4._launch_q4096_split4(inputs)
    return parent.launch_for_eval(inputs)

def _select_contract_shapes(shape_labels: str | tuple[str, ...] | list[str] | None):
    return parent._select_contract_shapes(shape_labels)

def knn_search_compile_and_launch_mma_split(*, benchmark: bool=True, shape_labels: str | tuple[str, ...] | list[str] | None=None) -> dict[str, Any]:
    from .._dispatch_runtime import evaluate
    result = evaluate(launch_for_eval, shapes=_select_contract_shapes(shape_labels), benchmark=benchmark)
    result['passed'] = bool(result.get('summary', {}).get('all_correct'))
    print(result)
    return result
