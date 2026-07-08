"""Round-15 Q4096 split-4 dispatch with tie-stable stream merge.

Minimum target architecture: sm_100a for the inherited tcgen05 MMA path. This
wrapper preserves the round-14 split-4 bucket dispatcher for all other shapes
and adds the repaired ``M=49152`` bucket after making the stream merge consume
only one equal-distance split head per output rank.
"""
from __future__ import annotations
from json import loads as _json_loads
from .._dispatch_runtime import _capture_cuTensorMapEncodeTiled, _decode_capture, _import_dispatch_module, _ir_proxy
from typing import Any
from . import knn_search_mma_split_v1 as mma
from . import knn_search_q4096_split4_0611_r14_4e2c_v1 as parent
Q4096_ROWS = parent.Q4096_ROWS
Q4096_SPLIT_M = parent.Q4096_SPLIT_M
Q4096_SPLIT4_M_ROWS = frozenset({*parent.Q4096_SPLIT4_M_ROWS, 49152})
ir = _decode_capture(_json_loads('{"__ir__": "knn_search_mma_split_partial_v1", "arg_keys": ["queries", "database", "partial_distances", "partial_indices", "B", "Q", "M", "split_m", "num_q_tiles", "total_m_tiles", "tiles_per_split"], "cluster_dims": [1, 1, 1], "computed_smem_bytes": 108800, "constants": [["K_MAX_", 10], ["EXPOSE_COL_COHORTS", 0], ["FULL_M_TILES", 0]], "cta_group": 1, "threads": 640}'))
Q4096_SPLIT4_TIESTABLE_SHAPES: list[dict[str, Any]] = [*parent.Q4096_SPLIT4_SHAPES, {'label': 'sweep_q4096_m49152_d128_k10', 'params': {'B': 1, 'Q': 4096, 'M': 49152, 'D': 128, 'K': 10, 'dtype': 'bfloat16', 'seed': 610212, 'self_search': False, 'min_recall': 0.999}}]

def _use_q4096_split4_tiestable(inputs: dict[str, Any]) -> bool:
    return int(inputs['Q']) == Q4096_ROWS and int(inputs['M']) in Q4096_SPLIT4_M_ROWS and (int(inputs['D']) == mma.D_STATIC) and (int(inputs['K']) <= mma.K_MAX) and mma._tcgen05_capable_arch()

def launch_for_eval(inputs: dict[str, Any]) -> dict[str, Any]:
    if _use_q4096_split4_tiestable(inputs):
        return parent._launch_q4096_split4(inputs)
    return parent.launch_for_eval(inputs)

def _select_contract_shapes(shape_labels: str | tuple[str, ...] | list[str] | None):
    return parent._select_contract_shapes(shape_labels)

def knn_search_compile_and_launch_q4096_split4_tiestable(*, benchmark: bool=True, shape_labels: str | tuple[str, ...] | list[str] | None=None) -> dict[str, Any]:
    from .._dispatch_runtime import evaluate
    result = evaluate(launch_for_eval, shapes=_select_contract_shapes(shape_labels), benchmark=benchmark)
    result['passed'] = bool(result.get('summary', {}).get('all_correct'))
    print(result)
    return result
