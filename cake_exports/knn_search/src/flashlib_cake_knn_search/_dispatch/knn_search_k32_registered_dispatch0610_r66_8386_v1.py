"""Round-66 registered K31/K32 guarded capacity route for exact BF16 kNN.

Minimum target architecture: sm_100a for the K31/K32 tcgen05 producer path.
This wrapper preserves the round-19 registered dispatcher for all existing
routes, and routes only ``B=1,Q=128,M>=131072,D=128,K=31..32`` through the
round-60 K32 producer-side cohort merge plus guarded Q128/split-148 final
merge.
"""
from __future__ import annotations
from json import loads as _json_loads
from .._dispatch_runtime import _capture_cuTensorMapEncodeTiled, _decode_capture, _import_dispatch_module, _ir_proxy
from typing import Any
from . import knn_search_k32_guardedmerge_dispatch0610_r60_v1 as k32_guarded
from . import knn_search_k64_q128register_0612_r19_4e2c_v1 as parent
ir = _decode_capture(_json_loads('{"__ir__": "knn_search_mma_split_partial_v1", "arg_keys": ["queries", "database", "partial_distances", "partial_indices", "B", "Q", "M", "split_m", "num_q_tiles", "total_m_tiles", "tiles_per_split"], "cluster_dims": [1, 1, 1], "computed_smem_bytes": 165120, "constants": [["K_MAX_", 32], ["EXPOSE_COL_COHORTS", 0]], "cta_group": 1, "threads": 512}'))
K32_REGISTERED_SHAPES: list[dict[str, Any]] = [{'label': 'ksweep_q128_m131072_d128_k31', 'params': {'B': 1, 'Q': 128, 'M': 131072, 'D': 128, 'K': 31, 'dtype': 'bfloat16', 'seed': 610314, 'self_search': False, 'min_recall': 0.999}}, {'label': 'ksweep_q128_m131072_d128_k32', 'params': {'B': 1, 'Q': 128, 'M': 131072, 'D': 128, 'K': 32, 'dtype': 'bfloat16', 'seed': 610315, 'self_search': False, 'min_recall': 0.999}}]

def _use_registered_k31_k32_guarded(inputs: dict[str, Any]) -> bool:
    return k32_guarded._use_k32_guarded_mma(inputs)

def launch_for_eval(inputs: dict[str, Any]) -> dict[str, Any]:
    if _use_registered_k31_k32_guarded(inputs):
        return k32_guarded._launch_k32_guarded_mma(inputs)
    return parent.launch_for_eval(inputs)

def _select_contract_shapes(shape_labels: str | tuple[str, ...] | list[str] | None):
    return parent._select_contract_shapes(shape_labels)

def knn_search_compile_and_launch_mma_split(*, benchmark: bool=True, shape_labels: str | tuple[str, ...] | list[str] | None=None) -> dict[str, Any]:
    from .._dispatch_runtime import evaluate
    result = evaluate(launch_for_eval, shapes=_select_contract_shapes(shape_labels), benchmark=benchmark)
    result['passed'] = bool(result.get('summary', {}).get('all_correct'))
    print(result)
    return result
