"""Round-16 Q128 small/mid-M split dispatch for BF16 squared-L2 kNN search.

Minimum target architecture: sm_100a for the inherited MMA paths. This wrapper
preserves the round-15 K12/K20/K30 extended-K dispatcher and only routes
``Q=128, 8192 <= M < 131072, D=128, K<=10`` through the source-policy-clean
Q128 split-count policy, avoiding the inherited single-CTA producer for those
contract-suite M-bucket shapes.
"""
from __future__ import annotations
from json import loads as _json_loads
from .._dispatch_runtime import _capture_cuTensorMapEncodeTiled, _decode_capture, _import_dispatch_module, _ir_proxy
from typing import Any
from . import knn_search_k20k30_mma_capacity_0611_r15_4e96_v1 as parent
from . import knn_search_q128_split_policy_f295_v1 as q128_policy
Q128_MIDM_MIN = 8192
Q128_MIDM_EXCLUSIVE_MAX = q128_policy.Q128_LARGE_M_THRESHOLD
ir = _decode_capture(_json_loads('{"__ir__": "knn_search_mma_split_partial_v1", "arg_keys": ["queries", "database", "partial_distances", "partial_indices", "B", "Q", "M", "split_m", "num_q_tiles", "total_m_tiles", "tiles_per_split"], "cluster_dims": [1, 1, 1], "computed_smem_bytes": 108800, "constants": [["K_MAX_", 10], ["EXPOSE_COL_COHORTS", 0], ["FULL_M_TILES", 0]], "cta_group": 1, "threads": 640}'))

def _use_q128_midm_split_policy(inputs: dict[str, Any]) -> bool:
    return int(inputs['Q']) == q128_policy.BLOCK_Q and Q128_MIDM_MIN <= int(inputs['M']) < Q128_MIDM_EXCLUSIVE_MAX and (int(inputs['D']) == q128_policy.D_STATIC) and (int(inputs['K']) <= q128_policy.K_MAX) and q128_policy._incumbent._tcgen05_capable_arch()

def launch_for_eval(inputs: dict[str, Any]) -> dict[str, Any]:
    if _use_q128_midm_split_policy(inputs):
        return q128_policy.launch_for_eval(inputs)
    return parent.launch_for_eval(inputs)

def _select_contract_shapes(shape_labels: str | tuple[str, ...] | list[str] | None):
    if shape_labels is None:
        return None
    from .._dispatch_runtime import load_contract
    from .._dispatch_runtime import CANONICAL_SHAPES
    labels = [shape_labels] if isinstance(shape_labels, str) else list(shape_labels)
    contract = load_contract('knn_search')
    shapes: dict[str, dict[str, Any]] = {str(shape['label']): shape for shape in CANONICAL_SHAPES}
    for suite in contract.raw.get('benchmark_shape_suites', {}).get('suites', []):
        for raw_shape in suite.get('shapes', []):
            label = str(raw_shape['label'])
            shapes[label] = {'label': label, 'params': {key: value for key, value in raw_shape.items() if key != 'label'}}
    missing = [label for label in labels if label not in shapes]
    if missing:
        raise ValueError(''.join(['unknown knn_search shape label(s): ', format(missing, '')]))
    return [shapes[label] for label in labels]

def knn_search_compile_and_launch_mma_split(*, benchmark: bool=True, shape_labels: str | tuple[str, ...] | list[str] | None=None) -> dict[str, Any]:
    from .._dispatch_runtime import evaluate
    result = evaluate(launch_for_eval, shapes=_select_contract_shapes(shape_labels), benchmark=benchmark)
    result['passed'] = bool(result.get('summary', {}).get('all_correct'))
    print(result)
    return result
