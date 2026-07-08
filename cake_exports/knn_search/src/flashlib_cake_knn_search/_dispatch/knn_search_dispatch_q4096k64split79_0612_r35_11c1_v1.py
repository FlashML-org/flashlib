"""Round-35 table-driven dispatcher for registered exact BF16 kNN routes.

Minimum target architecture: sm_100a for the registered Q4096/K64 tcgen05
route. This dispatcher consumes the round-34 exact
``B=1,Q=4096,M=20000,D=128,K=64`` split79 two-tile producer and keeps the
existing source-clean parent dispatcher for all other shapes. It is intentionally
an additive policy wrapper: no kernel body or fallback semantics are changed.
"""
from __future__ import annotations
from json import loads as _json_loads
from .._dispatch_runtime import _capture_cuTensorMapEncodeTiled, _decode_capture, _import_dispatch_module, _ir_proxy
from collections.abc import Callable
from typing import Any, NamedTuple
from . import knn_search_k64_q4096split79_twotile_oddevensort_fastmerge_0612_r34_11c1_v1 as q4096_k64_split79
partial_ir = _decode_capture(_json_loads('{"__ir__": "knn_search_k64_q4096split79_twotile_oddevensort_partial_0612_r34_11c1_v1", "arg_keys": ["queries", "database", "partial_distances", "partial_indices", "B", "Q", "M", "split_m", "num_q_tiles", "total_m_tiles", "tiles_per_split"], "cluster_dims": [1, 1, 1], "computed_smem_bytes": 165120, "constants": [["K_MAX_", 64]], "cta_group": 1, "threads": 512}'))
merge_ir = _decode_capture(_json_loads('{"__ir__": "knn_search_k64_q4096split79_indexfastmerge10_guarded_0612_r34_11c1_v1", "arg_keys": ["partial_distances", "partial_indices", "out_distances", "out_indices", "B", "Q", "K", "split_m", "num_q_tiles"], "cluster_dims": [1, 1, 1], "computed_smem_bytes": 0, "constants": [["K_MAX_", 64]], "cta_group": 1, "threads": 32}'))
ir = _decode_capture(_json_loads('{"__ir__": "knn_search_k64_q4096split79_twotile_oddevensort_partial_0612_r34_11c1_v1", "arg_keys": ["queries", "database", "partial_distances", "partial_indices", "B", "Q", "M", "split_m", "num_q_tiles", "total_m_tiles", "tiles_per_split"], "cluster_dims": [1, 1, 1], "computed_smem_bytes": 165120, "constants": [["K_MAX_", 64]], "cta_group": 1, "threads": 512}'))

class DispatchRoute(NamedTuple):
    name: str
    shape_key: str
    guard: str
    predicate: Callable[[dict[str, Any]], bool]
    launcher: Callable[[dict[str, Any]], dict[str, Any]]
    source_round_doc: str

def _use_q4096_k64_split79(inputs: dict[str, Any]) -> bool:
    return q4096_k64_split79._use_q4096_k64_split79_oddevensort_fastmerge(inputs)

def _launch_q4096_k64_split79(inputs: dict[str, Any]) -> dict[str, Any]:
    return q4096_k64_split79._launch_q4096_k64_split79_oddevensort_fastmerge(inputs)
DISPATCH_ROUTES: tuple[DispatchRoute, ...] = (DispatchRoute(name='q4096_m20000_d128_k64_split79_oddeven_fastmerge', shape_key='B1_Q4096_M20000_D128_K64', guard='B == 1 and Q == 4096 and M == 20000 and D == 128 and K == 64 and tcgen05_capable_arch', predicate=_use_q4096_k64_split79, launcher=_launch_q4096_k64_split79, source_round_doc='design_doc/active/weave_evolve_knn_search_round_34_11c1_q4096k64split79.md'),)
DISPATCH_REGISTRY = _decode_capture(_json_loads('{"__tuple__": [{"__dict_items__": [["name", "q4096_m20000_d128_k64_split79_oddeven_fastmerge"], ["shape_key", "B1_Q4096_M20000_D128_K64"], ["shape_guard", "B == 1 and Q == 4096 and M == 20000 and D == 128 and K == 64 and tcgen05_capable_arch"], ["kernel_ref", "loom.examples.weave.knn_search_k64_q4096split79_twotile_oddevensort_fastmerge_0612_r34_11c1_v1:launch_for_eval"], ["source_round_doc", "design_doc/active/weave_evolve_knn_search_round_34_11c1_q4096k64split79.md"]]}]}'))
DISPATCHER_R35_SHAPES: list[dict[str, Any]] = [{'label': 'rag_q128_m131072_d128_k10', 'params': {'B': 1, 'Q': 128, 'M': 131072, 'D': 128, 'K': 10, 'dtype': 'bfloat16', 'seed': 2, 'self_search': False, 'min_recall': 0.999}}, {'label': 'rag_q4096_m20000_d128_k10', 'params': {'B': 1, 'Q': 4096, 'M': 20000, 'D': 128, 'K': 10, 'dtype': 'bfloat16', 'seed': 3, 'self_search': False, 'min_recall': 0.999}}, *q4096_k64_split79.K64_Q4096_SPLIT79_ODDEVENSORT_FASTMERGE_SHAPES]

def selected_route_name(inputs: dict[str, Any]) -> str:
    for route in DISPATCH_ROUTES:
        if route.predicate(inputs):
            return route.name
    return 'parent_dispatcher'

def launch_for_eval(inputs: dict[str, Any]) -> dict[str, Any]:
    for route in DISPATCH_ROUTES:
        if route.predicate(inputs):
            return route.launcher(inputs)
    return q4096_k64_split79.launch_for_eval(inputs)

def _select_contract_shapes(shape_labels: str | tuple[str, ...] | list[str] | None):
    if shape_labels is None:
        return None
    from .._dispatch_runtime import select_named_shapes
    return select_named_shapes(shape_labels)

def knn_search_compile_and_launch_mma_split(*, benchmark: bool=True, shape_labels: str | tuple[str, ...] | list[str] | None=None) -> dict[str, Any]:
    from .._dispatch_runtime import evaluate
    result = evaluate(launch_for_eval, shapes=_select_contract_shapes(shape_labels), benchmark=benchmark)
    result['passed'] = bool(result.get('summary', {}).get('all_correct'))
    print(result)
    return result

def knn_search_compile_and_launch_dispatch_r35(*, benchmark: bool=True, shape_labels: str | tuple[str, ...] | list[str] | None=None, shapes: list[dict[str, Any]] | None=None) -> dict[str, Any]:
    from .._dispatch_runtime import evaluate
    selected = shapes
    if selected is None:
        selected = DISPATCHER_R35_SHAPES if shape_labels is None else _select_contract_shapes(shape_labels)
    result = evaluate(launch_for_eval, shapes=selected, benchmark=benchmark)
    result['passed'] = bool(result.get('summary', {}).get('all_correct'))
    print(result)
    return result
