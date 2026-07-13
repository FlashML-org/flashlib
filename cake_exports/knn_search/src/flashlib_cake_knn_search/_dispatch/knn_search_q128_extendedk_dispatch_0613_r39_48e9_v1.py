"""Round-39 Q128 extended-K dispatcher over the r38 kNN routes.

Minimum target architecture: sm_100a for the Q128/D128 tcgen05 capacity
routes. This shape-kernel continuation preserves the round-38 K<=10 dispatcher
and adds contract-visible Q128/D128 K11/K12/K20/K30/K32/K64 capacity routes
using existing source-clean Weave producers and merges.
"""
from __future__ import annotations
from json import loads as _json_loads
from .._dispatch_runtime import _capture_cuTensorMapEncodeTiled, _decode_capture, _import_dispatch_module, _ir_proxy
from typing import Any
from .._dispatch_runtime import select_named_shapes
from . import knn_search_d128q128_lowk_dispatch_0613_r38_48e9_v1 as parent
from . import knn_search_k12_mma_capacity_0611_r14_4e96_v1 as k12_route
from . import knn_search_k20k30_mma_q128register_0611_r20_4e96_v1 as k20k30_route
from . import knn_search_k31k32_staticmerge_dispatch0610_r68_8386_v1 as k31k32_route
from . import knn_search_k64_q128register_0612_r19_4e2c_v1 as k64_route
THREADS = k20k30_route.THREADS
BLOCK_Q = parent.BLOCK_Q
BLOCK_M = parent.BLOCK_M
D_STATIC = parent.D_STATIC
K10_MAX = parent.K_MAX
K12_MAX = k12_route.K12_MAX
K20_MAX = k20k30_route.K20_MAX
K30_MAX = k20k30_route.K30_MAX
K32_MAX = k31k32_route.K32_MAX
K64_MAX = k64_route.K64_MAX
ir = _decode_capture(_json_loads('{"__ir__": "knn_search_mma_split_partial_v1", "arg_keys": ["queries", "database", "partial_distances", "partial_indices", "B", "Q", "M", "split_m", "num_q_tiles", "total_m_tiles", "tiles_per_split"], "cluster_dims": [1, 1, 1], "computed_smem_bytes": 165120, "constants": [["K_MAX_", 20], ["EXPOSE_COL_COHORTS", 0]], "cta_group": 1, "threads": 512}'))
k12_ir = _decode_capture(_json_loads('{"__ir__": "knn_search_mma_split_partial_v1", "arg_keys": ["queries", "database", "partial_distances", "partial_indices", "B", "Q", "M", "split_m", "num_q_tiles", "total_m_tiles", "tiles_per_split"], "cluster_dims": [1, 1, 1], "computed_smem_bytes": 165120, "constants": [["K_MAX_", 12], ["EXPOSE_COL_COHORTS", 1]], "cta_group": 1, "threads": 512}'))
k20k30_ir = _decode_capture(_json_loads('{"__ir__": "knn_search_mma_split_partial_v1", "arg_keys": ["queries", "database", "partial_distances", "partial_indices", "B", "Q", "M", "split_m", "num_q_tiles", "total_m_tiles", "tiles_per_split"], "cluster_dims": [1, 1, 1], "computed_smem_bytes": 165120, "constants": [["K_MAX_", 20], ["EXPOSE_COL_COHORTS", 0]], "cta_group": 1, "threads": 512}'))
k31k32_ir = _decode_capture(_json_loads('{"__ir__": "knn_search_mma_split_partial_v1", "arg_keys": ["queries", "database", "partial_distances", "partial_indices", "B", "Q", "M", "split_m", "num_q_tiles", "total_m_tiles", "tiles_per_split"], "cluster_dims": [1, 1, 1], "computed_smem_bytes": 165120, "constants": [["K_MAX_", 32], ["EXPOSE_COL_COHORTS", 0]], "cta_group": 1, "threads": 512}'))
k64_ir = _decode_capture(_json_loads('{"__ir__": "knn_search_mma_split_partial_v1", "arg_keys": ["queries", "database", "partial_distances", "partial_indices", "B", "Q", "M", "split_m", "num_q_tiles", "total_m_tiles", "tiles_per_split"], "cluster_dims": [1, 1, 1], "computed_smem_bytes": 165120, "constants": [["K_MAX_", 64], ["EXPOSE_COL_COHORTS", 1]], "cta_group": 1, "threads": 512}'))
parent_ir = _decode_capture(_json_loads('{"__ir__": "knn_search_mma_split_partial_v1", "arg_keys": ["queries", "database", "partial_distances", "partial_indices", "B", "Q", "M", "split_m", "num_q_tiles", "total_m_tiles", "tiles_per_split"], "cluster_dims": [1, 1, 1], "computed_smem_bytes": 108800, "constants": [["K_MAX_", 10], ["EXPOSE_COL_COHORTS", 0], ["FULL_M_TILES", 0]], "cta_group": 1, "threads": 640}'))
Q128_EXTENDEDK_CONTRACT_LABELS: tuple[str, ...] = ('ksweep_q128_m131072_d128_k11', 'ksweep_q128_m131072_d128_k12', 'ksweep_q128_m131072_d128_k20', 'ksweep_q128_m131072_d128_k30', 'ksweep_q128_m131072_d128_k64')
Q128_EXTENDEDK_K32_SHAPE: dict[str, Any] = {'label': 'ksweep_q128_m131072_d128_k32', 'params': {'B': 1, 'Q': 128, 'M': 131072, 'D': 128, 'K': 32, 'dtype': 'bfloat16', 'seed': 610315, 'self_search': False, 'min_recall': 0.999}}
Q128_EXTENDEDK_DISPATCH_SHAPES = _decode_capture(_json_loads('[{"__dict_items__": [["label", "ksweep_q128_m131072_d128_k11"], ["params", {"__dict_items__": [["B", 1], ["Q", 128], ["M", 131072], ["D", 128], ["K", 11], ["dtype", "bfloat16"], ["seed", 610306], ["self_search", false], ["min_recall", 0.999]]}]]}, {"__dict_items__": [["label", "ksweep_q128_m131072_d128_k12"], ["params", {"__dict_items__": [["B", 1], ["Q", 128], ["M", 131072], ["D", 128], ["K", 12], ["dtype", "bfloat16"], ["seed", 610307], ["self_search", false], ["min_recall", 0.999]]}]]}, {"__dict_items__": [["label", "ksweep_q128_m131072_d128_k20"], ["params", {"__dict_items__": [["B", 1], ["Q", 128], ["M", 131072], ["D", 128], ["K", 20], ["dtype", "bfloat16"], ["seed", 610308], ["self_search", false], ["min_recall", 0.999]]}]]}, {"__dict_items__": [["label", "ksweep_q128_m131072_d128_k30"], ["params", {"__dict_items__": [["B", 1], ["Q", 128], ["M", 131072], ["D", 128], ["K", 30], ["dtype", "bfloat16"], ["seed", 610309], ["self_search", false], ["min_recall", 0.999]]}]]}, {"__dict_items__": [["label", "ksweep_q128_m131072_d128_k64"], ["params", {"__dict_items__": [["B", 1], ["Q", 128], ["M", 131072], ["D", 128], ["K", 64], ["dtype", "bfloat16"], ["seed", 610312], ["self_search", false], ["min_recall", 0.999]]}]]}, {"__dict_items__": [["label", "ksweep_q128_m131072_d128_k32"], ["params", {"__dict_items__": [["B", 1], ["Q", 128], ["M", 131072], ["D", 128], ["K", 32], ["dtype", "bfloat16"], ["seed", 610315], ["self_search", false], ["min_recall", 0.999]]}]]}]'))
ROUND39_PRESERVE_K10_SHAPES = parent.ROUND38_DISPATCHER_SHAPES
ROUND39_FULL_SHAPES = [*ROUND39_PRESERVE_K10_SHAPES, *Q128_EXTENDEDK_DISPATCH_SHAPES]
SHAPE_DISPATCH_REGISTRY: tuple[dict[str, str], ...] = ({'shape_key': 'd128_q128_k11_k12_capacity', 'guard': 'B == 1 and Q == 128 and M >= 131072 and D == 128 and 11 <= K <= 12 and tcgen05', 'route': 'round14_k12_tcgen05_capacity'}, {'shape_key': 'd128_q128_k13_k30_capacity', 'guard': 'B == 1 and Q == 128 and M >= 131072 and D == 128 and 13 <= K <= 30 and tcgen05', 'route': 'round20_k20_k30_tcgen05_capacity'}, {'shape_key': 'd128_q128_k32_capacity', 'guard': 'B == 1 and Q == 128 and M >= 131072 and D == 128 and K == 32 and tcgen05', 'route': 'round68_k32_static_merge_tcgen05_capacity'}, {'shape_key': 'd128_q128_k64_capacity', 'guard': 'B == 1 and Q == 128 and M >= 131072 and D == 128 and K == 64 and tcgen05', 'route': 'round19_k64_tcgen05_capacity'}, *parent.SHAPE_DISPATCH_REGISTRY)

def _use_k12_capacity(inputs: dict[str, Any]) -> bool:
    return bool(k12_route._use_k12_mma(inputs))

def _use_k20k30_capacity(inputs: dict[str, Any]) -> bool:
    return bool(k20k30_route._use_bucket_mma(inputs))

def _use_k32_capacity(inputs: dict[str, Any]) -> bool:
    return bool(k31k32_route._use_staticmerge_mma(inputs)) and int(inputs['K']) == K32_MAX

def _use_k64_capacity(inputs: dict[str, Any]) -> bool:
    return bool(k64_route._use_q128_k64(inputs))

def selected_route(inputs: dict[str, Any]) -> str:
    if _use_k12_capacity(inputs):
        return 'round14_k12_tcgen05_capacity'
    if _use_k20k30_capacity(inputs):
        return 'round20_k20_k30_tcgen05_capacity'
    if _use_k32_capacity(inputs):
        return 'round68_k32_static_merge_tcgen05_capacity'
    if _use_k64_capacity(inputs):
        return 'round19_k64_tcgen05_capacity'
    return parent.selected_route(inputs)

def launch_for_eval(inputs: dict[str, Any]) -> dict[str, Any]:
    route = selected_route(inputs)
    if route == 'round14_k12_tcgen05_capacity':
        return k12_route.launch_for_eval(inputs)
    if route == 'round20_k20_k30_tcgen05_capacity':
        return k20k30_route.launch_for_eval(inputs)
    if route == 'round68_k32_static_merge_tcgen05_capacity':
        return k31k32_route.launch_for_eval(inputs)
    if route == 'round19_k64_tcgen05_capacity':
        return k64_route.launch_for_eval(inputs)
    return parent.launch_for_eval(inputs)

def _select_contract_shapes(shape_labels: str | tuple[str, ...] | list[str] | None):
    if shape_labels is None:
        return None
    return select_named_shapes(shape_labels)

def knn_search_compile_and_launch_extendedk_dispatch(*, benchmark: bool=True, shapes: list[dict[str, Any]] | None=None) -> dict[str, Any]:
    from .._dispatch_runtime import evaluate
    result = evaluate(launch_for_eval, shapes=Q128_EXTENDEDK_DISPATCH_SHAPES if shapes is None else shapes, benchmark=benchmark)
    result['passed'] = bool(result.get('summary', {}).get('all_correct'))
    print(result)
    return result

def knn_search_compile_and_launch_round39_full(*, benchmark: bool=True, shapes: list[dict[str, Any]] | None=None) -> dict[str, Any]:
    from .._dispatch_runtime import evaluate
    result = evaluate(launch_for_eval, shapes=ROUND39_FULL_SHAPES if shapes is None else shapes, benchmark=benchmark)
    result['passed'] = bool(result.get('summary', {}).get('all_correct'))
    print(result)
    return result
