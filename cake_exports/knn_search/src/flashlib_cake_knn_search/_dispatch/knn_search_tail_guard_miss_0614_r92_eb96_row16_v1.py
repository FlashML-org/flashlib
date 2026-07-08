"""Round-92 low-Q small-M/tail row16 seed for exact BF16 kNN.

Minimum target architecture: sm_100a for the tcgen05 row16 route. This
shape-kernel candidate routes only
``B=1,Q in {8,16,32},32768<=M<131072,D=128,K=10`` through the existing
source-policy-clean ROW_16x256B partial producer and const-148 merge. It is an
additive seed for the tail guard-miss bucket; large-M rows and all guard misses
delegate to the round-55 default Weave dispatcher unchanged.
"""
from __future__ import annotations
from json import loads as _json_loads
from .._dispatch_runtime import _capture_cuTensorMapEncodeTiled, _decode_capture, _import_dispatch_module, _ir_proxy
import math
from typing import Any
from . import knn_search_lowq_q8q16q32q64_row16_registered_0613_r55_48e9_v1 as parent
from . import knn_search_lowq_row16_mma_dispatch0610_r18_d14a_vec16_bguard_v1 as row16
from . import knn_search_mma_split_v1 as _mma
THREADS = row16.THREADS
MERGE_THREADS = _mma.MERGE_THREADS
BLOCK_Q = row16.BLOCK_Q
BLOCK_M = row16.BLOCK_M
D_STATIC = row16.D_STATIC
K_MAX = row16.K_MAX
SPLIT_M = row16.SPLIT_M
ir = _decode_capture(_json_loads('{"__ir__": "knn_search_lowq_row16_mma_partial_dispatch0610_r18_d14a_vec16_bguard_v1", "arg_keys": ["queries", "database", "partial_distances", "partial_indices", "B", "Q", "M", "split_m", "num_q_tiles", "total_m_tiles"], "cluster_dims": [1, 1, 1], "computed_smem_bytes": 96000, "constants": [["K_MAX_", 10]], "cta_group": 1, "threads": 512}'))
partial_ir = _decode_capture(_json_loads('{"__ir__": "knn_search_lowq_row16_mma_partial_dispatch0610_r18_d14a_vec16_bguard_v1", "arg_keys": ["queries", "database", "partial_distances", "partial_indices", "B", "Q", "M", "split_m", "num_q_tiles", "total_m_tiles"], "cluster_dims": [1, 1, 1], "computed_smem_bytes": 96000, "constants": [["K_MAX_", 10]], "cta_group": 1, "threads": 512}'))
merge_ir = _decode_capture(_json_loads('{"__ir__": "knn_search_mma_split_merge_q128_const148_v1", "arg_keys": ["partial_distances", "partial_indices", "out_distances", "out_indices", "B", "Q", "K", "split_m", "num_q_tiles"], "cluster_dims": [1, 1, 1], "computed_smem_bytes": 0, "constants": [["K_MAX_", 10]], "cta_group": 1, "threads": 32}'))
parent_ir = _decode_capture(_json_loads('{"__ir__": "knn_search_lowq_row16_mma_partial_dispatch0610_r18_d14a_vec16_bguard_v1", "arg_keys": ["queries", "database", "partial_distances", "partial_indices", "B", "Q", "M", "split_m", "num_q_tiles", "total_m_tiles"], "cluster_dims": [1, 1, 1], "computed_smem_bytes": 96000, "constants": [["K_MAX_", 10]], "cta_group": 1, "threads": 512}'))
ROUTE_TAIL_GUARD_MISS_ROW16 = 'round92_tail_guard_miss_q8q16q32_row16_tcgen05'
ROUTE_PARENT_DEFAULT = 'round92_parent_round55_r55_default'
ROUTE_LOWQ_ROW16 = parent.ROUTE_LOWQ_ROW16
ROUTE_PARENT_R55_FALLBACK = parent.ROUTE_PARENT_DEFAULT
TAIL_GUARD_Q_VALUES: tuple[int, ...] = (8, 16, 32)
TAIL_GUARD_M_MIN = 32768
TAIL_GUARD_M_MAX_EXCLUSIVE = 131072

def _shape(label: str, *, q: int, m: int, d: int=128, k: int=10, b: int=1, seed: int=615116) -> dict[str, Any]:
    return {'label': label, 'params': {'B': b, 'Q': q, 'M': m, 'D': d, 'K': k, 'dtype': 'bfloat16', 'seed': seed, 'self_search': False, 'min_recall': 0.999 if q > 1 else 1.0}}

def _dedupe_shapes(shapes: list[dict[str, Any]]) -> list[dict[str, Any]]:
    seen: set[str] = set()
    result: list[dict[str, Any]] = []
    for shape in shapes:
        label = str(shape['label'])
        if label in seen:
            continue
        seen.add(label)
        result.append(shape)
    return result
TAIL_GUARD_TARGET_SHAPES = _decode_capture(_json_loads('[{"__dict_items__": [["label", "guardmiss_lowq_q16_m65535_d128_k10"], ["params", {"__dict_items__": [["B", 1], ["Q", 16], ["M", 65535], ["D", 128], ["K", 10], ["dtype", "bfloat16"], ["seed", 615116], ["self_search", false], ["min_recall", 0.999]]}]]}]'))
TAIL_GUARD_MIDTAIL_SHAPES = _decode_capture(_json_loads('[{"__dict_items__": [["label", "midtail_lowq_q16_m32768_d128_k10"], ["params", {"__dict_items__": [["B", 1], ["Q", 16], ["M", 32768], ["D", 128], ["K", 10], ["dtype", "bfloat16"], ["seed", 615120], ["self_search", false], ["min_recall", 0.999]]}]]}]'))
TAIL_GUARD_BOUNDARY_SHAPES = _decode_capture(_json_loads('[{"__dict_items__": [["label", "boundary_lowq_q16_m131071_d128_k10"], ["params", {"__dict_items__": [["B", 1], ["Q", 16], ["M", 131071], ["D", 128], ["K", 10], ["dtype", "bfloat16"], ["seed", 615121], ["self_search", false], ["min_recall", 0.999]]}]]}]'))
TAIL_GUARD_HELDOUT_SHAPES = _decode_capture(_json_loads('[{"__dict_items__": [["label", "heldout_tail_guard_q8_m65535_d128_k10"], ["params", {"__dict_items__": [["B", 1], ["Q", 8], ["M", 65535], ["D", 128], ["K", 10], ["dtype", "bfloat16"], ["seed", 615108], ["self_search", false], ["min_recall", 0.999]]}]]}, {"__dict_items__": [["label", "heldout_tail_guard_q32_m65535_d128_k10"], ["params", {"__dict_items__": [["B", 1], ["Q", 32], ["M", 65535], ["D", 128], ["K", 10], ["dtype", "bfloat16"], ["seed", 615132], ["self_search", false], ["min_recall", 0.999]]}]]}]'))
TAIL_GUARD_PRESERVE_LARGE_M_SHAPES = _decode_capture(_json_loads('[{"__dict_items__": [["label", "preserve_large_m_q8_m131072_d128_k10"], ["params", {"__dict_items__": [["B", 1], ["Q", 8], ["M", 131072], ["D", 128], ["K", 10], ["dtype", "bfloat16"], ["seed", 615208], ["self_search", false], ["min_recall", 0.999]]}]]}, {"__dict_items__": [["label", "preserve_large_m_q16_m131072_d128_k10"], ["params", {"__dict_items__": [["B", 1], ["Q", 16], ["M", 131072], ["D", 128], ["K", 10], ["dtype", "bfloat16"], ["seed", 615216], ["self_search", false], ["min_recall", 0.999]]}]]}, {"__dict_items__": [["label", "preserve_large_m_q32_m131072_d128_k10"], ["params", {"__dict_items__": [["B", 1], ["Q", 32], ["M", 131072], ["D", 128], ["K", 10], ["dtype", "bfloat16"], ["seed", 615232], ["self_search", false], ["min_recall", 0.999]]}]]}]'))
TAIL_GUARD_MISS_FALLBACK_SHAPES = _decode_capture(_json_loads('[{"__dict_items__": [["label", "below_tail_guard_q16_m32767_d128_k10"], ["params", {"__dict_items__": [["B", 1], ["Q", 16], ["M", 32767], ["D", 128], ["K", 10], ["dtype", "bfloat16"], ["seed", 615119], ["self_search", false], ["min_recall", 0.999]]}]]}, {"__dict_items__": [["label", "q64_tail_guard_miss_m65535_d128_k10"], ["params", {"__dict_items__": [["B", 1], ["Q", 64], ["M", 65535], ["D", 128], ["K", 10], ["dtype", "bfloat16"], ["seed", 615164], ["self_search", false], ["min_recall", 0.999]]}]]}, {"__dict_items__": [["label", "q2_tail_guard_miss_m65535_d128_k10"], ["params", {"__dict_items__": [["B", 1], ["Q", 2], ["M", 65535], ["D", 128], ["K", 10], ["dtype", "bfloat16"], ["seed", 615102], ["self_search", false], ["min_recall", 0.999]]}]]}, {"__dict_items__": [["label", "k8_tail_guard_miss_q16_m65535_d128_k8"], ["params", {"__dict_items__": [["B", 1], ["Q", 16], ["M", 65535], ["D", 128], ["K", 8], ["dtype", "bfloat16"], ["seed", 615117], ["self_search", false], ["min_recall", 0.999]]}]]}, {"__dict_items__": [["label", "d256_tail_guard_miss_q16_m65535_d256_k10"], ["params", {"__dict_items__": [["B", 1], ["Q", 16], ["M", 65535], ["D", 256], ["K", 10], ["dtype", "bfloat16"], ["seed", 615118], ["self_search", false], ["min_recall", 0.999]]}]]}, {"__dict_items__": [["label", "b2_tail_guard_miss_q16_m65535_d128_k10"], ["params", {"__dict_items__": [["B", 2], ["Q", 16], ["M", 65535], ["D", 128], ["K", 10], ["dtype", "bfloat16"], ["seed", 615119], ["self_search", false], ["min_recall", 0.999]]}]]}]'))
TAIL_GUARD_CORRECTNESS_SHAPES = _decode_capture(_json_loads('[{"__dict_items__": [["label", "guardmiss_lowq_q16_m65535_d128_k10"], ["params", {"__dict_items__": [["B", 1], ["Q", 16], ["M", 65535], ["D", 128], ["K", 10], ["dtype", "bfloat16"], ["seed", 615116], ["self_search", false], ["min_recall", 0.999]]}]]}, {"__dict_items__": [["label", "midtail_lowq_q16_m32768_d128_k10"], ["params", {"__dict_items__": [["B", 1], ["Q", 16], ["M", 32768], ["D", 128], ["K", 10], ["dtype", "bfloat16"], ["seed", 615120], ["self_search", false], ["min_recall", 0.999]]}]]}, {"__dict_items__": [["label", "boundary_lowq_q16_m131071_d128_k10"], ["params", {"__dict_items__": [["B", 1], ["Q", 16], ["M", 131071], ["D", 128], ["K", 10], ["dtype", "bfloat16"], ["seed", 615121], ["self_search", false], ["min_recall", 0.999]]}]]}, {"__dict_items__": [["label", "heldout_tail_guard_q8_m65535_d128_k10"], ["params", {"__dict_items__": [["B", 1], ["Q", 8], ["M", 65535], ["D", 128], ["K", 10], ["dtype", "bfloat16"], ["seed", 615108], ["self_search", false], ["min_recall", 0.999]]}]]}, {"__dict_items__": [["label", "heldout_tail_guard_q32_m65535_d128_k10"], ["params", {"__dict_items__": [["B", 1], ["Q", 32], ["M", 65535], ["D", 128], ["K", 10], ["dtype", "bfloat16"], ["seed", 615132], ["self_search", false], ["min_recall", 0.999]]}]]}]'))
TAIL_GUARD_PERFORMANCE_SHAPES = [*TAIL_GUARD_TARGET_SHAPES, *TAIL_GUARD_MIDTAIL_SHAPES, *TAIL_GUARD_BOUNDARY_SHAPES]
TAIL_GUARD_COVERAGE_CATEGORY_SHAPES: dict[str, list[dict[str, Any]]] = {'target': TAIL_GUARD_TARGET_SHAPES, 'midtail': TAIL_GUARD_MIDTAIL_SHAPES, 'boundary': TAIL_GUARD_BOUNDARY_SHAPES, 'heldout': TAIL_GUARD_HELDOUT_SHAPES, 'guard_overlap': TAIL_GUARD_PRESERVE_LARGE_M_SHAPES, 'guard_miss_fallback': TAIL_GUARD_MISS_FALLBACK_SHAPES, 'forced_fallback': TAIL_GUARD_TARGET_SHAPES}
SHAPE_DISPATCH_REGISTRY: tuple[dict[str, str], ...] = ({'shape_key': 'd128_lowq_q8_q16_q32_tail_guard_miss_row16_tcgen05', 'guard': 'B == 1 and Q in {8,16,32} and 32768 <= M < 131072 and D == 128 and K == 10 and tcgen05', 'route': ROUTE_TAIL_GUARD_MISS_ROW16}, *parent.SHAPE_DISPATCH_REGISTRY)

def _forced_fallback(inputs: dict[str, Any]) -> bool:
    return parent._forced_fallback(inputs)

def _use_tail_guard_miss_row16(inputs: dict[str, Any]) -> bool:
    q_rows = int(inputs['Q'])
    m_rows = int(inputs['M'])
    return int(inputs['B']) == 1 and q_rows in TAIL_GUARD_Q_VALUES and (TAIL_GUARD_M_MIN <= m_rows < TAIL_GUARD_M_MAX_EXCLUSIVE) and (int(inputs['D']) == D_STATIC) and (int(inputs['K']) == K_MAX) and row16._tcgen05_capable_arch()

def selected_route(inputs: dict[str, Any]) -> str:
    if _forced_fallback(inputs):
        return parent.selected_route(inputs)
    if _use_tail_guard_miss_row16(inputs):
        return ROUTE_TAIL_GUARD_MISS_ROW16
    return parent.selected_route(inputs)

def selected_route_name(inputs: dict[str, Any]) -> str:
    return selected_route(inputs)

def _selected_guard(inputs: dict[str, Any], route: str, parent_info: dict[str, Any] | None) -> str:
    if _forced_fallback(inputs):
        return 'force_fallback metadata/env'
    if route == ROUTE_TAIL_GUARD_MISS_ROW16:
        return SHAPE_DISPATCH_REGISTRY[0]['guard']
    if parent_info is not None:
        return str(parent_info.get('selected_guard', 'round92 guard miss; delegate to r55 parent'))
    return 'round92 guard miss; delegate to r55 parent'

def route_info(inputs: dict[str, Any]) -> dict[str, Any]:
    route = selected_route(inputs)
    if route != ROUTE_TAIL_GUARD_MISS_ROW16:
        info = dict(parent.route_info(inputs))
        guard_order = list(info.get('guard_order', []))
        info['guard_order'] = [SHAPE_DISPATCH_REGISTRY[0]['shape_key'], *guard_order]
        info['selected_guard'] = _selected_guard(inputs, route, info)
        return info
    parent_info = parent.route_info(inputs)
    return {'route': route, 'parent_route': parent_info['route'], 'route_kind': 'specialized', 'coverage_class': 'performance_route_tail_guard_miss_row16', 'coverage_only': False, 'production_policy': 'weave_only', 'external_fallback': None, 'guard_order': [entry['shape_key'] for entry in SHAPE_DISPATCH_REGISTRY], 'forced_fallback': False, 'selected_guard': _selected_guard(inputs, route, parent_info), 'fallback': None}

def route_trace_entry(label: str, inputs: dict[str, Any]) -> dict[str, Any]:
    info = route_info(inputs)
    return {'label': label, 'B': int(inputs['B']), 'Q': int(inputs['Q']), 'M': int(inputs['M']), 'D': int(inputs['D']), 'K': int(inputs['K']), **info}

def coverage_route_trace() -> list[dict[str, Any]]:
    entries: list[dict[str, Any]] = []
    for category, shapes in TAIL_GUARD_COVERAGE_CATEGORY_SHAPES.items():
        for shape in shapes:
            inputs = dict(shape['params'])
            if category == 'forced_fallback':
                inputs['force_fallback'] = True
            entry = route_trace_entry(str(shape['label']), inputs)
            entry['category'] = category
            entries.append(entry)
    return entries

def _launch_tail_guard_miss_row16(inputs: dict[str, Any]) -> dict[str, Any]:
    if not row16._KERNELS:
        row16._KERNELS.update(row16._compile_kernels())
    bsz = int(inputs['B'])
    q_rows = int(inputs['Q'])
    m_rows = int(inputs['M'])
    k = int(inputs['K'])
    num_q_tiles = math.ceil(q_rows / row16.BLOCK_Q)
    total_m_tiles = math.ceil(m_rows / row16.BLOCK_M)
    partial_dist, partial_idx = _mma._scratch(inputs, row16.SPLIT_M, num_q_tiles)
    row16._KERNELS['partial'].launch(grid=(bsz * num_q_tiles * row16.SPLIT_M, 1, 1), block=(row16.THREADS, 1, 1), args=[inputs['queries'], inputs['database'], partial_dist, partial_idx, bsz, q_rows, m_rows, row16.SPLIT_M, num_q_tiles, total_m_tiles], shared_mem=row16.SMEM_BYTES)
    row16._KERNELS['merge'].launch(grid=(bsz * q_rows, 1, 1), block=(_mma.MERGE_THREADS, 1, 1), args=[partial_dist, partial_idx, inputs['out_distances'], inputs['out_indices'], bsz, q_rows, k, row16.SPLIT_M, num_q_tiles], shared_mem=_mma.MERGE_SMEM_BYTES)
    return {'distances': inputs['out_distances'], 'indices': inputs['out_indices']}

def launch_for_eval(inputs: dict[str, Any]) -> dict[str, Any]:
    if not _forced_fallback(inputs) and _use_tail_guard_miss_row16(inputs):
        return _launch_tail_guard_miss_row16(inputs)
    return parent.launch_for_eval(inputs)

def knn_search_compile_and_launch_tail_guard_miss_row16(*, benchmark: bool=True, shapes: list[dict[str, Any]] | None=None) -> dict[str, Any]:
    from .._dispatch_runtime import evaluate
    selected = shapes
    if selected is None:
        selected = TAIL_GUARD_PERFORMANCE_SHAPES if benchmark else TAIL_GUARD_CORRECTNESS_SHAPES
    result = evaluate(launch_for_eval, shapes=selected, benchmark=benchmark)
    result['passed'] = bool(result.get('summary', {}).get('all_correct'))
    print(result)
    return result
