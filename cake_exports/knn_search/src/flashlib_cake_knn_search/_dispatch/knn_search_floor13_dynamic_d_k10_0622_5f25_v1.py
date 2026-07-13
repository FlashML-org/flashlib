"""Round-132/5f25 floor13 dynamic-D K10 seed bundle for kNN search.

Minimum target architecture: sm_100a for the tcgen05/TMEM producer path. This
additive bucket-kernel module targets the floor13 dynamic-D guard-miss rows:
``D33/Q128/M131072``, ``D80/Q256/M65536``, ``D160/Q64/M131072``, and
``D640/Q32/M32768``. It reuses source-clean dynamic-D Weave producer schedules
with exact shape guards and does not edit production dispatch.
"""
from __future__ import annotations
from json import loads as _json_loads
from .._dispatch_runtime import _capture_cuTensorMapEncodeTiled, _decode_capture, _import_dispatch_module, _ir_proxy
from typing import Any
from .._dispatch_runtime import select_named_shapes
from . import knn_search_dynamic_d768d1024_q32q16_tcgen05_0618_9286_v1 as d768d1024
from . import knn_search_dynamic_d_high_q128_directstride_tcgen05_0618_ccef_v1 as highd
from . import knn_search_dynamic_d_residual_capacity_0621_r125_3532_v1 as parent
from . import knn_search_dynamic_d_tiny_q128_m65536_tcgen05_0618_c8b9_v1 as tinyd
THREADS = highd.THREADS
MERGE_THREADS = highd.MERGE_THREADS
BLOCK_Q = highd.BLOCK_Q
BLOCK_M = highd.BLOCK_M
K_MAX = highd.K_MAX
ir = _decode_capture(_json_loads('{"__ir__": "knn_search_dynamic_d_high_directstride_tcgen05_partial_0618_ccef_v1", "arg_keys": ["queries", "database", "partial_distances", "partial_indices", "B", "Q", "M", "split_m", "num_q_tiles", "total_m_tiles", "tiles_per_split"], "cluster_dims": [1, 1, 1], "computed_smem_bytes": 126720, "constants": [["K_MAX_", 10], ["D_ORIG_", 512], ["NUM_D_PASSES_", 4], ["Q_NORM_PARTS_", 32]], "cta_group": 1, "threads": 640}'))
partial_ir = _decode_capture(_json_loads('{"__ir__": "knn_search_dynamic_d_high_directstride_tcgen05_partial_0618_ccef_v1", "arg_keys": ["queries", "database", "partial_distances", "partial_indices", "B", "Q", "M", "split_m", "num_q_tiles", "total_m_tiles", "tiles_per_split"], "cluster_dims": [1, 1, 1], "computed_smem_bytes": 126720, "constants": [["K_MAX_", 10], ["D_ORIG_", 512], ["NUM_D_PASSES_", 4], ["Q_NORM_PARTS_", 32]], "cta_group": 1, "threads": 640}'))
merge_ir = _decode_capture(_json_loads('{"__ir__": "knn_search_mma_split_merge_q128_const148_v1", "arg_keys": ["partial_distances", "partial_indices", "out_distances", "out_indices", "B", "Q", "K", "split_m", "num_q_tiles"], "cluster_dims": [1, 1, 1], "computed_smem_bytes": 0, "constants": [["K_MAX_", 10]], "cta_group": 1, "threads": 32}'))
tinyd_partial_ir = _decode_capture(_json_loads('{"__ir__": "knn_search_dynamic_d_tiny_q128_m65536_tcgen05_partial_0618_c8b9_v1", "arg_keys": ["queries", "database", "partial_distances", "partial_indices", "B", "Q", "M", "split_m", "num_q_tiles", "total_m_tiles", "tiles_per_split"], "cluster_dims": [1, 1, 1], "computed_smem_bytes": 112384, "constants": [["K_MAX_", 10], ["D_TOTAL_", 3], ["NUM_D_PASSES_", 1], ["Q_NORM_PARTS_", 1]], "cta_group": 1, "threads": 640}'))
tinyd_merge_ir = _decode_capture(_json_loads('{"__ir__": "knn_search_mma_split_merge_q128_const148_v1", "arg_keys": ["partial_distances", "partial_indices", "out_distances", "out_indices", "B", "Q", "K", "split_m", "num_q_tiles"], "cluster_dims": [1, 1, 1], "computed_smem_bytes": 0, "constants": [["K_MAX_", 10]], "cta_group": 1, "threads": 32}'))
d768d1024_partial_ir = _decode_capture(_json_loads('{"__ir__": "knn_search_dynamic_d768d1024_q32q16_tcgen05_partial_0618_9286_v1", "arg_keys": ["queries", "database", "partial_distances", "partial_indices", "B", "Q", "M", "split_m", "num_q_tiles", "total_m_tiles", "tiles_per_split"], "cluster_dims": [1, 1, 1], "computed_smem_bytes": 143104, "constants": [["K_MAX_", 10], ["D_ORIG_", 1024], ["NUM_D_PASSES_", 8], ["Q_NORM_PARTS_", 64]], "cta_group": 1, "threads": 640}'))
d768d1024_merge_ir = _decode_capture(_json_loads('{"__ir__": "knn_search_mma_split_merge_q128_const148_v1", "arg_keys": ["partial_distances", "partial_indices", "out_distances", "out_indices", "B", "Q", "K", "split_m", "num_q_tiles"], "cluster_dims": [1, 1, 1], "computed_smem_bytes": 0, "constants": [["K_MAX_", 10]], "cta_group": 1, "threads": 32}'))
parent_ir = _decode_capture(_json_loads('{"__ir__": "knn_search_dynamic_d_high_directstride_tcgen05_partial_0618_ccef_v1", "arg_keys": ["queries", "database", "partial_distances", "partial_indices", "B", "Q", "M", "split_m", "num_q_tiles", "total_m_tiles", "tiles_per_split"], "cluster_dims": [1, 1, 1], "computed_smem_bytes": 126720, "constants": [["K_MAX_", 10], ["D_ORIG_", 512], ["NUM_D_PASSES_", 4], ["Q_NORM_PARTS_", 32]], "cta_group": 1, "threads": 640}'))
ROUTE_D33_Q128_M131072 = '5f25_floor13_dyn_d33_q128_m131072_tcgen05'
ROUTE_D80_Q256_M65536 = '5f25_floor13_dyn_d80_q256_m65536_tcgen05'
ROUTE_D160_Q64_M131072 = '5f25_floor13_dyn_d160_q64_m131072_tcgen05'
ROUTE_D640_Q32_M32768 = '5f25_floor13_dyn_d640_q32_m32768_tcgen05'
ROUTE_PARENT_FALLBACK = parent.ROUTE_PARENT_FALLBACK
CONSUMED_D33_SEED = 'weave-evolve-knn-search-c8b9-tinyd-generated-d33'
CONSUMED_D80_D160_SEED = highd.CONSUMED_SEED
CONSUMED_D640_SEED = d768d1024.CONSUMED_SEED
CONSUMED_PARENT_SEEDS: tuple[str, ...] = parent.CONSUMED_SEEDS
CONSUMED_SEEDS: tuple[str, ...] = (*CONSUMED_PARENT_SEEDS, CONSUMED_D33_SEED, CONSUMED_D80_D160_SEED, CONSUMED_D640_SEED)
TARGET_LABELS: tuple[str, ...] = ('blind_0622_dyn_d33_q128_m131072_k10', 'blind_0622_dyn_d80_q256_m65536_k10', 'blind_0622_dyn_d160_q64_m131072_k10', 'blind_0622_dyn_d640_q32_m32768_k10')
TARGET_SHAPES = _decode_capture(_json_loads('[{"__dict_items__": [["label", "blind_0622_dyn_d33_q128_m131072_k10"], ["params", {"__dict_items__": [["B", 1], ["Q", 128], ["M", 131072], ["D", 33], ["K", 10], ["dtype", "bfloat16"], ["seed", 611005], ["self_search", false], ["min_recall", 0.999]]}]]}, {"__dict_items__": [["label", "blind_0622_dyn_d80_q256_m65536_k10"], ["params", {"__dict_items__": [["B", 1], ["Q", 256], ["M", 65536], ["D", 80], ["K", 10], ["dtype", "bfloat16"], ["seed", 611006], ["self_search", false], ["min_recall", 0.999]]}]]}, {"__dict_items__": [["label", "blind_0622_dyn_d160_q64_m131072_k10"], ["params", {"__dict_items__": [["B", 1], ["Q", 64], ["M", 131072], ["D", 160], ["K", 10], ["dtype", "bfloat16"], ["seed", 611007], ["self_search", false], ["min_recall", 0.999]]}]]}, {"__dict_items__": [["label", "blind_0622_dyn_d640_q32_m32768_k10"], ["params", {"__dict_items__": [["B", 1], ["Q", 32], ["M", 32768], ["D", 640], ["K", 10], ["dtype", "bfloat16"], ["seed", 611008], ["self_search", false], ["min_recall", 0.999]]}]]}]'))
_D33_ENTRY: dict[str, Any] = {'shape_key': '5f25_floor13_dyn_d33_q128_m131072_k10', 'label': 'blind_0622_dyn_d33_q128_m131072_k10', 'shape': (1, 128, 131072, 33, 10, False), 'guard': 'B == 1 and Q == 128 and M == 131072 and D == 33 and K == 10 and not self_search and not forced_fallback and arch in {sm_100a,sm_103a}', 'route': ROUTE_D33_Q128_M131072, 'entrypoint': 'loom.examples.weave.knn_search_floor13_dynamic_d_k10_0622_5f25_v1:launch_for_eval', 'source_entrypoint': 'loom.examples.weave.knn_search_dynamic_d_tiny_q128_m65536_tcgen05_0618_c8b9_v1:_launch_tiny_dynamic_d_mma', 'selected_seed': CONSUMED_D33_SEED, 'source_round_doc': 'design_doc/active/weave_evolve_knn_search_round_123_189c_dynamicd_alias_q128.md', 'coverage_class': 'bucket_seed_floor13_dynamic_d33_q128_m131072_k10', 'arch_requirement': 'sm_100a'}
_D80_ENTRY: dict[str, Any] = {'shape_key': '5f25_floor13_dyn_d80_q256_m65536_k10', 'label': 'blind_0622_dyn_d80_q256_m65536_k10', 'shape': (1, 256, 65536, 80, 10, False), 'guard': 'B == 1 and Q == 256 and M == 65536 and D == 80 and K == 10 and not self_search and not forced_fallback and arch in {sm_100a,sm_103a}', 'route': ROUTE_D80_Q256_M65536, 'entrypoint': 'loom.examples.weave.knn_search_floor13_dynamic_d_k10_0622_5f25_v1:launch_for_eval', 'source_entrypoint': 'loom.examples.weave.knn_search_dynamic_d_high_q128_directstride_tcgen05_0618_ccef_v1:_launch_high_dynamic_d_tcgen05', 'selected_seed': CONSUMED_D80_D160_SEED, 'source_round_doc': 'design_doc/active/weave_evolve_knn_search_round_123_189c_dynamicd_alias_q128.md', 'coverage_class': 'bucket_seed_floor13_dynamic_d80_q256_m65536_k10', 'arch_requirement': 'sm_100a'}
_D160_ENTRY: dict[str, Any] = {'shape_key': '5f25_floor13_dyn_d160_q64_m131072_k10', 'label': 'blind_0622_dyn_d160_q64_m131072_k10', 'shape': (1, 64, 131072, 160, 10, False), 'guard': 'B == 1 and Q == 64 and M == 131072 and D == 160 and K == 10 and not self_search and not forced_fallback and arch in {sm_100a,sm_103a}', 'route': ROUTE_D160_Q64_M131072, 'entrypoint': 'loom.examples.weave.knn_search_floor13_dynamic_d_k10_0622_5f25_v1:launch_for_eval', 'source_entrypoint': 'loom.examples.weave.knn_search_dynamic_d_high_q128_directstride_tcgen05_0618_ccef_v1:_launch_high_dynamic_d_tcgen05', 'selected_seed': CONSUMED_D80_D160_SEED, 'source_round_doc': 'design_doc/active/weave_evolve_knn_search_round_123_189c_dynamicd_alias_q128.md', 'coverage_class': 'bucket_seed_floor13_dynamic_d160_q64_m131072_k10', 'arch_requirement': 'sm_100a'}
_D640_ENTRY: dict[str, Any] = {'shape_key': '5f25_floor13_dyn_d640_q32_m32768_k10', 'label': 'blind_0622_dyn_d640_q32_m32768_k10', 'shape': (1, 32, 32768, 640, 10, False), 'guard': 'B == 1 and Q == 32 and M == 32768 and D == 640 and K == 10 and not self_search and not forced_fallback and arch in {sm_100a,sm_103a}', 'route': ROUTE_D640_Q32_M32768, 'entrypoint': 'loom.examples.weave.knn_search_floor13_dynamic_d_k10_0622_5f25_v1:launch_for_eval', 'source_entrypoint': 'loom.examples.weave.knn_search_dynamic_d768d1024_q32q16_tcgen05_0618_9286_v1:_launch_high_dynamic_d_tcgen05', 'selected_seed': CONSUMED_D640_SEED, 'source_round_doc': 'design_doc/active/weave_evolve_knn_search_round_123_f670_dynamicd_breakthrough.md', 'coverage_class': 'bucket_seed_floor13_dynamic_d640_q32_m32768_k10', 'arch_requirement': 'sm_100a'}
_ENTRIES: tuple[dict[str, Any], ...] = (_D33_ENTRY, _D80_ENTRY, _D160_ENTRY, _D640_ENTRY)
SHAPE_DISPATCH_REGISTRY: tuple[dict[str, Any], ...] = (*_ENTRIES, *parent.SHAPE_DISPATCH_REGISTRY)

def _shape_key(inputs: dict[str, Any]) -> tuple[int, int, int, int, int, bool]:
    return (int(inputs.get('B', 1)), int(inputs['Q']), int(inputs['M']), int(inputs['D']), int(inputs['K']), bool(inputs.get('self_search', False)))

def _forced_fallback(inputs: dict[str, Any]) -> bool:
    return bool(inputs.get('force_fallback', False))

def _tcgen05_capable_for(entry: dict[str, Any]) -> bool:
    if entry is _D33_ENTRY:
        return tinyd.mma._tcgen05_capable_arch()
    if entry is _D640_ENTRY:
        return d768d1024.mma._tcgen05_capable_arch()
    return highd.mma._tcgen05_capable_arch()

def _active_entry(inputs: dict[str, Any]) -> dict[str, Any] | None:
    if _forced_fallback(inputs):
        return None
    shape = _shape_key(inputs)
    for entry in _ENTRIES:
        if shape == entry['shape'] and _tcgen05_capable_for(entry):
            return entry
    return None

def _guard_order() -> list[str]:
    return [str(entry['shape_key']) for entry in SHAPE_DISPATCH_REGISTRY]

def selected_route(inputs: dict[str, Any]) -> str:
    entry = _active_entry(inputs)
    if entry is not None:
        return str(entry['route'])
    return parent.selected_route(inputs)

def selected_route_name(inputs: dict[str, Any]) -> str:
    return selected_route(inputs)

def _parent_info(inputs: dict[str, Any]) -> dict[str, Any]:
    info = dict(parent.route_info(inputs))
    info['guard_order'] = _guard_order()
    info.setdefault('production_policy', 'weave_only')
    info.setdefault('external_fallback', None)
    return info

def _entry_info(inputs: dict[str, Any], entry: dict[str, Any]) -> dict[str, Any]:
    parent_info = dict(parent.route_info(inputs))
    parent_route = str(parent_info.get('route') or parent_info.get('selected_route') or parent.selected_route(inputs))
    return {'route': entry['route'], 'selected_route': entry['route'], 'selected_entrypoint': entry['entrypoint'], 'source_entrypoint': entry['source_entrypoint'], 'parent_route': parent_route, 'replaced_route': parent_route, 'route_kind': 'specialized', 'route_source': 'shape-specific-seed', 'coverage_class': entry['coverage_class'], 'classification': 'seed-produced', 'coverage_only': False, 'production_policy': 'weave_only', 'external_fallback': None, 'guard_order': _guard_order(), 'guard_id': entry['shape_key'], 'selected_guard': entry['guard'], 'fallback': ROUTE_PARENT_FALLBACK, 'missing_weave_route': False, 'forced_fallback': False, 'selected_seed': entry['selected_seed'], 'source_round_doc': entry['source_round_doc'], 'arch_requirement': entry['arch_requirement']}

def route_info(inputs: dict[str, Any]) -> dict[str, Any]:
    entry = _active_entry(inputs)
    if entry is None:
        return _parent_info(inputs)
    return _entry_info(inputs, entry)

def route_trace_entry(label: str, inputs: dict[str, Any]) -> dict[str, Any]:
    return {'label': label, 'shape_key': label, 'B': int(inputs['B']), 'Q': int(inputs['Q']), 'M': int(inputs['M']), 'D': int(inputs['D']), 'K': int(inputs['K']), 'self_search': bool(inputs.get('self_search', False)), **route_info(inputs)}

def launch_for_eval(inputs: dict[str, Any]) -> dict[str, Any]:
    entry = _active_entry(inputs)
    if entry is _D33_ENTRY:
        return tinyd._launch_tiny_dynamic_d_mma(inputs)
    if entry is _D80_ENTRY or entry is _D160_ENTRY:
        return highd._launch_high_dynamic_d_tcgen05(inputs)
    if entry is _D640_ENTRY:
        return d768d1024._launch_high_dynamic_d_tcgen05(inputs)
    return parent.launch_for_eval(inputs)

def _select_contract_shapes(shape_labels: str | tuple[str, ...] | list[str] | None):
    if shape_labels is None:
        return TARGET_SHAPES
    return select_named_shapes(shape_labels)

def knn_search_compile_and_launch_floor13_dynamic_d_k10_0622_5f25(*, benchmark: bool=True, shape_labels: str | tuple[str, ...] | list[str] | None=TARGET_LABELS) -> dict[str, Any]:
    from .._dispatch_runtime import evaluate
    result = evaluate(launch_for_eval, shapes=_select_contract_shapes(shape_labels), benchmark=benchmark)
    result['passed'] = bool(result.get('summary', {}).get('all_correct'))
    print(result)
    return result
