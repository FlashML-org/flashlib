"""Extended dynamic-D scalar-capacity seed bundle.

Minimum target architecture: sm_80 for inherited CUDA-core D3 routes and
sm_100a for inherited tcgen05/TMEM routes. This additive bucket-kernel module
extends the round-123 dynamic-D breakthrough wrapper with the remaining 0621
priority dynamic-D rows that already have source-clean Weave seed producers.
It does not edit production dispatch.
"""
from __future__ import annotations
from json import loads as _json_loads
from .._dispatch_runtime import _capture_cuTensorMapEncodeTiled, _decode_capture, _import_dispatch_module, _ir_proxy
from typing import Any
from .._dispatch_runtime import select_named_shapes
from . import knn_search_dispatch0618_c492_ext_dynamic_highd_9286_v1 as ext_highd
from . import knn_search_dynamic_d_high_q128_directstride_tcgen05_0618_ccef_v1 as direct_highd
from . import knn_search_dynamic_d_scalar_breakthrough_0621_r123_f670_v1 as parent
from . import knn_search_dynamic_d768d1024_q32q16_tcgen05_0618_9286_v1 as d768d1024
THREADS = parent.THREADS
MERGE_THREADS = parent.MERGE_THREADS
BLOCK_Q = parent.BLOCK_Q
BLOCK_M = parent.BLOCK_M
K_MAX = parent.K_MAX
ir = _decode_capture(_json_loads('{"__ir__": "knn_search_dynamic_d_high_directstride_tcgen05_partial_0618_ccef_v1", "arg_keys": ["queries", "database", "partial_distances", "partial_indices", "B", "Q", "M", "split_m", "num_q_tiles", "total_m_tiles", "tiles_per_split"], "cluster_dims": [1, 1, 1], "computed_smem_bytes": 126720, "constants": [["K_MAX_", 10], ["D_ORIG_", 512], ["NUM_D_PASSES_", 4], ["Q_NORM_PARTS_", 32]], "cta_group": 1, "threads": 640}'))
partial_ir = _decode_capture(_json_loads('{"__ir__": "knn_search_dynamic_d_high_directstride_tcgen05_partial_0618_ccef_v1", "arg_keys": ["queries", "database", "partial_distances", "partial_indices", "B", "Q", "M", "split_m", "num_q_tiles", "total_m_tiles", "tiles_per_split"], "cluster_dims": [1, 1, 1], "computed_smem_bytes": 126720, "constants": [["K_MAX_", 10], ["D_ORIG_", 512], ["NUM_D_PASSES_", 4], ["Q_NORM_PARTS_", 32]], "cta_group": 1, "threads": 640}'))
merge_ir = _decode_capture(_json_loads('{"__ir__": "knn_search_mma_split_merge_q128_const148_v1", "arg_keys": ["partial_distances", "partial_indices", "out_distances", "out_indices", "B", "Q", "K", "split_m", "num_q_tiles"], "cluster_dims": [1, 1, 1], "computed_smem_bytes": 0, "constants": [["K_MAX_", 10]], "cta_group": 1, "threads": 32}'))
direct_highd_partial_ir = _decode_capture(_json_loads('{"__ir__": "knn_search_dynamic_d_high_directstride_tcgen05_partial_0618_ccef_v1", "arg_keys": ["queries", "database", "partial_distances", "partial_indices", "B", "Q", "M", "split_m", "num_q_tiles", "total_m_tiles", "tiles_per_split"], "cluster_dims": [1, 1, 1], "computed_smem_bytes": 126720, "constants": [["K_MAX_", 10], ["D_ORIG_", 512], ["NUM_D_PASSES_", 4], ["Q_NORM_PARTS_", 32]], "cta_group": 1, "threads": 640}'))
direct_highd_merge_ir = _decode_capture(_json_loads('{"__ir__": "knn_search_mma_split_merge_q128_const148_v1", "arg_keys": ["partial_distances", "partial_indices", "out_distances", "out_indices", "B", "Q", "K", "split_m", "num_q_tiles"], "cluster_dims": [1, 1, 1], "computed_smem_bytes": 0, "constants": [["K_MAX_", 10]], "cta_group": 1, "threads": 32}'))
d768d1024_partial_ir = _decode_capture(_json_loads('{"__ir__": "knn_search_dynamic_d768d1024_q32q16_tcgen05_partial_0618_9286_v1", "arg_keys": ["queries", "database", "partial_distances", "partial_indices", "B", "Q", "M", "split_m", "num_q_tiles", "total_m_tiles", "tiles_per_split"], "cluster_dims": [1, 1, 1], "computed_smem_bytes": 143104, "constants": [["K_MAX_", 10], ["D_ORIG_", 1024], ["NUM_D_PASSES_", 8], ["Q_NORM_PARTS_", 64]], "cta_group": 1, "threads": 640}'))
d768d1024_merge_ir = _decode_capture(_json_loads('{"__ir__": "knn_search_mma_split_merge_q128_const148_v1", "arg_keys": ["partial_distances", "partial_indices", "out_distances", "out_indices", "B", "Q", "K", "split_m", "num_q_tiles"], "cluster_dims": [1, 1, 1], "computed_smem_bytes": 0, "constants": [["K_MAX_", 10]], "cta_group": 1, "threads": 32}'))
parent_ir = _decode_capture(_json_loads('{"__ir__": "knn_search_dynamic_d_high_directstride_tcgen05_partial_0618_ccef_v1", "arg_keys": ["queries", "database", "partial_distances", "partial_indices", "B", "Q", "M", "split_m", "num_q_tiles", "total_m_tiles", "tiles_per_split"], "cluster_dims": [1, 1, 1], "computed_smem_bytes": 126720, "constants": [["K_MAX_", 10], ["D_ORIG_", 512], ["NUM_D_PASSES_", 4], ["Q_NORM_PARTS_", 32]], "cta_group": 1, "threads": 640}'))
ROUTE_DYNAMIC_D_BREAKTHROUGH_EXT_HIGHD_MORE = ext_highd.ROUTE_EXT_DYNAMIC_HIGHD_9286
ROUTE_DYNAMIC_D_BREAKTHROUGH_D129D257 = direct_highd.ROUTE_HIGH_DYNAMIC_D_TCGEN05
ROUTE_DYNAMIC_D_BREAKTHROUGH_D768D1024 = d768d1024.ROUTE_HIGH_DYNAMIC_D_TCGEN05
ROUTE_SCALAR_CAPACITY = parent.ROUTE_SCALAR_CAPACITY
CONSUMED_EXT_HIGHD_MORE_SEED = ext_highd.CONSUMED_EXT_HIGHD_SEED
CONSUMED_D129D257_SEED = direct_highd.CONSUMED_SEED
CONSUMED_D768D1024_SEED = d768d1024.CONSUMED_SEED
CONSUMED_SEEDS: tuple[str, ...] = (*parent.CONSUMED_SEEDS, CONSUMED_EXT_HIGHD_MORE_SEED, CONSUMED_D129D257_SEED, CONSUMED_D768D1024_SEED)
NEW_TARGET_LABELS: tuple[str, ...] = ('blind_ext_dyn_d65_q128_m65536_k10', 'blind_ext_dyn_d127_q128_m65536_k10', 'blind_dyn_d129_q128_m65536_k10', 'blind_ext_dyn_d255_q128_m65536_k10', 'blind_dyn_d257_q128_m65536_k10', 'blind_ext_dyn_d768_q32_m32768_k10', 'blind_ext_dyn_d1024_q16_m32768_k10')
TARGET_LABELS: tuple[str, ...] = (*parent.TARGET_LABELS, *NEW_TARGET_LABELS)
TARGET_SHAPES = _decode_capture(_json_loads('[{"__dict_items__": [["label", "blind_dyn_self_q2048_m2048_d3_k10"], ["params", {"__dict_items__": [["B", 1], ["Q", 2048], ["M", 2048], ["D", 3], ["K", 10], ["dtype", "bfloat16"], ["seed", 610810], ["self_search", true], ["min_recall", 0.999]]}]]}, {"__dict_items__": [["label", "blind_dyn_d3_q128_m65536_k10"], ["params", {"__dict_items__": [["B", 1], ["Q", 128], ["M", 65536], ["D", 3], ["K", 10], ["dtype", "bfloat16"], ["seed", 610801], ["self_search", false], ["min_recall", 0.999]]}]]}, {"__dict_items__": [["label", "blind_ext_dyn_d31_q128_m65536_k10"], ["params", {"__dict_items__": [["B", 1], ["Q", 128], ["M", 65536], ["D", 31], ["K", 10], ["dtype", "bfloat16"], ["seed", 610916], ["self_search", false], ["min_recall", 0.999]]}]]}, {"__dict_items__": [["label", "blind_dyn_d63_q128_m65536_k10"], ["params", {"__dict_items__": [["B", 1], ["Q", 128], ["M", 65536], ["D", 63], ["K", 10], ["dtype", "bfloat16"], ["seed", 610803], ["self_search", false], ["min_recall", 0.999]]}]]}, {"__dict_items__": [["label", "blind_ext_dyn_d512_q64_m65536_k10"], ["params", {"__dict_items__": [["B", 1], ["Q", 64], ["M", 65536], ["D", 512], ["K", 10], ["dtype", "bfloat16"], ["seed", 610922], ["self_search", false], ["min_recall", 0.999]]}]]}, {"__dict_items__": [["label", "blind_ext_dyn_d65_q128_m65536_k10"], ["params", {"__dict_items__": [["B", 1], ["Q", 128], ["M", 65536], ["D", 65], ["K", 10], ["dtype", "bfloat16"], ["seed", 610917], ["self_search", false], ["min_recall", 0.999]]}]]}, {"__dict_items__": [["label", "blind_ext_dyn_d127_q128_m65536_k10"], ["params", {"__dict_items__": [["B", 1], ["Q", 128], ["M", 65536], ["D", 127], ["K", 10], ["dtype", "bfloat16"], ["seed", 610918], ["self_search", false], ["min_recall", 0.999]]}]]}, {"__dict_items__": [["label", "blind_dyn_d129_q128_m65536_k10"], ["params", {"__dict_items__": [["B", 1], ["Q", 128], ["M", 65536], ["D", 129], ["K", 10], ["dtype", "bfloat16"], ["seed", 610804], ["self_search", false], ["min_recall", 0.999]]}]]}, {"__dict_items__": [["label", "blind_ext_dyn_d255_q128_m65536_k10"], ["params", {"__dict_items__": [["B", 1], ["Q", 128], ["M", 65536], ["D", 255], ["K", 10], ["dtype", "bfloat16"], ["seed", 610920], ["self_search", false], ["min_recall", 0.999]]}]]}, {"__dict_items__": [["label", "blind_dyn_d257_q128_m65536_k10"], ["params", {"__dict_items__": [["B", 1], ["Q", 128], ["M", 65536], ["D", 257], ["K", 10], ["dtype", "bfloat16"], ["seed", 610805], ["self_search", false], ["min_recall", 0.999]]}]]}, {"__dict_items__": [["label", "blind_ext_dyn_d768_q32_m32768_k10"], ["params", {"__dict_items__": [["B", 1], ["Q", 32], ["M", 32768], ["D", 768], ["K", 10], ["dtype", "bfloat16"], ["seed", 610923], ["self_search", false], ["min_recall", 0.999]]}]]}, {"__dict_items__": [["label", "blind_ext_dyn_d1024_q16_m32768_k10"], ["params", {"__dict_items__": [["B", 1], ["Q", 16], ["M", 32768], ["D", 1024], ["K", 10], ["dtype", "bfloat16"], ["seed", 610924], ["self_search", false], ["min_recall", 0.999]]}]]}]'))
NEW_TARGET_SHAPES = _decode_capture(_json_loads('[{"__dict_items__": [["label", "blind_ext_dyn_d65_q128_m65536_k10"], ["params", {"__dict_items__": [["B", 1], ["Q", 128], ["M", 65536], ["D", 65], ["K", 10], ["dtype", "bfloat16"], ["seed", 610917], ["self_search", false], ["min_recall", 0.999]]}]]}, {"__dict_items__": [["label", "blind_ext_dyn_d127_q128_m65536_k10"], ["params", {"__dict_items__": [["B", 1], ["Q", 128], ["M", 65536], ["D", 127], ["K", 10], ["dtype", "bfloat16"], ["seed", 610918], ["self_search", false], ["min_recall", 0.999]]}]]}, {"__dict_items__": [["label", "blind_dyn_d129_q128_m65536_k10"], ["params", {"__dict_items__": [["B", 1], ["Q", 128], ["M", 65536], ["D", 129], ["K", 10], ["dtype", "bfloat16"], ["seed", 610804], ["self_search", false], ["min_recall", 0.999]]}]]}, {"__dict_items__": [["label", "blind_ext_dyn_d255_q128_m65536_k10"], ["params", {"__dict_items__": [["B", 1], ["Q", 128], ["M", 65536], ["D", 255], ["K", 10], ["dtype", "bfloat16"], ["seed", 610920], ["self_search", false], ["min_recall", 0.999]]}]]}, {"__dict_items__": [["label", "blind_dyn_d257_q128_m65536_k10"], ["params", {"__dict_items__": [["B", 1], ["Q", 128], ["M", 65536], ["D", 257], ["K", 10], ["dtype", "bfloat16"], ["seed", 610805], ["self_search", false], ["min_recall", 0.999]]}]]}, {"__dict_items__": [["label", "blind_ext_dyn_d768_q32_m32768_k10"], ["params", {"__dict_items__": [["B", 1], ["Q", 32], ["M", 32768], ["D", 768], ["K", 10], ["dtype", "bfloat16"], ["seed", 610923], ["self_search", false], ["min_recall", 0.999]]}]]}, {"__dict_items__": [["label", "blind_ext_dyn_d1024_q16_m32768_k10"], ["params", {"__dict_items__": [["B", 1], ["Q", 16], ["M", 32768], ["D", 1024], ["K", 10], ["dtype", "bfloat16"], ["seed", 610924], ["self_search", false], ["min_recall", 0.999]]}]]}]'))
REQUESTED_BUT_MISSING_CONTRACT_LABELS: tuple[str, ...] = (*parent.REQUESTED_BUT_MISSING_CONTRACT_LABELS, 'blind_ext_dyn_d129_q128_m65536_k10', 'blind_ext_dyn_d257_q128_m65536_k10')
_EXT_HIGHD_MORE_DIMS: frozenset[int] = frozenset({65, 127, 255})
_D129D257_DIMS: frozenset[int] = frozenset({129, 257})
_EXT_HIGHD_MORE_ENTRY: dict[str, Any] = {'shape_key': 'f670_r124_ext_dynamic_d65_d127_d255_q128_m65536_k10', 'labels': ('blind_ext_dyn_d65_q128_m65536_k10', 'blind_ext_dyn_d127_q128_m65536_k10', 'blind_ext_dyn_d255_q128_m65536_k10'), 'guard': 'B == 1 and Q == 128 and M == 65536 and D in {65,127,255} and K == 10 and not self_search and not forced_fallback and arch in {sm_100a,sm_103a}', 'route': ROUTE_DYNAMIC_D_BREAKTHROUGH_EXT_HIGHD_MORE, 'entrypoint': 'loom.examples.weave.knn_search_dynamic_d_scalar_breakthrough_0621_r124_f670_extrows_v1:launch_for_eval', 'source_entrypoint': 'loom.examples.weave.knn_search_dispatch0618_c492_ext_dynamic_highd_9286_v1:launch_for_eval', 'selected_seed': CONSUMED_EXT_HIGHD_MORE_SEED, 'source_round_doc': 'design_doc/active/generalize_auto_tuning_knn_search_round_111_9286.md', 'coverage_class': 'bucket_seed_ext_dynamic_d65_d127_d255_q128_m65536_k10', 'arch_requirement': 'sm_100a'}
_D129D257_ENTRY: dict[str, Any] = {'shape_key': 'f670_r124_dynamic_d129_d257_q128_m65536_k10', 'labels': ('blind_dyn_d129_q128_m65536_k10', 'blind_dyn_d257_q128_m65536_k10'), 'requested_label_aliases': {'blind_ext_dyn_d129_q128_m65536_k10': 'blind_dyn_d129_q128_m65536_k10', 'blind_ext_dyn_d257_q128_m65536_k10': 'blind_dyn_d257_q128_m65536_k10'}, 'guard': 'B == 1 and Q == 128 and M == 65536 and D in {129,257} and K == 10 and not self_search and not forced_fallback and arch in {sm_100a,sm_103a}', 'route': ROUTE_DYNAMIC_D_BREAKTHROUGH_D129D257, 'entrypoint': 'loom.examples.weave.knn_search_dynamic_d_scalar_breakthrough_0621_r124_f670_extrows_v1:launch_for_eval', 'source_entrypoint': 'loom.examples.weave.knn_search_dynamic_d_high_q128_directstride_tcgen05_0618_ccef_v1:launch_for_eval', 'selected_seed': CONSUMED_D129D257_SEED, 'source_round_doc': 'design_doc/active/generalize_auto_tuning_knn_search_round_112_c2e0.md', 'coverage_class': 'bucket_seed_dynamic_d129_d257_q128_m65536_k10', 'arch_requirement': 'sm_100a'}
_D768D1024_ENTRY: dict[str, Any] = {'shape_key': 'f670_r124_ext_dynamic_d768_d1024_q32_q16_m32768_k10', 'labels': ('blind_ext_dyn_d768_q32_m32768_k10', 'blind_ext_dyn_d1024_q16_m32768_k10'), 'guard': 'B == 1 and M == 32768 and (Q,D) in {(32,768),(16,1024)} and K == 10 and not self_search and not forced_fallback and arch in {sm_100a,sm_103a}', 'route': ROUTE_DYNAMIC_D_BREAKTHROUGH_D768D1024, 'entrypoint': 'loom.examples.weave.knn_search_dynamic_d_scalar_breakthrough_0621_r124_f670_extrows_v1:launch_for_eval', 'source_entrypoint': 'loom.examples.weave.knn_search_dynamic_d768d1024_q32q16_tcgen05_0618_9286_v1:launch_for_eval', 'selected_seed': CONSUMED_D768D1024_SEED, 'source_round_doc': 'design_doc/active/generalize_auto_tuning_knn_search_round_112_c2e0.md', 'coverage_class': 'bucket_seed_ext_dynamic_d768d1024_q32q16_m32768_k10', 'arch_requirement': 'sm_100a'}
_NEW_ENTRIES: tuple[dict[str, Any], ...] = (_EXT_HIGHD_MORE_ENTRY, _D129D257_ENTRY, _D768D1024_ENTRY)
SHAPE_DISPATCH_REGISTRY: tuple[dict[str, Any], ...] = (*_NEW_ENTRIES, *parent.SHAPE_DISPATCH_REGISTRY)

def _shape_key(inputs: dict[str, Any]) -> tuple[int, int, int, int, int, bool]:
    return (int(inputs.get('B', 1)), int(inputs['Q']), int(inputs['M']), int(inputs['D']), int(inputs['K']), bool(inputs.get('self_search', False)))

def _forced_fallback(inputs: dict[str, Any]) -> bool:
    return bool(inputs.get('force_fallback', False))

def _use_ext_highd_more(inputs: dict[str, Any]) -> bool:
    return int(inputs['D']) in _EXT_HIGHD_MORE_DIMS and ext_highd._use_ext_dynamic_highd(inputs)

def _use_d129d257(inputs: dict[str, Any]) -> bool:
    if _forced_fallback(inputs):
        return False
    bsz, q_rows, m_rows, dim, k, self_search = _shape_key(inputs)
    return bsz == 1 and q_rows == 128 and (m_rows == 65536) and (dim in _D129D257_DIMS) and (k == 10) and (not self_search) and direct_highd.mma._tcgen05_capable_arch()

def _use_d768d1024(inputs: dict[str, Any]) -> bool:
    return d768d1024._use_high_dynamic_d_tcgen05(inputs)

def _active_new_entry(inputs: dict[str, Any]) -> dict[str, Any] | None:
    if _use_ext_highd_more(inputs):
        return _EXT_HIGHD_MORE_ENTRY
    if _use_d129d257(inputs):
        return _D129D257_ENTRY
    if _use_d768d1024(inputs):
        return _D768D1024_ENTRY
    return None

def _active_entry(inputs: dict[str, Any]) -> dict[str, Any] | None:
    new_entry = _active_new_entry(inputs)
    if new_entry is not None:
        return new_entry
    return parent._active_entry(inputs)

def _guard_order() -> list[str]:
    return [str(entry['shape_key']) for entry in SHAPE_DISPATCH_REGISTRY]

def selected_route(inputs: dict[str, Any]) -> str:
    entry = _active_new_entry(inputs)
    if entry is not None:
        return str(entry['route'])
    return parent.selected_route(inputs)

def selected_route_name(inputs: dict[str, Any]) -> str:
    return selected_route(inputs)

def _parent_info(inputs: dict[str, Any]) -> dict[str, Any]:
    info = dict(parent.route_info(inputs))
    info['guard_order'] = _guard_order()
    return info

def _new_entry_info(inputs: dict[str, Any], entry: dict[str, Any]) -> dict[str, Any]:
    parent_info = dict(parent.route_info(inputs))
    parent_route = str(parent_info.get('route') or parent_info.get('selected_route') or parent.selected_route(inputs))
    info = {'route': entry['route'], 'selected_route': entry['route'], 'selected_entrypoint': entry['entrypoint'], 'source_entrypoint': entry['source_entrypoint'], 'parent_route': parent_route, 'replaced_route': parent_route, 'route_kind': 'specialized', 'route_source': 'shape-specific-seed', 'coverage_class': entry['coverage_class'], 'classification': 'seed-produced', 'coverage_only': False, 'production_policy': 'weave_only', 'external_fallback': None, 'guard_order': _guard_order(), 'guard_id': entry['shape_key'], 'selected_guard': entry['guard'], 'fallback': parent_route, 'missing_weave_route': False, 'forced_fallback': False, 'selected_seed': entry['selected_seed'], 'source_round_doc': entry['source_round_doc'], 'arch_requirement': entry['arch_requirement']}
    if 'requested_label_aliases' in entry:
        info['requested_label_aliases'] = entry['requested_label_aliases']
    return info

def route_info(inputs: dict[str, Any]) -> dict[str, Any]:
    entry = _active_new_entry(inputs)
    if entry is None:
        return _parent_info(inputs)
    return _new_entry_info(inputs, entry)

def route_trace_entry(label: str, inputs: dict[str, Any]) -> dict[str, Any]:
    return {'label': label, 'shape_key': label, 'B': int(inputs['B']), 'Q': int(inputs['Q']), 'M': int(inputs['M']), 'D': int(inputs['D']), 'K': int(inputs['K']), 'self_search': bool(inputs.get('self_search', False)), **route_info(inputs)}

def launch_for_eval(inputs: dict[str, Any]) -> dict[str, Any]:
    entry = _active_new_entry(inputs)
    if entry is _EXT_HIGHD_MORE_ENTRY:
        return ext_highd.launch_for_eval(inputs)
    if entry is _D129D257_ENTRY:
        return direct_highd.launch_for_eval(inputs)
    if entry is _D768D1024_ENTRY:
        return d768d1024.launch_for_eval(inputs)
    return parent.launch_for_eval(inputs)

def _select_contract_shapes(shape_labels: str | tuple[str, ...] | list[str] | None):
    if shape_labels is None:
        return NEW_TARGET_SHAPES
    return select_named_shapes(shape_labels)

def knn_search_compile_and_launch_dynamic_d_scalar_breakthrough_0621_r124_f670_extrows(*, benchmark: bool=True, shape_labels: str | tuple[str, ...] | list[str] | None=NEW_TARGET_LABELS) -> dict[str, Any]:
    from .._dispatch_runtime import evaluate
    result = evaluate(launch_for_eval, shapes=_select_contract_shapes(shape_labels), benchmark=benchmark)
    result['passed'] = bool(result.get('summary', {}).get('all_correct'))
    print(result)
    return result
