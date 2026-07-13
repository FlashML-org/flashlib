"""Dynamic-D scalar-capacity requested-alias seed aggregator.

Minimum target architecture: sm_80 for the CUDA-core D3 routes and sm_100a for
the inherited tcgen05/TMEM routes. This additive bucket-kernel module keeps the
round-123 Weave-only seed routes intact, but resolves the 0621 request label
``blind_ext_dyn_d63_q128_m65536_k10`` to the ratified contract's identical D63
row before invoking the contract harness.
"""
from __future__ import annotations
from json import loads as _json_loads
from .._dispatch_runtime import _capture_cuTensorMapEncodeTiled, _decode_capture, _import_dispatch_module, _ir_proxy
from typing import Any
from .._dispatch_runtime import select_named_shapes
from . import knn_search_dynamic_d_scalar_breakthrough_0621_r123_f670_v1 as parent
THREADS = parent.THREADS
MERGE_THREADS = parent.MERGE_THREADS
BLOCK_Q = parent.BLOCK_Q
BLOCK_M = parent.BLOCK_M
K_MAX = parent.K_MAX
ir = _decode_capture(_json_loads('{"__ir__": "knn_search_dynamic_d_high_directstride_tcgen05_partial_0618_ccef_v1", "arg_keys": ["queries", "database", "partial_distances", "partial_indices", "B", "Q", "M", "split_m", "num_q_tiles", "total_m_tiles", "tiles_per_split"], "cluster_dims": [1, 1, 1], "computed_smem_bytes": 126720, "constants": [["K_MAX_", 10], ["D_ORIG_", 512], ["NUM_D_PASSES_", 4], ["Q_NORM_PARTS_", 32]], "cta_group": 1, "threads": 640}'))
partial_ir = _decode_capture(_json_loads('{"__ir__": "knn_search_dynamic_d_high_directstride_tcgen05_partial_0618_ccef_v1", "arg_keys": ["queries", "database", "partial_distances", "partial_indices", "B", "Q", "M", "split_m", "num_q_tiles", "total_m_tiles", "tiles_per_split"], "cluster_dims": [1, 1, 1], "computed_smem_bytes": 126720, "constants": [["K_MAX_", 10], ["D_ORIG_", 512], ["NUM_D_PASSES_", 4], ["Q_NORM_PARTS_", 32]], "cta_group": 1, "threads": 640}'))
merge_ir = _decode_capture(_json_loads('{"__ir__": "knn_search_mma_split_merge_q128_const148_v1", "arg_keys": ["partial_distances", "partial_indices", "out_distances", "out_indices", "B", "Q", "K", "split_m", "num_q_tiles"], "cluster_dims": [1, 1, 1], "computed_smem_bytes": 0, "constants": [["K_MAX_", 10]], "cta_group": 1, "threads": 32}'))
d3_partial_ir = _decode_capture(_json_loads('{"__ir__": "knn_search_dynamic_tinyd_tile_reduce_partial_0618_c8b9_v1", "arg_keys": ["queries", "database", "partial_distances", "partial_indices", "B", "Q", "M", "K", "num_m_tiles"], "cluster_dims": [1, 1, 1], "computed_smem_bytes": 10240, "constants": [["D_", 3], ["BLOCK_M_", 4096], ["ROWS_PER_WORKER_", 32], ["K_MAX_", 10]], "cta_group": 1, "threads": 128}'))
d3_merge_ir = _decode_capture(_json_loads('{"__ir__": "knn_search_dynamic_tinyd_tile_reduce_merge_0618_c8b9_v1", "arg_keys": ["partial_distances", "partial_indices", "out_distances", "out_indices", "B", "Q", "K", "num_m_tiles", "num_groups", "tiles_per_group"], "cluster_dims": [1, 1, 1], "computed_smem_bytes": 384, "constants": [["K_MAX_", 10]], "cta_group": 1, "threads": 128}'))
tinyd_partial_ir = _decode_capture(_json_loads('{"__ir__": "knn_search_dynamic_d_tiny_q128_m65536_tcgen05_partial_0618_c8b9_v1", "arg_keys": ["queries", "database", "partial_distances", "partial_indices", "B", "Q", "M", "split_m", "num_q_tiles", "total_m_tiles", "tiles_per_split"], "cluster_dims": [1, 1, 1], "computed_smem_bytes": 112384, "constants": [["K_MAX_", 10], ["D_TOTAL_", 3], ["NUM_D_PASSES_", 1], ["Q_NORM_PARTS_", 1]], "cta_group": 1, "threads": 640}'))
tinyd_merge_ir = _decode_capture(_json_loads('{"__ir__": "knn_search_mma_split_merge_q128_const148_v1", "arg_keys": ["partial_distances", "partial_indices", "out_distances", "out_indices", "B", "Q", "K", "split_m", "num_q_tiles"], "cluster_dims": [1, 1, 1], "computed_smem_bytes": 0, "constants": [["K_MAX_", 10]], "cta_group": 1, "threads": 32}'))
d512_partial_ir = _decode_capture(_json_loads('{"__ir__": "knn_search_dynamic_d512_q64_tcgen05_partial_0618_9286_v1", "arg_keys": ["queries", "database", "partial_distances", "partial_indices", "B", "Q", "M", "split_m", "num_q_tiles", "total_m_tiles"], "cluster_dims": [1, 1, 1], "computed_smem_bytes": 102144, "constants": [["K_MAX_", 10]], "cta_group": 1, "threads": 512}'))
d512_merge_ir = _decode_capture(_json_loads('{"__ir__": "knn_search_mma_split_merge_q128_const148_v1", "arg_keys": ["partial_distances", "partial_indices", "out_distances", "out_indices", "B", "Q", "K", "split_m", "num_q_tiles"], "cluster_dims": [1, 1, 1], "computed_smem_bytes": 0, "constants": [["K_MAX_", 10]], "cta_group": 1, "threads": 32}'))
self_d3_ir = _decode_capture(_json_loads('{"__ir__": "knn_search_dynamic_d3_tile_reduce_partial_0618_c8b9_v1", "arg_keys": ["queries", "database", "partial_distances", "partial_indices", "B", "Q", "M", "K", "num_m_tiles"], "cluster_dims": [1, 1, 1], "computed_smem_bytes": 10240, "constants": [["K_MAX_", 10]], "cta_group": 1, "threads": 256}'))
scalar_capacity_ir = _decode_capture(_json_loads('{"__ir__": "knn_search_scalar_capacity_partial_v1", "arg_keys": ["queries", "database", "partial_distances", "partial_indices", "B", "Q", "M", "num_m_tiles"], "cluster_dims": [1, 1, 1], "computed_smem_bytes": 0, "constants": [["D_", 8], ["K_CAP_", 64], ["BLOCK_M_", 512], ["NUM_WARPS_", 8]], "cta_group": 1, "threads": 256}'))
ROUTE_DYNAMIC_D_BREAKTHROUGH_D3_Q128 = parent.ROUTE_DYNAMIC_D_BREAKTHROUGH_D3_Q128
ROUTE_DYNAMIC_D_BREAKTHROUGH_D63_Q128 = parent.ROUTE_DYNAMIC_D_BREAKTHROUGH_D63_Q128
ROUTE_DYNAMIC_D_BREAKTHROUGH_D31_Q128 = parent.ROUTE_DYNAMIC_D_BREAKTHROUGH_D31_Q128
ROUTE_DYNAMIC_D_BREAKTHROUGH_SELF_D3 = parent.ROUTE_DYNAMIC_D_BREAKTHROUGH_SELF_D3
ROUTE_DYNAMIC_D_BREAKTHROUGH_D512_Q64 = parent.ROUTE_DYNAMIC_D_BREAKTHROUGH_D512_Q64
ROUTE_SCALAR_CAPACITY = parent.ROUTE_SCALAR_CAPACITY
CONSUMED_D3_Q128_SEED = parent.CONSUMED_D3_Q128_SEED
CONSUMED_D63_Q128_SEED = parent.CONSUMED_D63_Q128_SEED
CONSUMED_D31_Q128_SEED = parent.CONSUMED_D31_Q128_SEED
CONSUMED_SELF_D3_SEED = parent.CONSUMED_SELF_D3_SEED
CONSUMED_D512_Q64_SEED = parent.CONSUMED_D512_Q64_SEED
CONSUMED_SEEDS = parent.CONSUMED_SEEDS
REQUESTED_D63_ALIAS = 'blind_ext_dyn_d63_q128_m65536_k10'
CONTRACT_D63_LABEL = 'blind_dyn_d63_q128_m65536_k10'
ALIAS_TO_CONTRACT_LABEL: dict[str, str] = {REQUESTED_D63_ALIAS: CONTRACT_D63_LABEL}
MODULE_ENTRYPOINT = 'loom.examples.weave.knn_search_dynamic_d_scalar_breakthrough_0621_r124_f670_d63_alias_v1:launch_for_eval'
TARGET_LABELS: tuple[str, ...] = ('blind_dyn_self_q2048_m2048_d3_k10', 'blind_dyn_d3_q128_m65536_k10', 'blind_ext_dyn_d31_q128_m65536_k10', REQUESTED_D63_ALIAS, 'blind_ext_dyn_d512_q64_m65536_k10')
REQUESTED_BUT_MISSING_CONTRACT_LABELS: tuple[str, ...] = ()
RESOLVED_CONTRACT_ALIASES = _decode_capture(_json_loads('{"__dict_items__": [["blind_ext_dyn_d63_q128_m65536_k10", "blind_dyn_d63_q128_m65536_k10"]]}'))

def _with_entrypoint(entry: dict[str, Any]) -> dict[str, Any]:
    rewritten = dict(entry)
    rewritten['entrypoint'] = MODULE_ENTRYPOINT
    if rewritten.get('label') == CONTRACT_D63_LABEL:
        rewritten['label'] = REQUESTED_D63_ALIAS
        rewritten['shape_key'] = 'f670_requested_ext_dynamic_d63_q128_m65536_k10'
        rewritten['contract_label'] = CONTRACT_D63_LABEL
    return rewritten
SHAPE_DISPATCH_REGISTRY = _decode_capture(_json_loads('{"__tuple__": [{"__dict_items__": [["shape_key", "f670_dynamic_self_q2048_m2048_d3_k10"], ["label", "blind_dyn_self_q2048_m2048_d3_k10"], ["guard", "B == 1 and Q == 2048 and M == 2048 and D == 3 and K == 10 and self_search and not forced_fallback"], ["route", "ccef_dynamic_d_self_q2048_d3_tile_reduce"], ["entrypoint", "loom.examples.weave.knn_search_dynamic_d_scalar_breakthrough_0621_r124_f670_d63_alias_v1:launch_for_eval"], ["source_entrypoint", "loom.examples.weave.knn_search_dynamic_d_remaining_seeds_0618_ccef_v2:launch_for_eval"], ["selected_seed", "weave-evolve-knn-search-a2ab"], ["source_round_doc", "design_doc/active/generalize_auto_tuning_knn_search_round_4_generalize-auto-tuning-knn-search-04af.md"], ["coverage_class", "bucket_seed_dynamic_d_self_q2048_m2048_d3_k10"], ["arch_requirement", "sm_80"]]}, {"__dict_items__": [["shape_key", "f670_dynamic_d3_q128_m65536_k10"], ["label", "blind_dyn_d3_q128_m65536_k10"], ["guard", "B == 1 and Q == 128 and M == 65536 and D == 3 and K == 10 and not self_search and not forced_fallback"], ["route", "b5b2_dynamic_d3_tinyd_tile_reduce"], ["entrypoint", "loom.examples.weave.knn_search_dynamic_d_scalar_breakthrough_0621_r124_f670_d63_alias_v1:launch_for_eval"], ["source_entrypoint", "loom.examples.weave.knn_search_dynamic_tinyd_d3_tile_reduce_0618_b5b2_v1:launch_for_eval"], ["selected_seed", "weave-evolve-knn-search-b5b2"], ["source_round_doc", "design_doc/active/generalize_auto_tuning_knn_search_round_4_generalize-auto-tuning-knn-search-04af.md"], ["coverage_class", "bucket_seed_dynamic_d3_q128_m65536_k10"], ["arch_requirement", "sm_80"]]}, {"__dict_items__": [["shape_key", "f670_ext_dynamic_d31_q128_m65536_k10"], ["label", "blind_ext_dyn_d31_q128_m65536_k10"], ["guard", "B == 1 and Q == 128 and M == 65536 and D == 31 and K == 10 and not self_search and not forced_fallback and arch in {sm_100a,sm_103a}"], ["route", "9286_ext_dynamic_d_highd_generated_variants"], ["entrypoint", "loom.examples.weave.knn_search_dynamic_d_scalar_breakthrough_0621_r124_f670_d63_alias_v1:launch_for_eval"], ["source_entrypoint", "loom.examples.weave.knn_search_dispatch0618_c492_ext_dynamic_highd_9286_v1:launch_for_eval"], ["selected_seed", "weave-evolve-knn-search-ccef-highd-directstride"], ["source_round_doc", "design_doc/active/generalize_auto_tuning_knn_search_round_111_9286.md"], ["coverage_class", "bucket_seed_ext_dynamic_d31_q128_m65536_k10"], ["arch_requirement", "sm_100a"]]}, {"__dict_items__": [["shape_key", "f670_requested_ext_dynamic_d63_q128_m65536_k10"], ["label", "blind_ext_dyn_d63_q128_m65536_k10"], ["requested_label_alias", "blind_ext_dyn_d63_q128_m65536_k10"], ["guard", "B == 1 and Q == 128 and M == 65536 and D == 63 and K == 10 and not self_search and not forced_fallback and arch in {sm_100a,sm_103a}"], ["route", "f670_dynamic_d63_q128_no_pack_tcgen05"], ["entrypoint", "loom.examples.weave.knn_search_dynamic_d_scalar_breakthrough_0621_r124_f670_d63_alias_v1:launch_for_eval"], ["source_entrypoint", "loom.examples.weave.knn_search_dynamic_d_tiny_q128_m65536_tcgen05_0618_c8b9_v1:launch_for_eval"], ["selected_seed", "weave-evolve-knn-search-449d"], ["source_round_doc", "design_doc/active/generalize_auto_tuning_knn_search_round_4_generalize-auto-tuning-knn-search-04af.md"], ["coverage_class", "bucket_seed_dynamic_d63_q128_m65536_k10"], ["arch_requirement", "sm_100a"], ["contract_label", "blind_dyn_d63_q128_m65536_k10"]]}, {"__dict_items__": [["shape_key", "f670_ext_dynamic_d512_q64_m65536_k10"], ["label", "blind_ext_dyn_d512_q64_m65536_k10"], ["guard", "B == 1 and Q == 64 and M == 65536 and D == 512 and K == 10 and not self_search and not forced_fallback and arch in {sm_100a,sm_103a}"], ["route", "9286_d512_q64_row16_directstride_tcgen05"], ["entrypoint", "loom.examples.weave.knn_search_dynamic_d_scalar_breakthrough_0621_r124_f670_d63_alias_v1:launch_for_eval"], ["source_entrypoint", "loom.examples.weave.knn_search_dynamic_d512_q64_tcgen05_0618_9286_v1:launch_for_eval"], ["selected_seed", "weave-evolve-knn-search-9286-d512-q64-row16-directstride"], ["source_round_doc", "design_doc/active/weave_evolve_knn_search_round_112_9286.md"], ["coverage_class", "bucket_seed_ext_dynamic_d512_q64_m65536_k10"], ["arch_requirement", "sm_100a"]]}]}'))
TARGET_SHAPES = _decode_capture(_json_loads('[{"__dict_items__": [["label", "blind_dyn_self_q2048_m2048_d3_k10"], ["params", {"__dict_items__": [["B", 1], ["Q", 2048], ["M", 2048], ["D", 3], ["K", 10], ["dtype", "bfloat16"], ["seed", 610810], ["self_search", true], ["min_recall", 0.999]]}]]}, {"__dict_items__": [["label", "blind_dyn_d3_q128_m65536_k10"], ["params", {"__dict_items__": [["B", 1], ["Q", 128], ["M", 65536], ["D", 3], ["K", 10], ["dtype", "bfloat16"], ["seed", 610801], ["self_search", false], ["min_recall", 0.999]]}]]}, {"__dict_items__": [["label", "blind_ext_dyn_d31_q128_m65536_k10"], ["params", {"__dict_items__": [["B", 1], ["Q", 128], ["M", 65536], ["D", 31], ["K", 10], ["dtype", "bfloat16"], ["seed", 610916], ["self_search", false], ["min_recall", 0.999]]}]]}, {"__dict_items__": [["label", "blind_ext_dyn_d63_q128_m65536_k10"], ["params", {"__dict_items__": [["B", 1], ["Q", 128], ["M", 65536], ["D", 63], ["K", 10], ["dtype", "bfloat16"], ["seed", 610803], ["self_search", false], ["min_recall", 0.999]]}]]}, {"__dict_items__": [["label", "blind_ext_dyn_d512_q64_m65536_k10"], ["params", {"__dict_items__": [["B", 1], ["Q", 64], ["M", 65536], ["D", 512], ["K", 10], ["dtype", "bfloat16"], ["seed", 610922], ["self_search", false], ["min_recall", 0.999]]}]]}]'))

def _as_label_list(shape_labels: str | tuple[str, ...] | list[str]) -> list[str]:
    return [shape_labels] if isinstance(shape_labels, str) else list(shape_labels)

def _select_contract_shapes(shape_labels: str | tuple[str, ...] | list[str] | None):
    labels = list(TARGET_LABELS) if shape_labels is None else _as_label_list(shape_labels)
    selected: list[dict[str, Any]] = []
    for label in labels:
        contract_label = ALIAS_TO_CONTRACT_LABEL.get(label, label)
        shape = select_named_shapes(contract_label)[0]
        if contract_label == label:
            selected.append(shape)
            continue
        selected.append({'label': label, 'params': dict(shape['params'])})
    return selected
TARGET_SHAPES = _decode_capture(_json_loads('[{"__dict_items__": [["label", "blind_dyn_self_q2048_m2048_d3_k10"], ["params", {"__dict_items__": [["B", 1], ["Q", 2048], ["M", 2048], ["D", 3], ["K", 10], ["dtype", "bfloat16"], ["seed", 610810], ["self_search", true], ["min_recall", 0.999]]}]]}, {"__dict_items__": [["label", "blind_dyn_d3_q128_m65536_k10"], ["params", {"__dict_items__": [["B", 1], ["Q", 128], ["M", 65536], ["D", 3], ["K", 10], ["dtype", "bfloat16"], ["seed", 610801], ["self_search", false], ["min_recall", 0.999]]}]]}, {"__dict_items__": [["label", "blind_ext_dyn_d31_q128_m65536_k10"], ["params", {"__dict_items__": [["B", 1], ["Q", 128], ["M", 65536], ["D", 31], ["K", 10], ["dtype", "bfloat16"], ["seed", 610916], ["self_search", false], ["min_recall", 0.999]]}]]}, {"__dict_items__": [["label", "blind_ext_dyn_d63_q128_m65536_k10"], ["params", {"__dict_items__": [["B", 1], ["Q", 128], ["M", 65536], ["D", 63], ["K", 10], ["dtype", "bfloat16"], ["seed", 610803], ["self_search", false], ["min_recall", 0.999]]}]]}, {"__dict_items__": [["label", "blind_ext_dyn_d512_q64_m65536_k10"], ["params", {"__dict_items__": [["B", 1], ["Q", 64], ["M", 65536], ["D", 512], ["K", 10], ["dtype", "bfloat16"], ["seed", 610922], ["self_search", false], ["min_recall", 0.999]]}]]}]'))

def selected_route(inputs: dict[str, Any]) -> str:
    return parent.selected_route(inputs)

def selected_route_name(inputs: dict[str, Any]) -> str:
    return selected_route(inputs)

def route_info(inputs: dict[str, Any]) -> dict[str, Any]:
    info = dict(parent.route_info(inputs))
    if info.get('route_kind') == 'specialized':
        info['selected_entrypoint'] = MODULE_ENTRYPOINT
    if int(inputs.get('D', -1)) == 63 and info.get('selected_route') == ROUTE_DYNAMIC_D_BREAKTHROUGH_D63_Q128:
        info['requested_label'] = REQUESTED_D63_ALIAS
        info['contract_label'] = CONTRACT_D63_LABEL
    return info

def route_trace_entry(label: str, inputs: dict[str, Any]) -> dict[str, Any]:
    trace = parent.route_trace_entry(label, inputs)
    if trace.get('route_kind') == 'specialized':
        trace['selected_entrypoint'] = MODULE_ENTRYPOINT
    if label == REQUESTED_D63_ALIAS:
        trace['shape_key'] = REQUESTED_D63_ALIAS
        trace['requested_label'] = REQUESTED_D63_ALIAS
        trace['contract_label'] = CONTRACT_D63_LABEL
    return trace

def launch_for_eval(inputs: dict[str, Any]) -> dict[str, Any]:
    return parent.launch_for_eval(inputs)

def knn_search_compile_and_launch_dynamic_d_scalar_breakthrough_0621_r124_f670_d63_alias(*, benchmark: bool=True, shape_labels: str | tuple[str, ...] | list[str] | None=TARGET_LABELS) -> dict[str, Any]:
    from .._dispatch_runtime import evaluate
    result = evaluate(launch_for_eval, shapes=_select_contract_shapes(shape_labels), benchmark=benchmark)
    result['passed'] = bool(result.get('summary', {}).get('all_correct'))
    print(result)
    return result
