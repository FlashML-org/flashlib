"""Legacy D33/D80/D160 K10 kNN cohort with stride-safe direct staging.

Minimum target architecture: sm_100a.  This additive capability seed owns the
exact residual-full198 rows ``D33/Q128/M131072``, ``D80/Q256/M65536``, and
``D160/Q64/M131072``.  D33 retains scalar direct-stride loads because its
66-byte BF16 row stride is not safe for 32-byte vector loads.  D80 and D160
instead specialize the aligned 16-BF16 direct-vector producer used by the
existing D768/D1024 seed.  Every route performs kernel-local zero padding to a
K128 tcgen05 tile, feeds the split-M exact-top10 merge, and writes caller-owned
FP32 distances and INT32 indices.

The closest structural parent is
``knn_search_floor13_dynamic_d_k10_0622_5f25_v1``.  This module preserves that
parent as its fallback and changes only aligned D80/D160 operand staging; it
does not edit the production dispatcher or registry.
"""
from __future__ import annotations
from json import loads as _json_loads
from .._dispatch_runtime import _capture_cuTensorMapEncodeTiled, _decode_capture, _import_dispatch_module, _ir_proxy
from typing import Any
from .. import _dispatch_runtime as contract
from . import knn_search_dynamic_d768d1024_q32q16_tcgen05_0618_9286_v1 as aligned
from . import knn_search_dynamic_d_tiny_q128_m65536_tcgen05_0618_c8b9_v1 as odd_d
from . import knn_search_floor13_dynamic_d_k10_0622_5f25_v1 as parent
THREADS = aligned.THREADS
MERGE_THREADS = aligned.MERGE_THREADS
BLOCK_Q = aligned.BLOCK_Q
BLOCK_M = aligned.BLOCK_M
K_MAX = aligned.K_MAX
ir = _decode_capture(_json_loads('{"__ir__": "knn_search_dynamic_d768d1024_q32q16_tcgen05_partial_0618_9286_v1", "arg_keys": ["queries", "database", "partial_distances", "partial_indices", "B", "Q", "M", "split_m", "num_q_tiles", "total_m_tiles", "tiles_per_split"], "cluster_dims": [1, 1, 1], "computed_smem_bytes": 143104, "constants": [["K_MAX_", 10], ["D_ORIG_", 1024], ["NUM_D_PASSES_", 8], ["Q_NORM_PARTS_", 64]], "cta_group": 1, "threads": 640}'))
partial_ir = _decode_capture(_json_loads('{"__ir__": "knn_search_dynamic_d768d1024_q32q16_tcgen05_partial_0618_9286_v1", "arg_keys": ["queries", "database", "partial_distances", "partial_indices", "B", "Q", "M", "split_m", "num_q_tiles", "total_m_tiles", "tiles_per_split"], "cluster_dims": [1, 1, 1], "computed_smem_bytes": 143104, "constants": [["K_MAX_", 10], ["D_ORIG_", 1024], ["NUM_D_PASSES_", 8], ["Q_NORM_PARTS_", 64]], "cta_group": 1, "threads": 640}'))
merge_ir = _decode_capture(_json_loads('{"__ir__": "knn_search_mma_split_merge_q128_const148_v1", "arg_keys": ["partial_distances", "partial_indices", "out_distances", "out_indices", "B", "Q", "K", "split_m", "num_q_tiles"], "cluster_dims": [1, 1, 1], "computed_smem_bytes": 0, "constants": [["K_MAX_", 10]], "cta_group": 1, "threads": 32}'))
odd_d_partial_ir = _decode_capture(_json_loads('{"__ir__": "knn_search_dynamic_d_tiny_q128_m65536_tcgen05_partial_0618_c8b9_v1", "arg_keys": ["queries", "database", "partial_distances", "partial_indices", "B", "Q", "M", "split_m", "num_q_tiles", "total_m_tiles", "tiles_per_split"], "cluster_dims": [1, 1, 1], "computed_smem_bytes": 112384, "constants": [["K_MAX_", 10], ["D_TOTAL_", 3], ["NUM_D_PASSES_", 1], ["Q_NORM_PARTS_", 1]], "cta_group": 1, "threads": 640}'))
odd_d_merge_ir = _decode_capture(_json_loads('{"__ir__": "knn_search_mma_split_merge_q128_const148_v1", "arg_keys": ["partial_distances", "partial_indices", "out_distances", "out_indices", "B", "Q", "K", "split_m", "num_q_tiles"], "cluster_dims": [1, 1, 1], "computed_smem_bytes": 0, "constants": [["K_MAX_", 10]], "cta_group": 1, "threads": 32}'))
TASK_ID = 'weave-evolve-knn-search-dispatcher-residual-full198-cf73'
LANE_ID = 'legacy_dynamic_d_capability_deferred'
BUCKET_ID = 'legacy_dynamic_d/blind_0622_dyn_d80_q256_m65536_k10'
TARGET_SHAPE = 'blind_0622_dyn_d80_q256_m65536_k10'
TARGET_LABELS: tuple[str, ...] = ('blind_0622_dyn_d33_q128_m131072_k10', TARGET_SHAPE, 'blind_0622_dyn_d160_q64_m131072_k10')
TARGET_SHAPES: list[dict[str, Any]] = [shape for shape in contract.CANONICAL_SHAPES if shape['label'] in TARGET_LABELS]
if tuple((shape['label'] for shape in TARGET_SHAPES)) != TARGET_LABELS:
    raise RuntimeError('legacy dynamic-D target labels do not match the residual-full198 contract order')
ROUTE_D33 = 'cf73_legacy_dyn_d33_scalar_directstride_tcgen05'
ROUTE_D80 = 'cf73_legacy_dyn_d80_alignedvec_directstride_tcgen05'
ROUTE_D160 = 'cf73_legacy_dyn_d160_alignedvec_directstride_tcgen05'
ROUTE_PARENT_FALLBACK = '5f25_floor13_dynamic_d_parent'
ENTRYPOINT = 'loom.examples.weave.knn_search_legacy_dynamic_d_alignedvec_0707_cf73_v1:launch_for_eval'
_D33_ENTRY: dict[str, Any] = {'shape_key': TARGET_LABELS[0], 'shape': (1, 128, 131072, 33, 10, False), 'guard': 'B == 1 and Q == 128 and M == 131072 and D == 33 and K == 10 and not self_search and not forced_fallback and arch in {sm_100a,sm_103a}', 'route': ROUTE_D33, 'source_entrypoint': 'loom.examples.weave.knn_search_dynamic_d_tiny_q128_m65536_tcgen05_0618_c8b9_v1:_launch_tiny_dynamic_d_mma', 'selected_seed': 'cf73_legacy_dyn_d33_retained_scalar_directstride', 'coverage_class': 'bucket_seed_legacy_dynamic_d33_q128_m131072_k10', 'staging': 'scalar_bf16_direct_stride_with_kernel_local_k128_zero_fill', 'padding_tag': 'kernel_local_d33_to_d128', 'structural_delta': 'retained_odd_stride_parent_inside_hybrid_cohort'}
_D80_ENTRY: dict[str, Any] = {'shape_key': TARGET_LABELS[1], 'shape': (1, 256, 65536, 80, 10, False), 'guard': 'B == 1 and Q == 256 and M == 65536 and D == 80 and K == 10 and D % 16 == 0 and not self_search and not forced_fallback and arch in {sm_100a,sm_103a}', 'route': ROUTE_D80, 'source_entrypoint': 'loom.examples.weave.knn_search_dynamic_d768d1024_q32q16_tcgen05_0618_9286_v1:_launch_high_dynamic_d_tcgen05', 'selected_seed': 'cf73_legacy_dyn_d80_alignedvec_directstride', 'coverage_class': 'bucket_seed_legacy_dynamic_d80_q256_m65536_k10', 'staging': 'aligned_16xbf16_direct_vector_load_with_kernel_local_k128_zero_fill', 'padding_tag': 'kernel_local_d80_to_d128', 'structural_delta': 'scalar_to_aligned_vector_direct_stride_staging'}
_D160_ENTRY: dict[str, Any] = {'shape_key': TARGET_LABELS[2], 'shape': (1, 64, 131072, 160, 10, False), 'guard': 'B == 1 and Q == 64 and M == 131072 and D == 160 and K == 10 and D % 16 == 0 and not self_search and not forced_fallback and arch in {sm_100a,sm_103a}', 'route': ROUTE_D160, 'source_entrypoint': 'loom.examples.weave.knn_search_dynamic_d768d1024_q32q16_tcgen05_0618_9286_v1:_launch_high_dynamic_d_tcgen05', 'selected_seed': 'cf73_legacy_dyn_d160_alignedvec_directstride', 'coverage_class': 'bucket_seed_legacy_dynamic_d160_q64_m131072_k10', 'staging': 'aligned_16xbf16_direct_vector_load_with_kernel_local_k128_tail_zero_fill', 'padding_tag': 'kernel_local_d160_to_d256', 'structural_delta': 'scalar_to_aligned_vector_direct_stride_staging'}
_ENTRIES: tuple[dict[str, Any], ...] = (_D33_ENTRY, _D80_ENTRY, _D160_ENTRY)
SHAPE_DISPATCH_REGISTRY = _decode_capture(_json_loads('{"__tuple__": [{"__dict_items__": [["overlay", "cf73_legacy_dyn_d33_retained_scalar_directstride"], ["shape_key", "blind_0622_dyn_d33_q128_m131072_k10"], ["guard", "B == 1 and Q == 128 and M == 131072 and D == 33 and K == 10 and not self_search and not forced_fallback and arch in {sm_100a,sm_103a}"], ["route", "cf73_legacy_dyn_d33_scalar_directstride_tcgen05"], ["entrypoint", "loom.examples.weave.knn_search_legacy_dynamic_d_alignedvec_0707_cf73_v1:launch_for_eval"], ["source_entrypoint", "loom.examples.weave.knn_search_dynamic_d_tiny_q128_m65536_tcgen05_0618_c8b9_v1:_launch_tiny_dynamic_d_mma"], ["selected_seed", "cf73_legacy_dyn_d33_retained_scalar_directstride"], ["coverage_class", "bucket_seed_legacy_dynamic_d33_q128_m131072_k10"], ["workflow_mode", "generalize_auto_tuning"], ["auto_tuning_stage", "bucket-kernel"], ["evaluation_stage", "capability"], ["bucket_id", "legacy_dynamic_d/blind_0622_dyn_d80_q256_m65536_k10"], ["padding_tag", "kernel_local_d33_to_d128"], ["staging", "scalar_bf16_direct_stride_with_kernel_local_k128_zero_fill"], ["structural_delta", "retained_odd_stride_parent_inside_hybrid_cohort"]]}, {"__dict_items__": [["overlay", "cf73_legacy_dyn_d80_alignedvec_directstride"], ["shape_key", "blind_0622_dyn_d80_q256_m65536_k10"], ["guard", "B == 1 and Q == 256 and M == 65536 and D == 80 and K == 10 and D % 16 == 0 and not self_search and not forced_fallback and arch in {sm_100a,sm_103a}"], ["route", "cf73_legacy_dyn_d80_alignedvec_directstride_tcgen05"], ["entrypoint", "loom.examples.weave.knn_search_legacy_dynamic_d_alignedvec_0707_cf73_v1:launch_for_eval"], ["source_entrypoint", "loom.examples.weave.knn_search_dynamic_d768d1024_q32q16_tcgen05_0618_9286_v1:_launch_high_dynamic_d_tcgen05"], ["selected_seed", "cf73_legacy_dyn_d80_alignedvec_directstride"], ["coverage_class", "bucket_seed_legacy_dynamic_d80_q256_m65536_k10"], ["workflow_mode", "generalize_auto_tuning"], ["auto_tuning_stage", "bucket-kernel"], ["evaluation_stage", "capability"], ["bucket_id", "legacy_dynamic_d/blind_0622_dyn_d80_q256_m65536_k10"], ["padding_tag", "kernel_local_d80_to_d128"], ["staging", "aligned_16xbf16_direct_vector_load_with_kernel_local_k128_zero_fill"], ["structural_delta", "scalar_to_aligned_vector_direct_stride_staging"]]}, {"__dict_items__": [["overlay", "cf73_legacy_dyn_d160_alignedvec_directstride"], ["shape_key", "blind_0622_dyn_d160_q64_m131072_k10"], ["guard", "B == 1 and Q == 64 and M == 131072 and D == 160 and K == 10 and D % 16 == 0 and not self_search and not forced_fallback and arch in {sm_100a,sm_103a}"], ["route", "cf73_legacy_dyn_d160_alignedvec_directstride_tcgen05"], ["entrypoint", "loom.examples.weave.knn_search_legacy_dynamic_d_alignedvec_0707_cf73_v1:launch_for_eval"], ["source_entrypoint", "loom.examples.weave.knn_search_dynamic_d768d1024_q32q16_tcgen05_0618_9286_v1:_launch_high_dynamic_d_tcgen05"], ["selected_seed", "cf73_legacy_dyn_d160_alignedvec_directstride"], ["coverage_class", "bucket_seed_legacy_dynamic_d160_q64_m131072_k10"], ["workflow_mode", "generalize_auto_tuning"], ["auto_tuning_stage", "bucket-kernel"], ["evaluation_stage", "capability"], ["bucket_id", "legacy_dynamic_d/blind_0622_dyn_d80_q256_m65536_k10"], ["padding_tag", "kernel_local_d160_to_d256"], ["staging", "aligned_16xbf16_direct_vector_load_with_kernel_local_k128_tail_zero_fill"], ["structural_delta", "scalar_to_aligned_vector_direct_stride_staging"]]}]}'))

def _shape_key(inputs: dict[str, Any]) -> tuple[int, int, int, int, int, bool]:
    return (int(inputs.get('B', 1)), int(inputs['Q']), int(inputs['M']), int(inputs['D']), int(inputs['K']), bool(inputs.get('self_search', False)))

def _active_entry(inputs: dict[str, Any]) -> dict[str, Any] | None:
    if bool(inputs.get('force_fallback', False)):
        return None
    shape = _shape_key(inputs)
    for entry in _ENTRIES:
        if shape == entry['shape'] and aligned.mma._tcgen05_capable_arch():
            return entry
    return None

def selected_route(inputs: dict[str, Any]) -> str:
    entry = _active_entry(inputs)
    if entry is not None:
        return str(entry['route'])
    return parent.selected_route(inputs)

def selected_route_name(inputs: dict[str, Any]) -> str:
    return selected_route(inputs)

def _parent_route(inputs: dict[str, Any]) -> str:
    info = parent.route_info(inputs)
    return str(info.get('route') or info.get('selected_route') or parent.selected_route(inputs))

def route_info(inputs: dict[str, Any]) -> dict[str, Any]:
    entry = _active_entry(inputs)
    if entry is None:
        info = dict(parent.route_info(inputs))
        info['guard_order'] = [row['shape_key'] for row in SHAPE_DISPATCH_REGISTRY]
        return info
    return {'route': entry['route'], 'selected_route': entry['route'], 'selected_entrypoint': ENTRYPOINT, 'source_entrypoint': entry['source_entrypoint'], 'parent_route': _parent_route(inputs), 'replaced_route': _parent_route(inputs), 'route_kind': 'specialized', 'route_source': 'shape-specific-seed', 'coverage_class': entry['coverage_class'], 'classification': 'seed-produced', 'coverage_only': False, 'production_policy': 'weave_only', 'external_fallback': None, 'guard_order': [row['shape_key'] for row in SHAPE_DISPATCH_REGISTRY], 'guard_id': entry['shape_key'], 'selected_guard': entry['guard'], 'fallback': ROUTE_PARENT_FALLBACK, 'missing_weave_route': False, 'forced_fallback': False, 'selected_seed': entry['selected_seed'], 'arch_requirement': 'sm_100a', 'staging': entry['staging'], 'structural_delta': entry['structural_delta'], 'uses_kernel_padding': True, 'uses_materialized_padding': False, 'padding_overhead_timed': True, 'padding_tag': entry['padding_tag'], 'workspace_policy': 'shape-keyed cached split-M FP32-distance/INT32-index scratch', 'workspace_reuse': True, 'caller_owned_output_writes': True}

def route_trace_entry(label: str, inputs: dict[str, Any]) -> dict[str, Any]:
    return {'label': label, 'shape_key': label, 'B': int(inputs['B']), 'Q': int(inputs['Q']), 'M': int(inputs['M']), 'D': int(inputs['D']), 'K': int(inputs['K']), 'self_search': bool(inputs.get('self_search', False)), **route_info(inputs)}

def launch_for_eval(inputs: dict[str, Any]) -> dict[str, Any]:
    entry = _active_entry(inputs)
    if entry is _D33_ENTRY:
        return odd_d._launch_tiny_dynamic_d_mma(inputs)
    if entry is _D80_ENTRY or entry is _D160_ENTRY:
        return aligned._launch_high_dynamic_d_tcgen05(inputs)
    return parent.launch_for_eval(inputs)

def knn_search_compile_and_launch_legacy_dynamic_d_alignedvec_0707_cf73(*, benchmark: bool=True) -> dict[str, Any]:
    result = contract.evaluate(launch_for_eval, shapes=TARGET_SHAPES, benchmark=benchmark)
    result['passed'] = bool(result.get('summary', {}).get('all_correct'))
    print(result)
    return result
