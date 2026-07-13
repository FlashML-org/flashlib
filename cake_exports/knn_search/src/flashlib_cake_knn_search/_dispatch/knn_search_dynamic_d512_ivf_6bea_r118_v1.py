"""D512/K64 plus tiny-IVF repair seed for exact BF16 squared-L2 kNN.

Minimum target architecture: sm_80 for the IVF direct route; sm_100a for the
inherited D512/K64 tcgen05 route. This additive generalize-auto-tuning bucket
candidate targets the round-117 D512/IVF repair lane without editing the
production dispatcher. The D512 row delegates to the current 6bea-selected
packed tcgen05 seed; the IVF-like ``B=1,Q=12,M=100,D=64,K=20`` row uses a
single Weave CUDA-core CTA per query and a warp/block winner loop.
"""
from __future__ import annotations
from json import loads as _json_loads
from .._dispatch_runtime import _capture_cuTensorMapEncodeTiled, _decode_capture, _import_dispatch_module, _ir_proxy
from typing import Any
from .._dispatch_runtime import select_named_shapes
from . import knn_search_dynamic_lowd_k64_capacity_9d5c_r117_v1 as parent
IVF_THREADS = 128
IVF_NUM_WARPS = IVF_THREADS // 32
IVF_Q = 12
IVF_M = 100
IVF_D = 64
IVF_K = 20
IVF_WARP_DIST_BYTES = IVF_NUM_WARPS * 4
IVF_WARP_IDX_OFFSET = IVF_WARP_DIST_BYTES
IVF_WARP_THREAD_OFFSET = IVF_WARP_IDX_OFFSET + IVF_NUM_WARPS * 4
IVF_DIRECT_SMEM_BYTES = IVF_WARP_THREAD_OFFSET + IVF_NUM_WARPS * 4
ROUTE_IVF_Q12_M100_D64_K20_DIRECT = '6bea_r118_ivf_q12_m100_d64_k20_direct'
ROUTE_D512_K64_PARENT = parent.ROUTE_D512_K64_D512_PACKED
ROUTE_PARENT = '9d5c_r117_parent'
CONSUMED_SEED = 'weave-evolve-knn-search-6bea-r118-d512-ivf'
REUSED_D512_SEED = parent.CONSUMED_SEED
TARGET_LABELS: tuple[str, ...] = ('blind_ext_dyn_d512_k64_q32_m32768', 'blind_ext_ivf_q12_m100_d64_k20')
TARGET_SHAPES = _decode_capture(_json_loads('[{"__dict_items__": [["label", "blind_ext_dyn_d512_k64_q32_m32768"], ["params", {"__dict_items__": [["B", 1], ["Q", 32], ["M", 32768], ["D", 512], ["K", 64], ["dtype", "bfloat16"], ["seed", 610930], ["self_search", false], ["min_recall", 0.999]]}]]}, {"__dict_items__": [["label", "blind_ext_ivf_q12_m100_d64_k20"], ["params", {"__dict_items__": [["B", 1], ["Q", 12], ["M", 100], ["D", 64], ["K", 20], ["dtype", "bfloat16"], ["seed", 610931], ["self_search", false], ["min_recall", 1.0]]}]]}]'))
_IVF_KERNEL: dict[str, Any] = {}
knn_search_ivf_q12_m100_d64_k20_direct_6bea_r118_v1 = _decode_capture(_json_loads('{"__ir__": "knn_search_ivf_q12_m100_d64_k20_direct_6bea_r118_v1", "arg_keys": ["queries", "database", "out_distances", "out_indices", "B", "Q", "M", "K"], "cluster_dims": [1, 1, 1], "computed_smem_bytes": 128, "constants": [["D_", 64], ["K_MAX_", 20], ["NUM_WARPS_", 4]], "cta_group": 1, "threads": 128}'))
ir = _decode_capture(_json_loads('{"__ir__": "knn_search_ivf_q12_m100_d64_k20_direct_6bea_r118_v1", "arg_keys": ["queries", "database", "out_distances", "out_indices", "B", "Q", "M", "K"], "cluster_dims": [1, 1, 1], "computed_smem_bytes": 128, "constants": [["D_", 64], ["K_MAX_", 20], ["NUM_WARPS_", 4]], "cta_group": 1, "threads": 128}'))

def _use_ivf_direct(inputs: dict[str, Any]) -> bool:
    return int(inputs['B']) == 1 and int(inputs['Q']) == IVF_Q and (int(inputs['M']) == IVF_M) and (int(inputs['D']) == IVF_D) and (int(inputs['K']) == IVF_K) and (not bool(inputs.get('self_search', False))) and (not bool(inputs.get('force_fallback', False)))

def _compile_ivf_kernel():
    return _decode_capture(_json_loads('{"__kernel__": "dispatch_kernel_0108"}'))

def _ensure_ivf_kernel():
    kernel = _IVF_KERNEL.get('direct')
    if kernel is None:
        kernel = _compile_ivf_kernel()
        _IVF_KERNEL['direct'] = kernel
    return kernel

def _launch_ivf_direct(inputs: dict[str, Any]) -> dict[str, Any]:
    kernel = _ensure_ivf_kernel()
    kernel.launch(grid=(int(inputs['B']) * int(inputs['Q']), 1, 1), block=(IVF_THREADS, 1, 1), args=[inputs['queries'], inputs['database'], inputs['out_distances'], inputs['out_indices'], int(inputs['B']), int(inputs['Q']), int(inputs['M']), int(inputs['K'])], shared_mem=IVF_DIRECT_SMEM_BYTES)
    return {'distances': inputs['out_distances'], 'indices': inputs['out_indices']}

def selected_route(inputs: dict[str, Any]) -> str:
    if _use_ivf_direct(inputs):
        return ROUTE_IVF_Q12_M100_D64_K20_DIRECT
    if parent.selected_route(inputs) == parent.ROUTE_D512_K64_D512_PACKED:
        return ROUTE_D512_K64_PARENT
    return ROUTE_PARENT

def selected_route_name(inputs: dict[str, Any]) -> str:
    return selected_route(inputs)

def route_info(inputs: dict[str, Any]) -> dict[str, Any]:
    if _use_ivf_direct(inputs):
        return {'route': ROUTE_IVF_Q12_M100_D64_K20_DIRECT, 'selected_route': ROUTE_IVF_Q12_M100_D64_K20_DIRECT, 'selected_entrypoint': 'loom.examples.weave.knn_search_dynamic_d512_ivf_6bea_r118_v1:launch_for_eval', 'route_kind': 'specialized', 'route_source': 'bucket-specific-seed', 'coverage_class': 'bucket_seed_ivf_q12_m100_d64_k20', 'classification': 'seed-produced', 'coverage_only': False, 'production_policy': 'weave_only', 'external_fallback': None, 'guard_id': '6bea_r118_ivf_q12_m100_d64_k20', 'forced_fallback': False, 'selected_seed': CONSUMED_SEED, 'missing_weave_route': False}
    info = dict(parent.route_info(inputs))
    if info.get('selected_route') == parent.ROUTE_D512_K64_D512_PACKED:
        info['selected_entrypoint'] = 'loom.examples.weave.knn_search_dynamic_d512_ivf_6bea_r118_v1:launch_for_eval'
        info['selected_seed'] = REUSED_D512_SEED
        info['reused_seed'] = REUSED_D512_SEED
    return info

def route_trace_entry(label: str, inputs: dict[str, Any]) -> dict[str, Any]:
    return {'label': label, 'shape_key': label, 'B': int(inputs['B']), 'Q': int(inputs['Q']), 'M': int(inputs['M']), 'D': int(inputs['D']), 'K': int(inputs['K']), 'self_search': bool(inputs.get('self_search', False)), **route_info(inputs)}
SHAPE_DISPATCH_REGISTRY: tuple[dict[str, Any], ...] = ({'overlay': '6bea_r118_ivf_q12_m100_d64_k20_direct', 'shape_key': 'blind_ext_ivf_q12_m100_d64_k20', 'labels': ('blind_ext_ivf_q12_m100_d64_k20',), 'guard': 'B == 1 and Q == 12 and M == 100 and D == 64 and K == 20 and not self_search and not forced_fallback', 'route': ROUTE_IVF_Q12_M100_D64_K20_DIRECT, 'entrypoint': 'loom.examples.weave.knn_search_dynamic_d512_ivf_6bea_r118_v1:launch_for_eval', 'selected_seed': CONSUMED_SEED, 'source_task': CONSUMED_SEED, 'coverage_class': 'bucket_seed_ivf_q12_m100_d64_k20', 'workflow_mode': 'generalize_auto_tuning', 'auto_tuning_stage': 'bucket-kernel'}, {'overlay': '6bea_r118_inherited_d512_q32_m32768_k64', 'shape_key': 'blind_ext_dyn_d512_k64_q32_m32768', 'labels': ('blind_ext_dyn_d512_k64_q32_m32768',), 'guard': 'B == 1 and Q == 32 and M == 32768 and D == 512 and K == 64 and not self_search and not forced_fallback and tcgen05_capable_arch', 'route': ROUTE_D512_K64_PARENT, 'entrypoint': 'loom.examples.weave.knn_search_dynamic_d512_ivf_6bea_r118_v1:launch_for_eval', 'selected_seed': REUSED_D512_SEED, 'source_task': REUSED_D512_SEED, 'coverage_class': 'bucket_seed_dynamic_d512_k64_inherited', 'workflow_mode': 'generalize_auto_tuning', 'auto_tuning_stage': 'bucket-kernel'})

def launch_for_eval(inputs: dict[str, Any]) -> dict[str, Any]:
    if _use_ivf_direct(inputs):
        return _launch_ivf_direct(inputs)
    return parent.launch_for_eval(inputs)

def knn_search_compile_and_launch_dynamic_d512_ivf_6bea_r118(*, benchmark: bool=True, shapes: list[dict[str, Any]] | None=None) -> dict[str, Any]:
    from .._dispatch_runtime import evaluate
    selected = TARGET_SHAPES if shapes is None else shapes
    result = evaluate(launch_for_eval, shapes=selected, benchmark=benchmark)
    result['passed'] = bool(result.get('summary', {}).get('all_correct'))
    print(result)
    return result
