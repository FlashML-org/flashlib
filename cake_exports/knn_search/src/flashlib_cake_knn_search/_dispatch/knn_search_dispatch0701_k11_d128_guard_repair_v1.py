"""Guarded K11 and D128-tail repair over the exported 8ae1 dispatcher.

Minimum target architecture: sm_100a.  This wrapper leaves all seed schedules
unchanged.  It adapts the validated Q4096/M20000/D128/K64 Weave seed to K11
by copying its exact sorted prefix into the caller-owned outputs, uses an
explicit +infinity M-tail pack before the existing K10 Weave MMA seed, and
sends the Q tail to that seed directly.  Forced fallback remains owned by the
inherited exported dispatcher.
"""
from __future__ import annotations
from json import loads as _json_loads
from .._dispatch_runtime import _capture_cuTensorMapEncodeTiled, _decode_capture, _import_dispatch_module, _ir_proxy
from typing import Any
import torch
from . import knn_search_dispatch0618_d128_k10_c08b_f828_synth_9d5c_v1 as d128_restored
from . import knn_search_dispatch0622_current_portfolio_e472_v1 as floor13_restored
from . import knn_search_dispatch0701_8ae1_q4096_exported_vertical_slice_consumption_v1 as base
from . import knn_search_ext_k_capacity_0618_28ec_v1 as ext_k
from . import knn_search_floor13_k64_fast_specials_0622_5f25_v1 as floor13_k64
from . import knn_search_floor13_k80_prefix8_0622_f3ce_v1 as floor13_k80
from . import knn_search_k64_mma_capacity_0611_r22_4e96_v1 as k64_q128_capacity
from . import knn_search_k64_q4096m20000_prefixcert_fused_0615_576b_v1 as k64_seed
from . import knn_search_mma_split_v1 as mma_seed
from . import knn_search_q4096_split4_0611_r14_4e2c_v1 as mtail_seed
from . import knn_search_self_k10_q1024_split8_0622_f4bc_v1 as self_q1024_restored
ENTRYPOINT = 'loom.examples.weave.knn_search_dispatch0701_k11_d128_guard_repair_v1:launch_for_eval'
K11_SEED = 'round20-576b-q4096-m20000-d128-k64-prefix-adapter'
MTAIL_SEED = 'round14-4e2c-q4096-split4-mtail'
QTAIL_SEED = 'knn-search-mma-split-v1-qtail'
K64_ENTRYPOINT = 'loom.examples.weave.knn_search_k64_q4096m20000_prefixcert_fused_0615_576b_v1:launch_for_eval'
K64_Q128_ENTRYPOINT = 'loom.examples.weave.knn_search_k64_mma_capacity_0611_r22_4e96_v1:launch_for_eval'
MMA_ENTRYPOINT = 'loom.examples.weave.knn_search_mma_split_v1:launch_for_eval'
MTAIL_ENTRYPOINT = 'loom.examples.weave.knn_search_q4096_split4_0611_r14_4e2c_v1:launch_for_eval'
K11_GUARD_ID = 'q4096_m20000_d128_k11_k64_prefix_adapter'
MTAIL_GUARD_ID = 'q4096_m19999_d128_k10_infinity_mtail'
QTAIL_GUARD_ID = 'q4095_m20001_d128_k10_direct_qtail'
K64_REPAIR_ROUTE = k64_seed.ROUTE_Q4096_M20000_K64_PREFIX6CERT_FUSED
K64_Q128_CAPACITY_ROUTE = 'round22_q128_m131072_d128_k64_capacity'
_K11 = (1, 4096, 20000, 128, 11)
_MTAIL = (1, 4096, 19999, 128, 10)
_QTAIL = (1, 4095, 20001, 128, 10)
_K64_REPAIR = (1, 4096, 20000, 128, 64)
_K64_Q128_CAPACITY = (1, 128, 131072, 128, 64)
_D128_RESTORED_KEYS = frozenset({(1, 513, 98304, 128, 10, False), (1, 3072, 49152, 128, 10, False), (1, 3072, 3072, 128, 10, True)})
_D128_RESTORED_ROUTES = frozenset({d128_restored.ROUTE_Q513_M98304_F828_SPLIT32, d128_restored.ROUTE_Q3072_M49152_C08B, d128_restored.ROUTE_SELF_Q3072_M3072_C08B})
_FLOOR13_RESTORED_KEYS = frozenset({(1, 96, 131072, 128, 64, False), (1, 384, 98304, 128, 64, False), (1, 128, 65536, 128, 80, False), (1, 4096, 32768, 128, 80, False), (2, 256, 98304, 128, 64, False)})
_FLOOR13_RESTORED_ROUTES = frozenset({floor13_restored.ROUTE_Q96_K64_EXACT, floor13_restored.ROUTE_Q384_K64_EXACT, floor13_restored.ROUTE_FLOOR13_K80_PREFIX8, floor13_restored.ROUTE_FLOOR13_K64_PREFIX8})
_SELF_Q1024_RESTORED_KEY = (1, 1024, 1024, 128, 10, True)
_K11_COPY_THREADS = 256
_K11_COPY_WIDTH = 11
_K64_SEED_WIDTH = 64
_K11_COPY_KERNELS: dict[str, Any] = {}
ir = _decode_capture(_json_loads('{"__ir__": "knn_search_mma_split_partial_v1", "arg_keys": ["queries", "database", "partial_distances", "partial_indices", "B", "Q", "M", "split_m", "num_q_tiles", "total_m_tiles", "tiles_per_split"], "cluster_dims": [1, 1, 1], "computed_smem_bytes": 108800, "constants": [["K_MAX_", 10], ["EXPOSE_COL_COHORTS", 0], ["FULL_M_TILES", 0]], "cta_group": 1, "threads": 640}'))
SHAPE_DISPATCH_REGISTRY = (*floor13_k64.SHAPE_DISPATCH_REGISTRY[:2], floor13_k64.broad_k64.SHAPE_DISPATCH_REGISTRY[0], floor13_k80.SHAPE_DISPATCH_REGISTRY[0], {'shape_key': 'q128_m131072_d128_k64_capacity', 'route': K64_Q128_CAPACITY_ROUTE, 'entrypoint': K64_Q128_ENTRYPOINT}, {'shape_key': K11_GUARD_ID, 'route': 'k64_prefix_to_k11', 'entrypoint': K64_ENTRYPOINT}, {'shape_key': MTAIL_GUARD_ID, 'route': 'split4_k10_mtail', 'entrypoint': MTAIL_ENTRYPOINT}, {'shape_key': QTAIL_GUARD_ID, 'route': 'mma_k10_direct_qtail', 'entrypoint': MMA_ENTRYPOINT}, *base.SHAPE_DISPATCH_REGISTRY)
knn_search_k64_prefix_to_k11_copy_0705_v1 = _decode_capture(_json_loads('{"__ir__": "knn_search_k64_prefix_to_k11_copy_0705_v1", "arg_keys": ["source_distances", "source_indices", "out_distances", "out_indices", "B", "Q"], "cluster_dims": [1, 1, 1], "computed_smem_bytes": 0, "constants": [["SOURCE_K_", 64], ["OUTPUT_K_", 11]], "cta_group": 1, "threads": 256}'))

def _compile_k11_copy_kernel() -> Any:
    return _decode_capture(_json_loads('{"__kernel__": "dispatch_kernel_0291"}'))

def _key(inputs: dict[str, Any]) -> tuple[int, int, int, int, int]:
    return tuple((int(inputs[name]) for name in ('B', 'Q', 'M', 'D', 'K')))

def _eligible(inputs: dict[str, Any], key: tuple[int, int, int, int, int]) -> bool:
    return _key(inputs) == key and (not bool(inputs.get('self_search', False))) and (not bool(inputs.get('force_fallback', False)))

def _route(inputs: dict[str, Any]) -> str | None:
    if _eligible(inputs, _K11):
        return 'k64_prefix_to_k11'
    if _eligible(inputs, _MTAIL):
        return 'split4_k10_mtail'
    if _eligible(inputs, _QTAIL):
        return 'mma_k10_direct_qtail'
    return None

def _use_k64_repair(inputs: dict[str, Any]) -> bool:
    return _eligible(inputs, _K64_REPAIR)

def _use_k64_q128_capacity(inputs: dict[str, Any]) -> bool:
    return _eligible(inputs, _K64_Q128_CAPACITY)

def _use_floor13_k64(inputs: dict[str, Any]) -> bool:
    return floor13_k64.broad_k64.supports_floor13_k64_prefix8(inputs)

def _use_floor13_k80(inputs: dict[str, Any]) -> bool:
    return floor13_k80.supports_floor13_k80_prefix8(inputs)

def _route_key_with_self(inputs: dict[str, Any]) -> tuple[int, int, int, int, int, bool]:
    return (*(int(inputs[name]) for name in ('B', 'Q', 'M', 'D', 'K')), bool(inputs.get('self_search', False)))

def _restored_d128_route(inputs: dict[str, Any]) -> str | None:
    if _route_key_with_self(inputs) not in _D128_RESTORED_KEYS:
        return None
    route = d128_restored.selected_route(inputs)
    return route if route in _D128_RESTORED_ROUTES else None

def _restored_floor13_route(inputs: dict[str, Any]) -> str | None:
    if bool(inputs.get('force_fallback', False)) or _route_key_with_self(inputs) not in _FLOOR13_RESTORED_KEYS:
        return None
    route = floor13_restored.selected_route(inputs)
    return route if route in _FLOOR13_RESTORED_ROUTES else None

def _restored_self_q1024_route(inputs: dict[str, Any]) -> str | None:
    if _route_key_with_self(inputs) != _SELF_Q1024_RESTORED_KEY:
        return None
    route = self_q1024_restored.selected_route(inputs)
    return route if route == self_q1024_restored.ROUTE_SELF_K10_Q1024_SPLIT8 else None

def _use_ext_k(inputs: dict[str, Any]) -> bool:
    return bool(ext_k._use_q128_m131072_k40(inputs) or ext_k._use_q128_m65536_k56(inputs) or ext_k._use_q4096_m49152_k64(inputs))

def selected_route(inputs: dict[str, Any]) -> str:
    restored_route = _restored_d128_route(inputs) or _restored_floor13_route(inputs) or _restored_self_q1024_route(inputs)
    if restored_route is not None:
        return restored_route
    if _use_floor13_k64(inputs):
        return floor13_k64.selected_route(inputs)
    if _use_floor13_k80(inputs):
        return floor13_k80.selected_route(inputs)
    if _use_k64_q128_capacity(inputs):
        return K64_Q128_CAPACITY_ROUTE
    if _use_k64_repair(inputs):
        return K64_REPAIR_ROUTE
    if _use_ext_k(inputs):
        return ext_k.selected_route(inputs)
    return _route(inputs) or base.selected_route(inputs)

def selected_route_name(inputs: dict[str, Any]) -> str:
    return selected_route(inputs)

def route_info(inputs: dict[str, Any]) -> dict[str, Any]:
    if _restored_d128_route(inputs) is not None:
        info = dict(d128_restored.route_info(inputs))
        info['dispatcher_entrypoint'] = ENTRYPOINT
        return info
    if _restored_floor13_route(inputs) is not None:
        info = dict(floor13_restored.route_info(inputs))
        info['dispatcher_entrypoint'] = ENTRYPOINT
        return info
    if _restored_self_q1024_route(inputs) is not None:
        info = dict(self_q1024_restored.route_info(inputs))
        info['dispatcher_entrypoint'] = ENTRYPOINT
        return info
    if _use_floor13_k64(inputs):
        info = dict(floor13_k64.route_info(inputs))
        info['dispatcher_entrypoint'] = ENTRYPOINT
        return info
    if _use_floor13_k80(inputs):
        info = dict(floor13_k80.route_info(inputs))
        info['dispatcher_entrypoint'] = ENTRYPOINT
        return info
    if _use_k64_q128_capacity(inputs):
        return {'route': K64_Q128_CAPACITY_ROUTE, 'selected_route': K64_Q128_CAPACITY_ROUTE, 'selected_entrypoint': K64_Q128_ENTRYPOINT, 'selected_seed': 'round22-q128-k64-capacity', 'expected_seed': 'round22-q128-k64-capacity', 'guard_id': 'q128_m131072_d128_k64_capacity', 'guard_condition': 'B=1,Q=128,M=131072,D=128,K=64,nonself,not forced', 'selected_guard': 'B=1,Q=128,M=131072,D=128,K=64,nonself,not forced', 'route_kind': 'specialized', 'route_source': 'shape-specific-seed', 'classification': 'seed-consumed', 'coverage_only': False, 'production_policy': 'weave_only', 'external_fallback': None, 'dispatcher_entrypoint': ENTRYPOINT, 'guard_order': [entry['shape_key'] for entry in SHAPE_DISPATCH_REGISTRY]}
    if _use_k64_repair(inputs):
        info = dict(k64_seed.route_info(inputs))
        info['dispatcher_entrypoint'] = ENTRYPOINT
        return info
    if _use_ext_k(inputs):
        info = dict(ext_k.route_info(inputs))
        info['dispatcher_entrypoint'] = ENTRYPOINT
        return info
    route = _route(inputs)
    if route is None:
        info = dict(base.route_info(inputs))
        info.update({'dispatcher_entrypoint': ENTRYPOINT, 'guard_order': [entry['shape_key'] for entry in SHAPE_DISPATCH_REGISTRY]})
        return info
    guard_id, seed_id, entrypoint, condition = {'k64_prefix_to_k11': (K11_GUARD_ID, K11_SEED, K64_ENTRYPOINT, 'B=1,Q=4096,M=20000,D=128,K=11,nonself,not forced'), 'split4_k10_mtail': (MTAIL_GUARD_ID, MTAIL_SEED, MTAIL_ENTRYPOINT, 'B=1,Q=4096,M=19999,D=128,K=10,nonself,not forced'), 'mma_k10_direct_qtail': (QTAIL_GUARD_ID, QTAIL_SEED, MMA_ENTRYPOINT, 'B=1,Q=4095,M=20001,D=128,K=10,nonself,not forced')}[route]
    return {'route': route, 'selected_route': route, 'selected_entrypoint': entrypoint, 'selected_seed': seed_id, 'expected_seed': seed_id, 'guard_id': guard_id, 'guard_condition': condition, 'selected_guard': condition, 'route_kind': 'specialized', 'route_source': 'generated-variant', 'classification': 'seed-consumed', 'coverage_only': False, 'production_policy': 'weave_only', 'external_fallback': None, 'dispatcher_entrypoint': ENTRYPOINT, 'guard_order': [entry['shape_key'] for entry in SHAPE_DISPATCH_REGISTRY]}

def route_trace_entry(label: str, inputs: dict[str, Any]) -> dict[str, Any]:
    return {'label': label, 'shape_key': label, **route_info(inputs)}

def _launch_k11_prefix(inputs: dict[str, Any]) -> dict[str, Any]:
    if not _K11_COPY_KERNELS:
        _K11_COPY_KERNELS['copy'] = _compile_k11_copy_kernel()
    seed_inputs = dict(inputs)
    seed_inputs['K'] = _K64_SEED_WIDTH
    seed_inputs['out_distances'] = torch.empty((*inputs['out_distances'].shape[:-1], _K64_SEED_WIDTH), dtype=torch.float32, device=inputs['queries'].device)
    seed_inputs['out_indices'] = torch.empty((*inputs['out_indices'].shape[:-1], _K64_SEED_WIDTH), dtype=torch.int32, device=inputs['queries'].device)
    outputs = k64_seed.launch_for_eval(seed_inputs)
    out_distances = inputs['out_distances']
    out_indices = inputs['out_indices']
    count = int(inputs['B']) * int(inputs['Q']) * _K11_COPY_WIDTH
    _K11_COPY_KERNELS['copy'].launch(grid=((count + _K11_COPY_THREADS - 1) // _K11_COPY_THREADS, 1, 1), block=(_K11_COPY_THREADS, 1, 1), args=[outputs['distances'], outputs['indices'], out_distances, out_indices, int(inputs['B']), int(inputs['Q'])])
    return {'distances': out_distances, 'indices': out_indices}

def launch_for_eval(inputs: dict[str, Any]) -> dict[str, Any]:
    if _restored_d128_route(inputs) is not None:
        return d128_restored.launch_for_eval(inputs)
    if _restored_floor13_route(inputs) is not None:
        return floor13_restored.launch_for_eval(inputs)
    if _restored_self_q1024_route(inputs) is not None:
        return self_q1024_restored.launch_for_eval(inputs)
    if _use_floor13_k64(inputs):
        return floor13_k64.launch_for_eval(inputs)
    if _use_floor13_k80(inputs):
        return floor13_k80.launch_for_eval(inputs)
    if _use_k64_q128_capacity(inputs):
        return k64_q128_capacity.launch_for_eval(inputs)
    if _use_k64_repair(inputs):
        return k64_seed.launch_for_eval(inputs)
    if _use_ext_k(inputs):
        return ext_k.launch_for_eval(inputs)
    route = _route(inputs)
    if route == 'k64_prefix_to_k11':
        return _launch_k11_prefix(inputs)
    if route == 'split4_k10_mtail':
        return mtail_seed.launch_for_eval(inputs)
    if route == 'mma_k10_direct_qtail':
        return mma_seed.launch_for_eval(inputs)
    return base.launch_for_eval(inputs)
