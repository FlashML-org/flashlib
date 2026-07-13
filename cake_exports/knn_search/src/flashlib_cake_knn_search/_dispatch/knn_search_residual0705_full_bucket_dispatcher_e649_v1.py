"""Exact-guard residual-kNN portfolio over the 084a and 6675 bases.

Minimum target architecture: sm_100a.  The three leading guards consume the
rank-admitted tcgen05/TMEM seeds only on their exact proven shapes.  Every
guard miss, including forced fallback, delegates unchanged to the selected
Weave-only base dispatcher; no seed guard is widened by padding or inference.
"""
from __future__ import annotations
from json import loads as _json_loads
from .._dispatch_runtime import _capture_cuTensorMapEncodeTiled, _decode_capture, _import_dispatch_module, _ir_proxy
from typing import Any
from . import knn_search_dispatch0618_084a_lowd_d256_post_d384_k64_v1 as registered_base
from . import knn_search_dispatch0630_floor14_expanded_coverage_synthesis_6675_v1 as cohort_base
from . import knn_search_residual0705_q64_tail_last_tile_handoff_812c_v1 as tail_812c
from . import knn_search_residual0705_q64_warp_distributed_state_04b4_v1 as exact_04b4
from . import knn_search_residual_convergence_q128_stability_then_kernel_round301_a8f5_v1 as q128_a8f5
from .._dispatch_runtime import detect_gpu_arch
PROFILE_REGISTERED_084A = 'registered_084a_exact_guard_overlay_e649'
PROFILE_COHORT_6675 = 'cohort_6675_exact_guard_overlay_e649'
PROFILE_ALL = PROFILE_REGISTERED_084A
ENTRYPOINT = 'loom.examples.weave.knn_search_residual0705_full_bucket_dispatcher_e649_v1:launch_for_eval'
ENTRYPOINT_REGISTERED = 'loom.examples.weave.knn_search_residual0705_full_bucket_dispatcher_e649_v1:launch_registered_overlay_for_eval'
ENTRYPOINT_COHORT = 'loom.examples.weave.knn_search_residual0705_full_bucket_dispatcher_e649_v1:launch_cohort_overlay_for_eval'
REGISTERED_BASE_ENTRYPOINT = 'loom.examples.weave.knn_search_dispatch0618_084a_lowd_d256_post_d384_k64_v1:launch_for_eval'
COHORT_BASE_ENTRYPOINT = 'loom.examples.weave.knn_search_dispatch0630_floor14_expanded_coverage_synthesis_6675_v1:launch_for_eval'
_BASES: dict[str, tuple[Any, str, str]] = {PROFILE_REGISTERED_084A: (registered_base, REGISTERED_BASE_ENTRYPOINT, 'registered_084a'), PROFILE_COHORT_6675: (cohort_base, COHORT_BASE_ENTRYPOINT, 'historical_6675')}
_ENTRIES: tuple[dict[str, Any], ...] = ({'shape_key': 'residual_q64_exact', 'shape': (1, 64, 262144, 256, 64), 'guard_id': 'e649_exact_q64_m262144_04b4', 'guard': 'B == 1 and Q == 64 and M == 262144 and D == 256 and K == 64 and queries/database are contiguous BF16 CUDA tensors and outputs are contiguous FP32/INT32 CUDA tensors on one device and not self_search and not force_fallback and arch in {sm_100a,sm_103a}', 'route': exact_04b4.ROUTE, 'entrypoint': exact_04b4.ENTRYPOINT, 'seed': 'warp_distributed_state_04b4', 'source_task': 'weave-evolve-knn-search-residual-convergence-04b4', 'source_commit': 'fc6554c49e7c42ab4384585f1f52219bf67d8085', 'module': exact_04b4}, {'shape_key': 'residual_q64_m_tail_minus', 'shape': (1, 64, 262143, 256, 64), 'guard_id': 'e649_exact_q64_m262143_812c', 'guard': 'B == 1 and Q == 64 and M == 262143 and D == 256 and K == 64 and queries/database are contiguous BF16 CUDA tensors and outputs are contiguous FP32/INT32 CUDA tensors on one device and not self_search and not force_fallback and arch in {sm_100a,sm_103a}', 'route': tail_812c.ROUTE, 'entrypoint': tail_812c.ENTRYPOINT, 'seed': 'tail_last_tile_handoff_812c', 'source_task': 'weave-evolve-knn-search-residual-convergence-812c', 'source_commit': 'e7efffcace81a2e0f60c959df0daf42266efd093', 'module': tail_812c}, {'shape_key': 'residual_q128_stability', 'shape': (1, 128, 262144, 256, 64), 'guard_id': 'e649_exact_q128_m262144_a8f5', 'guard': 'B == 1 and Q == 128 and M == 262144 and D == 256 and K == 64 and queries/database are contiguous BF16 CUDA tensors and outputs are contiguous FP32/INT32 CUDA tensors on one device and not self_search and not force_fallback and arch in {sm_100a,sm_103a}', 'route': q128_a8f5.ROUTE, 'entrypoint': q128_a8f5.ENTRYPOINT, 'seed': 'split304_fullwaves_a8f5', 'source_task': 'weave-evolve-knn-search-residual-convergence-a8f5', 'source_commit': '93cfae6993ef1ef43a25f4d04bc9a507f947cdb8', 'module': q128_a8f5})
CONSUMED_SEEDS = _decode_capture(_json_loads('{"__tuple__": ["warp_distributed_state_04b4", "tail_last_tile_handoff_812c", "split304_fullwaves_a8f5"]}'))
SHAPE_DISPATCH_REGISTRY = _ENTRIES

def _base(profile: str) -> tuple[Any, str, str]:
    try:
        return _BASES[profile]
    except KeyError as exc:
        raise ValueError(''.join(['unknown residual e649 dispatcher profile: ', format(profile, '')])) from exc

def _shape_tuple(inputs: dict[str, Any]) -> tuple[int, int, int, int, int]:
    return tuple((int(inputs[name]) for name in ('B', 'Q', 'M', 'D', 'K')))

def _valid_contract_abi(inputs: dict[str, Any]) -> bool:
    tensors = {'queries': inputs.get('queries'), 'database': inputs.get('database'), 'out_distances': inputs.get('out_distances'), 'out_indices': inputs.get('out_indices')}
    if any((tensor is None for tensor in tensors.values())):
        return False
    queries = tensors['queries']
    database = tensors['database']
    out_distances = tensors['out_distances']
    out_indices = tensors['out_indices']
    assert queries is not None
    assert database is not None
    assert out_distances is not None
    assert out_indices is not None
    expected_shapes = {'queries': (int(inputs['B']), int(inputs['Q']), int(inputs['D'])), 'database': (int(inputs['B']), int(inputs['M']), int(inputs['D'])), 'out_distances': (int(inputs['B']), int(inputs['Q']), int(inputs['K'])), 'out_indices': (int(inputs['B']), int(inputs['Q']), int(inputs['K']))}
    return bool(str(queries.dtype) == 'torch.bfloat16' and str(database.dtype) == 'torch.bfloat16' and (str(out_distances.dtype) == 'torch.float32') and (str(out_indices.dtype) == 'torch.int32') and all((tensor.is_cuda for tensor in tensors.values())) and all((tensor.is_contiguous() for tensor in tensors.values())) and (tuple(queries.shape) == expected_shapes['queries']) and (tuple(database.shape) == expected_shapes['database']) and (tuple(out_distances.shape) == expected_shapes['out_distances']) and (tuple(out_indices.shape) == expected_shapes['out_indices']) and (len({str(tensor.device) for tensor in tensors.values()}) == 1))

def _active_entry(inputs: dict[str, Any]) -> dict[str, Any] | None:
    if bool(inputs.get('force_fallback', False)):
        return None
    if bool(inputs.get('self_search', False)) or not _valid_contract_abi(inputs):
        return None
    if detect_gpu_arch() not in {'sm_100a', 'sm_103a'}:
        return None
    shape = _shape_tuple(inputs)
    entry = next((entry for entry in _ENTRIES if entry['shape'] == shape), None)
    if entry is None:
        return None
    if entry['module'].selected_route(inputs) != entry['route']:
        return None
    return entry

def _guard_order(profile: str) -> list[str]:
    base, _, _ = _base(profile)
    order = [str(entry['guard_id']) for entry in _ENTRIES]
    order.extend((str(entry.get('shape_key', entry.get('guard_id', 'inherited'))) for entry in getattr(base, 'SHAPE_DISPATCH_REGISTRY', ())))
    return order

def selected_route_for_profile(inputs: dict[str, Any], profile: str) -> str:
    entry = _active_entry(inputs)
    if entry is not None:
        return str(entry['route'])
    base, _, _ = _base(profile)
    return str(base.selected_route(inputs))

def selected_route(inputs: dict[str, Any]) -> str:
    return selected_route_for_profile(inputs, PROFILE_ALL)

def selected_route_name(inputs: dict[str, Any]) -> str:
    return selected_route(inputs)

def route_info_for_profile(inputs: dict[str, Any], profile: str) -> dict[str, Any]:
    base, base_entrypoint, base_id = _base(profile)
    candidate_entrypoint = ENTRYPOINT_REGISTERED if profile == PROFILE_REGISTERED_084A else ENTRYPOINT_COHORT
    entry = _active_entry(inputs)
    if entry is None:
        info = dict(base.route_info(inputs))
        route = str(info.get('route') or info.get('selected_route'))
        info.update({'profile': profile, 'dispatcher_entrypoint': candidate_entrypoint, 'base_dispatcher': base_entrypoint, 'base_id': base_id, 'route': route, 'selected_route': route, 'guard_order': _guard_order(profile), 'forced_fallback': bool(inputs.get('force_fallback', False)), 'production_policy': 'weave_only', 'external_fallback': None})
        return info
    parent = dict(base.route_info(inputs))
    parent_route = str(parent.get('route') or parent.get('selected_route'))
    return {'profile': profile, 'dispatcher_entrypoint': candidate_entrypoint, 'base_dispatcher': base_entrypoint, 'base_id': base_id, 'route': entry['route'], 'selected_route': entry['route'], 'selected_entrypoint': entry['entrypoint'], 'selected_seed': entry['seed'], 'expected_seed': entry['seed'], 'selected_seed_task': entry['source_task'], 'source_task': entry['source_task'], 'source_commit': entry['source_commit'], 'parent_route': parent_route, 'replaced_route': parent_route, 'route_kind': 'specialized', 'route_source': 'shape-specific-seed', 'classification': 'seed-consumed', 'coverage_only': False, 'production_policy': 'weave_only', 'external_fallback': None, 'guard_order': _guard_order(profile), 'guard_id': entry['guard_id'], 'guard_condition': entry['guard'], 'selected_guard': entry['guard'], 'forced_fallback': False, 'fallback': base_entrypoint, 'missing_weave_route': False}

def route_info(inputs: dict[str, Any]) -> dict[str, Any]:
    return route_info_for_profile(inputs, PROFILE_ALL)

def route_trace_entry(label: str, inputs: dict[str, Any], profile: str=PROFILE_ALL) -> dict[str, Any]:
    return {'label': label, 'shape_key': label, 'B': int(inputs['B']), 'Q': int(inputs['Q']), 'M': int(inputs['M']), 'D': int(inputs['D']), 'K': int(inputs['K']), 'self_search': bool(inputs.get('self_search', False)), **route_info_for_profile(inputs, profile)}

def launch_for_profile(inputs: dict[str, Any], profile: str) -> dict[str, Any]:
    entry = _active_entry(inputs)
    if entry is not None:
        return entry['module'].launch_for_eval(inputs)
    base, _, _ = _base(profile)
    return base.launch_for_eval(inputs)

def launch_registered_overlay_for_eval(inputs: dict[str, Any]) -> dict[str, Any]:
    return launch_for_profile(inputs, PROFILE_REGISTERED_084A)

def launch_cohort_overlay_for_eval(inputs: dict[str, Any]) -> dict[str, Any]:
    return launch_for_profile(inputs, PROFILE_COHORT_6675)

def launch_for_eval(inputs: dict[str, Any]) -> dict[str, Any]:
    return launch_registered_overlay_for_eval(inputs)
launch_registered_overlay_for_eval.route_info = lambda inputs: route_info_for_profile(inputs, PROFILE_REGISTERED_084A)
launch_cohort_overlay_for_eval.route_info = lambda inputs: route_info_for_profile(inputs, PROFILE_COHORT_6675)
launch_for_eval.route_info = launch_registered_overlay_for_eval.route_info
