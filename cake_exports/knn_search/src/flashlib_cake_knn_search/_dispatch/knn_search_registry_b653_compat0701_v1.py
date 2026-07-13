"""Compose the 7ad3 exact route, b653, restored full133, and the 0701 surface.

Minimum architecture: sm_100a for registry shapes delegated to b653.  Other
shapes preserve the architecture requirements and behavior of the selected
full133 or 0701 child. This host-only dispatcher restores exact full133 routes
with current correctness and CUPTI evidence while preserving the newer b653
residual and 0701 public surfaces.  The exact D256/Q64/K64 candidate route is
7ad3; the immutable current-comparison surface retains its prior 04b4 owner.
"""
from __future__ import annotations
from json import loads as _json_loads
from .._dispatch_runtime import _capture_cuTensorMapEncodeTiled, _decode_capture, _import_dispatch_module, _ir_proxy
from dataclasses import dataclass
from typing import Any
from . import knn_search_dispatch0624_full133_eacf_lowd_cd65_v1 as restored_full133
from . import knn_search_dispatch0701_k11_d128_guard_repair_v1 as compat0701
from . import knn_search_residual0705_q64_tail_split152_full_bucket_dispatcher_b653_v1 as registered_b653
from . import knn_search_residual0707_d256_q64_split152_warp_state_7ad3_v1 as d256_q64_7ad3
from . import knn_search_residual_full198_full_bucket_portfolio_0707_09a4_v1 as portfolio_09a4
from .._dispatch_runtime import PreparedKernelSequence, capture_kernel_launches
ENTRYPOINT = 'loom.examples.weave.knn_search_registry_b653_compat0701_v1:launch_for_eval'
detect_gpu_arch = registered_b653.detect_gpu_arch
_B653_REGISTRY_SHAPES = frozenset({(1, 256, 256, 128, 5, True), (1, 1, 131072, 128, 10, False), (1, 128, 131072, 128, 10, False), (1, 4096, 20000, 128, 10, False), (1, 8, 131072, 128, 10, False), (1, 16, 131072, 128, 10, False), (1, 32, 131072, 128, 10, False), (1, 64, 131072, 128, 10, False), (1, 4096, 20000, 128, 1, False), (1, 4096, 20000, 128, 2, False), (1, 4096, 20000, 128, 64, False), (1, 128, 131072, 256, 10, False), (1, 128, 131072, 256, 64, False), (1, 8, 10, 32, 10, False), (1, 8, 20, 48, 10, False), (1, 1500, 1500, 2, 32, True), (1, 1500, 1500, 2, 64, True), (1, 4096, 20000, 128, 5, False), (1, 4096, 20000, 128, 8, False), (1, 4096, 16384, 128, 8, False), (1, 4096, 32768, 128, 8, False)})
_B653_RESIDUAL_SHAPES = frozenset({(1, 64, 262144, 256, 64, False), (1, 64, 262143, 256, 64, False), (1, 64, 262145, 256, 64, False), (1, 63, 262144, 256, 64, False), (1, 65, 262144, 256, 64, False), (1, 128, 262144, 256, 64, False)})
_B653_DOMAIN_SHAPES = _B653_REGISTRY_SHAPES | _B653_RESIDUAL_SHAPES
_D256_Q64_7AD3_KEY = (1, 64, 262144, 256, 64, False)
_D256_Q64_7AD3_SEED = 'residual0707_d256_q64_split152_warp_state_7ad3'
_D256_Q64_7AD3_GUARD_ID = 'residual0707_d256_q64_split152_warp_state_7ad3_exact'
_D256_Q64_7AD3_GUARD_CONDITION = 'B==1 && Q==64 && M==262144 && D==256 && K==64 && dtype==bfloat16 && contiguous(query,database,out_distances,out_indices) && out_distances.dtype==float32 && out_indices.dtype==int32 && same_cuda_device && !self_search && !force_fallback && arch in {sm_100a,sm_103a}'
_PORTFOLIO_09A4_SHAPES = frozenset((key for key, label in portfolio_09a4._LABEL_BY_KEY.items() if label in portfolio_09a4.SELECTED_TARGET_LABELS))
_RETAINED_09A4_SHAPES = frozenset((key for key, label in portfolio_09a4._LABEL_BY_KEY.items() if label in portfolio_09a4.RETAINED_TARGET_LABELS))
if _PORTFOLIO_09A4_SHAPES & _RETAINED_09A4_SHAPES:
    raise RuntimeError('09a4 selected and retained route ownership overlaps')
if _PORTFOLIO_09A4_SHAPES | _RETAINED_09A4_SHAPES != frozenset(portfolio_09a4._BINDING_BY_KEY):
    raise RuntimeError('09a4 route ownership does not cover the full denominator')
_FULL133_RESTORED_SHAPES = frozenset({(1, 12, 100, 64, 20, False), (1, 16, 32768, 1024, 10, False), (1, 32, 32768, 512, 64, False), (1, 32, 32768, 768, 10, False), (1, 32, 131072, 384, 10, False), (1, 64, 65536, 1, 10, False), (1, 64, 65536, 130, 64, False), (1, 64, 65536, 257, 64, False), (1, 64, 65536, 512, 10, False), (1, 127, 65536, 1, 10, False), (1, 127, 131071, 128, 10, False), (1, 128, 65535, 1, 10, False), (1, 128, 65536, 1, 10, False), (1, 128, 65536, 3, 10, False), (1, 128, 65536, 5, 10, False), (1, 128, 65536, 7, 10, False), (1, 128, 65536, 15, 10, False), (1, 128, 65536, 31, 10, False), (1, 128, 65536, 63, 10, False), (1, 128, 65536, 65, 10, False), (1, 128, 65536, 127, 10, False), (1, 128, 65536, 129, 10, False), (1, 128, 65536, 130, 10, False), (1, 128, 65536, 255, 10, False), (1, 128, 65536, 257, 10, False), (1, 128, 65536, 258, 10, False), (1, 128, 65536, 511, 10, False), (1, 128, 65537, 1, 10, False), (1, 128, 131072, 128, 64, False), (1, 129, 65536, 1, 10, False), (1, 256, 65536, 1, 10, False), (1, 2048, 2048, 3, 10, True), (1, 4096, 4096, 3, 32, True), (1, 4096, 20000, 128, 1, False), (1, 4096, 20000, 128, 64, False), (1, 4096, 32768, 128, 32, False), (1, 4096, 32768, 128, 48, False), (1, 4096, 32768, 128, 64, False), (1, 4096, 49152, 128, 64, False), (2, 64, 65536, 129, 10, False)})

def _shape_key(inputs: dict[str, Any]) -> tuple[int, int, int, int, int, bool]:
    return (int(inputs['B']), int(inputs['Q']), int(inputs['M']), int(inputs['D']), int(inputs['K']), bool(inputs.get('self_search', False)))

def _uses_registered_b653(inputs: dict[str, Any]) -> bool:
    shape_key = _shape_key(inputs)
    return shape_key in _B653_DOMAIN_SHAPES and shape_key not in _FULL133_RESTORED_SHAPES

def _uses_restored_full133(inputs: dict[str, Any]) -> bool:
    return _shape_key(inputs) in _FULL133_RESTORED_SHAPES

def _current_child(inputs: dict[str, Any]):
    if _uses_restored_full133(inputs):
        return restored_full133
    return registered_b653 if _uses_registered_b653(inputs) else compat0701

def _tensor_matches(tensor: Any, *, shape: tuple[int, ...], dtype: str) -> bool:
    if tensor is None:
        return False
    try:
        return tuple((int(dim) for dim in tensor.shape)) == shape and str(tensor.dtype) == dtype and bool(tensor.is_contiguous()) and (str(tensor.device).split(':', 1)[0] == 'cuda')
    except (AttributeError, TypeError, ValueError):
        return False

def _portfolio_09a4_binding(inputs: dict[str, Any]) -> dict[str, Any] | None:
    key = _shape_key(inputs)
    if key not in _PORTFOLIO_09A4_SHAPES or bool(inputs.get('force_fallback', False)):
        return None
    bsz, q_rows, m_rows, dim, top_k, _ = key
    tensors = (inputs.get('queries'), inputs.get('database'), inputs.get('out_distances'), inputs.get('out_indices'))
    if not (_tensor_matches(tensors[0], shape=(bsz, q_rows, dim), dtype='torch.bfloat16') and _tensor_matches(tensors[1], shape=(bsz, m_rows, dim), dtype='torch.bfloat16') and _tensor_matches(tensors[2], shape=(bsz, q_rows, top_k), dtype='torch.float32') and _tensor_matches(tensors[3], shape=(bsz, q_rows, top_k), dtype='torch.int32')):
        return None
    if len({str(tensor.device) for tensor in tensors}) != 1:
        return None
    if detect_gpu_arch() not in {'sm_100a', 'sm_103a'}:
        return None
    portfolio_09a4._require_owned_key(inputs)
    return portfolio_09a4._BINDING_BY_KEY[key]

def _uses_d256_q64_7ad3(inputs: dict[str, Any]) -> bool:
    key = _shape_key(inputs)
    if key != _D256_Q64_7AD3_KEY or bool(inputs.get('force_fallback', False)):
        return False
    bsz, q_rows, m_rows, dim, top_k, _ = key
    tensors = (inputs.get('queries'), inputs.get('database'), inputs.get('out_distances'), inputs.get('out_indices'))
    if not (_tensor_matches(tensors[0], shape=(bsz, q_rows, dim), dtype='torch.bfloat16') and _tensor_matches(tensors[1], shape=(bsz, m_rows, dim), dtype='torch.bfloat16') and _tensor_matches(tensors[2], shape=(bsz, q_rows, top_k), dtype='torch.float32') and _tensor_matches(tensors[3], shape=(bsz, q_rows, top_k), dtype='torch.int32')):
        return False
    if len({str(tensor.device) for tensor in tensors}) != 1:
        return False
    return detect_gpu_arch() in {'sm_100a', 'sm_103a'}

def _uses_portfolio_09a4(inputs: dict[str, Any]) -> bool:
    return _portfolio_09a4_binding(inputs) is not None

def _decorate_retained_09a4_identity(inputs: dict[str, Any], info: dict[str, Any]) -> dict[str, Any]:
    key = _shape_key(inputs)
    if key not in _RETAINED_09A4_SHAPES or bool(inputs.get('force_fallback', False)):
        return info
    label = portfolio_09a4._LABEL_BY_KEY[key]
    identity = portfolio_09a4.RETAINED_INCUMBENT_IDENTITIES[label]
    route = str(info.get('selected_route', info.get('route', '')))
    entrypoint = str(info.get('selected_entrypoint', ''))
    if route != identity['route'] or entrypoint != identity['entrypoint']:
        raise RuntimeError(''.join(['retained 09a4 incumbent identity changed for ', format(label, ''), ': route=', format(repr(route), ''), ', entrypoint=', format(repr(entrypoint), '')]))
    info.update({'selected_seed': identity['seed'], 'expected_seed': identity['seed'], 'retained_shape_key': label, 'retained_incumbent': True, 'consumed_by_production_dispatcher': False, 'classification': 'retained-incumbent'})
    return info

def _child(inputs: dict[str, Any]):
    if _uses_d256_q64_7ad3(inputs):
        return d256_q64_7ad3
    binding = _portfolio_09a4_binding(inputs)
    return binding['module'] if binding is not None else _current_child(inputs)

def current_selected_route(inputs: dict[str, Any]) -> str:
    return str(_current_child(inputs).selected_route(inputs))

def selected_route(inputs: dict[str, Any]) -> str:
    if _uses_d256_q64_7ad3(inputs):
        return d256_q64_7ad3.ROUTE
    if _uses_portfolio_09a4(inputs):
        return portfolio_09a4.selected_route(inputs)
    return current_selected_route(inputs)

def selected_route_name(inputs: dict[str, Any]) -> str:
    return selected_route(inputs)

def route_info(inputs: dict[str, Any]) -> dict[str, Any]:
    shape_key = _shape_key(inputs)
    if _uses_d256_q64_7ad3(inputs):
        info = dict(d256_q64_7ad3.route_info(inputs))
        info.update({'selected_entrypoint': ENTRYPOINT, 'seed_entrypoint': d256_q64_7ad3.ENTRYPOINT, 'selected_seed': _D256_Q64_7AD3_SEED, 'expected_seed': _D256_Q64_7AD3_SEED, 'seed_route_source': info.get('route_source'), 'route_source': 'shape-specific-seed', 'source_task': 'weave-evolve-knn-search-7ad3', 'source_commit': 'e918dfa5f00d879e32ec86cf264ee9f85c9cde68', 'guard_id': _D256_Q64_7AD3_GUARD_ID, 'guard_condition': _D256_Q64_7AD3_GUARD_CONDITION, 'export_composite_entrypoint': ENTRYPOINT, 'export_composite_branch': 'd256_q64_7ad3', 'registry_contract_shape': shape_key in _B653_REGISTRY_SHAPES, 'residual_convergence_shape': shape_key in _B653_RESIDUAL_SHAPES, 'registered_b653_selected': False, 'full133_restored_shape': False, 'replaced_route': current_selected_route(inputs), 'consumed_by_production_dispatcher': True, 'retained_incumbent': False, 'classification': 'seed-consumed', 'fallback': 'preserved_current_weave_dispatcher'})
        return info
    binding = _portfolio_09a4_binding(inputs)
    if binding is not None:
        info = dict(portfolio_09a4.route_info(inputs))
        info.update({'selected_entrypoint': ENTRYPOINT, 'seed_entrypoint': binding['entrypoint'], 'portfolio_entrypoint': portfolio_09a4.ENTRYPOINT, 'export_composite_entrypoint': ENTRYPOINT, 'export_composite_branch': 'portfolio_09a4', 'registry_contract_shape': shape_key in _B653_REGISTRY_SHAPES, 'residual_convergence_shape': shape_key in _B653_RESIDUAL_SHAPES, 'registered_b653_selected': False, 'full133_restored_shape': False, 'replaced_route': current_selected_route(inputs), 'consumed_by_production_dispatcher': True, 'classification': 'seed-consumed', 'fallback': 'preserved_current_weave_dispatcher'})
        return info
    uses_full133 = shape_key in _FULL133_RESTORED_SHAPES
    uses_registry = shape_key in _B653_DOMAIN_SHAPES and (not uses_full133)
    info = _decorate_retained_09a4_identity(inputs, dict(_current_child(inputs).route_info(inputs)))
    info.update({'export_composite_entrypoint': ENTRYPOINT, 'export_composite_branch': 'restored_full133' if uses_full133 else 'registered_b653' if uses_registry else 'compat0701', 'registry_contract_shape': shape_key in _B653_REGISTRY_SHAPES, 'residual_convergence_shape': shape_key in _B653_RESIDUAL_SHAPES, 'registered_b653_selected': uses_registry, 'full133_restored_shape': uses_full133})
    return info

def launch_for_eval(inputs: dict[str, Any]) -> dict[str, Any]:
    if _uses_d256_q64_7ad3(inputs):
        outputs = d256_q64_7ad3.launch_for_eval(inputs)
        if outputs.get('distances') is not inputs['out_distances'] or outputs.get('indices') is not inputs['out_indices']:
            raise RuntimeError('7ad3 production route did not preserve caller-owned outputs')
        return outputs
    if _uses_portfolio_09a4(inputs):
        outputs = portfolio_09a4.launch_for_eval(inputs)
        if outputs.get('distances') is not inputs['out_distances'] or outputs.get('indices') is not inputs['out_indices']:
            raise RuntimeError('09a4 production route did not preserve caller-owned outputs')
        return outputs
    return launch_current_for_eval(inputs)

def launch_current_for_eval(inputs: dict[str, Any]) -> dict[str, Any]:
    """Launch the immutable pre-09a4 dispatcher for same-session comparison."""
    if _uses_restored_full133(inputs):
        return restored_full133.launch_for_eval(inputs)
    if _uses_registered_b653(inputs):
        return registered_b653.launch_361b_f392_d436_split152_for_eval(inputs)
    return compat0701.launch_for_eval(inputs)

def current_route_info(inputs: dict[str, Any]) -> dict[str, Any]:
    shape_key = _shape_key(inputs)
    uses_full133 = shape_key in _FULL133_RESTORED_SHAPES
    uses_registry = shape_key in _B653_DOMAIN_SHAPES and (not uses_full133)
    info = _decorate_retained_09a4_identity(inputs, dict(_current_child(inputs).route_info(inputs)))
    info.update({'current_dispatcher_entrypoint': ''.join([format(ENTRYPOINT.rsplit(':', 1)[0], ''), ':launch_current_for_eval']), 'export_composite_entrypoint': ENTRYPOINT, 'export_composite_branch': 'restored_full133' if uses_full133 else 'registered_b653' if uses_registry else 'compat0701', 'registry_contract_shape': shape_key in _B653_REGISTRY_SHAPES, 'residual_convergence_shape': shape_key in _B653_RESIDUAL_SHAPES, 'registered_b653_selected': uses_registry, 'full133_restored_shape': uses_full133, 'comparison_role': 'pre_09a4_current_dispatcher'})
    return info
launch_current_for_eval.route_info = current_route_info

@dataclass(frozen=True)
class PreparedKnnSearchForEval:
    """One route-resolved, stream-bound sequence of direct CUDA launches."""
    inputs: dict[str, Any]
    route: dict[str, Any]
    direct_launcher: PreparedKernelSequence
    stream: Any
    stream_handle: int
    device_index: int
    scratch_pointers: tuple[int, ...]

    @property
    def launch_count(self) -> int:
        return int(self.direct_launcher.launch_count)

def _caller_owned_outputs(outputs: Any, inputs: dict[str, Any]) -> bool:
    return bool(isinstance(outputs, dict) and outputs.get('distances', outputs.get('out_distances')) is inputs['out_distances'] and (outputs.get('indices', outputs.get('out_indices')) is inputs['out_indices']))

def _prepared_scratch_pointers(direct_launcher: PreparedKernelSequence, inputs: dict[str, Any]) -> tuple[int, ...]:
    public_pointers = {int(value.data_ptr()) for key, value in inputs.items() if isinstance(key, str) and (not key.startswith('_')) and callable(getattr(value, 'data_ptr', None))}
    pointers: set[int] = set()
    for launch in direct_launcher._launches:
        for value in launch._keepalive:
            data_ptr = getattr(value, 'data_ptr', None)
            if callable(data_ptr):
                pointer = int(data_ptr())
                if pointer not in public_pointers:
                    pointers.add(pointer)
    return tuple(sorted(pointers))

def prepare_for_eval(inputs: dict[str, Any], *, stream: Any=None) -> PreparedKnnSearchForEval:
    """Resolve once and capture the candidate's exact leaf launch sequence."""
    import torch
    query = inputs.get('queries')
    if query is None or not bool(getattr(query, 'is_cuda', False)):
        raise TypeError('prepared KNN-search inputs require CUDA query tensors')
    device_index = int(query.device.index or 0)
    with torch.cuda.device(device_index):
        resolved_stream = torch.cuda.current_stream(device_index) if stream is None else stream
        stream_handle = int(resolved_stream.cuda_stream)
        prepared_inputs = dict(inputs)
        prepared_inputs['_knn_search_prepared_stream_key'] = (device_index, stream_handle)
        resolved_route = route_info(prepared_inputs)
        with capture_kernel_launches(stream=resolved_stream, arch=detect_gpu_arch(), inputs=prepared_inputs) as captured:
            outputs = launch_for_eval(prepared_inputs)
        if not _caller_owned_outputs(outputs, prepared_inputs):
            raise RuntimeError('prepared KNN-search capture did not bind caller-owned outputs')
        direct_launcher = captured.bind(outputs)
        scratch_pointers = _prepared_scratch_pointers(direct_launcher, prepared_inputs)
    return PreparedKnnSearchForEval(inputs=prepared_inputs, route=resolved_route, direct_launcher=direct_launcher, stream=resolved_stream, stream_handle=stream_handle, device_index=device_index, scratch_pointers=scratch_pointers)

def launch_prepared_for_eval(prepared: PreparedKnnSearchForEval) -> dict[str, Any]:
    """Submit a frozen direct sequence without root or parent dispatch."""
    if not isinstance(prepared, PreparedKnnSearchForEval):
        raise TypeError('prepared must be returned by prepare_for_eval')
    import torch
    with torch.cuda.device(prepared.device_index):
        current = torch.cuda.current_stream(prepared.device_index)
        if int(current.cuda_stream) != prepared.stream_handle:
            raise RuntimeError('prepared KNN-search launch is bound to its preparation stream')
        outputs = prepared.direct_launcher(prepared.inputs, stream=prepared.stream)
        prepared.direct_launcher.record_stream(prepared.stream)
    if not _caller_owned_outputs(outputs, prepared.inputs):
        raise RuntimeError('prepared KNN-search launch lost caller-owned outputs')
    return outputs

def prepared_route_info(prepared: PreparedKnnSearchForEval) -> dict[str, Any]:
    current_scratch = _prepared_scratch_pointers(prepared.direct_launcher, prepared.inputs)
    return {**prepared.route, 'prepared_entrypoint': ''.join([format(ENTRYPOINT.rsplit(':', 1)[0], ''), ':launch_prepared_for_eval']), 'resolved_direct_launcher': True, 'direct_launch_count': prepared.launch_count, 'root_route_traversals_per_launch': 0, 'parent_chain_traversals_per_launch': 0, 'caller_owned_outputs': True, 'scratch_reused': current_scratch == prepared.scratch_pointers, 'scratch_pointer_count': len(current_scratch), 'stream_bound': True, 'stream_handle': prepared.stream_handle, 'internal_device_synchronizations': 0}
launch_for_eval.route_info = route_info
