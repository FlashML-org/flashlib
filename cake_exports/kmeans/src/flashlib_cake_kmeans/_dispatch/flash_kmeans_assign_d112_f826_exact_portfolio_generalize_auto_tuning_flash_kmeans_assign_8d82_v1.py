"""Namespaced D112 portfolio with one exact f826 guard (minimum: sm_100a).

The guard ``B=5, N=2944, D=112, K=512`` selects the validated f826
two-owner warp-MMA seed.  K=1024 remains on the frozen c72b overlay and every
other admitted D112 row remains on frozen 4549.  This module is additive: it
does not alter the production dispatcher or any child kernel schedule.
"""
from __future__ import annotations
from json import loads as _json_loads
from .._dispatch_runtime import _capture_cuTensorMapEncodeTiled, _decode_capture, _import_dispatch_module, _ir_proxy
from dataclasses import dataclass
from threading import RLock
from typing import Any, Callable
from . import flash_kmeans_assign_d112_4549_c72b_k1024_overlay_weave_evolve_flash_kmeans_assign_3ae6_v1 as _frozen
from . import flash_kmeans_assign_d112_k512_two_owner_peer_only_mma_weave_evolve_flash_kmeans_assign_f826_v1 as _f826
FEAT_D = 112
EXACT_F826_KEY = (5, 2944, 112, 512)
SUPPORTED_K = _frozen.SUPPORTED_K
ROUTE_ID = 'd112_f826_exact_portfolio_generalize_auto_tuning_flash_kmeans_assign_8d82_v1'
SEED_ID = 'd112-overlay-plus-f826-exact-k512-consumption-v1'
TARGET_SHAPE = 'physical_D112_padded_14'
EXACT_SHAPE_KEY = 'adjacent_68cf_d112_tail_b5_n2944_k512_d112'
EXACT_GUARD_ID = 'guard_exact_b5_n2944_d112_k512_f826_v1'
EXACT_GUARD_CONDITION = 'B == 5 and N == 2944 and D == 112 and K == 512'
F826_SOURCE = {'id': 'f826_exact_k512', 'source_task': 'weave-evolve-flash-kmeans-assign-f826', 'implementation_commit': '124ef2ad24c3ac86c063e69da0cb0c5d7fca0174', 'implementation_source_blob': '3a1dee841855c15481a740f28ba07142e4fdbb29', 'implementation_source_sha256': '99c44085f258a699b7e3372115cc82aa969643addae0797bf47c0758cad8cf0f', 'measurement_source_blob': 'a3dcf64c5014d223127bf0a39b13c485cf140c48', 'measurement_source_sha256': 'fd3e97727367d3acbbd3b994a86cb25f563ce3f32544a724dbd09f05a70a88ce', 'final_source_blob': 'f30b546e70bfbc80a94547458d857ead0de5e320', 'final_source_sha256': '47c2d22bbb00b3061d268fae9c405e62257bde8ba43ba33fc1124d456572cd63', 'source_path': 'loom/examples/weave/flash_kmeans_assign_d112_k512_two_owner_peer_only_mma_weave_evolve_flash_kmeans_assign_f826_v1.py', 'entrypoint': ''.join([format(_f826.__name__, ''), ':launch_for_eval'])}
_PREPARE_LOCK = RLock()

def _shape(inputs: dict[str, Any]) -> tuple[int, int, int, int]:
    return tuple((int(inputs[key]) for key in ('B', 'N', 'D', 'K')))

def _is_exact_f826(*, B: int, N: int, D: int, K: int) -> bool:
    return (B, N, D, K) == EXACT_F826_KEY

def supports_shape(*, D: int, N: int, K: int, B: int | None=None) -> bool:
    del B
    return _frozen.supports_shape(D=D, N=N, K=K)

def source_for_shape(*, B: int, N: int, D: int, K: int) -> dict[str, Any]:
    if _is_exact_f826(B=B, N=N, D=D, K=K):
        return dict(F826_SOURCE)
    return dict(_frozen.source_for_k(K))

def _resolve_route(inputs: dict[str, Any], *, force_f826: bool=False) -> tuple[Callable[[dict[str, Any]], dict[str, Any]], dict[str, Any], str, str]:
    B, N, D, K = _shape(inputs)
    if not supports_shape(B=B, D=D, N=N, K=K):
        raise ValueError('8d82 D112 portfolio requires D=112, N%64=0, and K in (256, 512, 768, 1024, 4096, 8192)')
    exact = _is_exact_f826(B=B, N=N, D=D, K=K)
    if force_f826 and (not exact):
        raise ValueError('forced f826 route requires exact B5/N2944/D112/K512')
    if exact:
        return (_f826.launch_for_eval, dict(F826_SOURCE), EXACT_GUARD_ID, EXACT_GUARD_CONDITION)
    source = dict(_frozen.source_for_k(K))
    return (_frozen.launch_for_eval, source, 'guard_frozen_3ae6_k_ownership', 'K == 1024 selects c72b; other admitted K values select 4549')

def _finish_outputs(inputs: dict[str, Any], child_outputs: dict[str, Any], source: dict[str, Any], guard_id: str, guard_condition: str) -> dict[str, Any]:
    B, N, D, K = _shape(inputs)
    caller_out = inputs['out']
    cluster_ids = child_outputs.get('cluster_ids', child_outputs.get('out'))
    if cluster_ids is None:
        raise ValueError(''.join([format(source['id'], ''), ' did not return cluster_ids or out']))
    caller_ptr = int(caller_out.data_ptr())
    returned_ptr = int(cluster_ids.data_ptr())
    if returned_ptr != caller_ptr:
        raise ValueError(''.join([format(source['id'], ''), " did not retain caller-owned inputs['out']"]))
    child_trace = dict(child_outputs.get('route_trace') or {})
    exact = source['id'] == F826_SOURCE['id']
    return {'cluster_ids': cluster_ids, 'selected_route': ROUTE_ID, 'route_trace': {'shape_key': EXACT_SHAPE_KEY if exact else TARGET_SHAPE, 'selected_route': ROUTE_ID, 'selected_entrypoint': ''.join([format(__name__, ''), ':launch_for_eval']), 'selected_seed': _f826.SEED_ID if exact else child_trace.get('selected_seed'), 'expected_seed': _f826.SEED_ID if exact else child_trace.get('selected_seed'), 'portfolio_id': ROUTE_ID, 'portfolio_seed_id': SEED_ID, 'selected_source': source['id'], 'selected_source_task': source['source_task'], 'selected_source_revision': source.get('implementation_commit', source.get('source_revision')), 'selected_source_blob': source.get('final_source_blob', source.get('source_blob')), 'selected_source_sha256': source.get('final_source_sha256', source.get('source_sha256')), 'selected_source_path': source['source_path'], 'selected_source_entrypoint': source['entrypoint'], 'child_selected_route': child_outputs.get('selected_route'), 'child_route_trace': child_trace, 'route_kind': 'specialized' if exact else 'general', 'route_source': 'shape-specific-seed' if exact else 'generated-variant', 'guard_id': guard_id, 'guard_condition': guard_condition, 'classification': 'seed-consumed' if exact else 'route-ok', 'dispatch_key': 'B,N,D,K' if exact else 'K', 'dispatch_value': [B, N, D, K] if exact else K, 'caller_output_data_ptr': caller_ptr, 'returned_output_data_ptr': returned_ptr, 'caller_owned_output': True, 'output_dtype': 'int32', 'compute_kernel_count': 1, 'padding_count': 0, 'pack_count': 0, 'fallback_contract_regions': [], 'residual_contract_regions': [], 'global_workspace': False, 'materialized_padding': False, 'production_route': 'weave_only', 'launch_grid': child_trace.get('launch_grid'), 'producer_to_consumer': child_trace.get('producer_to_consumer'), 'B': B, 'N': N, 'D': D, 'K': K}}

def _launch(inputs: dict[str, Any], *, force_f826: bool=False) -> dict[str, Any]:
    child, source, guard_id, guard_condition = _resolve_route(inputs, force_f826=force_f826)
    return _finish_outputs(inputs, child(inputs), source, guard_id, guard_condition)

def launch_for_eval(inputs: dict[str, Any]) -> dict[str, Any]:
    """Launch the guarded namespaced portfolio through its public ABI."""
    return _launch(inputs)

def launch_for_eval_forced_f826(inputs: dict[str, Any]) -> dict[str, Any]:
    """Force the exact f826 route for exported-ABI correctness evidence."""
    return _launch(inputs, force_f826=True)

@dataclass(frozen=True)
class PreparedPortfolioPlan:
    inputs: dict[str, Any]
    selected_route: str
    selected_source: str
    guard_id: str
    direct_launcher: Callable[..., dict[str, Any]]
    launch_count: int
    device_index: int
    stream: Any
    stream_handle: int
    caller_output_data_ptr: int
    direct_launcher_token: str

    def launch(self, *, stream: Any=None, timeout_ms: float | None=None) -> dict[str, Any]:
        """Submit the captured child launch without re-entering either dispatcher."""
        import torch
        with torch.cuda.device(self.device_index):
            resolved_stream = self.stream if stream is None else stream
            if int(resolved_stream.cuda_stream) != self.stream_handle:
                raise RuntimeError('prepared 8d82 portfolio plan is stream-bound')
            if int(self.inputs['out'].data_ptr()) != self.caller_output_data_ptr:
                raise RuntimeError('prepared 8d82 portfolio caller output was replaced')
            return self.direct_launcher(self.inputs, stream=None, timeout_ms=timeout_ms)

def prepare_launch_plan(inputs: dict[str, Any], *, stream: Any=None, force_f826: bool=False) -> PreparedPortfolioPlan:
    """Resolve the exact route once and capture one stable direct launch token."""
    import torch
    from .._dispatch_runtime import detect_gpu_arch
    from .._dispatch_runtime import capture_kernel_launches
    device_index = inputs['x'].device.index
    if device_index is None:
        device_index = torch.cuda.current_device()
    device_index = int(device_index)
    with torch.cuda.device(device_index):
        resolved_stream = torch.cuda.current_stream(device_index) if stream is None else stream
        stream_device = getattr(resolved_stream, 'device', None)
        stream_device_index = getattr(stream_device, 'index', stream_device)
        if stream_device_index is not None and int(stream_device_index) != device_index:
            raise ValueError('prepared 8d82 stream device does not match input device')
        if detect_gpu_arch() not in {'sm_100a', 'sm_103a'}:
            raise ValueError('8d82 prepared routes require sm_100a or sm_103a')
        child, source, guard_id, guard_condition = _resolve_route(inputs, force_f826=force_f826)
        with _PREPARE_LOCK:
            with capture_kernel_launches(stream=resolved_stream) as captured:
                child_outputs = child(inputs)
            prepared_result = _finish_outputs(inputs, child_outputs, source, guard_id, guard_condition)
            direct_launcher = captured.bind(prepared_result)
    return PreparedPortfolioPlan(inputs=inputs, selected_route=ROUTE_ID, selected_source=source['id'], guard_id=guard_id, direct_launcher=direct_launcher, launch_count=direct_launcher.launch_count, device_index=device_index, stream=resolved_stream, stream_handle=int(resolved_stream.cuda_stream), caller_output_data_ptr=int(inputs['out'].data_ptr()), direct_launcher_token=''.join([format(type(direct_launcher).__module__, ''), '.', format(type(direct_launcher).__qualname__, ''), '@', format(id(direct_launcher), ''.join(['x']))]))

def launch_prepared(plan: PreparedPortfolioPlan, *, stream: Any=None, timeout_ms: float | None=None) -> dict[str, Any]:
    if not isinstance(plan, PreparedPortfolioPlan):
        raise TypeError('plan must be returned by prepare_launch_plan')
    return plan.launch(stream=stream, timeout_ms=timeout_ms)
