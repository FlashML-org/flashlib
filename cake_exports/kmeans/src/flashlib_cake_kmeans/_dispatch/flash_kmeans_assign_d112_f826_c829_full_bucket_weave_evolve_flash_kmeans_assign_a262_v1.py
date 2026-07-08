"""Exact-guard D112 c829/f826/4549 portfolio (minimum: sm_100a).

The two assigned B4/D112/K1024 rows select the frozen c829 owner-local
warp-MMA kernel.  B5/N2944/D112/K512 retains the frozen f826 two-owner
warp-MMA kernel through the 32b2 portfolio, and every other row in the exact
14-row D112 bucket retains the frozen 4549 child through that same portfolio.
The host-only composition adds no padding, packing, workspace, fallback, or
extra GPU launch.
"""
from __future__ import annotations
from json import loads as _json_loads
from .._dispatch_runtime import _capture_cuTensorMapEncodeTiled, _decode_capture, _import_dispatch_module, _ir_proxy
from dataclasses import dataclass
from threading import RLock
from typing import Any, Callable
from . import flash_kmeans_assign_d112_f826_exact_portfolio_generalize_auto_tuning_flash_kmeans_assign_8d82_v1 as _retained
from . import flash_kmeans_assign_d112_k1024_owner_local_warp_mma_distinct_workfeed_weave_evolve_flash_kmeans_assign_c829_v1 as _c829
FEAT_D = 112
ROUTE_ID = 'd112_f826_c829_full_bucket_weave_evolve_flash_kmeans_assign_a262_v1'
SEED_ID = 'd112-f826-c829-full-bucket-weave-evolve-flash-kmeans-assign-a262-v1'
TARGET_SHAPE = 'physical_D112_padded_14'
SHAPE_KEYS = {(2, 2048, 112, 512): 'post_d895_d112_b2_n2048_k512_d112', (4, 8192, 112, 1024): 'post_d895_d112_b4_n8192_k1024_d112', (1, 512, 112, 8192): 'post_d895_d112_b1_n512_k8192_d112', (1, 256, 112, 256): 'post_d895_d112_b1_n256_k256_d112', (3, 3840, 112, 768): 'adjacent_9ca0_d112_tail_div_b3_n3840_k768_d112', (2, 512, 112, 8192): 'adjacent_3328_d112_highk_low_n_b2_n512_k8192_d112', (5, 2176, 112, 512): 'adjacent_5600_d112_odd_tile_tail_b5_n2176_k512_d112', (4, 3456, 112, 256): 'adjacent_a2f8_d112_lowk_tail_b4_n3456_k256_d112', (1, 384, 112, 4096): 'adjacent_8f09_d112_small_highk_b1_n384_k4096_d112', (3, 768, 112, 8192): 'adjacent_d9d5_d112_highk_request_b3_n768_k8192_d112', (5, 2944, 112, 512): 'adjacent_68cf_d112_tail_b5_n2944_k512_d112', (2, 3200, 112, 768): 'adjacent_c44f_d112_odd_tail_b2_n3200_k768_d112', (1, 1408, 112, 4096): 'adjacent_1d49_d112_highk_tail_b1_n1408_k4096_d112', (4, 3712, 112, 1024): 'adjacent_ef00_d112_odd_tail_b4_n3712_k1024_d112'}
EXACT_C829_KEYS = frozenset({(4, 8192, 112, 1024), (4, 3712, 112, 1024)})
EXACT_F826_KEY = (5, 2944, 112, 512)
C829_SOURCE = {'id': 'c829_exact_k1024', 'source_task': 'weave-evolve-flash-kmeans-assign-ab9a', 'implementation_revision': 'e3d609c6c0e68796f1c9273a6e113205c5fcea1e', 'representative_measurement_revision': '2def42e0b313de745896952a8f17d1f3c78680a8', 'timed_source_blob': '6bb66ffdeb5da35094c3e2ee02932ee1cac2432e', 'timed_source_sha256': '3205f28d937bf9ff1616bbb0eb35e8765f13447a9802cb9807dc28d46b1ab27f', 'executable_body_blob': '9d0b7cb69e1274850fc60be2f6f1f78002f2b23f', 'executable_body_sha256': '72cc676d4ddcb32c40cabb01e7f2ff9ffc1138d8449ee49b38ec552c7a54b720', 'final_source_blob': '4ce799909b9b93a64c272c831b55ad4c13186fb3', 'final_source_sha256': '81bf56e0b8cbb8962f9852159096fa1df52c769350fd46412e782c78a7ded010', 'source_path': 'loom/examples/weave/flash_kmeans_assign_d112_k1024_owner_local_warp_mma_distinct_workfeed_weave_evolve_flash_kmeans_assign_c829_v1.py', 'entrypoint': ''.join([format(_c829.__name__, ''), ':launch_for_eval'])}
F826_SOURCE = _decode_capture(_json_loads('{"__dict_items__": [["id", "f826_exact_k512"], ["source_task", "weave-evolve-flash-kmeans-assign-f826"], ["implementation_commit", "124ef2ad24c3ac86c063e69da0cb0c5d7fca0174"], ["implementation_source_blob", "3a1dee841855c15481a740f28ba07142e4fdbb29"], ["implementation_source_sha256", "99c44085f258a699b7e3372115cc82aa969643addae0797bf47c0758cad8cf0f"], ["measurement_source_blob", "a3dcf64c5014d223127bf0a39b13c485cf140c48"], ["measurement_source_sha256", "fd3e97727367d3acbbd3b994a86cb25f563ce3f32544a724dbd09f05a70a88ce"], ["final_source_blob", "f30b546e70bfbc80a94547458d857ead0de5e320"], ["final_source_sha256", "47c2d22bbb00b3061d268fae9c405e62257bde8ba43ba33fc1124d456572cd63"], ["source_path", "loom/examples/weave/flash_kmeans_assign_d112_k512_two_owner_peer_only_mma_weave_evolve_flash_kmeans_assign_f826_v1.py"], ["entrypoint", "loom.examples.weave.flash_kmeans_assign_d112_k512_two_owner_peer_only_mma_weave_evolve_flash_kmeans_assign_f826_v1:launch_for_eval"]]}'))
BASE_4549_SOURCE = _decode_capture(_json_loads('{"__dict_items__": [["id", "base_portfolio"], ["source_task", "weave-evolve-flash-kmeans-assign-4549"], ["source_kernel_task", "weave-evolve-flash-kmeans-assign-4549"], ["source_revision", "21c48b897873f6a5e631c7fccd320c35d73ae6d5"], ["evidence_revision", "713ee91855069de8580152c20ec37b5ab4569c29"], ["source_blob", "9d4da373afd9869c2718fffaf62f4230f4bb2482"], ["source_sha256", "ef1b1b574a5ce0516c745a5c28fa381a47e8a5e5c9ad49e35a93bff496c43b64"], ["source_path", "loom/examples/weave/flash_kmeans_assign_d112_direct_handoff_full_bucket_weave_evolve_flash_kmeans_assign_4549_v1.py"], ["entrypoint", "loom.examples.weave.flash_kmeans_assign_d112_direct_handoff_full_bucket_weave_evolve_flash_kmeans_assign_4549_v1:launch_for_eval"]]}'))
RETAINED_PORTFOLIO_SOURCE = {'id': 'portfolio_32b2', 'source_task': 'generalize-auto-tuning-flash-kmeans-assign-32b2', 'source_blob': '2acba116338583b0f55b06b49b1882779b440b09', 'source_sha256': 'ba9da56a4ff50c211d201dcbfbb69d8d1cb31464c2628dd5110be1320b626087', 'source_path': 'loom/examples/weave/flash_kmeans_assign_d112_f826_exact_portfolio_generalize_auto_tuning_flash_kmeans_assign_8d82_v1.py', 'entrypoint': ''.join([format(_retained.__name__, ''), ':launch_for_eval'])}
_PREPARE_LOCK = RLock()

def _shape(inputs: dict[str, Any]) -> tuple[int, int, int, int]:
    return tuple((int(inputs[name]) for name in ('B', 'N', 'D', 'K')))

def _shape_key(key: tuple[int, int, int, int]) -> str:
    try:
        return SHAPE_KEYS[key]
    except KeyError as exc:
        raise ValueError(''.join(['a262 portfolio rejects unassigned shape ', format(repr(key), '')])) from exc

def supports_shape(*, B: int, D: int, N: int, K: int) -> bool:
    return (int(B), int(N), int(D), int(K)) in SHAPE_KEYS

def source_for_shape(*, B: int, N: int, D: int, K: int) -> dict[str, Any]:
    key = (int(B), int(N), int(D), int(K))
    _shape_key(key)
    if key in EXACT_C829_KEYS:
        return dict(C829_SOURCE)
    return dict(_retained.source_for_shape(B=B, N=N, D=D, K=K))

def _resolve_route(inputs: dict[str, Any], *, force_c829: bool=False, force_f826: bool=False) -> tuple[Callable[[dict[str, Any]], dict[str, Any]], dict[str, Any], str, str]:
    key = _shape(inputs)
    _shape_key(key)
    if force_c829 and force_f826:
        raise ValueError('only one forced route may be requested')
    if force_c829 and key not in EXACT_C829_KEYS:
        raise ValueError('forced c829 requires one of the two exact K1024 keys')
    if force_f826 and key != EXACT_F826_KEY:
        raise ValueError('forced f826 requires exact B5/N2944/D112/K512')
    if key in EXACT_C829_KEYS:
        return (_c829.launch_for_eval, dict(C829_SOURCE), 'guard_exact_two_k1024_keys_c829_v1', '(B, N, D, K) in {(4, 8192, 112, 1024), (4, 3712, 112, 1024)}')
    source = source_for_shape(B=key[0], N=key[1], D=key[2], K=key[3])
    if key == EXACT_F826_KEY:
        return (_retained.launch_for_eval_forced_f826, source, 'guard_exact_b5_n2944_d112_k512_f826_v1', 'B == 5 and N == 2944 and D == 112 and K == 512')
    return (_retained.launch_for_eval, source, 'guard_assigned_d112_bucket_retained_4549_v1', 'assigned D112 bucket key excluding the c829 and f826 exact guards')

def _source_value(source: dict[str, Any], suffix: str) -> Any:
    for prefix in ('final_source', 'source', 'executable_body'):
        value = source.get(''.join([format(prefix, ''), '_', format(suffix, '')]))
        if value is not None:
            return value
    return None

def _finish_outputs(inputs: dict[str, Any], child_outputs: dict[str, Any], source: dict[str, Any], guard_id: str, guard_condition: str) -> dict[str, Any]:
    B, N, D, K = _shape(inputs)
    key = (B, N, D, K)
    caller_out = inputs['out']
    cluster_ids = child_outputs.get('cluster_ids', child_outputs.get('out'))
    if cluster_ids is None:
        raise ValueError(''.join([format(source['id'], ''), ' did not return cluster_ids or out']))
    caller_ptr = int(caller_out.data_ptr())
    returned_ptr = int(cluster_ids.data_ptr())
    if returned_ptr != caller_ptr:
        raise ValueError(''.join([format(source['id'], ''), " replaced caller-owned inputs['out']"]))
    child_trace = dict(child_outputs.get('route_trace') or {})
    specialized = key in EXACT_C829_KEYS or key == EXACT_F826_KEY
    return {'cluster_ids': cluster_ids, 'selected_route': ROUTE_ID, 'route_trace': {'shape_key': _shape_key(key), 'selected_route': ROUTE_ID, 'selected_entrypoint': ''.join([format(__name__, ''), ':launch_for_eval']), 'selected_seed': _c829.SEED_ID if key in EXACT_C829_KEYS else child_trace.get('selected_seed'), 'portfolio_id': ROUTE_ID, 'portfolio_seed_id': SEED_ID, 'retained_portfolio': RETAINED_PORTFOLIO_SOURCE['id'], 'selected_source': source['id'], 'selected_source_task': source['source_task'], 'selected_source_revision': source.get('implementation_revision', source.get('implementation_commit', source.get('source_revision'))), 'selected_source_blob': _source_value(source, 'blob'), 'selected_source_sha256': _source_value(source, 'sha256'), 'selected_source_path': source['source_path'], 'selected_source_entrypoint': source['entrypoint'], 'child_selected_route': child_outputs.get('selected_route'), 'child_route_trace': child_trace, 'route_kind': 'specialized' if specialized else 'general', 'route_source': 'shape-specific-seed' if specialized else 'generated-variant', 'guard_id': guard_id, 'guard_condition': guard_condition, 'classification': 'seed-consumed' if specialized else 'route-ok', 'dispatch_key': 'B,N,D,K', 'dispatch_value': [B, N, D, K], 'caller_output_data_ptr': caller_ptr, 'returned_output_data_ptr': returned_ptr, 'caller_owned_output': True, 'output_dtype': 'int32', 'compute_kernel_count': 1, 'padding_count': 0, 'pack_count': 0, 'fallback_contract_regions': [], 'residual_contract_regions': [], 'global_workspace': False, 'materialized_padding': False, 'production_route': 'weave_only', 'launch_grid': child_trace.get('launch_grid'), 'producer_to_consumer': child_trace.get('producer_to_consumer'), 'primitive_family': child_trace.get('primitive_family', 'warp_mma_sync'), 'B': B, 'N': N, 'D': D, 'K': K}}

def _launch(inputs: dict[str, Any], *, force_c829: bool=False, force_f826: bool=False) -> dict[str, Any]:
    child, source, guard_id, guard_condition = _resolve_route(inputs, force_c829=force_c829, force_f826=force_f826)
    return _finish_outputs(inputs, child(inputs), source, guard_id, guard_condition)

def launch_for_eval(inputs: dict[str, Any]) -> dict[str, Any]:
    """Launch exactly one frozen child through the caller-owned output ABI."""
    return _launch(inputs)

def launch_for_eval_forced_c829(inputs: dict[str, Any]) -> dict[str, Any]:
    return _launch(inputs, force_c829=True)

def launch_for_eval_forced_f826(inputs: dict[str, Any]) -> dict[str, Any]:
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
        import torch
        with torch.cuda.device(self.device_index):
            resolved_stream = self.stream if stream is None else stream
            if int(resolved_stream.cuda_stream) != self.stream_handle:
                raise RuntimeError('prepared a262 portfolio plan is stream-bound')
            if int(self.inputs['out'].data_ptr()) != self.caller_output_data_ptr:
                raise RuntimeError('prepared a262 caller output was replaced')
            return self.direct_launcher(self.inputs, stream=None, timeout_ms=timeout_ms)

def prepare_launch_plan(inputs: dict[str, Any], *, stream: Any=None, force_c829: bool=False, force_f826: bool=False) -> PreparedPortfolioPlan:
    """Resolve and capture one stable direct child launch token."""
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
            raise ValueError('prepared a262 stream device does not match inputs')
        if detect_gpu_arch() not in {'sm_100a', 'sm_103a'}:
            raise ValueError('a262 prepared routes require sm_100a or sm_103a')
        child, source, guard_id, guard_condition = _resolve_route(inputs, force_c829=force_c829, force_f826=force_f826)
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
