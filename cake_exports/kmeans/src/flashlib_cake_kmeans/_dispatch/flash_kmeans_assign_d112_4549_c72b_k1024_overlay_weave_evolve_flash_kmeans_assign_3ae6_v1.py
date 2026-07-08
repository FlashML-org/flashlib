"""Frozen D112 K-owned portfolio composition (minimum architecture: sm_100a).

The c72b dual-issuer tcgen05/TMEM kernel owns exactly K=1024.  The 4549
direct-handoff portfolio owns K in {256, 512, 768, 4096, 8192}.  Both child
sources are imported byte-for-byte from their pinned source revisions; this
module adds only the host ownership guard and auditable route metadata.  Every
route writes caller-owned ``cluster_ids`` with one Weave compute launch and no
padding, packing, fallback, or global workspace.
"""
from __future__ import annotations
from json import loads as _json_loads
from .._dispatch_runtime import _capture_cuTensorMapEncodeTiled, _decode_capture, _import_dispatch_module, _ir_proxy
from typing import Any
from . import flash_kmeans_assign_d112_direct_handoff_full_bucket_weave_evolve_flash_kmeans_assign_4549_v1 as _base
from . import flash_kmeans_assign_d112_shared_point_dual_issuer_tcgen05_weave_evolve_flash_kmeans_assign_6e1e_v1 as _overlay
FEAT_D = 112
BASE_K = (256, 512, 768, 4096, 8192)
OVERLAY_K = 1024
SUPPORTED_K = (*BASE_K[:3], OVERLAY_K, *BASE_K[3:])
ROUTE_ID = 'd112_4549_c72b_k1024_overlay_weave_evolve_flash_kmeans_assign_3ae6_v1'
SEED_ID = 'd112-4549-c72b-k1024-overlay-weave-evolve-flash-kmeans-assign-3ae6-v1'
TARGET_SHAPE = 'physical_D112_padded_14'
BASE_SOURCE = {'id': 'base_portfolio', 'source_task': 'weave-evolve-flash-kmeans-assign-4549', 'source_kernel_task': 'weave-evolve-flash-kmeans-assign-4549', 'source_revision': '21c48b897873f6a5e631c7fccd320c35d73ae6d5', 'evidence_revision': '713ee91855069de8580152c20ec37b5ab4569c29', 'source_blob': '9d4da373afd9869c2718fffaf62f4230f4bb2482', 'source_sha256': 'ef1b1b574a5ce0516c745a5c28fa381a47e8a5e5c9ad49e35a93bff496c43b64', 'source_path': 'loom/examples/weave/flash_kmeans_assign_d112_direct_handoff_full_bucket_weave_evolve_flash_kmeans_assign_4549_v1.py', 'entrypoint': ''.join([format(_base.__name__, ''), ':launch_for_eval'])}
OVERLAY_SOURCE = {'id': 'k1024_overlay', 'source_task': 'weave-evolve-flash-kmeans-assign-c72b', 'source_kernel_task': 'weave-evolve-flash-kmeans-assign-6e1e', 'source_revision': 'a2415111e8f5d268760c97ccc65496173b289741', 'evidence_revision': '048037f7fada3aec532d102d1f1ca16fd1decc8b', 'source_blob': 'cf307b65bed35cd7384fbb78de2bfcf10bc29193', 'source_sha256': '2cf0d0da5e68abe23c22cf5a88315f8aede639095e6b84df40931a423aa94766', 'source_path': 'loom/examples/weave/flash_kmeans_assign_d112_shared_point_dual_issuer_tcgen05_weave_evolve_flash_kmeans_assign_6e1e_v1.py', 'entrypoint': ''.join([format(_overlay.__name__, ''), ':launch_for_eval'])}

def source_for_k(K: int) -> dict[str, str]:
    """Return the immutable source record for one admitted K regime."""
    if K == OVERLAY_K:
        return OVERLAY_SOURCE
    if K in BASE_K:
        return BASE_SOURCE
    raise ValueError('D112 4549/c72b portfolio requires K in (256, 512, 768, 1024, 4096, 8192)')

def supports_shape(*, D: int, N: int, K: int) -> bool:
    if D != FEAT_D:
        return False
    if K == OVERLAY_K:
        return _overlay.supports_shape(D=D, N=N, K=K)
    if K in BASE_K:
        return _base.supports_shape(D=D, N=N, K=K)
    return False

def launch_for_eval(inputs: dict[str, Any]) -> dict[str, Any]:
    B, N, D, K = (int(inputs[key]) for key in ('B', 'N', 'D', 'K'))
    if not supports_shape(D=D, N=N, K=K):
        raise ValueError('D112 4549/c72b portfolio requires D=112, an owned N multiple, and K in (256, 512, 768, 1024, 4096, 8192)')
    source = source_for_k(K)
    child = _overlay if K == OVERLAY_K else _base
    caller_out = inputs['out']
    caller_ptr = int(caller_out.data_ptr())
    child_outputs = child.launch_for_eval(inputs)
    cluster_ids = child_outputs.get('cluster_ids', child_outputs.get('out'))
    if cluster_ids is None:
        raise ValueError(''.join([format(source['id'], ''), ' did not return cluster_ids or out']))
    returned_ptr = int(cluster_ids.data_ptr())
    if returned_ptr != caller_ptr:
        raise ValueError(''.join([format(source['id'], ''), " did not return caller-owned inputs['out']"]))
    child_trace = dict(child_outputs.get('route_trace') or {})
    wrapper_entrypoint = ''.join([format(__name__, ''), ':launch_for_eval'])
    return {'cluster_ids': cluster_ids, 'selected_route': ROUTE_ID, 'route_trace': {'shape_key': TARGET_SHAPE, 'selected_route': ROUTE_ID, 'selected_entrypoint': wrapper_entrypoint, 'selected_seed': SEED_ID, 'selected_source': source['id'], 'selected_source_task': source['source_task'], 'selected_source_kernel_task': source['source_kernel_task'], 'selected_source_revision': source['source_revision'], 'selected_source_evidence_revision': source['evidence_revision'], 'selected_source_blob': source['source_blob'], 'selected_source_sha256': source['source_sha256'], 'selected_source_path': source['source_path'], 'selected_source_entrypoint': source['entrypoint'], 'child_selected_route': child_outputs.get('selected_route'), 'child_route_trace': child_trace, 'dispatch_key': 'K', 'dispatch_value': K, 'owned_k': [OVERLAY_K] if K == OVERLAY_K else list(BASE_K), 'caller_output_data_ptr': caller_ptr, 'returned_output_data_ptr': returned_ptr, 'caller_owned_output': True, 'output_dtype': 'int32', 'compute_kernel_count': 1, 'padding_count': 0, 'pack_count': 0, 'fallback_contract_regions': [], 'residual_contract_regions': [], 'global_workspace': False, 'materialized_padding': False, 'production_route': 'weave_only', 'launch_grid': child_trace.get('launch_grid'), 'producer_to_consumer': child_trace.get('producer_to_consumer'), 'B': B, 'N': N, 'D': D, 'K': K}}
