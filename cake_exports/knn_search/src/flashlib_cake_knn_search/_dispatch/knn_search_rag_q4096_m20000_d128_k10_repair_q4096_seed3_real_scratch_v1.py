"""Exact Q4096 seed-3 tcgen05 col4 -> real-72-list pairlocal merge repair.

Minimum target architecture: sm_100a.  The producer is the checked tcgen05
col4 path and the consumer is the physical 72-list (18 splits x 4 cohorts)
pairlocal merge.  This is an additive exact-shape candidate; it does not alter
the shared dispatcher or benchmark registry.
"""
from __future__ import annotations
from json import loads as _json_loads
from .._dispatch_runtime import _capture_cuTensorMapEncodeTiled, _decode_capture, _import_dispatch_module, _ir_proxy
import math
from typing import Any
from . import knn_search_mma_split_v1 as mma
from . import knn_search_rag_q4096_m20000_d128_k10_f1d4_col4_checked_partials_v1 as capture
TARGET_KEY = (1, 4096, 20000, 128, 10)
PARTIAL_LIST_COUNT = 72
ACTIVE_MERGE_LANES = 24
LISTS_PER_ACTIVE_LANE = 3
TARGET_ROUTE = 'rag_q4096_m20000_d128_k10_tcgen05_col4_real_scratch_pairlocal72_repair'
FORCED_FALLBACK_ROUTE = ''.join([format(TARGET_ROUTE, ''), '_forced'])
PRODUCER_ABI = capture.PRODUCER_ABI
CONSUMER_ABI = 'q4096_pairlocal_topk_72_lists_three_per_active_lane'
ir = _decode_capture(_json_loads('{"__ir__": "knn_search_mma_split_partial_v1", "arg_keys": ["queries", "database", "partial_distances", "partial_indices", "B", "Q", "M", "split_m", "num_q_tiles", "total_m_tiles", "tiles_per_split"], "cluster_dims": [1, 1, 1], "computed_smem_bytes": 108800, "constants": [["K_MAX_", 10], ["EXPOSE_COL_COHORTS", 0], ["FULL_M_TILES", 0]], "cta_group": 1, "threads": 640}'))

def _is_target(inputs: dict[str, Any]) -> bool:
    return (int(inputs.get('B', 1)), int(inputs['Q']), int(inputs['M']), int(inputs['D']), int(inputs['K'])) == TARGET_KEY and (not bool(inputs.get('self_search', False))) and mma._tcgen05_capable_arch()

def selected_route(inputs: dict[str, Any]) -> str:
    if not _is_target(inputs):
        return 'weave_only_fallback'
    return FORCED_FALLBACK_ROUTE if bool(inputs.get('force_fallback', False)) else TARGET_ROUTE

def route_info(inputs: dict[str, Any]) -> dict[str, Any]:
    route = selected_route(inputs)
    return {'route': route, 'selected_route': route, 'production_policy': 'weave_only', 'primitive_family': 'tcgen05_mma', 'producer_abi': PRODUCER_ABI, 'consumer_abi': CONSUMER_ABI, 'partial_list_count': PARTIAL_LIST_COUNT, 'active_merge_lanes': ACTIVE_MERGE_LANES, 'lists_per_active_lane': LISTS_PER_ACTIVE_LANE}

def _ensure_kernels() -> None:
    if not mma._KNN_SEARCH_KERNELS:
        mma._KNN_SEARCH_KERNELS.update(mma._compile_kernels())

def _launch_pairlocal_merge(partial_dist: Any, partial_idx: Any, inputs: dict[str, Any], out_dist: Any, out_idx: Any) -> None:
    _ensure_kernels()
    bsz, q_rows, k = (int(inputs['B']), int(inputs['Q']), int(inputs['K']))
    mma._KNN_SEARCH_KERNELS['merge_q4096_pairlocal'].launch(grid=(bsz * q_rows, 1, 1), block=(mma.MERGE_THREADS, 1, 1), args=[partial_dist, partial_idx, out_dist, out_idx, bsz, q_rows, k, PARTIAL_LIST_COUNT, math.ceil(q_rows / mma.BLOCK_Q)], shared_mem=mma.MERGE_SMEM_BYTES)

def launch_for_eval(inputs: dict[str, Any]) -> dict[str, Any]:
    """Run the tcgen05 producer and consume all physical lists with pairlocal72."""
    if not _is_target(inputs):
        raise ValueError('pairlocal72 repair only accepts rag_q4096_m20000_d128_k10 on sm_100a/sm_103a')
    required = ('queries', 'database', 'out_distances', 'out_indices')
    missing = [name for name in required if name not in inputs]
    if missing:
        raise ValueError(''.join(['pairlocal72 repair missing tensor fields: ', format(missing, '')]))
    partial_dist, partial_idx = capture.capture_col4_partials(inputs)
    _launch_pairlocal_merge(partial_dist, partial_idx, inputs, inputs['out_distances'], inputs['out_indices'])
    return {'distances': inputs['out_distances'], 'indices': inputs['out_indices']}

def trace_real_scratch(inputs: dict[str, Any]) -> dict[str, Any]:
    """Check the producer-to-merge handoff for every Q tile against its 72 lists.

    This is a correctness diagnostic only.  It replays the real producer scratch
    and compares the pairlocal output with a device-side top-k over the same
    72x10 candidates, so the first mismatch identifies a merge-side divergence
    without synthesizing partial lists.
    """
    import torch
    if not _is_target(inputs):
        raise ValueError('real-scratch trace only accepts the exact target shape')
    partial_dist, partial_idx = capture.capture_col4_partials(inputs)
    q_rows, k = (int(inputs['Q']), int(inputs['K']))
    trace_dist = torch.empty_like(inputs['out_distances'])
    trace_idx = torch.empty_like(inputs['out_indices'])
    _launch_pairlocal_merge(partial_dist, partial_idx, inputs, trace_dist, trace_idx)
    candidates_d = partial_dist.permute(0, 1, 3, 2, 4).reshape(1, q_rows, PARTIAL_LIST_COUNT * mma.K_MAX)
    candidates_i = partial_idx.permute(0, 1, 3, 2, 4).reshape(1, q_rows, PARTIAL_LIST_COUNT * mma.K_MAX)
    expected_d, positions = torch.topk(candidates_d, k=k, dim=-1, largest=False, sorted=True)
    expected_i = candidates_i.gather(-1, positions)
    mismatch = (trace_idx != expected_i) | ~torch.isclose(trace_dist, expected_d, atol=0.01, rtol=0.01)
    mismatch_rows = torch.nonzero(mismatch.any(dim=-1), as_tuple=False)
    first = None if mismatch_rows.numel() == 0 else int(mismatch_rows[0, 1].cpu())
    return {'passed': first is None, 'producer_abi': PRODUCER_ABI, 'consumer_abi': CONSUMER_ABI, 'partial_list_count': PARTIAL_LIST_COUNT, 'q_tiles_checked': math.ceil(q_rows / mma.BLOCK_Q), 'q_rows_checked': q_rows, 'first_divergence_q_row': first, 'divergent_q_rows': int(mismatch.any(dim=-1).sum().cpu())}
