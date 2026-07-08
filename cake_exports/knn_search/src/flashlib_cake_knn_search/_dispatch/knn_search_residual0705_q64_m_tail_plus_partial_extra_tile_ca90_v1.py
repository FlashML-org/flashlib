"""Exact M262145/Q64 split producer with a bounded extra-row owner.

Minimum target architecture: sm_100a.  The first 262144 database rows retain
04b4's native-Q64 tcgen05 producer, XOR-2 warp-distributed K32+K32 state, exact
group/final merges, and caller-owned output ABI.  A separate warp owns each
query's only valid row in the extra partial M64 tile, computes that row once,
and strictly inserts it into the already sorted caller K64 output.  The tail
handoff is device-side and stream-ordered; it does not rescan the full database,
materialize padding, or use a host/reference merge.
"""
from __future__ import annotations
from json import loads as _json_loads
from .._dispatch_runtime import _capture_cuTensorMapEncodeTiled, _decode_capture, _import_dispatch_module, _ir_proxy
import math
from typing import Any
from .. import _dispatch_runtime as contract
from . import knn_search_residual0705_q64_warp_distributed_state_04b4_v1 as full_04b4
from .._dispatch_runtime import detect_gpu_arch
FULL_M = 262144
EXTRA_ROWS = 1
D_STATIC = 256
K_MAX = 64
TAIL_THREADS = 256
TAIL_WARPS = TAIL_THREADS // 32
TARGET_SHAPE = 'residual_q64_m_tail_plus'
TARGET_SHAPES = [shape for shape in contract.CANONICAL_SHAPES if shape['label'] == TARGET_SHAPE]
TARGET_PARAMS = _decode_capture(_json_loads('{"__dict_items__": [["B", 1], ["Q", 64], ["M", 262145], ["D", 256], ["K", 64], ["dtype", "bfloat16"], ["seed", 704003], ["self_search", false], ["min_recall", 0.999], ["check_correctness", true], ["benchmark", true]]}'))
ROUTE = 'residual0705_q64_m_tail_plus_partial_extra_tile_ca90_tcgen05'
ENTRYPOINT = 'loom.examples.weave.knn_search_residual0705_q64_m_tail_plus_partial_extra_tile_ca90_v1:launch_for_eval'
ALLOWED_ARCHES = {'sm_100a', 'sm_103a'}
fallback = full_04b4.fallback
_TAIL_KERNELS: dict[str, Any] = {}
knn_search_q64_m_tail_plus_extra_row_merge_ca90_v1 = _decode_capture(_json_loads('{"__ir__": "knn_search_q64_m_tail_plus_extra_row_merge_ca90_v1", "arg_keys": ["queries", "database", "out_distances", "out_indices", "Q", "M"], "cluster_dims": [1, 1, 1], "computed_smem_bytes": 0, "constants": [["K_MAX_", 64]], "cta_group": 1, "threads": 256}'))
tail_ir = _decode_capture(_json_loads('{"__ir__": "knn_search_q64_m_tail_plus_extra_row_merge_ca90_v1", "arg_keys": ["queries", "database", "out_distances", "out_indices", "Q", "M"], "cluster_dims": [1, 1, 1], "computed_smem_bytes": 0, "constants": [["K_MAX_", 64]], "cta_group": 1, "threads": 256}'))
ir = _decode_capture(_json_loads('{"__ir__": "knn_search_q64_m_tail_plus_extra_row_merge_ca90_v1", "arg_keys": ["queries", "database", "out_distances", "out_indices", "Q", "M"], "cluster_dims": [1, 1, 1], "computed_smem_bytes": 0, "constants": [["K_MAX_", 64]], "cta_group": 1, "threads": 256}'))

def _compile_tail_kernel() -> Any:
    return _decode_capture(_json_loads('{"__kernel__": "dispatch_kernel_0504"}'))

def _valid_exact_abi(inputs: dict[str, Any]) -> bool:
    tensors = {'queries': inputs.get('queries'), 'database': inputs.get('database'), 'out_distances': inputs.get('out_distances'), 'out_indices': inputs.get('out_indices')}
    if any((tensor is None for tensor in tensors.values())):
        return False
    queries = tensors['queries']
    database = tensors['database']
    out_distances = tensors['out_distances']
    out_indices = tensors['out_indices']
    expected_shapes = {'queries': (1, 64, D_STATIC), 'database': (1, FULL_M + EXTRA_ROWS, D_STATIC), 'out_distances': (1, 64, K_MAX), 'out_indices': (1, 64, K_MAX)}
    return bool(str(queries.dtype) == 'torch.bfloat16' and str(database.dtype) == 'torch.bfloat16' and (str(out_distances.dtype) == 'torch.float32') and (str(out_indices.dtype) == 'torch.int32') and all((tensor.is_cuda for tensor in tensors.values())) and all((tensor.is_contiguous() for tensor in tensors.values())) and (tuple(queries.shape) == expected_shapes['queries']) and (tuple(database.shape) == expected_shapes['database']) and (tuple(out_distances.shape) == expected_shapes['out_distances']) and (tuple(out_indices.shape) == expected_shapes['out_indices']) and (len({str(tensor.device) for tensor in tensors.values()}) == 1))

def _use_target(inputs: dict[str, Any]) -> bool:
    return bool(not bool(inputs.get('force_fallback', False)) and (not bool(inputs.get('self_search', False))) and (int(inputs['B']) == TARGET_PARAMS['B']) and (int(inputs['Q']) == TARGET_PARAMS['Q']) and (int(inputs['M']) == TARGET_PARAMS['M']) and (int(inputs['D']) == TARGET_PARAMS['D']) and (int(inputs['K']) == TARGET_PARAMS['K']) and _valid_exact_abi(inputs) and (detect_gpu_arch() in ALLOWED_ARCHES))

def _full_inputs(inputs: dict[str, Any]) -> dict[str, Any]:
    full_inputs = dict(inputs)
    full_inputs['M'] = FULL_M
    return full_inputs

def launch_for_eval(inputs: dict[str, Any]) -> dict[str, Any]:
    if not _use_target(inputs):
        return fallback.launch_for_eval(inputs)
    if 'tail' not in _TAIL_KERNELS:
        _TAIL_KERNELS['tail'] = _compile_tail_kernel()
    full_04b4.launch_for_eval(_full_inputs(inputs))
    _TAIL_KERNELS['tail'].launch(grid=(math.ceil(int(inputs['Q']) / TAIL_WARPS), 1, 1), block=(TAIL_THREADS, 1, 1), args=[inputs['queries'], inputs['database'], inputs['out_distances'], inputs['out_indices'], int(inputs['Q']), int(inputs['M'])])
    return {'distances': inputs['out_distances'], 'indices': inputs['out_indices']}

def selected_route(inputs: dict[str, Any]) -> str:
    return ROUTE if _use_target(inputs) else fallback.selected_route(inputs)

def route_info(inputs: dict[str, Any]) -> dict[str, Any]:
    if not _use_target(inputs):
        return fallback.route_info(inputs)
    return {'route': ROUTE, 'selected_route': ROUTE, 'selected_entrypoint': ENTRYPOINT, 'route_kind': 'specialized', 'route_source': 'q64-m-tail-plus-partial-extra-tile-ca90', 'production_policy': 'weave_only', 'arch_requirement': 'sm_100a', 'supported_arches': sorted(ALLOWED_ARCHES), 'full_tile_rows': FULL_M, 'partial_tile_valid_rows': EXTRA_ROWS, 'full_producer': full_04b4.ROUTE, 'producer_layout': '04b4 M262144 tcgen05 scan with XOR-2 K32+K32 state; one warp-owned D256 reduction for the only valid extra-tile row', 'producer_to_consumer_abi': '04b4 sorted caller K64 -> strict device-side one-candidate insertion -> same caller FP32/INT32 outputs', 'consumer_layout': 'one warp per query; lane0 stable K64 carry merge', 'handoff_sync': 'same-stream kernel order after 04b4 final merge', 'state_ownership': '04b4 XOR-2 full-tile owners plus one bounded tail warp per query', 'padding_tag': 'none', 'uses_materialized_padding': False, 'uses_kernel_padding': False, 'padding_overhead_timed': False, 'database_scans': 'full rows once plus extra row once', 'workspace_reuse': '04b4 cached exact-shape scratch; no tail scratch'}
