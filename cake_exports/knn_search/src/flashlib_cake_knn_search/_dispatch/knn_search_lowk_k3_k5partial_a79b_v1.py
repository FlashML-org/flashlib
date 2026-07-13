"""A79B Q4096/M20000 K3 seed using K5-capacity partials.

Minimum target architecture: sm_100a. This additive bucket-kernel candidate
targets only ``B=1, Q=4096, M=20000, D=128, K=3`` and preserves the incumbent
split-4 route for guard misses. The eval path stays Weave-only: an inherited
tcgen05 K5 partial producer writes five candidates per split, then a K3 merge
consumer reads that five-slot layout and writes the contract top-3 outputs.
"""
from __future__ import annotations
from json import loads as _json_loads
from .._dispatch_runtime import _capture_cuTensorMapEncodeTiled, _decode_capture, _import_dispatch_module, _ir_proxy
import math
from typing import Any
from .._dispatch_runtime import dc as dc
from .._dispatch_runtime import select_named_shapes
from . import knn_search_mma_split_v1 as mma
from . import knn_search_q4096_lowk_k5partial_dispatch_0613_r49_48e9_v1 as k5
from . import knn_search_q4096_split4_tiestable_0612_r15_4e2c_v1 as parent
K3_TARGET = 3
K_PARTIAL_STRIDE = k5.K_PARTIAL_MAX
Q4096_ROWS = 4096
Q4096_LOWK_M = 20000
Q4096_LOWK_SPLIT_M = 4
THREADS = k5.THREADS
MERGE_THREADS = k5.MERGE_THREADS
BLOCK_Q = k5.BLOCK_Q
BLOCK_M = k5.BLOCK_M
D_STATIC = k5.D_STATIC
LOWK_MMA_SMEM_BYTES = k5.LOWK_MMA_SMEM_BYTES
MERGE_SMEM_BYTES = mma.MERGE_SMEM_BYTES
partial_ir = _decode_capture(_json_loads('{"__ir__": "knn_search_q4096_lowk_k5partial_0613_r49_48e9_v1", "arg_keys": ["queries", "database", "partial_distances", "partial_indices", "B", "Q", "M", "split_m", "num_q_tiles", "total_m_tiles", "tiles_per_split"], "cluster_dims": [1, 1, 1], "computed_smem_bytes": 108800, "constants": [], "cta_group": 1, "threads": 640}'))
merge_k3_ir = _decode_capture(_json_loads('{"__ir__": "knn_search_lowk_k3_k5partial_a79b_merge_v1", "arg_keys": ["partial_distances", "partial_indices", "out_distances", "out_indices", "B", "Q", "K", "split_m", "num_q_tiles"], "cluster_dims": [1, 1, 1], "computed_smem_bytes": 0, "constants": [["K_STRIDE_", 5], ["K_OUT_MAX_", 3]], "cta_group": 1, "threads": 32}'))
ir = _decode_capture(_json_loads('{"__ir__": "knn_search_lowk_k3_k5partial_a79b_merge_v1", "arg_keys": ["partial_distances", "partial_indices", "out_distances", "out_indices", "B", "Q", "K", "split_m", "num_q_tiles"], "cluster_dims": [1, 1, 1], "computed_smem_bytes": 0, "constants": [["K_STRIDE_", 5], ["K_OUT_MAX_", 3]], "cta_group": 1, "threads": 32}'))
Q4096_K3_LABELS: tuple[str, ...] = ('blind_lowk_q4096_m20000_d128_k3',)
Q4096_K3_SHAPES = _decode_capture(_json_loads('[{"__dict_items__": [["label", "blind_lowk_q4096_m20000_d128_k3"], ["params", {"__dict_items__": [["B", 1], ["Q", 4096], ["M", 20000], ["D", 128], ["K", 3], ["dtype", "bfloat16"], ["seed", 610607], ["self_search", false], ["min_recall", 0.999]]}]]}]'))
ROUTE_Q4096_K3_K5PARTIAL_A79B = 'a79b_q4096_m20000_k3_k5partial_split4'
SHAPE_DISPATCH_REGISTRY: tuple[dict[str, str], ...] = ({'shape_key': 'a79b_exact_q4096_m20000_d128_k3_k5partial_split4', 'guard': 'B == 1 and Q == 4096 and M == 20000 and D == 128 and K == 3 and tcgen05', 'route': ROUTE_Q4096_K3_K5PARTIAL_A79B},)
_KNN_SEARCH_K3_K5PARTIAL_KERNELS: dict[str, Any] = {}
_KNN_SEARCH_K3_K5PARTIAL_SCRATCH: dict[tuple[int, int, int, int, int, int, int, str], tuple[Any, Any]] = {}

def _compile_k3_k5partial_kernels() -> dict[str, Any]:
    return _decode_capture(_json_loads('{"merge": {"__kernel__": "dispatch_kernel_0251"}, "partial": {"__kernel__": "dispatch_kernel_0158"}}'))

def _scratch_k3_k5partial(inputs: dict[str, Any], split_m: int, num_q_tiles: int) -> tuple[Any, Any]:
    import torch
    key = (int(inputs['B']), int(inputs['Q']), int(inputs['M']), int(split_m), int(num_q_tiles), K_PARTIAL_STRIDE, int(inputs['queries'].device.index or 0), str(inputs['queries'].dtype))
    cached = _KNN_SEARCH_K3_K5PARTIAL_SCRATCH.get(key)
    if cached is None:
        shape = (int(inputs['B']), int(num_q_tiles), int(split_m), BLOCK_Q, K_PARTIAL_STRIDE)
        cached = (torch.empty(shape, dtype=torch.float32, device=inputs['queries'].device), torch.empty(shape, dtype=torch.int32, device=inputs['queries'].device))
        _KNN_SEARCH_K3_K5PARTIAL_SCRATCH[key] = cached
    return cached

def _use_q4096_k3_k5partial(inputs: dict[str, Any]) -> bool:
    return int(inputs['B']) == 1 and int(inputs['Q']) == Q4096_ROWS and (int(inputs['M']) == Q4096_LOWK_M) and (int(inputs['D']) == D_STATIC) and (int(inputs['K']) == K3_TARGET) and (not bool(inputs.get('self_search', False))) and (not bool(inputs.get('force_fallback', False))) and mma._tcgen05_capable_arch()

def selected_route(inputs: dict[str, Any]) -> str:
    if _use_q4096_k3_k5partial(inputs):
        return ROUTE_Q4096_K3_K5PARTIAL_A79B
    return 'parent_q4096_split4_tiestable'

def _launch_q4096_k3_k5partial(inputs: dict[str, Any]) -> dict[str, Any]:
    import torch
    if not _KNN_SEARCH_K3_K5PARTIAL_KERNELS:
        _KNN_SEARCH_K3_K5PARTIAL_KERNELS.update(_compile_k3_k5partial_kernels())
    bsz = int(inputs['B'])
    q_rows = int(inputs['Q'])
    m_rows = int(inputs['M'])
    k = int(inputs['K'])
    num_q_tiles = math.ceil(q_rows / BLOCK_Q)
    total_m_tiles = math.ceil(m_rows / BLOCK_M)
    split_m = min(Q4096_LOWK_SPLIT_M, total_m_tiles)
    tiles_per_split = math.ceil(total_m_tiles / split_m)
    partial_dist, partial_idx = _scratch_k3_k5partial(inputs, split_m, num_q_tiles)
    _KNN_SEARCH_K3_K5PARTIAL_KERNELS['partial'].launch(grid=(bsz * num_q_tiles * split_m, 1, 1), block=(THREADS, 1, 1), args=[inputs['queries'], inputs['database'], partial_dist, partial_idx, bsz, q_rows, m_rows, split_m, num_q_tiles, total_m_tiles, tiles_per_split], shared_mem=LOWK_MMA_SMEM_BYTES)
    _KNN_SEARCH_K3_K5PARTIAL_KERNELS['merge'].launch(grid=(bsz * q_rows, 1, 1), block=(MERGE_THREADS, 1, 1), args=[partial_dist, partial_idx, inputs['out_distances'], inputs['out_indices'], bsz, q_rows, k, split_m, num_q_tiles], shared_mem=MERGE_SMEM_BYTES)
    return {'distances': inputs['out_distances'], 'indices': inputs['out_indices']}

def launch_for_eval(inputs: dict[str, Any]) -> dict[str, Any]:
    if _use_q4096_k3_k5partial(inputs):
        return _launch_q4096_k3_k5partial(inputs)
    return parent.launch_for_eval(inputs)

def _select_contract_shapes(shape_labels: str | tuple[str, ...] | list[str] | None):
    if shape_labels is None:
        return Q4096_K3_SHAPES
    return select_named_shapes(shape_labels)

def knn_search_compile_and_launch_lowk_k3_k5partial_a79b(*, benchmark: bool=True, shape_labels: str | tuple[str, ...] | list[str] | None=None) -> dict[str, Any]:
    from .._dispatch_runtime import evaluate
    result = evaluate(launch_for_eval, shapes=_select_contract_shapes(shape_labels), benchmark=benchmark)
    result['passed'] = bool(result.get('summary', {}).get('all_correct'))
    print(result)
    return result
