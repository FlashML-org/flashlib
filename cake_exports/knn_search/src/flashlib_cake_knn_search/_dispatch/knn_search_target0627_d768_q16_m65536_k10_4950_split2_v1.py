"""Exact D768/Q16/M65536/K10 direct-stride tcgen05 split-2 kNN seed.

Minimum target architecture: sm_100a.  This additive auto-tuning candidate
keeps the N=256 tcgen05 producer and fixed-148 merge ABI, but groups two
database tiles per active producer CTA (128 active groups over 256 tiles).
The producer, partial top-10 lists, merge, and contract distances/indices are
all on the evaluated Weave path; dispatcher routing remains unchanged.
"""
from __future__ import annotations
from json import loads as _json_loads
from .._dispatch_runtime import _capture_cuTensorMapEncodeTiled, _decode_capture, _import_dispatch_module, _ir_proxy
from typing import Any
from .._dispatch_runtime import select_named_shapes
from . import knn_search_d768_q16_compatible_consumer_d810_v1 as n256
TARGET_LABELS: tuple[str, ...] = ('target0627_d768_q16_m65536_k10',)
TARGET_SHAPES = _decode_capture(_json_loads('[{"__dict_items__": [["label", "target0627_d768_q16_m65536_k10"], ["params", {"__dict_items__": [["B", 1], ["Q", 16], ["M", 65536], ["D", 768], ["K", 10], ["dtype", "bfloat16"], ["seed", 612107], ["self_search", false], ["min_recall", 0.999]]}]]}]'))
PHYSICAL_SPLIT_M = 148
ACTIVE_SPLIT_M = 128
TOTAL_M_TILES = 256
ROUTE = 'target0628_d768_q16_m65536_k10_4950_directstride_n256_split2_tcgen05'
partial_ir = _decode_capture(_json_loads('{"__ir__": "knn_search_q8_blockm256_two_stripe_tmem_8d4fe4ead6cd_v1", "arg_keys": ["queries", "database", "partial_distances", "partial_indices", "B", "Q", "M", "split_m", "num_q_tiles", "total_m_tiles", "active_split_m"], "cluster_dims": [1, 1, 1], "computed_smem_bytes": 170240, "constants": [["K_MAX_", 10], ["D_ORIG_", 768], ["NUM_D_PASSES_", 8], ["Q_NORM_PARTS_", 48]], "cta_group": 1, "threads": 640}'))
merge_ir = _decode_capture(_json_loads('{"__ir__": "knn_search_mma_split_merge_q128_const148_v1", "arg_keys": ["partial_distances", "partial_indices", "out_distances", "out_indices", "B", "Q", "K", "split_m", "num_q_tiles"], "cluster_dims": [1, 1, 1], "computed_smem_bytes": 0, "constants": [["K_MAX_", 10]], "cta_group": 1, "threads": 32}'))
ir = _decode_capture(_json_loads('{"__ir__": "knn_search_q8_blockm256_two_stripe_tmem_8d4fe4ead6cd_v1", "arg_keys": ["queries", "database", "partial_distances", "partial_indices", "B", "Q", "M", "split_m", "num_q_tiles", "total_m_tiles", "active_split_m"], "cluster_dims": [1, 1, 1], "computed_smem_bytes": 170240, "constants": [["K_MAX_", 10], ["D_ORIG_", 768], ["NUM_D_PASSES_", 8], ["Q_NORM_PARTS_", 48]], "cta_group": 1, "threads": 640}'))

def _matches(inputs: dict[str, Any]) -> bool:
    return n256.selected_route(inputs) == n256.TARGET_ROUTE

def selected_route_name(inputs: dict[str, Any]) -> str:
    return ROUTE if _matches(inputs) else 'unsupported_shape'

def route_info(inputs: dict[str, Any]) -> dict[str, Any]:
    if not _matches(inputs):
        return {'route': 'unsupported_shape', 'selected_route': 'unsupported_shape', 'production_policy': 'weave_only'}
    info = dict(n256.route_info(inputs))
    info.update({'route': ROUTE, 'selected_route': ROUTE, 'selected_entrypoint': 'loom.examples.weave.knn_search_target0627_d768_q16_m65536_k10_4950_split4_v1:launch_for_eval', 'route_source': 'tile-shape-search-directstride-n256-split2', 'selected_seed': 'weave-evolve-knn-search-4950-d768-q16-split2', 'tile_shape': [64, 256, 128], 'tile_grouping': 'split2: 128 active producer groups over 256 N256 database tiles', 'physical_split_m': PHYSICAL_SPLIT_M, 'active_split_m': ACTIVE_SPLIT_M})
    return info

def launch_for_eval(inputs: dict[str, Any]) -> dict[str, Any]:
    if not _matches(inputs):
        raise ValueError('supports only B=1,Q=16,M=65536,D=768,K=10 non-self search on sm_100a+')
    kernels = n256._DIRECT_MMA_KERNELS.get(n256.HIGH_D_MAX)
    if kernels is None:
        kernels = n256._compile_direct_mma_kernels(n256.HIGH_D_MAX)
        n256._DIRECT_MMA_KERNELS[n256.HIGH_D_MAX] = kernels
    partial_dist, partial_idx = n256._partial_scratch(inputs, PHYSICAL_SPLIT_M, 1)
    kernels['partial'].launch(grid=(PHYSICAL_SPLIT_M, 1, 1), block=(n256.THREADS, 1, 1), args=[inputs['queries'], inputs['database'], partial_dist, partial_idx, 1, n256.TARGET_Q, n256.TARGET_M, PHYSICAL_SPLIT_M, 1, TOTAL_M_TILES, ACTIVE_SPLIT_M], shared_mem=int(kernels['shared_mem']))
    n256._target_merge_kernel().launch(grid=(n256.TARGET_Q, 1, 1), block=(n256.MERGE_THREADS, 1, 1), args=[partial_dist, partial_idx, inputs['out_distances'], inputs['out_indices'], 1, n256.TARGET_Q, n256.K_MAX, PHYSICAL_SPLIT_M, 1], shared_mem=n256.MERGE_SMEM_BYTES)
    return {'distances': inputs['out_distances'], 'indices': inputs['out_indices']}
