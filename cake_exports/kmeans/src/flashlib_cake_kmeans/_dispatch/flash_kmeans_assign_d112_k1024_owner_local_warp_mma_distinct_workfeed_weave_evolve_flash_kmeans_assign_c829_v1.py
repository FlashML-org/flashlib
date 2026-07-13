"""D112 K1024 owner-local streamed warp-MMA path (minimum arch: sm_90a).

One ordinary CTA owns one N64 point tile.  Warp 0 loads the point tile once
and streams alternating K32 centroid slabs.  Four row-owner warps retain the
same 16 point rows for the full K1024 scan, consume each slab with
``ldmatrix`` + ``mma.sync``, and write caller-owned cluster IDs directly.

This physical DAG has no K-sharded CTA owners, cluster/DSM handoff, TMEM,
global workspace, padding, pack kernel, or post-kernel reduction.
"""
from __future__ import annotations
from json import loads as _json_loads
from .._dispatch_runtime import _capture_cuTensorMapEncodeTiled, _decode_capture, _import_dispatch_module, _ir_proxy
from functools import lru_cache
from typing import Any
BLOCK_N = 64
BLOCK_K = 32
FEAT_D = 112
FEATURE_TILES = FEAT_D // 16
LOAD_WARPS = 1
COMPUTE_WARPS = 4
THREADS = (LOAD_WARPS + COMPUTE_WARPS) * 32
K_TILES = 1024 // BLOCK_K
K_TILE_PAIRS = K_TILES // 2
X_ELEMS = BLOCK_N * FEAT_D
C_ELEMS = BLOCK_K * FEAT_D
X_BYTES = X_ELEMS * 2
C_BYTES = C_ELEMS * 2
X_OFFSET = 0
C0_OFFSET = X_OFFSET + X_BYTES
C1_OFFSET = C0_OFFSET + C_BYTES
SMEM_BYTES = C1_OFFSET + C_BYTES
ROUTE_ID = 'd112_k1024_owner_local_warp_mma_distinct_workfeed_weave_evolve_flash_kmeans_assign_c829_v1'
SEED_ID = 'd112-k1024-owner-local-warp-mma-distinct-workfeed-weave-evolve-flash-kmeans-assign-c829-v1'
TARGET_SHAPE = 'post_d895_d112_b4_n8192_k1024_d112'
CAPABILITY_SHAPE_KEYS = (TARGET_SHAPE, 'adjacent_ef00_d112_odd_tail_b4_n3712_k1024_d112')
flash_kmeans_assign_d112_k1024_owner_local_warp_mma_distinct_workfeed_c829_v1 = _decode_capture(_json_loads('{"__ir__": "flash_kmeans_assign_d112_k1024_owner_local_warp_mma_distinct_workfeed_c829_v1", "arg_keys": ["x", "centroids", "c_sq", "out", "B", "N", "D", "K", "num_n_tiles"], "cluster_dims": [1, 1, 1], "computed_smem_bytes": 29696, "constants": [], "cta_group": 1, "threads": 160}'))
ir = _decode_capture(_json_loads('{"__ir__": "flash_kmeans_assign_d112_k1024_owner_local_warp_mma_distinct_workfeed_c829_v1", "arg_keys": ["x", "centroids", "c_sq", "out", "B", "N", "D", "K", "num_n_tiles"], "cluster_dims": [1, 1, 1], "computed_smem_bytes": 29696, "constants": [], "cta_group": 1, "threads": 160}'))

def _cuda_include_dirs() -> list[str]:
    from .._dispatch_runtime import _cuda_include_dirs as include_dirs
    return include_dirs()

def _compiled() -> tuple[bytes, str, int, int]:
    return _decode_capture(_json_loads('{"__tuple__": [{"__kernel_source__": "dispatch_kernel_0035"}, "kernel_flash_kmeans_assign_d112_k1024_owner_local_warp_mma_distinct_workfeed_c829_v1", 29696, 160]}'))

def supports_shape(*, B: int, D: int, N: int, K: int) -> bool:
    return B == 4 and D == FEAT_D and (N in (8192, 3712)) and (K == 1024)

def _shape_key(*, B: int, N: int, D: int, K: int) -> str:
    if (B, N, D, K) == (4, 8192, 112, 1024):
        return CAPABILITY_SHAPE_KEYS[0]
    if (B, N, D, K) == (4, 3712, 112, 1024):
        return CAPABILITY_SHAPE_KEYS[1]
    raise ValueError(''.join(['unassigned D112 K1024 capability shape ', format(repr((B, N, D, K)), '')]))

def launch_for_eval(inputs: dict[str, Any]) -> dict[str, Any]:
    import torch
    from .._dispatch_runtime import CUDAKernel
    B, N, D, K = (int(inputs[key]) for key in ('B', 'N', 'D', 'K'))
    if not supports_shape(B=B, D=D, N=N, K=K):
        raise ValueError('c829 owner-local warp-MMA seed admits only B4/D112/K1024 with N in (8192, 3712)')
    cubin, name, smem, threads = _compiled()
    grid = (B * (N // BLOCK_N), 1, 1)
    with CUDAKernel(cubin, name) as kernel:
        kernel.launch(grid=grid, block=(threads, 1, 1), args=[inputs['x'], inputs['centroids'], inputs['c_sq'], inputs['out'], B, N, D, K, N // BLOCK_N], shared_mem=smem, stream=torch.cuda.current_stream(), timeout_ms=120000)
    shape_key = _shape_key(B=B, N=N, D=D, K=K)
    return {'cluster_ids': inputs['out'], 'selected_route': ROUTE_ID, 'route_trace': {'shape_key': shape_key, 'selected_route': ROUTE_ID, 'selected_entrypoint': ''.join([format(__name__, ''), ':launch_for_eval']), 'selected_seed': SEED_ID, 'route_kind': 'single_cta_owner_local_k32_streamed_warp_mma', 'primitive_family': 'warp_mma_sync', 'compute_kernel_count': 1, 'cluster_ctas': 1, 'cta_group': 1, 'point_tile': BLOCK_N, 'centroid_tile': BLOCK_K, 'loader_warps': LOAD_WARPS, 'row_owner_warps': COMPUTE_WARPS, 'producer_ownership': 'warp 0 stages one shared point tile and alternating full-K centroid slabs', 'consumer_ownership': 'four row-owner warps retain disjoint 16-row spans for all K1024', 'producer_to_consumer': 'bulk_gmem_to_double_buffered_smem_to_ldmatrix_to_mma_sync_to_owner_local_argmax_to_caller_cluster_ids', 'split_k_reduction': 'none', 'global_workspace': False, 'padding_count': 0, 'pack_count': 0, 'fallback_contract_regions': [], 'residual_contract_regions': [], 'launch_grid': grid, 'num_n_tiles': N // BLOCK_N, 'shared_pool_bytes': SMEM_BYTES, 'shared_storage_bytes': int(ir.computed_smem_bytes)}}
