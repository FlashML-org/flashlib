"""D112 K1024 direct consumer-scoped MMA handoff (minimum architecture: sm_90a).

Eight CTA ranks own disjoint K/8 spans and two four-warp consumer groups own
disjoint halves of each span.  TMA writes each group's two canonical 8x112
centroid stages directly.  The owning consumer group loads those stages with
``ldmatrix.x2`` into the ``mma.sync`` B fragment, so there is no
canonical-to-MMA-view materialization.  Independent 128-thread named barriers
protect stage reuse without a 256-thread per-tile rendezvous.
"""
from __future__ import annotations
from json import loads as _json_loads
from .._dispatch_runtime import _capture_cuTensorMapEncodeTiled, _decode_capture, _import_dispatch_module, _ir_proxy
from functools import lru_cache
from typing import Any
BLOCK_N, BLOCK_K, FEAT_D, THREADS = (64, 8, 112, 256)
CLUSTER_CTAS, CONSUMER_GROUPS, PIPELINE_STAGES = (8, 2, 2)
SUPPORTED_K = (1024,)
X_ELEMS, C_ELEMS = (BLOCK_N * FEAT_D, BLOCK_K * FEAT_D)
X_BYTES, C_BYTES = (X_ELEMS * 2, C_ELEMS * 2)
CENTROID_STAGE_COUNT = CONSUMER_GROUPS * PIPELINE_STAGES
SCORE_BYTES = CONSUMER_GROUPS * 4 * 16 * BLOCK_K * 4
LOCAL_KEY_BYTES = BLOCK_N * 8
GROUP_KEY_BYTES = CONSUMER_GROUPS * LOCAL_KEY_BYTES
CLUSTER_KEY_BYTES = CLUSTER_CTAS * LOCAL_KEY_BYTES
U32_MASK = 4294967295
X_RAW_OFFSET = 0
C_DIRECT_OFFSET = X_RAW_OFFSET + X_BYTES
X_MMA_OFFSET = C_DIRECT_OFFSET + CENTROID_STAGE_COUNT * C_BYTES
SCORE_OFFSET = X_MMA_OFFSET + X_BYTES
GROUP_KEY_OFFSET = SCORE_OFFSET + SCORE_BYTES
LOCAL_KEY_OFFSET = GROUP_KEY_OFFSET + GROUP_KEY_BYTES
CLUSTER_KEY_OFFSET = LOCAL_KEY_OFFSET + LOCAL_KEY_BYTES
SMEM_BYTES = CLUSTER_KEY_OFFSET + CLUSTER_KEY_BYTES
ROUTE_ID = 'd112_consumer_scoped_direct_mma_view_handoff_weave_evolve_flash_kmeans_assign_e5e1_v1'
SEED_ID = 'd112-consumer-scoped-direct-mma-view-handoff-weave-evolve-flash-kmeans-assign-e5e1-v1'
TARGET_SHAPE = 'physical_D112_padded_14'
flash_kmeans_assign_d112_consumer_scoped_direct_mma_view_handoff_e5e1_v1 = _decode_capture(_json_loads('{"__ir__": "flash_kmeans_assign_d112_consumer_scoped_direct_mma_view_handoff_e5e1_v1", "arg_keys": ["x", "centroids", "c_sq", "out", "B", "N", "D", "K", "num_n_tiles"], "cluster_dims": [8, 1, 1], "computed_smem_bytes": 46592, "constants": [], "cta_group": 1, "threads": 256}'))
cluster_ir = _decode_capture(_json_loads('{"__ir__": "flash_kmeans_assign_d112_consumer_scoped_direct_mma_view_handoff_e5e1_v1", "arg_keys": ["x", "centroids", "c_sq", "out", "B", "N", "D", "K", "num_n_tiles"], "cluster_dims": [8, 1, 1], "computed_smem_bytes": 46592, "constants": [], "cta_group": 1, "threads": 256}'))
ir = _decode_capture(_json_loads('{"__ir__": "flash_kmeans_assign_d112_consumer_scoped_direct_mma_view_handoff_e5e1_v1", "arg_keys": ["x", "centroids", "c_sq", "out", "B", "N", "D", "K", "num_n_tiles"], "cluster_dims": [8, 1, 1], "computed_smem_bytes": 46592, "constants": [], "cta_group": 1, "threads": 256}'))

def _cuda_include_dirs() -> list[str]:
    from .._dispatch_runtime import _cuda_include_dirs as dirs
    return dirs()

def _compiled_cluster() -> tuple[bytes, str, int, int]:
    return _decode_capture(_json_loads('{"__tuple__": [{"__kernel_source__": "dispatch_kernel_0024"}, "kernel_flash_kmeans_assign_d112_consumer_scoped_direct_mma_view_handoff_e5e1_v1", 46592, 256]}'))

def supports_shape(*, D: int, N: int, K: int) -> bool:
    return D == FEAT_D and N % BLOCK_N == 0 and (K in SUPPORTED_K)

def launch_for_eval(inputs: dict[str, Any]) -> dict[str, Any]:
    import torch
    from .._dispatch_runtime import CUDAKernel
    B, N, D, K = (int(inputs[key]) for key in ('B', 'N', 'D', 'K'))
    if not supports_shape(D=D, N=N, K=K):
        raise ValueError('consumer-scoped direct MMA handoff e5e1 requires D=112, N%64=0, and K=1024')
    cubin, name, smem, threads = _compiled_cluster()
    grid = (B * (N // BLOCK_N) * CLUSTER_CTAS, 1, 1)
    with CUDAKernel(cubin, name) as kernel:
        kernel.launch_cluster(grid=grid, block=(threads, 1, 1), args=[inputs['x'], inputs['centroids'], inputs['c_sq'], inputs['out'], B, N, D, K, N // BLOCK_N], cluster_dims=(CLUSTER_CTAS, 1, 1), shared_mem=smem, stream=torch.cuda.current_stream(), timeout_ms=120000)
    return {'cluster_ids': inputs['out'], 'selected_route': ROUTE_ID, 'route_trace': {'shape_key': TARGET_SHAPE, 'selected_route': ROUTE_ID, 'selected_entrypoint': ''.join([format(__name__, ''), ':launch_for_eval']), 'selected_seed': SEED_ID, 'route_kind': 'eight_owner_consumer_scoped_direct_mma_view_handoff', 'cluster_ctas': CLUSTER_CTAS, 'consumer_groups_per_owner': CONSUMER_GROUPS, 'centroid_pipeline_stages': PIPELINE_STAGES, 'centroids_per_owner': K // CLUSTER_CTAS, 'centroids_per_consumer_group': K // (CLUSTER_CTAS * CONSUMER_GROUPS), 'producer_ownership': 'ranks 0..7 stage points locally; each four-warp group owns two canonical centroid stages', 'reuse_mechanism': 'direct TMA canonical stages consumed by ldmatrix.x2 with independent 128-thread group barriers', 'producer_to_consumer': 'TMA_to_consumer_scoped_canonical_stage_to_ldmatrix_x2_to_mma_sync_to_argmax_to_onchip_cluster_ids', 'overlap_edge': 'centroid_t_plus_1_direct_TMA -> current_centroid_t_mma_and_argmax', 'stage_storage_bytes': C_BYTES, 'split_k_reduction': 'shared-memory intra-owner key merge followed by on-chip cluster merge; no global workspace', 'launch_grid': grid, 'residual_contract_regions': []}}
