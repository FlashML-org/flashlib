"""D112 K-general direct consumer handoff (minimum architecture: sm_90a).

This additive full-bucket entrypoint retains the admitted e5e1 Weave IR:
direct canonical centroid stages feed ``ldmatrix.x2`` and ``mma.sync`` before
the owner-local argmax/key merge writes caller-owned cluster IDs.  The lane is
compiled and validated explicitly for ``sm_100a``.  Every admitted K is a
multiple of the eight CTA owners, two consumer groups, and the K8 MMA tile, so
the same dynamic loop topology covers the complete 14-row D112 bucket.
"""
from __future__ import annotations
from json import loads as _json_loads
from .._dispatch_runtime import _capture_cuTensorMapEncodeTiled, _decode_capture, _import_dispatch_module, _ir_proxy
from typing import Any
from . import flash_kmeans_assign_d112_consumer_scoped_direct_mma_view_handoff_weave_evolve_flash_kmeans_assign_e5e1_v1 as _direct
BLOCK_N = _direct.BLOCK_N
BLOCK_K = _direct.BLOCK_K
FEAT_D = _direct.FEAT_D
THREADS = _direct.THREADS
CLUSTER_CTAS = _direct.CLUSTER_CTAS
CONSUMER_GROUPS = _direct.CONSUMER_GROUPS
PIPELINE_STAGES = _direct.PIPELINE_STAGES
SUPPORTED_K = (256, 512, 768, 1024, 4096, 8192)
K_PARTITION_QUANTUM = CLUSTER_CTAS * CONSUMER_GROUPS * BLOCK_K
ROUTE_ID = 'd112_direct_handoff_full_bucket_weave_evolve_flash_kmeans_assign_4549_v1'
SEED_ID = 'd112-direct-handoff-full-bucket-weave-evolve-flash-kmeans-assign-4549-v1'
TARGET_SHAPE = 'physical_D112_padded_14'
cluster_ir = _decode_capture(_json_loads('{"__ir__": "flash_kmeans_assign_d112_consumer_scoped_direct_mma_view_handoff_e5e1_v1", "arg_keys": ["x", "centroids", "c_sq", "out", "B", "N", "D", "K", "num_n_tiles"], "cluster_dims": [8, 1, 1], "computed_smem_bytes": 46592, "constants": [], "cta_group": 1, "threads": 256}'))
ir = _decode_capture(_json_loads('{"__ir__": "flash_kmeans_assign_d112_consumer_scoped_direct_mma_view_handoff_e5e1_v1", "arg_keys": ["x", "centroids", "c_sq", "out", "B", "N", "D", "K", "num_n_tiles"], "cluster_dims": [8, 1, 1], "computed_smem_bytes": 46592, "constants": [], "cta_group": 1, "threads": 256}'))

def supports_shape(*, D: int, N: int, K: int) -> bool:
    return D == FEAT_D and N % BLOCK_N == 0 and (K in SUPPORTED_K) and (K >= K_PARTITION_QUANTUM) and (K % K_PARTITION_QUANTUM == 0)

def launch_for_eval(inputs: dict[str, Any]) -> dict[str, Any]:
    import torch
    from .._dispatch_runtime import CUDAKernel
    B, N, D, K = (int(inputs[key]) for key in ('B', 'N', 'D', 'K'))
    if not supports_shape(D=D, N=N, K=K):
        raise ValueError('D112 direct-handoff full bucket requires D=112, N%64=0, and K in (256, 512, 768, 1024, 4096, 8192)')
    cubin, name, smem, threads = _direct._compiled_cluster()
    grid = (B * (N // BLOCK_N) * CLUSTER_CTAS, 1, 1)
    with CUDAKernel(cubin, name) as kernel:
        kernel.launch_cluster(grid=grid, block=(threads, 1, 1), args=[inputs['x'], inputs['centroids'], inputs['c_sq'], inputs['out'], B, N, D, K, N // BLOCK_N], cluster_dims=(CLUSTER_CTAS, 1, 1), shared_mem=smem, stream=torch.cuda.current_stream(), timeout_ms=120000)
    return {'cluster_ids': inputs['out'], 'selected_route': ROUTE_ID, 'route_trace': {'shape_key': TARGET_SHAPE, 'selected_route': ROUTE_ID, 'selected_entrypoint': ''.join([format(__name__, ''), ':launch_for_eval']), 'selected_seed': SEED_ID, 'route_kind': 'eight_owner_k_general_consumer_scoped_direct_mma_view_handoff', 'cluster_ctas': CLUSTER_CTAS, 'consumer_groups_per_owner': CONSUMER_GROUPS, 'centroid_pipeline_stages': PIPELINE_STAGES, 'centroids_per_owner': K // CLUSTER_CTAS, 'centroids_per_consumer_group': K // (CLUSTER_CTAS * CONSUMER_GROUPS), 'producer_ownership': 'ranks 0..7 stage points locally; each four-warp group owns two canonical centroid stages', 'reuse_mechanism': 'direct TMA canonical stages consumed by ldmatrix.x2 with independent 128-thread group barriers', 'producer_to_consumer': 'TMA_to_consumer_scoped_canonical_stage_to_ldmatrix_x2_to_mma_sync_to_argmax_to_onchip_cluster_ids', 'overlap_edge': 'centroid_t_plus_1_direct_TMA -> current_centroid_t_mma_and_argmax', 'stage_storage_bytes': _direct.C_BYTES, 'split_k_reduction': 'shared-memory intra-owner key merge followed by on-chip cluster merge; no global workspace', 'launch_grid': grid, 'supported_k': SUPPORTED_K, 'residual_contract_regions': []}}
