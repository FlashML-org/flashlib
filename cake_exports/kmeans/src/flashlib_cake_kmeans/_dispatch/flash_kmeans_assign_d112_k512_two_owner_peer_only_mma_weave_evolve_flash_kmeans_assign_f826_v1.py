"""D112 K512 two-owner peer-only warp-MMA path (minimum architecture: sm_90a).

Two independent ``cta_group=1`` owners split K512 into disjoint K256 spans.
Each owner uses two four-warp consumer groups with direct TMA centroid stages,
``ldmatrix.x2``, and ``mma.sync`` before locally merging packed argmax keys.
Rank 1 sends one 512-byte key slab to rank 0; rank 0 retains its local slab and
writes caller-owned ``int32`` cluster IDs without a self-copy or workspace.
"""
from __future__ import annotations
from json import loads as _json_loads
from .._dispatch_runtime import _capture_cuTensorMapEncodeTiled, _decode_capture, _import_dispatch_module, _ir_proxy
from functools import lru_cache
from typing import Any
BLOCK_N, BLOCK_K, FEAT_D, THREADS = (64, 8, 112, 256)
CLUSTER_CTAS, CONSUMER_GROUPS, PIPELINE_STAGES = (2, 2, 2)
SUPPORTED_K = (512,)
X_ELEMS, C_ELEMS = (BLOCK_N * FEAT_D, BLOCK_K * FEAT_D)
X_BYTES, C_BYTES = (X_ELEMS * 2, C_ELEMS * 2)
CENTROID_STAGE_COUNT = CONSUMER_GROUPS * PIPELINE_STAGES
SCORE_BYTES = CONSUMER_GROUPS * 4 * 16 * BLOCK_K * 4
LOCAL_KEY_BYTES = BLOCK_N * 8
GROUP_KEY_BYTES = CONSUMER_GROUPS * LOCAL_KEY_BYTES
PEER_KEY_BYTES = LOCAL_KEY_BYTES
U32_MASK = 4294967295
X_RAW_OFFSET = 0
C_DIRECT_OFFSET = X_RAW_OFFSET + X_BYTES
X_MMA_OFFSET = C_DIRECT_OFFSET + CENTROID_STAGE_COUNT * C_BYTES
SCORE_OFFSET = X_MMA_OFFSET + X_BYTES
GROUP_KEY_OFFSET = SCORE_OFFSET + SCORE_BYTES
LOCAL_KEY_OFFSET = GROUP_KEY_OFFSET + GROUP_KEY_BYTES
PEER_KEY_OFFSET = LOCAL_KEY_OFFSET + LOCAL_KEY_BYTES
SMEM_BYTES = PEER_KEY_OFFSET + PEER_KEY_BYTES
ROUTE_ID = 'd112_k512_two_owner_peer_only_mma_weave_evolve_flash_kmeans_assign_f826_v1'
SEED_ID = 'd112-k512-two-owner-peer-only-mma-weave-evolve-flash-kmeans-assign-f826-v1'
TARGET_SHAPE = 'adjacent_68cf_d112_tail_b5_n2944_k512_d112'
flash_kmeans_assign_d112_k512_two_owner_peer_only_mma_f826_v1 = _decode_capture(_json_loads('{"__ir__": "flash_kmeans_assign_d112_k512_two_owner_peer_only_mma_f826_v1", "arg_keys": ["x", "centroids", "c_sq", "out", "B", "N", "D", "K", "num_n_tiles"], "cluster_dims": [2, 1, 1], "computed_smem_bytes": 43008, "constants": [], "cta_group": 1, "threads": 256}'))
cluster_ir = _decode_capture(_json_loads('{"__ir__": "flash_kmeans_assign_d112_k512_two_owner_peer_only_mma_f826_v1", "arg_keys": ["x", "centroids", "c_sq", "out", "B", "N", "D", "K", "num_n_tiles"], "cluster_dims": [2, 1, 1], "computed_smem_bytes": 43008, "constants": [], "cta_group": 1, "threads": 256}'))
ir = _decode_capture(_json_loads('{"__ir__": "flash_kmeans_assign_d112_k512_two_owner_peer_only_mma_f826_v1", "arg_keys": ["x", "centroids", "c_sq", "out", "B", "N", "D", "K", "num_n_tiles"], "cluster_dims": [2, 1, 1], "computed_smem_bytes": 43008, "constants": [], "cta_group": 1, "threads": 256}'))

def _cuda_include_dirs() -> list[str]:
    from .._dispatch_runtime import _cuda_include_dirs as dirs
    return dirs()

def _compiled_cluster() -> tuple[bytes, str, int, int]:
    return _decode_capture(_json_loads('{"__tuple__": [{"__kernel_source__": "dispatch_kernel_0354"}, "kernel_flash_kmeans_assign_d112_k512_two_owner_peer_only_mma_f826_v1", 43008, 256]}'))

def supports_shape(*, D: int, N: int, K: int) -> bool:
    return D == FEAT_D and N % BLOCK_N == 0 and (K in SUPPORTED_K)

def launch_for_eval(inputs: dict[str, Any]) -> dict[str, Any]:
    import torch
    from .._dispatch_runtime import CUDAKernel
    B, N, D, K = (int(inputs[key]) for key in ('B', 'N', 'D', 'K'))
    if not supports_shape(D=D, N=N, K=K):
        raise ValueError('two-owner peer-only MMA f826 requires D=112, N%64=0, and K=512')
    caller_out = inputs['out']
    caller_ptr = int(caller_out.data_ptr())
    cubin, name, smem, threads = _compiled_cluster()
    num_n_tiles = N // BLOCK_N
    grid = (B * num_n_tiles * CLUSTER_CTAS, 1, 1)
    with CUDAKernel(cubin, name) as kernel:
        kernel.launch_cluster(grid=grid, block=(threads, 1, 1), args=[inputs['x'], inputs['centroids'], inputs['c_sq'], caller_out, B, N, D, K, num_n_tiles], cluster_dims=(CLUSTER_CTAS, 1, 1), shared_mem=smem, stream=torch.cuda.current_stream(), timeout_ms=120000)
    returned_ptr = int(caller_out.data_ptr())
    if returned_ptr != caller_ptr:
        raise ValueError('two-owner peer-only MMA did not retain caller-owned out')
    return {'cluster_ids': caller_out, 'selected_route': ROUTE_ID, 'route_trace': {'shape_key': TARGET_SHAPE, 'selected_route': ROUTE_ID, 'selected_entrypoint': ''.join([format(__name__, ''), ':launch_for_eval']), 'selected_seed': SEED_ID, 'route_kind': 'd112_k512_two_owner_peer_only_warp_mma', 'cluster_ctas': CLUSTER_CTAS, 'cta_group': 1, 'consumer_groups_per_owner': CONSUMER_GROUPS, 'centroid_pipeline_stages': PIPELINE_STAGES, 'centroids_per_owner': K // CLUSTER_CTAS, 'centroids_per_consumer_group': K // (CLUSTER_CTAS * CONSUMER_GROUPS), 'producer_ownership': 'ranks 0..1 independently own K256; each four-warp group owns two direct canonical centroid stages', 'reuse_mechanism': 'direct TMA canonical stages consumed by ldmatrix.x2 with independent 128-thread group barriers', 'producer_to_consumer': 'D112_TMA_to_ldmatrix_x2_to_mma_sync_to_local_two_group_argmax_to_one_peer_key_handoff_to_caller_cluster_ids', 'split_k_reduction': 'per-owner shared two-group key merge; one rank1-to-rank0 peer slab; rank0 retains local slab', 'remote_producer_count': 1, 'peer_key_transaction_bytes': PEER_KEY_BYTES, 'rank0_self_copy': False, 'caller_output_data_ptr': caller_ptr, 'returned_output_data_ptr': returned_ptr, 'caller_owned_output': True, 'output_dtype': 'int32', 'compute_kernel_count': 1, 'padding_count': 0, 'pack_count': 0, 'global_workspace': False, 'fallback_contract_regions': [], 'residual_contract_regions': [], 'launch_grid': grid, 'B': B, 'N': N, 'D': D, 'K': K, 'num_n_tiles': num_n_tiles}}
