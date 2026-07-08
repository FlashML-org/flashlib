"""D112 shared-point dual-independent-issuer tcgen05/TMEM assignment.

Minimum architecture: sm_100a.  A dedicated point producer stages the seven
native D16 slices of each N128 tile exactly once.  Two independently
synchronized centroid-owner pipelines consume that shared point tile through
distinct tcgen05 issuer warps and disjoint K64 TMEM score slots, merge ordered
argmax keys on chip, and write the caller-owned ``cluster_ids`` tensor.  Every
issuer owns its operand-ready, score-complete, readback-complete, and recycle
phase state.  The production launch is Weave-only: there is no D128 adapter,
DSM point fanout, global workspace, or external fallback.
"""
from __future__ import annotations
from json import loads as _json_loads
from .._dispatch_runtime import _capture_cuTensorMapEncodeTiled, _decode_capture, _import_dispatch_module, _ir_proxy
from functools import lru_cache
from typing import Any
BLOCK_N, BLOCK_K, FEAT_D, FEAT_CHUNK = (128, 64, 112, 16)
FEATURE_TILES = FEAT_D // FEAT_CHUNK
OWNER_SLOTS, COMPUTE_WARPS = (2, 4)
THREADS = 13 * 32
X_SLICE_BYTES = BLOCK_N * FEAT_CHUNK * 2
C_SLICE_BYTES = BLOCK_K * FEAT_CHUNK * 2
LOCAL_KEY_BYTES = BLOCK_N * 8
X_BYTES = FEATURE_TILES * X_SLICE_BYTES
C0_OFFSET = X_BYTES
C1_OFFSET = C0_OFFSET + C_SLICE_BYTES
KEY0_OFFSET = C1_OFFSET + C_SLICE_BYTES
KEY1_OFFSET = KEY0_OFFSET + LOCAL_KEY_BYTES
SMEM_BYTES = KEY1_OFFSET + LOCAL_KEY_BYTES
U32_MASK = 4294967295
TCGEN05_ARCHES = frozenset({'sm_100a', 'sm_103a'})
ROUTE_ID = 'd112_shared_point_dual_issuer_tcgen05_weave_evolve_flash_kmeans_assign_6e1e_v1'
SEED_ID = 'd112-shared-point-dual-issuer-tcgen05-weave-evolve-flash-kmeans-assign-6e1e-v1'
TARGET_SHAPE = 'physical_D112_padded_14'
CAPABILITY_SHAPE = 'post_d895_d112_b4_n8192_k1024_d112'
PHYSICAL_DAG = 'one_N128_D112_point_producer->seven_persistent_D16_slices->two_independent_K64_tcgen05_issuers->two_disjoint_TMEM_argmax_consumers->local_packed_key_merge->caller_cluster_ids'
flash_kmeans_assign_d112_shared_point_dual_issuer_tcgen05_6e1e_v1 = _decode_capture(_json_loads('{"__ir__": "flash_kmeans_assign_d112_shared_point_dual_issuer_tcgen05_6e1e_v1", "arg_keys": ["x_tmap", "c_tmap", "c_sq", "out", "B", "N", "D", "K", "num_n_tiles", "K_tiles"], "cluster_dims": [1, 1, 1], "computed_smem_bytes": 35840, "constants": [], "cta_group": 1, "threads": 416}'))
ir = _decode_capture(_json_loads('{"__ir__": "flash_kmeans_assign_d112_shared_point_dual_issuer_tcgen05_6e1e_v1", "arg_keys": ["x_tmap", "c_tmap", "c_sq", "out", "B", "N", "D", "K", "num_n_tiles", "K_tiles"], "cluster_dims": [1, 1, 1], "computed_smem_bytes": 35840, "constants": [], "cta_group": 1, "threads": 416}'))

def _cuda_include_dirs() -> list[str]:
    from .._dispatch_runtime import _cuda_include_dirs as include_dirs
    return include_dirs()

def _require_tcgen05_arch() -> str:
    from .._dispatch_runtime import detect_gpu_arch
    arch = detect_gpu_arch()
    if arch not in TCGEN05_ARCHES:
        supported = ', '.join(sorted(TCGEN05_ARCHES))
        raise RuntimeError(''.join([format(ROUTE_ID, ''), ' requires tcgen05/TMEM (', format(supported, ''), '); detected ', format(arch, '')]))
    return arch

def _compiled() -> tuple[bytes, str, int, int]:
    return _decode_capture(_json_loads('{"__tuple__": [{"__kernel_source__": "dispatch_kernel_0355"}, "kernel_flash_kmeans_assign_d112_shared_point_dual_issuer_tcgen05_6e1e_v1", 35840, 416]}'))

def supports_shape(*, D: int, N: int, K: int) -> bool:
    return D == FEAT_D and N % BLOCK_N == 0 and (K % (BLOCK_K * OWNER_SLOTS) == 0)

def _make_tmaps(inputs: dict[str, Any]) -> tuple[Any, Any]:
    from .._dispatch_runtime import create_tensor_map_3d_32b
    bsz, n_points, n_clusters = (int(inputs[key]) for key in ('B', 'N', 'K'))
    return (create_tensor_map_3d_32b(inputs['x'].data_ptr(), bsz * n_points, BLOCK_N, FEAT_D, FEAT_CHUNK), create_tensor_map_3d_32b(inputs['centroids'].data_ptr(), bsz * n_clusters, BLOCK_K, FEAT_D, FEAT_CHUNK))

def launch_for_eval(inputs: dict[str, Any]) -> dict[str, Any]:
    import torch
    from .._dispatch_runtime import CUDAKernel
    _require_tcgen05_arch()
    B, N, D, K = (int(inputs[key]) for key in ('B', 'N', 'D', 'K'))
    if not supports_shape(D=D, N=N, K=K):
        raise ValueError(''.join([format(ROUTE_ID, ''), ' requires D=112, N%128=0, and K%128=0']))
    x_tmap, c_tmap = _make_tmaps(inputs)
    cubin, name, smem, threads = _compiled()
    grid = (B * (N // BLOCK_N), 1, 1)
    with CUDAKernel(cubin, name) as kernel:
        kernel.launch(grid=grid, block=(threads, 1, 1), args=[x_tmap, c_tmap, inputs['c_sq'], inputs['out'], B, N, D, K, N // BLOCK_N, K // BLOCK_K], shared_mem=smem, stream=torch.cuda.current_stream())
    return {'cluster_ids': inputs['out'], 'selected_route': ROUTE_ID, 'route_trace': {'shape_key': TARGET_SHAPE, 'capability_shape': CAPABILITY_SHAPE, 'selected_route': ROUTE_ID, 'selected_entrypoint': ''.join([format(__name__, ''), ':launch_for_eval']), 'selected_seed': SEED_ID, 'primitive': 'native_D112_TMA+tcgen05_mma+TMEM', 'physical_dag': PHYSICAL_DAG, 'point_producer': 'warp0 loads seven persistent D16 slices once per N128 tile', 'centroid_ownership': 'slot0 owns even K64 tiles; slot1 owns odd K64 tiles', 'issuer_policy': 'two independent warp issuers with no shared completion barrier', 'issuer_ownership': [{'slot': 0, 'issuer_warp': 3, 'operand_ready': ['x_ready0', 'c_full0'], 'score_complete': 'score_full0', 'readback_complete': 'score_empty0', 'recycle': ['c_empty0', 'x_empty0']}, {'slot': 1, 'issuer_warp': 12, 'operand_ready': ['x_ready1', 'c_full1'], 'score_complete': 'score_full1', 'readback_complete': 'score_empty1', 'recycle': ['c_empty1', 'x_empty1']}], 'score_slots': [{'tmem_cols': [0, 64], 'issuer_warp': 3, 'compute_warps': [4, 5, 6, 7]}, {'tmem_cols': [64, 128], 'issuer_warp': 12, 'compute_warps': [8, 9, 10, 11]}], 'producer_to_consumer': 'shared_seven_D16_point_slices_to_two_independent_K64_tcgen05_issuers_to_disjoint_TMEM_argmax_consumers_to_local_ordered_key_merge_to_cluster_ids', 'output_ownership': 'caller_owned_inputs[out]', 'production_launches': [''.join(['kernel_', format(ir.symbol, '')])], 'fallback_contract_regions': [], 'residual_contract_regions': [], 'global_workspace': False, 'materialized_d128_adapter': False, 'launch_grid': grid}}

def benchmark_shared_point_dual_issuer_tcgen05_6e1e(*, shape_label: str=CAPABILITY_SHAPE, benchmark: bool=True) -> dict[str, Any]:
    """Registered one-row capability launcher for regression and sanitizer tools."""
    from .. import _dispatch_runtime as padded
    shape = next((row for row in padded.padded_bucket_shapes('d112_gap_pad_14') if row['label'] == shape_label))
    report = padded.evaluate(launch_for_eval, shapes=[shape], correctness=True, benchmark=benchmark, flashlib_baseline=False, benchmark_warmup_ms=20.0, benchmark_ms=100.0)
    row = report['per_shape'][shape_label]
    exact = bool(report['correctness']['all_correct']) and row.get('cluster_id_exact_match') is True
    result: dict[str, Any] = {'passed': exact, 'all_pass': exact, 'shape_label': shape_label, 'selected_route': row.get('selected_route'), 'cluster_id_exact_match': row.get('cluster_id_exact_match'), 'measurement_comparable': bool(row.get('measurement_comparable')) if benchmark else False}
    if benchmark:
        result.update({'tflops': float(row['tflops']), 'kernel_ms': float(row['kernel_ms']), 'timing_backend': row.get('timing_backend'), 'timing_backend_requested': row.get('timing_backend_requested'), 'timing_backend_fallback_reason': row.get('timing_backend_fallback_reason'), 'performance_comparable': bool(row.get('measurement_comparable')), 'measured_entrypoint': ''.join([format(__name__, ''), ':launch_for_eval'])})
    return result
