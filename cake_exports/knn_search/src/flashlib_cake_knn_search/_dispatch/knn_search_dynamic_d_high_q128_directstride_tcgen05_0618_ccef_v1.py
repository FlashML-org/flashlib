"""Dynamic high-D q128/m65536 K10 kNN seed using direct-stride tcgen05 routes.

Minimum target architecture: sm_100a. This additive bucket seed owns
``B=1,Q=128,M=65536,K=10,D in {129,257,511}`` without a packed side buffer.
The partial producer stages directly from the original feature stride, uses
tcgen05 MMA for every 128-wide chunk, and zero-fills only the final partial
feature vector before the existing Q128 split-M merge consumer.
"""
from __future__ import annotations
from json import loads as _json_loads
from .._dispatch_runtime import _capture_cuTensorMapEncodeTiled, _decode_capture, _import_dispatch_module, _ir_proxy
import math
from typing import Any
from . import knn_search_blind_lowd_non_d128_tcgen05_dispatch0610_r99_ec7c_v1 as lowd
from . import knn_search_mma_split_v1 as mma
from . import knn_search_scalar_capacity_0611_r22_4e96_v1 as scalar_capacity
THREADS = lowd.THREADS
BLOCK_Q = lowd.BLOCK_Q
BLOCK_M = lowd.BLOCK_M
D_STAGE = lowd.D_STAGE
K_MAX = lowd.K_MAX
MERGE_THREADS = lowd.MERGE_THREADS
MERGE_SMEM_BYTES = lowd.MERGE_SMEM_BYTES
HIGH_DYNAMIC_D_SPLIT_M = lowd.NON_D128_SPLIT_M
SUPPORTED_ORIGINAL_D = {129, 257, 511}
HIGH_D_MAX = 512
HIGH_Q_NORM_PARTS_MAX = HIGH_D_MAX // lowd.MMA_STAGE_VEC_ELEMS
HIGH_SMEM_A_BYTES = BLOCK_Q * D_STAGE * 2
HIGH_SMEM_B_BYTES = BLOCK_M * D_STAGE * 2
HIGH_SMEM_Q_NORM_PART_BYTES = BLOCK_Q * HIGH_Q_NORM_PARTS_MAX * 4
HIGH_SMEM_DB_NORM_PART_BYTES = BLOCK_M * lowd.MMA_DB_NORM_PARTS * 4
HIGH_SMEM_DB_NORM_BYTES = BLOCK_M * 4
HIGH_COHORT_TOPK_BYTES = BLOCK_Q * K_MAX * 4
HIGH_SMEM_B_OFFSET = HIGH_SMEM_A_BYTES
HIGH_SMEM_Q_NORM_PART_OFFSET = HIGH_SMEM_B_OFFSET + HIGH_SMEM_B_BYTES
HIGH_SMEM_DB_NORM_PART_OFFSET = HIGH_SMEM_Q_NORM_PART_OFFSET + HIGH_SMEM_Q_NORM_PART_BYTES
HIGH_SMEM_DB_NORM_OFFSET = HIGH_SMEM_DB_NORM_PART_OFFSET + HIGH_SMEM_DB_NORM_PART_BYTES
HIGH_COHORT_TOPK_D_OFFSET = HIGH_SMEM_DB_NORM_OFFSET + HIGH_SMEM_DB_NORM_BYTES
HIGH_COHORT_TOPK_I_OFFSET = HIGH_COHORT_TOPK_D_OFFSET + lowd.MMA_POST_MMA_COL_COHORTS * HIGH_COHORT_TOPK_BYTES
HIGH_SMEM_POOL_BYTES = HIGH_COHORT_TOPK_I_OFFSET + lowd.MMA_POST_MMA_COL_COHORTS * HIGH_COHORT_TOPK_BYTES + 256
HIGH_MMA_SMEM_BYTES = HIGH_SMEM_POOL_BYTES + mma.WEAVE_SMEM_SYSTEM_BYTES
_DIRECT_MMA_KERNELS: dict[int, dict[str, Any]] = {}
_DIRECT_SCRATCH: dict[tuple[int, int, int, int, int, int, int, str], tuple[Any, Any]] = {}
_knn_accumulate_q_norm_highd_direct_d = _ir_proxy('loom.examples.weave.knn_search_dynamic_d_high_q128_directstride_tcgen05_0618_ccef_v1:_knn_accumulate_q_norm_highd_direct_d', 256)
_knn_stage_q_pass_highd_direct_d = _ir_proxy('loom.examples.weave.knn_search_dynamic_d_high_q128_directstride_tcgen05_0618_ccef_v1:_knn_stage_q_pass_highd_direct_d', 256)
_knn_stage_database_pass_highd_direct_d = _ir_proxy('loom.examples.weave.knn_search_dynamic_d_high_q128_directstride_tcgen05_0618_ccef_v1:_knn_stage_database_pass_highd_direct_d', 256)
knn_search_dynamic_d_high_directstride_tcgen05_partial_0618_ccef_v1 = _decode_capture(_json_loads('{"__ir__": "knn_search_dynamic_d_high_directstride_tcgen05_partial_0618_ccef_v1", "arg_keys": ["queries", "database", "partial_distances", "partial_indices", "B", "Q", "M", "split_m", "num_q_tiles", "total_m_tiles", "tiles_per_split"], "cluster_dims": [1, 1, 1], "computed_smem_bytes": 126720, "constants": [["K_MAX_", 10], ["D_ORIG_", 512], ["NUM_D_PASSES_", 4], ["Q_NORM_PARTS_", 32]], "cta_group": 1, "threads": 640}'))
partial_ir = _decode_capture(_json_loads('{"__ir__": "knn_search_dynamic_d_high_directstride_tcgen05_partial_0618_ccef_v1", "arg_keys": ["queries", "database", "partial_distances", "partial_indices", "B", "Q", "M", "split_m", "num_q_tiles", "total_m_tiles", "tiles_per_split"], "cluster_dims": [1, 1, 1], "computed_smem_bytes": 126720, "constants": [["K_MAX_", 10], ["D_ORIG_", 512], ["NUM_D_PASSES_", 4], ["Q_NORM_PARTS_", 32]], "cta_group": 1, "threads": 640}'))
merge_ir = _decode_capture(_json_loads('{"__ir__": "knn_search_mma_split_merge_q128_const148_v1", "arg_keys": ["partial_distances", "partial_indices", "out_distances", "out_indices", "B", "Q", "K", "split_m", "num_q_tiles"], "cluster_dims": [1, 1, 1], "computed_smem_bytes": 0, "constants": [["K_MAX_", 10]], "cta_group": 1, "threads": 32}'))
ir = _decode_capture(_json_loads('{"__ir__": "knn_search_dynamic_d_high_directstride_tcgen05_partial_0618_ccef_v1", "arg_keys": ["queries", "database", "partial_distances", "partial_indices", "B", "Q", "M", "split_m", "num_q_tiles", "total_m_tiles", "tiles_per_split"], "cluster_dims": [1, 1, 1], "computed_smem_bytes": 126720, "constants": [["K_MAX_", 10], ["D_ORIG_", 512], ["NUM_D_PASSES_", 4], ["Q_NORM_PARTS_", 32]], "cta_group": 1, "threads": 640}'))
ROUTE_HIGH_DYNAMIC_D_TCGEN05 = 'ccef_dynamic_d_high_q128_directstride_tcgen05'
CONSUMED_SEED = 'weave-evolve-knn-search-ccef-highd-directstride'
HIGH_DYNAMIC_D_LABELS: tuple[str, ...] = ('blind_dyn_d129_q128_m65536_k10', 'blind_dyn_d257_q128_m65536_k10', 'blind_dyn_d511_q128_m65536_k10')
HIGH_DYNAMIC_D_SHAPES: list[dict[str, Any]] = [{'label': 'blind_dyn_d129_q128_m65536_k10', 'params': {'B': 1, 'Q': 128, 'M': 65536, 'D': 129, 'K': 10, 'dtype': 'bfloat16', 'seed': 610804, 'self_search': False, 'min_recall': 0.999}}, {'label': 'blind_dyn_d257_q128_m65536_k10', 'params': {'B': 1, 'Q': 128, 'M': 65536, 'D': 257, 'K': 10, 'dtype': 'bfloat16', 'seed': 610805, 'self_search': False, 'min_recall': 0.999}}, {'label': 'blind_dyn_d511_q128_m65536_k10', 'params': {'B': 1, 'Q': 128, 'M': 65536, 'D': 511, 'K': 10, 'dtype': 'bfloat16', 'seed': 610806, 'self_search': False, 'min_recall': 0.999}}]
SHAPE_DISPATCH_REGISTRY: tuple[dict[str, Any], ...] = ({'overlay': 'dynamic_d_high_q128_directstride_tcgen05_ccef', 'shape_key': 'B=1,Q=128,M=65536,D in {129,257,511},K=10', 'labels': HIGH_DYNAMIC_D_LABELS, 'guard': 'B == 1 and Q == 128 and M == 65536 and D in {129,257,511} and K == 10 and not self_search and not forced_fallback', 'route': ROUTE_HIGH_DYNAMIC_D_TCGEN05, 'entrypoint': 'loom.examples.weave.knn_search_dynamic_d_high_q128_directstride_tcgen05_0618_ccef_v1:launch_for_eval', 'selected_seed': CONSUMED_SEED, 'source_task': 'weave-evolve ccef dynamic-D high bucket direct-stride repair', 'coverage_class': 'bucket_seed_dynamic_d_high_q128_m65536_k10', 'workflow_mode': 'generalize_auto_tuning', 'auto_tuning_stage': 'bucket-kernel'},)

def _use_high_dynamic_d_tcgen05(inputs: dict[str, Any]) -> bool:
    return int(inputs['B']) == 1 and int(inputs['Q']) == BLOCK_Q and (int(inputs['M']) == 65536) and (int(inputs['D']) in SUPPORTED_ORIGINAL_D) and (int(inputs['K']) == K_MAX) and (not bool(inputs.get('self_search', False))) and (not bool(inputs.get('force_fallback', False))) and mma._tcgen05_capable_arch()

def _compile_direct_mma_kernels(original_d: int) -> dict[str, Any]:
    from .._dispatch_runtime import generate_kernel
    from .._dispatch_runtime import _cuda_include_dirs
    from .._dispatch_runtime import compile_cuda, detect_gpu_arch
    from .._dispatch_runtime import CUDAKernel
    arch = detect_gpu_arch()
    include_dirs = _cuda_include_dirs()
    partial_source = generate_kernel(knn_search_dynamic_d_high_directstride_tcgen05_partial_0618_ccef_v1, validate=False, smem_bytes=HIGH_MMA_SMEM_BYTES, D_ORIG_=int(original_d), NUM_D_PASSES_=int(math.ceil(original_d / D_STAGE)), Q_NORM_PARTS_=int(math.ceil(original_d / lowd.MMA_STAGE_VEC_ELEMS)))
    merge_source = generate_kernel(lowd.merge_ir, validate=False, smem_bytes=MERGE_SMEM_BYTES)
    partial_cubin = compile_cuda(partial_source, arch=arch, options=['--use_fast_math'], include_dirs=include_dirs)
    merge_cubin = compile_cuda(merge_source, arch=arch, options=['--use_fast_math'], include_dirs=include_dirs)
    return {'partial': CUDAKernel(partial_cubin, ''.join(['kernel_', format(knn_search_dynamic_d_high_directstride_tcgen05_partial_0618_ccef_v1.symbol, '')])), 'merge': CUDAKernel(merge_cubin, ''.join(['kernel_', format(lowd.merge_ir.symbol, '')])), 'shared_mem': HIGH_MMA_SMEM_BYTES}

def _partial_scratch(inputs: dict[str, Any], split_m: int, num_q_tiles: int) -> tuple[Any, Any]:
    import torch
    key = (int(inputs['B']), int(inputs['Q']), int(inputs['M']), int(inputs['D']), int(split_m), int(num_q_tiles), int(inputs['queries'].device.index or 0), str(inputs['queries'].dtype))
    cached = _DIRECT_SCRATCH.get(key)
    if cached is None:
        shape = (int(inputs['B']), int(num_q_tiles), int(split_m), BLOCK_Q, K_MAX)
        cached = (torch.empty(shape, dtype=torch.float32, device=inputs['queries'].device), torch.empty(shape, dtype=torch.int32, device=inputs['queries'].device))
        _DIRECT_SCRATCH[key] = cached
    return cached

def selected_route(inputs: dict[str, Any]) -> str:
    if _use_high_dynamic_d_tcgen05(inputs):
        return ROUTE_HIGH_DYNAMIC_D_TCGEN05
    return 'scalar_capacity_parent'

def selected_route_name(inputs: dict[str, Any]) -> str:
    return selected_route(inputs)

def route_info(inputs: dict[str, Any]) -> dict[str, Any]:
    if _use_high_dynamic_d_tcgen05(inputs):
        return {'route': ROUTE_HIGH_DYNAMIC_D_TCGEN05, 'selected_route': ROUTE_HIGH_DYNAMIC_D_TCGEN05, 'selected_entrypoint': 'loom.examples.weave.knn_search_dynamic_d_high_q128_directstride_tcgen05_0618_ccef_v1:launch_for_eval', 'route_kind': 'specialized', 'route_source': 'bucket-specific-seed', 'coverage_class': 'bucket_seed_dynamic_d_high_q128_m65536_k10', 'classification': 'seed-produced', 'coverage_only': False, 'production_policy': 'weave_only', 'external_fallback': None, 'guard_id': 'dynamic_d_high_q128_m65536_k10', 'forced_fallback': False, 'selected_guard': SHAPE_DISPATCH_REGISTRY[0]['guard'], 'fallback': 'scalar_capacity_parent', 'missing_weave_route': False, 'selected_seed': CONSUMED_SEED}
    return {'route': 'scalar_capacity_parent', 'selected_route': 'scalar_capacity_parent', 'selected_entrypoint': 'loom.examples.weave.knn_search_scalar_capacity_0611_r22_4e96_v1:launch_for_eval', 'route_kind': 'general', 'route_source': 'generic-weave-fallback', 'coverage_only': False, 'production_policy': 'weave_only', 'external_fallback': None, 'forced_fallback': bool(inputs.get('force_fallback', False))}

def _launch_high_dynamic_d_tcgen05(inputs: dict[str, Any]) -> dict[str, Any]:
    import torch
    original_d = int(inputs['D'])
    kernels = _DIRECT_MMA_KERNELS.get(original_d)
    if kernels is None:
        kernels = _compile_direct_mma_kernels(original_d)
        _DIRECT_MMA_KERNELS[original_d] = kernels
    bsz = int(inputs['B'])
    q_rows = int(inputs['Q'])
    m_rows = int(inputs['M'])
    k = int(inputs['K'])
    split_m = min(HIGH_DYNAMIC_D_SPLIT_M, math.ceil(m_rows / BLOCK_M))
    num_q_tiles = math.ceil(q_rows / BLOCK_Q)
    total_m_tiles = math.ceil(m_rows / BLOCK_M)
    tiles_per_split = math.ceil(total_m_tiles / split_m)
    partial_dist, partial_idx = _partial_scratch(inputs, split_m, num_q_tiles)
    kernels['partial'].launch(grid=(bsz * num_q_tiles * split_m, 1, 1), block=(THREADS, 1, 1), args=[inputs['queries'], inputs['database'], partial_dist, partial_idx, bsz, q_rows, m_rows, split_m, num_q_tiles, total_m_tiles, tiles_per_split], shared_mem=int(kernels['shared_mem']))
    kernels['merge'].launch(grid=(bsz * q_rows, 1, 1), block=(MERGE_THREADS, 1, 1), args=[partial_dist, partial_idx, inputs['out_distances'], inputs['out_indices'], bsz, q_rows, k, split_m, num_q_tiles], shared_mem=MERGE_SMEM_BYTES)
    return {'distances': inputs['out_distances'], 'indices': inputs['out_indices']}

def launch_for_eval(inputs: dict[str, Any]) -> dict[str, Any]:
    if _use_high_dynamic_d_tcgen05(inputs):
        return _launch_high_dynamic_d_tcgen05(inputs)
    return scalar_capacity.launch_for_eval(inputs)

def knn_search_compile_and_launch_dynamic_d_high_q128_tcgen05(*, benchmark: bool=True) -> dict[str, Any]:
    from .._dispatch_runtime import evaluate
    result = evaluate(launch_for_eval, shapes=HIGH_DYNAMIC_D_SHAPES, benchmark=benchmark)
    result['passed'] = bool(result.get('summary', {}).get('all_correct'))
    print(result)
    return result
