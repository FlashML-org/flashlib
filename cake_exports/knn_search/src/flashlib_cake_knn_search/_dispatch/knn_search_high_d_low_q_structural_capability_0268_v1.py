"""Exact five-row high-D/low-Q direct-stride tcgen05 capability portfolio.

Minimum target architecture: sm_100a.  This additive task-local portfolio owns
only the five rank-assigned BF16 K10 rows.  Its new D768/Q64 producer widens
database ownership from N128 to N256, reads two TMEM stripes, and feeds an
exact fixed-148 split-M merge.  D640, D2048, and D4096 rows reuse validated
exact-stride tcgen05 producer/merge seeds.  Every route writes caller-owned
FP32 distances and INT32 indices; no dispatcher or production registry is
modified.
"""
from __future__ import annotations
from json import loads as _json_loads
from .._dispatch_runtime import _capture_cuTensorMapEncodeTiled, _decode_capture, _import_dispatch_module, _ir_proxy
import math
from typing import Any
from .._dispatch_runtime import select_named_shapes
from . import knn_search_blind_lowd_non_d128_tcgen05_dispatch0610_r99_ec7c_v1 as lowd
from . import knn_search_mma_split_v1 as mma
from .knn_search_stream import current_stream_handle
THREADS = lowd.THREADS
BLOCK_Q = lowd.BLOCK_Q
BLOCK_M = 256
D_STAGE = lowd.D_STAGE
K_MAX = lowd.K_MAX
MERGE_THREADS = lowd.MERGE_THREADS
MERGE_SMEM_BYTES = lowd.MERGE_SMEM_BYTES
MERGE_SPLITS_PER_LANE_MAX = mma.MERGE_SPLITS_PER_LANE_MAX
Q128_SPLIT_M = mma.Q128_SPLIT_M
Q128_SLOT4_LANES = mma.Q128_SLOT4_LANES
HIGH_DYNAMIC_D_SPLIT_M = lowd.NON_D128_SPLIT_M
HIGH_D_MAX = 768
TARGET_Q = 64
TARGET_M = 65536
TARGET_SPLIT_M = 148
TARGET_LABEL = 'target_d768_q64_m65536_k10'
TARGET_ROUTE = '0268_high_d_low_q_d768_q64_n256_tcgen05'
HIGH_SMEM_A_BYTES = BLOCK_Q * D_STAGE * 2
HIGH_SMEM_B_BYTES = BLOCK_M * D_STAGE * 2
HIGH_SMEM_DB_NORM_PART_BYTES = BLOCK_M * lowd.MMA_DB_NORM_PARTS * 4
HIGH_SMEM_DB_NORM_BYTES = BLOCK_M * 4
HIGH_COHORT_TOPK_BYTES = BLOCK_Q * K_MAX * 4
HIGH_SMEM_B_OFFSET = HIGH_SMEM_A_BYTES
HIGH_SMEM_DB_NORM_PART_OFFSET = HIGH_SMEM_B_OFFSET + HIGH_SMEM_B_BYTES
HIGH_SMEM_DB_NORM_OFFSET = HIGH_SMEM_DB_NORM_PART_OFFSET + HIGH_SMEM_DB_NORM_PART_BYTES
HIGH_COHORT_TOPK_D_OFFSET = HIGH_SMEM_DB_NORM_OFFSET + HIGH_SMEM_DB_NORM_BYTES
HIGH_COHORT_TOPK_I_OFFSET = HIGH_COHORT_TOPK_D_OFFSET + lowd.MMA_POST_MMA_COL_COHORTS * HIGH_COHORT_TOPK_BYTES
HIGH_SMEM_POOL_BYTES = HIGH_COHORT_TOPK_I_OFFSET + lowd.MMA_POST_MMA_COL_COHORTS * HIGH_COHORT_TOPK_BYTES + 256
HIGH_MMA_SMEM_BYTES = HIGH_SMEM_POOL_BYTES + mma.WEAVE_SMEM_SYSTEM_BYTES
_PREVIOUS_LOWD_BLOCK_M = lowd.BLOCK_M
_PREVIOUS_MMA_BLOCK_M = mma.BLOCK_M
lowd.BLOCK_M = BLOCK_M
mma.BLOCK_M = BLOCK_M
_DIRECT_MMA_KERNELS: dict[int, dict[str, Any]] = {}
_DIRECT_SCRATCH: dict[tuple[int, int, int, int, int, int, int, str, int], tuple[Any, Any]] = {}
_knn_stage_q_pass_highd_direct_d = _ir_proxy('loom.examples.weave.knn_search_high_d_low_q_structural_capability_0268_v1:_knn_stage_q_pass_highd_direct_d', 256)
_knn_stage_database_pass_highd_direct_d = _ir_proxy('loom.examples.weave.knn_search_high_d_low_q_structural_capability_0268_v1:_knn_stage_database_pass_highd_direct_d', 256)
_knn_accumulate_db_norm_blockm256 = _ir_proxy('loom.examples.weave.knn_search_high_d_low_q_structural_capability_0268_v1:_knn_accumulate_db_norm_blockm256', 256)
_knn_mma_issue_blockm256 = _ir_proxy('loom.examples.weave.knn_search_high_d_low_q_structural_capability_0268_v1:_knn_mma_issue_blockm256', 256)
knn_search_high_d_low_q_d768_q64_norm_merge_0268_v1 = _decode_capture(_json_loads('{"__ir__": "knn_search_high_d_low_q_d768_q64_norm_merge_0268_v1", "arg_keys": ["queries", "partial_distances", "partial_indices", "out_distances", "out_indices", "B", "Q", "K", "split_m", "num_q_tiles"], "cluster_dims": [1, 1, 1], "computed_smem_bytes": 0, "constants": [["K_MAX_", 10]], "cta_group": 1, "threads": 32}'))
knn_search_high_d_low_q_d768_q64_n256_partial_0268_v1 = _decode_capture(_json_loads('{"__ir__": "knn_search_high_d_low_q_d768_q64_n256_partial_0268_v1", "arg_keys": ["queries", "database", "partial_distances", "partial_indices", "B", "Q", "M", "split_m", "num_q_tiles", "total_m_tiles", "active_split_m"], "cluster_dims": [1, 1, 1], "computed_smem_bytes": 145664, "constants": [["K_MAX_", 10], ["D_ORIG_", 768], ["NUM_D_PASSES_", 6]], "cta_group": 1, "threads": 640}'))
lowd.BLOCK_M = _PREVIOUS_LOWD_BLOCK_M
mma.BLOCK_M = _PREVIOUS_MMA_BLOCK_M
partial_ir = _decode_capture(_json_loads('{"__ir__": "knn_search_high_d_low_q_d768_q64_n256_partial_0268_v1", "arg_keys": ["queries", "database", "partial_distances", "partial_indices", "B", "Q", "M", "split_m", "num_q_tiles", "total_m_tiles", "active_split_m"], "cluster_dims": [1, 1, 1], "computed_smem_bytes": 145664, "constants": [["K_MAX_", 10], ["D_ORIG_", 768], ["NUM_D_PASSES_", 6]], "cta_group": 1, "threads": 640}'))
merge_ir = _decode_capture(_json_loads('{"__ir__": "knn_search_high_d_low_q_d768_q64_norm_merge_0268_v1", "arg_keys": ["queries", "partial_distances", "partial_indices", "out_distances", "out_indices", "B", "Q", "K", "split_m", "num_q_tiles"], "cluster_dims": [1, 1, 1], "computed_smem_bytes": 0, "constants": [["K_MAX_", 10]], "cta_group": 1, "threads": 32}'))
ir = _decode_capture(_json_loads('{"__ir__": "knn_search_high_d_low_q_d768_q64_n256_partial_0268_v1", "arg_keys": ["queries", "database", "partial_distances", "partial_indices", "B", "Q", "M", "split_m", "num_q_tiles", "total_m_tiles", "active_split_m"], "cluster_dims": [1, 1, 1], "computed_smem_bytes": 145664, "constants": [["K_MAX_", 10], ["D_ORIG_", 768], ["NUM_D_PASSES_", 6]], "cta_group": 1, "threads": 640}'))
ROUTE_HIGH_DYNAMIC_D_TCGEN05 = TARGET_ROUTE
CONSUMED_SEED = 'weave-evolve-knn-search-dispatcher-residual-full198-0268'
HIGH_DYNAMIC_D_LABELS: tuple[str, ...] = (TARGET_LABEL,)
HIGH_DYNAMIC_D_SHAPES = _decode_capture(_json_loads('[{"__dict_items__": [["label", "target_d768_q64_m65536_k10"], ["params", {"__dict_items__": [["B", 1], ["Q", 64], ["M", 65536], ["D", 768], ["K", 10], ["dtype", "bfloat16"], ["seed", 611108], ["self_search", false], ["min_recall", 0.999]]}]]}]'))
D768_SOURCE_REGISTRY: tuple[dict[str, Any], ...] = ()

def _compile_direct_mma_kernels(original_d: int) -> dict[str, Any]:
    from .._dispatch_runtime import generate_kernel
    from .._dispatch_runtime import _cuda_include_dirs
    from .._dispatch_runtime import compile_cuda, detect_gpu_arch
    from .._dispatch_runtime import CUDAKernel
    arch = detect_gpu_arch()
    include_dirs = _cuda_include_dirs()
    partial_source = generate_kernel(knn_search_high_d_low_q_d768_q64_n256_partial_0268_v1, validate=False, smem_bytes=HIGH_MMA_SMEM_BYTES, D_ORIG_=int(original_d), NUM_D_PASSES_=int(math.ceil(original_d / D_STAGE)))
    partial_cubin = compile_cuda(partial_source, arch=arch, options=['--use_fast_math'], include_dirs=include_dirs)
    return {'partial': CUDAKernel(partial_cubin, ''.join(['kernel_', format(knn_search_high_d_low_q_d768_q64_n256_partial_0268_v1.symbol, '')])), 'shared_mem': HIGH_MMA_SMEM_BYTES}
_TARGET_MERGE_KERNEL: Any | None = None

def _target_merge_kernel() -> Any:
    global _TARGET_MERGE_KERNEL
    if _TARGET_MERGE_KERNEL is None:
        from .._dispatch_runtime import generate_kernel
        from .._dispatch_runtime import _cuda_include_dirs
        from .._dispatch_runtime import compile_cuda, detect_gpu_arch
        from .._dispatch_runtime import CUDAKernel
        merge_ir = _decode_capture(_json_loads('{"__ir__": "knn_search_high_d_low_q_d768_q64_norm_merge_0268_v1", "arg_keys": ["queries", "partial_distances", "partial_indices", "out_distances", "out_indices", "B", "Q", "K", "split_m", "num_q_tiles"], "cluster_dims": [1, 1, 1], "computed_smem_bytes": 0, "constants": [["K_MAX_", 10]], "cta_group": 1, "threads": 32}'))
        source = generate_kernel(merge_ir, validate=False, smem_bytes=MERGE_SMEM_BYTES)
        cubin = compile_cuda(source, arch=detect_gpu_arch(), options=['--use_fast_math'], include_dirs=_cuda_include_dirs())
        _TARGET_MERGE_KERNEL = CUDAKernel(cubin, ''.join(['kernel_', format(merge_ir.symbol, '')]))
    return _TARGET_MERGE_KERNEL

def _partial_scratch(inputs: dict[str, Any], split_m: int, num_q_tiles: int) -> tuple[Any, Any]:
    import torch
    key = (int(inputs['B']), int(inputs['Q']), int(inputs['M']), int(inputs['D']), int(split_m), int(num_q_tiles), int(inputs['queries'].device.index or 0), str(inputs['queries'].dtype), current_stream_handle(inputs))
    cached = _DIRECT_SCRATCH.get(key)
    if cached is None:
        shape = (int(inputs['B']), int(num_q_tiles), int(split_m), BLOCK_Q, K_MAX)
        cached = (torch.empty(shape, dtype=torch.float32, device=inputs['queries'].device), torch.empty(shape, dtype=torch.int32, device=inputs['queries'].device))
        _DIRECT_SCRATCH[key] = cached
    return cached
from .. import _dispatch_runtime as residual_contract
from . import knn_search_floor13_dynamic_d_k10_0622_5f25_v1 as d640_seed
from . import knn_search_d2048_q8_m16384_k10_0623_6185_d51b_v1 as d2048_seed
from . import knn_search_d4096_q4q8_m8192m16384_k10_0623_5ff7_v1 as d4096_seed
ENTRYPOINT = 'loom.examples.weave.knn_search_high_d_low_q_structural_capability_0268_v1:launch_for_eval'
ROUTE_PORTFOLIO = '0268_high_d_low_q_directstride_splitm_tcgen05'
CAPABILITY_SHAPE_KEYS: tuple[str, ...] = ('target_d768_q64_m65536_k10', 'blind_0622_dyn_d640_q32_m32768_k10', 'target_d2048_q8_m16384_k10', 'target_d4096_q4_m8192_k10', 'target_d4096_q8_m16384_k10')
TARGET_SHAPE = 'target_d4096_q4_m8192_k10'
_shape_by_label = {shape['label']: shape for shape in residual_contract.CANONICAL_SHAPES}
TARGET_SHAPES: list[dict[str, Any]] = [_shape_by_label[label] for label in CAPABILITY_SHAPE_KEYS]
_PARTIAL_STRIDE_Q = 128

def _workspace_metadata(split_m: int) -> dict[str, Any]:
    shape = [1, 1, int(split_m), _PARTIAL_STRIDE_Q, K_MAX]
    elements = int(split_m) * _PARTIAL_STRIDE_Q * K_MAX
    allocation_bytes = elements * 4
    return {'allocation_policy': 'lazy seed-owned cache resolved inside launch_for_eval', 'partial_distances': {'shape': shape, 'dtype': 'torch.float32', 'bytes': allocation_bytes}, 'partial_indices': {'shape': shape, 'dtype': 'torch.int32', 'bytes': allocation_bytes}, 'total_bytes': 2 * allocation_bytes}
_D768_KEY = (1, 64, 65536, 768, 10, False)
_D640_KEY = (1, 32, 32768, 640, 10, False)
_D2048_KEY = (1, 8, 16384, 2048, 10, False)
_D4096_Q4_KEY = (1, 4, 8192, 4096, 10, False)
_D4096_Q8_KEY = (1, 8, 16384, 4096, 10, False)
_ROUTE_CONFIGS = _decode_capture(_json_loads('{"__tuple__": [{"__dict_items__": [["shape_key", "target_d768_q64_m65536_k10"], ["shape", {"__tuple__": [1, 64, 65536, 768, 10, false]}], ["route", "0268_high_d_low_q_d768_q64_n256_tcgen05"], ["source_entrypoint", "loom.examples.weave.knn_search_high_d_low_q_structural_capability_0268_v1:launch_for_eval"], ["guard", "B == 1 and Q == 64 and M == 65536 and D == 768 and K == 10 and not self_search and not force_fallback and arch in {sm_100a,sm_103a} and queries.dtype == database.dtype == torch.bfloat16 and out_distances.dtype == torch.float32 and out_indices.dtype == torch.int32"], ["workspace_reuse", "task-local N256 fixed148 partial scratch cache"], ["workspace_metadata", {"__dict_items__": [["allocation_policy", "lazy seed-owned cache resolved inside launch_for_eval"], ["partial_distances", {"__dict_items__": [["shape", [1, 1, 148, 128, 10]], ["dtype", "torch.float32"], ["bytes", 757760]]}], ["partial_indices", {"__dict_items__": [["shape", [1, 1, 148, 128, 10]], ["dtype", "torch.int32"], ["bytes", 757760]]}], ["total_bytes", 1515520]]}], ["partial_grid", [148, 1, 1]], ["merge_grid", [64, 1, 1]], ["partial_threads", 640], ["partial_smem_bytes", 145664], ["structural_delta", "N256_database_ownership_with_query_invariant_norm_deferred_to_merge"]]}, {"__dict_items__": [["shape_key", "blind_0622_dyn_d640_q32_m32768_k10"], ["shape", {"__tuple__": [1, 32, 32768, 640, 10, false]}], ["route", "0268_high_d_low_q_d640_q32_directstride_tcgen05"], ["source_entrypoint", "loom.examples.weave.knn_search_floor13_dynamic_d_k10_0622_5f25_v1:launch_for_eval"], ["guard", "B == 1 and Q == 32 and M == 32768 and D == 640 and K == 10 and not self_search and not force_fallback and arch in {sm_100a,sm_103a} and queries.dtype == database.dtype == torch.bfloat16 and out_distances.dtype == torch.float32 and out_indices.dtype == torch.int32"], ["workspace_reuse", "D640 direct-stride fixed split-M partial scratch cache"], ["workspace_metadata", {"__dict_items__": [["allocation_policy", "lazy seed-owned cache resolved inside launch_for_eval"], ["partial_distances", {"__dict_items__": [["shape", [1, 1, 148, 128, 10]], ["dtype", "torch.float32"], ["bytes", 757760]]}], ["partial_indices", {"__dict_items__": [["shape", [1, 1, 148, 128, 10]], ["dtype", "torch.int32"], ["bytes", 757760]]}], ["total_bytes", 1515520]]}], ["partial_grid", [148, 1, 1]], ["merge_grid", [32, 1, 1]], ["partial_threads", 640], ["partial_smem_bytes", 143104], ["structural_delta", "scalar_capacity_to_directstride_tcgen05_splitm"]]}, {"__dict_items__": [["shape_key", "target_d2048_q8_m16384_k10"], ["shape", {"__tuple__": [1, 8, 16384, 2048, 10, false]}], ["route", "0268_high_d_low_q_d2048_q8_directstride_tcgen05"], ["source_entrypoint", "loom.examples.weave.knn_search_d2048_q8_m16384_k10_0623_6185_d51b_v1:launch_for_eval"], ["guard", "B == 1 and Q == 8 and M == 16384 and D == 2048 and K == 10 and not self_search and not force_fallback and arch in {sm_100a,sm_103a} and queries.dtype == database.dtype == torch.bfloat16 and out_distances.dtype == torch.float32 and out_indices.dtype == torch.int32"], ["workspace_reuse", "D2048 direct-stride dynamic split-M partial scratch cache"], ["workspace_metadata", {"__dict_items__": [["allocation_policy", "lazy seed-owned cache resolved inside launch_for_eval"], ["partial_distances", {"__dict_items__": [["shape", [1, 1, 128, 128, 10]], ["dtype", "torch.float32"], ["bytes", 655360]]}], ["partial_indices", {"__dict_items__": [["shape", [1, 1, 128, 128, 10]], ["dtype", "torch.int32"], ["bytes", 655360]]}], ["total_bytes", 1310720]]}], ["partial_grid", [128, 1, 1]], ["merge_grid", [8, 1, 1]], ["partial_threads", 512], ["partial_smem_bytes", 126720], ["structural_delta", "scalar_capacity_to_Q64_directstride_tcgen05_splitm"]]}, {"__dict_items__": [["shape_key", "target_d4096_q4_m8192_k10"], ["shape", {"__tuple__": [1, 4, 8192, 4096, 10, false]}], ["route", "0268_high_d_low_q_d4096_q4_directstride_tcgen05"], ["source_entrypoint", "loom.examples.weave.knn_search_d4096_q4q8_m8192m16384_k10_0623_5ff7_v1:launch_for_eval"], ["guard", "B == 1 and Q == 4 and M == 8192 and D == 4096 and K == 10 and not self_search and not force_fallback and arch in {sm_100a,sm_103a} and queries.dtype == database.dtype == torch.bfloat16 and out_distances.dtype == torch.float32 and out_indices.dtype == torch.int32"], ["workspace_reuse", "D4096 direct-stride dynamic split-M partial scratch cache"], ["workspace_metadata", {"__dict_items__": [["allocation_policy", "lazy seed-owned cache resolved inside launch_for_eval"], ["partial_distances", {"__dict_items__": [["shape", [1, 1, 64, 128, 10]], ["dtype", "torch.float32"], ["bytes", 327680]]}], ["partial_indices", {"__dict_items__": [["shape", [1, 1, 64, 128, 10]], ["dtype", "torch.int32"], ["bytes", 327680]]}], ["total_bytes", 655360]]}], ["partial_grid", [64, 1, 1]], ["merge_grid", [4, 1, 1]], ["partial_threads", 512], ["partial_smem_bytes", 159488], ["structural_delta", "scalar_capacity_to_Q64_directstride_tcgen05_splitm"]]}, {"__dict_items__": [["shape_key", "target_d4096_q8_m16384_k10"], ["shape", {"__tuple__": [1, 8, 16384, 4096, 10, false]}], ["route", "0268_high_d_low_q_d4096_q8_directstride_tcgen05"], ["source_entrypoint", "loom.examples.weave.knn_search_d4096_q4q8_m8192m16384_k10_0623_5ff7_v1:launch_for_eval"], ["guard", "B == 1 and Q == 8 and M == 16384 and D == 4096 and K == 10 and not self_search and not force_fallback and arch in {sm_100a,sm_103a} and queries.dtype == database.dtype == torch.bfloat16 and out_distances.dtype == torch.float32 and out_indices.dtype == torch.int32"], ["workspace_reuse", "D4096 direct-stride dynamic split-M partial scratch cache"], ["workspace_metadata", {"__dict_items__": [["allocation_policy", "lazy seed-owned cache resolved inside launch_for_eval"], ["partial_distances", {"__dict_items__": [["shape", [1, 1, 128, 128, 10]], ["dtype", "torch.float32"], ["bytes", 655360]]}], ["partial_indices", {"__dict_items__": [["shape", [1, 1, 128, 128, 10]], ["dtype", "torch.int32"], ["bytes", 655360]]}], ["total_bytes", 1310720]]}], ["partial_grid", [128, 1, 1]], ["merge_grid", [8, 1, 1]], ["partial_threads", 512], ["partial_smem_bytes", 159488], ["structural_delta", "scalar_capacity_to_Q64_directstride_tcgen05_splitm"]]}]}'))
SHAPE_DISPATCH_REGISTRY = _ROUTE_CONFIGS
d768_partial_ir = _decode_capture(_json_loads('{"__ir__": "knn_search_high_d_low_q_d768_q64_n256_partial_0268_v1", "arg_keys": ["queries", "database", "partial_distances", "partial_indices", "B", "Q", "M", "split_m", "num_q_tiles", "total_m_tiles", "active_split_m"], "cluster_dims": [1, 1, 1], "computed_smem_bytes": 145664, "constants": [["K_MAX_", 10], ["D_ORIG_", 768], ["NUM_D_PASSES_", 6]], "cta_group": 1, "threads": 640}'))
d768_merge_ir = _decode_capture(_json_loads('{"__ir__": "knn_search_high_d_low_q_d768_q64_norm_merge_0268_v1", "arg_keys": ["queries", "partial_distances", "partial_indices", "out_distances", "out_indices", "B", "Q", "K", "split_m", "num_q_tiles"], "cluster_dims": [1, 1, 1], "computed_smem_bytes": 0, "constants": [["K_MAX_", 10]], "cta_group": 1, "threads": 32}'))
d640_partial_ir = _decode_capture(_json_loads('{"__ir__": "knn_search_dynamic_d768d1024_q32q16_tcgen05_partial_0618_9286_v1", "arg_keys": ["queries", "database", "partial_distances", "partial_indices", "B", "Q", "M", "split_m", "num_q_tiles", "total_m_tiles", "tiles_per_split"], "cluster_dims": [1, 1, 1], "computed_smem_bytes": 143104, "constants": [["K_MAX_", 10], ["D_ORIG_", 1024], ["NUM_D_PASSES_", 8], ["Q_NORM_PARTS_", 64]], "cta_group": 1, "threads": 640}'))
d640_merge_ir = _decode_capture(_json_loads('{"__ir__": "knn_search_mma_split_merge_q128_const148_v1", "arg_keys": ["partial_distances", "partial_indices", "out_distances", "out_indices", "B", "Q", "K", "split_m", "num_q_tiles"], "cluster_dims": [1, 1, 1], "computed_smem_bytes": 0, "constants": [["K_MAX_", 10]], "cta_group": 1, "threads": 32}'))
d2048_partial_ir = _decode_capture(_json_loads('{"__ir__": "knn_search_d2048_q8_m16384_k10_partial_0623_6185_d51b_v1", "arg_keys": ["queries", "database", "partial_distances", "partial_indices", "B", "Q", "M", "split_m", "num_q_tiles", "total_m_tiles"], "cluster_dims": [1, 1, 1], "computed_smem_bytes": 126720, "constants": [["K_MAX_", 10]], "cta_group": 1, "threads": 512}'))
d2048_merge_ir = _decode_capture(_json_loads('{"__ir__": "knn_search_d2048_q8_m16384_k10_merge_0623_6185_d51b_v1", "arg_keys": ["partial_distances", "partial_indices", "out_distances", "out_indices", "B", "Q", "K", "split_m", "num_q_tiles"], "cluster_dims": [1, 1, 1], "computed_smem_bytes": 0, "constants": [["K_MAX_", 10]], "cta_group": 1, "threads": 32}'))
d4096_partial_ir = _decode_capture(_json_loads('{"__ir__": "knn_search_d4096_q4q8_m8192m16384_k10_partial_0623_5ff7_v1", "arg_keys": ["queries", "database", "partial_distances", "partial_indices", "B", "Q", "M", "split_m", "num_q_tiles", "total_m_tiles"], "cluster_dims": [1, 1, 1], "computed_smem_bytes": 159488, "constants": [["K_MAX_", 10]], "cta_group": 1, "threads": 512}'))
d4096_merge_ir = _decode_capture(_json_loads('{"__ir__": "knn_search_d4096_q4q8_m8192m16384_k10_merge_0623_5ff7_v1", "arg_keys": ["partial_distances", "partial_indices", "out_distances", "out_indices", "B", "Q", "K", "split_m", "num_q_tiles"], "cluster_dims": [1, 1, 1], "computed_smem_bytes": 0, "constants": [["K_MAX_", 10]], "cta_group": 1, "threads": 32}'))
ir = _decode_capture(_json_loads('{"__ir__": "knn_search_high_d_low_q_d768_q64_n256_partial_0268_v1", "arg_keys": ["queries", "database", "partial_distances", "partial_indices", "B", "Q", "M", "split_m", "num_q_tiles", "total_m_tiles", "active_split_m"], "cluster_dims": [1, 1, 1], "computed_smem_bytes": 145664, "constants": [["K_MAX_", 10], ["D_ORIG_", 768], ["NUM_D_PASSES_", 6]], "cta_group": 1, "threads": 640}'))
merge_ir = _decode_capture(_json_loads('{"__ir__": "knn_search_high_d_low_q_d768_q64_norm_merge_0268_v1", "arg_keys": ["queries", "partial_distances", "partial_indices", "out_distances", "out_indices", "B", "Q", "K", "split_m", "num_q_tiles"], "cluster_dims": [1, 1, 1], "computed_smem_bytes": 0, "constants": [["K_MAX_", 10]], "cta_group": 1, "threads": 32}'))

def _shape_tuple(inputs: dict[str, Any]) -> tuple[int, int, int, int, int, bool]:
    return (int(inputs['B']), int(inputs['Q']), int(inputs['M']), int(inputs['D']), int(inputs['K']), bool(inputs.get('self_search', False)))

def _active_config(inputs: dict[str, Any]) -> dict[str, Any] | None:
    if bool(inputs.get('force_fallback', False)) or not mma._tcgen05_capable_arch():
        return None
    if str(getattr(inputs.get('queries'), 'dtype', None)) != 'torch.bfloat16' or str(getattr(inputs.get('database'), 'dtype', None)) != 'torch.bfloat16' or str(getattr(inputs.get('out_distances'), 'dtype', None)) != 'torch.float32' or (str(getattr(inputs.get('out_indices'), 'dtype', None)) != 'torch.int32'):
        return None
    shape = _shape_tuple(inputs)
    for config in _ROUTE_CONFIGS:
        if shape == config['shape']:
            return config
    return None

def selected_route(inputs: dict[str, Any]) -> str:
    config = _active_config(inputs)
    return str(config['route']) if config is not None else 'unsupported_shape'

def selected_route_name(inputs: dict[str, Any]) -> str:
    return selected_route(inputs)

def route_info(inputs: dict[str, Any]) -> dict[str, Any]:
    config = _active_config(inputs)
    if config is None:
        return {'route': 'unsupported_shape', 'selected_route': 'unsupported_shape', 'selected_entrypoint': None, 'route_kind': 'unsupported', 'production_policy': 'weave_only', 'external_fallback': None, 'forced_fallback': bool(inputs.get('force_fallback', False))}
    return {'route': config['route'], 'selected_route': config['route'], 'selected_entrypoint': ENTRYPOINT, 'source_entrypoint': config['source_entrypoint'], 'route_kind': 'specialized', 'route_source': 'task-local-capability-portfolio', 'coverage_class': 'high_d_low_q_structural_capability_0268', 'classification': 'seed-produced', 'coverage_only': False, 'production_policy': 'weave_only', 'external_fallback': None, 'guard_id': config['shape_key'], 'selected_guard': config['guard'], 'forced_fallback': False, 'fallback': None, 'missing_weave_route': False, 'selected_seed': 'weave-evolve-knn-search-dispatcher-residual-full198-0268', 'padding_tag': 'none', 'padded_D': int(inputs['D']), 'original_D': int(inputs['D']), 'uses_materialized_padding': False, 'uses_kernel_padding': False, 'padding_overhead_timed': False, 'workspace_reuse': config['workspace_reuse'], 'workspace_metadata': config['workspace_metadata'], 'resource_metadata': {'partial_grid': config['partial_grid'], 'merge_grid': config['merge_grid'], 'partial_threads': config['partial_threads'], 'partial_smem_bytes': config['partial_smem_bytes'], 'merge_threads': MERGE_THREADS, 'merge_smem_bytes': MERGE_SMEM_BYTES}, 'structural_delta': config['structural_delta']}

def _launch_d768_q64_n256(inputs: dict[str, Any]) -> dict[str, Any]:
    kernels = _DIRECT_MMA_KERNELS.get(HIGH_D_MAX)
    if kernels is None:
        kernels = _compile_direct_mma_kernels(HIGH_D_MAX)
        _DIRECT_MMA_KERNELS[HIGH_D_MAX] = kernels
    partial_dist, partial_idx = _partial_scratch(inputs, TARGET_SPLIT_M, 1)
    kernels['partial'].launch(grid=(TARGET_SPLIT_M, 1, 1), block=(THREADS, 1, 1), args=[inputs['queries'], inputs['database'], partial_dist, partial_idx, 1, TARGET_Q, TARGET_M, TARGET_SPLIT_M, 1, TARGET_M // BLOCK_M, TARGET_SPLIT_M], shared_mem=int(kernels['shared_mem']))
    _target_merge_kernel().launch(grid=(TARGET_Q, 1, 1), block=(MERGE_THREADS, 1, 1), args=[inputs['queries'], partial_dist, partial_idx, inputs['out_distances'], inputs['out_indices'], 1, TARGET_Q, K_MAX, TARGET_SPLIT_M, 1], shared_mem=MERGE_SMEM_BYTES)
    return {'distances': inputs['out_distances'], 'indices': inputs['out_indices']}

def launch_for_eval(inputs: dict[str, Any]) -> dict[str, Any]:
    config = _active_config(inputs)
    if config is None:
        raise ValueError('high_d_low_q_structural_capability_0268 supports only the exact five assigned B=1 BF16 K10 non-self-search rows on sm_100a/sm_103a')
    shape = config['shape']
    if shape == _D768_KEY:
        return _launch_d768_q64_n256(inputs)
    if shape == _D640_KEY:
        return d640_seed.launch_for_eval(inputs)
    if shape == _D2048_KEY:
        return d2048_seed.launch_for_eval(inputs)
    return d4096_seed.launch_for_eval(inputs)

def route_trace_entry(label: str, inputs: dict[str, Any]) -> dict[str, Any]:
    return {'label': label, 'shape_key': label, 'B': int(inputs['B']), 'Q': int(inputs['Q']), 'M': int(inputs['M']), 'D': int(inputs['D']), 'K': int(inputs['K']), 'self_search': bool(inputs.get('self_search', False)), **route_info(inputs)}

def knn_search_compile_and_launch_high_d_low_q_structural_capability_0268(*, benchmark: bool=True) -> dict[str, Any]:
    result = residual_contract.evaluate(launch_for_eval, shapes=TARGET_SHAPES, correctness=True, benchmark=benchmark)
    result['passed'] = bool(result.get('summary', {}).get('all_correct'))
    return result
