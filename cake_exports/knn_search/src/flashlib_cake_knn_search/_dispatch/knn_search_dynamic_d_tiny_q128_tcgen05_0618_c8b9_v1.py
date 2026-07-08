"""Dynamic tiny-D q128/m65536 K10 kNN seed using padded Weave routes.

Minimum target architecture: sm_100a. This additive bucket seed owns
``B=1,Q=128,M=65536,K=10,D in {3,7,63}`` by packing the BF16 query/database
tensors into seed-compatible feature strides with a Weave kernel. D7/D63 reuse
the inherited padded-D tcgen05 partial producer and Q128 merge consumer. D3
uses a direct FP32 CUDA-core tile-reduce producer because the tcgen05 dot path
loses near-tie recall at this tiny distance scale. Guard misses delegate to the
scalar-capacity parent route.
"""
from __future__ import annotations
from json import loads as _json_loads
from .._dispatch_runtime import _capture_cuTensorMapEncodeTiled, _decode_capture, _import_dispatch_module, _ir_proxy
import math
from typing import Any
from . import knn_search_blind_lowd_non_d128_tcgen05_dispatch0610_r99_ec7c_v1 as lowd
from . import knn_search_lowd_non128_tile_reduce_0615_7d36_v2 as lowd_tile
from . import knn_search_scalar_capacity_0611_r22_4e96_v1 as scalar_capacity
PACK_THREADS = 256
THREADS = lowd.THREADS
BLOCK_Q = lowd.BLOCK_Q
BLOCK_M = lowd.BLOCK_M
K_MAX = lowd.K_MAX
MERGE_THREADS = lowd.MERGE_THREADS
MMA_SMEM_BYTES = lowd.MMA_SMEM_BYTES
MERGE_SMEM_BYTES = lowd.MERGE_SMEM_BYTES
TINY_DYNAMIC_D_SPLIT_M = lowd.NON_D128_SPLIT_M
SUPPORTED_ORIGINAL_D = {3, 7, 63}
D3_THREADS = lowd_tile.THREADS
D3_BLOCK_M = 4096
D3_SUBWARP_WIDTH = 2
D3_SUBWARPS_PER_WARP = 16
D3_NUM_ROW_WORKERS = 128
D3_ROWS_PER_WORKER = D3_BLOCK_M // D3_NUM_ROW_WORKERS
_PACK_KERNELS: dict[tuple[int, int], Any] = {}
_PADDED_MMA_KERNELS: dict[int, dict[str, Any]] = {}
_D3_TILE_KERNELS: dict[str, Any] | None = None
_PADDED_SCRATCH: dict[tuple[int, int, int, int, int, int, int, str], tuple[Any, Any]] = {}
_D3_SCRATCH: dict[tuple[int, int, int, int, int, int, str], tuple[Any, Any]] = {}
_PADDED_INPUTS: dict[tuple[int, int, int, int, int, int, int, str], tuple[Any, Any]] = {}
knn_search_dynamic_d_tiny_pack_bf16_0618_c8b9_v1 = _decode_capture(_json_loads('{"__ir__": "knn_search_dynamic_d_tiny_pack_bf16_0618_c8b9_v1", "arg_keys": ["queries", "database", "padded_queries", "padded_database", "B", "Q", "M"], "cluster_dims": [1, 1, 1], "computed_smem_bytes": 0, "constants": [["D_ORIG_", 3], ["D_PAD_", 16]], "cta_group": 1, "threads": 256}'))
knn_search_dynamic_d3_tile_reduce_partial_0618_c8b9_v1 = _decode_capture(_json_loads('{"__ir__": "knn_search_dynamic_d3_tile_reduce_partial_0618_c8b9_v1", "arg_keys": ["queries", "database", "partial_distances", "partial_indices", "B", "Q", "M", "K", "num_m_tiles"], "cluster_dims": [1, 1, 1], "computed_smem_bytes": 10240, "constants": [["K_MAX_", 10]], "cta_group": 1, "threads": 256}'))
pack_ir = _decode_capture(_json_loads('{"__ir__": "knn_search_dynamic_d_tiny_pack_bf16_0618_c8b9_v1", "arg_keys": ["queries", "database", "padded_queries", "padded_database", "B", "Q", "M"], "cluster_dims": [1, 1, 1], "computed_smem_bytes": 0, "constants": [["D_ORIG_", 3], ["D_PAD_", 16]], "cta_group": 1, "threads": 256}'))
d3_tile_ir = _decode_capture(_json_loads('{"__ir__": "knn_search_dynamic_d3_tile_reduce_partial_0618_c8b9_v1", "arg_keys": ["queries", "database", "partial_distances", "partial_indices", "B", "Q", "M", "K", "num_m_tiles"], "cluster_dims": [1, 1, 1], "computed_smem_bytes": 10240, "constants": [["K_MAX_", 10]], "cta_group": 1, "threads": 256}'))
partial_ir = _decode_capture(_json_loads('{"__ir__": "knn_search_blind_lowd_non_d128_tcgen05_partial_0615_r99_v1", "arg_keys": ["queries", "database", "partial_distances", "partial_indices", "B", "Q", "M", "split_m", "num_q_tiles", "total_m_tiles", "tiles_per_split"], "cluster_dims": [1, 1, 1], "computed_smem_bytes": 120576, "constants": [["K_MAX_", 10], ["D_TOTAL_", 64], ["NUM_D_PASSES_", 1], ["Q_NORM_PARTS_", 4]], "cta_group": 1, "threads": 640}'))
merge_ir = _decode_capture(_json_loads('{"__ir__": "knn_search_mma_split_merge_q128_const148_v1", "arg_keys": ["partial_distances", "partial_indices", "out_distances", "out_indices", "B", "Q", "K", "split_m", "num_q_tiles"], "cluster_dims": [1, 1, 1], "computed_smem_bytes": 0, "constants": [["K_MAX_", 10]], "cta_group": 1, "threads": 32}'))
ir = _decode_capture(_json_loads('{"__ir__": "knn_search_dynamic_d3_tile_reduce_partial_0618_c8b9_v1", "arg_keys": ["queries", "database", "partial_distances", "partial_indices", "B", "Q", "M", "K", "num_m_tiles"], "cluster_dims": [1, 1, 1], "computed_smem_bytes": 10240, "constants": [["K_MAX_", 10]], "cta_group": 1, "threads": 256}'))
ROUTE_TINY_DYNAMIC_D_TCGEN05 = 'c8b9_dynamic_d_tiny_q128_tcgen05'
ROUTE_D3_TILE_REDUCE = 'c8b9_dynamic_d3_q128_tile_reduce'
CONSUMED_SEED = 'weave-evolve-knn-search-round-2-c8b9-tinyd'
TINY_DYNAMIC_D_LABELS: tuple[str, ...] = ('blind_dyn_d3_q128_m65536_k10', 'blind_dyn_d7_q128_m65536_k10', 'blind_dyn_d63_q128_m65536_k10')
TINY_DYNAMIC_D_SHAPES: list[dict[str, Any]] = [{'label': 'blind_dyn_d3_q128_m65536_k10', 'params': {'B': 1, 'Q': 128, 'M': 65536, 'D': 3, 'K': 10, 'dtype': 'bfloat16', 'seed': 610801, 'self_search': False, 'min_recall': 0.999}}, {'label': 'blind_dyn_d7_q128_m65536_k10', 'params': {'B': 1, 'Q': 128, 'M': 65536, 'D': 7, 'K': 10, 'dtype': 'bfloat16', 'seed': 610802, 'self_search': False, 'min_recall': 0.999}}, {'label': 'blind_dyn_d63_q128_m65536_k10', 'params': {'B': 1, 'Q': 128, 'M': 65536, 'D': 63, 'K': 10, 'dtype': 'bfloat16', 'seed': 610803, 'self_search': False, 'min_recall': 0.999}}]
SHAPE_DISPATCH_REGISTRY: tuple[dict[str, Any], ...] = ({'overlay': 'dynamic_d_tiny_q128_tcgen05_c8b9', 'shape_key': 'B=1,Q=128,M=65536,D in {3,7,63},K=10', 'labels': TINY_DYNAMIC_D_LABELS, 'guard': 'B == 1 and Q == 128 and M == 65536 and D in {3,7,63} and K == 10 and not self_search and not forced_fallback', 'route': ''.join([format(ROUTE_D3_TILE_REDUCE, ''), ' / ', format(ROUTE_TINY_DYNAMIC_D_TCGEN05, '')]), 'entrypoint': 'loom.examples.weave.knn_search_dynamic_d_tiny_q128_tcgen05_0618_c8b9_v1:launch_for_eval', 'selected_seed': CONSUMED_SEED, 'source_task': 'generalize-auto-tuning c8b9 priority-1 dynamic-D tiny bucket', 'coverage_class': 'bucket_seed_dynamic_d_tiny_q128_m65536_k10', 'workflow_mode': 'generalize_auto_tuning', 'auto_tuning_stage': 'bucket-kernel'},)

def _kernel_dim(original_d: int) -> int:
    if original_d <= 0:
        raise ValueError(''.join(['feature dimension must be positive, got D=', format(original_d, '')]))
    return max(16, (int(original_d) + 15) // 16 * 16)

def _use_tiny_dynamic_d_tcgen05(inputs: dict[str, Any]) -> bool:
    return int(inputs['B']) == 1 and int(inputs['Q']) == BLOCK_Q and (int(inputs['M']) == 65536) and (int(inputs['D']) in SUPPORTED_ORIGINAL_D) and (int(inputs['K']) == K_MAX) and (not bool(inputs.get('self_search', False))) and (not bool(inputs.get('force_fallback', False))) and lowd.mma._tcgen05_capable_arch()

def _compile_pack_kernel(original_d: int, padded_d: int):
    from .._dispatch_runtime import generate_kernel
    from .._dispatch_runtime import _cuda_include_dirs
    from .._dispatch_runtime import compile_cuda, detect_gpu_arch
    from .._dispatch_runtime import CUDAKernel
    source = generate_kernel(knn_search_dynamic_d_tiny_pack_bf16_0618_c8b9_v1, validate=False, smem_bytes=0, D_ORIG_=int(original_d), D_PAD_=int(padded_d))
    cubin = compile_cuda(source, arch=detect_gpu_arch(), options=['--use_fast_math'], include_dirs=_cuda_include_dirs())
    return CUDAKernel(cubin, ''.join(['kernel_', format(knn_search_dynamic_d_tiny_pack_bf16_0618_c8b9_v1.symbol, '')]))

def _compile_padded_mma_kernels(padded_d: int) -> dict[str, Any]:
    return lowd._compile_non_d128_mma_kernels(int(padded_d))

def _compile_d3_tile_reduce_kernels() -> dict[str, Any]:
    return _decode_capture(_json_loads('{"merge": {"__kernel__": "dispatch_kernel_0309"}, "partial": {"__kernel__": "dispatch_kernel_0308"}}'))

def _padded_buffers(inputs: dict[str, Any], padded_d: int) -> tuple[Any, Any]:
    import torch
    key = (int(inputs['B']), int(inputs['Q']), int(inputs['M']), int(inputs['D']), int(padded_d), int(inputs['queries'].device.index or 0), id(inputs['queries']), str(inputs['queries'].dtype))
    cached = _PADDED_INPUTS.get(key)
    if cached is None:
        cached = (torch.empty((int(inputs['B']), int(inputs['Q']), int(padded_d)), dtype=inputs['queries'].dtype, device=inputs['queries'].device), torch.empty((int(inputs['B']), int(inputs['M']), int(padded_d)), dtype=inputs['database'].dtype, device=inputs['database'].device))
        _PADDED_INPUTS[key] = cached
    return cached

def _d3_scratch(inputs: dict[str, Any], num_m_tiles: int) -> tuple[Any, Any]:
    import torch
    key = (int(inputs['B']), int(inputs['Q']), int(inputs['M']), int(inputs['D']), int(num_m_tiles), int(inputs['queries'].device.index or 0), str(inputs['queries'].dtype))
    cached = _D3_SCRATCH.get(key)
    if cached is None:
        shape = (int(inputs['B']), int(inputs['Q']), int(num_m_tiles), K_MAX)
        cached = (torch.empty(shape, dtype=torch.float32, device=inputs['queries'].device), torch.empty(shape, dtype=torch.int32, device=inputs['queries'].device))
        _D3_SCRATCH[key] = cached
    return cached

def _partial_scratch(inputs: dict[str, Any], padded_d: int, split_m: int, num_q_tiles: int) -> tuple[Any, Any]:
    import torch
    key = (int(inputs['B']), int(inputs['Q']), int(inputs['M']), int(padded_d), int(split_m), int(num_q_tiles), int(inputs['queries'].device.index or 0), str(inputs['queries'].dtype))
    cached = _PADDED_SCRATCH.get(key)
    if cached is None:
        shape = (int(inputs['B']), int(num_q_tiles), int(split_m), BLOCK_Q, K_MAX)
        cached = (torch.empty(shape, dtype=torch.float32, device=inputs['queries'].device), torch.empty(shape, dtype=torch.int32, device=inputs['queries'].device))
        _PADDED_SCRATCH[key] = cached
    return cached

def selected_route(inputs: dict[str, Any]) -> str:
    if _use_tiny_dynamic_d_tcgen05(inputs):
        if int(inputs['D']) == 3:
            return ROUTE_D3_TILE_REDUCE
        return ROUTE_TINY_DYNAMIC_D_TCGEN05
    return 'scalar_capacity_parent'

def selected_route_name(inputs: dict[str, Any]) -> str:
    return selected_route(inputs)

def route_info(inputs: dict[str, Any]) -> dict[str, Any]:
    if _use_tiny_dynamic_d_tcgen05(inputs):
        route = selected_route(inputs)
        return {'route': route, 'selected_route': route, 'selected_entrypoint': 'loom.examples.weave.knn_search_dynamic_d_tiny_q128_tcgen05_0618_c8b9_v1:launch_for_eval', 'route_kind': 'specialized', 'route_source': 'bucket-specific-seed', 'coverage_class': 'bucket_seed_dynamic_d_tiny_q128_m65536_k10', 'classification': 'seed-produced', 'coverage_only': False, 'production_policy': 'weave_only', 'external_fallback': None, 'guard_id': 'dynamic_d_tiny_q128_m65536_k10', 'forced_fallback': False, 'selected_guard': SHAPE_DISPATCH_REGISTRY[0]['guard'], 'fallback': 'scalar_capacity_parent', 'missing_weave_route': False, 'selected_seed': CONSUMED_SEED}
    return {'route': 'scalar_capacity_parent', 'selected_route': 'scalar_capacity_parent', 'selected_entrypoint': 'loom.examples.weave.knn_search_scalar_capacity_0611_r22_4e96_v1:launch_for_eval', 'route_kind': 'general', 'route_source': 'generic-weave-fallback', 'coverage_only': False, 'production_policy': 'weave_only', 'external_fallback': None, 'forced_fallback': bool(inputs.get('force_fallback', False))}

def _launch_tiny_dynamic_d_tcgen05(inputs: dict[str, Any]) -> dict[str, Any]:
    import torch
    original_d = int(inputs['D'])
    padded_d = _kernel_dim(original_d)
    pack_key = (original_d, padded_d)
    pack_kernel = _PACK_KERNELS.get(pack_key)
    if pack_kernel is None:
        pack_kernel = _compile_pack_kernel(original_d, padded_d)
        _PACK_KERNELS[pack_key] = pack_kernel
    kernels = _PADDED_MMA_KERNELS.get(padded_d)
    if kernels is None:
        kernels = _compile_padded_mma_kernels(padded_d)
        _PADDED_MMA_KERNELS[padded_d] = kernels
    padded_queries, padded_database = _padded_buffers(inputs, padded_d)
    bsz = int(inputs['B'])
    q_rows = int(inputs['Q'])
    m_rows = int(inputs['M'])
    k = int(inputs['K'])
    pack_elems = bsz * (q_rows + m_rows) * padded_d
    pack_kernel.launch(grid=(math.ceil(pack_elems / PACK_THREADS), 1, 1), block=(PACK_THREADS, 1, 1), args=[inputs['queries'], inputs['database'], padded_queries, padded_database, bsz, q_rows, m_rows], shared_mem=0)
    split_m = min(TINY_DYNAMIC_D_SPLIT_M, math.ceil(m_rows / BLOCK_M))
    num_q_tiles = math.ceil(q_rows / BLOCK_Q)
    total_m_tiles = math.ceil(m_rows / BLOCK_M)
    tiles_per_split = math.ceil(total_m_tiles / split_m)
    partial_dist, partial_idx = _partial_scratch(inputs, padded_d, split_m, num_q_tiles)
    padded_inputs = dict(inputs)
    padded_inputs['queries'] = padded_queries
    padded_inputs['database'] = padded_database
    padded_inputs['D'] = padded_d
    kernels['partial'].launch(grid=(bsz * num_q_tiles * split_m, 1, 1), block=(THREADS, 1, 1), args=[padded_inputs['queries'], padded_inputs['database'], partial_dist, partial_idx, bsz, q_rows, m_rows, split_m, num_q_tiles, total_m_tiles, tiles_per_split], shared_mem=MMA_SMEM_BYTES)
    kernels['merge'].launch(grid=(bsz * q_rows, 1, 1), block=(MERGE_THREADS, 1, 1), args=[partial_dist, partial_idx, inputs['out_distances'], inputs['out_indices'], bsz, q_rows, k, split_m, num_q_tiles], shared_mem=MERGE_SMEM_BYTES)
    return {'distances': inputs['out_distances'], 'indices': inputs['out_indices']}

def _launch_d3_tile_reduce(inputs: dict[str, Any]) -> dict[str, Any]:
    import torch
    global _D3_TILE_KERNELS
    if _D3_TILE_KERNELS is None:
        _D3_TILE_KERNELS = _compile_d3_tile_reduce_kernels()
    bsz = int(inputs['B'])
    q_rows = int(inputs['Q'])
    m_rows = int(inputs['M'])
    k = int(inputs['K'])
    num_m_tiles = math.ceil(m_rows / D3_BLOCK_M)
    tiles_per_group = max(lowd_tile.MERGE_TILES_PER_GROUP, math.ceil(num_m_tiles / lowd_tile.MERGE_WARPS))
    num_groups = math.ceil(num_m_tiles / tiles_per_group)
    partial_dist, partial_idx = _d3_scratch(inputs, num_m_tiles)
    _D3_TILE_KERNELS['partial'].launch(grid=(bsz * q_rows * num_m_tiles, 1, 1), block=(D3_THREADS, 1, 1), args=[inputs['queries'], inputs['database'], partial_dist, partial_idx, bsz, q_rows, m_rows, k, num_m_tiles], shared_mem=lowd_tile.TILE_SMEM_BYTES)
    _D3_TILE_KERNELS['merge'].launch(grid=(bsz * q_rows, 1, 1), block=(lowd_tile.MERGE_THREADS, 1, 1), args=[partial_dist, partial_idx, inputs['out_distances'], inputs['out_indices'], bsz, q_rows, k, num_m_tiles, num_groups, tiles_per_group], shared_mem=lowd_tile.MERGE_SMEM_BYTES)
    return {'distances': inputs['out_distances'], 'indices': inputs['out_indices']}

def launch_for_eval(inputs: dict[str, Any]) -> dict[str, Any]:
    if _use_tiny_dynamic_d_tcgen05(inputs):
        if int(inputs['D']) == 3:
            return _launch_d3_tile_reduce(inputs)
        return _launch_tiny_dynamic_d_tcgen05(inputs)
    return scalar_capacity.launch_for_eval(inputs)

def _select_contract_shapes(shape_labels: str | tuple[str, ...] | list[str] | None):
    from .._dispatch_runtime import select_named_shapes
    if shape_labels is None:
        return TINY_DYNAMIC_D_SHAPES
    return select_named_shapes(shape_labels)

def knn_search_compile_and_launch_dynamic_d_tiny_q128_tcgen05(*, benchmark: bool=True, shape_labels: str | tuple[str, ...] | list[str] | None=TINY_DYNAMIC_D_LABELS) -> dict[str, Any]:
    from .._dispatch_runtime import evaluate
    result = evaluate(launch_for_eval, shapes=_select_contract_shapes(shape_labels), benchmark=benchmark)
    result['passed'] = bool(result.get('summary', {}).get('all_correct'))
    print(result)
    return result
