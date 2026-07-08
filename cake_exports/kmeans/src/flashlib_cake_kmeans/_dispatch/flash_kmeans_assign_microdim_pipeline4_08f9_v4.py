"""Flash-KMeans high-N D16 four-stage pipeline with D32 raw-TMA sibling.

Minimum architecture: sm_100a.  D16 keeps one CTA per 128-point tile and
uses exact K16, 32B-swizzled TMA operands.  Four centroid stages and four TMEM
score stages overlap the eight K64 centroid chunks, while three point stages
feed a bounded persistent grid.  A single cooperative K512 c_sq preload
replaces the per-stage load/synchronize sequence, and four independent score
chains shorten the per-row argmax dependency.  D32 retains the correct
raw-TMA v1 child.  There is no global scratch, partial output, or
host/reference sidecar.  This module
is not intended for sm_120a/sm_121a where ptxas rejects tcgen05 instructions.
"""
from __future__ import annotations
from json import loads as _json_loads
from .._dispatch_runtime import _capture_cuTensorMapEncodeTiled, _decode_capture, _import_dispatch_module, _ir_proxy
from functools import lru_cache
from typing import Any
from . import flash_kmeans_assign_microdim_raw_tma_08f9_v1 as _d32
BLOCK_N = 128
BLOCK_K = 64
FEAT_D = 16
NUM_C_STAGES = 4
NUM_ACC_STAGES = 4
NUM_X_STAGES = 3
NUM_COMPUTE_WARPS = 4
THREADS = 192
GRID_CAP = 152
X_STAGE_BYTES = BLOCK_N * FEAT_D * 2
C_STAGE_BYTES = BLOCK_K * FEAT_D * 2
X_TOTAL_BYTES = X_STAGE_BYTES * NUM_X_STAGES
C_TOTAL_BYTES = C_STAGE_BYTES * NUM_C_STAGES
CSQ_ELEMS = 512
CSQ_BYTES = CSQ_ELEMS * 4
ROUTE_ID = 'microdim_pipeline4_08f9_v4'
SEED_ID = 'microdim-pipeline4-08f9-v4'
D16_CHILD_ROUTE_ID = 'microdim_d16_pipeline4_k64_08f9_v4'
D32_CHILD_ROUTE_ID = _d32.ROUTE_ID
TARGET_SHAPE = 'post_d895_d16_b8_n65536_k512_d16'
TARGET_SHAPES = (TARGET_SHAPE, 'post_d895_d32_b8_n65536_k512_d32')
BF16_DTYPE_NAMES = {'bfloat16', 'bf16', 'torch.bfloat16'}
_TMAP_CACHE: dict[tuple[int, int, int, int, int, int], Any] = {}
flash_kmeans_assign_microdim_d16_pipeline4_08f9_v4 = _decode_capture(_json_loads('{"__ir__": "flash_kmeans_assign_microdim_d16_pipeline4_08f9_v4", "arg_keys": ["x_tmap", "c_tmap", "c_sq", "out", "B", "N", "D", "K", "num_n_tiles", "K_tiles"], "cluster_dims": [1, 1, 1], "computed_smem_bytes": 23552, "constants": [], "cta_group": 1, "threads": 192}'))
ir = _decode_capture(_json_loads('{"__ir__": "flash_kmeans_assign_microdim_d16_pipeline4_08f9_v4", "arg_keys": ["x_tmap", "c_tmap", "c_sq", "out", "B", "N", "D", "K", "num_n_tiles", "K_tiles"], "cluster_dims": [1, 1, 1], "computed_smem_bytes": 23552, "constants": [], "cta_group": 1, "threads": 192}'))

def _cuda_include_dirs() -> list[str]:
    from .._dispatch_runtime import _cuda_include_dirs as _common_cuda_include_dirs
    return _common_cuda_include_dirs()

def _create_d16_tmap(data_ptr: int, *, global_height: int, shared_height: int):
    import torch
    from .._dispatch_runtime import create_tensor_map_3d_32b
    device_index = torch.cuda.current_device()
    key = (device_index, int(data_ptr), int(global_height), int(shared_height), FEAT_D, FEAT_D)
    cached = _TMAP_CACHE.get(key)
    if cached is not None:
        return cached
    cached = create_tensor_map_3d_32b(data_ptr, global_height, shared_height, FEAT_D, FEAT_D).to(device=torch.device('cuda', device_index))
    _TMAP_CACHE[key] = cached
    return cached

def _make_d16_tmaps(inputs: dict[str, Any]) -> tuple[Any, Any]:
    cache_key = (int(inputs['x'].data_ptr()), int(inputs['centroids'].data_ptr()), int(inputs['B']), int(inputs['N']), int(inputs['K']), FEAT_D)
    cached = inputs.get('_flash_kmeans_assign_microdim_pipeline4_08f9_v4_tmaps')
    if cached is not None and cached[0] == cache_key:
        return (cached[1], cached[2])
    bsz = int(inputs['B'])
    n_points = int(inputs['N'])
    n_clusters = int(inputs['K'])
    tmap_x = _create_d16_tmap(inputs['x'].data_ptr(), global_height=bsz * n_points, shared_height=BLOCK_N)
    tmap_c = _create_d16_tmap(inputs['centroids'].data_ptr(), global_height=bsz * n_clusters, shared_height=BLOCK_K)
    inputs['_flash_kmeans_assign_microdim_pipeline4_08f9_v4_tmaps'] = (cache_key, tmap_x, tmap_c)
    return (tmap_x, tmap_c)

def _compiled_d16_kernel() -> tuple[bytes, str, int, int]:
    return _decode_capture(_json_loads('{"__tuple__": [{"__kernel_source__": "dispatch_kernel_0330"}, "kernel_flash_kmeans_assign_microdim_d16_pipeline4_08f9_v4", 23552, 192]}'))

def _launch_d16(inputs: dict[str, Any]) -> None:
    from .._dispatch_runtime import CUDAKernel
    from .._dispatch_runtime import pack_kernel_args
    bsz = int(inputs['B'])
    n_points = int(inputs['N'])
    n_clusters = int(inputs['K'])
    num_n_tiles = n_points // BLOCK_N
    k_tiles = n_clusters // BLOCK_K
    tmap_x, tmap_c = _make_d16_tmaps(inputs)
    cubin, kernel_name, smem_bytes, threads = _compiled_d16_kernel()
    args = pack_kernel_args(ir, x_tmap=tmap_x, c_tmap=tmap_c, c_sq=inputs['c_sq'], out=inputs['out'], B=bsz, N=n_points, D=FEAT_D, K=n_clusters, num_n_tiles=num_n_tiles, K_tiles=k_tiles)
    grid = (min(bsz * num_n_tiles, GRID_CAP), 1, 1)
    with CUDAKernel(cubin, kernel_name) as kernel:
        kernel.launch(grid=grid, block=(threads, 1, 1), args=args, shared_mem=smem_bytes)

def launch_for_eval(inputs: dict[str, Any]) -> dict[str, Any]:
    bsz = int(inputs['B'])
    n_points = int(inputs['N'])
    dim = int(inputs['D'])
    n_clusters = int(inputs['K'])
    _validate_supported_shape(B=bsz, N=n_points, D=dim, K=n_clusters, dtype=_dtype_name(inputs))
    if dim == FEAT_D:
        _launch_d16(inputs)
        child_route = D16_CHILD_ROUTE_ID
        child_seed = 'microdim-d16-pipeline4-k64-08f9-v4'
    else:
        child_outputs = _d32.launch_for_eval(inputs)
        if child_outputs['cluster_ids'].data_ptr() != inputs['out'].data_ptr():
            raise AssertionError('D32 raw-TMA child must write the caller-owned output')
        child_route = D32_CHILD_ROUTE_ID
        child_seed = _d32.SEED_ID
    trace = _route_trace(inputs, child_route=child_route, child_seed=child_seed, pipelined=dim == FEAT_D)
    return {'cluster_ids': inputs['out'], 'selected_route': ROUTE_ID, 'route_trace': trace}

def compile_and_launch_flash_kmeans_assign_microdim_b8_highn_2(*, benchmark: bool=False) -> dict[str, Any]:
    from .._dispatch_runtime import evaluate
    shapes = _exact_shapes(benchmark=benchmark)

    def checked(inputs: dict[str, Any]) -> dict[str, Any]:
        outputs = launch_for_eval(inputs)
        if outputs['cluster_ids'].data_ptr() != inputs['out'].data_ptr():
            raise AssertionError('microdim pipeline4 seed must write the caller-owned output')
        if outputs['selected_route'] != ROUTE_ID:
            raise AssertionError('microdim pipeline4 seed returned unexpected route metadata')
        if outputs['route_trace']['shape_key'] != inputs['label']:
            raise AssertionError('microdim pipeline4 seed returned the wrong shape key')
        return outputs
    payload = evaluate(checked, shapes=shapes, correctness=True, benchmark=benchmark, time_triton_baseline=False)
    return {'passed': bool(payload['correctness']['all_correct']), 'contract_eval': payload}

def _exact_shapes(*, benchmark: bool) -> list[dict[str, Any]]:
    return [{'label': TARGET_SHAPES[0], 'params': {'B': 8, 'N': 65536, 'K': 512, 'D': 16, 'dtype': 'bfloat16', 'seed': 21601, 'check_correctness': True, 'benchmark': benchmark}}, {'label': TARGET_SHAPES[1], 'params': {'B': 8, 'N': 65536, 'K': 512, 'D': 32, 'dtype': 'bfloat16', 'seed': 23201, 'check_correctness': True, 'benchmark': benchmark}}]

def _validate_supported_shape(*, B: int, N: int, D: int, K: int, dtype: Any) -> None:
    dtype_name = str(dtype).replace('torch.', '')
    if dtype_name not in BF16_DTYPE_NAMES:
        raise ValueError(''.join([format(ROUTE_ID, ''), ' requires bfloat16 input, got ', format(dtype, '')]))
    if B != 8 or N != 65536 or K != 512 or (D not in {16, 32}):
        raise ValueError(''.join([format(ROUTE_ID, ''), ' owns B=8, N=65536, K=512, D in [16, 32]']))
    if N % BLOCK_N != 0 or K % BLOCK_K != 0:
        raise ValueError(''.join([format(ROUTE_ID, ''), ' requires N % ', format(BLOCK_N, ''), ' == 0 and K % ', format(BLOCK_K, ''), ' == 0']))

def _route_trace(inputs: dict[str, Any], *, child_route: str, child_seed: str, pipelined: bool) -> dict[str, Any]:
    return {'shape_key': _shape_key(inputs), 'selected_route': ROUTE_ID, 'selected_entrypoint': 'loom.examples.weave.flash_kmeans_assign_microdim_pipeline4_08f9_v4:launch_for_eval', 'selected_seed': SEED_ID, 'expected_seed': None, 'route_kind': 'specialized', 'route_source': 'shape-specific-seed', 'guard_id': 'guard_microdim_pipeline4_08f9_v4', 'guard_condition': 'B == 8 and N == 65536 and K == 512 and D in [16, 32] and dtype == bfloat16', 'classification': 'route-ok', 'dispatcher_kernel_ms': None, 'shape_specific_kernel_ms': None, 'relative_speedup_vs_baseline': None, 'child_route': child_route, 'child_seed': child_seed, 'pipelined': pipelined, 'caller_owned_output': True, 'residual_contract_regions': [], 'reason': 'D16 preloads K512 c_sq once and overlaps exact K16 K64 tiles through four C/TMEM stages; D32 retains the one-CTA raw-TMA sibling'}

def _shape_key(inputs: dict[str, Any]) -> str:
    label = inputs.get('label')
    if label:
        return str(label)
    return ''.join(['b', format(int(inputs['B']), ''), '_n', format(int(inputs['N']), ''), '_k', format(int(inputs['K']), ''), '_d', format(int(inputs['D']), '')])

def _dtype_name(inputs: dict[str, Any]) -> str:
    dtype = inputs.get('dtype')
    if dtype is not None:
        return str(dtype).replace('torch.', '')
    x = inputs.get('x')
    if x is not None and hasattr(x, 'dtype'):
        return str(x.dtype).replace('torch.', '')
    return 'bfloat16'
