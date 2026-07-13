"""Flash-KMeans high-N micro-D raw-TMA seed for D=16/32.

Minimum architecture: sm_100a.  This candidate keeps the proven 6cd2
tcgen05/TMEM score and argmax schedule, but replaces its global pad kernel and
rank-3 padded tensor maps with rank-2 tensor maps over the raw logical rows.
The 64-wide TMA box zero-fills columns beyond logical D, so the contract path
is one assignment kernel with no global scratch handoff.  It is not intended
for sm_120a/sm_121a where ptxas rejects tcgen05 instructions.
"""
from __future__ import annotations
from json import loads as _json_loads
from .._dispatch_runtime import _capture_cuTensorMapEncodeTiled, _decode_capture, _import_dispatch_module, _ir_proxy
from functools import lru_cache
from typing import Any
BLOCK_N = 128
BLOCK_K = 256
SCORE_CHUNK_K = 128
CSQ_VEC = 4
CSQ_STAGE_VEC = 4
FEAT_D_PAD = 64
SUPPORTED_D = {16, 32}
NUM_COMPUTE_WARPS = 4
X_TILE_BYTES = BLOCK_N * FEAT_D_PAD * 2
C_TILE_BYTES = BLOCK_K * FEAT_D_PAD * 2
CSQ_ELEMS = 512
CSQ_TILE_BYTES = CSQ_ELEMS * 4
ROUTE_ID = 'microdim_raw_tma_08f9_v1'
SEED_ID = 'microdim-raw-tma-08f9-v1'
BF16_DTYPE_NAMES = {'bfloat16', 'bf16', 'torch.bfloat16'}
TARGET_SHAPE = 'post_d895_d16_b8_n65536_k512_d16'
TARGET_SHAPES = (TARGET_SHAPE, 'post_d895_d32_b8_n65536_k512_d32')
flash_kmeans_assign_microdim_raw_tma_08f9_v1 = _decode_capture(_json_loads('{"__ir__": "flash_kmeans_assign_microdim_raw_tma_08f9_v1", "arg_keys": ["x_tmap", "c_tmap", "x_sq", "c_sq", "out", "B", "N", "D", "K", "num_n_tiles", "K_tiles"], "cluster_dims": [1, 1, 1], "computed_smem_bytes": 52224, "constants": [], "cta_group": 1, "threads": 192}'))
ir = _decode_capture(_json_loads('{"__ir__": "flash_kmeans_assign_microdim_raw_tma_08f9_v1", "arg_keys": ["x_tmap", "c_tmap", "x_sq", "c_sq", "out", "B", "N", "D", "K", "num_n_tiles", "K_tiles"], "cluster_dims": [1, 1, 1], "computed_smem_bytes": 52224, "constants": [], "cta_group": 1, "threads": 192}'))
_TMAP_CACHE: dict[tuple[int, int, int, int, int, int], Any] = {}

def _cuda_include_dirs() -> list[str]:
    from .._dispatch_runtime import _cuda_include_dirs as _common_cuda_include_dirs
    return _common_cuda_include_dirs()

def _create_raw_tensor_map(data_ptr: int, *, global_height: int, shared_height: int, dim: int):
    import torch
    from .._dispatch_runtime import create_tensor_map
    device_index = torch.cuda.current_device()
    key = (device_index, int(data_ptr), int(global_height), int(shared_height), int(dim), FEAT_D_PAD)
    cached = _TMAP_CACHE.get(key)
    if cached is not None:
        return cached
    cached = create_tensor_map(data_ptr, dim, global_height, FEAT_D_PAD, shared_height, dim * 2).to(device=torch.device('cuda', device_index))
    _TMAP_CACHE[key] = cached
    return cached

def _make_tmaps(inputs: dict[str, Any]) -> tuple[Any, Any]:
    cache_key = (int(inputs['x'].data_ptr()), int(inputs['centroids'].data_ptr()), int(inputs['B']), int(inputs['N']), int(inputs['D']), int(inputs['K']))
    cached = inputs.get('_flash_kmeans_assign_microdim_raw_tma_08f9_v1_tmaps')
    if cached is not None and cached[0] == cache_key:
        return (cached[1], cached[2])
    bsz = int(inputs['B'])
    n_points = int(inputs['N'])
    dim = int(inputs['D'])
    n_clusters = int(inputs['K'])
    tmap_x = _create_raw_tensor_map(inputs['x'].data_ptr(), global_height=bsz * n_points, shared_height=BLOCK_N, dim=dim)
    tmap_c = _create_raw_tensor_map(inputs['centroids'].data_ptr(), global_height=bsz * n_clusters, shared_height=BLOCK_K, dim=dim)
    inputs['_flash_kmeans_assign_microdim_raw_tma_08f9_v1_tmaps'] = (cache_key, tmap_x, tmap_c)
    return (tmap_x, tmap_c)

def _compiled_kernel() -> tuple[bytes, str, int, int]:
    return _decode_capture(_json_loads('{"__tuple__": [{"__kernel_source__": "dispatch_kernel_0023"}, "kernel_flash_kmeans_assign_microdim_raw_tma_08f9_v1", 52224, 192]}'))

def launch_for_eval(inputs: dict[str, Any]) -> dict[str, Any]:
    from .._dispatch_runtime import CUDAKernel
    from .._dispatch_runtime import pack_kernel_args
    bsz = int(inputs['B'])
    n_points = int(inputs['N'])
    dim = int(inputs['D'])
    n_clusters = int(inputs['K'])
    dtype_name = _dtype_name(inputs)
    _validate_supported_shape(B=bsz, N=n_points, D=dim, K=n_clusters, dtype=dtype_name)
    tmap_x, tmap_c = _make_tmaps(inputs)
    cubin, kernel_name, smem_bytes, threads = _compiled_kernel()
    num_n_tiles = n_points // BLOCK_N
    k_tiles = n_clusters // BLOCK_K
    grid = (bsz * num_n_tiles, 1, 1)
    args = pack_kernel_args(ir, x_tmap=tmap_x, c_tmap=tmap_c, x_sq=inputs['x_sq'], c_sq=inputs['c_sq'], out=inputs['out'], B=bsz, N=n_points, D=dim, K=n_clusters, num_n_tiles=num_n_tiles, K_tiles=k_tiles)
    with CUDAKernel(cubin, kernel_name) as kernel:
        kernel.launch(grid=grid, block=(threads, 1, 1), args=args, shared_mem=smem_bytes)
    trace = _route_trace(inputs, total_tiles=bsz * num_n_tiles, grid=grid)
    return {'cluster_ids': inputs['out'], 'selected_route': ROUTE_ID, 'route_trace': trace}

def _validate_supported_shape(*, B: int, N: int, D: int, K: int, dtype: Any) -> None:
    dtype_name = str(dtype).replace('torch.', '')
    if dtype_name not in BF16_DTYPE_NAMES:
        raise ValueError(''.join([format(ROUTE_ID, ''), ' requires bfloat16 input, got ', format(dtype, '')]))
    if B != 8 or N != 65536 or K != 512 or (D not in SUPPORTED_D):
        raise ValueError(''.join([format(ROUTE_ID, ''), ' owns B=8, N=65536, K=512, D in ', format(sorted(SUPPORTED_D), '')]))
    if N % BLOCK_N != 0 or K % BLOCK_K != 0:
        raise ValueError(''.join([format(ROUTE_ID, ''), ' requires N % ', format(BLOCK_N, ''), ' == 0 and K % ', format(BLOCK_K, ''), ' == 0']))

def _route_trace(inputs: dict[str, Any], *, total_tiles: int, grid: tuple[int, int, int]) -> dict[str, Any]:
    return {'shape_key': _shape_key(inputs), 'selected_route': ROUTE_ID, 'selected_entrypoint': 'loom.examples.weave.flash_kmeans_assign_microdim_raw_tma_08f9_v1:launch_for_eval', 'selected_seed': SEED_ID, 'expected_seed': None, 'route_kind': 'specialized', 'route_source': 'shape-specific-seed', 'guard_id': 'guard_microdim_raw_tma_08f9_v1', 'guard_condition': 'B == 8 and N == 65536 and K == 512 and D in [16, 32] and dtype == bfloat16', 'classification': 'route-ok', 'dispatcher_kernel_ms': None, 'shape_specific_kernel_ms': None, 'relative_speedup_vs_baseline': None, 'raw_tma_zero_fill': True, 'global_padding_scratch': False, 'total_tiles': total_tiles, 'launch_grid': grid, 'reason': 'raw rank-2 TMA zero-fills the 64-wide micro-D tile and removes the pack-kernel handoff'}

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
