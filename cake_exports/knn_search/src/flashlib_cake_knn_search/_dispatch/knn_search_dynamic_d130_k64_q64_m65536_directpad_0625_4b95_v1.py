"""D130/Q64/M65536/K64 direct-padded tcgen05 kNN seed.

Minimum target architecture: sm_100a for the tcgen05/TMEM producer path. This
additive bucket-kernel module targets ``B=1,Q=64,M=65536,D=130,K=64`` without
the prior materialized D512 padding workspace. The producer scans the original
stride-D130 tensors directly and zero-fills the unused portion of a D256
tcgen05 tile in shared memory, then reuses the D256/K64 partial-list merge ABI.
Guard misses delegate to the current cd65 dispatcher.
"""
from __future__ import annotations
from json import loads as _json_loads
from .._dispatch_runtime import _capture_cuTensorMapEncodeTiled, _decode_capture, _import_dispatch_module, _ir_proxy
import math
from typing import Any
from .._dispatch_runtime import select_named_shapes
from . import knn_search_dispatch0624_full133_eacf_lowd_cd65_v1 as parent
from . import knn_search_lowd_d256_mma64_over48e9_0612_r34_48e9_v1 as d256_mma
THREADS = d256_mma.THREADS
BLOCK_Q = d256_mma.BLOCK_Q
BLOCK_M = d256_mma.BLOCK_M
D_ORIGINAL = 130
D_PAD_INTERNAL = d256_mma.D_STATIC
K_MAX = d256_mma.K_MAX
D130_SPLIT_M = 148
MMA_POST_MMA_COL_COHORTS = d256_mma.MMA_POST_MMA_COL_COHORTS
MMA_SMEM_A_BYTES = BLOCK_Q * D_PAD_INTERNAL * 2
MMA_SMEM_B_BYTES = BLOCK_M * D_PAD_INTERNAL * 2
MMA_DB_NORM_PARTS = d256_mma.MMA_DB_NORM_PARTS
MMA_DB_NORM_CHUNK = D_PAD_INTERNAL // MMA_DB_NORM_PARTS
MMA_STAGE_VEC_ELEMS = d256_mma.MMA_STAGE_VEC_ELEMS
MMA_STAGE_PACK_WORDS = d256_mma.MMA_STAGE_PACK_WORDS
MMA_Q_STAGE_VECS = BLOCK_Q * D_PAD_INTERNAL // MMA_STAGE_VEC_ELEMS
MMA_Q_NORM_PARTS = D_PAD_INTERNAL // MMA_STAGE_VEC_ELEMS
MMA_DB_NORM_PART_VECS = MMA_DB_NORM_CHUNK // MMA_STAGE_VEC_ELEMS
MMA_SMEM_Q_NORM_PART_BYTES = BLOCK_Q * MMA_Q_NORM_PARTS * 4
MMA_SMEM_DB_NORM_PART_BYTES = BLOCK_M * MMA_DB_NORM_PARTS * 4
MMA_SMEM_DB_NORM_BYTES = BLOCK_M * 4
MMA_SMEM_B0_OFFSET = MMA_SMEM_A_BYTES
MMA_SMEM_B1_OFFSET = MMA_SMEM_B0_OFFSET + MMA_SMEM_B_BYTES
MMA_SMEM_Q_NORM_PART_OFFSET = MMA_SMEM_B1_OFFSET + MMA_SMEM_B_BYTES
MMA_SMEM_DB_NORM_PART0_OFFSET = MMA_SMEM_Q_NORM_PART_OFFSET + MMA_SMEM_Q_NORM_PART_BYTES
MMA_SMEM_DB_NORM_PART1_OFFSET = MMA_SMEM_DB_NORM_PART0_OFFSET + MMA_SMEM_DB_NORM_PART_BYTES
MMA_SMEM_DB_NORM0_OFFSET = MMA_SMEM_DB_NORM_PART1_OFFSET + MMA_SMEM_DB_NORM_PART_BYTES
MMA_SMEM_DB_NORM1_OFFSET = MMA_SMEM_DB_NORM0_OFFSET + MMA_SMEM_DB_NORM_BYTES
MMA_STAGING_END = MMA_SMEM_DB_NORM1_OFFSET + MMA_SMEM_DB_NORM_BYTES
WEAVE_SMEM_SYSTEM_BYTES = 1024
MMA_SMEM_POOL_BYTES = MMA_STAGING_END + 256
MMA_SMEM_BYTES = MMA_SMEM_POOL_BYTES + WEAVE_SMEM_SYSTEM_BYTES
MERGE_THREADS = d256_mma.MERGE_THREADS
MERGE_SMEM_BYTES = d256_mma.MERGE_SMEM_BYTES
ROUTE_D130_Q64_K64_DIRECTPAD = '4b95_d130_q64_m65536_k64_direct_d256pad_tcgen05'
SELECTED_SEED = 'weave-evolve-knn-search-4b95-d130-k64-directpad'
PARENT_ROUTE = '9d5c_r117_dynamic_d130_q64_k64_d512packed_tcgen05'
ENTRYPOINT = 'loom.examples.weave.knn_search_dynamic_d130_k64_q64_m65536_directpad_0625_4b95_v1:launch_for_eval'
TARGET_LABELS: tuple[str, ...] = ('blind_ext_dyn_d130_k64_q64_m65536',)
TARGET_SHAPES = _decode_capture(_json_loads('[{"__dict_items__": [["label", "blind_ext_dyn_d130_k64_q64_m65536"], ["params", {"__dict_items__": [["B", 1], ["Q", 64], ["M", 65536], ["D", 130], ["K", 64], ["dtype", "bfloat16"], ["seed", 610929], ["self_search", false], ["min_recall", 0.999]]}]]}]'))
_KERNELS: dict[str, Any] = {}
_SCRATCH: dict[tuple[int, int, int, int, int, int, int, str], tuple[Any, Any]] = {}
_mma_issue_d130 = _ir_proxy('loom.examples.weave.knn_search_dynamic_d130_k64_q64_m65536_directpad_0625_4b95_v1:_mma_issue_d130', 256)
_stage_database_tile_d130 = _ir_proxy('loom.examples.weave.knn_search_dynamic_d130_k64_q64_m65536_directpad_0625_4b95_v1:_stage_database_tile_d130', 256)
knn_search_d130_q64_k64_directpad_partial_0625_4b95_v1 = _decode_capture(_json_loads('{"__ir__": "knn_search_d130_q64_k64_directpad_partial_0625_4b95_v1", "arg_keys": ["queries", "database", "partial_distances", "partial_indices", "B", "Q", "M", "split_m", "num_q_tiles", "total_m_tiles", "tiles_per_split"], "cluster_dims": [1, 1, 1], "computed_smem_bytes": 143104, "constants": [["K_MAX_", 64]], "cta_group": 1, "threads": 256}'))
partial_ir = _decode_capture(_json_loads('{"__ir__": "knn_search_d130_q64_k64_directpad_partial_0625_4b95_v1", "arg_keys": ["queries", "database", "partial_distances", "partial_indices", "B", "Q", "M", "split_m", "num_q_tiles", "total_m_tiles", "tiles_per_split"], "cluster_dims": [1, 1, 1], "computed_smem_bytes": 143104, "constants": [["K_MAX_", 64]], "cta_group": 1, "threads": 256}'))
merge_ir = _decode_capture(_json_loads('{"__ir__": "knn_search_mma_split_merge_v1", "arg_keys": ["partial_distances", "partial_indices", "out_distances", "out_indices", "B", "Q", "K", "split_m", "num_q_tiles"], "cluster_dims": [1, 1, 1], "computed_smem_bytes": 0, "constants": [["K_MAX_", 64]], "cta_group": 1, "threads": 32}'))
ir = _decode_capture(_json_loads('{"__ir__": "knn_search_d130_q64_k64_directpad_partial_0625_4b95_v1", "arg_keys": ["queries", "database", "partial_distances", "partial_indices", "B", "Q", "M", "split_m", "num_q_tiles", "total_m_tiles", "tiles_per_split"], "cluster_dims": [1, 1, 1], "computed_smem_bytes": 143104, "constants": [["K_MAX_", 64]], "cta_group": 1, "threads": 256}'))
SHAPE_DISPATCH_REGISTRY: tuple[dict[str, Any], ...] = ({'shape_key': '4b95_d130_q64_m65536_k64_directpad', 'label': 'blind_ext_dyn_d130_k64_q64_m65536', 'labels': TARGET_LABELS, 'shape': (1, 64, 65536, D_ORIGINAL, K_MAX, False), 'guard': 'B == 1 and Q == 64 and M == 65536 and D == 130 and K == 64 and not self_search and not forced_fallback and arch in {sm_100a,sm_103a}', 'route': ROUTE_D130_Q64_K64_DIRECTPAD, 'entrypoint': ENTRYPOINT, 'parent_route': PARENT_ROUTE, 'selected_seed': SELECTED_SEED, 'source_round_doc': 'design_doc/active/weave_evolve_knn_search_round_156_4b95_d130_directpad.md', 'coverage_class': 'bucket_seed_d130_q64_m65536_k64_direct_d256pad', 'arch_requirement': 'sm_100a'},)

def _shape_key(inputs: dict[str, Any]) -> tuple[int, int, int, int, int, bool]:
    return (int(inputs.get('B', 1)), int(inputs['Q']), int(inputs['M']), int(inputs['D']), int(inputs['K']), bool(inputs.get('self_search', False)))

def _tcgen05_capable_arch() -> bool:
    return bool(d256_mma._tcgen05_capable_arch())

def _active_entry(inputs: dict[str, Any]) -> dict[str, Any] | None:
    if bool(inputs.get('force_fallback', False)) or not _tcgen05_capable_arch():
        return None
    shape = _shape_key(inputs)
    for entry in SHAPE_DISPATCH_REGISTRY:
        if shape == entry['shape']:
            return entry
    return None

def selected_route(inputs: dict[str, Any]) -> str:
    entry = _active_entry(inputs)
    if entry is not None:
        return str(entry['route'])
    return parent.selected_route(inputs)

def selected_route_name(inputs: dict[str, Any]) -> str:
    return selected_route(inputs)

def _entry_info(inputs: dict[str, Any], entry: dict[str, Any]) -> dict[str, Any]:
    parent_info = dict(parent.route_info(inputs))
    parent_route = str(parent_info.get('selected_route') or parent_info.get('route') or parent.selected_route(inputs))
    return {'route': entry['route'], 'selected_route': entry['route'], 'selected_entrypoint': entry['entrypoint'], 'parent_route': parent_route, 'replaced_route': parent_route, 'route_kind': 'specialized', 'route_source': 'bucket-specific-seed', 'coverage_class': entry['coverage_class'], 'classification': 'seed-produced', 'coverage_only': False, 'production_policy': 'weave_only', 'external_fallback': None, 'guard_id': entry['shape_key'], 'selected_guard': entry['guard'], 'guard_condition': entry['guard'], 'fallback': parent_route, 'missing_weave_route': False, 'forced_fallback': False, 'selected_seed': entry['selected_seed'], 'selected_seed_task': entry['selected_seed'], 'source_round_doc': entry['source_round_doc'], 'arch_requirement': entry['arch_requirement'], 'pack_route': 'none', 'scan_route': entry['route'], 'padding_tag': 'kernel_padded_d_schedule', 'uses_materialized_padding': False, 'uses_kernel_padding': True, 'padding_overhead_timed': True, 'original_D': D_ORIGINAL, 'padded_D': D_PAD_INTERNAL, 'padding_ratio': D_PAD_INTERNAL / D_ORIGINAL, 'workspace_reuse': 'module_cache_by_shape_device_input_identity'}

def route_info(inputs: dict[str, Any]) -> dict[str, Any]:
    entry = _active_entry(inputs)
    if entry is not None:
        return _entry_info(inputs, entry)
    info = dict(parent.route_info(inputs))
    info.setdefault('production_policy', 'weave_only')
    info.setdefault('external_fallback', None)
    return info

def route_trace_entry(label: str, inputs: dict[str, Any]) -> dict[str, Any]:
    return {'label': label, 'shape_key': label, 'B': int(inputs['B']), 'Q': int(inputs['Q']), 'M': int(inputs['M']), 'D': int(inputs['D']), 'K': int(inputs['K']), 'self_search': bool(inputs.get('self_search', False)), 'force_fallback': bool(inputs.get('force_fallback', False)), **route_info(inputs)}

def _compile_kernels() -> dict[str, Any]:
    return _decode_capture(_json_loads('{"merge": {"__kernel__": "dispatch_kernel_0279"}, "partial": {"__kernel__": "dispatch_kernel_0278"}}'))

def _ensure_kernels() -> dict[str, Any]:
    if not _KERNELS:
        _KERNELS.update(_compile_kernels())
    return _KERNELS

def _scratch(inputs: dict[str, Any], partial_list_count: int, num_q_tiles: int) -> tuple[Any, Any]:
    import torch
    key = (int(inputs['B']), int(inputs['Q']), int(inputs['M']), int(partial_list_count), int(num_q_tiles), K_MAX, int(inputs['queries'].device.index or 0), str(inputs['queries'].dtype))
    cached = _SCRATCH.get(key)
    if cached is None:
        shape = (int(inputs['B']), int(num_q_tiles), int(partial_list_count), BLOCK_Q, K_MAX)
        cached = (torch.empty(shape, dtype=torch.float32, device=inputs['queries'].device), torch.empty(shape, dtype=torch.int32, device=inputs['queries'].device))
        _SCRATCH[key] = cached
    return cached

def _launch_d130_q64_k64_directpad(inputs: dict[str, Any]) -> dict[str, Any]:
    kernels = _ensure_kernels()
    bsz = int(inputs['B'])
    q_rows = int(inputs['Q'])
    m_rows = int(inputs['M'])
    k = int(inputs['K'])
    num_q_tiles = math.ceil(q_rows / BLOCK_Q)
    total_m_tiles = math.ceil(m_rows / BLOCK_M)
    split_m = min(D130_SPLIT_M, total_m_tiles)
    tiles_per_split = math.ceil(total_m_tiles / split_m)
    partial_list_count = split_m * MMA_POST_MMA_COL_COHORTS
    partial_dist, partial_idx = _scratch(inputs, partial_list_count, num_q_tiles)
    kernels['partial'].launch(grid=(bsz * num_q_tiles * split_m, 1, 1), block=(THREADS, 1, 1), args=[inputs['queries'], inputs['database'], partial_dist, partial_idx, bsz, q_rows, m_rows, split_m, num_q_tiles, total_m_tiles, tiles_per_split], shared_mem=MMA_SMEM_BYTES)
    kernels['merge'].launch(grid=(bsz * q_rows, 1, 1), block=(MERGE_THREADS, 1, 1), args=[partial_dist, partial_idx, inputs['out_distances'], inputs['out_indices'], bsz, q_rows, k, partial_list_count, num_q_tiles], shared_mem=MERGE_SMEM_BYTES)
    return {'distances': inputs['out_distances'], 'indices': inputs['out_indices']}

def launch_for_eval(inputs: dict[str, Any]) -> dict[str, Any]:
    if _active_entry(inputs) is not None:
        return _launch_d130_q64_k64_directpad(inputs)
    return parent.launch_for_eval(inputs)

def knn_search_compile_and_launch_d130_q64_k64_directpad(*, benchmark: bool=True, shapes: list[dict[str, Any]] | None=None) -> dict[str, Any]:
    from .._dispatch_runtime import evaluate
    result = evaluate(launch_for_eval, shapes=TARGET_SHAPES if shapes is None else shapes, benchmark=benchmark)
    result['passed'] = bool(result.get('summary', {}).get('all_correct'))
    print(result)
    return result
