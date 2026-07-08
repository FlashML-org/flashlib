"""kNN search K20 rectangular-tail work-feed route.

Minimum target architecture: sm_100a. This additive shape candidate targets the
BF16 D128 non-build K20 rectangular rows:
``search_rect_b1_q1536_m65536_d128_k20`` and
``search_rect_b1_q4096_m65536_d128_k20``.

This module keeps Q4096 on the inherited fast split4 path and gives Q1536 a
higher-fanout split8/split16 unordered tcgen05 producer with an eight-warp K20
merge. The objective is to increase stage work feed without reintroducing the
slow bucket fallback. All other shapes delegate to the validated v41 Weave
dispatcher fallback.
"""
from __future__ import annotations
from json import loads as _json_loads
from .._dispatch_runtime import _capture_cuTensorMapEncodeTiled, _decode_capture, _import_dispatch_module, _ir_proxy
import os
from functools import lru_cache
from typing import Any
from .. import _dispatch_runtime as eval_mod
from . import knn_build_k20_large_lowfanout_de1a_v1 as de1a
from . import knn_build_k20_mergeown_08ec_v3 as mergeown
from .._dispatch_runtime import pack_kernel_args
parent_v20 = de1a.parent_v20
parent_split = de1a.parent_split
base_v1 = de1a.base_v1
parent_v41 = de1a.parent_v41
BLOCK_Q = de1a.BLOCK_Q
BLOCK_M = de1a.BLOCK_M
FEAT_D = de1a.FEAT_D
STAGE1_THREADS = de1a.STAGE1_THREADS
K20_MERGE_THREADS = de1a.K20_MERGE_THREADS
GRID_DIM_DEFAULT = de1a.GRID_DIM_DEFAULT
CTA_GROUP = de1a.CTA_GROUP
TOP_K_K20 = de1a.TOP_K_K20
WORKFEED_TAIL_SPLIT_COUNT = 8
WORKFEED_SPLIT_COUNTS = (8, 16)
WARP8_MERGE_THREADS = mergeown.WARP8_MERGE_THREADS
EXACT_SHAPE_LABELS = ('search_rect_b1_q1536_m65536_d128_k20', 'search_rect_b1_q4096_m65536_d128_k20')
TAIL_Q_M = (1536, 65536)
FAST_Q_M = (4096, 65536)
EXACT_SHAPE_DIMS = {TAIL_Q_M, FAST_Q_M}
merge_k20_tail_s8_warp8_ir = _decode_capture(_json_loads('{"__ir__": "knn_build_k20_mergeown_08ec_warp8_select_q1536s8warp8", "arg_keys": ["partial_dists", "partial_indices", "out_dists", "out_indices", "total_queries"], "cluster_dims": [1, 1, 1], "computed_smem_bytes": 0, "constants": [["TOP_K_MAX", 20], ["SPLIT_COUNT", 8]], "cta_group": 1, "threads": 256}'))
merge_k20_tail_s16_warp8_ir = _decode_capture(_json_loads('{"__ir__": "knn_build_k20_mergeown_08ec_warp8_select_q1536s16warp8", "arg_keys": ["partial_dists", "partial_indices", "out_dists", "out_indices", "total_queries"], "cluster_dims": [1, 1, 1], "computed_smem_bytes": 0, "constants": [["TOP_K_MAX", 20], ["SPLIT_COUNT", 16]], "cta_group": 1, "threads": 256}'))

def _verify_export_ir() -> Any:
    verify_kernel = os.environ.get('LOOM_KNN_K20_TAIL_9B9F_WFEED_VERIFY_KERNEL')
    if verify_kernel in {'stage1_k20_tail', 'stage1_k20_unordered'}:
        return parent_v20.stage1_k20_unordered_ir
    if verify_kernel in {'merge_k20_tail_s4', 'merge_k20_unordered_warp_select'}:
        return parent_v20.merge_k20_unordered_warp_select_ir
    if verify_kernel == 'merge_k20_tail_s2':
        return de1a.merge_k20_s2_warp_select_ir
    if verify_kernel == 'merge_k20_tail_s8_warp8':
        return merge_k20_tail_s8_warp8_ir
    if verify_kernel == 'merge_k20_tail_s16_warp8':
        return merge_k20_tail_s16_warp8_ir
    return parent_v20.stage1_k20_unordered_ir
ir = _decode_capture(_json_loads('{"__ir__": "knn_build_evolve_7bfc_split_cg2_u2_stage1_k32_unordered_k20unordered", "arg_keys": ["tmap_query", "tmap_database", "query_sq", "database_sq", "partial_dists", "partial_indices", "B", "Q", "M", "K", "num_q_tile_pairs", "db_tiles_per_split", "split_count", "total_work"], "cluster_dims": [2, 1, 1], "computed_smem_bytes": 50432, "constants": [["BLOCK_Q", 128], ["BLOCK_M", 64], ["FEAT_D", 128], ["TOP_K_MAX", 20]], "cta_group": 1, "threads": 192}'))

def _forced_tail_split_count() -> int | None:
    split_text = os.environ.get('LOOM_KNN_K20_TAIL_9B9F_WFEED_SPLITS')
    if not split_text:
        return None
    split_count = int(split_text)
    if split_count not in (2, 4, *WORKFEED_SPLIT_COUNTS):
        raise ValueError(''.join(['unsupported K20 Q1536 tail split count: ', format(split_count, '')]))
    return split_count

def _tail_split_count() -> int:
    return _forced_tail_split_count() or WORKFEED_TAIL_SPLIT_COUNT

def _merge_k20_tail_warp8_ir(split_count: int) -> Any:
    if split_count == 8:
        return merge_k20_tail_s8_warp8_ir
    if split_count == 16:
        return merge_k20_tail_s16_warp8_ir
    raise ValueError(''.join(['unsupported K20 Q1536 tail warp8 split count: ', format(split_count, '')]))

@lru_cache(maxsize=2)
def _compiled_merge_k20_tail_warp8(split_count: int):
    return parent_v20._compile_ir(_merge_k20_tail_warp8_ir(split_count))

def _eligible_k20_tail(inputs: dict[str, Any]) -> bool:
    q_m = (int(inputs['Q']), int(inputs['M']))
    return not bool(inputs.get('build', False)) and str(inputs['query'].dtype) == 'torch.bfloat16' and (str(inputs['database'].dtype) == 'torch.bfloat16') and (int(inputs['B']) == 1) and (q_m in EXACT_SHAPE_DIMS) and (int(inputs['D']) == FEAT_D) and (int(inputs['K']) == TOP_K_K20)

def _launch_q1536_forced_unordered(inputs: dict[str, Any], *, split_count: int) -> None:
    query = inputs['query']
    database = inputs['database']
    bsz = int(inputs['B'])
    n_query = int(inputs['Q'])
    n_database = int(inputs['M'])
    dim = int(inputs['D'])
    top_k = int(inputs['K'])
    num_q_tiles = (n_query + BLOCK_Q - 1) // BLOCK_Q
    num_q_tile_pairs = (num_q_tiles + CTA_GROUP - 1) // CTA_GROUP
    num_db_tiles = (n_database + BLOCK_M - 1) // BLOCK_M
    db_tiles_per_split = (num_db_tiles + split_count - 1) // split_count
    total_work = bsz * num_q_tile_pairs * split_count
    stage1_grid = min(total_work * CTA_GROUP, GRID_DIM_DEFAULT)
    partial_dists, partial_indices = parent_split._partial_buffers(split_count=split_count, bsz=bsz, n_query=n_query, top_k=top_k, device=query.device)
    tmap_query = base_v1._create_tensor_map_3d_oob_zero(query.data_ptr(), bsz * n_query, BLOCK_Q, dim, dim)
    tmap_database = base_v1._create_tensor_map_3d_oob_zero(database.data_ptr(), bsz * n_database, BLOCK_M, dim, dim)
    stage1_ir_obj = parent_v20.stage1_k20_unordered_ir
    stage1_kernel = parent_v20._compiled_stage1_unordered_for_exact_k(top_k)
    stage1_kernel.launch_cluster(grid=(stage1_grid, 1, 1), block=(STAGE1_THREADS, 1, 1), args=pack_kernel_args(stage1_ir_obj, tmap_query=tmap_query, tmap_database=tmap_database, query_sq=inputs['query_sq'], database_sq=inputs['database_sq'], partial_dists=partial_dists, partial_indices=partial_indices, B=bsz, Q=n_query, M=n_database, K=top_k, num_q_tile_pairs=num_q_tile_pairs, db_tiles_per_split=db_tiles_per_split, split_count=split_count, total_work=total_work), cluster_dims=(CTA_GROUP, 1, 1), shared_mem=stage1_ir_obj.computed_smem_bytes)
    if split_count == 2:
        merge_ir = de1a.merge_k20_s2_warp_select_ir
        merge_kernel = de1a._compiled_merge_k20_s2_warp_select()
        merge_threads = K20_MERGE_THREADS
        merge_grid = (bsz * n_query + 3) // 4
    elif split_count == 4:
        merge_ir = parent_v20.merge_k20_unordered_warp_select_ir
        merge_kernel = parent_v20._compiled_merge_k20_unordered_warp_select()
        merge_threads = K20_MERGE_THREADS
        merge_grid = (bsz * n_query + 3) // 4
    else:
        merge_ir = _merge_k20_tail_warp8_ir(split_count)
        merge_kernel = _compiled_merge_k20_tail_warp8(split_count)
        merge_threads = WARP8_MERGE_THREADS
        merge_grid = (bsz * n_query + 7) // 8
    merge_kernel.launch(grid=(merge_grid, 1, 1), block=(merge_threads, 1, 1), args=[partial_dists, partial_indices, inputs['out_dists'], inputs['out_indices'], bsz * n_query], shared_mem=merge_ir.computed_smem_bytes)

def _launch_k20_tail(inputs: dict[str, Any]) -> None:
    q_m = (int(inputs['Q']), int(inputs['M']))
    if q_m == TAIL_Q_M:
        _launch_q1536_forced_unordered(inputs, split_count=_tail_split_count())
        return
    parent_v20._launch_k32_split_path(inputs, split_count=de1a.SEED_SPLIT_COUNT)

def launch_from_contract_inputs(inputs: dict[str, Any]) -> None:
    if _eligible_k20_tail(inputs):
        _launch_k20_tail(inputs)
        return
    parent_v41.launch_from_contract_inputs(inputs)

def candidate(inputs: dict[str, Any]):
    launch_from_contract_inputs(inputs)
    return None

def evaluate_contract(*, shapes=None, correctness: bool=True, benchmark: bool=True) -> dict[str, Any]:
    from .._dispatch_runtime import evaluate
    return evaluate(candidate, shapes=shapes, correctness=correctness, benchmark=benchmark)

def _select_contract_shapes(shape_labels):
    return de1a._select_contract_shapes(shape_labels)

def compile_and_launch_knn_build(*, shape_labels=EXACT_SHAPE_LABELS, benchmark: bool=False) -> dict[str, Any]:
    report = evaluate_contract(shapes=_select_contract_shapes(shape_labels), correctness=True, benchmark=benchmark)
    passed = bool(report.get('summary', {}).get('all_correct', False))
    report['passed'] = passed
    report['all_pass'] = passed
    return report

def _set_bench_backend(use_cupti: bool):
    previous = eval_mod.CONTRACT.bench.get('use_cupti', True)
    eval_mod.CONTRACT.bench['use_cupti'] = bool(use_cupti)
    return previous

def benchmark_k20_tail_q1536_9b9f_wfeed_v2(*, use_cupti: bool=True) -> dict[str, Any]:
    prior_use_cupti = _set_bench_backend(use_cupti)
    try:
        report = evaluate_contract(shapes=_select_contract_shapes(EXACT_SHAPE_LABELS), correctness=True, benchmark=True)
    finally:
        eval_mod.CONTRACT.bench['use_cupti'] = prior_use_cupti
    rows = report['per_shape']
    split_counts = {label: _tail_split_count() if (int(rows[label]['Q']), int(rows[label]['M'])) == TAIL_Q_M else de1a.SEED_SPLIT_COUNT for label in EXACT_SHAPE_LABELS if label in rows}
    timing_backends = sorted({str(row.get('timing_backend')) for row in rows.values() if row.get('timing_backend')})
    return {'tflops': report['summary']['primary_mean'] or 0.0, 'all_correct': report['summary']['all_correct'], 'performance_comparable': report['summary']['performance_comparable'], 'invalid_performance_reason': report['summary']['invalid_performance_reason'], 'timing_backends': timing_backends, 'timing_backend_requested': 'cupti' if use_cupti else 'cuda_event', 'use_cupti': use_cupti, 'split_count_by_shape': split_counts, 'measured_entrypoint': 'loom.examples.weave.knn_build_k20_tail_q1536_9b9f_wfeed_v2:benchmark_k20_tail_q1536_9b9f_wfeed_v2', 'report': report}
