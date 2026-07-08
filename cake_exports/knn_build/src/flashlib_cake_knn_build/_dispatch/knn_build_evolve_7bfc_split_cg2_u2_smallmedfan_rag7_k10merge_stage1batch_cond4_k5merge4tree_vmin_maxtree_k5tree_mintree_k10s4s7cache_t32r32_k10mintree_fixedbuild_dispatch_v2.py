"""kNN build/search K=10 fixed-build split-dispatch v2 candidate.

Minimum target architecture: sm_100a. This variant inherits the validated
tcgen05/TMA stage-1 kernel, sorted split-local K=10 output, and cached K=10
merge kernels from the selected parent. It changes only host dispatch for
fixed-build ``Q == M`` K=10 shapes: underfilled 1024/2048-style rows use the
7-split cached merge path to expose more independent CTA-group work, midlarge
rows whose CTA-grouped query tile count is 38..42 use the 7-split path, and
other fixed-build rows use the 4-split cached path.
"""
from __future__ import annotations
from json import loads as _json_loads
from .._dispatch_runtime import _capture_cuTensorMapEncodeTiled, _decode_capture, _import_dispatch_module, _ir_proxy
from typing import Any
from . import knn_build_evolve_7bfc_split_cg2_u2_smallmedfan_rag7_k10merge_stage1batch_cond4_k5merge4tree_vmin_maxtree_k5tree_mintree_k10s4s7cache_t32r32_k10mintree_v1 as parent
BLOCK_Q = parent.BLOCK_Q
BLOCK_M = parent.BLOCK_M
FEAT_D = parent.FEAT_D
TOP_K_MAX = parent.TOP_K_MAX
STAGE1_THREADS = parent.STAGE1_THREADS
MERGE_THREADS = parent.MERGE_THREADS
GRID_DIM_DEFAULT = parent.GRID_DIM_DEFAULT
CTA_GROUP = parent.CTA_GROUP
SMALL_SHAPE_MAX = parent.SMALL_SHAPE_MAX
MEDIUM_SPLITS = parent.MEDIUM_SPLITS
RAG_SPLITS = parent.RAG_SPLITS
stage1_ir = _decode_capture(_json_loads('{"__ir__": "knn_build_evolve_7bfc_split_cg2_stage1_batch8_cond4_vmin_maxtree", "arg_keys": ["tmap_query", "tmap_database", "query_sq", "database_sq", "partial_dists", "partial_indices", "B", "Q", "M", "K", "num_q_tile_pairs", "db_tiles_per_split", "split_count", "total_work"], "cluster_dims": [2, 1, 1], "computed_smem_bytes": 50432, "constants": [["BLOCK_Q", 128], ["BLOCK_M", 64], ["FEAT_D", 128], ["TOP_K_MAX", 10]], "cta_group": 1, "threads": 192}'))
ir = _decode_capture(_json_loads('{"__ir__": "knn_build_evolve_7bfc_split_cg2_stage1_batch8_cond4_vmin_maxtree", "arg_keys": ["tmap_query", "tmap_database", "query_sq", "database_sq", "partial_dists", "partial_indices", "B", "Q", "M", "K", "num_q_tile_pairs", "db_tiles_per_split", "split_count", "total_work"], "cluster_dims": [2, 1, 1], "computed_smem_bytes": 50432, "constants": [["BLOCK_Q", 128], ["BLOCK_M", 64], ["FEAT_D", 128], ["TOP_K_MAX", 10]], "cta_group": 1, "threads": 192}'))

def _eligible_fixed_build_k10(inputs: dict[str, Any]) -> bool:
    return bool(inputs.get('build', False)) and str(inputs['query'].dtype) == 'torch.bfloat16' and (str(inputs['database'].dtype) == 'torch.bfloat16') and (int(inputs['D']) == FEAT_D) and (int(inputs['K']) == TOP_K_MAX) and (int(inputs['Q']) == int(inputs['M']))

def _fixed_build_k10_split_count(inputs: dict[str, Any]) -> int | None:
    if not _eligible_fixed_build_k10(inputs):
        return None
    n_query = int(inputs['Q'])
    if n_query <= SMALL_SHAPE_MAX:
        return MEDIUM_SPLITS
    if n_query < 4096:
        return RAG_SPLITS
    num_q_tiles = (n_query + BLOCK_Q - 1) // BLOCK_Q
    num_q_tile_pairs = (num_q_tiles + CTA_GROUP - 1) // CTA_GROUP
    if 38 <= num_q_tile_pairs <= 42:
        return RAG_SPLITS
    return MEDIUM_SPLITS

def _launch_fixed_build_k10(inputs: dict[str, Any], *, split_count: int) -> None:
    if split_count == RAG_SPLITS:
        parent._launch_k10_cached_path(inputs, split_count=split_count, merge_threads=parent.parent_cached.RAG_MERGE_THREADS, merge_kernel=parent.parent_cached._compiled_merge_k10_s7_cache(), merge_ir=parent.parent_cached.merge_k10_s7_cache_ir)
        return
    parent._launch_k10_cached_path(inputs, split_count=MEDIUM_SPLITS, merge_threads=parent.parent_cached64.MEDIUM_MERGE_THREADS, merge_kernel=parent.parent_cached64._compiled_merge_k10_s4_cache(), merge_ir=parent.parent_cached64.merge_k10_s4_cache_ir)

def launch_from_contract_inputs(inputs: dict[str, Any]) -> None:
    split_count = _fixed_build_k10_split_count(inputs)
    if split_count is not None:
        _launch_fixed_build_k10(inputs, split_count=split_count)
        return
    parent.launch_from_contract_inputs(inputs)

def candidate(inputs: dict[str, Any]):
    launch_from_contract_inputs(inputs)
    return None

def evaluate_contract(*, shapes=None, correctness: bool=True, benchmark: bool=True) -> dict[str, Any]:
    from .._dispatch_runtime import evaluate
    return evaluate(candidate, shapes=shapes, correctness=correctness, benchmark=benchmark)

def _select_contract_shapes(shape_labels):
    from .._dispatch_runtime import CANONICAL_SHAPES
    if shape_labels is None:
        return list(CANONICAL_SHAPES)
    wanted = {str(label) for label in shape_labels}
    selected = [shape for shape in CANONICAL_SHAPES if shape['label'] in wanted]
    missing = wanted - {shape['label'] for shape in selected}
    if missing:
        raise ValueError(''.join(['unknown kNN build contract shape(s): ', format(sorted(missing), '')]))
    return selected

def compile_and_launch_knn_build(*, shape_labels=('flashml_correctness_b1_q256_m256_d128_k5',), benchmark: bool=False) -> dict[str, Any]:
    """e2e-test entrypoint: run real contract correctness for selected shapes."""
    report = evaluate_contract(shapes=_select_contract_shapes(shape_labels), correctness=True, benchmark=benchmark)
    passed = bool(report.get('summary', {}).get('all_correct', False))
    report['passed'] = passed
    report['all_pass'] = passed
    return report
