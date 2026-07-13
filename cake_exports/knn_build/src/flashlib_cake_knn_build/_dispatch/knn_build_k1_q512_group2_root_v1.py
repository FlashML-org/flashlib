"""Exact Q512/M512/K1 grouping-search seed (minimum arch: sm_100a).

This additive seed keeps the validated TMA + tcgen05 producer and exact
split-list merge on the contract path.  It changes the Q512 K1 work grouping
from the dispatched four splits to two splits: each cluster consumes four
database tiles rather than two.  Non-target inputs stay on the inherited
Weave-only route.

Benchmark evidence (CUPTI, 2026-06-30): exact BF16 build
``B=1,Q=M=512,D=128,K=1`` took 0.051296 ms versus FlashLib's 0.106945 ms
(2.084860x).  The current measurement is recorded in
``artifacts/weave_evolve_k1_q512_group2_root/k1_q512_group2_cupti.json``.
"""
from __future__ import annotations
from json import loads as _json_loads
from .._dispatch_runtime import _capture_cuTensorMapEncodeTiled, _decode_capture, _import_dispatch_module, _ir_proxy
from typing import Any
from . import knn_build_lowk_f8c3_q512_q1024_v1 as seed
TARGET_SHAPE = 'build_k_sweep_qm512_k1'
Q512_SPLIT_COUNT = 2
ROUTE_PREFIX = 'loom.examples.weave.knn_build_k1_q512_group2_root_v1'
ir = _decode_capture(_json_loads('{"__ir__": "knn_build_evolve_7bfc_split_cg2_stage1_batch8_cond4_vmin_maxtree", "arg_keys": ["tmap_query", "tmap_database", "query_sq", "database_sq", "partial_dists", "partial_indices", "B", "Q", "M", "K", "num_q_tile_pairs", "db_tiles_per_split", "split_count", "total_work"], "cluster_dims": [2, 1, 1], "computed_smem_bytes": 50432, "constants": [["BLOCK_Q", 128], ["BLOCK_M", 64], ["FEAT_D", 128], ["TOP_K_MAX", 10]], "cta_group": 1, "threads": 192}'))

def _eligible(inputs: dict[str, Any]) -> bool:
    return str(inputs.get('label', TARGET_SHAPE)) == TARGET_SHAPE and seed._is_bf16_build(inputs, q=512, k=1)

def route_for_contract_inputs(inputs: dict[str, Any]) -> str:
    if _eligible(inputs):
        return ''.join([format(ROUTE_PREFIX, ''), ':q512_k1_s', format(Q512_SPLIT_COUNT, '')])
    return seed.route_for_contract_inputs(inputs)

def launch_from_contract_inputs(inputs: dict[str, Any]) -> None:
    if _eligible(inputs):
        seed._launch_q512_lowk_split(inputs, split_count=Q512_SPLIT_COUNT)
        return
    seed.launch_from_contract_inputs(inputs)

def candidate(inputs: dict[str, Any]) -> None:
    launch_from_contract_inputs(inputs)

def evaluate_contract(*, shapes=None, correctness: bool=True, benchmark: bool=True) -> dict[str, Any]:
    from .._dispatch_runtime import evaluate
    return evaluate(candidate, shapes=shapes, correctness=correctness, benchmark=benchmark)

def _target_shapes() -> list[dict[str, Any]]:
    from .._dispatch_runtime import CANONICAL_SHAPES
    return [shape for shape in CANONICAL_SHAPES if shape['label'] == TARGET_SHAPE]

def benchmark_candidate(*, use_cupti: bool=True) -> dict[str, Any]:
    from .. import _dispatch_runtime as eval_mod
    previous = eval_mod.CONTRACT.bench.get('use_cupti', True)
    eval_mod.CONTRACT.bench['use_cupti'] = bool(use_cupti)
    try:
        return evaluate_contract(shapes=_target_shapes(), correctness=True, benchmark=True)
    finally:
        eval_mod.CONTRACT.bench['use_cupti'] = previous
