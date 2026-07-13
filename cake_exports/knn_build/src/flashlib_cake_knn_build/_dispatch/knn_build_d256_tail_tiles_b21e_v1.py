"""D256/K32 bounded tail-work search for the exact long-M kNN-build row.

Minimum target architecture: sm_100a.  This additive candidate keeps the
validated tcgen05/TMA D256 producer and warp-row K32 merge on the contract
path, while changing only its split/tail work partition for
``rag_stream_largek_common_d256_b1_q128_m100000_k32``.  The physical MMA tile
remains M64/N64/K128x2. The winning assigned (128, 128, 256) tail partition
uses 64 split work items (25 M64 tiles per item), rather than pretending that
the existing M64/N64 primitive has widened.
"""
from __future__ import annotations
from json import loads as _json_loads
from .._dispatch_runtime import _capture_cuTensorMapEncodeTiled, _decode_capture, _import_dispatch_module, _ir_proxy
import argparse
import json
import os
from typing import Any, Callable
from .. import _dispatch_runtime as eval_mod
from . import knn_build_v12_d256_k32_tail_59fe_v1 as parent
MODULE = 'loom.examples.weave.knn_build_d256_tail_tiles_b21e_v1'
ROUTE_ENTRYPOINT = ''.join([format(MODULE, ''), ':launch_from_contract_inputs'])
BENCHMARK_ENTRYPOINT = ''.join([format(MODULE, ''), ':benchmark_knn_build_d256_tail_tiles_b21e_v1'])
CANDIDATE_ID = 'knn_build_d256_tail_tiles_b21e_v1'
TARGET_SHAPE = 'rag_stream_largek_common_d256_b1_q128_m100000_k32'
TARGET_SHAPES = (TARGET_SHAPE,)
TILE_SHAPE = _decode_capture(_json_loads('{"__tuple__": [128, 128, 256]}'))
SPLIT_COUNT = _decode_capture(_json_loads('64'))
ir = _decode_capture(_json_loads('{"__ir__": "knn_build_v12_d256_k32_tail_59fe_v1_stage1_rowld", "arg_keys": ["tmap_query", "tmap_database", "query_sq", "database_sq", "partial_dists", "partial_indices", "B", "Q", "M", "K", "num_q_tiles", "db_tiles_per_split", "split_count", "total_work"], "cluster_dims": [1, 1, 1], "computed_smem_bytes": 99584, "constants": [["BLOCK_Q", 64], ["BLOCK_M", 64], ["K_TILE", 128], ["FEATURE_CHUNKS", 2], ["TOP_K_MAX", 32]], "cta_group": 1, "threads": 192}'))

def _eligible(inputs: dict[str, Any]) -> bool:
    return parent._target_label_for_inputs(inputs) == TARGET_SHAPE

def route_for_contract_inputs(inputs: dict[str, Any], *, force_fallback: bool=False) -> str:
    if not force_fallback and _eligible(inputs):
        tail_tiles = (100000 // 64 + SPLIT_COUNT - 1) // SPLIT_COUNT
        return ''.join(['d256_tail_tiles_b21e:exact_q128_m100000_d256_k32:s', format(SPLIT_COUNT, ''), ':m64n64k256:tail', format(tail_tiles, '')])
    return parent.route_for_contract_inputs(inputs, force_fallback=force_fallback)

def launch_from_contract_inputs(inputs: dict[str, Any], *, force_fallback: bool=False) -> None:
    if not force_fallback and _eligible(inputs):
        parent._launch_d256_k32_rowld_warpmerge(inputs, TARGET_SHAPE, split_count=SPLIT_COUNT)
        return
    parent.launch_from_contract_inputs(inputs, force_fallback=force_fallback)

def candidate(inputs: dict[str, Any]) -> None:
    launch_from_contract_inputs(inputs)

def evaluate_contract(*, shapes=None, correctness: bool=True, benchmark: bool=True, kernel_fn: Callable[[dict[str, Any]], Any]=candidate) -> dict[str, Any]:
    return eval_mod.evaluate(kernel_fn, shapes=shapes, correctness=correctness, benchmark=benchmark)

def _select_contract_shapes(shape_labels=TARGET_SHAPES):
    wanted = set(shape_labels)
    selected = [shape for shape in eval_mod.CANONICAL_SHAPES if str(shape['label']) in wanted]
    if {str(shape['label']) for shape in selected} != wanted:
        raise ValueError(''.join(['unknown shape labels: ', format(sorted(wanted), '')]))
    return selected

def _run(*, use_cupti: bool, shape_labels=TARGET_SHAPES) -> dict[str, Any]:
    previous = eval_mod.CONTRACT.bench.get('use_cupti', True)
    eval_mod.CONTRACT.bench['use_cupti'] = bool(use_cupti)
    try:
        return evaluate_contract(shapes=_select_contract_shapes(shape_labels), correctness=True, benchmark=True)
    finally:
        eval_mod.CONTRACT.bench['use_cupti'] = previous

def benchmark_knn_build_d256_tail_tiles_b21e_v1(*, use_cupti: bool=True) -> dict[str, Any]:
    report = _run(use_cupti=use_cupti)
    return {'contract': report['contract'], 'contract_version': report['contract_version'], 'candidate_id': CANDIDATE_ID, 'candidate_entrypoint': ROUTE_ENTRYPOINT, 'measured_entrypoint': BENCHMARK_ENTRYPOINT, 'tile_shape_candidate': list(TILE_SHAPE), 'physical_mma_tile': [64, 64, 256], 'split_count': SPLIT_COUNT, 'db_tiles_per_split': (100000 // 64 + SPLIT_COUNT - 1) // SPLIT_COUNT, 'summary': report['summary'], 'performance': report['performance'], 'correctness': report['correctness'], 'rank_objective': report.get('rank_objective'), 'per_shape': report.get('per_shape', {}), 'report': report}
