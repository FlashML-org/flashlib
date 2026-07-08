"""D320/K10 BLOCK_K tile search seed for the exact rectangular-search bucket.

Minimum target architecture: sm_100a.  The selected contract-visible path is
Weave-only: a tcgen05 producer, split-local K10 selection, then the existing
Weave split merge writes distances and indices.  ``BLOCK_K=320`` uses the
exact K128+K128+K64 producer; ``BLOCK_K=384`` is the otherwise-identical
Weave padded-D384 producer.  This module is additive and only accepts the
assigned BF16 D320 rectangular-search shape.
"""
from __future__ import annotations
from json import loads as _json_loads
from .._dispatch_runtime import _capture_cuTensorMapEncodeTiled, _decode_capture, _import_dispatch_module, _ir_proxy
import argparse
import json
import os
from typing import Any, Callable
from .. import _dispatch_runtime as eval_mod
from . import knn_build_non128_frontier_8227_d320tail_v1 as exact_d320
from . import knn_build_non128_frontier_8227_wide_m64_v1 as padded_d384
MODULE = 'loom.examples.weave.knn_build_d320_blockk_b21e_v1'
ROUTE_PREFIX = 'knn_build_d320_blockk_b21e_v1'
TARGET_SHAPE = 'search_rect_highd_b1_q512_m12000_d320_k10'
TARGET_SHAPES = (TARGET_SHAPE,)
BLOCK_K_CHOICES = (320, 384)
DEFAULT_BLOCK_K = _decode_capture(_json_loads('384'))

def _block_k() -> int:
    block_k = int(os.environ.get('LOOM_KNN_D320_BLOCKK_B21E', str(DEFAULT_BLOCK_K)))
    if block_k not in BLOCK_K_CHOICES:
        raise ValueError(''.join(['BLOCK_K must be one of ', format(BLOCK_K_CHOICES, ''), ', got ', format(block_k, '')]))
    return block_k

def _is_target(inputs: dict[str, Any]) -> bool:
    query = inputs.get('query')
    database = inputs.get('database')
    return str(inputs.get('label', '')) == TARGET_SHAPE and str(getattr(query, 'dtype', '')).endswith('bfloat16') and str(getattr(database, 'dtype', '')).endswith('bfloat16') and (int(inputs.get('B', -1)) == 1) and (int(inputs.get('Q', -1)) == 512) and (int(inputs.get('M', -1)) == 12000) and (int(inputs.get('D', -1)) == 320) and (int(inputs.get('K', -1)) == 10) and (not bool(inputs.get('build', True)))

def _verify_export_ir() -> Any:
    return exact_d320.stage1_d320tail_ir if _block_k() == 320 else padded_d384.stage1_d384_ir
ir = _decode_capture(_json_loads('{"__ir__": "knn_build_non128_frontier_8199_d384_stage1", "arg_keys": ["tmap_query", "tmap_database", "query_sq", "database_sq", "partial_dists", "partial_indices", "B", "Q", "M", "K", "num_q_tiles", "db_tiles_per_split", "split_count", "total_work"], "cluster_dims": [1, 1, 1], "computed_smem_bytes": 148736, "constants": [["BLOCK_Q", 128], ["BLOCK_M", 64], ["TOP_K_MAX", 10]], "cta_group": 1, "threads": 192}'))

def route_for_contract_inputs(inputs: dict[str, Any], *, force_fallback: bool=False) -> str:
    if force_fallback or not _is_target(inputs):
        return exact_d320.route_for_contract_inputs(inputs, force_fallback=force_fallback)
    producer = 'exact_d320_k128_k128_k64' if _block_k() == 320 else 'padded_d384_k128_x3'
    return ''.join([format(ROUTE_PREFIX, ''), ':', format(TARGET_SHAPE, ''), ':block_k', format(_block_k(), ''), ':', format(producer, '')])

def launch_from_contract_inputs(inputs: dict[str, Any], *, force_fallback: bool=False) -> None:
    if force_fallback or not _is_target(inputs):
        exact_d320.launch_from_contract_inputs(inputs, force_fallback=force_fallback)
    elif _block_k() == 320:
        exact_d320.launch_from_contract_inputs(inputs)
    else:
        padded_d384.launch_from_contract_inputs(inputs)

def candidate(inputs: dict[str, Any]) -> None:
    launch_from_contract_inputs(inputs)

def evaluate_contract(*, shapes=None, correctness: bool=True, benchmark: bool=True, kernel_fn: Callable[[dict[str, Any]], Any]=candidate) -> dict[str, Any]:
    return eval_mod.evaluate(kernel_fn, shapes=shapes, correctness=correctness, benchmark=benchmark)

def _select_contract_shapes(shape_labels=TARGET_SHAPES) -> list[dict[str, Any]]:
    wanted = set(shape_labels)
    return [shape for shape in eval_mod.CANONICAL_SHAPES if str(shape['label']) in wanted]
