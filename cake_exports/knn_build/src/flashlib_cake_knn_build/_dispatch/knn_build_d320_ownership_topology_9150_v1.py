"""D320 split-ownership topology candidate for the exact kNN search bucket.

Minimum target architecture: sm_100a.  This module retains the exact D320
TMA/tcgen05 producer (K128 + K128 + K64) and the existing Weave split merge,
but assigns four database tiles to each of 48 split-local K10 producers.  The
192 producer work items (four Q128 tiles times 48 splits) raise the producer
grid above the B200's 148 SMs while preserving the contract output ABI.
"""
from __future__ import annotations
from json import loads as _json_loads
from .._dispatch_runtime import _capture_cuTensorMapEncodeTiled, _decode_capture, _import_dispatch_module, _ir_proxy
import argparse
import json
import os
from contextlib import contextmanager
from typing import Any, Callable, Iterator
from .. import _dispatch_runtime as eval_mod
from . import knn_build_non128_frontier_8227_d320tail_v1 as exact_d320
MODULE = 'loom.examples.weave.knn_build_d320_ownership_topology_9150_v1'
ROUTE_PREFIX = 'knn_build_d320_ownership_topology_9150_v1'
TARGET_SHAPE = 'search_rect_highd_b1_q512_m12000_d320_k10'
TARGET_SHAPES = (TARGET_SHAPE,)
SPLIT_COUNT = 48
DB_TILES_PER_SPLIT = 4
TOTAL_WORK = 192
_SPLIT_ENV = 'LOOM_KNN_NON128_FRONTIER_8227_D320TAIL_SPLITS_SEARCH_RECT_HIGHD_B1_Q512_M12000_D320_K10'
ir = _decode_capture(_json_loads('{"__ir__": "knn_build_non128_frontier_8227_d320tail_stage1", "arg_keys": ["tmap_query", "tmap_database", "query_sq", "database_sq", "partial_dists", "partial_indices", "B", "Q", "M", "K", "num_q_tiles", "db_tiles_per_split", "split_count", "total_work"], "cluster_dims": [1, 1, 1], "computed_smem_bytes": 124160, "constants": [["BLOCK_Q", 128], ["BLOCK_M", 64], ["TOP_K_MAX", 10]], "cta_group": 1, "threads": 192}'))

def _is_target(inputs: dict[str, Any]) -> bool:
    query = inputs.get('query')
    database = inputs.get('database')
    return str(inputs.get('label', '')) == TARGET_SHAPE and str(getattr(query, 'dtype', '')) == 'torch.bfloat16' and (str(getattr(database, 'dtype', '')) == 'torch.bfloat16') and (int(inputs.get('B', -1)) == 1) and (int(inputs.get('Q', -1)) == 512) and (int(inputs.get('M', -1)) == 12000) and (int(inputs.get('D', -1)) == 320) and (int(inputs.get('K', -1)) == 10) and (not bool(inputs.get('build', True)))

@contextmanager
def _split_ownership() -> Iterator[None]:
    previous = os.environ.get(_SPLIT_ENV)
    os.environ[_SPLIT_ENV] = str(SPLIT_COUNT)
    try:
        yield
    finally:
        if previous is None:
            os.environ.pop(_SPLIT_ENV, None)
        else:
            os.environ[_SPLIT_ENV] = previous

def route_for_contract_inputs(inputs: dict[str, Any], *, force_fallback: bool=False) -> str:
    if force_fallback or not _is_target(inputs):
        return exact_d320.route_for_contract_inputs(inputs, force_fallback=force_fallback)
    return ''.join([format(ROUTE_PREFIX, ''), ':', format(TARGET_SHAPE, ''), ':exact_d320_k128_k128_k64:splits', format(SPLIT_COUNT, ''), ':dbtiles', format(DB_TILES_PER_SPLIT, ''), ':work', format(TOTAL_WORK, '')])

def launch_from_contract_inputs(inputs: dict[str, Any], *, force_fallback: bool=False) -> None:
    if force_fallback or not _is_target(inputs):
        exact_d320.launch_from_contract_inputs(inputs, force_fallback=force_fallback)
        return
    with _split_ownership():
        exact_d320.launch_from_contract_inputs(inputs)

def candidate(inputs: dict[str, Any]) -> None:
    launch_from_contract_inputs(inputs)

def evaluate_contract(*, shapes=None, correctness: bool=True, benchmark: bool=True, kernel_fn: Callable[[dict[str, Any]], Any]=candidate) -> dict[str, Any]:
    return eval_mod.evaluate(kernel_fn, shapes=shapes, correctness=correctness, benchmark=benchmark)

def _select_contract_shapes(shape_labels=TARGET_SHAPES) -> list[dict[str, Any]]:
    wanted = set(shape_labels)
    return [shape for shape in eval_mod.CANONICAL_SHAPES if str(shape['label']) in wanted]
