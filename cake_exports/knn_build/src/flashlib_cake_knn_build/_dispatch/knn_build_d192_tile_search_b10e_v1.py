"""Exact D192 build tile-grouping candidate for the Q2048/M2048/K10 bucket.

Minimum target architecture: sm_100a.  This additive candidate keeps the
validated D256-wide TMA/tcgen05 producer and exact Weave split merge from the
non-D128 frontier, but exposes its eight-way database grouping as an isolated
exact-shape seed.  The producer writes split-local K10 lists which the merge
consumes to produce the contract distances and indices; no host or reference
work is on the specialized path.
"""
from __future__ import annotations
from json import loads as _json_loads
from .._dispatch_runtime import _capture_cuTensorMapEncodeTiled, _decode_capture, _import_dispatch_module, _ir_proxy
from typing import Any, Callable
from .. import _dispatch_runtime as eval_mod
from . import knn_build_dispatch_v11_common_d_seed_portfolio_a4ec_v1 as a4ec
from . import knn_build_non128_frontier_8199_widecombine_v1 as widecombine
MODULE = 'loom.examples.weave.knn_build_d192_tile_search_b10e_v1'
TARGET_SHAPE = 'build_dim_sweep_b1_q2048_m2048_d192_k10'
TARGET_SHAPES = (TARGET_SHAPE,)
SPLIT_COUNT = 8
ir = _decode_capture(_json_loads('{"__ir__": "knn_build_dim_midk_df2f_d256_split_stage1", "arg_keys": ["tmap_query", "tmap_database", "query_sq", "database_sq", "partial_dists", "partial_indices", "B", "Q", "M", "K", "num_q_tiles", "db_tiles_per_split", "split_count", "total_work"], "cluster_dims": [1, 1, 1], "computed_smem_bytes": 99584, "constants": [["BLOCK_Q", 128], ["BLOCK_M", 64], ["TOP_K_MAX", 10]], "cta_group": 1, "threads": 192}'))

def _eligible(inputs: dict[str, Any]) -> bool:
    query = inputs.get('query')
    dtype = str(query.dtype).replace('torch.', '') if query is not None else str(inputs.get('dtype', ''))
    return dtype == 'bfloat16' and bool(inputs.get('build', False)) and (int(inputs['B']) == 1) and (int(inputs['Q']) == 2048) and (int(inputs['M']) == 2048) and (int(inputs['D']) == 192) and (int(inputs['K']) == 10)

def route_for_contract_inputs(inputs: dict[str, Any], *, force_fallback: bool=False) -> str:
    if not force_fallback and _eligible(inputs):
        return ''.join([format(MODULE, ''), ':exact_d192_wide256_s', format(SPLIT_COUNT, '')])
    return a4ec.route_for_contract_inputs(inputs, force_fallback=force_fallback)

def launch_from_contract_inputs(inputs: dict[str, Any], *, force_fallback: bool=False) -> None:
    if not force_fallback and _eligible(inputs):
        widecombine._launch_wide_stage(inputs, TARGET_SHAPE, feature_dim=256, kernel=widecombine.wide_d256._compiled_d256_stage1(), stage1_ir=widecombine.stage1_d256_ir)
        return
    a4ec.launch_from_contract_inputs(inputs, force_fallback=force_fallback)

def candidate(inputs: dict[str, Any]) -> None:
    launch_from_contract_inputs(inputs)

def evaluate_contract(*, shapes=None, correctness: bool=True, benchmark: bool=True, kernel_fn: Callable[[dict[str, Any]], Any]=candidate) -> dict[str, Any]:
    return eval_mod.evaluate(kernel_fn, shapes=shapes, correctness=correctness, benchmark=benchmark)

def _select_contract_shapes(shape_labels=TARGET_SHAPES):
    wanted = {str(label) for label in shape_labels}
    selected = [shape for shape in eval_mod.CANONICAL_SHAPES if str(shape['label']) in wanted]
    if wanted - {str(shape['label']) for shape in selected}:
        raise ValueError(''.join(['unknown contract shapes: ', format(sorted(wanted), '')]))
    return selected
