"""Exact D128 Q128/M100000 K32 split-group tile-search seed.

Minimum target architecture: sm_100a.  This additive candidate keeps the
validated TMA/tcgen05 ``128x64x128`` producer and fused K32 merge on the
contract-visible path, while exposing the producer split grouping for the
exact BF16 non-build M100000 bucket.  A true 128-column producer is a distinct
static Weave layout, not a runtime parameter of the inherited 64-column IR.
"""
from __future__ import annotations
from json import loads as _json_loads
from .._dispatch_runtime import _capture_cuTensorMapEncodeTiled, _decode_capture, _import_dispatch_module, _ir_proxy
import json
from pathlib import Path
from typing import Any, Callable
from .. import _dispatch_runtime as eval_mod
from . import knn_build_rag_stream_k32_q128m100000_ad64_v1 as parent
MODULE = 'loom.examples.weave.knn_build_rag_stream_k32_q128m100000_tile_937e_v1'
TARGET_SHAPE = parent.Q128_M100000_K32_SHAPE
TARGET_SHAPES = (TARGET_SHAPE,)
TOPOLOGY_CANDIDATES = ((64, 8), (72, 8), (80, 8))
SPLIT_COUNT, GROUP_COUNT = TOPOLOGY_CANDIDATES[1]
SEED_ID = 'rag_stream_k32_q128_m100000_tile_937e_v1_s72g8'
ROUTE_ENTRYPOINT = ''.join([format(MODULE, ''), ':launch_from_contract_inputs'])
BENCHMARK_ENTRYPOINT = ''.join([format(MODULE, ''), ':benchmark_knn_build_rag_stream_k32_q128m100000_tile_937e_v1'])
ir = _decode_capture(_json_loads('{"__ir__": "knn_build_rag_frontier_4fbf_stage1_k32_sort4earlystop_tailinf", "arg_keys": ["tmap_query", "tmap_database", "query_sq", "database_sq", "partial_dists", "partial_indices", "B", "Q", "M", "K", "num_q_tile_pairs", "db_tiles_per_split", "split_count", "total_work"], "cluster_dims": [2, 1, 1], "computed_smem_bytes": 50432, "constants": [["BLOCK_Q", 128], ["BLOCK_M", 64], ["FEAT_D", 128], ["TOP_K_MAX", 32]], "cta_group": 1, "threads": 192}'))

def _eligible(inputs: dict[str, Any]) -> bool:
    return parent._eligible_q128_m100000(inputs)

def _select_contract_shapes(labels: tuple[str, ...]) -> list[dict[str, Any]]:
    return parent._select_contract_shapes(labels)

def _validate_topology(split_count: int, group_count: int) -> None:
    if (split_count, group_count) not in TOPOLOGY_CANDIDATES:
        raise ValueError(''.join(['unsupported tile-search topology ', format(repr((split_count, group_count)), '')]))

def route_for_contract_inputs(inputs: dict[str, Any], *, split_count: int=SPLIT_COUNT, group_count: int=GROUP_COUNT, force_fallback: bool=False) -> str:
    _validate_topology(split_count, group_count)
    if not force_fallback and _eligible(inputs):
        return ''.join([format(SEED_ID, ''), ':s', format(split_count, ''), '_g', format(group_count, ''), '_tcgen05_128x64x128'])
    return parent.route_for_contract_inputs(inputs, force_fallback=force_fallback)

def launch_from_contract_inputs(inputs: dict[str, Any], *, split_count: int=SPLIT_COUNT, group_count: int=GROUP_COUNT, force_fallback: bool=False) -> None:
    _validate_topology(split_count, group_count)
    if not force_fallback and _eligible(inputs):
        parent._launch_q128_m100000_s72g8(inputs, split_count=split_count, group_count=group_count)
        return
    parent.launch_from_contract_inputs(inputs, force_fallback=force_fallback)

def candidate_with_topology(split_count: int, group_count: int) -> Callable[[dict[str, Any]], None]:
    _validate_topology(split_count, group_count)

    def _candidate(inputs: dict[str, Any]) -> None:
        launch_from_contract_inputs(inputs, split_count=split_count, group_count=group_count)
    return _candidate

def candidate(inputs: dict[str, Any]) -> None:
    launch_from_contract_inputs(inputs)

def evaluate_contract(*, shapes=None, correctness: bool=True, benchmark: bool=True, kernel_fn: Callable[[dict[str, Any]], Any]=candidate) -> dict[str, Any]:
    return eval_mod.evaluate(kernel_fn, shapes=shapes, correctness=correctness, benchmark=benchmark)

def benchmark_knn_build_rag_stream_k32_q128m100000_tile_937e_v1(*, use_cupti: bool=True, split_count: int=SPLIT_COUNT, group_count: int=GROUP_COUNT) -> dict[str, Any]:
    _validate_topology(split_count, group_count)
    old = eval_mod.CONTRACT.bench.get('use_cupti', True)
    eval_mod.CONTRACT.bench['use_cupti'] = bool(use_cupti)
    try:
        report = evaluate_contract(shapes=_select_contract_shapes(TARGET_SHAPES), kernel_fn=candidate_with_topology(split_count, group_count))
    finally:
        eval_mod.CONTRACT.bench['use_cupti'] = old
    return {'candidate_id': ''.join([format(SEED_ID, ''), '_s', format(split_count, ''), '_g', format(group_count, '')]), 'candidate_entrypoint': ROUTE_ENTRYPOINT, 'measured_entrypoint': BENCHMARK_ENTRYPOINT, 'measured_shape_labels': list(TARGET_SHAPES), 'producer_topology': {'mma_tile': [128, 64, 128], 'split_count': split_count, 'db_tiles_per_split': 25 if split_count == 64 else 22 if split_count == 72 else 20}, 'merge_topology': {'kind': '4fbf fused cooperative K32 merge', 'group_count': group_count}, 'summary': report['summary'], 'performance': report['performance'], 'correctness': report['correctness'], 'rank_objective': report['rank_objective'], 'report': report, 'timing_backend_requested': 'cupti' if use_cupti else 'cuda_event'}

def write_benchmark_artifact(path: str | Path, *, use_cupti: bool=True, split_count: int=SPLIT_COUNT, group_count: int=GROUP_COUNT) -> dict[str, Any]:
    payload = benchmark_knn_build_rag_stream_k32_q128m100000_tile_937e_v1(use_cupti=use_cupti, split_count=split_count, group_count=group_count)
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2, sort_keys=True) + '\n')
    return payload
