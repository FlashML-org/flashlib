"""Checked tcgen05-col4 partial-list capture for RAG Q4096/M20000/D128/K10.

Minimum target architecture: sm_100a.  This diagnostic launches the exact
tcgen05 producer used by the R260 exported route, before its pairlocal72
consumer, and checks the physical ``list = split * 4 + cohort`` ABI against
an exact squared-L2 subproblem.  It is deliberately not a benchmark path.
"""
from __future__ import annotations
from json import loads as _json_loads
from .._dispatch_runtime import _capture_cuTensorMapEncodeTiled, _decode_capture, _import_dispatch_module, _ir_proxy
import math
from collections.abc import Iterable
from typing import Any
from . import knn_search_mma_split_v1 as mma
TARGET_KEY = (1, 4096, 20000, 128, 10)
PARTIAL_LIST_COUNT = 72
PRODUCER_ABI = 'tcgen05_col4_partial_topk'
CONSUMER_ABI = 'q4096_pairlocal_topk_72_lists'
DEFAULT_ROWS = (0, 2048, 4095)
ir = _decode_capture(_json_loads('{"__ir__": "knn_search_mma_split_partial_v1", "arg_keys": ["queries", "database", "partial_distances", "partial_indices", "B", "Q", "M", "split_m", "num_q_tiles", "total_m_tiles", "tiles_per_split"], "cluster_dims": [1, 1, 1], "computed_smem_bytes": 108800, "constants": [["K_MAX_", 10], ["EXPOSE_COL_COHORTS", 0], ["FULL_M_TILES", 0]], "cta_group": 1, "threads": 640}'))

def _is_target(inputs: dict[str, Any]) -> bool:
    return (int(inputs.get('B', 1)), int(inputs['Q']), int(inputs['M']), int(inputs['D']), int(inputs['K'])) == TARGET_KEY and (not bool(inputs.get('self_search', False))) and mma._tcgen05_capable_arch()

def _validate_inputs(inputs: dict[str, Any]) -> None:
    if not _is_target(inputs):
        raise ValueError('checked col4 capture only accepts rag_q4096_m20000_d128_k10 on sm_100a/sm_103a')
    missing = [name for name in ('queries', 'database') if name not in inputs]
    if missing:
        raise ValueError(''.join(['checked col4 capture missing tensor fields: ', format(missing, '')]))

def _producer_kernel() -> Any:
    if not mma._KNN_SEARCH_KERNELS:
        mma._KNN_SEARCH_KERNELS.update(mma._compile_kernels())
    return mma._KNN_SEARCH_KERNELS['partial_col4']

def capture_col4_partials(inputs: dict[str, Any]) -> tuple[Any, Any]:
    """Run only the contract producer and return its [Q_tiles,72,128,K] lists."""
    import torch
    _validate_inputs(inputs)
    bsz, q_rows, m_rows = (int(inputs['B']), int(inputs['Q']), int(inputs['M']))
    split_m = mma._select_split_m(q_rows, m_rows)
    num_q_tiles = math.ceil(q_rows / mma.BLOCK_Q)
    total_m_tiles = math.ceil(m_rows / mma.BLOCK_M)
    if split_m != 18 or split_m * mma.MMA_POST_MMA_COL_COHORTS != PARTIAL_LIST_COUNT:
        raise RuntimeError(''.join(['unexpected checked producer ABI: split_m=', format(split_m, '')]))
    partial_dist = torch.empty((bsz, num_q_tiles, PARTIAL_LIST_COUNT, mma.BLOCK_Q, mma.K_MAX), device=inputs['queries'].device, dtype=torch.float32)
    partial_idx = torch.empty_like(partial_dist, dtype=torch.int32)
    _producer_kernel().launch(grid=(bsz * num_q_tiles * split_m, 1, 1), block=(mma.THREADS, 1, 1), args=[inputs['queries'], inputs['database'], partial_dist, partial_idx, bsz, q_rows, m_rows, split_m, num_q_tiles, total_m_tiles, math.ceil(total_m_tiles / split_m)], shared_mem=mma.MMA_SMEM_BYTES)
    return (partial_dist, partial_idx)

def _reference_list(query: Any, database: Any, split_id: int, cohort: int) -> tuple[Any, Any]:
    import torch
    total_tiles = math.ceil(database.shape[0] / mma.BLOCK_M)
    tile_begin = split_id * total_tiles // 18
    tile_end = (split_id + 1) * total_tiles // 18
    rows = torch.cat([torch.arange(tile * mma.BLOCK_M + cohort * 32, tile * mma.BLOCK_M + (cohort + 1) * 32, device=database.device) for tile in range(tile_begin, tile_end)])
    rows = rows[rows < database.shape[0]]
    distances = (database[rows].float() - query.float()).square().sum(dim=1)
    values, positions = torch.topk(distances, k=mma.K_MAX, largest=False, sorted=True)
    return (values, rows[positions].to(torch.int32))

def check_col4_partials(inputs: dict[str, Any], rows: Iterable[int]=DEFAULT_ROWS) -> dict[str, Any]:
    """Compare exact selected-row producer lists; no result feeds a benchmark."""
    import torch
    partial_dist, partial_idx = capture_col4_partials(inputs)
    checked_rows = tuple((int(row) for row in rows))
    failures: list[dict[str, Any]] = []
    for q_row in checked_rows:
        q_tile, q_local = divmod(q_row, mma.BLOCK_Q)
        for split_id in range(18):
            for cohort in range(mma.MMA_POST_MMA_COL_COHORTS):
                list_id = split_id * mma.MMA_POST_MMA_COL_COHORTS + cohort
                expected_d, expected_i = _reference_list(inputs['queries'][0, q_row], inputs['database'][0], split_id, cohort)
                actual_d = partial_dist[0, q_tile, list_id, q_local]
                actual_i = partial_idx[0, q_tile, list_id, q_local]
                indices_match = bool(torch.equal(actual_i, expected_i))
                values_match = bool(torch.allclose(actual_d, expected_d, atol=0.01, rtol=0.01))
                if not indices_match or not values_match:
                    failures.append({'query_row': q_row, 'split_id': split_id, 'cohort': cohort, 'list_id': list_id, 'indices_match': indices_match, 'values_match': values_match, 'actual_indices': actual_i.cpu().tolist(), 'expected_indices': expected_i.cpu().tolist(), 'max_abs': float((actual_d - expected_d).abs().max().cpu())})
    return {'passed': not failures, 'producer_abi': PRODUCER_ABI, 'consumer_abi': CONSUMER_ABI, 'partial_list_count': PARTIAL_LIST_COUNT, 'checked_rows': list(checked_rows), 'failure_count': len(failures), 'first_failure': failures[0] if failures else None}
