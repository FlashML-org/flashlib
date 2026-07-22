"""Generated workspace-shape table for the dispatch-family .so loader."""

from __future__ import annotations

WORKSPACE_KEYS = ('partial_dists', 'partial_indices', 'padded_db')
WORKSPACE_DTYPES = {'partial_dists': 'float32', 'partial_indices': 'int32', 'padded_db': 'bfloat16'}
SCALAR_NAMES = ('B', 'Q', 'M', 'D', 'K', 'dtype', 'build')
ROUTE_IDS = ('knn_build_d192_wide256_s8',)
ROUTE_LAUNCH_COUNTS = (3,)
ROUTE_INPUT_TENSORS = (('database', 'database_sq', 'out_dists', 'out_indices', 'query_sq'),)

def workspace_shapes(route_index: int, B, Q, M, D, K, dtype, build):
    if route_index == 0:
        return {"partial_dists": (8, B, Q, K,), "partial_indices": (8, B, Q, K,), "padded_db": ((B * M), 256,)}
    return {}
