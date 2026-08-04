"""Generated workspace-shape table for the dispatch-family .so loader."""

from __future__ import annotations

WORKSPACE_KEYS = ('partial_distances', 'partial_indices', 'group_distances', 'group_indices')
WORKSPACE_DTYPES = {'partial_distances': 'float32', 'partial_indices': 'int32', 'group_distances': 'float32', 'group_indices': 'int32'}
SCALAR_NAMES = ('B', 'Q', 'M', 'D', 'K', 'self_search')
ROUTE_IDS = ('residual0707_d256_q64_split152_warp_state_7ad3_tcgen05',)
ROUTE_LAUNCH_COUNTS = (3,)
ROUTE_INPUT_TENSORS = (('database', 'out_distances', 'out_indices', 'queries'),)

def workspace_shapes(route_index: int, B, Q, M, D, K, self_search):
    if route_index == 0:
        return {"partial_distances": (B, (((Q + 64) - 1) // 64), 304, 64, 64,), "partial_indices": (B, (((Q + 64) - 1) // 64), 304, 64, 64,), "group_distances": (B, Q, 4, 64,), "group_indices": (B, Q, 4, 64,)}
    return {}
