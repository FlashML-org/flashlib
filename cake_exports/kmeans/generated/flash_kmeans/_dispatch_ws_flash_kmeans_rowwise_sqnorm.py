"""Generated workspace-shape table for the dispatch-family .so loader."""

from __future__ import annotations

WORKSPACE_KEYS = ()
WORKSPACE_DTYPES = {}
SCALAR_NAMES = ('ROWS', 'D')
ROUTE_IDS = ('rowwise_sqnorm_v1',)
ROUTE_LAUNCH_COUNTS = (1,)
ROUTE_INPUT_TENSORS = (('out_sq', 'src'),)

def workspace_shapes(route_index: int, ROWS, D):
    return {}
