"""Generated outer first-match cascade over the per-family dispatch-family .so loaders.

Runtime-free (no ``loom`` import): the exported package interface imports this
module and drives the sub-families through their compiled ``.so`` C-ABI.  The
family order is the plan's cascade order; ``resolve`` returns the first family
whose ``route_of`` covers the shape, mirroring the in-repo outer dispatcher's
``select_index`` first-match.
"""

from __future__ import annotations

from typing import Any

from . import _dispatch_knn_search_d256_q64_7ad3 as _f0
from . import _dispatch_knn_search_portfolio_09a4 as _f1
from . import _dispatch_knn_search_registered_b653 as _f2
from . import _dispatch_knn_search_restored_full133 as _f3
from . import _dispatch_knn_search_compat0701 as _f4

#: The sub-family loaders in first-match cascade order.
FAMILY_LOADERS = (_f0, _f1, _f2, _f3, _f4,)
#: Parallel ``DispatchFamily.name`` per loader (the ``route_family`` the interface reports).
FAMILY_NAMES = ('knn_search_d256_q64_7ad3', 'knn_search_portfolio_09a4', 'knn_search_registered_b653', 'knn_search_restored_full133', 'knn_search_compat0701')
#: The common scalar key order every sub-family declares (interface forwards these).
SCALAR_NAMES = ('B', 'Q', 'M', 'D', 'K', 'self_search')


def resolve(*scalars: int, arch: str | None = None) -> tuple[int, int]:
    """First family (cascade order) whose ``.so`` covers the shape.

    Returns ``(family_index, route_index)``, or ``(-1, -1)`` when no sub-family
    owns the shape (the outer no-match the interface maps to a rejection).
    """
    for family_index, loader in enumerate(FAMILY_LOADERS):
        route_index = loader.route_of(*scalars, arch=arch)
        if route_index >= 0:
            return family_index, route_index
    return -1, -1


def route_id(family_index: int, route_index: int) -> str:
    return str(FAMILY_LOADERS[family_index].ROUTE_IDS[route_index])


def launch_count(family_index: int, route_index: int) -> int:
    return int(FAMILY_LOADERS[family_index].ROUTE_LAUNCH_COUNTS[route_index])


def input_tensors(family_index: int, route_index: int) -> tuple[str, ...]:
    return tuple(FAMILY_LOADERS[family_index].ROUTE_INPUT_TENSORS[route_index])


def allocate_workspaces(family_index: int, *scalars: int, device: Any, arch: str | None = None) -> list[Any]:
    """Allocate the selected family's route workspaces (family slot order)."""
    return FAMILY_LOADERS[family_index].allocate_workspaces(*scalars, device=device, arch=arch)


def run(family_index: int, *args: Any, arch: str | None = None) -> None:
    """Dispatch through the selected family's ``.so``: tensors, workspaces, scalars."""
    FAMILY_LOADERS[family_index].run(*args, arch=arch)
