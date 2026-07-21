"""Vendored by tools/export-generated-programs (runnable adapter).

Provenance: loom @ unknown. Flash K-Means assignment; dispatches through the compiled family .so C-ABI.
Runtime-free: this module imports no ``loom`` package.
"""

from __future__ import annotations

from typing import Any

from . import _dispatch_family as _loader

#: Signature scalars, in the order the compiled ``.so`` selector expects them.
SCALAR_NAMES = ('B', 'N', 'D', 'K')
#: Family run-ABI tensor order (inputs, then any norms, then outputs).
TENSOR_KEYS = ('x', 'centroids', 'x_sq', 'c_sq', 'out')
N_SCALARS = len(SCALAR_NAMES)


def _split(args: tuple[Any, ...]) -> tuple[tuple[Any, ...], tuple[int, ...]]:
    if len(args) < N_SCALARS:
        raise ValueError(f"expected the {len(TENSOR_KEYS)} tensors then {N_SCALARS} scalars, got {len(args)} args")
    cut = len(args) - N_SCALARS
    return args[:cut], tuple(int(s) for s in args[cut:])


def route_of(*scalars: int, arch: str | None = None) -> int:
    """Ordered first-match route index for a shape, or -1 (no route)."""
    return int(_loader.route_of(*(int(s) for s in scalars), arch=arch))


def covered(*scalars: int, arch: str | None = None) -> bool:
    return route_of(*scalars, arch=arch) >= 0


def route_id(*scalars: int, arch: str | None = None) -> str:
    index = route_of(*scalars, arch=arch)
    if index < 0:
        raise ValueError(f"no dispatch route for scalars {scalars!r}")
    return str(_loader.ROUTE_IDS[index])


def allocate_workspaces(*scalars: int, device: Any, arch: str | None = None) -> list[Any]:
    return _loader.allocate_workspaces(*(int(s) for s in scalars), device=device, arch=arch)


def run(*args: Any, arch: str | None = None) -> None:
    """Route + launch: pass the family tensors then the scalars; workspaces are
    sized and allocated here from the same table the ``.so`` validates against."""
    tensors, scalars = _split(args)
    if not tensors:
        raise ValueError("run requires at least one tensor argument")
    workspaces = _loader.allocate_workspaces(*scalars, device=tensors[0].device, arch=arch)
    _loader.run(*tensors, *workspaces, *scalars, arch=arch)
