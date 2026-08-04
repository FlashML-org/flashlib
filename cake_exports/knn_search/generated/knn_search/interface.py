"""Vendored by tools/export-generated-programs (runnable adapter).

Provenance: loom @ ff502f39df09ffdb317efc57ebdac3a668bb3aa4. k-NN search (top-K nearest neighbours); dispatches through the compiled family .so C-ABI.
Runtime-free: this module imports no ``loom`` package.
"""

from __future__ import annotations

from typing import Any

from . import _dispatch_families as _fam

#: Composite scalars shared by every sub-family, in selector order.
SCALAR_NAMES = ('B', 'Q', 'M', 'D', 'K', 'self_search')
#: Family run-ABI tensor order (inputs, then any norms, then outputs).
TENSOR_KEYS = ('queries', 'database', 'out_distances', 'out_indices')
N_SCALARS = len(SCALAR_NAMES)
#: The sub-family names in first-match cascade order.
FAMILY_NAMES = _fam.FAMILY_NAMES


def _split(args: tuple[Any, ...]) -> tuple[tuple[Any, ...], tuple[int, ...]]:
    if len(args) < N_SCALARS:
        raise ValueError(f"expected the {len(TENSOR_KEYS)} tensors then {N_SCALARS} scalars, got {len(args)} args")
    cut = len(args) - N_SCALARS
    return args[:cut], tuple(int(s) for s in args[cut:])


def resolve(*scalars: int, arch: str | None = None) -> tuple[int, int]:
    """First covering ``(family_index, route_index)``, or ``(-1, -1)``."""
    return _fam.resolve(*(int(s) for s in scalars), arch=arch)


def covered(*scalars: int, arch: str | None = None) -> bool:
    return resolve(*scalars, arch=arch)[0] >= 0


def route_family(*scalars: int, arch: str | None = None) -> str:
    family_index, _ = resolve(*scalars, arch=arch)
    if family_index < 0:
        raise ValueError(f"no sub-family covers scalars {scalars!r}")
    return str(FAMILY_NAMES[family_index])


def route_id(*scalars: int, arch: str | None = None) -> str:
    family_index, route_index = resolve(*scalars, arch=arch)
    if family_index < 0:
        raise ValueError(f"no sub-family covers scalars {scalars!r}")
    return _fam.route_id(family_index, route_index)


def allocate_workspaces(*scalars: int, device: Any, arch: str | None = None) -> list[Any]:
    family_index, _ = resolve(*scalars, arch=arch)
    if family_index < 0:
        raise ValueError(f"no sub-family covers scalars {scalars!r}")
    return _fam.allocate_workspaces(family_index, *(int(s) for s in scalars), device=device, arch=arch)


def run(*args: Any, arch: str | None = None) -> None:
    """Route through the sub-family cascade + launch: family tensors then scalars."""
    tensors, scalars = _split(args)
    if not tensors:
        raise ValueError("run requires at least one tensor argument")
    family_index, _ = _fam.resolve(*scalars, arch=arch)
    if family_index < 0:
        raise ValueError(f"no sub-family covers scalars {scalars!r}")
    workspaces = _fam.allocate_workspaces(family_index, *scalars, device=tensors[0].device, arch=arch)
    _fam.run(family_index, *tensors, *workspaces, *scalars, arch=arch)
