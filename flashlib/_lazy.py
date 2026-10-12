"""Shared lazy-import helper.

Each primitive's public API surfaces a small handful of CuteDSL backend
symbols. Importing CuteDSL eagerly pulls in the full ``cutlass`` /
``cutlass.cute`` stack which is heavy and hardware-gated (Hopper-only).
We therefore expose those names as thin wrappers that resolve their
backing module on first call.

This used to be duplicated as ``_lazy_cutedsl_attr`` inside every
``primitives/*/__init__.py``. Now it lives here so each primitive's
``__init__`` is one line per name::

    from flashlib._lazy import lazy_attr

    cutedsl_foo = lazy_attr("flashlib.primitives.foo.cutedsl", "cutedsl_foo")

The returned callable forwards ``*args, **kwargs`` to the resolved
attribute, so the wrapper is indistinguishable from the real function
at the call site.
"""
from __future__ import annotations

import importlib
import sys
from typing import Any, Callable

from flashlib._optional import has_cutedsl, require_cutedsl

# These ``cutedsl`` backends import and run without the CuTe DSL (torch / Triton / cuBLAS, or a Triton fallback).
_CUTEDSL_NAME_ONLY = frozenset({
    "flashlib.linalg.cholesky.cutedsl",
    "flashlib.linalg.qr.cutedsl",
    "flashlib.linalg.trsm.cutedsl",
    "flashlib.primitives.linear_regression.cutedsl",
    "flashlib.primitives.pca.cutedsl",
    "flashlib.primitives.random_forest.cutedsl",
})


def lazy_attr(module_path: str, name: str) -> Callable[..., Any]:
    """Return a callable that resolves ``getattr(import(module_path), name)``
    on first invocation and forwards ``*args, **kwargs`` to it.

    Parameters
    ----------
    module_path:
        Fully-qualified import path of the module that hosts ``name``
        (e.g. ``"flashlib.primitives.knn.cutedsl"``).
    name:
        Attribute (typically a function or class) to fetch from the
        module.
    """
    needs_cutedsl = ".cutedsl" in module_path and module_path not in _CUTEDSL_NAME_ONLY
    # Capability probes answer False instead of raising when the backend is missing.
    is_probe = name.endswith(("_available", "_supported"))

    def _f(*args, **kwargs):
        if needs_cutedsl and not has_cutedsl():
            if is_probe:
                return False
            require_cutedsl(name)
        mod = importlib.import_module(module_path)
        return getattr(mod, name)(*args, **kwargs)

    _f.__name__ = name
    _f.__qualname__ = name
    # Same trick as collections.namedtuple: lets pickle find the proxy in the module that defines it.
    _f.__module__ = sys._getframe(1).f_globals.get("__name__", __name__)
    _f.__doc__ = f"Lazy proxy for ``{module_path}.{name}``."
    return _f


__all__ = ["lazy_attr"]
