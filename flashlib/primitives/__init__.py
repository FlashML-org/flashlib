"""Algorithm primitives. Each subpackage is independently callable.

All primitives are first-class:

    from flashlib import flash_kmeans, flash_knn, flash_pca, ...    # top-level
    from flashlib.primitives.kmeans import batch_kmeans_Euclid       # qualified

Subpackages are loaded lazily — importing flashlib.primitives itself does
not pull in any of the heavy primitive modules (e.g. hdbscan builds C
helpers on first use, random_forest is 1400 LOC).

v0.1 surface (15 ops):
  - clustering:        kmeans, dbscan, hdbscan, spectral_clustering
  - neighbors:         knn
  - decomposition:     pca, truncated_svd
  - regression:        linear_regression, ridge, logistic_regression
  - manifold:          umap, tsne
  - classification:    multinomial_nb, random_forest
  - preprocessing:     standard_scaler
"""
from __future__ import annotations

import importlib

_SUBPACKAGES = (
    "standard_scaler",
    "kmeans", "knn",
    "pca", "truncated_svd",
    "linear_regression", "ridge", "logistic_regression",
    "dbscan", "hdbscan",
    "umap", "tsne",
    "multinomial_nb", "random_forest",
    "spectral_clustering",
)


def __getattr__(name: str):
    if name in _SUBPACKAGES:
        mod = importlib.import_module(f"{__name__}.{name}")
        globals()[name] = mod
        return mod
    raise AttributeError(f"module 'flashlib.primitives' has no attribute {name!r}")


def __dir__() -> list[str]:
    return sorted(set(globals()) | set(_SUBPACKAGES))


__all__ = list(_SUBPACKAGES)
