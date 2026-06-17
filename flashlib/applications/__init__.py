"""sklearn-style classes wrapping primitives. For users who don't want to
read kernel code — just import and call fit/predict/transform.

Lazy-loaded so importing this package does not eagerly pull every primitive
(which would create import cycles, e.g. spectral_clustering -> kmeans
application class -> applications/__init__.py -> ...).
"""
from __future__ import annotations

import importlib

_LAZY_CLASSES: dict[str, str] = {
    "KMeans":                "flashlib.applications.kmeans",
    "FlashKMeans":           "flashlib.applications.kmeans",
    "NearestNeighbors":      "flashlib.applications.knn",
    "IVFFlat":               "flashlib.applications.ivf_flat",
    "IVFPQ":                 "flashlib.applications.ivf_pq",
    "PCA":                   "flashlib.applications.pca",
    "StandardScaler":        "flashlib.applications.standard_scaler",
    "DBSCAN":                "flashlib.applications.dbscan",
    "TruncatedSVD":          "flashlib.applications.truncated_svd",
    "LinearRegression":      "flashlib.applications.linear_regression",
    "Ridge":                 "flashlib.applications.ridge",
    "LogisticRegression":    "flashlib.applications.logistic_regression",
    "HDBSCAN":               "flashlib.applications.hdbscan",
    "UMAP":                  "flashlib.applications.umap",
    "TSNE":                  "flashlib.applications.tsne",
    "MultinomialNB":         "flashlib.applications.multinomial_nb",
    "RandomForestClassifier":"flashlib.applications.random_forest",
    "SpectralClustering":    "flashlib.applications.spectral_clustering",
}


def __getattr__(name: str):
    if name in _LAZY_CLASSES:
        mod = importlib.import_module(_LAZY_CLASSES[name])
        cls = getattr(mod, name)
        globals()[name] = cls
        return cls
    raise AttributeError(f"module 'flashlib.applications' has no attribute {name!r}")


def __dir__() -> list[str]:
    return sorted(set(globals()) | set(_LAZY_CLASSES))


__all__ = list(_LAZY_CLASSES.keys())
