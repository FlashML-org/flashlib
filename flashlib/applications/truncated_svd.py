"""TruncatedSVD — sklearn-style wrapper around primitives.truncated_svd."""
from __future__ import annotations

import torch

from flashlib.primitives.truncated_svd import flash_truncated_svd


class TruncatedSVD:
    """Truncated singular value decomposition (randomized SVD).

    Args:
        n_components: number of components to keep.

    Attributes set after fit():
        components_:          (K, D) right singular vectors
        singular_values_:     (K,)   top singular values
        explained_variance_:  (K,)   variance per component
    """

    def __init__(self, n_components: int):
        self.n_components = int(n_components)
        self.components_ = None
        self.singular_values_ = None
        self.explained_variance_ = None

    def fit(self, X: torch.Tensor) -> "TruncatedSVD":
        if not X.is_cuda or X.ndim != 2:
            raise ValueError("TruncatedSVD requires a 2D CUDA tensor")
        S, Vt = flash_truncated_svd(X, self.n_components)
        self.components_ = Vt.contiguous()
        self.singular_values_ = S
        self.explained_variance_ = (S ** 2) / max(X.shape[0] - 1, 1)
        return self

    def transform(self, X: torch.Tensor) -> torch.Tensor:
        if self.components_ is None:
            raise RuntimeError("TruncatedSVD not fitted; call fit() first.")
        return X @ self.components_.T

    def fit_transform(self, X: torch.Tensor) -> torch.Tensor:
        self.fit(X)
        return self.transform(X)


__all__ = ["TruncatedSVD"]
