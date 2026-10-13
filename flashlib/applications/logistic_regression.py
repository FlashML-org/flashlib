"""LogisticRegression — sklearn-style wrapper around primitives.logistic_regression."""
from __future__ import annotations

import torch

from flashlib.primitives.logistic_regression import flash_logistic_regression


class LogisticRegression:
    """Binary logistic regression via L-BFGS with fused fwd/bwd Triton kernel.

    Args:
        C: inverse regularization strength (sklearn convention).
        max_iter: maximum L-BFGS iterations.

    Attributes set after fit():
        coef_:       (D,) weights
        intercept_:  scalar bias
    """

    def __init__(self, C: float = 1.0, max_iter: int = 100):
        self.C = float(C)
        self.max_iter = int(max_iter)
        self.coef_ = None
        self.intercept_ = 0.0

    def fit(self, X: torch.Tensor, y: torch.Tensor) -> "LogisticRegression":
        if not X.is_cuda or X.ndim != 2 or y.ndim != 1:
            raise ValueError("LogisticRegression requires 2D X and 1D y on CUDA")
        result = flash_logistic_regression(X, y, C=self.C, n_iter=self.max_iter)
        if isinstance(result, tuple) and len(result) == 2:
            self.coef_, self.intercept_ = result
        else:
            self.coef_ = result
        return self

    def predict_proba(self, X: torch.Tensor) -> torch.Tensor:
        if self.coef_ is None:
            raise RuntimeError("LogisticRegression not fitted; call fit() first.")
        logits = X @ self.coef_ + self.intercept_
        p1 = torch.sigmoid(logits)
        return torch.stack([1 - p1, p1], dim=1)

    def predict(self, X: torch.Tensor) -> torch.Tensor:
        return (self.predict_proba(X)[:, 1] >= 0.5).long()


__all__ = ["LogisticRegression"]
