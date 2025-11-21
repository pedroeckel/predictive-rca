from __future__ import annotations

from typing import Any, Dict
import numpy as np
from sklearn.linear_model import LogisticRegression

from .base_model import BaseModel


class LogisticRegressionModel(BaseModel):
    """Wrapper da regressão logística."""

    def __init__(self, **params: Any) -> None:
        params.setdefault("max_iter", 500)
        self.params: Dict[str, Any] = params
        self.model = LogisticRegression(**params)

    def train(self, X: np.ndarray, y: np.ndarray) -> None:
        self.model.fit(X, y)

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self.model.predict(X)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        return self.model.predict_proba(X)
