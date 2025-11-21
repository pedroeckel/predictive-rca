from __future__ import annotations

from typing import Any, Dict
import numpy as np
from catboost import CatBoostClassifier

from .base_model import BaseModel


class CatBoostModel(BaseModel):
    """Wrapper para CatBoost seguindo BaseModel."""

    def __init__(self, **params: Any) -> None:
        params.setdefault("verbose", False)
        self.params: Dict[str, Any] = params
        self.model = CatBoostClassifier(**params)

    def train(self, X: np.ndarray, y: np.ndarray) -> None:
        self.model.fit(X, y)

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self.model.predict(X)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        return self.model.predict_proba(X)
