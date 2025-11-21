from __future__ import annotations

from typing import Any, Dict

import numpy as np
from lightgbm import LGBMClassifier

from .base_model import BaseModel


class LightGBMModel(BaseModel):
    """Wrapper simples para LGBMClassifier seguindo a interface BaseModel."""

    def __init__(self, **params: Any) -> None:
        self.params: Dict[str, Any] = params
        self.model = LGBMClassifier(**params)

    def train(self, X: np.ndarray, y: np.ndarray) -> None:
        self.model.fit(X, y)

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self.model.predict(X)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        return self.model.predict_proba(X)
