from __future__ import annotations

from typing import Any, Dict
import numpy as np
from xgboost import XGBClassifier

from .base_model import BaseModel


class XGBoostModel(BaseModel):
    """Wrapper para XGBClassifier."""

    def __init__(self, **params: Any) -> None:
        params.setdefault("eval_metric", "logloss")
        params.setdefault("use_label_encoder", False)
        self.params: Dict[str, Any] = params
        self.model = XGBClassifier(**params)

    def train(self, X: np.ndarray, y: np.ndarray) -> None:
        self.model.fit(X, y)

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self.model.predict(X)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        return self.model.predict_proba(X)
