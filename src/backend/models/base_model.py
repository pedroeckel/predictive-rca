from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Protocol

import numpy as np


class BaseModel(ABC):
    """Interface abstrata para modelos de classificação."""

    @abstractmethod
    def train(self, X: np.ndarray, y: np.ndarray) -> None:
        ...

    @abstractmethod
    def predict(self, X: np.ndarray) -> np.ndarray:
        ...

    @abstractmethod
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        ...


class ProbabilisticClassifier(Protocol):
    """Protocolo para permitir tipagem em funções genéricas de avaliação."""

    def predict(self, X: np.ndarray) -> np.ndarray:  # pragma: no cover
        ...

    def predict_proba(self, X: np.ndarray) -> np.ndarray:  # pragma: no cover
        ...
