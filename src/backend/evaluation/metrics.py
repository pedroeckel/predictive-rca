from __future__ import annotations

from typing import Any, Dict

import numpy as np
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score

from src.backend.models.base_model import ProbabilisticClassifier


def evaluate_binary_classification(
    model: ProbabilisticClassifier,
    X: np.ndarray,
    y_true: np.ndarray,
    threshold: float = 0.5,
    verbose: bool = True,
) -> Dict[str, Any]:
    """
    Avalia um modelo binário com AUC-ROC, acurácia, classificação e matriz de confusão.
    """
    y_proba = model.predict_proba(X)[:, 1]
    y_pred = (y_proba >= threshold).astype(int)

    auc = roc_auc_score(y_true, y_proba)
    accuracy = np.mean(y_true == y_pred)
    report_str = classification_report(y_true, y_pred)
    matrix = confusion_matrix(y_true, y_pred)

    if verbose:
        print("AUC-ROC:", auc)
        print("Accuracy:", accuracy)
        print("\nClassification report:\n", report_str)
        print("\nConfusion matrix:\n", matrix)

    return {
        "auc": float(auc),
        "accuracy": float(accuracy),
        "classification_report": report_str,
        "confusion_matrix": matrix,
    }
