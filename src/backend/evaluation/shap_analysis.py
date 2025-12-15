from __future__ import annotations

from typing import Any, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import shap
from lightgbm import LGBMClassifier
from pandas import DataFrame
from scipy.sparse import issparse


def _to_dense(X: np.ndarray) -> np.ndarray:
    """Converte matriz potencialmente esparsa em densa para uso no pandas/SHAP."""
    return X.toarray() if issparse(X) else np.asarray(X)


def _select_shap_values(
    shap_values: Any, positive_class_index: int
) -> np.ndarray:
    """Seleciona a classe positiva (se lista) e garante formato denso."""
    if isinstance(shap_values, list):
        values = shap_values[positive_class_index]
    else:
        values = shap_values
    return _to_dense(values)


def compute_shap_values(
    model: Any,
    X_pre: np.ndarray,
) -> Tuple[Any, Any]:
    """
    Computa valores SHAP usando TreeExplainer.
    """
    model_type = type(model).__name__
    if 'LogisticRegression' in model_type:
        explainer = shap.LinearExplainer(model, X_pre)
    else:
        explainer = shap.TreeExplainer(model)

    shap_values = explainer.shap_values(X_pre)
    return explainer, shap_values


def plot_shap_summary(
    shap_values: Any,
    X_pre: np.ndarray,
    feature_names: List[str],
    positive_class_index: int = 1,
) -> None:
    """
    Plota SHAP summary plot global.
    """
    shap.initjs()

    shap_values_pos = _select_shap_values(shap_values, positive_class_index)
    X_dense = _to_dense(X_pre)
    X_df = DataFrame(X_dense, columns=feature_names)

    plt.figure(figsize=(10, 6))
    shap.summary_plot(shap_values_pos, X_df, show=False)
    plt.tight_layout()
    plt.show()


def plot_shap_dependence_top_feature(
    shap_values: Any,
    X_pre: np.ndarray,
    feature_names: List[str],
    positive_class_index: int = 1,
) -> None:
    """
    Plota SHAP dependence plot para a feature mais importante.
    """
    if isinstance(shap_values, list):
        shap_values_pos = shap_values[positive_class_index]
    elif len(shap_values.shape) == 3:
        shap_values_pos = shap_values[:, :, positive_class_index]
    else:
        shap_values_pos = shap_values

    X_dense = _to_dense(X_pre)
    shap_dense = _to_dense(shap_values_pos)
    X_df = DataFrame(X_dense, columns=feature_names)

    importances = np.mean(np.abs(shap_dense), axis=0)
    top_idx = int(np.argmax(importances))
    feat = feature_names[top_idx]

    plt.figure(figsize=(8, 5))
    shap.dependence_plot(top_idx, shap_dense, X_df, show=False)
    plt.tight_layout()
    plt.show()


def plot_shap_force_single(
    explainer: Any,
    shap_values: Any,
    X_pre: np.ndarray,
    feature_names: List[str],
    index: int = 0,
    positive_class_index: int = 1,
) -> None:
    """
    Plota explicação local SHAP (force_plot) para um único exemplo.
    """
    shap_values_pos = _select_shap_values(shap_values, positive_class_index)
    shap_dense = _to_dense(shap_values_pos)

    if hasattr(explainer, 'expected_value'):
        if isinstance(explainer.expected_value, (list, np.ndarray)):
            expected_value = explainer.expected_value[positive_class_index]
        else:
            expected_value = explainer.expected_value
    else:
        expected_value = 0

    X_dense = _to_dense(X_pre)
    X_df = DataFrame(X_dense, columns=feature_names)

    shap.force_plot(
        base_value=expected_value,
        shap_values=shap_dense[index, :],
        features=X_df.iloc[index, :],
        feature_names=feature_names,
        matplotlib=True,
        show=False
    )
