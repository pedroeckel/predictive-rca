from __future__ import annotations

from typing import Any, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import shap
from lightgbm import LGBMClassifier
from pandas import DataFrame


def compute_shap_values(
    model: LGBMClassifier,
    X_pre: np.ndarray,
) -> Tuple[shap.TreeExplainer, Any]:
    """
    Computa valores SHAP usando TreeExplainer.
    """
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

    if isinstance(shap_values, list):
        shap_values_pos = shap_values[positive_class_index]
    else:
        shap_values_pos = shap_values

    X_df = DataFrame(X_pre, columns=feature_names)

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
    else:
        shap_values_pos = shap_values

    X_df = DataFrame(X_pre, columns=feature_names)

    importances = np.mean(np.abs(shap_values_pos), axis=0)
    top_idx = int(np.argmax(importances))
    feat = feature_names[top_idx]

    plt.figure(figsize=(8, 5))
    shap.dependence_plot(feat, shap_values_pos, X_df, show=False)
    plt.tight_layout()
    plt.show()


def plot_shap_force_single(
    explainer: shap.TreeExplainer,
    shap_values: Any,
    X_pre: np.ndarray,
    feature_names: List[str],
    index: int = 0,
    positive_class_index: int = 1,
) -> None:
    """
    Plota explicação local SHAP (force_plot) para um único exemplo.
    """
    if isinstance(shap_values, list):
        shap_values_pos = shap_values[positive_class_index]
        expected_value = explainer.expected_value[positive_class_index]
    else:
        shap_values_pos = shap_values
        expected_value = explainer.expected_value

    X_df = DataFrame(X_pre, columns=feature_names)

    shap.force_plot(
        expected_value,
        shap_values_pos[index, :],
        X_df.iloc[index, :],
    )
