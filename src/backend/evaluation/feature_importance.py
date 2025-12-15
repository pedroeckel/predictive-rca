from __future__ import annotations

from typing import List, Tuple

import matplotlib.pyplot as plt
import numpy as np
from lightgbm import LGBMClassifier
from matplotlib.figure import Figure


def plot_feature_importance(
    model: LGBMClassifier,
    feature_names: List[str],
    max_features: int | None = 15,
    figsize: Tuple[int, int] = (10, 6),
    show: bool = True,
) -> Figure:
    """
    Plota importância de features para modelos baseados em árvore (LightGBM).
    """
    importances = model.feature_importances_
    order = np.argsort(importances)[::-1]

    if max_features is not None:
        order = order[:max_features]

    sorted_features = []
    for i in order:
        try:
            sorted_features.append(feature_names[i])
        except IndexError:
            sorted_features.append(f"feature_{i}")
    sorted_importances = importances[order]

    indices = np.arange(len(sorted_features))
    fig, ax = plt.subplots(figsize=figsize)
    ax.bar(indices, sorted_importances)
    ax.set_xticks(indices)
    ax.set_title("Importância das Features (LightGBM)")
    ax.set_xticklabels(sorted_features, rotation=90)
    fig.tight_layout()
    if show:
        plt.show()
    return fig
