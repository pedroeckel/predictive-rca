from __future__ import annotations

from typing import List, Tuple

import matplotlib.pyplot as plt
import numpy as np
from lightgbm import LGBMClassifier


def plot_feature_importance(
    model: LGBMClassifier,
    feature_names: List[str],
    max_features: int | None = 15,
    figsize: Tuple[int, int] = (10, 6),
) -> None:
    """
    Plota importância de features para modelos baseados em árvore (LightGBM).
    """
    importances = model.feature_importances_
    order = np.argsort(importances)[::-1]

    if max_features is not None:
        order = order[:max_features]

    sorted_features = [feature_names[i] for i in order]
    sorted_importances = importances[order]

    plt.figure(figsize=figsize)
    plt.bar(sorted_features, sorted_importances)
    plt.title("Importância das Features (LightGBM)")
    plt.xticks(rotation=90)
    plt.tight_layout()
    plt.show()
