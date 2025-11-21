from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder

from .split import DatasetSplits


@dataclass
class PreprocessArtifacts:
    preprocessor: ColumnTransformer
    numeric_features: List[str]
    categorical_features: List[str]
    feature_names: List[str]


def build_preprocessor(
    numeric_features: List[str],
    categorical_features: List[str],
) -> ColumnTransformer:
    """Cria ColumnTransformer numérico + categórico (OneHot)."""
    numeric_transformer = "passthrough"
    categorical_transformer = OneHotEncoder(handle_unknown="ignore")

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, numeric_features),
            ("cat", categorical_transformer, categorical_features),
        ]
    )

    return preprocessor


def fit_preprocessor_and_transform(
    preprocessor: ColumnTransformer,
    splits: DatasetSplits,
    numeric_features: List[str],
    categorical_features: List[str],
) -> Tuple[PreprocessArtifacts, np.ndarray, np.ndarray, np.ndarray]:
    """
    Ajusta o preprocessor em X_train e transforma X_train, X_val, X_test.
    """
    X_train_pre = preprocessor.fit_transform(splits.X_train)
    X_val_pre = preprocessor.transform(splits.X_val)
    X_test_pre = preprocessor.transform(splits.X_test)

    feature_names: List[str] = []

    if numeric_features:
        feature_names.extend(numeric_features)

    if categorical_features:
        ohe: OneHotEncoder = preprocessor.named_transformers_["cat"]  # type: ignore
        cat_names = ohe.get_feature_names_out(categorical_features).tolist()
        feature_names.extend(cat_names)

    artifacts = PreprocessArtifacts(
        preprocessor=preprocessor,
        numeric_features=numeric_features,
        categorical_features=categorical_features,
        feature_names=feature_names,
    )

    return artifacts, X_train_pre, X_val_pre, X_test_pre
