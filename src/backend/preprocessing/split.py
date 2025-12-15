from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split


@dataclass
class DatasetSplits:
    X_train: pd.DataFrame
    X_val: pd.DataFrame
    X_test: pd.DataFrame
    y_train: pd.Series
    y_val: pd.Series
    y_test: pd.Series


def stratified_case_split(
    df_cases: pd.DataFrame,
    target_col: str = "sla_violated",
    id_col: str = "case_id",
    test_size: float = 0.2,
    val_size: float = 0.25,
    random_state: int = 42,
) -> Tuple[DatasetSplits, List[str], List[str]]:
    """
    Divide dataset em nível de caso em treino / validação / teste.

    Returns
    -------
    DatasetSplits
    numeric_features
    categorical_features
    """
    feature_cols = [c for c in df_cases.columns if c not in [target_col, id_col]]

    X = df_cases[feature_cols].copy()
    y = df_cases[target_col].copy()

    X_train_full, X_test, y_train_full, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        stratify=y,
        random_state=random_state,
    )

    X_train, X_val, y_train, y_val = train_test_split(
        X_train_full,
        y_train_full,
        test_size=val_size,
        stratify=y_train_full,
        random_state=random_state,
    )

    numeric_features = X.select_dtypes(include=[np.number]).columns.tolist()
    categorical_features = X.select_dtypes(exclude=[np.number]).columns.tolist()

    splits = DatasetSplits(
        X_train=X_train,
        X_val=X_val,
        X_test=X_test,
        y_train=y_train,
        y_val=y_val,
        y_test=y_test,
    )

    return splits, numeric_features, categorical_features
