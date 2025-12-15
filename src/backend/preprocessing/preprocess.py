from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

from .split import DatasetSplits


@dataclass
class PreprocessArtifacts:
    preprocessor: ColumnTransformer
    numeric_features: List[str]
    categorical_features: List[str]
    feature_names: List[str]


def _build_readable_ohe_names(
    ohe: OneHotEncoder, categorical_features: List[str]
) -> List[str]:
    """
    Constrói nomes amigáveis (coluna=valor) para cada dummy gerado pelo OneHotEncoder.
    Usa categories_ para preservar a ordem exata das colunas codificadas.
    """
    if not hasattr(ohe, "categories_"):
        return []

    categories = getattr(ohe, "categories_", [])
    if len(categories) != len(categorical_features):
        return []

    drop_idx = getattr(ohe, "drop_idx_", None)
    drop_idx_list = list(drop_idx) if drop_idx is not None else [None] * len(categorical_features)

    names: List[str] = []
    for col, cats, drop in zip(categorical_features, categories, drop_idx_list):
        for idx, cat in enumerate(cats):
            if drop is not None and idx == drop:
                continue
            names.append(f"{col}={cat}")

    try:
        expected = len(ohe.get_feature_names_out())
    except Exception:
        expected = None

    return names if expected is None or expected == len(names) else []


def _prettify_raw_ohe_name(name: str) -> str:
    """
    Limpa prefixos técnicos e deixa explícito o valor (coluna=valor).
    Ex.: 'cat__status_Aprovado' -> 'status=Aprovado'
    """
    cleaned = name.replace("cat__", "", 1)
    if "__" in cleaned:
        cleaned = cleaned.split("__", 1)[1]
    if "_" in cleaned:
        col, val = cleaned.rsplit("_", 1)
        return f"{col}={val}"
    return cleaned


def build_preprocessor(
    numeric_features: List[str],
    categorical_features: List[str],
) -> ColumnTransformer:
    """Cria ColumnTransformer numérico + categórico (OneHot)."""
    numeric_transformer = Pipeline(
        steps=[
            ("imputer_median", SimpleImputer(strategy="median")),
            # Garantir preenchimento mesmo se a coluna inteira for NaN
            ("imputer_constant", SimpleImputer(strategy="constant", fill_value=0)),
        ]
    )
    categorical_transformer = Pipeline(
        steps=[
            ("imputer_freq", SimpleImputer(strategy="most_frequent")),
            # Se toda a coluna for NaN, força valor padrão
            ("imputer_constant", SimpleImputer(strategy="constant", fill_value="missing")),
            ("encoder", OneHotEncoder(handle_unknown="ignore")),
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, numeric_features),
            ("cat", categorical_transformer, categorical_features),
        ]
        ,
        verbose_feature_names_out=False,  # evita prefixos como "cat__col"
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
        cat_pipeline: Pipeline = preprocessor.named_transformers_["cat"]  # type: ignore
        ohe: OneHotEncoder = cat_pipeline.named_steps["encoder"]
        cat_names = _build_readable_ohe_names(ohe, categorical_features)
        if not cat_names:
            try:
                cat_names = ohe.get_feature_names_out(categorical_features).tolist()
            except ValueError:
                # Se a lista passada não casar com o que o encoder viu, use o que ele registrou
                cat_names = ohe.get_feature_names_out().tolist()
            cat_names = [_prettify_raw_ohe_name(name) for name in cat_names]
        feature_names.extend(cat_names)

    artifacts = PreprocessArtifacts(
        preprocessor=preprocessor,
        numeric_features=numeric_features,
        categorical_features=categorical_features,
        feature_names=feature_names,
    )

    return artifacts, X_train_pre, X_val_pre, X_test_pre
