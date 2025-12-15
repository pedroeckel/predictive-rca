from __future__ import annotations

from typing import Dict, Iterable, Tuple, Type

import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.metrics import log_loss

from src.backend.config.settings import CONFIG
from src.backend.preprocessing.build_features import TargetBuilder, build_case_features
from src.backend.preprocessing.split import (
    DatasetSplits,
    stratified_case_split,
)
from src.backend.preprocessing.preprocess import (
    PreprocessArtifacts,
    build_preprocessor,
    fit_preprocessor_and_transform,
)
from src.backend.models.base_model import BaseModel
from src.backend.models.lightgbm_model import LightGBMModel
from src.backend.optimization.bayesian import optimize_bayesian
from src.backend.evaluation.metrics import evaluate_binary_classification
from src.backend.evaluation.feature_importance import plot_feature_importance
from src.backend.evaluation.shap_analysis import (
    compute_shap_values,
    plot_shap_summary,
    plot_shap_dependence_top_feature,
    plot_shap_force_single,
)
from src.backend.utils.logs import get_logger
from src.backend.utils.timer import timer


class PipelineBuilder:
    """
    Orquestra o pipeline completo, agora ACEITANDO QUALQUER MODELO:
    - log de eventos → features por caso
    - split
    - pré-processamento
    - (opcional) otimização bayesiana
    - treino final
    - avaliação + SHAP
    """

    def __init__(
        self,
        model_class: Type[BaseModel] = LightGBMModel,
        model_params: Dict | None = None,
        optimize_hyperparams: bool = True,
    ) -> None:
        """
        Parameters
        ----------
        model_class : Type[BaseModel]
            Classe do modelo (LightGBMModel, XGBoostModel, etc.)
        model_params : dict
            Parâmetros iniciais do modelo
        optimize_hyperparams : bool
            Se True, aplica otimização bayesiana (apenas LightGBM por enquanto)
        """
        self.model_class = model_class
        self.model_params = model_params or {}
        self.optimize_hyperparams = optimize_hyperparams
        self.logger = get_logger(self.__class__.__name__)

    # ---------------------------------------------------------
    # CARREGAMENTO DO LOG
    # ---------------------------------------------------------
    def _load_event_log(self, path: str) -> pd.DataFrame:
        df = pd.read_csv(path)
        if "timestamp" in df.columns:
            df["timestamp"] = pd.to_datetime(df["timestamp"])
        return df

    # ---------------------------------------------------------
    # FUNÇÃO DE AVALIAÇÃO PARA LIGHTGBM (OTIMIZAÇÃO BAYESIANA)
    # ---------------------------------------------------------
    def _make_lightgbm_eval_fn(
        self,
        X_train_pre: np.ndarray,
        y_train: np.ndarray,
        X_val_pre: np.ndarray,
        y_val: np.ndarray,
    ):
        def evaluate(
            num_leaves: float,
            max_depth: float,
            learning_rate: float,
            min_data_in_leaf: float,
            feature_fraction: float,
        ) -> float:

            model = LGBMClassifier(
                num_leaves=int(num_leaves),
                max_depth=int(max_depth),
                learning_rate=float(learning_rate),
                min_data_in_leaf=int(min_data_in_leaf),
                feature_fraction=float(feature_fraction),
                n_estimators=300,
                objective="binary",
                random_state=CONFIG.RANDOM_STATE,
                n_jobs=-1,
            )

            model.fit(
                X_train_pre,
                y_train,
                eval_set=[(X_val_pre, y_val)],
                eval_metric="logloss",
                callbacks=[],
            )

            y_proba = model.predict_proba(X_val_pre)[:, 1]
            loss = log_loss(y_val, y_proba)
            return -loss

        return evaluate

    # ---------------------------------------------------------
    # PIPELINE PRINCIPAL
    # ---------------------------------------------------------
    def run_from_event_log(
        self,
        log_path: str,
        sla_hours: float | None = None,
        target_builders: Iterable[TargetBuilder] | None = None,
        include_default_target: bool = True,
        target_col: str = "sla_violated",
        throughput_col: str = "throughput_hours",
        feature_importance_top_k: int | None = 15,
    ):
        if sla_hours is None:
            sla_hours = CONFIG.SLA_HOURS

        self.logger.info("Carregando log de eventos...")
        df_events = self._load_event_log(log_path)

        self.logger.info("Construindo features em nível de caso...")
        df_cases = build_case_features(
            df_events,
            sla_hours=sla_hours,
            target_builders=target_builders,
            include_default_target=include_default_target,
            default_target_name=target_col,
            throughput_col=throughput_col,
        )

        self.logger.info("Realizando split estratificado...")
        splits, numeric_features, categorical_features = stratified_case_split(
            df_cases,
            target_col=target_col,
            id_col="case_id",
            random_state=CONFIG.RANDOM_STATE,
        )

        self.logger.info("Construindo preprocessor...")
        preprocessor = build_preprocessor(numeric_features, categorical_features)

        self.logger.info("Transformando dados...")
        artifacts, X_train_pre, X_val_pre, X_test_pre = fit_preprocessor_and_transform(
            preprocessor,
            splits,
            numeric_features,
            categorical_features,
        )

        y_train = splits.y_train.to_numpy()
        y_val = splits.y_val.to_numpy()
        y_test = splits.y_test.to_numpy()

        # ---------------------------------------------------------
        #  OPTIMIZAÇÃO BAYESIANA (APENAS LIGHTGBM)
        # ---------------------------------------------------------
        if self.optimize_hyperparams and self.model_class is LightGBMModel:
            self.logger.info("Iniciando otimização bayesiana para LightGBM...")
            eval_fn = self._make_lightgbm_eval_fn(
                X_train_pre, y_train, X_val_pre, y_val
            )

            pbounds: Dict[str, Tuple[float, float]] = {
                "num_leaves": (16, 120),
                "max_depth": (3, 18),
                "learning_rate": (0.01, 0.3),
                "min_data_in_leaf": (5, 80),
                "feature_fraction": (0.5, 1.0),
            }

            with timer("Otimização Bayesiana"):
                best_params = optimize_bayesian(
                    eval_fn,
                    pbounds,
                    init_points=8,
                    n_iter=20,
                    random_state=CONFIG.RANDOM_STATE,
                )

            self.model_params.update(
                {
                    "num_leaves": int(best_params["num_leaves"]),
                    "max_depth": int(best_params["max_depth"]),
                    "learning_rate": float(best_params["learning_rate"]),
                    "min_data_in_leaf": int(best_params["min_data_in_leaf"]),
                    "feature_fraction": float(best_params["feature_fraction"]),
                    "n_estimators": 300,
                    "objective": "binary",
                    "random_state": CONFIG.RANDOM_STATE,
                    "n_jobs": -1,
                }
            )

            self.logger.info(f"Melhores hiperparâmetros: {self.model_params}")

        # ---------------------------------------------------------
        #  TREINO FINAL COM QUALQUER MODELO
        # ---------------------------------------------------------
        self.logger.info(f"Treinando modelo final: {self.model_class.__name__}...")
        model = self.model_class(**self.model_params)
        model.train(X_train_pre, y_train)

        # ---------------------------------------------------------
        # AVALIAÇÃO
        # ---------------------------------------------------------
        self.logger.info("Avaliando em validação...")
        evaluate_binary_classification(model, X_val_pre, y_val)

        self.logger.info("Avaliando em teste...")
        evaluate_binary_classification(model, X_test_pre, y_test)

        # ---------------------------------------------------------
        #  EXPLICABILIDADE
        # ---------------------------------------------------------
        if hasattr(model, "model") and hasattr(model.model, "feature_importances_"):
            self.logger.info("Plotando feature importance...")
            plot_feature_importance(
                model.model,
                artifacts.feature_names,
                max_features=feature_importance_top_k,
            )

        if hasattr(model, "model"):
            self.logger.info("Computando SHAP values...")
            explainer, shap_values = compute_shap_values(model.model, X_test_pre)

            self.logger.info("Plotando SHAP summary...")
            plot_shap_summary(shap_values, X_test_pre, artifacts.feature_names)

            plot_shap_dependence_top_feature(
                shap_values, X_test_pre, artifacts.feature_names
            )

            plot_shap_force_single(
                explainer,
                shap_values,
                X_test_pre,
                artifacts.feature_names,
                index=0,
            )

        return model, artifacts, splits
