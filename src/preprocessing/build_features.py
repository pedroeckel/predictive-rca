from __future__ import annotations

from typing import Callable, Iterable

import pandas as pd


TargetBuilder = Callable[[pd.DataFrame], pd.Series | pd.DataFrame]


def build_sla_target_builder(
    sla_hours: float,
    target_col: str = "sla_violated",
    throughput_col: str = "throughput_hours",
) -> TargetBuilder:
    """
    Retorna uma função que cria o alvo binário a partir do SLA.
    """

    def _builder(features: pd.DataFrame) -> pd.Series:
        target = (features[throughput_col] > sla_hours).astype(int)
        target.name = target_col
        return target

    return _builder


def _normalize_target_output(target_output: pd.Series | pd.DataFrame) -> pd.DataFrame:
    """
    Garante que o alvo retornado por um builder seja DataFrame para concatenação.
    """
    if isinstance(target_output, pd.Series):
        return target_output.to_frame()
    if isinstance(target_output, pd.DataFrame):
        return target_output
    raise TypeError("Target builders devem retornar pd.Series ou pd.DataFrame.")


def _apply_target_builders(
    features: pd.DataFrame, target_builders: Iterable[TargetBuilder] | None
) -> pd.DataFrame:
    """
    Executa uma lista de builders de alvo e concatena os resultados às features.
    """
    if not target_builders:
        return features

    target_frames = [
        _normalize_target_output(builder(features)) for builder in target_builders
    ]
    return pd.concat([features, *target_frames], axis=1)


def build_case_features(
    df_events: pd.DataFrame,
    sla_hours: float,
    case_id_col: str = "case_id",
    activity_col: str = "activity",
    timestamp_col: str = "timestamp",
    resource_col: str | None = "resource",
    cost_col: str | None = "cost",
    target_builders: Iterable[TargetBuilder] | None = None,
    include_default_target: bool = True,
    default_target_name: str = "sla_violated",
    throughput_col: str = "throughput_hours",
    extra_numeric_cols: list[str] | None = None,
    include_throughput_feature: bool = True,
    include_num_events: bool = True,
    include_num_unique_activities: bool = True,
    include_rework_count: bool = True,
    include_start_activity: bool = True,
    include_end_activity: bool = True,
    include_start_resource: bool = True,
    include_mean_cost: bool = True,
    include_extra_numeric_cols: bool = True,
    drop_feature_cols: list[str] | None = None,
) -> pd.DataFrame:
    """
    Constrói features em nível de caso a partir de um log de eventos.

    Parameters
    ----------
    df_events : pd.DataFrame
        Log de eventos.
    sla_hours : float
        Limiar de SLA em horas.
    case_id_col : str
        Nome da coluna que identifica o caso.
    activity_col : str
        Nome da coluna de atividade.
    timestamp_col : str
        Nome da coluna de timestamp.
    resource_col : str | None
        Nome da coluna de recurso (opcional).
    cost_col : str | None
        Nome da coluna de custo (opcional).
    extra_numeric_cols : list[str] | None
        Colunas numéricas adicionais para agregação (média) por caso.
    include_throughput_feature : bool
        Quando False, remove a coluna de throughput das features finais
        (mas mantém para cálculo de alvo, se necessário).
    include_num_events : bool
        Inclui contagem de eventos por caso.
    include_num_unique_activities : bool
        Inclui contagem de atividades únicas por caso.
    include_rework_count : bool
        Inclui contagem de rework (num_events - num_unique_activities).
    include_start_activity : bool
        Inclui atividade inicial.
    include_end_activity : bool
        Inclui atividade final.
    include_start_resource : bool
        Inclui recurso inicial (se existir).
    include_mean_cost : bool
        Inclui custo médio (se existir).
    include_extra_numeric_cols : bool
        Inclui as colunas numéricas extras agregadas.
    drop_feature_cols : list[str] | None
        Lista de colunas a remover das features finais (ex.: variável-alvo para evitar leakage).
    target_builders : Iterable[TargetBuilder] | None
        Funções opcionais que recebem o DataFrame de features por caso
        e retornam uma ou mais colunas de alvo.
    include_default_target : bool
        Quando True, adiciona automaticamente o alvo binário de SLA.
    default_target_name : str
        Nome da coluna do alvo de SLA padrão.
    throughput_col : str
        Nome da coluna que contém o tempo de ciclo em horas.

    Returns
    -------
    pd.DataFrame
        Dataset em nível de caso, com colunas-alvo construídas pelos builders.
    """
    df = df_events.copy()

    if timestamp_col in df.columns:
        df[timestamp_col] = pd.to_datetime(df[timestamp_col])

    df = df.sort_values([case_id_col, timestamp_col])

    agg_time = df.groupby(case_id_col)[timestamp_col].agg(["min", "max"])
    agg_time[throughput_col] = (
        agg_time["max"] - agg_time["min"]
    ).dt.total_seconds() / 3600.0

    features_to_concat: list[pd.Series | pd.DataFrame] = []

    if include_throughput_feature:
        features_to_concat.append(agg_time[[throughput_col]])
    else:
        # mantém apenas min/max para eventuais alvos; removerá depois se não forem necessários
        features_to_concat.append(agg_time[["min", "max"]])

    if include_num_events:
        num_events = df.groupby(case_id_col).size().rename("num_events")
        features_to_concat.append(num_events)

    if include_num_unique_activities:
        num_unique_acts = (
            df.groupby(case_id_col)[activity_col]
            .nunique()
            .rename("num_unique_activities")
        )
        features_to_concat.append(num_unique_acts)
    else:
        num_unique_acts = None

    if include_rework_count:
        if "num_events" in [s.name for s in features_to_concat if hasattr(s, "name")] and num_unique_acts is not None:
            rework_count = (num_events - num_unique_acts).rename("rework_count")  # type: ignore
            features_to_concat.append(rework_count)

    if include_start_activity:
        first_activity = (
            df.groupby(case_id_col).first()[activity_col].rename("start_activity")
        )
        features_to_concat.append(first_activity)

    if include_end_activity:
        last_activity = (
            df.groupby(case_id_col).last()[activity_col].rename("end_activity")
        )
        features_to_concat.append(last_activity)

    if include_start_resource:
        if resource_col is not None and resource_col in df.columns:
            first_resource = (
                df.groupby(case_id_col).first()[resource_col].rename("start_resource")
            )
        else:
            first_resource = pd.Series(
                index=df[case_id_col].unique(),
                dtype="object",
                name="start_resource",
            )
        features_to_concat.append(first_resource)

    if include_mean_cost:
        if cost_col is not None and cost_col in df.columns:
            mean_cost = (
                df.groupby(case_id_col)[cost_col].mean().rename("mean_cost")
            )
        else:
            mean_cost = pd.Series(
                index=df[case_id_col].unique(),
                dtype="float",
                name="mean_cost",
            )
        features_to_concat.append(mean_cost)

    features = pd.concat(features_to_concat, axis=1)

    if include_extra_numeric_cols and extra_numeric_cols:
        extra_cols = [col for col in extra_numeric_cols if col in df.columns]
        if extra_cols:
            extra_numeric = df.groupby(case_id_col)[extra_cols].mean()
            features = pd.concat([features, extra_numeric], axis=1)

    builders = list(target_builders or [])
    if include_default_target:
        builders.insert(
            0,
            build_sla_target_builder(
                sla_hours=sla_hours,
                target_col=default_target_name,
                throughput_col=throughput_col,
            ),
        )

    features = _apply_target_builders(features, builders)

    if not include_throughput_feature and throughput_col in features.columns:
        features = features.drop(columns=[throughput_col])
    if drop_feature_cols:
        drop_cols = [c for c in drop_feature_cols if c in features.columns]
        if drop_cols:
            features = features.drop(columns=drop_cols)

    # padroniza o nome da coluna de identificador para uso posterior no split
    features = features.reset_index().rename(columns={case_id_col: "case_id"})

    return features
