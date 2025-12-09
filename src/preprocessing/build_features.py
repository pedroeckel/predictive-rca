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
    case_id_col: str = "id_caso",
    activity_col: str = "atividade",
    timestamp_col: str = "timestamp",
    resource_col: str | None = "resource",
    cost_col: str | None = "cost",
    target_builders: Iterable[TargetBuilder] | None = None,
    include_default_target: bool = True,
    default_target_name: str = "sla_violated",
    throughput_col: str = "throughput_hours",
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

    num_events = df.groupby(case_id_col).size().rename("num_events")
    num_unique_acts = (
        df.groupby(case_id_col)[activity_col]
        .nunique()
        .rename("num_unique_activities")
    )
    rework_count = (num_events - num_unique_acts).rename("rework_count")

    first_activity = (
        df.groupby(case_id_col).first()[activity_col].rename("start_activity")
    )
    last_activity = (
        df.groupby(case_id_col).last()[activity_col].rename("end_activity")
    )

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

    features = pd.concat(
        [
            agg_time[[throughput_col]],
            num_events,
            num_unique_acts,
            rework_count,
            first_activity,
            last_activity,
            first_resource,
        ],
        axis=1,
    )

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

    features = features.reset_index().rename(columns={case_id_col: "id_caso"})

    return features
