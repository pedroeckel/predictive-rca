from __future__ import annotations

from typing import List

import numpy as np
import pandas as pd


def build_case_features(
    df_events: pd.DataFrame,
    sla_hours: float,
    case_id_col: str = "case_id",
    activity_col: str = "activity",
    timestamp_col: str = "timestamp",
    resource_col: str | None = "resource",
    cost_col: str | None = "cost",
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

    Returns
    -------
    pd.DataFrame
        Dataset em nível de caso, com coluna alvo `sla_violated`.
    """
    df = df_events.copy()

    if timestamp_col in df.columns:
        df[timestamp_col] = pd.to_datetime(df[timestamp_col])

    df = df.sort_values([case_id_col, timestamp_col])

    agg_time = df.groupby(case_id_col)[timestamp_col].agg(["min", "max"])
    agg_time["throughput_hours"] = (
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
            agg_time[["throughput_hours"]],
            num_events,
            num_unique_acts,
            rework_count,
            first_activity,
            last_activity,
            first_resource,
            mean_cost,
        ],
        axis=1,
    )

    features["sla_violated"] = (
        features["throughput_hours"] > sla_hours
    ).astype(int)

    features = features.reset_index().rename(columns={case_id_col: "case_id"})

    return features
