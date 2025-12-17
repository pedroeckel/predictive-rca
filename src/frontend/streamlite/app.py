# =========================================================
# ANÁLISE DE CAUSA RAIZ DE ATRASOS EM PROCESSOS (ACR)
# Streamlit — Página Única Linear
# =========================================================

from __future__ import annotations

from pathlib import Path
import sys
import tempfile
import warnings
from dataclasses import dataclass
from typing import Any, Optional

import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px

warnings.filterwarnings("ignore")

# =========================================================
# Configuração inicial
# =========================================================

ROOT_DIR = Path(__file__).resolve().parents[3]
if str(ROOT_DIR) not in sys.path:
    sys.path.append(str(ROOT_DIR))

from src.backend.pipeline.pipeline_builder import PipelineBuilder
from src.backend.models.lightgbm_model import LightGBMModel
from src.backend.models.xgboost_model import XGBoostModel
from src.backend.models.catboost_model import CatBoostModel
from src.backend.models.random_forest import RandomForestModel
from src.backend.models.logistic_regression import LogisticRegressionModel
from src.backend.evaluation.metrics import evaluate_binary_classification

from src.backend.preprocessing.eda import (
    check_required_columns,
    filter_cases_by_sla,
    filter_cases_by_boxplot,
    filter_cases_by_desired_date,
    calculate_frequency,
    final_trace_frequency,
    get_traces_with_highest_lead_time,
    get_transitions_with_highest_time,
    get_traces_grouped,
    get_transition_grouped,
)


@st.cache_data
def _corr_cached(df):
    """Calcula correlação com cache para melhor performance."""
    return df.corr()

# =========================================================
# UI: Page config + CSS
# =========================================================

st.set_page_config(
    page_title="ACR — Análise de Causa Raiz em Processos",
    page_icon="🧪",
    layout="wide",
)

st.markdown(
    """
<style>
.block-container { padding-top: 1.2rem; padding-bottom: 2.2rem; }

div[data-testid="stMetric"]{
  background: rgba(250,250,250,0.04);
  border: 1px solid rgba(255,255,255,0.08);
  padding: 14px 14px 10px 14px;
  border-radius: 14px;
}

div[data-testid="stDataFrame"]{
  border-radius: 14px;
  overflow: hidden;
}

.section-title{
  font-size: 1.05rem;
  font-weight: 700;
  margin: 1.1rem 0 .4rem 0;
}
.small-muted{
  font-size: .9rem;
  opacity: .75;
}
</style>
""",
    unsafe_allow_html=True,
)

# =========================================================
# Catálogo de modelos
# =========================================================

MODEL_CATALOG: dict[str, Any] = {
    "LightGBM (recomendado)": LightGBMModel,
    "XGBoost": XGBoostModel,
    "CatBoost": CatBoostModel,
    "Random Forest": RandomForestModel,
    "Regressão Logística (baseline)": LogisticRegressionModel,
}

# =========================================================
# Estruturas e utilidades
# =========================================================

@dataclass
class AppState:
    df_raw: Optional[pd.DataFrame] = None
    df_complete: Optional[pd.DataFrame] = None
    df_for_model: Optional[pd.DataFrame] = None

    model: Any = None
    artifacts: Any = None
    splits: Any = None
    pipeline: Any = None


def _get_state() -> AppState:
    if "acr_state" not in st.session_state:
        st.session_state["acr_state"] = AppState()
    return st.session_state["acr_state"]


state = _get_state()


def _normalize_colname(x: str) -> str:
    return str(x).strip().lower().replace("-", "_").replace(" ", "_")


def infer_column_any(df: pd.DataFrame, candidates: list[str]) -> Optional[str]:
    cols_norm = {_normalize_colname(c): c for c in df.columns}
    for cand in candidates:
        cand_norm = _normalize_colname(cand)
        if cand_norm in cols_norm:
            return cols_norm[cand_norm]
        if cand in df.columns:
            return cand
    return None


def safe_to_datetime(s: pd.Series) -> pd.Series:
    return pd.to_datetime(s, errors="coerce", utc=False)


def dataframe_quality_report(df: pd.DataFrame) -> pd.DataFrame:
    info_df = pd.DataFrame(
        {
            "coluna": df.columns,
            "tipo": df.dtypes.astype(str),
            "nulos": df.isnull().sum(),
            "% nulos": (df.isnull().mean() * 100).round(2),
            "n_unicos": [df[c].nunique(dropna=True) for c in df.columns],
        }
    )
    return info_df.sort_values(["% nulos", "n_unicos"], ascending=[False, False]).reset_index(drop=True)


def _period_text(df: pd.DataFrame) -> str:
    if "timestamp" not in df.columns:
        return "N/A"
    s = pd.to_datetime(df["timestamp"], errors="coerce")
    if s.isna().all():
        return "N/A"
    return f"{s.min().date()} → {s.max().date()}"


def kpi_row(df: pd.DataFrame) -> None:
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Linhas (eventos)", f"{len(df):,}".replace(",", "."))
    c2.metric("Casos (id_caso)", f"{df['id_caso'].nunique():,}".replace(",", ".") if "id_caso" in df.columns else "N/A")
    c3.metric("Atividades", f"{df['atividade'].nunique():,}".replace(",", ".") if "atividade" in df.columns else "N/A")
    c4.metric("Período", _period_text(df))


def _download_csv_button(df: pd.DataFrame, label: str, filename: str) -> None:
    csv_bytes = df.to_csv(index=False).encode("utf-8")
    st.download_button(
        label=label,
        data=csv_bytes,
        file_name=filename,
        mime="text/csv",
        use_container_width=True,
    )


def _drop_duplicate_columns(df: pd.DataFrame) -> pd.DataFrame:
    if df.columns.duplicated().any():
        dup = df.columns[df.columns.duplicated()].tolist()
        st.warning(f"Foram detectadas colunas duplicadas e serão removidas (mantendo a primeira ocorrência): {dup}")
        df = df.loc[:, ~df.columns.duplicated()].copy()
    return df


def _safe_apply_semantic_mapping(
    df: pd.DataFrame,
    *,
    col_case: str,
    col_act: str,
    col_time: str,
) -> pd.DataFrame:
    df = df.copy()
    df = _drop_duplicate_columns(df)

    semantic_targets = {"case_id", "activity", "timestamp"}
    selected = {col_case, col_act, col_time}

    for sem in semantic_targets:
        if sem in df.columns and sem not in selected:
            df = df.rename(columns={sem: f"{sem}_orig"})

    df = df.rename(
        columns={
            col_case: "case_id",
            col_act: "activity",
            col_time: "timestamp",
        }
    )

    df = _drop_duplicate_columns(df)
    df["timestamp"] = safe_to_datetime(df["timestamp"])
    return df


# =========================================================
# Header
# =========================================================

st.title("Análise de Causa Raiz em Processos (ACR)")
st.caption("Upload → Alvo → Indicadores → Modelagem → Explicabilidade")

st.markdown("---")

# =========================================================
# 1) Upload e validação do log de análise
# =========================================================

st.markdown('<div class="section-title">1. Upload e validação do log de análise</div>', unsafe_allow_html=True)

uploaded_file = st.file_uploader("Faça upload do CSV", type=["csv"])
if uploaded_file is None:
    st.stop()

df_raw = pd.read_csv(uploaded_file)
df_raw = _drop_duplicate_columns(df_raw)

if not check_required_columns(df_raw):
    st.error("Colunas obrigatórias ausentes.")
    st.stop()

df_raw["timestamp"] = safe_to_datetime(df_raw["timestamp"])
state.df_raw = df_raw

kpi_row(df_raw)

with st.expander("Prévia do log"):
    st.dataframe(df_raw.head(200), use_container_width=True)

st.markdown("---")

# =========================================================
# 2) Definição do atraso (variável alvo)
# =========================================================

st.markdown('<div class="section-title">2. Definição do atraso (variável alvo)</div>', unsafe_allow_html=True)

target_method = st.selectbox(
    "Método",
    ["SLA", "Boxplot de lead time (Q3)", "Data desejada"],
)

df_base = state.df_raw.copy()

if target_method == "SLA":
    sla = st.number_input("SLA (horas)", min_value=1, value=24)
    df_complete = filter_cases_by_sla(df_base, sla)
elif target_method == "Boxplot de lead time (Q3)":
    df_complete = filter_cases_by_boxplot(df_base)
else:
    df_complete = filter_cases_by_desired_date(df_base)

df_complete["target"] = df_complete["atraso"].astype(int)
state.df_complete = df_complete

aggregated_cases = df_complete.drop_duplicates(subset=['id_caso'])
target_counts = aggregated_cases["target"].value_counts()

st.bar_chart(target_counts)

st.markdown("---")

# =========================================================
# 3) Indicadores de Causa Raiz Geral
# =========================================================


st.markdown('<div class="section-title">3. Indicadores de Causa Raiz Geral</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="small-muted">Análises exploratórias integradas: qualidade, processo, gargalos e contrato semântico para o pipeline.</div>',
    unsafe_allow_html=True,
)

tabQ, tabP, tabG, tabM = st.tabs(
    ["Qualidade (EDA)", "Processo (EDA)", "Gargalos (atrasos)", "Mapeamento (Pipeline)"]
)

# --- (2) Qualidade ---
with tabQ:
    kpi_row(df_raw)
    tab1, tab2, tab3 = st.tabs(["Resumo", "Perfil de colunas", "Numéricos (opcional)"])

    with tab1:
        st.markdown("**Checklist de qualidade (sinais de risco)**")
        risk = []
        if "timestamp" in df_raw.columns and df_raw["timestamp"].isna().mean() > 0.01:
            risk.append("Timestamps inválidos > 1%")
        if "id_caso" in df_raw.columns and df_raw["id_caso"].isna().any():
            risk.append("Há casos com id_caso nulo")
        if "atividade" in df_raw.columns and df_raw["atividade"].isna().any():
            risk.append("Há eventos sem atividade")
        if len(risk) == 0:
            st.success("Não foram detectados riscos óbvios com base em regras simples.")
        else:
            st.warning("Riscos detectados:")
            for r in risk:
                st.write(f"- {r}")

    with tab2:
        info_df = dataframe_quality_report(df_raw)
        st.dataframe(info_df, use_container_width=True, height=520)

    with tab3:
        num_cols = df_raw.select_dtypes(include=np.number).columns.tolist()
        if not num_cols:
            st.info("Não há colunas numéricas para análise descritiva/correlação.")
        else:
            st.markdown("**Selecione colunas numéricas para análise** (evita poluir a UI).")
            cols_sel = st.multiselect("Colunas numéricas", num_cols, default=num_cols[: min(6, len(num_cols))])

            if cols_sel:
                st.subheader("Estatísticas descritivas")
                st.dataframe(df_raw[cols_sel].describe().T, use_container_width=True)

                st.subheader("Outliers (boxplots)")
                col_box = st.selectbox("Coluna para boxplot", cols_sel)
                st.plotly_chart(px.box(df_raw, y=col_box, title=f"Boxplot — {col_box}"), use_container_width=True)

                if len(cols_sel) >= 2:
                    st.subheader("Correlação (numéricas)")
                    corr = _corr_cached(df_raw[cols_sel])
                    st.plotly_chart(px.imshow(corr, text_auto=".2f", title="Matriz de correlação"), use_container_width=True)

# --- (3) Processo ---
with tabP:
    kpi_row(df_raw)

    with st.expander("Configurações de visualização", expanded=True):
        c1, c2, c3, c4 = st.columns(4)
        with c1:
            show_tables = st.toggle("Mostrar tabelas (Top-100)", value=True, key="tabP_show_tables")
        with c2:
            show_traces = st.toggle("Analisar variantes (trace)", value=("trace" in df_raw.columns), key="tabP_show_traces")
        with c3:
            show_trans = st.toggle("Analisar transições (transicao)", value=("transicao" in df_raw.columns), key="tabP_show_trans")
        with c4:
            top_k = st.number_input("Top N", min_value=5, max_value=100, value=20, step=5, key="tabP_top_k")

    col1, col2 = st.columns(2)

    with col1:
        if "atividade" in df_raw.columns:
            freq_act = calculate_frequency(df_raw, "atividade")
            st.plotly_chart(px.bar(freq_act.head(top_k), x="atividade", y="quantidade", title="Top atividades"), use_container_width=True)
            if show_tables:
                st.dataframe(freq_act.head(100), use_container_width=True, height=320)

    with col2:
        if "recurso" in df_raw.columns:
            freq_res = calculate_frequency(df_raw, "recurso")
            st.plotly_chart(px.bar(freq_res.head(top_k), x="recurso", y="quantidade", title="Top recursos"), use_container_width=True)
            if show_tables:
                st.dataframe(freq_res.head(100), use_container_width=True, height=320)

    if show_traces and "trace" in df_raw.columns:
        st.subheader("Variantes (traces)")
        trace_freq = calculate_frequency(df_raw, "trace")
        fig_trace = px.bar(trace_freq.head(top_k), x="trace", y="quantidade", title="Top traces")
        fig_trace.update_layout(xaxis_tickangle=45)
        st.plotly_chart(fig_trace, use_container_width=True)
        if show_tables:
            st.dataframe(trace_freq.head(100), use_container_width=True, height=320)

    if show_trans and "transicao" in df_raw.columns:
        st.subheader("Transições")
        trans_freq = calculate_frequency(df_raw, "transicao")
        fig_trans = px.bar(trans_freq.head(top_k), x="transicao", y="quantidade", title="Top transições")
        fig_trans.update_layout(xaxis_tickangle=45)
        st.plotly_chart(fig_trans, use_container_width=True)
        if show_tables:
            st.dataframe(trans_freq.head(100), use_container_width=True, height=320)

# --- (5) Gargalos: depende do alvo (df_complete) ---
with tabG:
    if state.df_complete is None:
        st.info("Defina o **alvo (atraso)** na seção 3 para habilitar esta aba de gargalos.")
    else:
        df_complete = state.df_complete
        st.markdown("**Análise focada nos casos atrasados**")

        df_delayed = df_complete[df_complete["target"] == 1].copy()
        if df_delayed.empty:
            st.warning("Nenhum caso atrasado foi identificado com o critério atual. Ajuste a definição do alvo.")
        else:
            unique_cases = df_delayed.drop_duplicates(subset=["id_caso"]) if "id_caso" in df_delayed.columns else df_delayed

            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Casos atrasados", int(unique_cases["id_caso"].nunique()) if "id_caso" in unique_cases.columns else len(unique_cases))
            c2.metric("Lead time médio (dias)", f"{unique_cases['lead_time'].mean():.2f}" if "lead_time" in unique_cases else "N/A")
            c3.metric("Lead time máx (dias)", f"{unique_cases['lead_time'].max():.2f}" if "lead_time" in unique_cases else "N/A")
            c4.metric("Lead time mín (dias)", f"{unique_cases['lead_time'].min():.2f}" if "lead_time" in unique_cases else "N/A")

            tabA, tabB, tabC, tabD = st.tabs(["Atividades/Recursos", "Traces/Transições", "Lead time extremo", "Dados (amostra)"])

            with tabA:
                if "atividade" in df_delayed.columns:
                    activity_freq = calculate_frequency(df_delayed, "atividade")
                    st.plotly_chart(px.bar(activity_freq.head(top_k), x="atividade", y="quantidade", title="Top atividades (atrasados)"), use_container_width=True)
                    st.dataframe(activity_freq.head(100), use_container_width=True, height=320)

                if "recurso" in df_delayed.columns:
                    resource_freq = calculate_frequency(df_delayed, "recurso")
                    st.plotly_chart(px.bar(resource_freq.head(top_k), x="recurso", y="quantidade", title="Top recursos (atrasados)"), use_container_width=True)
                    st.dataframe(resource_freq.head(100), use_container_width=True, height=320)

            with tabB:
                if "trace" in df_delayed.columns:
                    trace_freq = calculate_frequency(df_delayed, "trace")
                    fig = px.bar(trace_freq.head(top_k), x="trace", y="quantidade", title="Top traces (atrasados)")
                    fig.update_layout(xaxis_tickangle=45)
                    st.plotly_chart(fig, use_container_width=True)
                    st.dataframe(trace_freq.head(100), use_container_width=True, height=320)

                if "transicao" in df_delayed.columns:
                    transition_freq = calculate_frequency(df_delayed, "transicao")
                    fig = px.bar(transition_freq.head(top_k), x="transicao", y="quantidade", title="Top transições (atrasados)")
                    fig.update_layout(xaxis_tickangle=45)
                    st.plotly_chart(fig, use_container_width=True)
                    st.dataframe(transition_freq.head(100), use_container_width=True, height=320)

                if "trace" in df_delayed.columns and "gpm_nota" in df_delayed.columns:
                    final_trace_freq = final_trace_frequency(df_delayed)
                    gpm_freq = calculate_frequency(df_delayed, "gpm_nota")

                    c1, c2 = st.columns(2)
                    with c1:
                        fig = px.bar(final_trace_freq.head(top_k), x="trace final", y="quantidade", title="Top traces completos (atrasados)")
                        fig.update_layout(xaxis_tickangle=45)
                        st.plotly_chart(fig, use_container_width=True)
                        st.dataframe(final_trace_freq.head(100), use_container_width=True, height=320)

                    with c2:
                        fig = px.bar(gpm_freq.head(top_k), x="gpm_nota", y="quantidade", title="Top GPMs (atrasados)")
                        fig.update_layout(xaxis_tickangle=45)
                        st.plotly_chart(fig, use_container_width=True)
                        st.dataframe(gpm_freq.head(100), use_container_width=True, height=320)

            with tabC:
                if all(col in df_delayed.columns for col in ["trace", "lead_time"]):
                    top_traces = get_traces_with_highest_lead_time(df_delayed, top_n=top_k)
                    st.plotly_chart(px.bar(top_traces, x="trace final", y="lead time (dias)", title="Traces individuais com maior lead time"), use_container_width=True)
                    st.dataframe(top_traces, use_container_width=True, height=320)

                if all(col in df_delayed.columns for col in ["transicao", "tempo_atividade_execucao"]):
                    top_transitions = get_transitions_with_highest_time(df_delayed, top_n=top_k).dropna()
                    if "transição" in top_transitions.columns:
                        top_transitions = top_transitions[top_transitions["transição"].astype(str).str.lower() != "none"]

                    x_col = "transição" if "transição" in top_transitions.columns else top_transitions.columns[0]
                    y_col = "tempo entre atividades (dias)" if "tempo entre atividades (dias)" in top_transitions.columns else top_transitions.columns[1]
                    st.plotly_chart(px.bar(top_transitions, x=x_col, y=y_col, title="Transições individuais mais demoradas"), use_container_width=True)
                    st.dataframe(top_transitions, use_container_width=True, height=320)

                if all(c in df_delayed.columns for c in ["trace", "lead_time", "transicao", "tempo_atividade_execucao"]):
                    traces_grouped = get_traces_grouped(df_delayed, top_n=min(15, top_k))
                    transitions_grouped = get_transition_grouped(df_delayed, top_n=min(15, top_k))

                    c1, c2 = st.columns(2)
                    with c1:
                        if not traces_grouped.empty:
                            st.plotly_chart(px.bar(traces_grouped, x="trace", y="media_lead_time", title="Traces (média do lead time)"), use_container_width=True)
                    with c2:
                        if not traces_grouped.empty:
                            st.plotly_chart(px.bar(traces_grouped, x="trace", y="mediana_lead_time", title="Traces (mediana do lead time)"), use_container_width=True)
                    if not traces_grouped.empty:
                        st.dataframe(traces_grouped.head(100), use_container_width=True, height=320)

                    c3, c4 = st.columns(2)
                    with c3:
                        if not transitions_grouped.empty:
                            st.plotly_chart(px.bar(transitions_grouped, x="transicao", y="media_tempo", title="Transições (média do tempo)"), use_container_width=True)
                    with c4:
                        if not transitions_grouped.empty:
                            st.plotly_chart(px.bar(transitions_grouped, x="transicao", y="mediana_tempo", title="Transições (mediana do tempo)"), use_container_width=True)
                    if not transitions_grouped.empty:
                        st.dataframe(transitions_grouped.head(100), use_container_width=True, height=320)

            with tabD:
                st.dataframe(df_delayed.head(200), use_container_width=True, height=520)

            _download_csv_button(df_delayed, "Baixar CSV (apenas atrasados)", "acr_log_atrasados.csv")

# --- (6) Mapeamento: aplica-se APÓS alvo existir; e evita duplicação ---
with tabM:
    if state.df_complete is None:
        st.info("Defina o **alvo (atraso)** na seção 3. Depois, aplique o mapeamento aqui para habilitar a modelagem.")
    else:
        df_complete = state.df_complete

        st.markdown("**Contrato do pipeline**: selecionar quais colunas representam case/activity/timestamp.")
        st.markdown('<div class="small-muted">O mapeamento é aplicado sobre o dataset já rotulado (com target).</div>', unsafe_allow_html=True)

        # Sugestão automática (baseada no df_complete)
        REQUIRED_FOR_PIPELINE: dict[str, list[str]] = {
            "case_id": ["case_id", "id_caso", "case", "idcase", "caseid", "id caso"],
            "activity": ["activity", "atividade", "act", "evento", "event", "task", "atividade_nome"],
            "timestamp": ["timestamp", "time", "data_hora", "datetime", "datahora", "dt", "data_hora_evento"],
        }

        inferred_map = {
            semantic: infer_column_any(df_complete, candidates)
            for semantic, candidates in REQUIRED_FOR_PIPELINE.items()
        }

        cols = df_complete.columns.tolist()
        c1, c2, c3 = st.columns(3)
        with c1:
            col_case = st.selectbox(
                "Coluna para **case_id**",
                cols,
                index=cols.index(inferred_map["case_id"]) if inferred_map["case_id"] in cols else 0,
            )
        with c2:
            col_act = st.selectbox(
                "Coluna para **activity**",
                cols,
                index=cols.index(inferred_map["activity"]) if inferred_map["activity"] in cols else 0,
            )
        with c3:
            col_time = st.selectbox(
                "Coluna para **timestamp**",
                cols,
                index=cols.index(inferred_map["timestamp"]) if inferred_map["timestamp"] in cols else 0,
            )

        if len({col_case, col_act, col_time}) < 3:
            st.error("Uma mesma coluna foi usada para mais de um papel. Ajuste o mapeamento.")
        else:
            df_for_model = _safe_apply_semantic_mapping(
                df_complete,
                col_case=col_case,
                col_act=col_act,
                col_time=col_time,
            )

            # Garantias mínimas do pipeline
            missing = [c for c in ["case_id", "activity", "timestamp"] if c not in df_for_model.columns]
            if missing:
                st.error(f"Colunas semânticas ausentes para o pipeline: {missing}")
            else:
                state.df_for_model = df_for_model
                # Invalida modelo se alterar mapping
                state.model = None
                state.artifacts = None
                state.splits = None
                state.pipeline = None

                st.success("Contrato aplicado: **case_id / activity / timestamp** (sem colisões).")
                with st.expander("Prévia do dataset de modelagem", expanded=True):
                    max_rows_preview = st.slider(
                        "Linhas exibidas na prévia",
                        min_value=20,
                        max_value=500,
                        value=200,
                        step=20,
                        help="Controla quantas linhas são mostradas nesta amostra.",
                    )
                    st.dataframe(df_for_model.head(max_rows_preview), use_container_width=True, height=420)

                _download_csv_button(df_for_model, "Baixar CSV (para modelagem)", "acr_dataset_modelagem.csv")

st.markdown("---")

# =========================================================
# 4) Modelagem
# =========================================================

st.markdown('<div class="section-title">4. Modelagem preditiva</div>', unsafe_allow_html=True)
st.markdown('<div class="small-muted">Treine um classificador para estimar probabilidade de atraso.</div>', unsafe_allow_html=True)

if state.df_for_model is None:
    st.info("Aplique o **Mapeamento (Pipeline)** na seção 2 (aba Mapeamento) para habilitar a modelagem.")
    st.stop()

df_for_model = state.df_for_model

c1, c2, c3 = st.columns([2, 1, 1])
with c1:
    model_name = st.selectbox("Modelo", list(MODEL_CATALOG.keys()), index=0)
with c2:
    run_model = st.button("Treinar modelo", type="primary", use_container_width=True)
with c3:
    def _clear_model_state() -> None:
        state.model = None
        state.artifacts = None
        state.splits = None
        state.pipeline = None
        st.toast("Artefatos do modelo limpos. Re-treine.", icon="🧹")

    st.button("Re-treinar (limpar artefatos)", use_container_width=True, on_click=_clear_model_state)

if run_model:
    with st.status("Treinamento em execução...", expanded=True) as status:
        try:
            status.update(label="Serializando dataset temporário...", state="running")
            with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as tmp:
                df_for_model.to_csv(tmp.name, index=False)
                tmp_path = tmp.name

            status.update(label="Executando pipeline de features + treino...", state="running")
            pipeline = PipelineBuilder(model_class=MODEL_CATALOG[model_name])
            model, artifacts, splits = pipeline.run_from_event_log(tmp_path, target_col="target")

            state.model = model
            state.artifacts = artifacts
            state.splits = splits
            state.pipeline = pipeline

            status.update(label="Avaliando no conjunto de teste...", state="running")
            X_test = artifacts.preprocessor.transform(splits.X_test)
            y_test = splits.y_test.to_numpy()
            metrics = evaluate_binary_classification(model, X_test, y_test)

            status.update(label="Concluído.", state="complete")

            m1, m2 = st.columns(2)
            m1.metric("AUC (teste)", f"{metrics['auc']:.4f}")
            m2.metric("Acurácia (teste)", f"{metrics['accuracy']:.4f}")

            with st.expander("Relatório de classificação", expanded=True):
                st.text(metrics["classification_report"])

        except Exception as e:
            status.update(label="Falha no treinamento.", state="error")
            st.exception(e)

st.markdown("---")

# =========================================================
# 5) Explicabilidade
# =========================================================

st.markdown('<div class="section-title">5. Explicabilidade e análise de causa raiz (SHAP)</div>', unsafe_allow_html=True)
st.markdown('<div class="small-muted">Interprete como associação explicativa (não causalidade garantida).</div>', unsafe_allow_html=True)

if state.model is None or state.pipeline is None:
    st.info("Treine um modelo na seção 4 para habilitar a explicabilidade.")
    st.stop()

model = state.model
artifacts = state.artifacts
splits = state.splits
pipeline = state.pipeline

with st.spinner("Computando explicabilidade..."):
    explainability = pipeline.compute_explainability(
        model,
        artifacts,
        splits,
        dataset_split="test",
    )

shap_values = explainability.get("shap_values")
X_pre = explainability.get("X_pre")
feature_names = explainability.get("feature_names", [])

def _to_dense(arr):
    from scipy.sparse import issparse
    if issparse(arr):
        return arr.toarray()
    return np.asarray(arr)

def _select_shap_array(values):
    if values is None:
        return None
    if hasattr(values, "values"):  # shap.Explanation
        values = values.values
    if isinstance(values, list):
        idx = 1 if len(values) > 1 else 0
        return np.asarray(values[idx])
    if getattr(values, "ndim", 0) == 3:
        return np.asarray(values)[:, :, -1]
    return np.asarray(values)

def _resolve_feature_names(*, X_raw, X_pre_obj, preprocessor, provided, n_features: int) -> list[str]:
    if hasattr(X_pre_obj, "columns"):
        cols = list(X_pre_obj.columns)
        if len(cols) == n_features:
            return cols

    if preprocessor is not None and hasattr(preprocessor, "get_feature_names_out"):
        try:
            if X_raw is not None and hasattr(X_raw, "columns"):
                names = list(preprocessor.get_feature_names_out(X_raw.columns))
            else:
                names = list(preprocessor.get_feature_names_out())
            if len(names) == n_features:
                return names
        except Exception:
            pass

    if provided and isinstance(provided, (list, tuple)) and len(provided) == n_features:
        return list(provided)

    return [f"feature_{i}" for i in range(n_features)]

shap_matrix = _select_shap_array(shap_values)
if shap_matrix is None or X_pre is None:
    st.info("Não foi possível gerar SHAP para este modelo/dataset.")
    st.stop()

X_dense = _to_dense(X_pre)
if shap_matrix.shape[1] != X_dense.shape[1]:
    st.error(f"Inconsistência dimensional: SHAP={shap_matrix.shape[1]} vs X_pre={X_dense.shape[1]}.")
    st.stop()

X_raw_test = splits.X_test if isinstance(getattr(splits, "X_test", None), pd.DataFrame) else None
feature_names = _resolve_feature_names(
    X_raw=X_raw_test,
    X_pre_obj=X_pre,
    preprocessor=getattr(artifacts, "preprocessor", None),
    provided=feature_names,
    n_features=X_dense.shape[1],
)

tabG, tabD, tabL = st.tabs(["Global", "Dependência", "Local"])

with tabG:
    st.subheader("Importância global das features")
    importance = np.mean(np.abs(shap_matrix), axis=0)
    imp_df = pd.DataFrame({"feature": feature_names, "importance": importance}).sort_values("importance", ascending=False)

    k = st.slider("Top features", 5, min(80, len(imp_df)), min(20, len(imp_df)))
    imp_top = imp_df.head(k)

    st.plotly_chart(
        px.bar(
            imp_top,
            x="feature",
            y="importance",
            labels={"importance": "mean(|SHAP|)", "feature": "Feature"},
            title=f"Top-{k} — Importância global (mean(|SHAP|))",
        ),
        use_container_width=True,
    )

    with st.expander("Tabela (global)", expanded=False):
        st.dataframe(imp_df.head(200), use_container_width=True, height=420)

with tabD:
    st.subheader("Dependência SHAP (feature → impacto)")
    shortlist = imp_df.head(min(50, len(imp_df)))["feature"].tolist()
    feature_dep = st.selectbox("Selecione a feature", shortlist, index=0)
    dep_idx = feature_names.index(feature_dep)

    st.plotly_chart(
        px.scatter(
            x=X_dense[:, dep_idx],
            y=shap_matrix[:, dep_idx],
            labels={"x": feature_dep, "y": "SHAP"},
            title=f"Dependência — {feature_dep}",
        ),
        use_container_width=True,
    )

    with st.expander("Resumo estatístico", expanded=False):
        s = pd.Series(shap_matrix[:, dep_idx], name="shap")
        st.write(s.describe())

with tabL:
    st.subheader("Explicação local (instância individual)")
    idx = st.slider("Índice do caso (teste)", 0, X_dense.shape[0] - 1, 0)

    local_df = pd.DataFrame({"feature": feature_names, "shap": shap_matrix[idx]})
    local_df["abs"] = local_df["shap"].abs()
    local_df = local_df.sort_values("abs", ascending=False)

    k_local = st.slider("Top features (local)", 5, min(60, len(local_df)), 15)
    local_top = local_df.head(k_local).copy()

    st.plotly_chart(
        px.bar(
            local_top.sort_values("shap"),
            x="shap",
            y="feature",
            orientation="h",
            labels={"shap": "SHAP", "feature": "Feature"},
            title=f"Explicação local — Top-{k_local} (índice {idx})",
        ),
        use_container_width=True,
    )

    with st.expander("Tabela (local)", expanded=False):
        st.dataframe(local_df.head(200), use_container_width=True, height=420)
