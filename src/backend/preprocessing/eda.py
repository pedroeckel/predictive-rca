from typing import Tuple, List, Dict, Any
import pandas as pd

def check_required_columns(df: pd.DataFrame) -> bool:
    required_columns = [
        'id_caso',
        'atividade', 
        'timestamp',
        'abreviacao_status',
        'sequenciamento',
        'recurso',
        'gpm_nota',
        'prioridade_nota',
        'cod_finalidade_nota',
        'tipo_intervencao_nota',
        'data_fim_desejado_nota'
    ]
    
    missing_cols = []
    
    for col in required_columns:
        if col not in df.columns:
            missing_cols.append(col)
    
    if missing_cols:
        return False
    
    return True

def generate_trace_list(
    df: pd.DataFrame,
    case_id_col: str = 'id_caso',
    timestamp_col: str = 'timestamp',
    status_col: str = 'abreviacao_status',
    sequencing_col: str = 'sequenciamento',
) -> pd.DataFrame:

    if df.empty:
        df = df.copy()
        df['trace_list'] = []
        df['trace'] = []
        return df
    
    df = df.copy()

    try:
        df[timestamp_col] = pd.to_datetime(df[timestamp_col])
    except Exception as e:
        raise ValueError(f'Erro ao converter datas: {e}')
    
    df = df.sort_values([case_id_col, timestamp_col, sequencing_col])

    all_sequences = []
    current_activities = []
    current_case = None
    
    for i in range(len(df)):
        case_id = df[case_id_col].iloc[i]
        activity = df[status_col].iloc[i]

        if case_id == current_case:
            current_activities.append(activity)
        else:
            current_case = case_id
            current_activities = [activity]

        all_sequences.append(current_activities.copy())

    df['trace_list'] = all_sequences
    df['trace'] = df['trace_list'].apply(lambda x: ' -> '.join(map(str, x)))

    return df


def generate_transitions(
    df: pd.DataFrame,
    case_id_col: str = 'id_caso',
    status_col: str = 'abreviacao_status',
    timestamp_col: str = 'timestamp',
    sequencing_col: str = 'sequenciamento',
) -> pd.DataFrame:
    
    df = df.sort_values([case_id_col, timestamp_col, sequencing_col])

    if df.empty:
        df = df.copy()
        df['transicao'] = None
        return df
    
    df = df.copy()
    next_status = df.groupby(case_id_col, sort=False)[status_col].shift(-1)
    mask_valid = next_status.notna()
    df['transicao'] = None
    df.loc[mask_valid, 'transicao'] = (
        df.loc[mask_valid, status_col] + ' -> ' + next_status[mask_valid]
    )
    
    return df

def add_time_between_activities(
    df: pd.DataFrame,
    case_id_col: str = 'id_caso',
    timestamp_col: str = 'timestamp',
    sequencing_col: str = 'sequenciamento'
) -> pd.DataFrame:
    
    df = df.sort_values([case_id_col, timestamp_col, sequencing_col])

    if df.empty:
        df = df.copy()
        df['tempo_atividade_execucao'] = None
        return df
    
    df = df.copy()

    next_timestamp = df.groupby(case_id_col, sort=False)[timestamp_col].shift(-1)
    df['tempo_atividade_execucao'] = (next_timestamp - df[timestamp_col]).dt.total_seconds() / 86400
    mask_last = df.groupby(case_id_col, sort=False).cumcount(ascending=False) == 0
    df.loc[mask_last, 'tempo_atividade_execucao'] = 0
    
    return df


def add_lead_time(
    df: pd.DataFrame,
    case_id_col: str = 'id_caso',
    timestamp_col: str = 'timestamp',
    sequencing_col: str = 'sequenciamento'
) -> pd.DataFrame:
    
    df = df.sort_values([case_id_col, timestamp_col, sequencing_col])

    if df.empty:
        df = df.copy()
        df['lead_time'] = None
        return df
    
    df = df.copy()

    case_start = df.groupby(case_id_col)[timestamp_col].transform('min')
    case_end = df.groupby(case_id_col)[timestamp_col].transform('max')
    df['lead_time'] = (case_end - case_start).dt.total_seconds() / 86400

    return df

def filter_cases_by_sla(
    df: pd.DataFrame,
    sla_hours: float,
    case_id_col: str = 'id_caso',
    timestamp_col: str = 'timestamp',
    sequencing_col: str = 'sequenciamento'
) -> pd.DataFrame:
    
    if df.empty:
        return pd.DataFrame()
    
    df = df.copy()

    try:
        df[timestamp_col] = pd.to_datetime(df[timestamp_col])
    except Exception as e:
        raise ValueError(f'Erro ao converter datas: {e}')
    
    df_trace = generate_trace_list(df)
    df_transition = generate_transitions(df_trace)
    df_time = add_time_between_activities(df_transition)
    df_all_cases = add_lead_time(df_time)
    
    unique_cases = df_all_cases.drop_duplicates(subset=[case_id_col])[[case_id_col, 'lead_time']]
    sla_days = sla_hours / 24
    
    atraso_map = {}
    tempo_atraso_map = {}
    
    for _, row in unique_cases.iterrows():
        case_id = row[case_id_col]
        lead_time = row['lead_time']
        atraso = 1 if lead_time > sla_days else 0
        atraso_map[case_id] = atraso
        tempo_atraso_map[case_id] = max(0, lead_time - sla_days) if atraso == 1 else 0
    
    df_all_cases['atraso'] = df_all_cases[case_id_col].map(atraso_map)
    df_all_cases['tempo_atraso'] = df_all_cases[case_id_col].map(tempo_atraso_map)
    
    df_all_cases = df_all_cases.sort_values([case_id_col, timestamp_col, sequencing_col]).reset_index(drop=True)
    
    return df_all_cases


def filter_cases_by_boxplot(
    df: pd.DataFrame,
    case_id_col: str = 'id_caso',
    timestamp_col: str = 'timestamp',
    sequencing_col: str = 'sequenciamento'
) -> pd.DataFrame:
    
    if df.empty:
        return pd.DataFrame()
    
    df = df.copy()

    try:
        df[timestamp_col] = pd.to_datetime(df[timestamp_col])
    except Exception as e:
        raise ValueError(f'Erro ao converter datas: {e}')

    df_trace = generate_trace_list(df)
    df_transition = generate_transitions(df_trace)
    df_time = add_time_between_activities(df_transition)
    df_all_cases = add_lead_time(df_time)
    
    unique_cases = df_all_cases.drop_duplicates(subset=[case_id_col])[[case_id_col, 'lead_time']]
    Q1 = unique_cases['lead_time'].quantile(0.25)
    Q3 = unique_cases['lead_time'].quantile(0.75)
    IQR = Q3 - Q1
    upper_bound = Q3 + 1.5 * IQR
    
    atraso_map = {}
    tempo_atraso_map = {}
    
    for _, row in unique_cases.iterrows():
        case_id = row[case_id_col]
        lead_time = row['lead_time']
        atraso = 1 if lead_time > upper_bound else 0
        atraso_map[case_id] = atraso
        tempo_atraso_map[case_id] = max(0, lead_time - upper_bound) if atraso == 1 else 0
    
    df_all_cases['atraso'] = df_all_cases[case_id_col].map(atraso_map)
    df_all_cases['tempo_atraso'] = df_all_cases[case_id_col].map(tempo_atraso_map)
    df_all_cases['limite_boxplot'] = upper_bound
    
    df_all_cases = df_all_cases.sort_values([case_id_col, timestamp_col, sequencing_col]).reset_index(drop=True)
    
    return df_all_cases


def filter_cases_by_desired_date(
    df: pd.DataFrame,
    case_id_col: str = 'id_caso',
    timestamp_col: str = 'timestamp',
    desired_end_col: str = 'data_fim_desejado_nota',
    sequencing_col: str = 'sequenciamento'
) -> pd.DataFrame:

    if df.empty:
        return pd.DataFrame()
        
    df = df.copy()

    try:
        df[timestamp_col] = pd.to_datetime(df[timestamp_col])
        df[desired_end_col] = pd.to_datetime(df[desired_end_col], errors='coerce')
    except Exception as e:
        raise ValueError(f'Erro ao converter datas: {e}')

    unique_cases = df.groupby(case_id_col).agg({
        timestamp_col: 'max',
        desired_end_col: 'max'
    }).reset_index()
    
    unique_cases['atraso'] = (
        unique_cases[timestamp_col] > unique_cases[desired_end_col]
    ).fillna(False).astype(int)
    
    unique_cases['tempo_atraso'] = (
        unique_cases[timestamp_col] - unique_cases[desired_end_col]
    ).dt.total_seconds() / 86400
    
    unique_cases.loc[unique_cases['atraso'] == 0, 'tempo_atraso'] = 0
    unique_cases.loc[unique_cases['tempo_atraso'] < 0, 'tempo_atraso'] = 0
    
    atraso_map = dict(zip(unique_cases[case_id_col], unique_cases['atraso']))
    tempo_atraso_map = dict(zip(unique_cases[case_id_col], unique_cases['tempo_atraso']))
    
    df_trace = generate_trace_list(df)
    df_transition = generate_transitions(df_trace)
    df_time = add_time_between_activities(df_transition)
    df_all_cases = add_lead_time(df_time)
    
    df_all_cases['atraso'] = df_all_cases[case_id_col].map(atraso_map).fillna(0).astype(int)
    df_all_cases['tempo_atraso'] = df_all_cases[case_id_col].map(tempo_atraso_map).fillna(0)
    
    df_all_cases = df_all_cases.sort_values([case_id_col, timestamp_col, sequencing_col]).reset_index(drop=True)
    
    return df_all_cases

def calculate_frequency(
    df: pd.DataFrame,
    column: str
) -> pd.DataFrame:
    
    freq = df[column].value_counts()
    result = pd.DataFrame({
        column: freq.index,
        'quantidade': freq.values,
    })
    return result


def final_trace_frequency(
    df: pd.DataFrame,
    case_id_col: str = 'id_caso',
    trace_col: str = 'trace'
) -> pd.DataFrame:
    
    final_traces = df.groupby(case_id_col, sort=False)[trace_col].last().value_counts()
    result = pd.DataFrame({
        'trace final': final_traces.index,
        'quantidade': final_traces.values,
    })
    return result

    

def gpm_case_frequency(
    df: pd.DataFrame, 
    case_id_col: str = 'id_caso',
    gpm_col: str = 'gpm_nota'
) -> pd.DataFrame:

    gpm_by_case = df.groupby(case_id_col, sort=False)[gpm_col].apply(
        lambda x: x.dropna().unique()
    ).explode().dropna()
    
    freq = gpm_by_case.value_counts()
    return pd.DataFrame({
        'gpm_nota': freq.index,
        'casos': freq.values
    })


def priority_case_frequency(
    df: pd.DataFrame, 
    case_id_col: str = 'id_caso',
    priority_col: str = 'prioridade_nota'
) -> pd.DataFrame:

    priority_by_case = df.groupby(case_id_col, sort=False)[priority_col].apply(
        lambda x: x.dropna().unique()
    ).explode().dropna()
    
    freq = priority_by_case.value_counts()
    return pd.DataFrame({
        'prioridade_nota': freq.index,
        'casos': freq.values
    })


def intervention_case_frequency(
    df: pd.DataFrame, 
    case_id_col: str = 'id_caso',
    intervention_col: str = 'tipo_intervencao_nota'
) -> pd.DataFrame:

    intervention_by_case = df.groupby(case_id_col, sort=False)[intervention_col].apply(
        lambda x: x.dropna().unique()
    ).explode().dropna()
    
    freq = intervention_by_case.value_counts()
    result = pd.DataFrame({
        'tipo_intervencao_nota': freq.index,
        'casos': freq.values
    })
    return result


def purpose_case_frequency(
    df: pd.DataFrame, 
    case_id_col: str = 'id_caso',
    purpose_col: str = 'cod_finalidade_nota'
) -> pd.DataFrame:

    purpose_by_case = df.groupby(case_id_col, sort=False)[purpose_col].apply(
        lambda x: x.dropna().unique()
    ).explode().dropna()
    
    freq = purpose_by_case.value_counts()
    result = pd.DataFrame({
        'cod_finalidade_nota': freq.index,
        'casos': freq.values
    })
    return result


def get_traces_with_highest_lead_time(
    df: pd.DataFrame,
    case_id_col: str = 'id_caso',
    trace_col: str = 'trace',
    timestamp_col: str = 'timestamp',
    lead_time_col: str = 'lead_time',
    sequencing_col: str = 'sequenciamento',
    top_n: int = 30
) -> pd.DataFrame:
    
    df_sorted = df.sort_values([case_id_col, timestamp_col, sequencing_col])

    df_final_traces = (
        df_sorted
        .groupby(case_id_col, sort=False)
        .agg({
            trace_col: 'last',
            lead_time_col: 'first' 
        })
        .reset_index()
    )
    result = (
        df_final_traces
        .sort_values(lead_time_col, ascending=False)
        .head(top_n)
        [[trace_col, lead_time_col]]
        .reset_index(drop=True)
    )
    
    result.columns = ['trace final', 'lead time (dias)']
    return result


def get_transitions_with_highest_time(
    df: pd.DataFrame,
    transition_col: str = 'transicao',
    time_col: str = 'tempo_atividade_execucao',
    top_n: int = 30
) -> pd.DataFrame:
    
    result = (
        df
        .dropna(subset=[time_col])
        .sort_values(time_col, ascending=False)
        .head(top_n)
        [[transition_col, time_col]]
        .reset_index(drop=True)
    )
    
    result.columns = ['transição', 'tempo entre atividades (dias)']
    return result


def get_traces_grouped(
    df: pd.DataFrame,
    case_id_col: str = 'id_caso',
    timestamp_col: str = 'timestamp',
    sequencing_col: str = 'sequenciamento',
    trace_col: str = 'trace',
    lead_time_col: str = 'lead_time',
    top_n: int = 30
) -> pd.DataFrame:
    
    if df.empty or trace_col not in df.columns or lead_time_col not in df.columns:
        return pd.DataFrame()
    
    final_traces = (
        df
        .sort_values([case_id_col, timestamp_col, sequencing_col])
        .groupby(case_id_col, sort=False)
        .agg({
            trace_col: 'last',
            lead_time_col: 'first'
        })
        .reset_index()
    )
    
    trace_stats = (
        final_traces
        .groupby(trace_col)
        .agg({
            lead_time_col: ['mean', 'median']
        })
        .reset_index()
    )
    
    trace_stats.columns = [
        'trace',
        'media_lead_time',
        'mediana_lead_time'
    ]
    
    result = (
        trace_stats
        .sort_values('media_lead_time', ascending=False)
        .head(top_n)
        .reset_index(drop=True)
    )
    
    return result


def get_transition_grouped(
    df: pd.DataFrame,
    transition_col: str = 'transicao',
    time_col: str = 'tempo_atividade_execucao',
    top_n: int = 30
) -> pd.DataFrame:
    
    if df.empty or transition_col not in df.columns or time_col not in df.columns:
        return pd.DataFrame()
    
    valid_transitions = df.dropna(subset=[transition_col, time_col]).copy()
    
    if valid_transitions.empty:
        return pd.DataFrame()
    
    transition_stats = (
        valid_transitions
        .groupby(transition_col)
        .agg({
            time_col: ['mean', 'median']
        })
        .reset_index()
    )
    
    transition_stats.columns = [
        'transicao',
        'media_tempo',
        'mediana_tempo'
    ]
    
    result = (
        transition_stats
        .sort_values('media_tempo', ascending=False)
        .head(top_n)
        .reset_index(drop=True)
    )
    
    return result