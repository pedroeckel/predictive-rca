from typing import Tuple, List, Dict, Any
import pandas as pd

def check_required_columns(
    df: pd.DataFrame
) -> bool:

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
        print(f"Colunas faltando: {', '.join(missing_cols)}")
        return False
    
    print('Todas as colunas necessárias estão presentes')
    return True


def preprocess_delayed_cases(
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
        df[desired_end_col] = pd.to_datetime(df[desired_end_col])
    except Exception as e:
        raise ValueError(f'Erro ao converter datas: {e}')

    last_timestamp = df.groupby(case_id_col, sort=False)[timestamp_col].max().reset_index()
    desired_end = df.groupby(case_id_col, sort=False)[desired_end_col].first().reset_index()

    df_cases = last_timestamp.merge(desired_end, on=case_id_col)
    df_cases['atraso'] = (df_cases[timestamp_col] > df_cases[desired_end_col]).astype(int)

    delayed_case_ids = df_cases[df_cases['atraso'] == 1][case_id_col].unique().tolist()

    if not delayed_case_ids:
        return pd.DataFrame()

    df_delayed = df[df[case_id_col].isin(delayed_case_ids)].copy()
    df_delayed = df_delayed.sort_values([case_id_col, timestamp_col, sequencing_col]).reset_index(drop=True)

    return df_delayed


def generate_trace_list(
    df: pd.DataFrame,
    case_id_col: str = 'id_caso',
    timestamp_col: str = 'timestamp',
    status_col: str = 'abreviacao_status',
    sequencing_col: str = 'sequenciamento',
) -> pd.DataFrame:

    if df.empty:
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
    df['trace'] = df['trace_list'].apply(lambda x: ' -> '.join(x))

    return df


def generate_transitions(
    df: pd.DataFrame,
    case_id_col: str = 'id_caso',
    status_col: str = 'abreviacao_status',
    timestamp_col: str = 'timestamp',
    sequencing_col: str = 'sequenciamento',
) -> pd.DataFrame:
    
    if df.empty:
        df['transicao'] = None
        return df
    
    df = df.copy()
    df = df.sort_values([case_id_col, timestamp_col, sequencing_col])
    next_status = df.groupby(case_id_col)[status_col].shift(-1)
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
    
    if df.empty:
        df['tempo_atividade_execucao'] = None
        return df
    
    df = df.copy()
    df = df.sort_values([case_id_col, timestamp_col, sequencing_col])
    next_timestamp = df.groupby(case_id_col)[timestamp_col].shift(-1)
    df['tempo_atividade_execucao'] = (next_timestamp - df[timestamp_col]).dt.total_seconds() / 86400
    mask_last = df.groupby(case_id_col).cumcount(ascending=False) == 0
    df.loc[mask_last, 'tempo_atividade_execucao'] = None
    
    return df


def add_lead_time(
    df: pd.DataFrame,
    case_id_col: str = 'id_caso',
    timestamp_col: str = 'timestamp',
    sequencing_col: str = 'sequenciamento'
) -> pd.DataFrame:
    
    if df.empty:
        df['lead_time'] = None
        return df
    
    df = df.copy()
    df = df.sort_values([case_id_col, timestamp_col, sequencing_col])
    case_start = df.groupby(case_id_col)[timestamp_col].transform('min')
    case_end = df.groupby(case_id_col)[timestamp_col].transform('max')
    df['lead_time'] = (case_end - case_start).dt.total_seconds() / 86400

    return df


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
    
    final_traces = df.groupby(case_id_col)[trace_col].last().value_counts()
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

    gpm_by_case = df.groupby(case_id_col)[gpm_col].apply(
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

    priority_by_case = df.groupby(case_id_col)[priority_col].apply(
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

    intervention_by_case = df.groupby(case_id_col)[intervention_col].apply(
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

    purpose_by_case = df.groupby(case_id_col)[purpose_col].apply(
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
        .groupby(case_id_col)
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


def pipeline(df: pd.DataFrame) -> pd.DataFrame:
    
    if not check_required_columns(df):
        return pd.DataFrame()
    
    df_delayed = preprocess_delayed_cases(df)
    
    if df_delayed.empty:
        print('Nenhum caso atrasado encontrado')
        return pd.DataFrame()
    
    df_traces = generate_trace_list(df_delayed)
    df_transitions = generate_transitions(df_traces)
    df_times = add_time_between_activities(df_transitions)
    df_final = add_lead_time(df_times)
    
    print('\nFREQUÊNCIA DE ATIVIDADES:')
    activity_freq = calculate_frequency(df_final, 'atividade')
    print(activity_freq.to_string(index=False))
    
    print('\nFREQUÊNCIA DE TRANSIÇÕES:')
    transition_freq = calculate_frequency(df_final, 'transicao')
    print(transition_freq.to_string(index=False))
    
    print('\nFREQUÊNCIA DE TRACES:')
    trace_freq = calculate_frequency(df_final, 'trace')
    print(trace_freq.to_string(index=False))
    
    print('\nFREQUÊNCIA DE TRACES FINAIS:')
    final_trace_freq = final_trace_frequency(df_final)
    print(final_trace_freq.to_string(index=False))
    
    print('\nFREQUÊNCIA DE RECURSOS:')
    resource_freq = calculate_frequency(df_final, 'recurso')
    print(resource_freq.to_string(index=False))
    
    print('\nFREQUÊNCIA DE GPM (LINHA):')
    gpm_line_freq = calculate_frequency(df_final, 'gpm_nota')
    print(gpm_line_freq.to_string(index=False))
    
    print('\nFREQUÊNCIA DE GPM (CASO):')
    gpm_case_freq = gpm_case_frequency(df_final)
    print(gpm_case_freq.to_string(index=False))
    
    print('\nFREQUÊNCIA DE PRIORIDADE (LINHA):')
    priority_line_freq = calculate_frequency(df_final, 'prioridade_nota')
    print(priority_line_freq.to_string(index=False))
    
    print('\nFREQUÊNCIA DE PRIORIDADE (CASO):')
    priority_case_freq = priority_case_frequency(df_final)
    print(priority_case_freq.to_string(index=False))
    
    print('\nFREQUÊNCIA DE TIPO INTERVENÇÃO (LINHA):')
    intervention_line_freq = calculate_frequency(df_final, 'tipo_intervencao_nota')
    print(intervention_line_freq.to_string(index=False))
    
    print('\nFREQUÊNCIA DE TIPO INTERVENÇÃO (CASO):')
    intervention_case_freq = intervention_case_frequency(df_final)
    print(intervention_case_freq.to_string(index=False))
    
    print('\nFREQUÊNCIA DE CÓDIGO FINALIDADE (LINHA):')
    purpose_line_freq = calculate_frequency(df_final, 'cod_finalidade_nota')
    print(purpose_line_freq.to_string(index=False))
    
    print('\nFREQUÊNCIA DE CÓDIGO FINALIDADE (CASO):')
    purpose_case_freq = purpose_case_frequency(df_final)
    print(purpose_case_freq.to_string(index=False))
    
    print('\nTRACES COM MAIOR LEAD TIME:')
    top_lead_traces = get_traces_with_highest_lead_time(df_final)
    print(top_lead_traces.to_string(index=False))
    
    print('\nTRANSIÇÕES MAIS DEMORADAS:')
    top_transitions = get_transitions_with_highest_time(df_final)
    print(top_transitions.to_string(index=False))
    
    return df_final


df = pd.read_csv(r'C:\Users\Helton\Documents\Análise de Causa Raiz\predictive-rca\data\raw\csv\log_test3.csv')
processed_df = pipeline(df)
