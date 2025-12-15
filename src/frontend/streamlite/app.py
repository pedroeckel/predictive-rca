from pathlib import Path
import sys

# Garantir que o diretório raiz do projeto esteja no sys.path para resolver `src.*`
ROOT_DIR = Path(__file__).resolve().parents[3]
if str(ROOT_DIR) not in sys.path:
    sys.path.append(str(ROOT_DIR))

import streamlit as st
import pandas as pd
from src.backend.pipeline.pipeline_builder import PipelineBuilder
from src.backend.models.lightgbm_model import LightGBMModel
from src.backend.models.xgboost_model import XGBoostModel
from src.backend.models.catboost_model import CatBoostModel
from src.backend.models.random_forest import RandomForestModel
from src.backend.models.logistic_regression import LogisticRegressionModel
import numpy as np
import plotly.express as px
from src.backend.preprocessing.eda import (
            check_required_columns,
            filter_cases_by_sla,
            filter_cases_by_boxplot,
            filter_cases_by_desired_date,
            calculate_frequency,
            final_trace_frequency,
            gpm_case_frequency,
            priority_case_frequency,
            intervention_case_frequency,
            purpose_case_frequency,
            generate_trace_list,
            generate_transitions,
            add_lead_time,
            add_time_between_activities,
            get_traces_with_highest_lead_time,
            get_transitions_with_highest_time,
            get_traces_grouped,
            get_transition_grouped
        )

st.set_page_config(layout='wide')

st.title('ACR')
st.markdown('---')

uploaded_file = st.file_uploader('Faça upload do CSV', type=['csv'])

if uploaded_file is not None:
    try:
        df = pd.read_csv(uploaded_file)
        
        if not check_required_columns(df):
            st.error('O arquivo não contém todas as colunas necessárias')
            required_columns = [
                'id_caso', 'atividade', 'timestamp', 'abreviacao_status',
                'sequenciamento', 'recurso', 'gpm_nota', 'prioridade_nota',
                'cod_finalidade_nota', 'tipo_intervencao_nota', 'data_fim_desejado_nota'
            ]
            
            st.write('Colunas necessárias:')
            for col in required_columns:
                if col in df.columns:
                    st.write(f'{col}: OK')
                else:
                    st.write(f'{col}: FALTANDO')
            
            st.write('Colunas presentes no arquivo:')
            st.write(list(df.columns))
            st.stop()
        
        st.session_state.df = df
        st.success(f'Arquivo carregado: {uploaded_file.name}')
        
    except Exception as e:
        st.error(f'Erro ao carregar arquivo: {e}')

st.subheader('Definição de casos atrasados')

target_method = st.selectbox(
    'Método para definir casos atrasados:',
    options=[
        'SLA',
        'Boxplot de lead time (acima do 3º quartil)', 
        'Em relação ao desejado (data_fim_desejado_nota)',
    ],
    help='Escolha como identificar os casos considerados atrasados'
)

st.markdown('---')
st.header('Indicadores de atrasos gerais')

if 'df' in st.session_state:
    df = st.session_state.df

    if target_method == 'SLA':
        sla_value = st.number_input(
            'Digite o valor do SLA (em horas):',
            min_value=1,
            max_value=1000000,
            value=24,
            help='Casos com lead time maior que este valor serão considerados atrasados'
        )
        df_complete = filter_cases_by_sla(df, sla_value)
        df_delayed = df_complete[df_complete['atraso'] == 1].copy()
        
    elif target_method == 'Boxplot de lead time (acima do 3º quartil)':
        df_complete = filter_cases_by_boxplot(df)
        df_delayed = df_complete[df_complete['atraso'] == 1].copy()

    elif target_method == 'Em relação ao desejado (data_fim_desejado_nota)':  
        df_complete = filter_cases_by_desired_date(df)
        df_delayed = df_complete[df_complete['atraso'] == 1].copy()

    if df_delayed.empty:
        st.warning(f'Nenhum caso atrasado encontrado com o método: {target_method}')

    else:
        col1, col2, col3, col4 = st.columns(4)

        with col1:
            total_cases = df_delayed['id_caso'].nunique()
            st.metric('Total de casos atrasados', total_cases)

        if 'lead_time' in df_delayed.columns:
            unique_cases = df_delayed.drop_duplicates(subset=['id_caso'])
            has_lead_time = True
        else:
            has_lead_time = False

        with col2:
            if has_lead_time:
                lead_time_mean = unique_cases['lead_time'].mean()
                st.metric('Lead time médio (dias)', f'{lead_time_mean:.2f}')
            else:
                st.metric('Lead time médio', 'N/A')

        with col3:
            if has_lead_time:
                lead_time_max = unique_cases['lead_time'].max()
                st.metric('Lead time máximo (dias)', f'{lead_time_max:.2f}')
            else:
                st.metric('Lead time máximo', 'N/A')

        with col4:
            if has_lead_time:
                lead_time_min = unique_cases['lead_time'].min()
                st.metric('Lead time mínimo (dias)', f'{lead_time_min:.2f}')
            else:
                st.metric('Lead time mínimo', 'N/A')
        
        col5, col6, col7, col8 = st.columns(4)

        with col5:
            total_cases_original = df['id_caso'].nunique()
            delayed_cases = df_delayed['id_caso'].nunique()
            if total_cases_original > 0:
                porcentagem_atrasados = (delayed_cases / total_cases_original) * 100
                st.metric('% Casos atrasados', f'{porcentagem_atrasados:.1f}%')
            else:
                st.metric('% Casos atrasados', 'N/A')

        with col6:
            if 'tempo_atraso' in df_delayed.columns:
                delayed_time_mean = unique_cases['tempo_atraso'].mean()
                st.metric('Atraso médio (dias)', f'{delayed_time_mean:.2f}')
            else:
                st.metric('Atraso médio', 'N/A')

        with col7:
            if 'tempo_atraso' in df_delayed.columns:
                delayed_time_max = unique_cases['tempo_atraso'].max()
                st.metric('Atraso máximo (dias)', f'{delayed_time_max:.2f}')
            else:
                st.metric('Atraso máximo', 'N/A')

        with col8:
            if 'tempo_atraso' in df_delayed.columns:
                delayed_time_min = unique_cases['tempo_atraso'].min()
                st.metric('Atraso mínimo (dias)', f'{delayed_time_min:.2f}')
            else:
                st.metric('Atraso mínimo', 'N/A')

        st.markdown('---')

        if 'atividade' in df_delayed.columns and 'recurso' in df_delayed.columns:
            activity_freq = calculate_frequency(df_delayed, 'atividade')
            resource_freq = calculate_frequency(df_delayed, 'recurso')

            col_graph1, col_graph2 = st.columns(2)
            
            with col_graph1:
                freq_top_activity = activity_freq.head(7)
                fig_activity = px.bar(
                    freq_top_activity,
                    x='atividade',
                    y='quantidade',
                    title=f'Top atividades'
                )
                st.plotly_chart(fig_activity, use_container_width=True)
            
            with col_graph2:
                freq_top_resource = resource_freq.head(7)
                fig_resource = px.bar(
                    freq_top_resource,
                    x='recurso',
                    y='quantidade',
                    title=f'Top recursos'
                )
                st.plotly_chart(fig_resource, use_container_width=True)

            col_table1, col_table2 = st.columns(2)
            
            with col_table1:
                st.dataframe(activity_freq.head(100), use_container_width=True, height=300)
            
            with col_table2:
                st.dataframe(resource_freq.head(100), use_container_width=True, height=300)

        st.markdown('---') 

        if 'trace' in df_delayed.columns and 'transicao' in df_delayed.columns:
            trace_freq = calculate_frequency(df_delayed, 'trace')
            transition_freq = calculate_frequency(df_delayed, 'transicao')
            col_graph3, col_graph4 = st.columns(2)
            
            with col_graph3:
                freq_top_trace = trace_freq.head(7)
                fig_trace = px.bar(
                    freq_top_trace,
                    x='trace',
                    y='quantidade',
                    title=f'Top traces'
                )
                fig_trace.update_layout(xaxis_tickangle=45)
                st.plotly_chart(fig_trace, use_container_width=True)
            
            with col_graph4:
                freq_top_transition = transition_freq.head(7)
                fig_transition = px.bar(
                    freq_top_transition,
                    x='transicao',
                    y='quantidade',
                    title=f'Top transições'
                )
                fig_transition.update_layout(xaxis_tickangle=45)
                st.plotly_chart(fig_transition, use_container_width=True)

            col_table3, col_table4 = st.columns(2)
            
            with col_table3:
                st.dataframe(trace_freq.head(100), use_container_width=True, height=300)
            
            with col_table4:
                st.dataframe(transition_freq.head(100), use_container_width=True, height=300)

            st.markdown('---')

        if 'trace' in df_delayed.columns and 'gpm_nota' in df_delayed.columns:
            final_trace_freq = final_trace_frequency(df_delayed)
            gpm_freq = calculate_frequency(df_delayed, 'gpm_nota')
            col_graph5, col_graph6 = st.columns(2)

            with col_graph5:
                freq_top_final_trace = final_trace_freq.head(7)
                fig_final_trace = px.bar(
                    freq_top_final_trace,
                    x='trace final',
                    y='quantidade',
                    title=f'Top traces completos'
                )
                fig_final_trace.update_layout(xaxis_tickangle=45)
                st.plotly_chart(fig_final_trace, use_container_width=True)
            
            with col_graph6:
                freq_top_gpm = gpm_freq.head(7)
                fig_gpm = px.bar(
                    freq_top_gpm,
                    x='gpm_nota',
                    y='quantidade',
                    title=f'Top gpms'
                )
                fig_gpm.update_layout(xaxis_tickangle=45)
                st.plotly_chart(fig_gpm, use_container_width=True)

            col_table5, col_table6 = st.columns(2)
            
            with col_table5:
                st.dataframe(final_trace_freq.head(100), use_container_width=True, height=300)
            
            with col_table6:
                st.dataframe(gpm_freq.head(100), use_container_width=True, height=300)

            st.markdown('---')

        if 'trace' in df_delayed.columns and 'lead_time' in df_delayed.columns:

            col_lead, col_time = st.columns(2)
            
            with col_lead:
                if all(col in df_delayed.columns for col in ['trace', 'lead_time']):
                    top_traces = get_traces_with_highest_lead_time(df_delayed, top_n=10)
                    
                    fig_lead = px.bar(
                        top_traces,
                        x='trace final',
                        y='lead time (dias)',
                        title='Top traces completos individuais com maior lead time'
                    )
                    fig_lead.update_layout(xaxis_tickangle=45)
                    st.plotly_chart(fig_lead, use_container_width=True)
                    
                    st.dataframe(top_traces, use_container_width=True, height=300)
            
            with col_time:
                if all(col in df_delayed.columns for col in ['transicao', 'tempo_atividade_execucao']):
                    top_transitions = get_transitions_with_highest_time(df_delayed, top_n=10)
                    top_transitions = top_transitions.dropna(subset=['transição', 'tempo entre atividades (dias)'])
                    top_transitions = top_transitions[top_transitions['transição'] != 'None']
                    top_transitions = top_transitions[top_transitions['transição'].notna()]
                        
                    fig_time = px.bar(
                        top_transitions,
                        x='transição',
                        y='tempo entre atividades (dias)',
                        title='Top transições individuais mais demoradas'
                    )
                    fig_time.update_layout(xaxis_tickangle=45)
                    st.plotly_chart(fig_time, use_container_width=True)
                    
                    st.dataframe(top_transitions, use_container_width=True, height=300)

            st.markdown('---')

        if 'trace' in df_delayed.columns and 'lead_time' in df_delayed.columns and 'transicao' in df_delayed.columns and 'tempo_atividade_execucao' in df_delayed.columns:
            
            col_group1, col_group2 = st.columns(2)
            
            with col_group1:
                traces_grouped = get_traces_grouped(df_delayed, top_n=7)
                
                if not traces_grouped.empty:
                    fig_trace_media = px.bar(
                        traces_grouped,
                        x='trace',
                        y='media_lead_time',
                        title='Top traces completos agrupados pela média do lead time'
                    )
                    st.plotly_chart(fig_trace_media, use_container_width=True)
            
            with col_group2:
                
                if not traces_grouped.empty:
                    fig_trace_mediana = px.bar(
                        traces_grouped,
                        x='trace',
                        y='mediana_lead_time',
                        title='Top traces completos agrupados pela mediana do lead time'
                    )
                    st.plotly_chart(fig_trace_mediana, use_container_width=True)

            if not traces_grouped.empty:
                st.dataframe(traces_grouped.head(100), use_container_width=True, height=300)

            col_group3, col_group4 = st.columns(2)
            
            with col_group3:
                transitions_grouped = get_transition_grouped(df_delayed, top_n=7)
                
                if not transitions_grouped.empty:
                    fig_trans_media = px.bar(
                        transitions_grouped,
                        x='transicao',
                        y='media_tempo',
                        title='Top transições agrupadas pela média do tempo entre atividades'
                    )
                    st.plotly_chart(fig_trans_media, use_container_width=True)
            
            with col_group4:
                
                if not transitions_grouped.empty:
                    fig_trans_mediana = px.bar(
                        transitions_grouped,
                        x='transicao',
                        y='mediana_tempo',
                        title='Top transições agrupadas pela mediana do tempo entre atividades'
                    )
                    st.plotly_chart(fig_trans_mediana, use_container_width=True)

            if not transitions_grouped.empty:
                st.dataframe(transitions_grouped.head(100), use_container_width=True, height=300)  
        

            st.markdown('---')
        
        st.header('Amostra dos dados processados')
        st.dataframe(df_delayed.head(100), use_container_width=True)


        st.markdown('---')


        st.markdown('## Modelos')

        models = ['LightGBM', 'XGBoost', 'CatBoost', 'Random Forest', 'Regressão Logística']
        model_selected = st.selectbox('Escolha o modelo:', models)

    

else:
    st.info('Faça upload de um arquivo para visualizar os indicadores')
