# Predictive RCA

Plataforma enxuta para **análise preditiva de causa raiz** aplicada a logs de processo. Do CSV de eventos ao dataset por caso, o pipeline treina diferentes classificadores, permite otimizar o LightGBM e entrega explicações globais e locais (SHAP) sobre violações de SLA.

## Principais recursos
- Interface `BaseModel` com wrappers prontos: LightGBM, XGBoost, CatBoost, RandomForest e Regressão Logística; escolha via CLI ou código.
- Pipeline CLI interativo (`python -m src.backend.main`) para informar log, modelos e SLA, executando cada modelo em sequência; flags opcionais permitem pular prompts.
- Otimização bayesiana embutida para LightGBM (log loss negativo em validação).
- Engenharia de atributos por caso (tempo total, número de eventos, retrabalho, atividade inicial/final, recurso inicial, custo médio).
- Split estratificado em treino/validação/teste mantendo o balanceamento da classe alvo.
 - Pré-processamento com `ColumnTransformer` (numéricos em *passthrough* + One-Hot para categóricos, com nomes legíveis no formato `coluna=valor` para os gráficos de explicabilidade).
- Avaliação com AUC-ROC, classification report e matriz de confusão; importância de features (árvores) e gráficos SHAP (summary, dependence, force plot).
- Arquitetura separada em **backend** (pipeline e modelos) e **frontend** (app Streamlit) para explorar indicadores e testar o pipeline por UI.

## Estrutura do projeto
- `src/backend/main.py`: CLI interativo/parametrizável (log, modelos, SLA, otimização).
- `src/backend/pipeline/pipeline_builder.py`: orquestra leitura do log, *feature engineering*, split, pré-processamento, treino, avaliação e SHAP.
- `src/backend/preprocessing/`: construção de features por caso (`build_case_features`), split estratificado e pré-processamento (`ColumnTransformer`).
- `src/backend/models/`: wrappers que seguem `BaseModel` (LightGBM, XGBoost, CatBoost, RandomForest, LogisticRegression).
- `src/backend/evaluation/`: métricas, importância de features e análises SHAP.
- `src/backend/optimization/bayesian.py`: otimização bayesiana usada pelo LightGBM.
- `src/frontend/streamlite/app.py`: frontend em Streamlit para upload de logs, definição de alvo e exploração de atrasos/indicadores.
- `docs/`: referências rápidas (`comparativo_modelos.md`, `otimizadores.md`, material de RCA).
- `data/raw/`: logs de eventos de entrada (ex.: `event_log_sintetico_2000_cases.csv`).

## Frontend Streamlit
- URL pública: https://predictive-rca.streamlit.app/ (deploy da UI para testar sem instalar nada).
- Rodar local:
  ```bash
  pipenv install        # se ainda não instalou
  pipenv run start      # atalho definido no Pipfile → streamlit run src/frontend/streamlite/app.py
  ```
  ou, manualmente: `pipenv run streamlit run src/frontend/streamlite/app.py`.
- O app aceita upload de CSV, permite definir o método de atraso (SLA, boxplot, data desejada) e mostra métricas/indicadores visuais para os casos atrasados.
- Controles de exploração: escolha o *Top N* de atividades/recursos/traces exibidos e ajuste quantas linhas mostrar na prévia do dataset de modelagem.

## Requisitos
- Python 3.13 (definido em `Pipfile`).
- `pipenv` para gerenciamento de ambiente.
- Dependências principais: `pandas`, `scikit-learn`, `lightgbm`, `xgboost`, `catboost`, `bayesian-optimization`, `matplotlib`, `shap` (ver `Pipfile` completo).

## Como rodar
1. Instale as dependências:
   ```bash
   pipenv install
   ```
2. Rode o ambiente python:
   ```bash
   pipenv shell
   ```
2. Execute o pipeline (interativo por padrão):
   ```bash
   pipenv run python -m src.backend.main
   ```
   O script pedirá o caminho do CSV do log (ex.: `data/raw/event_log_sintetico_2000_cases.csv`), modelos desejados (`lightgbm` padrão) e SLA (48h por default no prompt).
3. Para pular prompts, use os argumentos:
   ```bash
   pipenv run python -m src.backend.main \
     --log data/raw/event_log_sintetico_2000_cases.csv \
     --models lightgbm,random_forest,catboost \
     --sla_hours 24 \
     --optimize              # ativa busca bayesiana só para LightGBM
   ```

### Usando outro log ou modelo
```python
from src.backend.pipeline.pipeline_builder import PipelineBuilder
from src.backend.models.catboost_model import CatBoostModel

pipeline = PipelineBuilder(
    model_class=CatBoostModel,
    model_params={"depth": 8, "learning_rate": 0.1, "iterations": 400},
    optimize_hyperparams=False,  # bayesiana só afeta LightGBM
)
pipeline.run_from_event_log("data/raw/seu_log.csv", sla_hours=24)
```

## Formato esperado do log
- CSV com colunas mínimas: `case_id`, `activity`, `timestamp` (parseável via `pandas.to_datetime`).
- Colunas opcionais: `resource`, `cost`.
- A engenharia de atributos gera, por caso:
  - `throughput_hours`, `num_events`, `num_unique_activities`, `rework_count`,
  - `start_activity`, `end_activity`, `start_resource` (se existir), `mean_cost` (se existir),
  - `sla_violated` (alvo binário: `throughput_hours` > `sla_hours`).

## Saídas e diagnósticos
- Impressão de métricas em validação e teste (AUC-ROC, classification report, matriz de confusão).
- Gráficos interativos/matplotlib:
  - Importância de features (para modelos com `feature_importances_`).
  - SHAP summary, dependence plot da feature mais importante e force plot para um exemplo.
- Artefatos do pré-processamento (nomes de features expandidas) retornados pelo `PipelineBuilder`.

## Configurações úteis
- `src/backend/config/settings.py`: `RANDOM_STATE`, `SLA_HOURS` (12h na lib) e diretórios padrão. O CLI usa 48h como valor sugerido no prompt, ajustável com `--sla_hours`.
- `optimize_hyperparams` em `PipelineBuilder` ativa/desativa a otimização bayesiana do LightGBM.
- Substitua o dataset em `src/backend/main.py` (argumento `--log`) ou passe o caminho diretamente em `run_from_event_log`.

## Documentação de apoio
- Comparativo dos modelos integrados: `docs/comparativo_modelos.md`.
- Estratégias de busca de hiperparâmetros: `docs/otimizadores.md`.
- Visão conceitual de RCA em mineração de processos: `docs/Root Cause Analysis – RCA.md`.

## Notebooks
Use `pipenv run jupyter notebook` para explorar `notebooks/` com o ambiente configurado.

## Licença
MIT — veja `LICENSE`.
