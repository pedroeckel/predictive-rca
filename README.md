# Predictive RCA

Plataforma enxuta para **análise preditiva de causa raiz** aplicada a logs de processo. O pipeline transforma um log de eventos em um dataset em nível de caso, treina modelos de classificação, otimiza hiperparâmetros e entrega explicações globais e locais (SHAP) sobre violações de SLA.

## Principais recursos
- Engenharia de atributos por caso (tempo total, número de eventos, retrabalho, atividade inicial/final, custo médio).
- Split estratificado em treino/validação/teste preservando balanceamento da classe alvo.
- Pré-processamento com `ColumnTransformer` (numéricos + One-Hot para categóricos).
- Modelos plugáveis via interface `BaseModel` (LightGBM, RandomForest, Regressão Logística); otimização bayesiana disponível para LightGBM.
- Avaliação com AUC-ROC, relatório de classificação e matriz de confusão.
- Explicabilidade: importância de features (árvores) e gráficos SHAP (summary, dependence, force plot).

## Estrutura do projeto
- `src/main.py`: ponto de entrada para rodar o pipeline completo.
- `src/pipeline/pipeline_builder.py`: orquestra todo o fluxo, da leitura do log até SHAP.
- `src/preprocessing/`: construção de features de caso, split estratificado e pré-processamento.
- `src/models/`: implementações que seguem `BaseModel` (LightGBM, RandomForest, LogisticRegression).
- `src/evaluation/`: métricas de classificação, importância de features e análises SHAP.
- `src/optimization/bayesian.py`: wrapper para otimização bayesiana de hiperparâmetros.
- `data/raw/`: logs de eventos de entrada (ex.: `event_log_sintetico_2000_cases.csv`).
- `docs/`: materiais de apoio sobre RCA.

## Requisitos
- Python 3.13 (definido em `Pipfile`).
- `pipenv` para gerenciamento de ambiente.
- Dependências principais: `pandas`, `scikit-learn`, `lightgbm`, `bayesian-optimization`, `matplotlib`, `shap` (ver `Pipfile` completo).

## Como rodar
1. Instale as dependências:
   ```bash
   pipenv install
   ```
2. Ative o ambiente virtual:
   ```bash
   pipenv shell
   ```
3. Execute o pipeline (a partir da raiz do repositório):
   ```bash
   python -m src.main
   ```
   O script usa por padrão `data/raw/event_log_sintetico_2000_cases.csv` e SLA de 12h (`CONFIG.SLA_HOURS`).

### Usando outro log ou modelo
```python
from src.pipeline.pipeline_builder import PipelineBuilder
from src.models.random_forest import RandomForestModel

pipeline = PipelineBuilder(
    model_class=RandomForestModel,
    model_params={"n_estimators": 400, "max_depth": 12},
    optimize_hyperparams=False,  # otimização bayesiana só para LightGBM
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
- `src/config/settings.py`: `RANDOM_STATE`, `SLA_HOURS` e diretórios padrão.
- Ajuste `optimize_hyperparams` em `PipelineBuilder` para ativar/desativar a otimização bayesiana do LightGBM.
- Substitua o dataset padrão em `src/main.py` ou passe o caminho diretamente em `run_from_event_log`.

## Notebooks
Use `pipenv run jupyter notebook` para explorar `notebooks/` com o ambiente configurado.

## Licença
MIT — veja `LICENSE`.
