# Comparativo dos Modelos

Visão geral dos classificadores integrados ao pipeline de *Predictive RCA*, incluindo orientações sobre quando utilizá-los e quais parâmetros de entrada tendem a ser mais relevantes.

## Como os modelos são encaixados

* Todos seguem a interface `BaseModel` (`train`, `predict`, `predict_proba`) e recebem vetores já pré-processados (`numpy.ndarray`).
* O `PipelineBuilder` constrói o dataset a partir do log de eventos (`build_case_features`), realiza *split* estratificado e aplica um `ColumnTransformer` com *numeric passthrough* para variáveis numéricas e `OneHotEncoder` para variáveis categóricas.
* A coluna-alvo é binária (`sla_violated`), construída com base na regra `throughput_hours > sla_hours`.

## Entradas do dataset

* Espera-se um log CSV com colunas mínimas: `case_id`, `activity`, `timestamp`; opcionais: `resource`, `cost`.
* As *features* geradas por caso incluem:
  `throughput_hours`, `num_events`, `num_unique_activities`, `rework_count`, `start_activity`, `end_activity`, `start_resource` (quando disponível), `mean_cost` (quando disponível) e a variável alvo `sla_violated`.

## Tabela rápida

| Modelo                      | Tipo / Biblioteca               | Pontos fortes                                                                             | Cuidados                                                                               |
| --------------------------- | ------------------------------- | ----------------------------------------------------------------------------------------- | -------------------------------------------------------------------------------------- |
| **LightGBMModel**           | Gradient Boosting (LightGBM)    | Rápido, robusto com *features* mistas, suporta otimização bayesiana integrada ao pipeline | Sensível à escolha de folhas/profundidade; risco de *overfitting* em datasets pequenos |
| **XGBoostModel**            | Gradient Boosting (XGBoost)     | Robusto para interações não lineares; forte regularização                                 | Mais lento que LightGBM; ajustar `n_estimators` e `eta` para evitar *overfit*          |
| **CatBoostModel**           | Gradient Boosting (CatBoost)    | Excelente desempenho com variáveis categóricas; exige pouca *tunagem*                     | No pipeline atual, entra após *One-Hot*, perdendo o tratamento nativo de categóricos   |
| **RandomForestModel**       | *Ensemble* de árvores (sklearn) | Baseline sólido; interpretável via *feature importance*                                   | Pode demandar muitos estimadores; não extrapola bem além do espaço de treino           |
| **LogisticRegressionModel** | Modelo linear (sklearn)         | Rápido, explicável, adequado para relações aproximadamente lineares                       | Depende de *features* bem informativas; desempenho inferior em relações não lineares   |

## Parâmetros de entrada por modelo

### **LightGBMModel** (`src/backend/models/lightgbm_model.py`)

* *Wrapper* direto para `LGBMClassifier`; parâmetros são repassados conforme `model_params`.
* Quando `optimize_hyperparams=True` no `PipelineBuilder`, aplica otimização bayesiana com limites:
  `num_leaves (16–120)`, `max_depth (3–18)`, `learning_rate (0.01–0.3)`,
  `min_data_in_leaf (5–80)`, `feature_fraction (0.5–1.0)`.
* Fixa por padrão:
  `n_estimators=300`, `objective="binary"`, `random_state=42`, `n_jobs=-1`.

### **XGBoostModel** (`src/backend/models/xgboost_model.py`)

* Utiliza por padrão `eval_metric="logloss"` e `use_label_encoder=False`; demais parâmetros seguem os *defaults* do `XGBClassifier`.
* Parâmetros mais sensíveis:
  `n_estimators`, `max_depth`, `learning_rate (eta)`, `subsample`,
  `colsample_bytree`, `min_child_weight`, `gamma`.

### **CatBoostModel** (`src/backend/models/catboost_model.py`)

* Configura `verbose=False` por padrão; demais argumentos são repassados ao `CatBoostClassifier`.
* Parâmetros com maior impacto:
  `depth`, `learning_rate`, `l2_leaf_reg`, `iterations`, `loss_function="Logloss"`.
* Como usamos *One-Hot Encoding*, não é necessário definir `cat_features` neste *wrapper*.

### **RandomForestModel** (`src/backend/models/random_forest.py`)

* Usa *defaults* do `RandomForestClassifier` (`n_estimators=100`, `max_depth=None`, `max_features="sqrt"`).
* Ajustes recomendados:
  `n_estimators`, `max_depth`, `min_samples_split`,
  `min_samples_leaf`, `class_weight="balanced"` para lidar com desbalanceamento.

### **LogisticRegressionModel** (`src/backend/models/logistic_regression.py`)

* Define `max_iter=500` para garantir convergência; demais parâmetros seguem os *defaults*.
* Parâmetros mais importantes:
  `C` (regularização), `penalty` (default `l2`), `class_weight`.

## Como passar parâmetros

### Exemplo via código

```python
from src.backend.pipeline.pipeline_builder import PipelineBuilder
from src.backend.models.xgboost_model import XGBoostModel

pipeline = PipelineBuilder(
    model_class=XGBoostModel,
    model_params={"n_estimators": 400, "max_depth": 6, "learning_rate": 0.05},
    optimize_hyperparams=False,  # apenas afeta LightGBM
)
pipeline.run_from_event_log("data/raw/event_log.csv", sla_hours=24)
```

### Via CLI (`src/main.py`)

* Seleção de modelos:
  `--models lightgbm,xgboost`
* Definição de SLA:
  `--sla_hours 24`
* A otimização bayesiana é ativada apenas para LightGBM:
  `--optimize`
* Parâmetros personalizados devem ser passados via código ou com extensão do CLI.
