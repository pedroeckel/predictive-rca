# Otimizadores de hiperparâmetros

Resumo das abordagens previstas para o projeto e o motivo de adotarmos a otimização bayesiana como padrão no LightGBM.

## Opções previstas
- **Grid search** (`src/optimization/gridsearch.py`, placeholder): varre combinações em grade. Costuma ser simples, mas cresce exponencialmente com o número de hiperparâmetros e não é eficiente para espaços contínuos amplos.
- **Random search** (`src/optimization/randomsearch.py`, placeholder): amostra combinações aleatórias. Cobertura melhor que grid em alto dimensional, mas ainda pode gastar muitas tentativas até achar regiões promissoras.
- **Otimização bayesiana** (`src/optimization/bayesian.py`, implementado): modela a função de avaliação como processo bayesiano e escolhe novos pontos com base em aquisições (exploração vs. exploração), buscando maximizar o ganho esperado a cada iteração.

## Por que usamos bayesiana no pipeline
- **Eficiência com orçamento pequeno**: encontramos bons hiperparâmetros em poucas iterações, útil quando o treino do modelo é mais caro que a avaliação pontual da função objetivo.
- **Espaço contínuo/misto**: lida bem com faixas contínuas (ex.: `learning_rate`, `num_leaves`) e evita a discretização rígida do grid.
- **Menos tentativas cegas**: diferente do random search, usa informação acumulada (superfície posterior) para propor a próxima avaliação.

## Como funciona no código
- O `PipelineBuilder` ativa a otimização bayesiana apenas para **LightGBMModel** quando `optimize_hyperparams=True`.
- Função objetivo: usa log loss negativo em validação (`_make_lightgbm_eval_fn` em `src/pipeline/pipeline_builder.py`).
- Espaço de busca padrão (`pbounds`):
  - `num_leaves`: 16–120
  - `max_depth`: 3–18
  - `learning_rate`: 0.01–0.3
  - `min_data_in_leaf`: 5–80
  - `feature_fraction`: 0.5–1.0
- Hiperparâmetros fixados para o treino final após a busca: `n_estimators=300`, `objective="binary"`, `random_state=42`, `n_jobs=-1`.
- Implementação em `src/optimization/bayesian.py` usando `bayes_opt.BayesianOptimization`.

## Como ativar
- **CLI**: `python -m src.main --optimize --models lightgbm --log path/para/log.csv --sla_hours 24`
- **Código**:
  ```python
  from src.pipeline.pipeline_builder import PipelineBuilder
  from src.models.lightgbm_model import LightGBMModel

  pipeline = PipelineBuilder(
      model_class=LightGBMModel,
      optimize_hyperparams=True,  # liga a busca bayesiana
  )
  pipeline.run_from_event_log("data/raw/event_log.csv", sla_hours=24)
  ```

## Quando considerar outras buscas
- **Grid search**: para poucos hiperparâmetros discretos e barato de treinar (útil em modelos lineares ou árvores rasas).
- **Random search**: quando o espaço é muito grande e queremos apenas um baseline rápido sem custo de modelagem da função objetivo.

No contexto atual (árvores de boosting com hiperparâmetros contínuos e custo moderado de treino), a otimização bayesiana oferece melhor trade-off entre qualidade de solução e número de experimentos.
