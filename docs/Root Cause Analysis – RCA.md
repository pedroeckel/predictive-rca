# **Análise de Causa Raiz em Mineração de Processos (Process Mining RCA)**

A investigação de causa raiz em mineração de processos tem como objetivo identificar **fatores estruturais, comportamentais ou contextuais** que contribuem para um desvio indesejado no fluxo de trabalho. Esses desvios podem assumir a forma de:

* **atrasos sistemáticos** (long throughput / remaining time),
* **não conformidades** (violação de modelo normativo),
* **variações estruturais** (ramos alternativos, loops, skips),
* **eventos específicos de falha** (cancelamentos, retrabalho, exceções),
* **atrasos induzidos por recurso humano/equipamento**,
* **falhas operacionais** documentadas em logs.

A abordagem quantitativa moderna fundamenta-se em três pilares:

1. **Modelagem do processo e extração de variáveis explicativas**
2. **Identificação estatística/análise discriminativa de padrões anômalos**
3. **Modelagem preditiva orientada à explicabilidade (XAI)** para inferir contribuição causal de atributos

A seguir é descrita cada etapa:

---

## **1. Modelagem e Transformações do Log**

Qualquer investigação de causa raiz exige um log devidamente estruturado; no padrão XES/CSV enriquecido isso implica extrair:

### **Atributos de Caso (case attributes)**

* tipo de ordem/tarefa
* complexidade da manutenção
* unidade/instalação
* criticidade do ativo
* origem da demanda
* data de abertura, SLAs, etc.

### **Atributos de Evento (event attributes)**

* recurso humano/máquina
* tempo de início/fim
* transição referencial
* códigos de falha
* documentos associados
* categorias de exceção

### **Atributos de Desvio (target)**

Alvo da análise de causa raiz:

* atraso acima do percentil 80/90
* violação de regra normativa (token missing, unexpected, out-of-order)
* permanência excessiva em atividade específica
* desvio de variante padrão

A consolidação desses atributos produz um dataset tabular com granularidade de caso (ou evento), adequado para modelagem.

---

## **2. Análise Estrutural: Identificação de Deviations no Modelo**

Um método típico é usar:

* **Token-Based Replay** para medir:

  * *missing tokens*
  * *remaining tokens*
  * *fitness*
  * *precisão e generalização*

* **Alignment-Based Conformance Checking**, obtendo:

  * custo de alinhamento,
  * atividades movidas no log (*log moves*)
  * atividades movidas no modelo (*model moves*).

Essas métricas geram indicadores que funcionam como *variáveis explicativas* para RCA.

Exemplo :

> Casos com *out-of-order tokens* em “Inspeção Preliminar” apresentam atraso médio 38% maior.

---

## **3. Análise Estatística Inicial (RCA Clássico)**

Métodos básicos incluem:

* **Testes de associação** (Qui-quadrado, Cramér’s V)
* **Modelos lineares generalizados (GLM)**
* **Decision trees para detecção de splits dominantes**
* **Contrast set mining / subgroup discovery**

Perguntas típicas:

* *Quais atributos diferenciam casos atrasados vs. normais?*
* *Quais combinações de atividade+recurso explicam maior probabilidade de exceção?*

---

# **4. Modelagem Preditiva para RCA (XGBoost/LightGBM + SHAP)**

A abordagem contemporânea mais robusta para causa raiz utiliza **modelos de gradiente boosting** associados a **métodos interpretáveis (SHAP)**.

### **4.1. Treinamento do Modelo**

O pipeline é:

1. Construção do dataset explicativo
2. Divisão temporal (train/test)
3. Treino com XGBoost ou LightGBM
4. Métricas de predição para validação
5. Explicabilidade global e local

O modelo aprende padrões complexos como:

* interações entre atividades,
* dependências temporais,
* variações por instalação,
* desvios induzidos por recursos específicos.

### **4.2. SHAP – Explicabilidade**

O método SHAP fornece:

* **importância média global dos atributos (SHAP mean)**
* **efeito marginal (dependence plots)**
* **explicação individual por caso (force plot)**
* **interações entre atributos**

Exemplo prático típico:

> “A presença do recurso *Equipe_Manutenção_03* na atividade *Constatação de Falha* aumenta em média +1.8 dias o tempo remanescente, com contribuição SHAP de +0.62.”

Isso não é apenas correlação; é análise de contribuição marginal do modelo.

---

# **5. RCA Orientado a Variantes (Variant-Based RCA)**

Aqui, investigamos:

* qual variante leva a maior atraso,
* qual equipamento/unidade apresenta padrões atípicos,
* diferenças estruturais entre grupos de casos (subgroup discovery).

Ferramentas matemáticas utilizadas:

* *Local Process Models*
* *Trace clustering*
* *Sequence mining* (prefixos frequentes)
* *Suffix prediction*

Aplicações típicas:
Detectar se variantes contendo a sequência “Inspeção → Reabertura → Aprovação” são responsáveis por 60% dos atrasos acima do SLA.

---

# **6. RCA Temporal (Remaining Time & Bottleneck Analysis)**

Combinamos:

* **modelos de estimativa do tempo remanescente** (LSTM, LightGBM, GRU, Temporal XGBoost)
* **simulação temporal local** (predictive monitoring)
* **heatmaps de espera entre atividades**

Perguntas investigadas:

* Onde se forma o gargalo?
* Qual etapa sofre maior variação temporal?
* Que condições externas impactam a duração total?

---

# **7. Modelo Integrado de RCA**

Um pipeline completo:

1. **Extração do log** (SAP PM, ERP, Maximo etc.)
2. **Pré-processamento e enriquecimento**
3. **Descoberta de processo (Inductive Miner / Split Miner)**
4. **Conformance checking**
5. **Feature engineering** (tempo, recurso, duração, variantes, métricas de alinhamento)
6. **Treino de XGBoost/LightGBM para predição de atraso/exceção**
7. **Explicabilidade via SHAP**
8. **Dashboard interativo** (Streamlit) com:

   * importância global,
   * causa raiz por caso,
   * comparação entre variantes,
   * simulação do impacto de intervenção (what-if analysis).

---

# **8. Saída Esperada em RCA (Exemplos)**

* “A combinação **TipoOrdem = Corretiva + Ativo de Média Criticidade + Variante V12** explica 67% dos casos atrasados.”
* “A atividade **Emissão de Documento Operacional** apresenta variância temporal 5× acima do baseline.”
* “Recursos B7 e R14 são responsáveis por 42% das ocorrências de token missing.”
* “Casos com *log move* em ‘Análise Técnica’ apresentam probabilidade 3,1× maior de violação de SLA.”

---

# **9. Quando Usar Cada Abordagem**

| Técnica                  | Objetivo                            | Cenário ideal                         |
| ------------------------ | ----------------------------------- | ------------------------------------- |
| Conformance Checking     | Explicar violações estruturais      | Aderência a modelo normativo          |
| RCA Estatística          | Relações simples e interpretáveis   | Pequenas variações e poucos atributos |
| XGBoost/SHAP             | Padrões complexos e explicabilidade | Logs grandes e multidimensionais      |
| Trace Clustering         | Descobrir grupos atípicos           | Processos muito variantes             |
| Prefix/Suffix Prediction | Antecipar desvios                   | Monitoramento operacional             |

---

# **10. Conclusão**

RCA em mineração de processos evoluiu do paradigma puramente qualitativo para uma **abordagem quantitativa explicável**, onde modelos supervisionados de alto desempenho — acompanhados de análise SHAP — fornecem evidências robustas sobre o impacto de cada fator no comportamento real do processo.

Essa abordagem não substitui a análise de especialistas, mas amplia a capacidade de diagnóstico com rigor estatístico e replicabilidade.

