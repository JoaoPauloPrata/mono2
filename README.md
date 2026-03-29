# The Hybridization Paradox: Impact of Algorithm Combination on Fairness and Risk in Recommender Systems

> **[English](#english) | [Português](#português)**

---

<a name="english"></a>
# English

Replication code for the paper submitted to SBBD 2026. The study evaluates weighted hybridization (stacking) of collaborative filtering algorithms against constituent methods on **MovieLens 1M**, using **20 sliding time windows × 5 independent executions** per window.

**Key finding:** Hybridization acts as a *generic leveler* — it amplifies fairness disparities between gender/activity groups and increases system risk (GeoRisk), despite achieving statistically equivalent prediction error (RMSE/MAE).

---

## Dataset

Download **MovieLens 1M** and place the files inside `data/ml-1m/`:

```
data/ml-1m/ratings.dat
data/ml-1m/users.dat
data/ml-1m/movies.dat
```

Download: https://grouplens.org/datasets/movielens/1m/

---

## Environment Setup

```bash
conda create -n yourenv python=3.10
conda activate yourenv
pip install -e .
```

---

## Reproducing the Experiment

Run the scripts below **in order**. Each stage depends on the outputs of the previous one.

### Stage 1 — Data Split (sliding windows)

Splits the dataset into 20 temporal windows (15-month span, 1-month step). Each window generates train/test splits for both the constituent methods (12+3 months) and the hybrid meta-model training (9+3+3 months).

```bash
python scripts/prepare_data.py
```

Output: `data/windows/`

---

### Stage 2 — Constituent Method Predictions

Trains and generates predictions for the 4 base (Level 0) algorithms — BiasedSVD, BiasedMF, NMF, StochasticItemKNN — for every window and execution.

```bash
python scripts/run_constituent_predictions.py
```

Output: `data/predictions/`

---

### Stage 3 — Post-processing (Filter Common Pairs)

Filters the constituent prediction files to keep only the (user, item) pairs that received a valid prediction from **every** constituent method. Also aligns the test splits to those pairs. This ensures the hybrid meta-model trains and evaluates on a consistent set.

```bash
python scripts/run_post_processor.py
```

Output: `data/filtered_predictions/`, `data/windows/processed/`

---

### Stage 4 — Hybrid Predictions + Quality Metrics

Trains the 8 regression meta-models (Level 1: BayesianRidge, Ridge, Tweedie, RandomForest, Bagging, AdaBoost, GradientBoosting, LinearSVR) using the constituent predictions as features, then evaluates RMSE, NDCG@10, F1@3.5, and MAE for all methods.

```bash
python scripts/run_pipeline.py
```

Output: `data/MetricsForMethods/MetricsForWindow{w}_{exec}.csv`

---

### Stage 5 — Fairness Evaluation (Gender + Activity Groups)

Computes the Absolute Difference (AD) fairness metric for two group dimensions:
- **Gender**: Male vs. Female (from `users.dat`)
- **Activity**: High vs. Low (KMeans K=2 on interaction count per window)

```bash
python scripts/run_fairness_evaluation.py
```

Output: `data/MetricsForMethods/Fairness/`

---

### Stage 6 — Per-user Metrics

Computes RMSE, NDCG, F1, and MAE at the individual user level across all windows and executions.

```bash
python -m recsys.pipeline.user_metric_calculator
```

Output: `data/MetricsForMethods/ByUser/`

---

### Stage 7 — Metric Matrix Build

Organizes per-user metrics into matrices (users × algorithms) per window/execution/metric, as input for GeoRisk computation.

```bash
python -m recsys.pipeline.metric_matrix_builder
```

Output: `data/MetricsForMethods/ByMetric/`

---

### Stage 8 — GeoRisk

Applies the GeoRisk formula (α=0.05) to the metric matrices to measure risk-sensitiveness of each algorithm.

```bash
python -m recsys.analysis.georisk_runner
```

Output: `data/MetricsForMethods/GeoRisk/`

---

### Stage 9 — Aggregation

Aggregates results across windows and executions (mean, std, median, min, max, 95% CI) for quality metrics, fairness, and GeoRisk.

```bash
python -m recsys.aggregation.quality_aggregator
python -m recsys.aggregation.final_result_aggregator
python -m recsys.aggregation.fairness_ratio_aggregator
python -m recsys.aggregation.group_quality_aggregator
python -m recsys.aggregation.vulnerable_group_aggregator
```

Output: `data/MetricsForMethods/`

---

### Stage 10 — Statistical Analysis (ANOVA + Tukey HSD)

Runs one-way ANOVA (α=0.05) across the 100 samples (20 windows × 5 executions) per algorithm, followed by Tukey HSD post-hoc pairwise comparisons.

```bash
python -m recsys.analysis.anova
python -m recsys.analysis.post_hoc
```

Output: `data/MetricsForMethods/anova_results/`

---

### Stage 11 — Result Splitting

Splits the aggregated results into separate files per metric and analysis type for easier reporting.

```bash
python -m recsys.reporting.result_splitter
python -m recsys.reporting.result_splitter_lite
python -m recsys.reporting.fairness_ratio_splitter
python -m recsys.reporting.quality_splitter
python -m recsys.reporting.vulnerability_splitter
```

Output: `data/MetricsForMethods/final_results_split/`

---

### Stage 12 — Charts and LaTeX Tables

Generates the bar charts (with 95% CI error bars) and LaTeX tables used in the paper.

```bash
python -m recsys.reporting.chart_generator
python -m recsys.reporting.group_chart_generator
python -m recsys.reporting.latex_table_generator
```

Output: `data/charts/`, `data/charts_groups/`, `data/MetricsForMethods/latex_tables/`

---

## Running Tests

```bash
pytest tests/
```

---

## Project Structure

```
./
├── data/ml-1m/              # Dataset (download separately)
├── scripts/                 # Entry points (run in order above)
│   ├── run_pipeline.py      # Stages 1–3
│   └── run_fairness_evaluation.py  # Stage 4
├── src/recsys/
│   ├── data/                # Splitting, post-processing, group segmentation
│   ├── models/              # SVD, BiasedMF, NMF, StochasticItemKNN, hybrid ensemble
│   ├── evaluation/          # RMSE, NDCG, F1, MAE, GeoRisk, fairness metrics
│   ├── pipeline/            # Per-user metrics, metric matrix builder
│   ├── aggregation/         # Cross-window/execution aggregators
│   ├── analysis/            # ANOVA, Tukey HSD, GeoRisk runner, Pearson correlation
│   └── reporting/           # Result splitters, charts, LaTeX tables
└── tests/
```

---

## Methods Evaluated

| Type | Algorithm |
|------|-----------|
| Constituent (Level 0) | BiasedSVD, BiasedMF, NMF, StochasticItemKNN |
| Hybrid (Level 1) | BayesianRidge, Ridge, Tweedie, RandomForest, Bagging, AdaBoost, GradientBoosting, LinearSVR |

Hyperparameter optimization: `RandomizedSearchCV` (3-fold CV, 15 iterations, MSE target).

## Metrics

| Metric | Description | Direction |
|--------|-------------|-----------|
| RMSE | Root Mean Squared Error | lower = better |
| MAE | Mean Absolute Error | lower = better |
| F1 | F1-score at threshold 3.5 | higher = better |
| NDCG@10 | Normalized Discounted Cumulative Gain | higher = better |
| GeoRisk | Risk-sensitiveness robustness score | higher = better |
| Fairness AD | Absolute Difference between groups | lower = better |

---

---

<a name="português"></a>
# Português

Código de replicação para o artigo submetido ao SBBD 2026. O estudo avalia a hibridização ponderada (stacking) de algoritmos de filtragem colaborativa em comparação com os métodos constituintes no **MovieLens 1M**, utilizando **20 janelas temporais deslizantes × 5 execuções independentes** por janela.

**Principal descoberta:** A hibridização atua como um *nivelador genérico* — amplifica disparidades de equidade entre grupos de gênero/atividade e aumenta o risco do sistema (GeoRisk), apesar de alcançar erro de predição estatisticamente equivalente (RMSE/MAE).

---

## Dataset

Baixe o **MovieLens 1M** e coloque os arquivos dentro de `data/ml-1m/`:

```
data/ml-1m/ratings.dat
data/ml-1m/users.dat
data/ml-1m/movies.dat
```

Download: https://grouplens.org/datasets/movielens/1m/

---

## Configuração do Ambiente

```bash
conda create -n yourenv python=3.10
conda activate yourenv
pip install -e .
```

---

## Reproduzindo o Experimento

Execute os scripts abaixo **em ordem**. Cada etapa depende das saídas da etapa anterior.

### Etapa 1 — Divisão dos Dados (janelas deslizantes)

Divide o dataset em 20 janelas temporais (intervalo de 15 meses, passo de 1 mês). Cada janela gera divisões treino/teste para os métodos constituintes (12+3 meses) e para o treinamento do meta-modelo híbrido (9+3+3 meses).

```bash
python scripts/prepare_data.py
```

Saída: `data/windows/`

---

### Etapa 2 — Predições dos Métodos Constituintes

Treina e gera predições para os 4 algoritmos base (Nível 0) — BiasedSVD, BiasedMF, NMF, StochasticItemKNN — para cada janela e execução.

```bash
python scripts/run_constituent_predictions.py
```

Saída: `data/predictions/`

---

### Etapa 3 — Pós-processamento (Filtragem de Pares Comuns)

Filtra os arquivos de predição constituintes para manter apenas os pares (usuário, item) que receberam predição válida de **todos** os métodos constituintes. Também alinha as divisões de teste a esses pares, garantindo que o meta-modelo híbrido treine e avalie sobre um conjunto consistente.

```bash
python scripts/run_post_processor.py
```

Saída: `data/filtered_predictions/`, `data/windows/processed/`

---

### Etapa 4 — Predições Híbridas + Métricas de Qualidade

Treina os 8 meta-modelos de regressão (Nível 1: BayesianRidge, Ridge, Tweedie, RandomForest, Bagging, AdaBoost, GradientBoosting, LinearSVR) usando as predições constituintes como features, e então avalia RMSE, NDCG@10, F1@3.5 e MAE para todos os métodos.

```bash
python scripts/run_pipeline.py
```

Saída: `data/MetricsForMethods/MetricsForWindow{w}_{exec}.csv`

---

### Etapa 5 — Avaliação de Equidade (Grupos de Gênero + Atividade)

Calcula a métrica de equidade Diferença Absoluta (DA) para duas dimensões de grupos:
- **Gênero**: Masculino vs. Feminino (de `users.dat`)
- **Atividade**: Alta vs. Baixa (KMeans K=2 sobre contagem de interações por janela)

```bash
python scripts/run_fairness_evaluation.py
```

Saída: `data/MetricsForMethods/Fairness/`

---

### Etapa 6 — Métricas por Usuário

Calcula RMSE, NDCG, F1 e MAE no nível individual do usuário ao longo de todas as janelas e execuções.

```bash
python -m recsys.pipeline.user_metric_calculator
```

Saída: `data/MetricsForMethods/ByUser/`

---

### Etapa 7 — Construção da Matriz de Métricas

Organiza as métricas por usuário em matrizes (usuários × algoritmos) por janela/execução/métrica, como entrada para o cálculo do GeoRisk.

```bash
python -m recsys.pipeline.metric_matrix_builder
```

Saída: `data/MetricsForMethods/ByMetric/`

---

### Etapa 8 — GeoRisk

Aplica a fórmula do GeoRisk (α=0.05) às matrizes de métricas para medir a sensibilidade ao risco de cada algoritmo.

```bash
python -m recsys.analysis.georisk_runner
```

Saída: `data/MetricsForMethods/GeoRisk/`

---

### Etapa 9 — Agregação

Agrega os resultados entre janelas e execuções (média, desvio padrão, mediana, mín., máx., IC 95%) para métricas de qualidade, equidade e GeoRisk.

```bash
python -m recsys.aggregation.quality_aggregator
python -m recsys.aggregation.final_result_aggregator
python -m recsys.aggregation.fairness_ratio_aggregator
python -m recsys.aggregation.group_quality_aggregator
python -m recsys.aggregation.vulnerable_group_aggregator
```

Saída: `data/MetricsForMethods/`

---

### Etapa 10 — Análise Estatística (ANOVA + Tukey HSD)

Executa ANOVA unidirecional (α=0.05) sobre as 100 amostras (20 janelas × 5 execuções) por algoritmo, seguido de comparações par a par Tukey HSD post-hoc.

```bash
python -m recsys.analysis.anova
python -m recsys.analysis.post_hoc
```

Saída: `data/MetricsForMethods/anova_results/`

---

### Etapa 11 — Divisão dos Resultados

Divide os resultados agregados em arquivos separados por métrica e tipo de análise para facilitar o reporte.

```bash
python -m recsys.reporting.result_splitter
python -m recsys.reporting.result_splitter_lite
python -m recsys.reporting.fairness_ratio_splitter
python -m recsys.reporting.quality_splitter
python -m recsys.reporting.vulnerability_splitter
```

Saída: `data/MetricsForMethods/final_results_split/`

---

### Etapa 12 — Gráficos e Tabelas LaTeX

Gera os gráficos de barras (com barras de erro IC 95%) e tabelas LaTeX utilizados no artigo.

```bash
python -m recsys.reporting.chart_generator
python -m recsys.reporting.group_chart_generator
python -m recsys.reporting.latex_table_generator
```

Saída: `data/charts/`, `data/charts_groups/`, `data/MetricsForMethods/latex_tables/`

---

## Executando os Testes

```bash
pytest tests/
```

---

## Estrutura do Projeto

```
./
├── data/ml-1m/              # Dataset (baixar separadamente)
├── scripts/                 # Pontos de entrada (executar na ordem acima)
│   ├── run_pipeline.py      # Etapas 1–3
│   └── run_fairness_evaluation.py  # Etapa 4
├── src/recsys/
│   ├── data/                # Divisão, pós-processamento, segmentação de grupos
│   ├── models/              # SVD, BiasedMF, NMF, StochasticItemKNN, ensemble híbrido
│   ├── evaluation/          # RMSE, NDCG, F1, MAE, GeoRisk, métricas de equidade
│   ├── pipeline/            # Métricas por usuário, construtor de matriz de métricas
│   ├── aggregation/         # Agregadores entre janelas/execuções
│   ├── analysis/            # ANOVA, Tukey HSD, GeoRisk runner, correlação de Pearson
│   └── reporting/           # Divisores de resultados, gráficos, tabelas LaTeX
└── tests/
```

---

## Métodos Avaliados

| Tipo | Algoritmo |
|------|-----------|
| Constituinte (Nível 0) | BiasedSVD, BiasedMF, NMF, StochasticItemKNN |
| Híbrido (Nível 1) | BayesianRidge, Ridge, Tweedie, RandomForest, Bagging, AdaBoost, GradientBoosting, LinearSVR |

Otimização de hiperparâmetros: `RandomizedSearchCV` (3-fold CV, 15 iterações, objetivo MSE).

## Métricas

| Métrica | Descrição | Direção |
|---------|-----------|---------|
| RMSE | Raiz do Erro Quadrático Médio | menor = melhor |
| MAE | Erro Absoluto Médio | menor = melhor |
| F1 | F1-score com limiar 3.5 | maior = melhor |
| NDCG@10 | Ganho Cumulativo Descontado Normalizado | maior = melhor |
| GeoRisk | Pontuação de robustez à sensibilidade ao risco | maior = melhor |
| Equidade DA | Diferença Absoluta entre grupos | menor = melhor |
