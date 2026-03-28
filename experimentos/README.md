# The Hybridization Paradox: Impact of Algorithm Combination on Fairness and Risk in Recommender Systems

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
python -c "
from scripts.run_pipeline import split_data, split_full_windows
split_data()
split_full_windows()
"
```

Output: `data/windows/`

---

### Stage 2 — Constituent Method Predictions

Trains and generates predictions for the 4 base (Level 0) algorithms — BiasedSVD, BiasedMF, NMF, StochasticItemKNN — for every window and execution.

```bash
python -c "
import pandas as pd
from scripts.run_pipeline import load_data_and_run
for exec_number in range(1, 6):
    for window_number in range(1, 21):
        load_data_and_run(window_number, exec_number)
"
```

Output: `data/predictions/`

---

### Stage 3 — Hybrid Predictions + Quality Metrics

Trains the 8 regression meta-models (Level 1: BayesianRidge, Ridge, Tweedie, RandomForest, Bagging, AdaBoost, GradientBoosting, LinearSVR) using the constituent predictions as features, then evaluates RMSE, NDCG@10, F1@3.5, and MAE for all methods.

```bash
python scripts/run_pipeline.py
```

Output: `data/MetricsForMethods/MetricsForWindow{w}_{exec}.csv`

---

### Stage 4 — Fairness Evaluation (Gender + Activity Groups)

Computes the Absolute Difference (AD) fairness metric for two group dimensions:
- **Gender**: Male vs. Female (from `users.dat`)
- **Activity**: High vs. Low (KMeans K=2 on interaction count per window)

```bash
python scripts/run_fairness_evaluation.py
```

Output: `data/MetricsForMethods/Fairness/`

---

### Stage 5 — Per-user Metrics

Computes RMSE, NDCG, F1, and MAE at the individual user level across all windows and executions.

```bash
python -m recsys.pipeline.user_metric_calculator
```

Output: `data/MetricsForMethods/ByUser/`

---

### Stage 6 — Metric Matrix Build

Organizes per-user metrics into matrices (users × algorithms) per window/execution/metric, as input for GeoRisk computation.

```bash
python -m recsys.pipeline.metric_matrix_builder
```

Output: `data/MetricsForMethods/ByMetric/`

---

### Stage 7 — GeoRisk

Applies the GeoRisk formula (α=0.05) to the metric matrices to measure risk-sensitiveness of each algorithm.

```bash
python -m recsys.analysis.georisk_runner
```

Output: `data/MetricsForMethods/GeoRisk/`

---

### Stage 8 — Aggregation

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

### Stage 9 — Statistical Analysis (ANOVA + Tukey HSD)

Runs one-way ANOVA (α=0.05) across the 100 samples (20 windows × 5 executions) per algorithm, followed by Tukey HSD post-hoc pairwise comparisons.

```bash
python -m recsys.analysis.anova
python -m recsys.analysis.post_hoc
```

Output: `data/MetricsForMethods/anova_results/`

---

### Stage 10 — Result Splitting

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

### Stage 11 — Charts and LaTeX Tables

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
experimentos/
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
