# CLAUDE.md — Recommender System Evaluation Framework

## Project Overview

This is a research codebase for a **monograph/thesis** comparing collaborative filtering vs hybrid ensemble recommender systems, with explicit fairness, statistical significance, and temporal robustness analysis on the MovieLens 1M dataset.

The `data/` folder is **output-only** — never edit files inside it.

---

## Architecture

### Pipeline Stages (run in order)

1. **Data Splitting** — `main.py` → calls `src/DataProcessing/TimePeriodSpliter.py`
2. **Base Predictions** — `src/Recommender.py` → calls `src/Methods/ConstituentMethods.py`
3. **Filtering & Alignment** — `src/DataProcessing/PosProcess.py`
4. **Hybrid Predictions** — `src/Methods/RegressionMethodsWithFineTuning.py`
5. **Quality Metrics** — `src/Metrics/Evaluator.py`
6. **Fairness Metrics** — `gender.py` / `kmeans.py` → `src/Metrics/absDiffCalculator.py`
7. **Statistical Analysis** — `anovaAnalysis.py`, `postHocAnalysis.py`, `risk.py`, `pearsonCorrelation.py`
8. **Aggregation** — `evaluateQuality.py`, `finalResult.py`, `finalResultFairnessRatio.py`, `finalResultSplitter.py`
9. **Reporting** — `chartGenerator.py`, `latexTableGenerator.py`, `byMetric.py`, `byUser.py`

---

## Methods Under Evaluation

### Constituent (Base) Methods — 4 total
| Method | File | Algorithm |
|--------|------|-----------|
| SVD | `ConstituentMethods.py` | BiasedSVD, 50 features, randomized |
| BiasedMF | `ConstituentMethods.py` | Matrix Factorization, 50 features, 20 iters, CD |
| NMF | `ConstituentMethods.py` | Non-negative MF, 15 factors, 50 epochs |
| StochasticItemKNN | `StochasticItemKNN.py` | Item-item cosine + temperature-based stochastic sampling |

### Hybrid (Ensemble Regression) Methods — 8 total
BayesianRidge, Ridge, Tweedie, RandomForest, Bagging, AdaBoost, GradientBoosting, LinearSVR

- Input features: predictions from the 4 constituent methods
- Weights (fixed): 15%, 15%, 12%, 15%, 12%, 10%, 13%, 8%
- Hyperparameters: optimized via `RandomizedSearchCV`, saved to `data/optimized_parameters/`

---

## Experimental Design

- **Dataset**: MovieLens 1M (`data/ml-1m/ratings.dat`, `data/ml-1m/users.dat`)
- **Windows**: 20 sliding temporal windows (15-month span, 1-month step)
- **Executions**: 5 independent runs per window (for statistical robustness)
- **Window split**:
  - `train_to_get_regression_train_data`: months 0–12
  - `test_to_get_regression_train_data`: months 12–15
  - `train_to_get_constituent_methods`: months 0–12
  - `test_to_get_constituent_methods`: months 12–15

---

## Metrics

| Metric | Formula | Direction |
|--------|---------|-----------|
| RMSE | sqrt(mean((ŷ−y)²)) | lower = better |
| NDCG@10 | DCG@10 / IDCG@10 | higher = better |
| F1@3.5 | Binary relevance at threshold 3.5 | higher = better |
| MAE | mean(|ŷ−y|) | lower = better |
| GeoRisk | sqrt((S/c) × Φ(z/c)), α=0.05 | lower = better |
| Fairness Diff | |metric_groupA − metric_groupB| | lower = better |

### Fairness Groups
- **Gender**: Male / Female (from `data/ml-1m/users.dat`)
- **Activity**: Low / High (KMeans K=2 on interaction count)

---

## Key Source Files

| File | Responsibility |
|------|---------------|
| `main.py` | Top-level pipeline orchestrator |
| `src/Recommender.py` | Runs constituent methods across all windows/execs |
| `src/Methods/ConstituentMethods.py` | SVD, BiasedMF, NMF implementations |
| `src/Methods/StochasticItemKNN.py` | Custom stochastic KNN recommender |
| `src/Methods/RegressionMethodsWithFineTuning.py` | 8 hybrid regression ensembles |
| `src/DataProcessing/TimePeriodSpliter.py` | Sliding-window temporal splits |
| `src/DataProcessing/PosProcess.py` | Prediction filtering and alignment |
| `src/Metrics/Evaluator.py` | RMSE, NDCG, F1, MAE computation |
| `src/Metrics/absDiffCalculator.py` | Per-group fairness metrics |
| `run_fairness_evaluation.py` | Orchestrates full fairness analysis |
| `anovaAnalysis.py` | One-way ANOVA across methods |
| `postHocAnalysis.py` | Tukey HSD post-hoc pairwise tests |
| `risk.py` | GeoRisk calculator |
| `pearsonCorrelation.py` | Pairwise method correlation |
| `overlapValidator.py` | Validates no train/test leakage |
| `finalResultSplitter.py` | Splits aggregated results per metric/analysis |
| `latexTableGenerator.py` | Generates LaTeX tables for the thesis |

---

## Dependencies

Defined in `src/Methods/requirements.txt`:
- `pandas`, `numpy`, `scikit-learn`, `scipy`
- `lenskit` — for SVD and BiasedMF
- `surprise` — for NMF

---

## Output Structure (data/ — do not edit)

```
data/
├── windows/                         temporal train/test splits
├── predictions/                     constituent method TSV predictions
├── filtered_predictions/            aligned predictions (common user-item pairs)
├── HybridPredictions/               ensemble regression outputs
├── optimized_parameters/            saved hyperparameter configs
├── MetricsForMethods/
│   ├── MetricsForWindow{w}_{e}.csv  quality metrics per window/exec
│   ├── Fairness/                    per-group fairness metrics
│   │   ├── gender/
│   │   └── kmeans/
│   ├── GeoRisk/                     risk scores per window/exec
│   ├── ByUser/                      per-user metric breakdowns
│   ├── ByMetric/                    method × user matrices
│   ├── anova_results/               ANOVA + Tukey HSD outputs
│   ├── final_results.csv
│   ├── quality_results.csv
│   └── fairness_ratio_results.csv
├── charts/                          PNG bar charts with CI error bars
└── latex_tables_lite/               LaTeX table .txt files
```

---

## Notes

- All prediction files are TSV format with columns: `user`, `item`, `prediction`
- Filtering step (`PosProcess.py`) ensures all methods share the same user-item pairs before metric computation
- ANOVA groups statistically similar methods; Tukey HSD identifies which pairs differ (p < 0.05)
- The 5-execution design enables confidence intervals and guards against random seed sensitivity
