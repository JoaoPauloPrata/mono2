# CLAUDE.md — Recommender System Evaluation Framework

## Project Overview

Research codebase for a **monograph/thesis** comparing collaborative filtering vs hybrid ensemble recommender systems, with fairness, statistical significance, and temporal robustness analysis on MovieLens 1M.

The `data/` folder is **output-only** — never edit files inside it.

---

## Project Layout

```
experimentos/
├── pyproject.toml               # dependencies + package config
├── conftest.py                  # adds src/ to sys.path for pytest
│
├── src/recsys/                  # installable package (src layout)
│   ├── data/                    # data loading & splitting
│   │   ├── time_period_splitter.py    # sliding-window temporal splits
│   │   ├── post_processor.py          # filter/align predictions
│   │   ├── gender_splitter.py         # M/F user group files
│   │   └── activity_splitter.py       # KMeans high/low activity groups
│   │
│   ├── models/                  # recommender algorithms
│   │   ├── stochastic_item_knn.py     # custom stochastic item-item KNN
│   │   ├── constituent_methods.py     # SVD, BiasedMF, NMF, StochasticKNN wrappers
│   │   ├── hybrid_ensemble.py         # 8 regression ensemble methods
│   │   └── recommender.py             # orchestrator over models
│   │
│   ├── evaluation/              # metric computation
│   │   ├── quality_metrics.py         # RMSE, NDCG, F1, MAE, GeoRisk
│   │   ├── fairness_metrics.py        # per-group abs diff + runner functions
│   │   └── overlap_validator.py       # train/test leakage check
│   │
│   ├── aggregation/             # cross-window/exec result aggregation
│   │   ├── quality_aggregator.py
│   │   ├── final_result_aggregator.py
│   │   ├── fairness_ratio_aggregator.py
│   │   ├── group_quality_aggregator.py
│   │   └── vulnerable_group_aggregator.py
│   │
│   ├── analysis/                # statistical tests
│   │   ├── anova.py
│   │   ├── post_hoc.py                # Tukey HSD
│   │   ├── pearson_correlation.py
│   │   └── georisk_runner.py
│   │
│   ├── reporting/               # output artefacts
│   │   ├── result_splitter.py
│   │   ├── result_splitter_lite.py
│   │   ├── fairness_ratio_splitter.py
│   │   ├── quality_splitter.py
│   │   ├── vulnerability_splitter.py
│   │   ├── latex_table_generator.py
│   │   ├── chart_generator.py
│   │   └── group_chart_generator.py
│   │
│   └── pipeline/                # per-user/per-metric helpers
│       ├── user_metric_calculator.py
│       └── metric_matrix_builder.py
│
├── scripts/                     # runnable entry points (no logic)
│   ├── run_pipeline.py          # main pipeline (data split → predict → evaluate)
│   ├── run_fairness_evaluation.py   # runs kmeans + gender group fairness
│   ├── splitter_view.py         # exploratory: rating count histogram
│   └── analyze_fairness_simple.py   # exploratory: simple fairness printout
│
└── tests/
    ├── unit/
    │   ├── test_quality_metrics.py
    │   ├── test_georisk.py
    │   └── test_final_result_aggregator.py
    └── integration/
```

---

## Running

```bash
# activate env
conda activate monoenv

# run tests
pytest tests/

# run main pipeline (predict + evaluate)
python scripts/run_pipeline.py

# run fairness evaluation
python scripts/run_fairness_evaluation.py
```

---

## Pipeline Stages (run in order)

1. **Data Split** — `scripts/run_pipeline.py` → `recsys.data.time_period_splitter`
2. **Base Predictions** — `recsys.models.recommender` → `recsys.models.constituent_methods`
3. **Filter & Align** — `recsys.data.post_processor`
4. **Hybrid Predictions** — `recsys.models.hybrid_ensemble`
5. **Quality Metrics** — `recsys.evaluation.quality_metrics.Evaluator.evaluateAllMetricsForAllMethods`
6. **Fairness Groups** — `recsys.data.gender_splitter` / `recsys.data.activity_splitter`
7. **Fairness Metrics** — `recsys.evaluation.fairness_metrics` (`kmeansGroupCalculator`, `genderGroupCalculator`)
8. **Per-user Metrics** — `recsys.pipeline.user_metric_calculator`
9. **Matrix Build** — `recsys.pipeline.metric_matrix_builder`
10. **GeoRisk** — `recsys.analysis.georisk_runner`
11. **Aggregation** — `recsys.aggregation.*`
12. **ANOVA + Post-Hoc** — `recsys.analysis.anova` / `recsys.analysis.post_hoc`
13. **Split Results** — `recsys.reporting.*_splitter`
14. **Charts + LaTeX** — `recsys.reporting.chart_generator` / `latex_table_generator`

---

## Methods Under Evaluation

### Constituent (4)
| Method | File | Algorithm |
|--------|------|-----------|
| SVD | `constituent_methods.py` | BiasedSVD, 50 features |
| BiasedMF | `constituent_methods.py` | MF, 50 features, 20 iters, CD |
| NMF | `constituent_methods.py` | NMF, 15 factors, 50 epochs |
| StochasticItemKNN | `stochastic_item_knn.py` | Item-item cosine + temperature sampling |

### Hybrid Ensemble (8)
BayesianRidge, Ridge, Tweedie, RandomForest, Bagging, AdaBoost, GradientBoosting, LinearSVR

- Input: predictions from the 4 constituent methods as features
- Hyperparameters: `RandomizedSearchCV`, saved to `data/optimized_parameters/`

---

## Experimental Design

- **Dataset**: MovieLens 1M (`data/ml-1m/ratings.dat`, `data/ml-1m/users.dat`)
- **Windows**: 20 sliding windows (15-month span, 1-month step)
- **Executions**: 5 independent runs per window

---

## Metrics

| Metric | Direction |
|--------|-----------|
| RMSE | lower = better |
| NDCG@10 | higher = better |
| F1@3.5 | higher = better |
| MAE | lower = better |
| GeoRisk | higher = better |
| Fairness Diff (|groupA − groupB|) | lower = better |

### Fairness Groups
- **Gender**: Male / Female (from `data/ml-1m/users.dat`)
- **Activity**: Low / High (KMeans K=2 on interaction count)

---

## Import Conventions

All internal imports use the `recsys` package name:

```python
from recsys.evaluation.quality_metrics import Evaluator
from recsys.models.recommender import Recommender
from recsys.evaluation.fairness_metrics import GroupMetricsDiffCalculator
```

`conftest.py` at the project root adds `src/` to `sys.path` automatically for pytest. For scripts, either run `pip install -e .` or ensure `src/` is on the Python path.

---

## `__file__`-Relative Paths

Classes in `src/recsys/reporting/` and `src/recsys/aggregation/` that resolve paths relative to the project root use `Path(__file__).resolve().parents[3]` (3 levels up: `module.py` → subpackage → `recsys` → `src` → project root).
