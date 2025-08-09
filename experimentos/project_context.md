# Project Context and Objectives

## Overview
This project is part of a Computer Science final thesis focused on **evaluating fairness and robustness** in recommender systems, with a special interest in how **hybrid recommendation methods** perform compared to classical algorithms.

We employ a **time-based sliding window evaluation** methodology to measure the performance and fairness of different recommendation algorithms over time. This approach ensures that training and testing respect the chronological order of events, preventing data leakage and allowing us to observe temporal variations in performance.

## Algorithms Evaluated
The experiments include both **classical** and **hybrid** recommendation algorithms:

- Classical:
  - ItemKNN
  - UserKNN
  - SVD
  - BIAS
  - BIASEDMF

- Hybrid:
  - Variants combining collaborative filtering and content-based filtering or other auxiliary data.

## Evaluation Metrics
We measure the performance of each algorithm using traditional metrics and one robustness metric:

- **RMSE** (Root Mean Squared Error): Measures the average prediction error magnitude.
- **NDCG** (Normalized Discounted Cumulative Gain): Evaluates the quality of ranked recommendations.
- **F1-score**: Balances precision and recall for binary relevance tasks.
- **MAE** (Mean Absolute Error): Measures the average absolute deviation of predictions.
- **GeoRisk**: A risk-sensitive metric that combines average performance and robustness, penalizing algorithms that perform poorly on subsets of the evaluation (e.g., certain metrics or time periods).

Fairness is analyzed by comparing **absolute differences** in metric values between groups or algorithms, and by evaluating GeoRisk scores as an indicator of consistency across scenarios.

## Sliding Window Methodology
The dataset (MovieLens 1M) is split into **20 overlapping time windows** using a sliding window approach:

- Each window covers **15 months** of data.
- The window advances by **1 month** for each iteration.
- Within each window, specific sub-periods are used for **training** and **testing** the algorithms.

This methodology allows for:
- Multiple evaluation points over the dataset's timeline.
- Analysis of how algorithm performance changes over time.
- Assessment of algorithm robustness and fairness in dynamic settings.

## Key Research Goals
1. **Compare classical vs. hybrid methods** in terms of accuracy, robustness, and fairness.
2. **Investigate whether hybridization improves fairness**, and under what conditions (e.g., after fine-tuning).
3. **Analyze the effect of fine-tuning**: initial results showed hybrids performing worse than classical methods; after fine-tuning, hybrids outperformed classical ones, suggesting tuning is crucial.
4. Use **GeoRisk** to quantify robustness across evaluation metrics and time periods.

## Interpretation Notes for the Codebase
- Algorithms are treated as "systems" and evaluation metrics as "queries" in the GeoRisk implementation.
- Performance matrices are normalized where necessary to avoid scale dominance between metrics.
- The code is structured to allow both **global hyperparameter tuning** and **per-window tuning** to study robustness vs. adaptability trade-offs.

## Importance
This work contributes to understanding:
- How hybrid recommender systems behave in dynamic environments.
- The relationship between robustness (via GeoRisk) and fairness.
- The role of parameter tuning in enabling hybrids to reach their potential.
