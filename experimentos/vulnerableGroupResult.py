import os
from typing import Dict, List, Iterable

import numpy as np
import pandas as pd


class VulnerableGroupAggregator:
    """
    Agrega métricas (RMSE, NDCG, F1, MAE) apenas dos grupos vulneráveis:
    - Gênero: female
    - Atividade (kmeans): group_low
    Usa arquivos gerados por absDiffCalculator.py.
    """

    def __init__(
        self,
        fairness_gender_path: str = "./data/MetricsForMethods/Fairness/gender",
        fairness_activity_path: str = "./data/MetricsForMethods/Fairness/kmeans",
        output_path: str = "./data/MetricsForMethods/vulnerable_group_results.csv",
        executions: int = 5,
    ) -> None:
        self.fairness_gender_path = fairness_gender_path.rstrip("/\\")
        self.fairness_activity_path = fairness_activity_path.rstrip("/\\")
        self.output_path = output_path
        self.executions = executions

        self.metrics = ["rmse", "f1", "ndcg", "mae"]
        self.constituent_algorithms = ["SVD", "BIASEDMF", "NMF", "StochasticItemKNN"]
        self.hybrid_algorithms = [
            "BayesianRidge",
            "Tweedie",
            "Ridge",
            "RandomForest",
            "Bagging",
            "AdaBoost",
            "GradientBoosting",
            "LinearSVR",
        ]
        self.algorithms = self.constituent_algorithms + self.hybrid_algorithms
        self.window_range = range(1, 21)

    @staticmethod
    def _init_result_dict(metrics: Iterable[str]) -> Dict[str, List[float]]:
        return {m: [] for m in metrics}

    @staticmethod
    def _stats(values: List[float]) -> Dict[str, float]:
        arr = np.array(values, dtype=float)
        return {
            "mean": float(np.mean(arr)),
            "std": float(np.std(arr, ddof=1)) if len(arr) > 1 else 0.0,
            "median": float(np.median(arr)),
            "min": float(np.min(arr)),
            "max": float(np.max(arr)),
            "n": int(len(arr)),
        }

    @staticmethod
    def _ci95(mean: float, std: float, n: int) -> Dict[str, float]:
        if n <= 1:
            return {"ci_lower": None, "ci_upper": None}
        margin = 1.96 * (std / np.sqrt(n))
        return {"ci_lower": float(mean - margin), "ci_upper": float(mean + margin)}

    def _gather_from_path(
        self, base_path: str, group_folder: str
    ) -> Dict[str, Dict[str, List[float]]]:
        acc: Dict[str, Dict[str, List[float]]] = {
            algo: self._init_result_dict(self.metrics) for algo in self.algorithms
        }
        metric_map = {"rmse": "RMSE", "f1": "F1", "ndcg": "NDCG", "mae": "MAE"}

        for window in self.window_range:
            for exec_number in range(1, self.executions + 1):
                for algo in self.algorithms:
                    group = "constituent" if algo in self.constituent_algorithms else "hybrid"
                    path = f"{base_path}/{group}/window_{window}/execution_{exec_number}/{algo}_{group_folder}.csv"
                    if not os.path.exists(path):
                        continue
                    df = pd.read_csv(path)
                    if df.empty:
                        continue
                    row = df.iloc[0]
                    for m in self.metrics:
                        col = metric_map[m]
                        val = row.get(col)
                        if pd.notna(val):
                            acc[algo][m].append(float(val))
        return acc

    def run(self) -> None:
        rows: List[Dict[str, object]] = []

        gender_vuln = self._gather_from_path(self.fairness_gender_path, "group_female")
        activity_vuln = self._gather_from_path(self.fairness_activity_path, "group_low")

        for analysis_type, data in [
            ("vulnerable_gender", gender_vuln),
            ("vulnerable_activity", activity_vuln),
        ]:
            for algo, metric_map in data.items():
                for metric, values in metric_map.items():
                    if not values:
                        continue
                    stats = self._stats(values)
                    ci = self._ci95(stats["mean"], stats["std"], stats["n"])
                    rows.append(
                        {
                            "method": algo,
                            "metric": metric,
                            "analysis_type": analysis_type,
                            **stats,
                            **ci,
                        }
                    )

        if not rows:
            print("Nenhum dado encontrado para agregação de grupos vulneráveis.")
            return

        out_df = pd.DataFrame(rows)
        out_df = out_df[
            [
                "method",
                "metric",
                "analysis_type",
                "mean",
                "std",
                "median",
                "min",
                "max",
                "n",
                "ci_lower",
                "ci_upper",
            ]
        ].sort_values(by=["analysis_type", "metric", "mean"], ascending=[True, True, True])

        os.makedirs(os.path.dirname(self.output_path), exist_ok=True)
        out_df.to_csv(self.output_path, index=False)
        print(f"Resultados para grupos vulneráveis salvos em {self.output_path}")


if __name__ == "__main__":
    agg = VulnerableGroupAggregator()
    agg.run()
