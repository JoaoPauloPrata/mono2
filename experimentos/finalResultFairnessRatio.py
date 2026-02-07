import os
from typing import Dict, List, Iterable

import numpy as np
import pandas as pd


class FairnessRatioAggregator:
    """
    Agrega os arquivos de razão gerados por reasonCalculator.py
    (FairnessRatio) ao longo das janelas e execuções.

    Produz estatísticas (mean, std, median, min, max, n, ci) por método e métrica
    para cada análise: gender e kmeans (activity).
    """

    def __init__(
        self,
        ratio_base: str = "./data/MetricsForMethods/FairnessRatio",
        output_path: str = "./data/MetricsForMethods/fairness_ratio_results.csv",
        executions: int = 5,
    ) -> None:
        self.ratio_base = ratio_base.rstrip("/\\")
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

    def _gather(self, analysis: str) -> Dict[str, Dict[str, List[float]]]:
        acc: Dict[str, Dict[str, List[float]]] = {
            algo: self._init_result_dict(self.metrics) for algo in self.algorithms
        }
        for window in self.window_range:
            for exec_number in range(1, self.executions + 1):
                group = "constituent" if analysis == "gender" else "constituent"
                # Constituents
                for algo in self.constituent_algorithms:
                    path = f"{self.ratio_base}/{analysis}/constituent/window_{window}/execution_{exec_number}/{algo}_ratio_.csv"
                    if os.path.exists(path):
                        df = pd.read_csv(path)
                        if not df.empty:
                            row = df.iloc[0]
                            for m in self.metrics:
                                key = m.upper() if m in ("rmse", "mae") else m.upper()
                                val = row.get(key)
                                if pd.notna(val):
                                    acc[algo][m].append(float(val))

                # Hybrids
                for algo in self.hybrid_algorithms:
                    path = f"{self.ratio_base}/{analysis}/hybrid/window_{window}/execution_{exec_number}/{algo}_ratio_.csv"
                    if os.path.exists(path):
                        df = pd.read_csv(path)
                        if not df.empty:
                            row = df.iloc[0]
                            for m in self.metrics:
                                key = m.upper() if m in ("rmse", "mae") else m.upper()
                                val = row.get(key)
                                if pd.notna(val):
                                    acc[algo][m].append(float(val))
        return acc

    def _accumulate_rows(
        self, data: Dict[str, Dict[str, List[float]]], analysis_type: str
    ) -> List[Dict[str, object]]:
        rows: List[Dict[str, object]] = []
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
        return rows

    def run(self) -> None:
        rows: List[Dict[str, object]] = []

        gender_data = self._gather("gender")
        rows.extend(self._accumulate_rows(gender_data, "fairness_ratio_gender"))

        kmeans_data = self._gather("kmeans")
        rows.extend(self._accumulate_rows(kmeans_data, "fairness_ratio_activity"))

        if not rows:
            print("Nenhum dado encontrado para agregação.")
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
        ].sort_values(by=["analysis_type", "metric", "method"])

        os.makedirs(os.path.dirname(self.output_path), exist_ok=True)
        out_df.to_csv(self.output_path, index=False)
        print(f"Resultados agregados (Fairness Ratio) salvos em {self.output_path}")


if __name__ == "__main__":
    agg = FairnessRatioAggregator()
    agg.run()
