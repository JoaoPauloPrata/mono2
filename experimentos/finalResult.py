import os
from typing import Dict, List, Iterable

import numpy as np
import pandas as pd


class FinalResultAggregator:
    """
    Agrega GeoRisk, fairness de gênero e de grupo (kmeans/activity) ao longo das janelas,
    produzindo métricas de média, desvio padrão, mediana, mínimo e máximo por método e métrica.
    """

    def __init__(
        self,
        georisk_path: str = "./data/MetricsForMethods/GeoRisk",
        fairness_gender_path: str = "./data/MetricsForMethods/Fairness/gender",
        fairness_activity_path: str = "./data/MetricsForMethods/Fairness/kmeans",
        output_path: str = "./data/MetricsForMethods/final_results.csv",
        executions: int = 5,
    ) -> None:
        self.georisk_path = georisk_path.rstrip("/\\")
        self.fairness_gender_path = fairness_gender_path.rstrip("/\\")
        self.fairness_activity_path = fairness_activity_path.rstrip("/\\")
        self.output_path = output_path
        self.executions = executions
        
        self.metrics = ["rmse", "f1", "ndcg", "mae"]
        self.algorithms = [
            "StochasticItemKNN",
            "NMF",
            "SVD",
            "BIASEDMF",
            "BayesianRidge",
            "Tweedie",
            "Ridge",
            "RandomForest",
            "Bagging",
            "AdaBoost",
            "GradientBoosting",
            "LinearSVR",
        ]
        self.window_range = range(1, 21)

    @staticmethod
    @staticmethod
    def _stats(values: List[float]) -> Dict[str, float]:
        arr = np.array(values, dtype=float)
        return {
            "mean": float(np.mean(arr)),
            "std": float(np.std(arr)),
            "median": float(np.median(arr)),
            "min": float(np.min(arr)),
            "max": float(np.max(arr)),
            "n": int(len(arr)),
        }

    @staticmethod
    def _ci95(mean: float, std: float, n: int) -> Dict[str, float]:
        if n <= 0:
            return {"ci_lower": None, "ci_upper": None}
        # 95% CI usando z=1.96 (amostras pequenas ok para uso exploratório)
        margin = 1.96 * (std / np.sqrt(n))
        return {"ci_lower": float(mean - margin), "ci_upper": float(mean + margin)}

    @staticmethod
    def _init_result_dict(metrics: Iterable[str]) -> Dict[str, List[float]]:
        return {m: [] for m in metrics}

    def _gather_georisk(self) -> Dict[str, Dict[str, List[float]]]:
        # method -> metric -> list of values over windows/execs
        acc: Dict[str, Dict[str, List[float]]] = {
            algo: self._init_result_dict(self.metrics) for algo in self.algorithms
        }
        for window in self.window_range:
            for exec_number in range(1, self.executions + 1):
                for metric in self.metrics:
                    path = f"{self.georisk_path}/window_{window}/execution_{exec_number}/{metric}.csv"
                    if not os.path.exists(path):
                        continue
                    df = pd.read_csv(path)
                    if "algorithm" not in df.columns or "georisk" not in df.columns:
                        continue
                    for _, row in df.iterrows():
                        algo = row["algorithm"]
                        if algo not in acc:
                            acc[algo] = self._init_result_dict(self.metrics)
                        val = row["georisk"]
                        if pd.notna(val):
                            acc[algo][metric].append(float(val))
        return acc

    def _gather_fairness(self, base_path: str) -> Dict[str, Dict[str, List[float]]]:
        acc: Dict[str, Dict[str, List[float]]] = {
            algo: self._init_result_dict(self.metrics) for algo in self.algorithms
        }
        metric_map = {"rmse": "RMSE", "ndcg": "NDCG", "f1": "F1", "mae": "MAE"}
        for window in self.window_range:
            for exec_number in range(1, self.executions + 1):
                for group in ("constituent", "hybrid"):
                    for algo in self.algorithms:
                        path = f"{base_path}/{group}/window_{window}/execution_{exec_number}/{algo}_absDiff_.csv"
                        if not os.path.exists(path):
                            continue
                        df = pd.read_csv(path)
                        if df.empty:
                            continue
                        row = df.iloc[0]
                        for m in self.metrics:
                            col = metric_map[m]
                            if col in row and pd.notna(row[col]):
                                acc[algo][m].append(float(row[col]))
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

        # GeoRisk
        georisk_data = self._gather_georisk()
        rows.extend(self._accumulate_rows(georisk_data, "georisk"))

        # Fairness gender
        fairness_gender = self._gather_fairness(self.fairness_gender_path)
        rows.extend(self._accumulate_rows(fairness_gender, "fairness_gender"))

        # Fairness activity/group (kmeans)
        fairness_activity = self._gather_fairness(self.fairness_activity_path)
        rows.extend(self._accumulate_rows(fairness_activity, "fairness_activity"))

        if not rows:
            print("Nenhum dado encontrado para agregação.")
            return

        out_df = pd.DataFrame(rows)
        # Ordena para facilitar leitura
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
                "ci_lower",
                "ci_upper",
            ]
        ]
        os.makedirs(os.path.dirname(self.output_path), exist_ok=True)
        out_df.to_csv(self.output_path, index=False)
        print(f"Resultados agregados salvos em {self.output_path}")


if __name__ == "__main__":
    aggregator = FinalResultAggregator()
    aggregator.run()
