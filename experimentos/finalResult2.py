import os
from typing import Dict, List, Iterable

import numpy as np
import pandas as pd


# t crítico para IC 95% (bicaudal), graus de liberdade = n - 1
_T_CRITICAL_95 = {
    1: 12.706,
    2: 4.303,
    3: 3.182,
    4: 2.776,
    5: 2.571,
    6: 2.447,
    7: 2.365,
    8: 2.306,
    9: 2.262,
    10: 2.228,
    11: 2.201,
    12: 2.179,
    13: 2.160,
    14: 2.145,
    15: 2.131,
    16: 2.120,
    17: 2.110,
    18: 2.101,
    19: 2.093,
    20: 2.086,
    21: 2.080,
    22: 2.074,
    23: 2.069,
    24: 2.064,
    25: 2.060,
    26: 2.056,
    27: 2.052,
    28: 2.048,
    29: 2.045,
}


class FinalResultAggregator:
    """
    Agrega GeoRisk, fairness de gênero e fairness por atividade (kmeans) ao longo das janelas,
    produzindo estatísticas (média, desvio, mediana, min, max) e IC 95% por método e métrica.
    """

    def __init__(
        self,
        georisk_path: str = "./data/MetricsForMethods/GeoRisk",
        fairness_gender_path: str = "./data/MetricsForMethods/Fairness/gender",
        fairness_activity_path: str = "./data/MetricsForMethods/Fairness/kmeans",
        output_path: str = "./data/MetricsForMethods/final_results.csv",
    ) -> None:
        self.georisk_path = georisk_path.rstrip("/\\")
        self.fairness_gender_path = fairness_gender_path.rstrip("/\\")
        self.fairness_activity_path = fairness_activity_path.rstrip("/\\")
        self.output_path = output_path

        self.metrics = ["rmse", "f1", "ndcg", "mae"]

        # Separa por tipo para evitar misturar fairness de folders diferentes
        self.constituent_algorithms = ["itemKNN", "BIAS", "userKNN", "SVD", "BIASEDMF"]
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
    def _stats(values: List[float]) -> Dict[str, float]:
        arr = np.array(values, dtype=float)
        # std amostral (ddof=1) se tiver pelo menos 2 observações
        std = float(np.std(arr, ddof=1)) if len(arr) > 1 else 0.0
        return {
            "mean": float(np.mean(arr)),
            "std": std,
            "median": float(np.median(arr)),
            "min": float(np.min(arr)),
            "max": float(np.max(arr)),
            "n": int(len(arr)),
        }

    @staticmethod
    def _ci95(mean: float, std: float, n: int) -> Dict[str, float]:
        """
        Intervalo de confiança de 95% para a média.
        - Usa t de Student para amostras pequenas (n < 30)
        - Usa aproximação normal (z=1.96) para n >= 30
        """
        if n <= 1:
            return {"ci_lower": None, "ci_upper": None, "ci_method": None}

        sem = std / np.sqrt(n)  # erro padrão da média

        if n < 30:
            df = n - 1
            t_crit = _T_CRITICAL_95.get(df, 1.96)  # fallback defensivo
            margin = t_crit * sem
            method = "t-student"
        else:
            margin = 1.96 * sem
            method = "normal-z"

        return {
            "ci_lower": float(mean - margin),
            "ci_upper": float(mean + margin),
            "ci_method": method,
        }

    @staticmethod
    def _init_result_dict(metrics: Iterable[str]) -> Dict[str, List[float]]:
        return {m: [] for m in metrics}

    def _gather_georisk(self) -> Dict[str, Dict[str, List[float]]]:
        # method -> metric -> list of values over windows
        acc: Dict[str, Dict[str, List[float]]] = {
            algo: self._init_result_dict(self.metrics) for algo in self.algorithms
        }

        for window in self.window_range:
            for metric in self.metrics:
                path = f"{self.georisk_path}/window_{window}/{metric}.csv"
                if not os.path.exists(path):
                    continue

                df = pd.read_csv(path)
                if "algorithm" not in df.columns or "georisk" not in df.columns:
                    continue

                for _, row in df.iterrows():
                    algo = str(row["algorithm"]).strip()
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

        def group_for_algo(algo: str) -> str:
            return "constituent" if algo in self.constituent_algorithms else "hybrid"

        for window in self.window_range:
            for algo in self.algorithms:
                group = group_for_algo(algo)
                path = f"{base_path}/{group}/window_{window}/{algo}_absDiff_.csv"
                if not os.path.exists(path):
                    continue

                df = pd.read_csv(path)
                if df.empty:
                    continue

                # Se for um summary de 1 linha, ok:
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
                "n",
                "ci_lower",
                "ci_upper",
                "ci_method",
            ]
        ].sort_values(by=["analysis_type", "metric", "method"], ascending=True)

        os.makedirs(os.path.dirname(self.output_path), exist_ok=True)
        out_df.to_csv(self.output_path, index=False)
        print(f"Resultados agregados salvos em {self.output_path}")


if __name__ == "__main__":
    aggregator = FinalResultAggregator()
    aggregator.run()
