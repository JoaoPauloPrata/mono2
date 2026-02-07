import os
from typing import Dict, List, Iterable

import numpy as np
import pandas as pd


class QualityAggregator:
    """
    Agrega métricas de qualidade (RMSE, NDCG, F1, MAE) a partir dos arquivos
    MetricsForWindow{w}_{exec}.csv gerados por evaluateAllMetricsForAllMethods.
    Produz estatísticas (mean, std, median, min, max, n) por método e métrica.
    """

    def __init__(
        self,
        source_pattern: str = "./data/MetricsForMethods/MetricsForWindow{window}_{exec}.csv",
        output_path: str = "./data/MetricsForMethods/quality_results.csv",
        executions: int = 5,
    ) -> None:
        self.source_pattern = source_pattern
        self.output_path = output_path
        self.executions = executions
        self.metrics = ["RMSE", "NDCG", "F1", "MAE"]
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

    def _collect(self) -> Dict[str, Dict[str, List[float]]]:
        acc: Dict[str, Dict[str, List[float]]] = {}
        for window in self.window_range:
            for exec_number in range(1, self.executions + 1):
                path = self.source_pattern.format(window=window, exec=exec_number)
                if not os.path.exists(path):
                    continue
                df = pd.read_csv(path)
                if "method" not in df.columns:
                    continue
                for _, row in df.iterrows():
                    method = str(row["method"])
                    if method not in acc:
                        acc[method] = self._init_result_dict(self.metrics)
                    for metric in self.metrics:
                        if metric in row and pd.notna(row[metric]):
                            acc[method][metric].append(float(row[metric]))
        return acc

    def run(self) -> None:
        data = self._collect()
        rows: List[Dict[str, object]] = []
        for method, metric_map in data.items():
            for metric, values in metric_map.items():
                if not values:
                    continue
                stats = self._stats(values)
                ci = self._ci95(stats["mean"], stats["std"], stats["n"])
                rows.append(
                    {
                        "method": method,
                        "metric": metric.lower(),  # harmoniza com outros arquivos
                        **stats,
                        **ci,
                    }
                )

        if not rows:
            print("Nenhum dado encontrado para agregar.")
            return

        out_df = pd.DataFrame(rows)
        out_df = out_df[
            ["method", "metric", "mean", "std", "median", "min", "max", "n", "ci_lower", "ci_upper"]
        ].sort_values(by=["metric", "mean"], ascending=[True, True])

        os.makedirs(os.path.dirname(self.output_path), exist_ok=True)
        out_df.to_csv(self.output_path, index=False)
        print(f"Resultados de qualidade salvos em {self.output_path}")


if __name__ == "__main__":
    agg = QualityAggregator()
    agg.run()
