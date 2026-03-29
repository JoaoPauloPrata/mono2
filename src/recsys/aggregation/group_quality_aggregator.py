import os
from math import sqrt
from pathlib import Path
from typing import Dict, Iterable, List

import pandas as pd


class GroupQualityResultsLite:
    """
    Gera CSVs (um por analise e metrica) com medias e IC 95% das metricas
    separadas por grupo:
      - atividade: high (mais ativos) e low (menos ativos)
      - genero: male e female

    Saida: analysis_type_group_metric.csv contendo colunas
    analysis_type,group,method,metric,mean,ci_lower,ci_upper.
    """

    def __init__(
        self,
        fairness_base: str = "data/MetricsForMethods/Fairness",
        output_dir: str = "data/MetricsForMethods/fairness_group_means_split",
        executions: int = 5,
        window_range: Iterable[int] = range(1, 21),
    ) -> None:
        base_dir = Path(__file__).resolve().parents[3]
        self.fairness_base = (base_dir / fairness_base).resolve()
        self.output_dir = (base_dir / output_dir).resolve()
        self.executions = executions
        self.window_range = list(window_range)

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
        self.metrics = ["rmse", "ndcg", "f1", "mae"]
        self.higher_is_better = {"f1": True, "ndcg": True, "rmse": False, "mae": False}

    @staticmethod
    def _read_metrics_file(path: Path) -> Dict[str, float]:
        if not path.exists():
            return {}
        df = pd.read_csv(path)
        if df.empty:
            return {}
        row = df.iloc[0]
        metrics: Dict[str, float] = {}
        for m in ["RMSE", "NDCG", "F1", "MAE"]:
            val = row.get(m)
            if pd.notna(val):
                metrics[m.lower()] = float(val)
        return metrics

    def _gather_records(self) -> List[Dict[str, object]]:
        records: List[Dict[str, object]] = []

        def collect_for_analysis(analysis: str, folder: str, group_map: Dict[str, str]) -> None:
            for algo_type, algorithms in (("constituent", self.constituent_algorithms), ("hybrid", self.hybrid_algorithms)):
                for window in self.window_range:
                    for execution in range(1, self.executions + 1):
                        base = self.fairness_base / folder / algo_type / f"window_{window}" / f"execution_{execution}"
                        for group_key, group_label in group_map.items():
                            for algo in algorithms:
                                metrics = self._read_metrics_file(base / f"{algo}_group_{group_key}.csv")
                                if not metrics:
                                    continue
                                for metric, value in metrics.items():
                                    records.append(
                                        {
                                            "analysis_type": analysis,
                                            "group": group_label,
                                            "method": algo,
                                            "metric": metric,
                                            "value": value,
                                        }
                                    )

        collect_for_analysis(
            analysis="fairness_activity",
            folder="kmeans",
            group_map={"low": "low", "high": "high"},
        )
        collect_for_analysis(
            analysis="fairness_gender",
            folder="gender",
            group_map={"female": "female", "male": "male"},
        )
        return records

    @staticmethod
    def _ci95(mean: float, std: float, n: int) -> Dict[str, float]:
        if n <= 1:
            return {"ci_lower": None, "ci_upper": None}
        margin = 1.96 * (std / sqrt(n))
        return {"ci_lower": mean - margin, "ci_upper": mean + margin}

    def _aggregate(self, records: List[Dict[str, object]]) -> pd.DataFrame:
        if not records:
            return pd.DataFrame(columns=["analysis_type", "group", "method", "metric", "mean", "ci_lower", "ci_upper"])

        df = pd.DataFrame(records)
        rows: List[Dict[str, object]] = []

        for keys, subset in df.groupby(["analysis_type", "group", "method", "metric"]):
            analysis_type, group, method, metric = keys
            values = subset["value"].astype(float)
            n = len(values)
            mean_val = values.mean()
            std_val = values.std(ddof=1) if n > 1 else 0.0
            ci = self._ci95(mean_val, std_val, n)
            rows.append(
                {
                    "analysis_type": analysis_type,
                    "group": group,
                    "method": method,
                    "metric": metric,
                    "mean": mean_val,
                    "ci_lower": ci["ci_lower"],
                    "ci_upper": ci["ci_upper"],
                }
            )

        return pd.DataFrame(rows)

    def run(self) -> None:
        records = self._gather_records()
        aggregated = self._aggregate(records)

        if aggregated.empty:
            print("Nenhum dado encontrado para processar.")
            return

        os.makedirs(self.output_dir, exist_ok=True)

        for (analysis_type, metric), subset in aggregated.groupby(["analysis_type", "metric"]):
            ascending = not self.higher_is_better.get(metric, True)
            ordered = subset.sort_values(by="mean", ascending=ascending)
            filename = f"{analysis_type}_{metric}.csv"
            out_path = self.output_dir / filename
            ordered.to_csv(out_path, index=False)
            print(f"Gerado (grupo/lite): {out_path} ({len(ordered)} linhas)")


if __name__ == "__main__":
    splitter = GroupQualityResultsLite()
    splitter.run()
