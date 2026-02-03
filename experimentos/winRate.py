import os
from typing import Dict, List, Tuple

import pandas as pd


class WinRateCalculator:
    """
    Calcula o ranking top-5 por janela e métrica (georisk, fairness gênero, fairness atividade),
    considerando métodos híbridos e constituintes em um único ranking.
    Também gera um resumo com a ordem dos top-5 por janela e a frequência de aparição.
    """

    def __init__(
        self,
        georisk_path: str = "./data/MetricsForMethods/GeoRisk",
        fairness_gender_path: str = "./data/MetricsForMethods/Fairness/gender",
        fairness_activity_path: str = "./data/MetricsForMethods/Fairness/kmeans",
        output_dir: str = "./data/MetricsForMethods/winrate",
    ) -> None:
        self.georisk_path = georisk_path.rstrip("/\\")
        self.fairness_gender_path = fairness_gender_path.rstrip("/\\")
        self.fairness_activity_path = fairness_activity_path.rstrip("/\\")
        self.output_dir = output_dir.rstrip("/\\")

        self.metrics = ["rmse", "f1", "ndcg", "mae"]
        self.algorithms = [
            "itemKNN",
            "BIAS",
            "userKNN",
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
    def _best_is_max(analysis_type: str) -> bool:
        # GeoRisk quanto maior melhor; fairness (abs diff) quanto menor melhor
        return analysis_type == "georisk"

    def _load_georisk(self, window: int, metric: str) -> pd.DataFrame:
        path = f"{self.georisk_path}/window_{window}/{metric}.csv"
        if not os.path.exists(path):
            return pd.DataFrame(columns=["algorithm", "value"])
        df = pd.read_csv(path)
        return df.rename(columns={"georisk": "value"})[["algorithm", "value"]]

    def _load_fairness(self, base_path: str, window: int) -> pd.DataFrame:
        rows = []
        metric_map = {"rmse": "RMSE", "f1": "F1", "ndcg": "NDCG", "mae": "MAE"}
        for algo in self.algorithms:
            group = "constituent" if algo in ["itemKNN", "BIAS", "userKNN", "SVD", "BIASEDMF"] else "hybrid"
            path = f"{base_path}/{group}/window_{window}/{algo}_absDiff_.csv"
            if not os.path.exists(path):
                continue
            df = pd.read_csv(path)
            if df.empty:
                continue
            row = df.iloc[0]
            for m, col in metric_map.items():
                if col in row:
                    rows.append({"algorithm": algo, "metric": m, "value": float(row[col])})
        return pd.DataFrame(rows)

    def _rank_top5(self, df: pd.DataFrame, analysis_type: str, metric: str) -> List[Tuple[str, float]]:
        if df.empty:
            return []
        df_metric = df[df["metric"] == metric] if "metric" in df.columns else df
        if df_metric.empty:
            return []
        ascending = not self._best_is_max(analysis_type)
        ranked = df_metric.sort_values(by="value", ascending=ascending)
        return list(ranked[["algorithm", "value"]].head(5).itertuples(index=False, name=None))

    def _save_window_top(self, analysis_type: str, metric: str, window: int, top5: List[Tuple[str, float]]):
        out_dir = f"{self.output_dir}/{analysis_type}/{metric}"
        os.makedirs(out_dir, exist_ok=True)
        path = f"{out_dir}/window_{window}.csv"
        pd.DataFrame(top5, columns=["algorithm", "value"]).to_csv(path, index=False)

    def _save_summary(self, analysis_type: str, metric: str, rankings: Dict[int, List[Tuple[str, float]]]):
        rows = []
        counts: Dict[str, int] = {algo: 0 for algo in self.algorithms}
        for window, top in rankings.items():
            ranks = [algo for algo, _ in top]
            for algo in ranks:
                counts[algo] += 1
            row = {"window": window}
            for i, algo in enumerate(ranks, start=1):
                row[f"rank{i}"] = algo
            rows.append(row)

        summary_df = pd.DataFrame(rows).sort_values("window")
        out_dir = f"{self.output_dir}/{analysis_type}"
        os.makedirs(out_dir, exist_ok=True)
        summary_path = f"{out_dir}/summary_{metric}.csv"
        summary_df.to_csv(summary_path, index=False)

        counts_df = pd.DataFrame(
            [{"algorithm": algo, "appearances": cnt} for algo, cnt in counts.items() if cnt > 0]
        ).sort_values(by="appearances", ascending=False)
        counts_path = f"{out_dir}/counts_{metric}.csv"
        counts_df.to_csv(counts_path, index=False)

        print(f"Resumo gerado: {summary_path} ; Contagens: {counts_path}")

    def run(self) -> None:
        for analysis_type in ["georisk", "fairness_gender", "fairness_activity"]:
            for metric in self.metrics:
                rankings: Dict[int, List[Tuple[str, float]]] = {}
                for window in self.window_range:
                    if analysis_type == "georisk":
                        df = self._load_georisk(window, metric)
                    elif analysis_type == "fairness_gender":
                        df = self._load_fairness(self.fairness_gender_path, window)
                    else:
                        df = self._load_fairness(self.fairness_activity_path, window)

                    top5 = self._rank_top5(df, analysis_type, metric)
                    if top5:
                        rankings[window] = top5
                        self._save_window_top(analysis_type, metric, window, top5)

                if rankings:
                    self._save_summary(analysis_type, metric, rankings)


if __name__ == "__main__":
    calc = WinRateCalculator()
    calc.run()
