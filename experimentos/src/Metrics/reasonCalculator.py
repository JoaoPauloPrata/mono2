import os
import pandas as pd
from typing import Tuple, Dict, Optional

from Evaluator import Evaluator


class GroupMetricsRatioCalculator:
    """
    Calcula a razão (%) do desempenho do grupo menos favorecido em relação ao mais favorecido.
    - Grupos mais favorecidos: usuários mais ativos / gênero masculino
    - Grupos menos favorecidos: usuários menos ativos / gênero feminino
    Para métricas onde menor é melhor (RMSE, MAE) a razão usa (melhor / pior),
    para métricas onde maior é melhor (F1, NDCG) usa (pior / melhor).
    """

    def __init__(self, k_ndcg: int = 5, f1_threshold: float = 3.5):
        self.k_ndcg = k_ndcg
        self.f1_threshold = f1_threshold
        self.evaluator = Evaluator()
        self.higher_is_better = {"RMSE": False, "MAE": False, "NDCG": True, "F1": True, "rmse": False, "mae": False, "ndcg": True, "f1": True}

    @staticmethod
    def _read_predictions_tsv(path: str) -> pd.DataFrame:
        df = pd.read_csv(path, sep="\t")
        expected = {"user", "item", "prediction"}
        if not expected.issubset(df.columns):
            raise ValueError(f"Arquivo de predições deve conter colunas: {expected}")
        return df[["user", "item", "prediction"]]

    @staticmethod
    def _read_truth(path: str) -> pd.DataFrame:
        sep = "\t" if path.lower().endswith(".tsv") else ","
        df = pd.read_csv(path, sep=sep)
        if "true_value" in df.columns:
            rating_col = "true_value"
        elif "rating" in df.columns:
            rating_col = "rating"
        else:
            raise ValueError("Arquivo de avaliações reais deve conter 'true_value' ou 'rating'.")
        return df.rename(columns={rating_col: "true_value"})[["user", "item", "true_value"]]

    @staticmethod
    def _read_group_ids(path: str) -> pd.Series:
        df = pd.read_csv(path)
        for col in ["user", "user_id", "userid", "uid", "id"]:
            if col in df.columns:
                return df[col].dropna().astype(int)
        first_col = df.columns[0]
        return df[first_col].dropna().astype(int)

    @staticmethod
    def _filter_by_users(df: pd.DataFrame, users: pd.Series, cols: Tuple[str, ...]) -> pd.DataFrame:
        filtered = df[df["user"].isin(set(users))]
        return filtered[list(cols)].reset_index(drop=True)

    @staticmethod
    def _save_metrics_csv(path: str, metrics: Dict[str, Optional[float]]):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        header = "RMSE,NDCG,F1,MAE\n"
        line = f"{metrics.get('RMSE')},{metrics.get('NDCG')},{metrics.get('F1')},{metrics.get('MAE')}\n"
        with open(path, "w", encoding="utf-8") as f:
            f.write(header)
            f.write(line)

    def _ratio_metrics(self, m_less: Dict[str, Optional[float]], m_more: Dict[str, Optional[float]]) -> Dict[str, Optional[float]]:
        ratios = {}
        for metric, higher in self.higher_is_better.items():
            a = m_less.get(metric)
            b = m_more.get(metric)
            if a is None or b is None or b == 0 or a == 0:
                ratios[metric] = None
                continue
            if higher:
                ratios[metric] = float(a) / float(b) * 100.0
            else:
                # menor é melhor -> usa melhor/pior
                ratios[metric] = float(b) / float(a) * 100.0
        return ratios

    def _read_group_metrics_file(self, path: str) -> Dict[str, Optional[float]]:
        df = pd.read_csv(path)
        if df.empty:
            return {}
        row = df.iloc[0]
        return {
            "RMSE": row.get("RMSE"),
            "NDCG": row.get("NDCG"),
            "F1": row.get("F1"),
            "MAE": row.get("MAE"),
        }

    def _run_single(
        self,
        less_metrics_path: str,
        more_metrics_path: str,
        out_ratio_path: str,
    ) -> None:
        """
        Lê métricas já calculadas (Fairness) e salva apenas o arquivo de razão em FairnessRatio.
        """
        if not (os.path.exists(less_metrics_path) and os.path.exists(more_metrics_path)):
            return

        metrics_less = self._read_group_metrics_file(less_metrics_path)
        metrics_more = self._read_group_metrics_file(more_metrics_path)

        ratios = self._ratio_metrics(metrics_less, metrics_more)
        self._save_metrics_csv(out_ratio_path, ratios)


calculator = GroupMetricsRatioCalculator(k_ndcg=5, f1_threshold=3.5)


def kmeansRatioCalculator():
    constituent_algorithms = ["SVD", "BIASEDMF", "NMF", "StochasticItemKNN"]
    hybrid_algorithms = ["BayesianRidge", "Tweedie", "Ridge", "RandomForest", "Bagging", "AdaBoost", "GradientBoosting", "LinearSVR"]

    for windowCount in range(1, 21):
        for execution_number in range(1, 6):
            for algorithm in constituent_algorithms:
                path = f"../../data/filtered_predictions/window_{windowCount}_{execution_number}_constituent_methods_{algorithm}.tsv"
                truth_file = f"../../data/windows/test_to_get_constituent_methods_{windowCount}.csv"
                group_less_file = f"../../data/windows/kmeansGroup/constituent/window_{windowCount}/window_{windowCount}_group_low.csv"
                group_more_file = f"../../data/windows/kmeansGroup/constituent/window_{windowCount}/window_{windowCount}_group_high.csv"
                metrics_less_path = f"../../data/MetricsForMethods/Fairness/kmeans/constituent/window_{windowCount}/execution_{execution_number}/{algorithm}_group_low.csv"
                metrics_more_path = f"../../data/MetricsForMethods/Fairness/kmeans/constituent/window_{windowCount}/execution_{execution_number}/{algorithm}_group_high.csv"
                out_ratio_file = f"../../data/MetricsForMethods/FairnessRatio/kmeans/constituent/window_{windowCount}/execution_{execution_number}/{algorithm}_ratio_.csv"

                calculator._run_single(
                    less_metrics_path=metrics_less_path,
                    more_metrics_path=metrics_more_path,
                    out_ratio_path=out_ratio_file,
                )

            for algorithm in hybrid_algorithms:
                path = f"../../data/HybridPredictions/window_{windowCount}_{execution_number}_predicted{algorithm}.tsv"
                truth_file = f"../../data/windows/test_to_get_constituent_methods_{windowCount}.csv"
                group_less_file = f"../../data/windows/kmeansGroup/hybrid/window_{windowCount}/window_{windowCount}_group_low.csv"
                group_more_file = f"../../data/windows/kmeansGroup/hybrid/window_{windowCount}/window_{windowCount}_group_high.csv"
                metrics_less_path = f"../../data/MetricsForMethods/Fairness/kmeans/hybrid/window_{windowCount}/execution_{execution_number}/{algorithm}_group_low.csv"
                metrics_more_path = f"../../data/MetricsForMethods/Fairness/kmeans/hybrid/window_{windowCount}/execution_{execution_number}/{algorithm}_group_high.csv"
                out_ratio_file = f"../../data/MetricsForMethods/FairnessRatio/kmeans/hybrid/window_{windowCount}/execution_{execution_number}/{algorithm}_ratio_.csv"

                calculator._run_single(
                    less_metrics_path=metrics_less_path,
                    more_metrics_path=metrics_more_path,
                    out_ratio_path=out_ratio_file,
                )


def genderRatioCalculator():
    constituent_algorithms = ["SVD", "BIASEDMF", "NMF", "StochasticItemKNN"]
    hybrid_algorithms = ["BayesianRidge", "Tweedie", "Ridge", "RandomForest", "Bagging", "AdaBoost", "GradientBoosting", "LinearSVR"]

    for windowCount in range(1, 21):
        for execution_number in range(1, 6):
            for algorithm in constituent_algorithms:
                path = f"../../data/filtered_predictions/window_{windowCount}_{execution_number}_constituent_methods_{algorithm}.tsv"
                truth_file = f"../../data/windows/test_to_get_constituent_methods_{windowCount}.csv"
                group_less_file = f"../../data/windows/gender/constituent/window_{windowCount}/female/window_{windowCount}_female.csv"
                group_more_file = f"../../data/windows/gender/constituent/window_{windowCount}/male/window_{windowCount}_male.csv"
                metrics_less_path = f"../../data/MetricsForMethods/Fairness/gender/constituent/window_{windowCount}/execution_{execution_number}/{algorithm}_group_female.csv"
                metrics_more_path = f"../../data/MetricsForMethods/Fairness/gender/constituent/window_{windowCount}/execution_{execution_number}/{algorithm}_group_male.csv"
                out_ratio_file = f"../../data/MetricsForMethods/FairnessRatio/gender/constituent/window_{windowCount}/execution_{execution_number}/{algorithm}_ratio_.csv"

                calculator._run_single(
                    less_metrics_path=metrics_less_path,
                    more_metrics_path=metrics_more_path,
                    out_ratio_path=out_ratio_file,
                )

            for algorithm in hybrid_algorithms:
                path = f"../../data/HybridPredictions/window_{windowCount}_{execution_number}_predicted{algorithm}.tsv"
                truth_file = f"../../data/windows/test_to_get_constituent_methods_{windowCount}.csv"
                group_less_file = f"../../data/windows/gender/hybrid/window_{windowCount}/female/window_{windowCount}_female.csv"
                group_more_file = f"../../data/windows/gender/hybrid/window_{windowCount}/male/window_{windowCount}_male.csv"
                metrics_less_path = f"../../data/MetricsForMethods/Fairness/gender/hybrid/window_{windowCount}/execution_{execution_number}/{algorithm}_group_female.csv"
                metrics_more_path = f"../../data/MetricsForMethods/Fairness/gender/hybrid/window_{windowCount}/execution_{execution_number}/{algorithm}_group_male.csv"
                out_ratio_file = f"../../data/MetricsForMethods/FairnessRatio/gender/hybrid/window_{windowCount}/execution_{execution_number}/{algorithm}_ratio_.csv"

                calculator._run_single(
                    less_metrics_path=metrics_less_path,
                    more_metrics_path=metrics_more_path,
                    out_ratio_path=out_ratio_file,
                )


if __name__ == "__main__":
    kmeansRatioCalculator()
    genderRatioCalculator()
