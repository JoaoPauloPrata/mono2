import os
from typing import List

import numpy as np
import pandas as pd

from src.Metrics.Evaluator import Evaluator


class GeoRiskRunner:
    """
    Aplica o GeoRisk (Evaluator.getGeoRisk) nas matrizes geradas por byMetric.py.
    Cada matriz (window, metric) gera um CSV com uma linha por algoritmo (colunas da matriz).
    """

    def __init__(
        self,
        metric_path: str = "./data/MetricsForMethods/ByMetric",
        output_path: str = "./data/MetricsForMethods/GeoRisk",
        alpha: float = 0.05,
    ) -> None:
        self.metric_path = metric_path.rstrip("/\\")
        self.output_path = output_path.rstrip("/\\")
        self.metrics = ["rmse", "f1", "ndcg", "mae"]
        self.alpha = alpha
        self.evaluator = Evaluator()

    @staticmethod
    def _read_matrix(path: str) -> pd.DataFrame:
        df = pd.read_csv(path)
        if "user" not in df.columns:
            raise ValueError(f"Arquivo {path} precisa ter coluna 'user'.")
        return df

    @staticmethod
    def _to_numeric_matrix(df: pd.DataFrame) -> np.ndarray:
        # Remove coluna de usuário e converte resto para float (NaN vira 0)
        mat_df = df.drop(columns=["user"])
        return mat_df.astype(float).fillna(0.0).to_numpy()

    @staticmethod
    def _save_georisk(path: str, algorithms: List[str], scores: np.ndarray) -> None:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        out_df = pd.DataFrame({"algorithm": algorithms, "georisk": scores})
        out_df.to_csv(path, index=False)

    def run(self) -> None:
        for window in range(1, 21):
            for metric in self.metrics:
                matrix_path = f"{self.metric_path}/window_{window}/{metric}.csv"
                if not os.path.exists(matrix_path):
                    continue

                df = self._read_matrix(matrix_path)
                algo_names = [c for c in df.columns if c != "user"]
                mat = self._to_numeric_matrix(df)

                geo_scores = Evaluator.getGeoRisk(mat, self.alpha)

                out_path = f"{self.output_path}/window_{window}/{metric}.csv"
                self._save_georisk(out_path, algo_names, geo_scores)
                print(
                    f"GeoRisk salvo em {out_path} (window={window}, metric={metric}, algos={len(algo_names)})"
                )


if __name__ == "__main__":
    runner = GeoRiskRunner()
    runner.run()
