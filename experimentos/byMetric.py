import os
from typing import Dict, Iterable, List

import pandas as pd


class MetricMatrixBuilder:
    """
    Constrói matrizes (usuário x método) para cada métrica e janela,
    usando os arquivos gerados em ByUser (um CSV por algoritmo).
    """

    def __init__(
        self,
        input_path: str = "./data/MetricsForMethods/ByUser",
        output_path: str = "./data/MetricsForMethods/ByMetric",
    ) -> None:
        self.input_path = input_path.rstrip("/\\")
        self.output_path = output_path.rstrip("/\\")
        self.metrics = ["rmse", "f1", "ndcg", "mae"]
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

    @staticmethod
    def _read_algo_user_metrics(path: str) -> pd.DataFrame:
        df = pd.read_csv(path)
        expected_cols = {"user", "rmse", "f1", "ndcg", "mae"}
        missing = expected_cols.difference(df.columns)
        if missing:
            raise ValueError(f"Arquivo {path} está faltando colunas: {missing}")
        return df

    @staticmethod
    def _collect_users(dfs: Iterable[pd.DataFrame]) -> List[int]:
        all_users = set()
        for df in dfs:
            all_users.update(df["user"].unique().tolist())
        return sorted(all_users)

    def _build_matrix(
        self,
        algo_to_df: Dict[str, pd.DataFrame],
        metric: str,
        users: List[int],
        algorithms: List[str],
    ) -> pd.DataFrame:
        """
        Retorna df onde cada linha é um usuário e cada coluna (além da 1ª) é um algoritmo.
        """
        series_by_algo = {
            algo: df.set_index("user")[metric] for algo, df in algo_to_df.items()
        }
        rows = []
        for user in users:
            row = [user]
            for algo in algorithms:
                row.append(series_by_algo[algo].get(user, float("nan")))
            rows.append(row)

        columns = ["user"] + algorithms
        return pd.DataFrame(rows, columns=columns)

    @staticmethod
    def _save_matrix(df: pd.DataFrame, path: str) -> None:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        df.to_csv(path, index=False)

    def _process_all_algorithms(self) -> None:
        for window in range(1, 21):
            # Carrega todos os CSVs de usuários (constituent + hybrid)
            algo_to_df: Dict[str, pd.DataFrame] = {}
            algorithms_order: List[str] = []
            for group, algorithms in (
                ("constituent", self.constituent_algorithms),
                ("hybrid", self.hybrid_algorithms),
            ):
                for algo in algorithms:
                    file_path = f"{self.input_path}/{group}/window_{window}/{algo}.csv"
                    if os.path.exists(file_path):
                        algo_to_df[algo] = self._read_algo_user_metrics(file_path)
                        algorithms_order.append(algo)

            if not algo_to_df:
                continue

            users = self._collect_users(algo_to_df.values())

            for metric in self.metrics:
                matrix_df = self._build_matrix(
                    algo_to_df, metric, users, algorithms_order
                )
                out_path = f"{self.output_path}/window_{window}/{metric}.csv"
                self._save_matrix(matrix_df, out_path)
                print(
                    f"Matriz salva em {out_path} (metric={metric}, window={window}, users={len(users)}, algos={len(algo_to_df)})"
                )

    def build(self) -> None:
        self._process_all_algorithms()


if __name__ == "__main__":
    builder = MetricMatrixBuilder()
    builder.build()
