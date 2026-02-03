import os
import tempfile
import unittest

import numpy as np
import pandas as pd

from finalResult import FinalResultAggregator


def _write_csv(path: str, data: pd.DataFrame) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    data.to_csv(path, index=False)


class FinalResultAggregatorTest(unittest.TestCase):
    def setUp(self):
        self.tmpdir = tempfile.TemporaryDirectory()
        self.geo_dir = os.path.join(self.tmpdir.name, "geo")
        self.gender_dir = os.path.join(self.tmpdir.name, "gender")
        self.activity_dir = os.path.join(self.tmpdir.name, "activity")
        self.output_path = os.path.join(self.tmpdir.name, "final.csv")

        # Configuração simplificada: 2 janelas, 2 métricas, 2 algoritmos
        self.metrics = ["f1", "rmse"]
        self.algos = ["AlgA", "AlgB"]

        self._build_georisk_samples()
        self._build_fairness_samples(self.gender_dir, prefix="gender", f1_vals=(0.3, 0.5), rmse_vals=(0.1, 0.2))
        self._build_fairness_samples(self.activity_dir, prefix="activity", f1_vals=(0.4, 0.6), rmse_vals=(0.2, 0.4))

    def tearDown(self):
        self.tmpdir.cleanup()

    def _build_georisk_samples(self):
        # GeoRisk: valores por algoritmo/métrica em 2 janelas
        values = {
            1: {"f1": [1.0, 2.0], "rmse": [2.0, 4.0]},
            2: {"f1": [3.0, 4.0], "rmse": [4.0, 6.0]},
        }
        for window, metric_map in values.items():
            for metric, vals in metric_map.items():
                df = pd.DataFrame({"algorithm": self.algos, "georisk": vals})
                _write_csv(f"{self.geo_dir}/window_{window}/{metric}.csv", df)

    def _build_fairness_samples(self, base_dir: str, prefix: str, f1_vals, rmse_vals):
        # Fairness CSVs: uma linha com RMSE,NDCG,F1,MAE
        for window, (f1_v, rmse_v) in enumerate(zip(f1_vals, rmse_vals), start=1):
            for group in ("constituent", "hybrid"):
                for algo in self.algos:
                    df = pd.DataFrame(
                        [{"RMSE": rmse_v, "NDCG": 0.0, "F1": f1_v, "MAE": 0.0}]
                    )
                    _write_csv(
                        f"{base_dir}/{group}/window_{window}/{algo}_absDiff_.csv", df
                    )

    def _run_aggregator(self):
        agg = FinalResultAggregator(
            georisk_path=self.geo_dir,
            fairness_gender_path=self.gender_dir,
            fairness_activity_path=self.activity_dir,
            output_path=self.output_path,
        )
        # Restringe janelas/métricas/algoritmos ao conjunto do teste
        agg.window_range = range(1, 3)
        agg.metrics = self.metrics
        agg.algorithms = self.algos
        agg.run()
        return pd.read_csv(self.output_path)

    def _assert_row(self, df, method, metric, analysis_type, expected):
        row = df[
            (df["method"] == method)
            & (df["metric"] == metric)
            & (df["analysis_type"] == analysis_type)
        ].iloc[0]
        for key, val in expected.items():
            self.assertAlmostEqual(row[key], val, places=6, msg=f"{method}-{metric}-{analysis_type}-{key}")

    def test_final_result_aggregation(self):
        df = self._run_aggregator()
        # Esperamos 2 métodos * 2 métricas * 3 análises = 12 linhas
        self.assertEqual(len(df), 12)

        # GeoRisk: AlgA f1 -> [1,3] => mean 2, std 1, median 2, min 1, max 3
        self._assert_row(
            df,
            "AlgA",
            "f1",
            "georisk",
            {
                "mean": 2.0,
                "std": 1.0,
                "median": 2.0,
                "min": 1.0,
                "max": 3.0,
                "ci_lower": 2.0 - 1.96 * 1.0 / np.sqrt(2),
                "ci_upper": 2.0 + 1.96 * 1.0 / np.sqrt(2),
            },
        )

        # Fairness gênero: AlgA f1 -> [0.3, 0.5]
        # Valores duplicados por const./hybrid => [0.3,0.3,0.5,0.5]
        self._assert_row(
            df,
            "AlgA",
            "f1",
            "fairness_gender",
            {
                "mean": 0.4,
                "std": np.std([0.3, 0.3, 0.5, 0.5]),
                "median": 0.4,
                "min": 0.3,
                "max": 0.5,
                "ci_lower": 0.4 - 1.96 * np.std([0.3, 0.3, 0.5, 0.5]) / np.sqrt(4),
                "ci_upper": 0.4 + 1.96 * np.std([0.3, 0.3, 0.5, 0.5]) / np.sqrt(4),
            },
        )

        # Fairness atividade: AlgB rmse -> [0.2, 0.4]
        # Valores duplicados por const./hybrid => [0.2,0.2,0.4,0.4]
        self._assert_row(
            df,
            "AlgB",
            "rmse",
            "fairness_activity",
            {
                "mean": 0.3,
                "std": np.std([0.2, 0.2, 0.4, 0.4]),
                "median": 0.3,
                "min": 0.2,
                "max": 0.4,
                "ci_lower": 0.3 - 1.96 * np.std([0.2, 0.2, 0.4, 0.4]) / np.sqrt(4),
                "ci_upper": 0.3 + 1.96 * np.std([0.2, 0.2, 0.4, 0.4]) / np.sqrt(4),
            },
        )


if __name__ == "__main__":
    unittest.main()
