import os
import tempfile
import unittest

import numpy as np
import pandas as pd

from recsys.aggregation.final_result_aggregator import FinalResultAggregator


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

        self.metrics = ["f1", "rmse"]
        self.algos = ["AlgA", "AlgB"]
        # Use 1 execution in test to match the path format the aggregator expects
        self.executions = 1

        self._build_georisk_samples()
        self._build_fairness_samples(self.gender_dir, f1_vals=(0.3, 0.5), rmse_vals=(0.1, 0.2))
        self._build_fairness_samples(self.activity_dir, f1_vals=(0.4, 0.6), rmse_vals=(0.2, 0.4))

    def tearDown(self):
        self.tmpdir.cleanup()

    def _build_georisk_samples(self):
        # GeoRisk: window_{w}/execution_{e}/{metric}.csv
        values = {
            1: {"f1": [1.0, 2.0], "rmse": [2.0, 4.0]},
            2: {"f1": [3.0, 4.0], "rmse": [4.0, 6.0]},
        }
        for window, metric_map in values.items():
            for metric, vals in metric_map.items():
                df = pd.DataFrame({"algorithm": self.algos, "georisk": vals})
                _write_csv(
                    f"{self.geo_dir}/window_{window}/execution_1/{metric}.csv", df
                )

    def _build_fairness_samples(self, base_dir: str, f1_vals, rmse_vals):
        # Fairness: {group}/window_{w}/execution_{e}/{algo}_absDiff_.csv
        for window, (f1_v, rmse_v) in enumerate(zip(f1_vals, rmse_vals), start=1):
            for group in ("constituent", "hybrid"):
                for algo in self.algos:
                    df = pd.DataFrame(
                        [{"RMSE": rmse_v, "NDCG": 0.0, "F1": f1_v, "MAE": 0.0}]
                    )
                    _write_csv(
                        f"{base_dir}/{group}/window_{window}/execution_1/{algo}_absDiff_.csv",
                        df,
                    )

    def _run_aggregator(self):
        agg = FinalResultAggregator(
            georisk_path=self.geo_dir,
            fairness_gender_path=self.gender_dir,
            fairness_activity_path=self.activity_dir,
            output_path=self.output_path,
            executions=self.executions,
        )
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
            self.assertAlmostEqual(
                row[key], val, places=6,
                msg=f"{method}-{metric}-{analysis_type}-{key}"
            )

    def test_final_result_aggregation(self):
        df = self._run_aggregator()
        # 2 methods * 2 metrics * 3 analysis types = 12 rows
        self.assertEqual(len(df), 12)

        # GeoRisk: AlgA f1 -> [1, 3] (windows 1 and 2, exec 1)
        self._assert_row(df, "AlgA", "f1", "georisk", {
            "mean": 2.0,
            "std": 1.0,
            "median": 2.0,
            "min": 1.0,
            "max": 3.0,
            "ci_lower": 2.0 - 1.96 * 1.0 / np.sqrt(2),
            "ci_upper": 2.0 + 1.96 * 1.0 / np.sqrt(2),
        })

        # Fairness gender AlgA f1: const+hybrid both write same value per window
        # window 1 f1=0.3 (x2 groups), window 2 f1=0.5 (x2 groups) => [0.3,0.3,0.5,0.5]
        vals = [0.3, 0.3, 0.5, 0.5]
        self._assert_row(df, "AlgA", "f1", "fairness_gender", {
            "mean": np.mean(vals),
            "min": 0.3,
            "max": 0.5,
        })

        # Fairness activity AlgB rmse: same pattern with rmse_vals=(0.2, 0.4)
        vals2 = [0.2, 0.2, 0.4, 0.4]
        self._assert_row(df, "AlgB", "rmse", "fairness_activity", {
            "mean": np.mean(vals2),
            "min": 0.2,
            "max": 0.4,
        })


if __name__ == "__main__":
    unittest.main()
