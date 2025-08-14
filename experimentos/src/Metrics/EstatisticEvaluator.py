import os
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from scipy import stats


class EstatisticEvaluator:
    CLASSICAL_METHODS: List[str] = [
        "itemKNN",
        "BIAS",
        "userKNN",
        "SVD",
        "BIASEDMF",
    ]

    HYBRID_METHODS: List[str] = [
        "BayesianRidge",
        "Tweedie",
        "Ridge",
        "RandomForest",
        "Bagging",
        "AdaBoost",
        "GradientBoosting",
        "LinearSVR",
    ]

    METRICS: List[str] = ["RMSE", "NDCG", "F1", "MAE"]

    # Para definir direção do que é melhor em cada métrica
    HIGHER_IS_BETTER = {
        "RMSE": False,
        "MAE": False,
        "NDCG": True,
        "F1": True,
    }

    def __init__(self, metrics_folder: str = "data/MetricsForMethods"):
        self.metrics_folder = metrics_folder

    def _read_metrics_file(self, window_number: int) -> pd.DataFrame:
        path = os.path.join(self.metrics_folder, f"MetricsForWindow{window_number}.csv")
        if not os.path.exists(path):
            raise FileNotFoundError(f"Arquivo de métricas não encontrado: {path}")

        df = pd.read_csv(path)
        # Garante coluna 'method'
        if 'method' not in df.columns:
            # Tenta promover índice a coluna de método
            df = pd.read_csv(path, index_col=0)
            df = df.reset_index().rename(columns={df.columns[0]: 'method'})

        # Mantém somente colunas relevantes
        keep_cols = ['method'] + [m for m in self.METRICS if m in df.columns]
        df = df[keep_cols]
        return df

    def _split_groups(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
        classic_df = df[df['method'].isin(self.CLASSICAL_METHODS)].copy()
        hybrid_df = df[df['method'].isin(self.HYBRID_METHODS)].copy()
        return classic_df, hybrid_df

    def _agg_by_group_per_window(self, df: pd.DataFrame) -> Dict[str, Dict[str, float]]:
        classic_df, hybrid_df = self._split_groups(df)
        summary: Dict[str, Dict[str, float]] = {metric: {} for metric in self.METRICS if metric in df.columns}

        for metric in list(summary.keys()):
            higher_better = self.HIGHER_IS_BETTER[metric]

            # Remove NaNs
            c_vals = classic_df[metric].dropna().values
            h_vals = hybrid_df[metric].dropna().values
            if len(c_vals) == 0 or len(h_vals) == 0:
                # Se faltar valores, ignora este metric nesta janela
                del summary[metric]
                continue

            if higher_better:
                best_classic = float(np.max(c_vals))
                best_hybrid = float(np.max(h_vals))
            else:
                best_classic = float(np.min(c_vals))
                best_hybrid = float(np.min(h_vals))

            mean_classic = float(np.mean(c_vals))
            mean_hybrid = float(np.mean(h_vals))

            summary[metric] = {
                'best_classic': best_classic,
                'best_hybrid': best_hybrid,
                'mean_classic': mean_classic,
                'mean_hybrid': mean_hybrid,
            }

        return summary

    def _collect_series_across_windows(
        self,
        windows: List[int]
    ):
        # Estruturas: {metric: {'best_classic': [], 'best_hybrid': [], 'mean_classic': [], 'mean_hybrid': []}}
        data: Dict[str, Dict[str, List[float]]] = {}

        for w in windows:
            try:
                df = self._read_metrics_file(w)
            except FileNotFoundError:
                continue

            per_window = self._agg_by_group_per_window(df)
            for metric, stats_map in per_window.items():
                if metric not in data:
                    data[metric] = {
                        'best_classic': [], 'best_hybrid': [],
                        'mean_classic': [], 'mean_hybrid': []
                    }
                for key in data[metric].keys():
                    data[metric][key].append(stats_map[key])

        return data

    @staticmethod
    def _paired_tests(x: np.ndarray, y: np.ndarray) -> Tuple[float, float, float, float]:
        # t pareado (duas caudas)
        t_stat, t_p = stats.ttest_rel(x, y, nan_policy='omit')
        # Wilcoxon (duas caudas). Necessita pelo menos 1 par não-zero
        try:
            w_stat, w_p = stats.wilcoxon(x, y, zero_method='wilcox', alternative='two-sided')
        except ValueError:
            w_stat, w_p = np.nan, np.nan
        return float(t_stat), float(t_p), float(w_stat), float(w_p)

    def _build_results_frame(
        self,
        data: Dict[str, Dict[str, List[float]]],
        comparison: str
    ) -> pd.DataFrame:
        """
        comparison: 'best' ou 'mean'
        """
        rows = []
        for metric, series_map in data.items():
            higher_better = self.HIGHER_IS_BETTER.get(metric, True)
            c_key = f"{comparison}_classic"
            h_key = f"{comparison}_hybrid"
            if c_key not in series_map or h_key not in series_map:
                continue
            x = np.array(series_map[h_key], dtype=float)
            y = np.array(series_map[c_key], dtype=float)
            # diferença no sentido de "melhora do híbrido"
            diff = x - y if higher_better else y - x

            t_stat, t_p, w_stat, w_p = self._paired_tests(x, y)
            rows.append({
                'metric': metric,
                'comparison': comparison,
                'n_windows': int(len(x)),
                'mean_diff_hybrid_vs_classic': float(np.mean(diff)),
                'std_diff': float(np.std(diff, ddof=1)) if len(diff) > 1 else 0.0,
                't_stat': t_stat,
                't_pvalue': t_p,
                'wilcoxon_stat': w_stat,
                'wilcoxon_pvalue': w_p,
                'higher_is_better': higher_better,
                'improved_windows': int(np.sum(diff > 0)),
            })
        return pd.DataFrame(rows)

    def run_validation(
        self,
        windows: List[int] = None,
        save_csv: bool = True
    ) -> pd.DataFrame:
        """
        Executa testes estatísticos (t pareado e Wilcoxon) comparando o desempenho
        de métodos híbridos vs clássicos por janela.

        - "best": compara melhor híbrido vs melhor clássico por janela
        - "mean": compara média do grupo híbrido vs média do grupo clássico por janela
        """
        if windows is None:
            windows = list(range(1, 21))

        data = self._collect_series_across_windows(windows)
        best_df = self._build_results_frame(data, comparison='best')
        mean_df = self._build_results_frame(data, comparison='mean')

        results = pd.concat([best_df, mean_df], ignore_index=True)

        if save_csv:
            os.makedirs(self.metrics_folder, exist_ok=True)
            out_path = os.path.join(self.metrics_folder, "StatisticalValidation.csv")
            results.to_csv(out_path, index=False)
            print(f"Validação estatística salva em: {out_path}")

        return results


if __name__ == "__main__":
    evaluator = EstatisticEvaluator()
    evaluator.run_validation()
