import os
import pandas as pd
from typing import Tuple, Dict, Optional

from Evaluator import Evaluator


class GroupMetricsDiffCalculator:
    """
    Calcula métricas (RMSE, NDCG, F1, MAE) para dois grupos de usuários
    a partir de arquivos de predições (TSV) e avaliações reais (CSV/TSV),
    salva os resultados por grupo e a diferença absoluta entre grupos.
    """

    def __init__(self, k_ndcg: int = 5, f1_threshold: float = 3.5):
        self.k_ndcg = k_ndcg
        self.f1_threshold = f1_threshold
        self.evaluator = Evaluator()

    @staticmethod
    def _read_predictions_tsv(path: str) -> pd.DataFrame:
        df = pd.read_csv(path, sep='\t')
        # garante colunas
        expected = {'user', 'item', 'prediction'}
        if not expected.issubset(df.columns):
            raise ValueError(f"Arquivo de predições deve conter colunas: {expected}")
        return df[['user', 'item', 'prediction']]

    @staticmethod
    def _read_truth(path: str) -> pd.DataFrame:
        # aceita CSV ou TSV; tenta detectar separador
        sep = '\t' if path.lower().endswith('.tsv') else ','
        df = pd.read_csv(path, sep=sep)
        # tenta normalizar nome do rating
        if 'true_value' in df.columns:
            rating_col = 'true_value'
        elif 'rating' in df.columns:
            rating_col = 'rating'
        else:
            raise ValueError("Arquivo de avaliações reais deve conter 'true_value' ou 'rating'.")
        return df.rename(columns={rating_col: 'true_value'})[['user', 'item', 'true_value']]

    @staticmethod
    def _read_group_ids(path: str) -> pd.Series:
        # aceita csv com uma coluna contendo ids de usuário
        df = pd.read_csv(path)
        # heurística: tenta achar coluna chamada 'user' ou 'user_id' ou a primeira numérica
        for col in ['user', 'user_id', 'userid', 'uid', 'id']:
            if col in df.columns:
                return df[col].dropna().astype(int)
        # fallback: primeira coluna
        first_col = df.columns[0]
        return df[first_col].dropna().astype(int)

    @staticmethod
    def _filter_by_users(df: pd.DataFrame, users: pd.Series, cols: Tuple[str, ...]) -> pd.DataFrame:
        filtered = df[df['user'].isin(set(users))]
        return filtered[list(cols)].reset_index(drop=True)

    def _compute_metrics(self, preds: pd.DataFrame, truth: pd.DataFrame) -> Dict[str, Optional[float]]:
        rmse = self.evaluator.calculate_rmse(preds, truth)
        ndcg = self.evaluator.compute_ndcg(preds, truth, self.k_ndcg)
        f1 = self.evaluator.calculate_f1_global(preds, truth, self.f1_threshold)
        mae = self.evaluator.calculate_mae(preds, truth)
        return {"RMSE": rmse, "NDCG": ndcg, "F1": f1, "MAE": mae}

    @staticmethod
    def _save_metrics_csv(path: str, metrics: Dict[str, Optional[float]]):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        header = "RMSE,NDCG,F1,MAE\n"
        line = f"{metrics['RMSE']},{metrics['NDCG']},{metrics['F1']},{metrics['MAE']}\n"
        with open(path, 'w', encoding='utf-8') as f:
            f.write(header)
            f.write(line)

    @staticmethod
    def _abs_diff_metrics(m1: Dict[str, Optional[float]], m2: Dict[str, Optional[float]]):
        def ad(a, b):
            if a is None or b is None:
                return None
            return abs(float(a) - float(b))
        return {
            'RMSE': ad(m1.get('RMSE'), m2.get('RMSE')),
            'NDCG': ad(m1.get('NDCG'), m2.get('NDCG')),
            'F1': ad(m1.get('F1'), m2.get('F1')),
            'MAE': ad(m1.get('MAE'), m2.get('MAE')),
        }

    def drop_users_with_less_then_n_ratings(self, df: pd.DataFrame, n: int) -> pd.DataFrame:
        user_counts = df['user'].value_counts()
        valid_users = user_counts[user_counts >= n].index
        return df[df['user'].isin(valid_users)].reset_index(drop=True)

    def run(self,
            predictions_tsv: str,
            truth_path: str,
            group_a_csv: str,
            group_b_csv: str,
            out_group_a_csv: str,
            out_group_b_csv: str,
            out_absdiff_csv: str) -> None:
        preds = self._read_predictions_tsv(predictions_tsv)
        truth = self._read_truth(truth_path)
       
        users_a = self._read_group_ids(group_a_csv)
        users_b = self._read_group_ids(group_b_csv)

        preds_a = self._filter_by_users(preds, users_a, ('user', 'item', 'prediction'))
        truth_a = self._filter_by_users(truth, users_a, ('user', 'item', 'true_value'))
        
        preds_a = self.drop_users_with_less_then_n_ratings(preds_a, 5)
        truth_a = self.drop_users_with_less_then_n_ratings(truth_a, 5)
        
        preds_b = self._filter_by_users(preds, users_b, ('user', 'item', 'prediction'))
        truth_b = self._filter_by_users(truth, users_b, ('user', 'item', 'true_value'))
        
        preds_b = self.drop_users_with_less_then_n_ratings(preds_b, 5)
        truth_b = self.drop_users_with_less_then_n_ratings(truth_b, 5)


        metrics_a = self._compute_metrics(preds_a, truth_a)
        metrics_b = self._compute_metrics(preds_b, truth_b)

        self._save_metrics_csv(out_group_a_csv, metrics_a)
        self._save_metrics_csv(out_group_b_csv, metrics_b)

        absdiff = self._abs_diff_metrics(metrics_a, metrics_b)
        self._save_metrics_csv(out_absdiff_csv, absdiff)

__all__ = ["GroupMetricsDiffCalculator"]

calculator = GroupMetricsDiffCalculator(k_ndcg=5, f1_threshold=3.5)



def kmeansGroupCalculator():
    constituent_algorithms = ["SVD", "BIASEDMF", "NMF", "StochasticItemKNN"]
    hybrid_algorithms = ["BayesianRidge", "Tweedie", "Ridge", "RandomForest", "Bagging", "AdaBoost", "GradientBoosting", "LinearSVR"]

    for windowCount in range(1, 21):
        for execution_number in range(1, 6):
            for algorithm in constituent_algorithms:
                path = f"../../data/filtered_predictions/window_{windowCount}_{execution_number}_constituent_methods_{algorithm}.tsv"
                truth_file = f"../../data/windows/test_to_get_constituent_methods_{windowCount}.csv"
                group_a_file = f"../../data/windows/kmeansGroup/constituent/window_{windowCount}/window_{windowCount}_group_high.csv"
                group_b_file = f"../../data/windows/kmeansGroup/constituent/window_{windowCount}/window_{windowCount}_group_low.csv"
                out_a_file = f"../../data/MetricsForMethods/Fairness/kmeans/constituent/window_{windowCount}/execution_{execution_number}/{algorithm}_group_high.csv"
                out_b_file = f"../../data/MetricsForMethods/Fairness/kmeans/constituent/window_{windowCount}/execution_{execution_number}/{algorithm}_group_low.csv"
                out_diff_file = f"../../data/MetricsForMethods/Fairness/kmeans/constituent/window_{windowCount}/execution_{execution_number}/{algorithm}_absDiff_.csv"

                calculator.run(
                    predictions_tsv=path,
                    truth_path=truth_file,
                    group_a_csv=group_a_file,
                    group_b_csv=group_b_file,
                    out_group_a_csv=out_a_file,
                    out_group_b_csv=out_b_file,
                    out_absdiff_csv=out_diff_file
                )


    for windowCount in range(1, 21):
        for execution_number in range(1, 6):
            for algorithm in hybrid_algorithms:
                path = f"../../data/HybridPredictions/window_{windowCount}_{execution_number}_predicted{algorithm}.tsv"
                truth_file = f"../../data/windows/test_to_get_constituent_methods_{windowCount}.csv"
                group_a_file = f"../../data/windows/kmeansGroup/hybrid/window_{windowCount}/window_{windowCount}_group_high.csv"
                group_b_file = f"../../data/windows/kmeansGroup/hybrid/window_{windowCount}/window_{windowCount}_group_low.csv"
                out_a_file = f"../../data/MetricsForMethods/Fairness/kmeans/hybrid/window_{windowCount}/execution_{execution_number}/{algorithm}_group_high.csv"
                out_b_file = f"../../data/MetricsForMethods/Fairness/kmeans/hybrid/window_{windowCount}/execution_{execution_number}/{algorithm}_group_low.csv"
                out_diff_file = f"../../data/MetricsForMethods/Fairness/kmeans/hybrid/window_{windowCount}/execution_{execution_number}/{algorithm}_absDiff_.csv"
                calculator.run(
                    predictions_tsv=path,
                    truth_path=truth_file,
                    group_a_csv=group_a_file,
                    group_b_csv=group_b_file,
                    out_group_a_csv=out_a_file,
                    out_group_b_csv=out_b_file,
                    out_absdiff_csv=out_diff_file
                )

def genderGroupCalculator():
    constituent_algorithms = ["SVD", "BIASEDMF", "NMF", "StochasticItemKNN"]
    hybrid_algorithms = ["BayesianRidge", "Tweedie", "Ridge", "RandomForest", "Bagging", "AdaBoost", "GradientBoosting", "LinearSVR"]

    for windowCount in range(1, 21):
        for execution_number in range(1, 6):
            for algorithm in constituent_algorithms:
                path = f"../../data/filtered_predictions/window_{windowCount}_{execution_number}_constituent_methods_{algorithm}.tsv"
                truth_file = f"../../data/windows/test_to_get_constituent_methods_{windowCount}.csv"
                group_a_file = f"../../data/windows/gender/constituent/window_{windowCount}/female/window_{windowCount}_female.csv"
                group_b_file = f"../../data/windows/gender/constituent/window_{windowCount}/male/window_{windowCount}_male.csv"      
                out_a_file = f"../../data/MetricsForMethods/Fairness/gender/constituent/window_{windowCount}/execution_{execution_number}/{algorithm}_group_female.csv"
                out_b_file = f"../../data/MetricsForMethods/Fairness/gender/constituent/window_{windowCount}/execution_{execution_number}/{algorithm}_group_male.csv"
                out_diff_file = f"../../data/MetricsForMethods/Fairness/gender/constituent/window_{windowCount}/execution_{execution_number}/{algorithm}_absDiff_.csv"
                calculator.run(
                    predictions_tsv=path,
                    truth_path=truth_file,
                    group_a_csv=group_a_file,
                    group_b_csv=group_b_file,
                    out_group_a_csv=out_a_file,
                    out_group_b_csv=out_b_file,
                    out_absdiff_csv=out_diff_file
                )
        
    for windowCount in range(1, 21):
        for execution_number in range(1, 6):
            for algorithm in hybrid_algorithms:
                path = f"../../data/HybridPredictions/window_{windowCount}_{execution_number}_predicted{algorithm}.tsv"
                truth_file = f"../../data/windows/test_to_get_constituent_methods_{windowCount}.csv"
                group_a_file = f"../../data/windows/gender/hybrid/window_{windowCount}/female/window_{windowCount}_female.csv"
                group_b_file = f"../../data/windows/gender/hybrid/window_{windowCount}/male/window_{windowCount}_male.csv"
                out_a_file = f"../../data/MetricsForMethods/Fairness/gender/hybrid/window_{windowCount}/execution_{execution_number}/{algorithm}_group_female.csv"
                out_b_file = f"../../data/MetricsForMethods/Fairness/gender/hybrid/window_{windowCount}/execution_{execution_number}/{algorithm}_group_male.csv"
                out_diff_file = f"../../data/MetricsForMethods/Fairness/gender/hybrid/window_{windowCount}/execution_{execution_number}/{algorithm}_absDiff_.csv"
                calculator.run(
                    predictions_tsv=path,
                    truth_path=truth_file,
                    group_a_csv=group_a_file,
                    group_b_csv=group_b_file,
                    out_group_a_csv=out_a_file,
                    out_group_b_csv=out_b_file,
                    out_absdiff_csv=out_diff_file
                )

if __name__ == "__main__":
    kmeansGroupCalculator()
    genderGroupCalculator()
