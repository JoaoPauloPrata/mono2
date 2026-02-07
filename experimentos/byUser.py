import os
from typing import Iterable, List

import pandas as pd

from src.Metrics.Evaluator import Evaluator


class UserMetricsByRating:
    def __init__(self, evaluator: Evaluator):   
        self.evaluator = evaluator

    def split_in_users(self, df) -> List[int]:
        users = df[['user']].drop_duplicates()
        return users["user"].unique().tolist()


    def evaluate_user(self, userId, df_truth, df_pred) -> dict:
        user_truth = df_truth[df_truth['user'] == userId]
        
        user_pred = df_pred[df_pred['user'] == userId]

        if user_truth.empty:
            return {}

        user_truth = user_truth.rename(columns={'rating': 'true_value'})

        rmse = self.evaluator.calculate_rmse(user_pred, user_truth)
        f1 = self.evaluator.calculate_f1_user(user_pred, user_truth, threshold=3.5)
        ndcg = self.evaluator.compute_ndcg(user_pred, user_truth, k=5)

        mae = self.evaluator.calculate_mae(user_pred, user_truth)
        return {"rmse": rmse, "f1": f1, "ndcg": ndcg, "mae": mae}
    

    def drop_users_with_less_than_n_ratings(self, df: pd.DataFrame, n: int) -> pd.DataFrame:
        print(df)
        user_counts = df['user'].value_counts()
        users_to_keep = user_counts[user_counts >= n].index
        return df[df['user'].isin(users_to_keep)]

    @staticmethod
    def _save_user_metrics(results: List[dict], output_file: str) -> None:
        """Save per-user metrics in the required column order."""
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        columns = ["user", "rmse", "f1", "ndcg", "mae"]
        df_out = pd.DataFrame(results, columns=columns)
        df_out.to_csv(output_file, index=False)
    
    def evaluateAllUsers(self, output_path: str = "./data/MetricsForMethods/ByUser") -> None:
        constituent_algorithms = ["SVD", "BIASEDMF", "NMF", "StochasticItemKNN"]
        hybrid_algorithms = ["BayesianRidge", "Tweedie", "Ridge", "RandomForest", "Bagging", "AdaBoost", "GradientBoosting", "LinearSVR"]
        output_path = output_path.rstrip("/\\")
       
    
        # for algorithm in constituent_algorithms:
        #     for execution_number in range(1, 6):
        #         for windowCount in range(1, 21):
        #             output_dir_constituent = f"{output_path}/constituent/window_{windowCount}/{algorithm}_execution{execution_number}.csv"
        #             df_pred_path = f"./data/filtered_predictions/window_{windowCount}_{execution_number}_constituent_methods_{algorithm}.tsv"
        #             truth_file_path = f"./data/windows/test_to_get_constituent_methods_{windowCount}.csv"
        #             df_pred = pd.read_csv(df_pred_path, sep="\t")
        #             df_pred = self.drop_users_with_less_than_n_ratings(df_pred, 5)
        #             userList = self.split_in_users(df_pred)
        #             truth_file = pd.read_csv(truth_file_path)
        #             results = []

        #             usersWithMoreThan10Ratings = truth_file['user'].value_counts()
        #             usersWithMoreThan10Ratings = usersWithMoreThan10Ratings[usersWithMoreThan10Ratings > 10].index.tolist()
        #             print(f"Total user count for {algorithm} in window {windowCount}: {len(userList)}")
        #             print(f"Total users with more than 10 ratings: {len(usersWithMoreThan10Ratings)}")

        #             for userId in userList:
        #                 user_metrics = self.evaluate_user(userId, truth_file, df_pred)
        #                 if user_metrics:
        #                     user_metrics["user"] = userId
        #                     results.append(user_metrics)
        #             self._save_user_metrics(results, output_dir_constituent)
        for algorithm in hybrid_algorithms:
            for execution_number in range(1, 6):
                for windowCount in range(1, 21):
                    output_dir_hybrid = f"{output_path}/hybrid/window_{windowCount}/{algorithm}_execution{execution_number}.csv"
                    df_pred_path = f"./data/HybridPredictions/window_{windowCount}_{execution_number}_predicted{algorithm}.tsv"
                    truth_file_path = f"./data/windows/test_to_get_constituent_methods_{windowCount}.csv"
                   
                    df_pred = pd.read_csv(df_pred_path, sep="\t")
                    df_pred = self.drop_users_with_less_than_n_ratings(df_pred, 5)
                 
                    userList = self.split_in_users(df_pred)
                    truth_file = pd.read_csv(truth_file_path)
                    results = []
                    for userId in userList:
                        user_metrics = self.evaluate_user(userId, truth_file, df_pred)
                        if user_metrics:
                            user_metrics["user"] = userId
                            results.append(user_metrics)
                    self._save_user_metrics(results, output_dir_hybrid)
    

metricFicator = UserMetricsByRating(Evaluator())
metricFicator.evaluateAllUsers(output_path="./data/MetricsForMethods/ByUser")
