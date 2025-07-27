import pandas as pd
from scipy.stats import norm
import numpy as np
import math
from sklearn.metrics import mean_squared_error, ndcg_score, precision_recall_fscore_support

class Evaluator:
    @staticmethod
    def calculate_rmse(predictions, true_values):
        """
        Calcula o RMSE usando sklearn.
        """
        pred_df = pd.DataFrame(predictions, columns=['user', 'item', 'prediction'])
        true_df = pd.DataFrame(true_values, columns=['user', 'item', 'true_value'])
        merged = pd.merge(pred_df, true_df, on=['user', 'item'])
        return np.sqrt(mean_squared_error(merged['true_value'], merged['prediction']))

    @staticmethod
    def calculate_user_ndcg(predictions, true_values, k):
        """
        Calcula o NDCG para cada usuário individualmente e retorna a média.
        """
        pred_df = pd.DataFrame(predictions, columns=['user', 'item', 'prediction'])
        true_df = pd.DataFrame(true_values, columns=['user', 'item', 'true_value'])

        ndcgs = []
        for user in pred_df['user'].unique():
            user_pred = pred_df[pred_df['user'] == user]
            user_true = true_df[true_df['user'] == user]

            if user_true.empty:
                continue

            true_relevance = user_true.set_index('item').reindex(user_pred['item'], fill_value=0)['true_value']
            pred_scores = user_pred['prediction']

            ndcgs.append(ndcg_score([true_relevance], [pred_scores], k=k))

        return np.mean(ndcgs) if ndcgs else None

    @staticmethod
    def calculate_global_ndcg(predictions, true_values, k):
        """
        Calcula o NDCG global considerando todas as previsões e valores reais.
        """
        pred_df = pd.DataFrame(predictions, columns=['user', 'item', 'prediction'])
        true_df = pd.DataFrame(true_values, columns=['user', 'item', 'true_value'])

        merged = pd.merge(pred_df, true_df, on=['user', 'item'], how='outer').fillna(0)
        true_relevance = merged['true_value'].values
        pred_scores = merged['prediction'].values

        return ndcg_score([true_relevance], [pred_scores], k=k)

    @staticmethod
    def calculate_f1_user(predictions, true_values, threshold):
        """
        Calcula o F1 para cada usuário individualmente e retorna a média.
        """
        pred_df = pd.DataFrame(predictions, columns=['user', 'item', 'prediction'])
        true_df = pd.DataFrame(true_values, columns=['user', 'item', 'true_value'])

        f1_scores = []
        for user in pred_df['user'].unique():
            user_pred = pred_df[pred_df['user'] == user]
            user_true = true_df[true_df['user'] == user]

            if user_true.empty:
                continue

            true_labels = (user_true.set_index('item').reindex(user_pred['item'], fill_value=0)['true_value'] >= threshold).astype(int)
            pred_labels = (user_pred['prediction'] >= threshold).astype(int)

            _, _, f1, _ = precision_recall_fscore_support(true_labels, pred_labels, average='binary', zero_division=0)
            f1_scores.append(f1)

        return np.mean(f1_scores) if f1_scores else None

    @staticmethod
    def calculate_f1_global(predictions, true_values, threshold):
        """
        Calcula o F1 global considerando todas as previsões e valores reais.
        """
        pred_df = pd.DataFrame(predictions, columns=['user', 'item', 'prediction'])
        true_df = pd.DataFrame(true_values, columns=['user', 'item', 'true_value'])

        merged = pd.merge(pred_df, true_df, on=['user', 'item'], how='outer').fillna(0)
        true_labels = (merged['true_value'] >= threshold).astype(int)
        pred_labels = (merged['prediction'] >= threshold).astype(int)

        _, _, f1, _ = precision_recall_fscore_support(true_labels, pred_labels, average='binary', zero_division=0)
        return f1

    def evaluateAllMetricsForAllMethods(self, window_count, top_n=None):
        constituentAlgoritms = ["itemKNN", "BIAS", "userKNN", "SVD", "BIASEDMF"]
        hybridAlgoritms = ["BayesianRidge", "Tweedie", "Ridge", "RandomForest", "Bagging", "AdaBoost", "GradientBoosting", "LinearSVR"] 
        original_ratings = pd.read_csv(f"data/dataSplited/test_to_get_constituent_methods_{window_count}_.csv")
        print("============= CONSTITUENTS ==============")
        for constituent in constituentAlgoritms:
            recs_from_constituent = pd.read_csv(f"data/filtered_predictions/window_{window_count}_constituent_methods__{constituent}.tsv", delimiter='\t')
            recs_from_constituent = recs_from_constituent.groupby('user').filter(lambda x: len(x) >= 10)
            partialOriginal = original_ratings[original_ratings['user'].isin(recs_from_constituent['user'])]
            recs_from_constituent = recs_from_constituent.reset_index(drop=True)
            partialOriginal = partialOriginal.reset_index(drop=True)
            predictions = list(zip(recs_from_constituent['user'], recs_from_constituent['item'], recs_from_constituent['prediction']))
            truth = list(zip(partialOriginal['user'], partialOriginal['item'], partialOriginal['rating']))
            print(f"\nMétodo: {constituent}")
            print("F1 global:", Evaluator.calculate_f1_global(predictions, truth, threshold=3.5))

        print("\n============= HYBRIDS ==============")
        for hybrid in hybridAlgoritms:
            filePath = f"data/HybridPredictions/window_{window_count}_predicted{hybrid}.tsv"
            recs_from_hybrid = pd.read_csv(filePath, delimiter='\t')
            recs_from_hybrid = recs_from_hybrid.groupby('user').filter(lambda x: len(x) >= 10)
            partialOriginal = original_ratings[original_ratings['user'].isin(recs_from_hybrid['user'])]
            recs_from_hybrid = recs_from_hybrid.reset_index(drop=True)
            partialOriginal = partialOriginal.reset_index(drop=True)
            predictions = list(zip(recs_from_hybrid['user'], recs_from_hybrid['item'], recs_from_hybrid['prediction']))
            truth = list(zip(partialOriginal['user'], partialOriginal['item'], partialOriginal['rating']))
            print(f"\nMétodo: {hybrid}")
            print("F1 global:", Evaluator.calculate_f1_global(predictions, truth, threshold=3.5))

    def getGeoRisk(self, mat, alpha):
        ##### IMPORTANT
        # This function takes a matrix of number of rows as a number of queries, and the number of collumns as the number of systems.
        ##############
        numSystems = mat.shape[0]
        numQueries = mat.shape[1]


        Tj = np.array([0.0] * numQueries)
        Si = np.array([0.0] * numSystems)
        geoRisk = np.array([0.0] * numSystems)
        zRisk = np.array([0.0] * numSystems)
        mSi = np.array([0.0] * numSystems)

        for i in range(numSystems):
            Si[i] = np.sum(mat[:, i])
            mSi[i] = np.mean(mat[:, i])

        for j in range(numSystems):
            Tj[j] = np.sum(mat[j, :])

        N = np.sum(Tj)

        for i in range(numSystems):
            tempZRisk = 0
            for j in range(numQueries):
                eij = Si[i] * (Tj[j] / N)
                xij_eij = mat[i, j] - eij
                if eij != 0:
                    ziq = xij_eij / math.sqrt(eij)
                else:
                    ziq = 0
                if xij_eij < 0:
                    ziq = (1 + alpha) * ziq
                tempZRisk = tempZRisk + ziq
            zRisk[i] = tempZRisk

        c = numQueries
        for i in range(numSystems):
            ncd = norm.cdf(zRisk[i] / c)
            geoRisk[i] = math.sqrt((Si[i] / c) * ncd)

        return geoRisk
   