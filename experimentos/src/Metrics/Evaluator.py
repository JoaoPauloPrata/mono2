import os
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

    @staticmethod
    def calculate_mae(predictions, true_values):
        pred_df = pd.DataFrame(predictions, columns=['user', 'item', 'prediction'])
        true_df = pd.DataFrame(true_values, columns=['user', 'item', 'true_value'])
        merged = pd.merge(pred_df, true_df, on=['user', 'item'])
        return float(np.mean(np.abs(merged['true_value'] - merged['prediction'])))

    def evaluateAllMetricsForAllMethods(self, window_count, execution,  top_n=None):
        constituent_algorithms = ["itemKNN", "BIAS", "userKNN", "SVD", "BIASEDMF"]
        hybrid_algorithms = ["BayesianRidge", "Tweedie", "Ridge", "RandomForest", "Bagging", "AdaBoost", "GradientBoosting", "LinearSVR"]

        # Carrega ground truth
        truth_df = pd.read_csv(f"data/windows/test_to_get_constituent_methods_{window_count}_.csv")
        truth_df = truth_df.rename(columns={"rating": "true_value"})

        # NDCG@k
        k = 10 if top_n is None else int(top_n)

        # Resultados agregados
        results = []

        def compute_metrics_from_frames(pred_df: pd.DataFrame, truth_subset: pd.DataFrame, method_name: str):
            # Filtra usuários com pelo menos 10 itens
            pred_df = pred_df.groupby('user').filter(lambda x: len(x) >= 10)
            truth_subset = truth_subset[truth_subset['user'].isin(pred_df['user'])]

            if pred_df.empty or truth_subset.empty:
                return

            pred_df = pred_df.reset_index(drop=True)
            truth_subset = truth_subset.reset_index(drop=True)

            # Merge para RMSE/MAE
            merged = pd.merge(pred_df, truth_subset, on=['user', 'item'])
            if merged.empty:
                return

            rmse = float(np.sqrt(mean_squared_error(merged['true_value'], merged['prediction'])))
            mae = float(np.mean(np.abs(merged['true_value'] - merged['prediction'])))

            # NDCG por usuário
            ndcgs = []
            for user_id, g in pred_df.groupby('user'):
                true_user = truth_subset[truth_subset['user'] == user_id]
                if true_user.empty:
                    continue
                aligned = g.merge(true_user, on='item', how='left').fillna({'true_value': 0})
                ndcgs.append(ndcg_score([aligned['true_value'].values], [aligned['prediction'].values], k=k))
            ndcg_mean = float(np.mean(ndcgs)) if ndcgs else None

            # F1 global (threshold fixo 3.5)
            predictions_list = list(zip(merged['user'], merged['item'], merged['prediction']))
            truth_list = list(zip(merged['user'], merged['item'], merged['true_value']))
            f1 = float(Evaluator.calculate_f1_global(predictions_list, truth_list, threshold=3.5))

            results.append({
                'method': method_name,
                'RMSE': rmse,
                'NDCG': ndcg_mean,
                'F1': f1,
                'MAE': mae,
            })

        # Constituents
        for constituent in constituent_algorithms:
            path = f"data/filtered_predictions/window_{window_count}_execution_{execution}_constituent_methods__{constituent}.tsv"
            recs_df = pd.read_csv(path, delimiter='\t')
            recs_df = recs_df[['user', 'item', 'prediction']]
            compute_metrics_from_frames(recs_df, truth_df, constituent)

        # Hybrids: alinhar arquivos simples (apenas coluna de score) com um template
        # Usar BIAS como template (mesma ordenação usada na geração das predições híbridas)
        template_path = f"data/filtered_predictions/window_{window_count}_execution_{execution}_constituent_methods__BIAS.tsv"
        template_df = pd.read_csv(template_path, delimiter='\t').sort_values('user').reset_index(drop=True)
        template_df = template_df[['user', 'item']]

        for hybrid in hybrid_algorithms:
            file_path = f"data/HybridPredictions/window_{window_count}_execution_{execution}_predicted{hybrid}.tsv"
            hyb_df = pd.read_csv(file_path, delimiter='\t')

            # Se não houver colunas user/item, constrói usando o template
            if not set(['user', 'item']).issubset(hyb_df.columns):
                # Primeira coluna contém as predições
                scores = hyb_df.iloc[:, 0].values
                if len(scores) != len(template_df):
                    # Tenta alinhar sem sort se tamanhos não batem
                    template_alt = pd.read_csv(template_path, delimiter='\t')
                    template_alt = template_alt[['user', 'item']]
                    if len(scores) == len(template_alt):
                        template_use = template_alt
                    else:
                        # Não é possível alinhar, pula método
                        continue
                else:
                    template_use = template_df
                pred_df = pd.DataFrame({
                    'user': template_use['user'].values,
                    'item': template_use['item'].values,
                    'prediction': scores
                })
            else:
                # Já veio no formato esperado
                pred_df = hyb_df[['user', 'item', 'prediction']]

            compute_metrics_from_frames(pred_df, truth_df, hybrid)

        # Salva CSV agregado
        os.makedirs('data/MetricsForMethods', exist_ok=True)
        out_path = f"data/MetricsForMethods/MetricsForWindow_{window_count}_Execution_{execution}.csv"
        out_df = pd.DataFrame(results).set_index('method')
        out_df = out_df.loc[[*constituent_algorithms, *hybrid_algorithms]]
        out_df.to_csv(out_path)
        print(f"Métricas salvas em: {out_path}")

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
   