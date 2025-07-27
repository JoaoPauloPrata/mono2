import pandas as pd
import numpy as np
from sklearn import linear_model
from sklearn.linear_model import TweedieRegressor, BayesianRidge
from sklearn.ensemble import (BaggingRegressor, RandomForestRegressor, 
                            AdaBoostRegressor, GradientBoostingRegressor)
from sklearn.svm import LinearSVR
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_score, GridSearchCV, RandomizedSearchCV
from scipy.stats import uniform, randint
import os
from datetime import datetime

class RegressionMethodsWithFineTuning:
    def __init__(self):
        self.relativePath = "data/filtered_predictions/window_"
        self.scaler = StandardScaler()
        # Inicializa modelos base que serão otimizados
        self.initialize_base_models()

    def initialize_base_models(self):
        """
        Inicializa os modelos com configurações básicas
        """
        self.regBayesianRidge = BayesianRidge()
        self.regRidge = linear_model.Ridge()
        self.regTweedie = TweedieRegressor()
        self.regRandomForest = RandomForestRegressor(n_jobs=-1)
        self.regBagging = BaggingRegressor(n_jobs=-1)
        self.regAdaBoost = AdaBoostRegressor()
        self.regGradientBoosting = GradientBoostingRegressor()
        self.regLinearSVR = LinearSVR()

    def save_optimized_parameters(self, optimized_models, window_count):
        """
        Salva os parâmetros otimizados em um arquivo TXT
        """
        # Cria o diretório se não existir
        os.makedirs("data/optimized_parameters", exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"data/optimized_parameters/optimized_params_window_{window_count}_{timestamp}.txt"
        
        with open(filename, 'w') as f:
            f.write(f"Parâmetros Otimizados - Janela {window_count}\n")
            f.write(f"Data/Hora: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write("="*60 + "\n\n")
            
            for model_name, model in optimized_models.items():
                f.write(f"{model_name}:\n")
                f.write("-" * 30 + "\n")
                
                # Extrai os parâmetros do modelo
                params = model.get_params()
                for param, value in params.items():
                    f.write(f"  {param}: {value}\n")
                f.write("\n")
        
        print(f"Parâmetros salvos em: {filename}")

    def optimize_models(self, X_train, y_train):
        """
        Otimiza os hiperparâmetros de cada modelo usando RandomizedSearchCV
        """
        # Definição dos espaços de busca para cada modelo
        param_distributions = {
            'BayesianRidge': {
                'n_iter': randint(200, 1000),
                'alpha_1': uniform(1e-7, 1e-5),
                'alpha_2': uniform(1e-7, 1e-5),
                'lambda_1': uniform(1e-7, 1e-5),
                'lambda_2': uniform(1e-7, 1e-5)
            },
            'Ridge': {
                'alpha': uniform(0.1, 2.0),
                'max_iter': randint(500, 2000),
                'solver': ['auto', 'svd', 'cholesky', 'lsqr']
            },
            'Tweedie': {
                'power': uniform(1.0, 2.0),
                'alpha': uniform(0.1, 2.0),
                'max_iter': randint(500, 1500)
            },
            'RandomForest': {
                'n_estimators': randint(50, 300),  # Reduzido para evitar travamento
                'max_depth': randint(5, 25),
                'min_samples_split': randint(2, 10),
                'min_samples_leaf': randint(1, 5),
                'max_features': ['sqrt', 'log2', None]
            },
            'Bagging': {
                'n_estimators': randint(10, 100),
                'max_samples': uniform(0.5, 1.0),
                'max_features': uniform(0.5, 1.0),
                'bootstrap': [True, False]
            },
            'AdaBoost': {
                'n_estimators': randint(50, 200),
                'learning_rate': uniform(0.01, 1.0),
                'loss': ['linear', 'square', 'exponential']
            },
            'GradientBoosting': {
                'n_estimators': randint(50, 300),
                'learning_rate': uniform(0.01, 0.3),
                'max_depth': randint(3, 10),
                'subsample': uniform(0.6, 1.0),
                'min_samples_split': randint(2, 10)
            },
            'LinearSVR': {
                'epsilon': uniform(0.0, 0.5),
                'C': uniform(0.1, 2.0),
                'max_iter': randint(1000, 5000),
                'tol': uniform(1e-5, 1e-3)
            }
        }

        # Otimização de cada modelo
        optimized_models = {}
        
        for model_name, params in param_distributions.items():
            print(f"\nOtimizando {model_name}...")
            
            if model_name == 'BayesianRidge':
                base_model = self.regBayesianRidge
            elif model_name == 'Ridge':
                base_model = self.regRidge
            elif model_name == 'Tweedie':
                base_model = self.regTweedie
            # elif model_name == 'RandomForest':
            #     base_model = self.regRandomForest
            elif model_name == 'Bagging':
                base_model = self.regBagging
            elif model_name == 'AdaBoost':
                base_model = self.regAdaBoost
            elif model_name == 'GradientBoosting':
                base_model = self.regGradientBoosting
            elif model_name == 'LinearSVR':
                base_model = self.regLinearSVR

            # Usando RandomizedSearchCV para otimização com limite de iterações
            n_iter = 10 if model_name == 'RandomForest' else 15  # Menos iterações para RandomForest
            
            random_search = RandomizedSearchCV(
                estimator=base_model,
                param_distributions=params,
                n_iter=n_iter,
                cv=3,  # Reduzido para 3 folds para acelerar
                scoring='neg_mean_squared_error',
                n_jobs=1 if model_name == 'RandomForest' else -1,  # 1 job para RandomForest
                random_state=42,
                verbose=1  # Para mostrar progresso
            )
            
            try:
                random_search.fit(X_train, y_train)
                
                print(f"Melhores parâmetros para {model_name}:")
                print(random_search.best_params_)
                print(f"Melhor score: {-random_search.best_score_:.4f} (RMSE)") 
                
                optimized_models[model_name] = random_search.best_estimator_
            except Exception as e:
                print(f"Erro ao otimizar {model_name}: {e}")
                # Use o modelo base se a otimização falhar
                optimized_models[model_name] = base_model

        return optimized_models

    def preprocess_data(self, combined_ratings):
        """
        Pré-processa os dados aplicando normalização e tratamento de valores ausentes
        """
        # Normaliza os dados
        scaled_data = self.scaler.fit_transform(combined_ratings)
        # Trata valores ausentes
        return np.nan_to_num(scaled_data, nan=0.0, posinf=5.0, neginf=1.0)

    def evaluate_model(self, model, X, y):
        """
        Avalia o modelo usando validação cruzada
        """
        scores = cross_val_score(model, X, y, cv=5, scoring='neg_mean_squared_error')
        rmse_scores = np.sqrt(-scores)
        return np.mean(rmse_scores), np.std(rmse_scores)

    def ensemble_predictions(self, predictions_list):
        """
        Combina as predições usando média ponderada
        """
        weights = [0.15, 0.15, 0.12, 0.15, 0.12, 0.1, 0.13, 0.08]  # Pesos para cada modelo
        weighted_predictions = np.zeros_like(predictions_list[0])
        
        for pred, weight in zip(predictions_list, weights):
            weighted_predictions += pred * weight
            
        return weighted_predictions

    def loadAndPredict(self, window_count):
        recs_from_SVD = pd.read_csv(self.relativePath +str(window_count)+ "_constituent_methods__SVD.tsv", delimiter='\t')
        recs_from_BIAS = pd.read_csv(self.relativePath +str(window_count)+ "_constituent_methods__BIAS.tsv", delimiter='\t')
        recs_from_userKNN = pd.read_csv(self.relativePath +str(window_count)+ "_constituent_methods__userKNN.tsv", delimiter='\t')
        recs_from_itemKNN = pd.read_csv(self.relativePath +str(window_count)+ "_constituent_methods__itemKNN.tsv", delimiter='\t')
        recs_from_biasedMF = pd.read_csv(self.relativePath +str(window_count)+ "_constituent_methods__BIASEDMF.tsv", delimiter='\t')
        columns = ['user', 'item', 'prediction']
        recs_from_SVD = recs_from_SVD.sort_values('user')
        recs_from_BIAS = recs_from_BIAS.sort_values('user')
        recs_from_userKNN = recs_from_userKNN.sort_values('user')
        recs_from_itemKNN = recs_from_itemKNN.sort_values('user')
        recs_from_biasedMF = recs_from_biasedMF.sort_values('user')
       
        scikit_test_data = pd.read_csv('data/dataSplited/processed/test_to_get_regression_train_data_' + str(window_count) + "_.csv", sep=',')

        recs_from_SVD_to_train_scikit = pd.read_csv(self.relativePath +str(window_count)+ "_scikit_train__SVD.tsv", delimiter='\t')
        recs_from_BIAS_to_train_scikit = pd.read_csv(self.relativePath +str(window_count)+ "_scikit_train__BIAS.tsv", delimiter='\t')
        recs_from_userKNN_to_train_scikit = pd.read_csv(self.relativePath +str(window_count)+ "_scikit_train__userKNN.tsv", delimiter='\t')
        recs_from_itemKNN_to_train_scikit = pd.read_csv(self.relativePath +str(window_count)+ "_scikit_train__itemKNN.tsv", delimiter='\t')
        recs_from_biasedMF_to_train_scikit = pd.read_csv(self.relativePath +str(window_count)+ "_scikit_train__BIASEDMF.tsv", delimiter='\t')

        
        ratings_BIAS_to_use_in_prediction = recs_from_BIAS['prediction'].values
        ratings_SVD_to_use_in_prediction = recs_from_SVD['prediction'].values
        ratings_userKNN_to_use_in_prediction = recs_from_userKNN['prediction'].values
        ratings_itemKNN_to_use_in_prediction = recs_from_itemKNN['prediction'].values
        ratings_biasedMF_to_use_in_prediction = recs_from_biasedMF['prediction'].values
        print(ratings_BIAS_to_use_in_prediction)
        combined_ratings = [[r, s, t, u, v] for r, s, t, u, v in zip(ratings_BIAS_to_use_in_prediction, ratings_SVD_to_use_in_prediction, ratings_userKNN_to_use_in_prediction, ratings_itemKNN_to_use_in_prediction, ratings_biasedMF_to_use_in_prediction)]
        combined_ratings = np.nan_to_num(combined_ratings)

        ratings_BIAS_to_use_in_train = recs_from_BIAS_to_train_scikit['prediction'].values
        ratings_SVD_to_use_in_train = recs_from_SVD_to_train_scikit['prediction'].values
        ratings_userKNN_to_use_in_train = recs_from_userKNN_to_train_scikit['prediction'].values
        ratings_itemKNN_to_use_in_train = recs_from_itemKNN_to_train_scikit['prediction'].values
        ratings_biasedMF_to_use_in_train = recs_from_biasedMF_to_train_scikit['prediction'].values
        
        # Adicionando a linha que estava faltando
        original_ratings_scikit = scikit_test_data['rating'].values
        
        cobined_ratings_train = [[r, s, t, u, v] for r, s,t,u,v in zip(ratings_BIAS_to_use_in_train, ratings_SVD_to_use_in_train, ratings_userKNN_to_use_in_train, ratings_itemKNN_to_use_in_train, ratings_biasedMF_to_use_in_train)]
        cobined_ratings_train = np.nan_to_num(cobined_ratings_train)

        # Prepara os dados para otimização
        combined_ratings_train = self.preprocess_data(cobined_ratings_train)
        
        # Otimiza os modelos
        print("Iniciando otimização de parâmetros...")
        optimized_models = self.optimize_models(combined_ratings_train, original_ratings_scikit)
        
        # Salva os parâmetros otimizados
        self.save_optimized_parameters(optimized_models, window_count)
        
        # Atualiza os modelos com as versões otimizadas
        self.regBayesianRidge = optimized_models['BayesianRidge']
        self.regRidge = optimized_models['Ridge']
        self.regTweedie = optimized_models['Tweedie']
        self.regRandomForest = optimized_models['RandomForest']
        self.regBagging = optimized_models['Bagging']
        self.regAdaBoost = optimized_models['AdaBoost']
        self.regGradientBoosting = optimized_models['GradientBoosting']
        self.regLinearSVR = optimized_models['LinearSVR']

        # Faz as predições usando os modelos otimizados
        print("Fazendo predições com modelos otimizados...")
        predictedBayesianRidge = self.regBayesianRidge.predict(combined_ratings)
        predictedTweedie = self.regTweedie.predict(combined_ratings)
        predicted = self.regRidge.predict(combined_ratings)
        predictedRandomForest = self.regRandomForest.predict(combined_ratings)
        predictedBagging = self.regBagging.predict(combined_ratings)
        predictedAdaBoost = self.regAdaBoost.predict(combined_ratings)
        predictedGradientBoosting = self.regGradientBoosting.predict(combined_ratings)
        predictedLinearSVR = self.regLinearSVR.predict(combined_ratings)
        
        # Salva as predições
        predictedBayesianRidge = pd.DataFrame(predictedBayesianRidge)   
        predictedBayesianRidge.to_csv("data/HybridPredictions/window_"+str(window_count)+"_predictedBayesianRidge.tsv", sep='\t', index=False)
        predictedTweedie = pd.DataFrame(predictedTweedie)
        predictedTweedie.to_csv("data/HybridPredictions/window_"+str(window_count)+"_predictedTweedie.tsv", sep='\t', index=False)
        predicted = pd.DataFrame(predicted)
        predicted.to_csv("data/HybridPredictions/window_"+str(window_count)+"_predictedRidge.tsv", sep='\t', index=False)
        predictedRandomForest = pd.DataFrame(predictedRandomForest)
        predictedRandomForest.to_csv("data/HybridPredictions/window_"+str(window_count)+"_predictedRandomForest.tsv", sep='\t', index=False)
        predictedBagging = pd.DataFrame(predictedBagging)
        predictedBagging.to_csv("data/HybridPredictions/window_"+str(window_count)+"_predictedBagging.tsv", sep='\t', index=False)
        predictedAdaBoost = pd.DataFrame(predictedAdaBoost)
        predictedAdaBoost.to_csv("data/HybridPredictions/window_"+str(window_count)+"_predictedAdaBoost.tsv", sep='\t', index=False)
        predictedGradientBoosting = pd.DataFrame(predictedGradientBoosting)
        predictedGradientBoosting.to_csv("data/HybridPredictions/window_"+str(window_count)+"_predictedGradientBoosting.tsv", sep='\t', index=False)
        predictedLinearSVR = pd.DataFrame(predictedLinearSVR)
        predictedLinearSVR.to_csv("data/HybridPredictions/window_"+str(window_count)+"_predictedLinearSVR.tsv", sep='\t', index=False)

        print(f"Predições salvas para janela {window_count}")

