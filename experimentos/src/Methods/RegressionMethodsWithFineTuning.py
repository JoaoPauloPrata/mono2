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
import ast
from glob import glob
from datetime import datetime

class RegressionMethodsWithFineTuning:
    def __init__(self):
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

    def load_parameters(self, window_count):
        """
        Carrega os parâmetros otimizados de um arquivo TXT
        """
        # Prioriza arquivo exato; caso não exista, busca o mais recente com timestamp
        filename_exact = f"data/optimized_parameters/optimized_params_window_{window_count}.txt"

        candidate_path = None
        if os.path.exists(filename_exact):
            candidate_path = filename_exact
        else:
            pattern = f"data/optimized_parameters/optimized_params_window_{window_count}_*.txt"
            matches = sorted(glob(pattern))
            if matches:
                candidate_path = matches[-1]

        if not candidate_path:
            print(f"Arquivo de parâmetros para window {window_count} não encontrado.")
            return None

        optimized_models_params = {}

        with open(candidate_path, 'r') as f:
            lines = f.readlines()
            current_model = None

            for raw_line in lines:
                line = raw_line.strip()
                if not line:
                    continue
                if line.endswith(':') and '-' not in line:
                    # Cabeçalho do modelo
                    current_model = line[:-1]
                    optimized_models_params[current_model] = {}
                    continue
                if set(line) <= {'-'}:
                    # linha separadora "-----"
                    continue
                if current_model and ':' in line:
                    # linha de parâmetro
                    param, value = line.split(':', 1)
                    param = param.strip()
                    value_str = value.strip()
                    # Converte para tipo nativo quando possível
                    try:
                        parsed_value = ast.literal_eval(value_str)
                    except Exception:
                        parsed_value = value_str
                    optimized_models_params[current_model][param] = parsed_value
        return optimized_models_params

    def save_optimized_parameters(self, optimized_models, window_count):
        """
        Salva os parâmetros otimizados em um arquivo TXT
        """
        # Cria o diretório se não existir
        os.makedirs("data/optimized_parameters", exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"data/optimized_parameters/optimized_params_window_{window_count}_{timestamp}.txt"
        
        with open(filename, 'w') as f:
            for model_name, model in optimized_models.items():
                f.write(f"{model_name}:\n")
                f.write("-" * 30 + "\n")
                # Extrai os parâmetros do modelo
                params = model.get_params()
                for param, value in params.items():
                    f.write(f"  {param}: {value}\n")
                f.write("\n")
        
        print(f"Parâmetros salvos em: {filename}")
        return filename

    def optimize_models(self, X_train, y_train):
        """
        Otimiza os hiperparâmetros de cada modelo usando RandomizedSearchCV
        """
        # Definição dos espaços de busca para cada modelo
        param_distributions = {
            'BayesianRidge': {
                'max_iter': randint(200, 1000),
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
                'n_estimators': randint(50, 150),
                'max_depth': randint(3, 12),
                'min_samples_split': randint(2, 10),
                'min_samples_leaf': randint(1, 5),
                'max_features': ['sqrt', 'log2'],
                # use somente se sua versão do sklearn suportar:
                # 'max_samples': uniform(0.6, 0.3)  # 0.6 a 0.9
            },
               'Bagging': {
                    'n_estimators': randint(10, 100),
                    'max_samples': uniform(0.5, 0.5),  # Gera valores entre 0.5 e 1.0
                    'max_features': uniform(0.5, 0.5),  # Gera valores entre 0.5 e 1.0
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
                'subsample': uniform(0.6, 0.4),
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
            base_model = None
            if model_name == 'BayesianRidge':
                base_model = self.regBayesianRidge
            if model_name == 'Ridge':
              base_model = self.regRidge
            if model_name == 'Tweedie':
              base_model = self.regTweedie
            if model_name == 'RandomForest':
              base_model = self.regRandomForest
            if model_name == 'Bagging':
              base_model = self.regBagging
            if model_name == 'AdaBoost':
              base_model = self.regAdaBoost
            if model_name == 'GradientBoosting':
              base_model = self.regGradientBoosting
            if model_name == 'LinearSVR':
              base_model = self.regLinearSVR

            # Antes do fit do RF, subamostre:
            X_fit, y_fit = X_train, y_train
            if model_name == 'RandomForest' and len(X_train) > 150_000:
                rng = np.random.RandomState(42)
                idx = rng.choice(len(X_train), size=150_000, replace=False)
                X_fit, y_fit = X_train[idx], y_train[idx]

            # E reduza o esforço do RF especificamente:
            n_iter = 8 if model_name == 'RandomForest' else 30
            cv = 2 if model_name == 'RandomForest' else 5

            random_search = RandomizedSearchCV(
                estimator=base_model,
                param_distributions=params,
                n_iter=n_iter,
                cv=cv,
                scoring='neg_mean_squared_error',
                n_jobs=1 if model_name == 'RandomForest' else -1,
                random_state=42,
                verbose=1
            )
            
            try:
                random_search.fit(X_fit, y_fit)
                
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
        scaled_data = self.scaler.fit_transform(combined_ratings)
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

    # def loandAndOptimize(self, window_count):
    #     recs_from_SVD = pd.read_csv(self.relativePath +str(window_count)+ "_constituent_methods__SVD.tsv", delimiter='\t')
    #     recs_from_BIAS = pd.read_csv(self.relativePath +str(window_count)+ "_constituent_methods__BIAS.tsv", delimiter='\t')
    #     recs_from_userKNN = pd.read_csv(self.relativePath +str(window_count)+ "_constituent_methods__userKNN.tsv", delimiter='\t')
    #     recs_from_itemKNN = pd.read_csv(self.relativePath +str(window_count)+ "_constituent_methods__itemKNN.tsv", delimiter='\t')
    #     recs_from_biasedMF = pd.read_csv(self.relativePath +str(window_count)+ "_constituent_methods__BIASEDMF.tsv", delimiter='\t')
    #     columns = ['user', 'item', 'prediction']
    #     recs_from_SVD = recs_from_SVD.sort_values('user')
    #     recs_from_BIAS = recs_from_BIAS.sort_values('user')
    #     recs_from_userKNN = recs_from_userKNN.sort_values('user')
    #     recs_from_itemKNN = recs_from_itemKNN.sort_values('user')
    #     recs_from_biasedMF = recs_from_biasedMF.sort_values('user')
       
    #     scikit_test_data = pd.read_csv('data/windows/processed/test_to_get_regression_train_data_' + str(window_count) + "_.csv", sep=',')

    #     recs_from_SVD_to_train_scikit = pd.read_csv(self.relativePath +str(window_count)+ "_scikit_train__SVD.tsv", delimiter='\t')
    #     recs_from_BIAS_to_train_scikit = pd.read_csv(self.relativePath +str(window_count)+ "_scikit_train__BIAS.tsv", delimiter='\t')
    #     recs_from_userKNN_to_train_scikit = pd.read_csv(self.relativePath +str(window_count)+ "_scikit_train__userKNN.tsv", delimiter='\t')
    #     recs_from_itemKNN_to_train_scikit = pd.read_csv(self.relativePath +str(window_count)+ "_scikit_train__itemKNN.tsv", delimiter='\t')
    #     recs_from_biasedMF_to_train_scikit = pd.read_csv(self.relativePath +str(window_count)+ "_scikit_train__BIASEDMF.tsv", delimiter='\t')

        
    #     ratings_BIAS_to_use_in_prediction = recs_from_BIAS['prediction'].values
    #     ratings_SVD_to_use_in_prediction = recs_from_SVD['prediction'].values
    #     ratings_userKNN_to_use_in_prediction = recs_from_userKNN['prediction'].values
    #     ratings_itemKNN_to_use_in_prediction = recs_from_itemKNN['prediction'].values
    #     ratings_biasedMF_to_use_in_prediction = recs_from_biasedMF['prediction'].values
    #     print(ratings_BIAS_to_use_in_prediction)
    #     combined_ratings = [[r, s, t, u, v] for r, s, t, u, v in zip(ratings_BIAS_to_use_in_prediction, ratings_SVD_to_use_in_prediction, ratings_userKNN_to_use_in_prediction, ratings_itemKNN_to_use_in_prediction, ratings_biasedMF_to_use_in_prediction)]
    #     combined_ratings = np.nan_to_num(combined_ratings)

    #     ratings_BIAS_to_use_in_train = recs_from_BIAS_to_train_scikit['prediction'].values
    #     ratings_SVD_to_use_in_train = recs_from_SVD_to_train_scikit['prediction'].values
    #     ratings_userKNN_to_use_in_train = recs_from_userKNN_to_train_scikit['prediction'].values
    #     ratings_itemKNN_to_use_in_train = recs_from_itemKNN_to_train_scikit['prediction'].values
    #     ratings_biasedMF_to_use_in_train = recs_from_biasedMF_to_train_scikit['prediction'].values
        
    #     # Adicionando a linha que estava faltando
    #     original_ratings_scikit = scikit_test_data['rating'].values
        
    #     cobined_ratings_train = [[r, s, t, u, v] for r, s,t,u,v in zip(ratings_BIAS_to_use_in_train, ratings_SVD_to_use_in_train, ratings_userKNN_to_use_in_train, ratings_itemKNN_to_use_in_train, ratings_biasedMF_to_use_in_train)]
    #     cobined_ratings_train = np.nan_to_num(cobined_ratings_train)

    #     # Prepara os dados para otimização
    #     combined_ratings_train = self.preprocess_data(cobined_ratings_train)
        
    #     # Otimiza os modelos
    #     print("Iniciando otimização de parâmetros...")
    #     optimized_models = self.optimize_models(combined_ratings_train, original_ratings_scikit)
        
    #     # Salva os parâmetros otimizados
    #     self.save_optimized_parameters(optimized_models, window_count)

    def loadAndPredictWithOptimizedModels(self, window_count):
        # Carrega as predições base e dados de treino/teste
        recs_from_SVD = pd.read_csv(f"data/filtered_predictions/window_{window_count}_constituent_methods_SVD.tsv", delimiter='\t')
        recs_from_BIAS = pd.read_csv(f"data/filtered_predictions/window_{window_count}_constituent_methods_BIAS.tsv", delimiter='\t')
        recs_from_userKNN = pd.read_csv(f"data/filtered_predictions/window_{window_count}_constituent_methods_userKNN.tsv", delimiter='\t')
        recs_from_itemKNN = pd.read_csv(f"data/filtered_predictions/window_{window_count}_constituent_methods_itemKNN.tsv", delimiter='\t')
        recs_from_biasedMF = pd.read_csv(f"data/filtered_predictions/window_{window_count}_constituent_methods_BIASEDMF.tsv", delimiter='\t')
        recs_from_SVD = recs_from_SVD.sort_values('user')
        recs_from_BIAS = recs_from_BIAS.sort_values('user')
        recs_from_userKNN = recs_from_userKNN.sort_values('user')
        recs_from_itemKNN = recs_from_itemKNN.sort_values('user')
        recs_from_biasedMF = recs_from_biasedMF.sort_values('user')

        scikit_test_data = pd.read_csv(f'data/windows/processed/test_to_get_regression_train_data_{window_count}_filtered.csv', sep=',')

        recs_from_SVD_to_train_scikit = pd.read_csv(f"data/filtered_predictions/window_{window_count}_scikit_train_SVD.tsv", delimiter='\t')
        recs_from_BIAS_to_train_scikit = pd.read_csv(f"data/filtered_predictions/window_{window_count}_scikit_train_BIAS.tsv", delimiter='\t')
        recs_from_userKNN_to_train_scikit = pd.read_csv(f"data/filtered_predictions/window_{window_count}_scikit_train_userKNN.tsv", delimiter='\t')
        recs_from_itemKNN_to_train_scikit = pd.read_csv(f"data/filtered_predictions/window_{window_count}_scikit_train_itemKNN.tsv", delimiter='\t')
        recs_from_biasedMF_to_train_scikit = pd.read_csv(f"data/filtered_predictions/window_{window_count}_scikit_train_BIASEDMF.tsv", delimiter='\t')
        

        print(f"Filtered scikit_test_data: {len(scikit_test_data)} linhas (mantém apenas pares com predições)")
        # X para predição
        ratings_BIAS_to_use_in_prediction = recs_from_BIAS['prediction'].values
        ratings_SVD_to_use_in_prediction = recs_from_SVD['prediction'].values
        ratings_userKNN_to_use_in_prediction = recs_from_userKNN['prediction'].values
        ratings_itemKNN_to_use_in_prediction = recs_from_itemKNN['prediction'].values
        ratings_biasedMF_to_use_in_prediction = recs_from_biasedMF['prediction'].values
        combined_ratings = [[r, s, t, u, v] for r, s, t, u, v in zip(
            ratings_BIAS_to_use_in_prediction,
            ratings_SVD_to_use_in_prediction,
            ratings_userKNN_to_use_in_prediction,
            ratings_itemKNN_to_use_in_prediction,
            ratings_biasedMF_to_use_in_prediction
        )]
        combined_ratings = np.nan_to_num(combined_ratings)

        # X de treino e y
        ratings_BIAS_to_use_in_train = recs_from_BIAS_to_train_scikit['prediction'].values
        ratings_SVD_to_use_in_train = recs_from_SVD_to_train_scikit['prediction'].values
        ratings_userKNN_to_use_in_train = recs_from_userKNN_to_train_scikit['prediction'].values
        ratings_itemKNN_to_use_in_train = recs_from_itemKNN_to_train_scikit['prediction'].values
        ratings_biasedMF_to_use_in_train = recs_from_biasedMF_to_train_scikit['prediction'].values
        original_ratings_scikit = scikit_test_data['rating'].values
        cobined_ratings_train = [[r, s, t, u, v] for r, s, t, u, v in zip(
            ratings_BIAS_to_use_in_train,
            ratings_SVD_to_use_in_train,
            ratings_userKNN_to_use_in_train,
            ratings_itemKNN_to_use_in_train,
            ratings_biasedMF_to_use_in_train
        )]
        cobined_ratings_train = np.nan_to_num(cobined_ratings_train)

        # Escala os dados (fit no treino, transform no preditivo)
        combined_ratings_train_scaled = self.preprocess_data(cobined_ratings_train)
        combined_ratings_scaled = np.nan_to_num(self.scaler.transform(combined_ratings))

        # Carrega hiperparâmetros
        loaded_params = self.load_parameters(window_count)
        if loaded_params is None:
            print("Parâmetros não encontrados. Executando otimização completa...")
            # Executa otimização e salva os parâmetros
            print("Iniciando otimização de parâmetros...")
            optimized_models = self.optimize_models(combined_ratings_train_scaled, original_ratings_scikit)
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
        else:
            # Aplica hiperparâmetros carregados
            self.initialize_base_models()
            
            model_attr_map = {
                'BayesianRidge': 'regBayesianRidge',
                'Ridge': 'regRidge',
                'Tweedie': 'regTweedie',
                'RandomForest': 'regRandomForest',
                'Bagging': 'regBagging',
                'AdaBoost': 'regAdaBoost',
                'GradientBoosting': 'regGradientBoosting',
                'LinearSVR': 'regLinearSVR',
            }

            for model_name, attr_name in model_attr_map.items():
                if model_name in loaded_params:
                    try:
                        getattr(self, attr_name).set_params(**loaded_params[model_name])
                    except ValueError as e:
                        print(f"Aviso: não foi possível aplicar todos os parâmetros de {model_name}: {e}")

            # Refit dos modelos com os hiperparâmetros carregados
            self.regBayesianRidge.fit(combined_ratings_train_scaled, original_ratings_scikit)
            self.regRidge.fit(combined_ratings_train_scaled, original_ratings_scikit)
            self.regTweedie.fit(combined_ratings_train_scaled, original_ratings_scikit)
            self.regRandomForest.fit(combined_ratings_train_scaled, original_ratings_scikit)
            self.regBagging.fit(combined_ratings_train_scaled, original_ratings_scikit)
            self.regAdaBoost.fit(combined_ratings_train_scaled, original_ratings_scikit)
            self.regGradientBoosting.fit(combined_ratings_train_scaled, original_ratings_scikit)
            self.regLinearSVR.fit(combined_ratings_train_scaled, original_ratings_scikit)

        # Predição com dados na mesma escala
        predictedBayesianRidge = self.regBayesianRidge.predict(combined_ratings_scaled)
        predictedTweedie = self.regTweedie.predict(combined_ratings_scaled)
        predicted = self.regRidge.predict(combined_ratings_scaled)
        predictedRandomForest = self.regRandomForest.predict(combined_ratings_scaled)
        predictedBagging = self.regBagging.predict(combined_ratings_scaled)
        predictedAdaBoost = self.regAdaBoost.predict(combined_ratings_scaled)
        predictedGradientBoosting = self.regGradientBoosting.predict(combined_ratings_scaled)
        predictedLinearSVR = self.regLinearSVR.predict(combined_ratings_scaled)

        # Salva as predições
        pd.DataFrame(predictedBayesianRidge).to_csv(
            f"data/HybridPredictions/window_{window_count}_predictedBayesianRidge.tsv", sep='\t', index=False
        )
        pd.DataFrame(predictedTweedie).to_csv(
            f"data/HybridPredictions/window_{window_count}_predictedTweedie.tsv", sep='\t', index=False
        )
        pd.DataFrame(predicted).to_csv(
            f"data/HybridPredictions/window_{window_count}_predictedRidge.tsv", sep='\t', index=False
        )
        pd.DataFrame(predictedRandomForest).to_csv(
            f"data/HybridPredictions/window_{window_count}_predictedRandomForest.tsv", sep='\t', index=False
        )
        pd.DataFrame(predictedBagging).to_csv(
            f"data/HybridPredictions/window_{window_count}_predictedBagging.tsv", sep='\t', index=False
        )
        pd.DataFrame(predictedAdaBoost).to_csv(
            f"data/HybridPredictions/window_{window_count}_predictedAdaBoost.tsv", sep='\t', index=False
        )
        pd.DataFrame(predictedGradientBoosting).to_csv(
            f"data/HybridPredictions/window_{window_count}_predictedGradientBoosting.tsv", sep='\t', index=False
        )
        pd.DataFrame(predictedLinearSVR).to_csv(
            f"data/HybridPredictions/window_{window_count}_predictedLinearSVR.tsv", sep='\t', index=False
        )

        print(f"Predições (com parâmetros carregados) salvas para janela {window_count}")



