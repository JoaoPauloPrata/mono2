import pandas as pd
import numpy as np
from sklearn import linear_model
from sklearn.linear_model import TweedieRegressor
from sklearn.linear_model import BayesianRidge
from sklearn.ensemble import BaggingRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import AdaBoostRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.svm import LinearSVR

#Arquivos contendo as predições geradas a partir do arquivo com 1 item de rating 5 e 1000 itens nao avaliados
recs_from_SVD = pd.read_csv('./SimpleMethodsPredictions/SVD.tsv', delimiter='\t')
recs_from_BIAS = pd.read_csv('./SimpleMethodsPredictions/BIAS.tsv', delimiter='\t')
recs_from_userKNN = pd.read_csv('./SimpleMethodsPredictions/userKNN.tsv', delimiter='\t')
recs_from_itemKNN = pd.read_csv('./SimpleMethodsPredictions/itemKNN.tsv', delimiter='\t')
recs_from_biasedMF = pd.read_csv('./SimpleMethodsPredictions/BIASEDMF.tsv', delimiter='\t')

recs_from_SVD = recs_from_SVD.sort_values('user')
recs_from_BIAS = recs_from_BIAS.sort_values('user')
recs_from_userKNN = recs_from_userKNN.sort_values('user')
recs_from_itemKNN = recs_from_itemKNN.sort_values('user')
recs_from_biasedMF = recs_from_biasedMF.sort_values('user')

print(recs_from_userKNN)

#Arquivo contendo os dados de teste para o scikit. Esse arquivo contem todas as avaliações do conjunto de prova, exceto as de rating 5 que foram selecionadas para o metodo do Cremonesi
scikit_test_data = pd.read_csv('./test_to_train_scikit.tsv', delimiter='\t', header=None, names=['user', 'item', 'rating', 'timestamp'])

#Arquivos contendo as predições geradas a partir do conjunto de teste test_to_train_scikit
recs_from_SVD_to_train_scikit = pd.read_csv('./TrainScikitData/scikit_train_SVD.tsv', delimiter='\t')
recs_from_BIAS_to_train_scikit = pd.read_csv('./TrainScikitData/scikit_train_BIAS.tsv', delimiter='\t')
recs_from_userKNN_to_train_scikit = pd.read_csv('./TrainScikitData/scikit_train_userKNN.tsv', delimiter='\t')
recs_from_itemKNN_to_train_scikit = pd.read_csv('./TrainScikitData/scikit_train_itemKNN.tsv', delimiter='\t')
recs_from_biasedMF_to_train_scikit = pd.read_csv('./TrainScikitData/scikit_train_BIASEDMF.tsv', delimiter='\t')

#Separando os ratings do que usarei nas predições(Ratings gerados a partir do conjunto de teste com um item 5 e 1000 itens nao avaliados) 
ratings_BIAS_to_use_in_prediction = recs_from_BIAS['rating'].values
ratings_SVD_to_use_in_prediction = recs_from_SVD['rating'].values
ratings_userKNN_to_use_in_prediction = recs_from_userKNN['rating'].values
ratings_itemKNN_to_use_in_prediction = recs_from_itemKNN['rating'].values
ratings_biasedMF_to_use_in_prediction = recs_from_biasedMF['rating'].values

combined_ratings = [[r, s, t, u, v] for r, s, t, u, v in zip(ratings_BIAS_to_use_in_prediction, ratings_SVD_to_use_in_prediction, ratings_userKNN_to_use_in_prediction, ratings_itemKNN_to_use_in_prediction, ratings_biasedMF_to_use_in_prediction)]
combined_ratings = np.nan_to_num(combined_ratings)

#Separando os ratings do que usarei no treinamento do modelo de regressão
ratings_BIAS_to_use_in_train = recs_from_BIAS_to_train_scikit['prediction'].values
ratings_SVD_to_use_in_train = recs_from_SVD_to_train_scikit['prediction'].values
ratings_userKNN_to_use_in_train = recs_from_userKNN_to_train_scikit['prediction'].values
ratings_itemKNN_to_use_in_train = recs_from_itemKNN_to_train_scikit['prediction'].values
ratings_biasedMF_to_use_in_train = recs_from_biasedMF_to_train_scikit['prediction'].values
original_ratings_scikit = scikit_test_data[['rating']].values
cobined_ratings_train = [[r, s, t, u, v] for r, s,t,u,v in zip(ratings_BIAS_to_use_in_train, ratings_SVD_to_use_in_train, ratings_userKNN_to_use_in_train, ratings_itemKNN_to_use_in_train, ratings_biasedMF_to_use_in_train)]
cobined_ratings_train = np.nan_to_num(cobined_ratings_train)

regBayesianRidge = BayesianRidge()
regRidge = linear_model.Ridge(alpha=.5)
regTweedie = TweedieRegressor(power=1, alpha=0.5, link='log')
regRandomForest = RandomForestRegressor(n_jobs=-1, random_state=0, bootstrap=True, max_depth=22, max_leaf_nodes=40, min_samples_leaf=41, min_samples_split=48, min_weight_fraction_leaf=0.01144979520270445, n_estimators=348)  
regBagging = BaggingRegressor(n_jobs=1, random_state=0, bootstrap=True)
regAdaBoost = AdaBoostRegressor(random_state=0)
regGradientBoosting = GradientBoostingRegressor(random_state=0)
regLinearSVR = LinearSVR(random_state=0, max_iter=5000)

regBayesianRidge.fit(cobined_ratings_train, original_ratings_scikit)
regTweedie.fit(cobined_ratings_train, original_ratings_scikit)
regRidge.fit(cobined_ratings_train, original_ratings_scikit)
regRandomForest.fit(cobined_ratings_train, original_ratings_scikit)
regBagging.fit(cobined_ratings_train, original_ratings_scikit)
regAdaBoost.fit(cobined_ratings_train, original_ratings_scikit)
regGradientBoosting.fit(cobined_ratings_train, original_ratings_scikit)
regLinearSVR.fit(cobined_ratings_train, original_ratings_scikit)
predictedBayesianRidge = regBayesianRidge.predict(combined_ratings)
predictedTweedie = regTweedie.predict(combined_ratings)
predicted = regRidge.predict(combined_ratings)
predictedRandomForest = regRandomForest.predict(combined_ratings)
predictedBagging = regBagging.predict(combined_ratings)
predictedAdaBoost = regAdaBoost.predict(combined_ratings)
predictedGradientBoosting = regGradientBoosting.predict(combined_ratings)
predictedLinearSVR = regLinearSVR.predict(combined_ratings)
predictedBayesianRidge = pd.DataFrame(predictedBayesianRidge)   
predictedBayesianRidge.to_csv('./HybridPredictions/predictedBayesianRidge.tsv', sep='\t', index=False)
predictedTweedie = pd.DataFrame(predictedTweedie)
predictedTweedie.to_csv('./HybridPredictions/predictedTweedie.tsv', sep='\t', index=False)
predicted = pd.DataFrame(predicted)
predicted.to_csv('./HybridPredictions/predicted.tsv', sep='\t', index=False)
predictedRandomForest = pd.DataFrame(predictedRandomForest)
predictedRandomForest.to_csv('./HybridPredictions/predictedRandomForest.tsv', sep='\t', index=False)
predictedBagging = pd.DataFrame(predictedBagging)
predictedBagging.to_csv('./HybridPredictions/predictedBagging.tsv', sep='\t', index=False)
predictedAdaBoost = pd.DataFrame(predictedAdaBoost)
predictedAdaBoost.to_csv('./HybridPredictions/predictedAdaBoost.tsv', sep='\t', index=False)
predictedGradientBoosting = pd.DataFrame(predictedGradientBoosting)
predictedGradientBoosting.to_csv('./HybridPredictions/predictedGradientBoosting.tsv', sep='\t', index=False)
predictedLinearSVR = pd.DataFrame(predictedLinearSVR)
predictedLinearSVR.to_csv('./HybridPredictions/predictedLinearSVR.tsv', sep='\t', index=False)