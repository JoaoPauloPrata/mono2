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



class RegressionMethods:
    def __init__(self):
        self.regBayesianRidge = BayesianRidge()
        self.relativePath = "data/filtered_predictions/window_"
        self.regRidge = linear_model.Ridge(alpha=.5)
        self.regTweedie = TweedieRegressor(power=1, alpha=0.5, link='log')
        self.regRandomForest = RandomForestRegressor(n_jobs=-1, random_state=0, bootstrap=True, max_depth=22, max_leaf_nodes=40, min_samples_leaf=41, min_samples_split=48, min_weight_fraction_leaf=0.01144979520270445, n_estimators=348)  
        self.regBagging = BaggingRegressor(n_jobs=1, random_state=0, bootstrap=True)
        self.regAdaBoost = AdaBoostRegressor(random_state=0)
        self.regGradientBoosting = GradientBoostingRegressor(random_state=0)
        self.regLinearSVR = LinearSVR(random_state=0, max_iter=5000)

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
       
        scikit_test_data = pd.read_csv('data/windows/processed/test_to_get_regression_train_data_' + str(window_count) + "_.csv", sep=',')

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
        original_ratings_scikit = scikit_test_data[['rating']].values
        cobined_ratings_train = [[r, s, t, u, v] for r, s,t,u,v in zip(ratings_BIAS_to_use_in_train, ratings_SVD_to_use_in_train, ratings_userKNN_to_use_in_train, ratings_itemKNN_to_use_in_train, ratings_biasedMF_to_use_in_train)]
        cobined_ratings_train = np.nan_to_num(cobined_ratings_train)
        self.regBayesianRidge.fit(cobined_ratings_train, original_ratings_scikit)
        self.regTweedie.fit(cobined_ratings_train, original_ratings_scikit)
        self.regRidge.fit(cobined_ratings_train, original_ratings_scikit)
        self.regRandomForest.fit(cobined_ratings_train, original_ratings_scikit)
        self.regBagging.fit(cobined_ratings_train, original_ratings_scikit)
        self.regAdaBoost.fit(cobined_ratings_train, original_ratings_scikit)
        self.regGradientBoosting.fit(cobined_ratings_train, original_ratings_scikit)
        self.regLinearSVR.fit(cobined_ratings_train, original_ratings_scikit)
        predictedBayesianRidge = self.regBayesianRidge.predict(combined_ratings)
        predictedTweedie = self.regTweedie.predict(combined_ratings)
        predicted = self.regRidge.predict(combined_ratings)
        predictedRandomForest = self.regRandomForest.predict(combined_ratings)
        predictedBagging = self.regBagging.predict(combined_ratings)
        predictedAdaBoost = self.regAdaBoost.predict(combined_ratings)
        predictedGradientBoosting = self.regGradientBoosting.predict(combined_ratings)
        predictedLinearSVR = self.regLinearSVR.predict(combined_ratings)
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

