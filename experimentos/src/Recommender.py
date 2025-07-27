import pandas as pd
from src.Methods.ConstituentMethods import ConstituentMethods 
from src.Methods.RegressionMethods import RegressionMethods
from src.Methods.RegressionMethodsWithFineTuning import RegressionMethodsWithFineTuning

class Recommender:
    def __init__(self):
        pass 
    def runRecomendations(self, train, test, window_number, path):
        print("Running recommendations for window " + str(window_number))
        recommender = ConstituentMethods()
        recommender.recommenderWithItemKNN(train, test,  "window_" + str(window_number) + str(path) + "_itemKNN.tsv" )
        recommender.recommenderWithUserKNN(train, test,   "window_" + str(window_number) + str(path) +"_userKNN.tsv")
        recommender.recommenderWithSvd(train, test,   "window_" + str(window_number) + str(path) +"_SVD.tsv")  
        recommender.recommenderWithBiasedMF(train, test,   "window_" + str(window_number) + str(path) +"_BIASEDMF.tsv")        
        recommender.recommenderWithBias(train, test,   "window_" + str(window_number) + str(path) +"_BIAS.tsv")

    def run_hybrid_methods(self):
        for i in range(1, 21):  
            regression = RegressionMethodsWithFineTuning()
            regression.loadAndPredict(i)