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
    def runOptimization(self):
        for i in range(1, 21):  
            regression = RegressionMethodsWithFineTuning()
            regression.loandAndOptimize(i)
        print("Hybrid methods with fine tuning completed.")
    def _normalize_windows(self, windows):
        """Normaliza o parâmetro de janelas para um iterável de inteiros."""
        if windows is None:
            return range(1, 21)
        if isinstance(windows, int):
            return [windows]
        return windows

    def runOptimization(self, windows=None):
        """
        Executa apenas o fine-tuning (otimização e salvamento de hiperparâmetros) por janela.
        """
        for i in self._normalize_windows(windows):
            regression = RegressionMethodsWithFineTuning()
            regression.loandAndOptimize(i)
        print("Fine-tuning concluído para as janelas especificadas.")

    def run_hybrid_methods_after_finetuning(self, windows=None):
        """
        Realiza recomendações híbridas CARREGANDO hiperparâmetros salvos previamente.
        """
        for i in self._normalize_windows(windows):
            regression = RegressionMethodsWithFineTuning()
            regression.loadAndPredictWithOptimizedModels(i)
        print("Recomendações híbridas (após fine-tuning) concluídas.")

    def run_hybrid_methods_without_finetuning(self, windows=None):
        """
        Realiza recomendações com os métodos sem fine-tuning (parâmetros padrão/estáticos).
        """
        for i in self._normalize_windows(windows):
            regression = RegressionMethods()
            regression.loadAndPredict(i)
        print("Recomendações sem fine-tuning concluídas.")

    # Mantido por compatibilidade: passa a usar o fluxo "após fine-tuning"
    def run_hybrid_methods(self, windows=None):
        for i in self._normalize_windows(windows):  
            regression = RegressionMethodsWithFineTuning()
            regression.loadAndPredictWithOptimizedModels(i)
        print("Recomendações híbridas (compat) concluídas.")
