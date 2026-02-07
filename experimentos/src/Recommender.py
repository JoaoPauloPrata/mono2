from src.Methods.ConstituentMethods import ConstituentMethods 
from src.Methods.RegressionMethodsWithFineTuning import RegressionMethodsWithFineTuning


class Recommender:
    def __init__(self):
        pass 
    def runRecomendations(self, train, test, window_number, exec_number,path):
        print("Running recommendations for window " + str(window_number))
        recommender = ConstituentMethods()
        recommender.recommenderWithStochasticItemKNN(train, test,   f"window_{window_number}_{exec_number}_{path}_StochasticItemKNN.tsv")
        recommender.recommenderWithSvd(train, test,   f"window_{window_number}_{exec_number}_{path}_SVD.tsv")  
        recommender.recommenderWithBiasedMF(train, test,   f"window_{window_number}_{exec_number}_{path}_BIASEDMF.tsv")      
        recommender.recommenderWithNMF(train, test,   f"window_{window_number}_{exec_number}_{path}_NMF.tsv")  

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

    # Mantido por compatibilidade: passa a usar o fluxo "após fine-tuning"
    def run_hybrid_methods(self, window_number, exec_number=None):
        regression = RegressionMethodsWithFineTuning()
        regression.loadAndPredictWithOptimizedModels(window_number, exec_number)
        print("Recomendações híbridas (compat) concluídas.")