from src.DataProcessing.TimePeriodSpliter import TimePeriodSpliter
import pandas as pd
from src.Recommender import Recommender
from sklearn.model_selection import train_test_split
import time
from src.Metrics.Evaluator import Evaluator
def split_data():
    datapath = "./ml-1m/"
    ratings_cols = ['user', 'item', 'rating', 'timestamp']
    ratings = pd.read_csv(datapath + 'ratings.dat', sep='::', names=ratings_cols, engine='python', encoding='latin-1')
    spliter = TimePeriodSpliter(sliding_window_size=15, step_size=1, dataset=ratings)
    count = 1
    time_window = spliter.get_window(count)
    while(not time_window.empty):
        train_to_get_regression_train_data = spliter.get_partial_data(time_window, 0, 9)
        test_to_get_regression_train_data = spliter.get_partial_data(time_window, 9, 3)
        
        train_to_get_regression_train_data.to_csv("./dataSplited/train_to_get_regression_train_data_"+ str(count) + "_.csv", index = False)
        test_to_get_regression_train_data.to_csv("./dataSplited/test_to_get_regression_train_data_"+ str(count) + "_.csv" , index = False)

        train_to_get_constituent_methods = spliter.get_partial_data(time_window, 0, 12)
        test_to_get_constituent_methods = spliter.get_partial_data(time_window, 12, 3)
        train_to_get_constituent_methods.to_csv("./dataSplited/train_to_get_constituent_methods_"+ str(count) + "_.csv" , index = False)
        test_to_get_constituent_methods.to_csv("./dataSplited/test_to_get_constituent_methods_"+ str(count) + "_.csv" , index = False)

        count += 1
        time_window = spliter.get_window(count)

def load_data_and_run(count):
    train_to_get_regression_train_data = pd.read_csv("./dataSplited/train_to_get_regression_train_data_"+str(count)+"_.csv")
    test_to_get_regression_train_data = pd.read_csv("./dataSplited/test_to_get_regression_train_data_"+str(count)+"_.csv")
    train_to_get_constituent_methods = pd.read_csv("./dataSplited/train_to_get_constituent_methods_"+str(count)+"_.csv")
    test_to_get_constituent_methods = pd.read_csv("./dataSplited/test_to_get_constituent_methods_"+str(count)+"_.csv")
    recommend = Recommender()
    recommend.runRecomendations(train_to_get_regression_train_data, test_to_get_regression_train_data, count, "_scikit_train_")
    recommend.runRecomendations(train_to_get_constituent_methods, test_to_get_constituent_methods, count, "_constituent_methods_")

def run_constituent_methods():
    start_time = time.time()
    count = 1
    while(count <= 20):
        load_data_and_run(count)
        count += 1
    end_time = time.time()

    elapsed_time = end_time - start_time
    elapsed_minutes = elapsed_time / 60

    print("Tempo demorado: {:.2f} minutos".format(elapsed_minutes))

def run_hybrid_methods():
    recommender = Recommender()
    recommender.run_hybrid_methods(2)

def runOptimization():
    recommender = Recommender()
    recommender.runOptimization()

def evaluate_based_on_metric(metric):
    evaluator = Evaluator()
    evaluator.evaluateAllMetricsForAllMethods(metric)

run_hybrid_methods()