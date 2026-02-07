from src.DataProcessing.TimePeriodSpliter import TimePeriodSpliter
import pandas as pd
from src.Recommender import Recommender
import time
from src.Metrics.Evaluator import Evaluator
import time
def split_data():
    datapath = "./data/ml-1m/"
    ratings_cols = ['user', 'item', 'rating', 'timestamp']
    ratings = pd.read_csv(datapath + 'ratings.dat', sep='::', names=ratings_cols, engine='python', encoding='latin-1')
    spliter = TimePeriodSpliter(sliding_window_size=15, step_size=1, dataset=ratings)
    count = 1
    time_window = spliter.get_window(count)
    while(not time_window.empty):
        train_to_get_regression_train_data = spliter.get_partial_data(time_window, 0, 9)
        test_to_get_regression_train_data = spliter.get_partial_data(time_window, 9, 3)

        train_to_get_regression_train_data.to_csv("./data/windows/train_to_get_regression_train_data_"+ str(count) + ".csv", index = False)
        test_to_get_regression_train_data.to_csv("./data/windows/test_to_get_regression_train_data_"+ str(count) + ".csv" , index = False)

        train_to_get_constituent_methods = spliter.get_partial_data(time_window, 0, 12)
        test_to_get_constituent_methods = spliter.get_partial_data(time_window, 12, 3)
        train_to_get_constituent_methods.to_csv("./data/windows/train_to_get_constituent_methods_"+ str(count) + ".csv" , index = False)
        test_to_get_constituent_methods.to_csv("./data/windows/test_to_get_constituent_methods_"+ str(count) + ".csv" , index = False)

        count += 1
        time_window = spliter.get_window(count)

def load_data_and_run(window_number, exec_number):
    train_to_get_regression_train_data = pd.read_csv("./data/windows/train_to_get_regression_train_data_"+str(window_number)+".csv")
    test_to_get_regression_train_data = pd.read_csv("./data/windows/test_to_get_regression_train_data_"+str(window_number)+".csv")
    train_to_get_constituent_methods = pd.read_csv("./data/windows/train_to_get_constituent_methods_"+str(window_number)+".csv")
    test_to_get_constituent_methods = pd.read_csv("./data/windows/test_to_get_constituent_methods_"+str(window_number)+".csv")
    recommend = Recommender()
    recommend.runRecomendations(train_to_get_regression_train_data, test_to_get_regression_train_data, window_number, exec_number, "scikit_train")
    recommend.runRecomendations(train_to_get_constituent_methods, test_to_get_constituent_methods, window_number, exec_number, "constituent_methods")


def split_full_windows():
    datapath = "./data/ml-1m/"
    ratings_cols = ['user', 'item', 'rating', 'timestamp']
    ratings = pd.read_csv(datapath + 'ratings.dat', sep='::', names=ratings_cols, engine='python', encoding='latin-1')
    spliter = TimePeriodSpliter(sliding_window_size=15, step_size=1, dataset=ratings)
    count = 1
    time_window = spliter.get_window(count)
    while(not time_window.empty):
        full_window = spliter.get_partial_data(time_window, 0, 15)
        full_window.to_csv("./data/windows/full/window_"+ str(count) + ".csv", index = False)
        count += 1
        time_window = spliter.get_window(count)


startExecutionTime = time.time()
recommender = Recommender()
evaluator = Evaluator()
for exec_number in range(1, 6):
    print(f"Starting execution number {exec_number}...")
    for window_number in range(1, 21):
        start_window_time = time.time()
        # print(f"Processing window {window_number}...")
        # load_data_and_run(window_number, exec_number)
        recommender.run_hybrid_methods(window_number, exec_number)
        # print(f"Recommendations for window {window_number} completed in {time.time() - start_window_time} seconds.")
        evaluator.evaluateAllMetricsForAllMethods(window_number, exec_number)

# finishExecutionTime = time.time()
# print(f"Executions finished at {time.strftime('%Y-%m-%d %H:%M:%S')}")
# print(f"Total execution time: {finishExecutionTime - startExecutionTime} seconds")
# load_data_and_run(1, 1)
# load_data_and_run(1)

