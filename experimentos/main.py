from src.DataProcessing.TimePeriodSpliter import TimePeriodSpliter
import pandas as pd
from src.Recommender import Recommender
from sklearn.model_selection import train_test_split
import time
from src.Metrics.Evaluator import Evaluator
def split_data(execution):
    datapath = "./data/ml-1m/"
    ratings_cols = ['user', 'item', 'rating', 'timestamp']
    ratings = pd.read_csv(datapath + 'ratings.dat', sep='::', names=ratings_cols, engine='python', encoding='latin-1')
    spliter = TimePeriodSpliter(sliding_window_size=15, step_size=1, dataset=ratings)
    count = 1
    time_window = spliter.get_window(count)
    while(not time_window.empty):
        train_to_get_regression_train_data = spliter.get_partial_data(time_window, 0, 9)
        test_to_get_regression_train_data = spliter.get_partial_data(time_window, 9, 3)

        train_to_get_regression_train_data.to_csv("./data/dataSplited/train_to_get_regression_train_data_"+ str(count) +"_execution_" + str(execution) + ".csv", index = False)
        test_to_get_regression_train_data.to_csv("./data/dataSplited/test_to_get_regression_train_data_"+ str(count) + "_execution_" + str(execution) + ".csv" , index = False)

        train_to_get_constituent_methods = spliter.get_partial_data(time_window, 0, 12)
        test_to_get_constituent_methods = spliter.get_partial_data(time_window, 12, 3)
        train_to_get_constituent_methods.to_csv("./data/dataSplited/train_to_get_constituent_methods_"+ str(count) + "_execution_" + str(execution) + ".csv" , index = False)
        test_to_get_constituent_methods.to_csv("./data/dataSplited/test_to_get_constituent_methods_"+ str(count) + "_execution_" + str(execution) + ".csv" , index = False)

        count += 1
        time_window = spliter.get_window(count)

def load_data_and_run(count, execution):
    train_to_get_regression_train_data = pd.read_csv("./data/dataSplited/train_to_get_regression_train_data_"+str(count)+"_execution_" + str(execution) + ".csv")
    test_to_get_regression_train_data = pd.read_csv("./data/dataSplited/test_to_get_regression_train_data_"+str(count)+"_execution_" + str(execution) + ".csv")
    train_to_get_constituent_methods = pd.read_csv("./data/dataSplited/train_to_get_constituent_methods_"+str(count)+"_execution_" + str(execution) + ".csv")
    test_to_get_constituent_methods = pd.read_csv("./data/dataSplited/test_to_get_constituent_methods_"+str(count)+"_execution_" + str(execution) + ".csv")
    recommend = Recommender()
    recommend.runRecomendations(train_to_get_regression_train_data, test_to_get_regression_train_data, count, execution, "scikit_train2")
    recommend.runRecomendations(train_to_get_constituent_methods, test_to_get_constituent_methods, count, execution, "constituent_methods2")





# for i in range(1, 101):
#     split_data(i)

# startExecutionTime = time.time()
# print(f"Executions started at {time.strftime('%Y-%m-%d %H:%M:%S')}")
# recommender = Recommender()
# evaluator = Evaluator()
# for j in range(1, 101):
#     partialExecutionTotalTime = time.time()
#     print(f"Execution {j} started at {time.strftime('%Y-%m-%d %H:%M:%S')}")
#     for i in range(1, 21):
#         load_data_and_run(i, j)
#         # recommender.run_hybrid_methods(i, j)
#         # evaluator.evaluateAllMetricsForAllMethods(i, j)
#     print(f"Execution {j} finished at {time.strftime('%Y-%m-%d %H:%M:%S')}")
#     print(f"Total execution time for execution {j}: {time.time() - partialExecutionTotalTime} seconds")

# finishExecutionTime = time.time()
# print(f"Executions finished at {time.strftime('%Y-%m-%d %H:%M:%S')}")
# print(f"Total execution time: {finishExecutionTime - startExecutionTime} seconds")
load_data_and_run(1, 1)
# load_data_and_run(1)