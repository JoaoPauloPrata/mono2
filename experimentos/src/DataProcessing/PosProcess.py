import pandas as pd
from functools import reduce

def filter_common_pairs_in_files(file_paths, output_dir, i):
    # Carrega todos os arquivos em uma lista de DataFrames
    dataframes = [pd.read_csv(file, sep='\t') for file in file_paths]
    all_pairs_to_drop = []
    for df in dataframes:
        missing_predictions = df[df['prediction'].isna()][['user', 'item']]
        for pair in missing_predictions.values:
            pair_tuple = tuple(pair)
            if pair_tuple not in all_pairs_to_drop:
                all_pairs_to_drop.append(pair_tuple)

    # print(all_pairs_to_drop)
    trainDatasetToClean =  pd.read_csv('data/dataSplited/test_to_get_constituent_methods_' + str(i) + "_.csv", sep=',')
    trainDatasetToClean.drop(trainDatasetToClean[trainDatasetToClean[['user', 'item']].apply(tuple, axis=1).isin(all_pairs_to_drop)].index, inplace=True)
    trainDatasetToClean.to_csv('data/dataSplited/processed/test_to_get_constituent_methods_' + str(i) + "_.csv", sep=',', index=False)

    # for file, df in zip(file_paths, dataframes):
    #     df.drop(df[df[['user', 'item']].apply(tuple, axis=1).isin(all_pairs_to_drop)].index, inplace=True)
    #     output_file = output_dir + '/' + file.split('/')[-1]
    #     df.to_csv(output_file, sep='\t', index=False)#

def drop_all_non_recommended():
    algo = ["itemKNN", "BIAS", "userKNN", "SVD", "BIASEDMF"]
    output_dir = "../../data/filtered_predictions"
    for i in range (1, 21):
        allPaths = []
        for a in algo:
            relativePath = f"window_{i}_scikit_train__{a}.tsv"
            allPaths.append(f"../../data/predictions/{a}/{relativePath}")
        filter_common_pairs_in_files(allPaths, output_dir, i)

def count_user_per_recs_count():
    pass



def add_header_to_hybrid():

    algo = ["BayesianRidge", "Tweedie", "Ridge", "RandomForest", "Bagging", "AdaBoost", "GradientBoosting", "LinearSVR"]
    # Carregar o DataFrame dfUsedToExtractUserItem
    for i in range (1, 21):
        usedToExtractUserItem = "window_" +str(i)+ "_constituent_methods__SVD.tsv"
        dfUsedToExtractUserItem = pd.read_csv("../../data/filtered_predictions/" + usedToExtractUserItem, delimiter='\t')
        user_item_pairs = dfUsedToExtractUserItem[['user', 'item']]

        for file in algo:
            filePath = f"../../data/HybridPredictions/window_{i}_predicted{file}.tsv"
            df_hybrid = pd.read_csv(filePath, delimiter='\t', header=None, names=['prediction'], skiprows=1)

            # Adicionar as colunas 'user' e 'item' do DataFrame dfUsedToExtractUserItem
            df_hybrid['user'] = user_item_pairs['user']
            df_hybrid['item'] = user_item_pairs['item']

            # Reordenar as colunas para ter 'user', 'item' e 'prediction'
            df_hybrid = df_hybrid[['user', 'item', 'prediction']]

            # Salvar o DataFrame atualizado de volta no arquivo
            df_hybrid.to_csv(filePath, sep='\t', index=False)



add_header_to_hybrid()

