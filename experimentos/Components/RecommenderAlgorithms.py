import pandas as pd
from lenskit.algorithms import item_knn, user_knn
from lenskit.algorithms.svd import BiasedSVD
from lenskit.algorithms.als import BiasedMF
from lenskit.algorithms.bias import Bias
class RecommenderAlgorithms:
    def __init__(self):
        pass

    def recommenderWithItemKNN(self, train, test, test_to_scikit, path):
        itemKnn = item_knn.ItemItem(1001, feedback='explicit')
        itemKnn.fit(train)
        grouped_test_by_group = test.groupby('user')
        all_recs = []
        for group in grouped_test_by_group:
            item_ids = group[1]['item'].unique()
            user_id = group[1]['user'].iloc[0]
            recs = itemKnn.predict_for_user(user_id, item_ids)
            recs = recs.sort_values(ascending=False)
            recs_df = pd.DataFrame({
                'user': [user_id]*len(recs),
                'item': recs.index,
                'rating': recs.values
            })
            all_recs.append(recs_df)

        #RESULTADO DAS RECOMENDAÇÕES    
        all_recs_df = pd.concat(all_recs)
        all_recs_df.to_csv(path, sep='\t', index=False)

        #RECOMENDAÇÕES APENAS PARA TREINAR METODOS DE REGRESSÃO
        recs_to_scikit = itemKnn.predict(test_to_scikit)
        recs_to_scikit.to_csv( "scikit_train_" + path , sep='\t', index=False)


    def recommenderWithUserKNN(self, train, test, test_to_scikit, path):
        userKnn = user_knn.UserUser(1001, min_nbrs=5, center=True, aggregate='weighted-average', feedback='explicit')
        userKnn.fit(train)
        grouped_test_by_group = test.groupby('user')
        all_recs = []
        for group in grouped_test_by_group:
            item_ids = group[1]['item'].unique()
            user_id = group[1]['user'].iloc[0]
            recs = userKnn.predict_for_user(user_id, item_ids)
            recs = recs.sort_values(ascending=False)
            recs_df = pd.DataFrame({
                'user': [user_id]*len(recs),
                'item': recs.index,
                'rating': recs.values
            })
            all_recs.append(recs_df)
        all_recs_df = pd.concat(all_recs)
        all_recs_df.to_csv(path, sep='\t', index=False)
        recs_to_scikit = userKnn.predict(test_to_scikit)
        recs_to_scikit.to_csv( "scikit_train_" + path , sep='\t', index=False)

    def recommenderWithSvd(self, train, test, test_to_scikit, path):
        svd = BiasedSVD(features=50, damping=5, bias=True, algorithm='randomized')
        svd.fit(train)
        # grouped_test_by_group = test.groupby('user')
        all_recs = []
        grupos = test.groupby(test.index // 1001)
        grouped_test_by_group = grupos
        for group in grouped_test_by_group:
            user_id = group[1]['user'].iloc[0]
            item_ids = group[1]['item'].unique()
            item_ids = list(map(int, item_ids))  # Convert item_ids to a list of integers
            recs = svd.predict_for_user(user_id, item_ids)
            recs_df = pd.DataFrame({
                'user': [user_id]*len(recs),
                'item': recs.index,
                'rating': recs.values
            })
            all_recs.append(recs_df)
        all_recs_df = pd.concat(all_recs)
        # all_recs_df = all_recs_df.sort_values(by='user')
        all_recs_df.to_csv(path, sep='\t', index=False)

        recs_to_scikit = svd.predict(test_to_scikit)
        recs_to_scikit.to_csv( "scikit_train_" + path , sep='\t', index=False)

    def recommenderWithBiasedMF(self, train, test, test_to_scikit, path):
        biasedMF = BiasedMF(features=50, iterations=20, reg=0.1, damping=5, bias=True, method='cd')
        biasedMF.fit(train)
        grouped_test_by_group = test.groupby('user')
        all_recs = []
        for group in grouped_test_by_group:
            item_ids = group[1]['item'].unique()
            item_ids = list(map(int, item_ids))  # Convert item_ids to a list of integers
            user_id = group[1]['user'].iloc[0]
            recs = biasedMF.predict_for_user(user_id, item_ids)
            recs_df = pd.DataFrame({
                'user': [user_id]*len(recs),
                'item': recs.index,
                'rating': recs.values
            })
            all_recs.append(recs_df)
        all_recs_df = pd.concat(all_recs)
        all_recs_df = all_recs_df.sort_values(by='user')
        all_recs_df.to_csv(path, sep='\t', index=False)

        recs_to_scikit = biasedMF.predict(test_to_scikit)
        recs_to_scikit.to_csv( "scikitt_train_" + path , sep='\t', index=False)

    def recommenderWithBias(self, train, test, test_to_scikit, path):
        bias = Bias(items=True, users=True, damping=5.0)
        bias.fit(train)
        grouped_test_by_group = test.groupby('user')
        all_recs = []
        for group in grouped_test_by_group:
            item_ids = group[1]['item'].unique()
            item_ids = list(map(int, item_ids))  # Convert item_ids to a list of integers
            user_id = group[1]['user'].iloc[0]
            recs = bias.predict_for_user(user_id, item_ids)
            recs_df = pd.DataFrame({
                'user': [user_id]*len(recs),
                'item': recs.index,
                'rating': recs.values
            })
            all_recs.append(recs_df)
        all_recs_df = pd.concat(all_recs)
        all_recs_df = all_recs_df.sort_values(by='user')
        all_recs_df.to_csv(path, sep='\t', index=False)
        recs_to_scikit = bias.predict(test_to_scikit)
        recs_to_scikit.to_csv( "scikit_train_" + path , sep='\t', index=False)