import pandas as pd
from lenskit.algorithms import item_knn, user_knn
from lenskit.algorithms.svd import BiasedSVD
from lenskit.algorithms.als import BiasedMF
from lenskit.algorithms.bias import Bias
class ConstituentMethods:
    def __init__(self):
        pass

    def recommenderWithItemKNN(self, train, test, path):
        itemKnn = item_knn.ItemItem(20, feedback='explicit')
        itemKnn.fit(train)
        predictions = itemKnn.predict(test)
        recs = pd.DataFrame({
            'user': test['user'],
            'item': test['item'],
            'prediction': predictions
        })
        recs.to_csv(".../data/data/predictions/itemKNN/" + path, sep='\t', index=False)


    def recommenderWithUserKNN(self, train, test, path):
        userKnn = user_knn.UserUser(20, min_nbrs=5, center=True, aggregate='weighted-average', feedback='explicit')
        userKnn.fit(train)
        predictions = userKnn.predict(test)
        recs = pd.DataFrame({
            'user': test['user'],
            'item': test['item'],
            'prediction': predictions
        })
        recs.to_csv("../data/predictions/userKNN/"+path, sep='\t', index=False)
    

    def recommenderWithSvd(self, train, test, path):
        svd = BiasedSVD(features=50, damping=5, bias=True, algorithm='randomized')
        svd.fit(train)
        predictions = svd.predict(test)
        recs = pd.DataFrame({
            'user': test['user'],
            'item': test['item'],
            'prediction': predictions
        })
        recs.to_csv("../data/predictions/svd/" + path, sep='\t', index=False)

    def recommenderWithBiasedMF(self, train, test, path):
        biasedMF = BiasedMF(features=50, iterations=20, reg=0.1, damping=5, bias=True, method='cd')
        biasedMF.fit(train)
        predictions = biasedMF.predict(test)
        recs = pd.DataFrame({
            'user': test['user'],
            'item': test['item'],
            'prediction': predictions
        })
        recs.to_csv("../data/predictions/biasedMF/" +path, sep='\t', index=False)

    def recommenderWithBias(self, train, test, path):
        bias = Bias(items=True, users=True, damping=5.0)
        bias.fit(train)
        predictions = bias.predict(test)
        recs = pd.DataFrame({
            'user': test['user'],
            'item': test['item'],
            'prediction': predictions
        })
        recs.to_csv("../data/predictions/bias/" + path, sep='\t', index=False)