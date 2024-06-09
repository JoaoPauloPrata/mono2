from lenskit.metrics.predict import user_metric, rmse
import pandas as pd
from lenskit.metrics.topn import ndcg
import numpy as np

def hitCount(recs):
   
    test = pd.read_csv('../TrainTestData/expanded_test.tsv', delimiter='\t', header=None, names=['user', 'item', 'rating', 'timestamp'])
    count = 0
    group_from_tests= test.groupby('user')
    grups_from_recs = recs.groupby('user')   
    
    for group in grups_from_recs:
        groupUser =group[1]['user'].iloc[0]
        groupFromTests = group_from_tests.get_group(groupUser)    
        itemWithRating5 = groupFromTests['item'].iloc[0]    
        top_20_recs = group[1].nlargest(20, 'rating')['item'].values
        if(itemWithRating5 in top_20_recs):
            count += 1  
    return count


def recall(hits):
    return hits/1001

def precision(hits, n):
    return hits/(n*1001)    

def f1_score(hits, n):
    p = precision(hits, n)  
    r = recall(hits)
    if p + r == 0:
        return 0
    else:
        return 2 * (p * r) / (p + r)


svdPreds = pd.read_csv('../SimpleMethodsPredictions/SVD.tsv', delimiter='\t',  header=None, names=['user', 'item', 'rating'])
biasPreds = pd.read_csv('../SimpleMethodsPredictions/BIAS.tsv', delimiter='\t',  header=None, names=['user', 'item', 'rating'])
biasedMFPreds = pd.read_csv('../SimpleMethodsPredictions/BIASEDMF.tsv', delimiter='\t',  header=None, names=['user', 'item', 'rating'])
itemKNNPreds = pd.read_csv('../SimpleMethodsPredictions/itemKNN.tsv', delimiter='\t',  header=None, names=['user', 'item', 'rating'])
userKNNPreds = pd.read_csv('../SimpleMethodsPredictions/userKNN.tsv', delimiter='\t',  header=None, names=['user', 'item', 'rating'])






print("SVD F1: ", f1_score(hitCount(svdPreds), 20))
print("BIAS F1: ", f1_score(hitCount(biasPreds), 20))
print("BIASEDMF F1: ", f1_score(hitCount(biasedMFPreds), 20))
print("ITEM KNN F1: ", f1_score(hitCount(itemKNNPreds), 20))
print("USER KNN F1: ", f1_score(hitCount(userKNNPreds), 20))







