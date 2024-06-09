import pandas as pd
from Components import RecommenderAlgorithms 

def main():
    train = pd.read_csv('./train.tsv', delimiter='\t', header=None, names=['user', 'item', 'rating', 'timestamp'])
    test = pd.read_csv('./expanded_test.tsv', delimiter='\t', header=None, names=['user', 'item', 'rating'])
    test_to_scikit = pd.read_csv('./test_to_train_scikit.tsv', delimiter='\t', header=None, names=['user', 'item', 'rating', 'timestamp'])
    recommender = RecommenderAlgorithms.RecommenderAlgorithms()
    recommender.recommenderWithItemKNN(train, test, test_to_scikit,  "itemKNN.tsv")
    recommender.recommenderWithUserKNN(train, test, test_to_scikit,"userKNN.tsv")
    recommender.recommenderWithSvd(train, test, test_to_scikit, "SVD.tsv")  
    recommender.recommenderWithBiasedMF(train, test, test_to_scikit,"BIASEDMF.tsv")        
    recommender.recommenderWithBias(train, test, test_to_scikit,"BIAS.tsv")

if __name__ == '__main__':
    main()
    # hitCount("itemKNN.tsv")  