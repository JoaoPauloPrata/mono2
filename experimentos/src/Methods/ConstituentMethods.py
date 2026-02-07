
import pandas as pd
from pathlib import Path
from lenskit.algorithms.item_knn import ItemItem
from lenskit.algorithms.svd import BiasedSVD
from lenskit.algorithms.als import BiasedMF
from surprise import Dataset, Reader, NMF
from src.Methods.StochasticItemKNN import StochasticItemKNN

class ConstituentMethods:
    def __init__(self):
        # Resolve o diretório raiz do projeto a partir deste arquivo: .../src/Methods/ -> .../
        self.project_root = Path(__file__).resolve().parents[2]
        self.predictions_base_dir = self.project_root / 'data' / 'predictions'

    def recommenderWithStochasticItemKNN(self, train, test, path):
        print("Running Stochastic Item KNN Recommender...")
        stiknn = StochasticItemKNN(k=20, temperature=0.2)
        stiknn.fit(train)
        predictions = []
        for _, row in test.iterrows():
            user = row['user']
            item = row['item']
            pred = stiknn._predict_single(user, item)
            predictions.append(pred)
        recs = pd.DataFrame({
            'user': test['user'],
            'item': test['item'],
            'prediction': predictions
        })
        out_dir = self.predictions_base_dir / 'StochasticItemKNN'
        out_dir.mkdir(parents=True, exist_ok=True)
        recs.to_csv(str(out_dir / path), sep='\t', index=False)

    def recommenderWithNMF(self, train, test, path):
        reader = Reader(rating_scale=(1, 5))
        data = Dataset.load_from_df(train[['user', 'item', 'rating']], reader)
        trainset = data.build_full_trainset()
        nmf = NMF(n_factors=15, n_epochs=50, reg_pu=0.06, reg_qi=0.06)
        nmf.fit(trainset)
        testset = list(zip(test['user'], test['item'], test['rating']))
        predictions = nmf.test(testset)
        recs = pd.DataFrame({
            'user': [pred.uid for pred in predictions],
            'item': [pred.iid for pred in predictions],
            'prediction': [pred.est for pred in predictions]
        })
        out_dir = self.predictions_base_dir / 'NMF'
        out_dir.mkdir(parents=True, exist_ok=True)
        recs.to_csv(str(out_dir / path), sep='\t', index=False)

        
    def recommenderWithSvd(self, train, test, path):
        svd = BiasedSVD(features=50, damping=5, bias=True, algorithm='randomized')
        svd.fit(train)
        predictions = svd.predict(test)
        recs = pd.DataFrame({
            'user': test['user'],
            'item': test['item'],
            'prediction': predictions
        })
        out_dir = self.predictions_base_dir / 'SVD'
        out_dir.mkdir(parents=True, exist_ok=True)
        recs.to_csv(str(out_dir / path), sep='\t', index=False)

    def recommenderWithBiasedMF(self, train, test, path):
        biasedMF = BiasedMF(features=50, iterations=20, reg=0.1, damping=5, bias=True, method='cd')
        biasedMF.fit(train)
        predictions = biasedMF.predict(test)
        recs = pd.DataFrame({
            'user': test['user'],
            'item': test['item'],
            'prediction': predictions
        })
        out_dir = self.predictions_base_dir / 'BIASEDMF'
        out_dir.mkdir(parents=True, exist_ok=True)
        recs.to_csv(str(out_dir / path), sep='\t', index=False)
