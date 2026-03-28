import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from sklearn.metrics.pairwise import cosine_similarity

class StochasticItemKNN:
    def __init__(self, k=20, temperature=0.1, random_state=None):
        self.k = k
        self.temperature = temperature
        self.rng = np.random.default_rng(random_state)
        self.sim_matrix = None
        self.item_mapper = {}     # ID Real -> Índice Matriz
        self.user_mapper = {}     # ID Real -> Índice Matriz
        self.train_matrix = None
        self.global_mean = 3.0
        
    def fit(self, df):
        # 1. Mapeamento seguro de IDs
        users = df['user'].unique()
        items = df['item'].unique()
        
        self.user_mapper = {u: i for i, u in enumerate(users)}
        self.item_mapper = {i: idx for idx, i in enumerate(items)}
        
        # 2. Criar Matriz Esparsa
        rows = df['user'].map(self.user_mapper)
        cols = df['item'].map(self.item_mapper)
        data = df['rating']
        
        self.train_matrix = csr_matrix((data, (rows, cols)), shape=(len(users), len(items)))
        self.global_mean = float(np.mean(data)) if len(data) else 3.0
        
        # 3. Similaridade Item-Item
        print("Calculando matriz de similaridade (Dense)...")
        # Transpomos para (Item x User) para calcular similaridade entre itens
        item_user_matrix = self.train_matrix.T
        self.sim_matrix = cosine_similarity(item_user_matrix, dense_output=True)
        
        # Zerar diagonal e valores negativos
        np.fill_diagonal(self.sim_matrix, 0)
        self.sim_matrix[self.sim_matrix < 0] = 0
        
        print("Treinamento concluído.")

    def _predict_single(self, user_id, item_id):
        # Lógica de Cold Start segura
        if user_id not in self.user_mapper or item_id not in self.item_mapper:
            return self.global_mean

        user_idx = self.user_mapper[user_id]
        item_idx = self.item_mapper[item_id]

        # Acesso otimizado à linha esparsa do usuário
        # Em vez de converter tudo para denso, pegamos apenas os índices relevantes
        u_row = self.train_matrix[user_idx]
        rated_indices = u_row.indices # Índices dos itens que o usuário avaliou
        user_ratings = u_row.data     # Notas dadas

        if len(rated_indices) == 0:
            return self.global_mean

        # Similaridades entre o Item Alvo e os itens que o usuário viu
        similarities = self.sim_matrix[item_idx, rated_indices]
        
        if np.sum(similarities) == 0:
            return self.global_mean

        # --- SAMPLING ESTOCÁSTICO (Com correção numérica) ---
        
        # 1. Ajuste de temperatura
        logits = similarities / self.temperature
        
        # 2. Shift trick para estabilidade numérica (evita overflow do exp)
        logits_shifted = logits - np.max(logits)
        exp_sim = np.exp(logits_shifted)
        probs = exp_sim / np.sum(exp_sim)
        
        # 3. Definição do K real
        n_neighbors = min(self.k, len(rated_indices))
        
        # 4. Sorteio
        # Escolhemos ÍNDICES do vetor 'rated_indices', não os índices de itens globais ainda
        chosen_local_indices = self.rng.choice(
            np.arange(len(rated_indices)), 
            size=n_neighbors, 
            replace=False, 
            p=probs
        )
        
        chosen_sims = similarities[chosen_local_indices]
        chosen_ratings = user_ratings[chosen_local_indices]
        
        # Predição
        if np.sum(chosen_sims) == 0:
            return np.mean(chosen_ratings)
            
        pred = np.dot(chosen_ratings, chosen_sims) / np.sum(chosen_sims)
        return pred

    def predict(self, test_df):
        print(f"Gerando predições estocásticas para {len(test_df)} pares...")
        preds = []
        
        test_users = test_df['user'].values
        test_items = test_df['item'].values
        
        # Iteração simples e segura
        # Passamos os IDs REAIS para _predict_single lidar com o mapeamento
        for u, i in zip(test_users, test_items):
            try:
                pred = self._predict_single(u, i)
                preds.append(pred)
            except Exception as e:
                # Fallback de segurança extrema
                preds.append(self.global_mean)
                
        return np.array(preds)