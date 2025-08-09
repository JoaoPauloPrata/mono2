import pandas as pd
from lenskit.algorithms.user_knn import UserUser
from lenskit import topn
from lenskit.datasets import ML100K

# Carregar o dataset MovieLens 100k incluído no LensKit
ml100k = ML100K('ml-100k')  # Baixa automaticamente se não estiver no cache
ratings = ml100k.ratings

# Verificar os primeiros registros
print("Exemplo de ratings:")
print(ratings.head())

# Criar o modelo de recomendação User-User
model = UserUser(k=5)  # k = número de vizinhos

# Treinar o modelo com todos os dados
model.fit(ratings)

# Escolher um usuário de exemplo
user = 42

# Gerar top 10 recomendações para o usuário
recs = topn.recommend(model, [user], n=10)

# Mostrar as recomendações
print(f"\nTop 10 recomendações para o usuário {user}:")
print(recs)
