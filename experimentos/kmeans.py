import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
import os

def splitGroups(base_path: str, output_dir: str, window: int):
    # 1. Ler o arquivo ratings.csv (esperado com colunas: user,item,rating,timestamp,date)
    ratings = pd.read_csv(base_path)

    # 2. Contar número de interações por usuário
    user_activity = ratings.groupby('user').size().reset_index(name='num_interactions')

    # 3. Aplicar KMeans com 3 clusters
    X = user_activity[['num_interactions']].values
    kmeans = KMeans(n_clusters=2, random_state=42)
    user_activity['activity_group'] = kmeans.fit_predict(X)

    # 4. Reordenar os clusters com base na média de interações
    group_order = user_activity.groupby('activity_group')['num_interactions'].mean().sort_values().index
    group_mapping = {old: new for new, old in enumerate(group_order)}
    user_activity['activity_group'] = user_activity['activity_group'].map(group_mapping)

    # 5. Criar diretório de saída se não existir
    os.makedirs(output_dir, exist_ok=True)

    # 6. Salvar os IDs por grupo em arquivos CSV
    for group in range(2):
        group_name = ['low', 'high'][group]
        users = user_activity[user_activity['activity_group'] == group]['user']
        output_path = os.path.join(output_dir, f'window_{window}_group_{group_name}.csv')
        users.to_csv(output_path, index=False, header=False)
        print(f"Grupo {group_name.capitalize()} salvo em: {output_path} (n = {len(users)})")

# 🔧 Exemplo de uso
for i in range(1, 21):
    base_path = f"./data/windows/test_to_get_regression_train_data_{i}.csv"
    output_dir = f"./data/windows/kmeansGroup/hybrid/window_{i}/"
    splitGroups(base_path, output_dir, i)

for i in range(1, 21):
    base_path = f"./data/windows/train_to_get_constituent_methods_{i}.csv"
    output_dir = f"./data/windows/kmeansGroup/constituent/window_{i}/"
    splitGroups(base_path, output_dir, i)
