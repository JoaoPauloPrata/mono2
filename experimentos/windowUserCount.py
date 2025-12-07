import pandas as pd

def count_users_in_window(base_path: str) -> int:
    """
    Lê um arquivo CSV de uma janela e retorna o número de usuários únicos.
    Espera que a coluna de usuário se chame 'user'.
    """
    df = pd.read_csv(base_path)
    num_users = df['user'].nunique()
    print(f"Número de usuários únicos em {base_path}: {num_users}")
    return num_users

# Exemplo de uso
for i in range(1, 21):
    base_path = f"./data/windows/train_to_get_constituent_methods_{i}.csv"
    count_users_in_window(base_path)