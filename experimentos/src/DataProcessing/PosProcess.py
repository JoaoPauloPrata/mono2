import pandas as pd
from functools import reduce
import os
from glob import glob

def filter_common_pairs_in_files(input_dir, output_dir):
    # Percorre todos os arquivos na pasta input_dir
    file_paths = [os.path.join(input_dir, f) for f in os.listdir(input_dir) if f.endswith('.csv') or f.endswith('.tsv')]
    dataframes = [pd.read_csv(file, sep='\t') for file in file_paths]
    all_pairs_to_drop = []
    for df in dataframes:
        missing_predictions = df[df['prediction'].isna()][['user', 'item']]
        for pair in missing_predictions.values:
            pair_tuple = tuple(pair)
            if pair_tuple not in all_pairs_to_drop:
                all_pairs_to_drop.append(pair_tuple)

    for file_path in file_paths:
        trainDatasetToClean = pd.read_csv(file_path, sep=',' if file_path.endswith('.csv') else '\t')
        trainDatasetToClean.drop(trainDatasetToClean[trainDatasetToClean[['user', 'item']].apply(tuple, axis=1).isin(all_pairs_to_drop)].index, inplace=True)
        output_file = os.path.join(output_dir, os.path.basename(file_path))
        trainDatasetToClean.to_csv(output_file, sep=',' if file_path.endswith('.csv') else '\t', index=False)

def get_common_user_item_pairs(file_paths):
    """
    Recebe uma lista de caminhos de arquivos (csv/tsv) e retorna um DataFrame
    contendo apenas os pares (user, item) presentes em todos os arquivos.
    """
    if not file_paths:
        return pd.DataFrame(columns=['user', 'item'])
    
    dfs = []
    for file_path in file_paths:
        if not os.path.exists(file_path):
            print(f"Aviso: Arquivo não encontrado: {file_path}")
            continue
        df = pd.read_csv(file_path, sep='\t' if file_path.endswith('.tsv') else ',')
        # Garante que os pares são únicos e ordena por user, item
        df_pairs = df[['user', 'item']].drop_duplicates().sort_values(['user', 'item']).reset_index(drop=True)
        dfs.append(df_pairs)

    if not dfs:
        return pd.DataFrame(columns=['user', 'item'])

    # Faz interseção dos pares (user, item) entre todos os DataFrames
    common_pairs = reduce(lambda left, right: pd.merge(left, right, on=['user', 'item']), dfs)
    print(f"Pares comuns encontrados: {len(common_pairs)}")
    return common_pairs

def filter_to_common_pairs_by_window_and_type():
    """
    Filtra todos os arquivos de predições para garantir que tenham os mesmos pares user-item
    por janela e tipo (constituent_methods ou scikit_train).
    """
    base_dir = "../../data/predictions"
    output_dir = "../../data/filtered_predictions"
    
    # Cria diretório de saída se não existir
    os.makedirs(output_dir, exist_ok=True)
    
    # Lista de métodos disponíveis
    methods = ["BIAS", "SVD", "userKNN", "itemKNN", "BIASEDMF"]
    
    # Busca todas as janelas disponíveis
    windows = set()
    for method in methods:
        method_dir = os.path.join(base_dir, method)
        if os.path.exists(method_dir):
            files = os.listdir(method_dir)
            for file in files:
                if file.startswith('window_') and file.endswith('.tsv'):
                    # Extrai número da janela
                    parts = file.split('_')
                    if len(parts) >= 2:
                        window_num = parts[1]
                        windows.add(int(window_num))
    
    windows = sorted(windows)
    print(f"Janelas encontradas: {windows}")
    
    # Tipos de arquivo
    file_types = ["constituent_methods", "scikit_train"]
    
    for window in windows:
        for file_type in file_types:
            print(f"\nProcessando janela {window}, tipo {file_type}")
            
            # Coleta todos os arquivos para esta janela e tipo
            file_paths = []
            for method in methods:
                file_path = os.path.join(base_dir, method, f"window_{window}_{file_type}_{method}.tsv")
                if os.path.exists(file_path):
                    file_paths.append(file_path)
                else:
                    print(f"Arquivo não encontrado: {file_path}")
            
            if len(file_paths) < len(methods):
                print(f"Aviso: Apenas {len(file_paths)}/{len(methods)} arquivos encontrados para janela {window}, tipo {file_type}")
                continue
            
            # Encontra pares comuns
            common_pairs = get_common_user_item_pairs(file_paths)
            
            if common_pairs.empty:
                print(f"Nenhum par comum encontrado para janela {window}, tipo {file_type}")
                continue
                
            # Filtra cada arquivo para manter apenas os pares comuns
            for file_path in file_paths:
                method = os.path.basename(file_path).split('_')[-1].replace('.tsv', '')
                
                # Carrega o arquivo original
                df = pd.read_csv(file_path, sep='\t')
                print(f"  {method}: {len(df)} linhas -> ", end="")
                
                # Faz merge com pares comuns para filtrar
                df_filtered = pd.merge(common_pairs, df, on=['user', 'item'], how='left')
                
                # Ordena por user, item para consistência
                df_filtered = df_filtered.sort_values(['user', 'item']).reset_index(drop=True)
                
                print(f"{len(df_filtered)} linhas")
                
                # Salva arquivo filtrado
                output_file = os.path.join(output_dir, f"window_{window}_{file_type}_{method}.tsv")
                df_filtered.to_csv(output_file, sep='\t', index=False)
            
            print(f"Janela {window}, tipo {file_type}: {len(common_pairs)} pares comuns mantidos")

def add_header_to_hybrid():
    algo = ["BayesianRidge", "Tweedie", "Ridge", "RandomForest", "Bagging", "AdaBoost", "GradientBoosting", "LinearSVR"]
    filtered_dir = "data/filtered_predictions"
    for file_name in os.listdir(filtered_dir):
        if file_name.endswith('.tsv'):
            filePath = os.path.join(filtered_dir, file_name)
            df = pd.read_csv(filePath, delimiter='\t', header=None, names=['user', 'item', 'prediction'])
            df.to_csv(filePath, sep='\t', index=False)

# Executa o filtro
if __name__ == "__main__":
    print("Iniciando filtragem de pares comuns...")
    filter_to_common_pairs_by_window_and_type()
    print("Filtragem concluída!")