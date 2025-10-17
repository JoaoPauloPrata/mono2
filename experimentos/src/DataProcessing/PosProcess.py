import pandas as pd
from functools import reduce
import os
from glob import glob
import numpy as np

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
    contendo apenas os pares (user, item) presentes em todos os arquivos
    e que têm predições válidas (não-NaN) em todos os métodos.
    """
    if not file_paths:
        return pd.DataFrame(columns=['user', 'item'])
    
    dfs = []
    for file_path in file_paths:
        if not os.path.exists(file_path):
            print(f"Aviso: Arquivo não encontrado: {file_path}")
            continue
        df = pd.read_csv(file_path, sep='\t' if file_path.endswith('.tsv') else ',')
        
        # Remove pares que não receberam recomendação (NaN, infinito, ou vazios)
        df = df.dropna(subset=['prediction'])
        df = df[np.isfinite(df['prediction'])]
        df = df[df['prediction'].notna()]
        
        print(f"  Arquivo {os.path.basename(file_path)}: {len(df)} pares válidos (após remoção de NaN)")
        
        # Garante que os pares são únicos e ordena por user, item
        df_pairs = df[['user', 'item']].drop_duplicates().sort_values(['user', 'item']).reset_index(drop=True)
        dfs.append(df_pairs)

    if not dfs:
        return pd.DataFrame(columns=['user', 'item'])

    # Faz interseção dos pares (user, item) entre todos os DataFrames
    common_pairs = reduce(lambda left, right: pd.merge(left, right, on=['user', 'item']), dfs)
    print(f"Pares comuns (com predições válidas em todos os métodos): {len(common_pairs)}")
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
            
            # Encontra pares comuns (apenas com predições válidas)
            print(f"Analisando pares válidos em cada método:")
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
                
                # Remove qualquer linha que ainda possa ter predição NaN após o merge
                initial_count = len(df_filtered)
                df_filtered = df_filtered.dropna(subset=['prediction'])
                df_filtered = df_filtered[np.isfinite(df_filtered['prediction'])]
                final_count = len(df_filtered)
                
                if initial_count != final_count:
                    print(f"Removidos {initial_count - final_count} pares com predições inválidas")
                
                # Ordena por user, item para consistência
                df_filtered = df_filtered.sort_values(['user', 'item']).reset_index(drop=True)
                
                print(f"{len(df_filtered)} linhas")
                
                # Salva arquivo filtrado
                output_file = os.path.join(output_dir, f"window_{window}_{file_type}_{method}.tsv")
                df_filtered.to_csv(output_file, sep='\t', index=False)
            
            print(f"Janela {window}, tipo {file_type}: {len(common_pairs)} pares comuns mantidos")

def add_header_to_hybrid():
    """
    Adiciona header e colunas user/item aos arquivos de predições híbridas
    usando os pares user/item dos arquivos filtered_predictions
    """
    algo = ["BayesianRidge", "Tweedie", "Ridge", "RandomForest", "Bagging", "AdaBoost", "GradientBoosting", "LinearSVR"]
    hybrid_dir = "../../data/HybridPredictions"
    filtered_dir = "../../data/filtered_predictions"
    
    # Busca todas as janelas disponíveis nos arquivos híbridos
    windows = set()
    for file_name in os.listdir(hybrid_dir):
        if file_name.startswith('window_') and file_name.endswith('.tsv'):
            # Extrai número da janela
            parts = file_name.split('_')
            if len(parts) >= 2:
                window_num = parts[1]
                try:
                    windows.add(int(window_num))
                except ValueError:
                    continue
    
    windows = sorted(windows)
    print(f"Janelas encontradas para correção: {windows}")
    
    for window in windows:
        print(f"\nProcessando janela {window}...")
        
        # Carrega o arquivo de referência para obter pares user/item
        reference_file = os.path.join(filtered_dir, f"window_{window}_constituent_methods_BIAS.tsv")
        
        if not os.path.exists(reference_file):
            print(f"Arquivo de referência não encontrado: {reference_file}")
            continue
            
        # Carrega pares user/item do arquivo de referência
        reference_df = pd.read_csv(reference_file, sep='\t')
        user_item_pairs = reference_df[['user', 'item']].copy()
        
        print(f"  Pares user/item carregados: {len(user_item_pairs)}")
        
        # Processa cada algoritmo híbrido
        for method in algo:
            hybrid_file = os.path.join(hybrid_dir, f"window_{window}_predicted{method}.tsv")
            
            if not os.path.exists(hybrid_file):
                print(f"  Arquivo híbrido não encontrado: {hybrid_file}")
                continue
            
            # Carrega predições híbridas (apenas valores) - pula a primeira linha (linha com "0")
            predictions_df = pd.read_csv(hybrid_file, sep='\t', header=None, names=['prediction'], skiprows=1)
            
            print(f"  {method}: {len(predictions_df)} predições -> ", end="")
            
            # Verifica se o número de predições coincide com os pares
            if len(predictions_df) != len(user_item_pairs):
                print(f"ERRO: {len(predictions_df)} predições != {len(user_item_pairs)} pares!")
                continue
            
            # Combina pares user/item com predições
            final_df = user_item_pairs.copy()
            final_df['prediction'] = predictions_df['prediction'].values
            
            # Salva arquivo corrigido
            final_df.to_csv(hybrid_file, sep='\t', index=False)
            print(f"corrigido com header")
        
        print(f"Janela {window} processada com sucesso!")
    
    print("\nTodos os arquivos híbridos foram corrigidos!")

def filter_test_data_to_match_predictions():
    """
    Filtra os arquivos de teste (test_to_get_regression_train_data) para que contenham
    apenas os pares user-item presentes nos arquivos de predição filtrados.
    Salva os arquivos filtrados em data/windows/processed para preservar os originais.
    """
    filtered_dir = "../../data/filtered_predictions"
    windows_dir = "../../data/windows"
    processed_dir = "../../data/windows/processed"
    
    # Cria diretório processed se não existir
    os.makedirs(processed_dir, exist_ok=True)
    
    # Busca todas as janelas disponíveis nos arquivos filtrados
    windows = set()
    if os.path.exists(filtered_dir):
        files = os.listdir(filtered_dir)
        for file in files:
            if file.startswith('window_') and 'constituent_methods' in file and file.endswith('.tsv'):
                # Extrai número da janela
                parts = file.split('_')
                if len(parts) >= 2:
                    window_num = parts[1]
                    try:
                        windows.add(int(window_num))
                    except ValueError:
                        continue
    
    windows = sorted(windows)
    print(f"Janelas encontradas para filtragem de dados de teste: {windows}")
    
    for window in windows:
        print(f"\nProcessando dados de teste para janela {window}...")
        
        # Define arquivo de referência baseado no tipo de teste
        test_files_with_references = [
            {
                "file": f"test_to_get_regression_train_data_{window}.csv",
                "reference": f"window_{window}_scikit_train_BIAS.tsv",
                "description": "scikit_train"
            },
            {
                "file": f"test_to_get_constituent_methods_{window}.csv", 
                "reference": f"window_{window}_constituent_methods_BIAS.tsv",
                "description": "constituent_methods"
            }
        ]
        
        for test_config in test_files_with_references:
            test_file = test_config["file"]
            reference_file_name = test_config["reference"]
            description = test_config["description"]
            
            test_file_path = os.path.join(windows_dir, test_file)
            reference_file_path = os.path.join(filtered_dir, reference_file_name)
            
            print(f"\n  Processando {test_file} (referência: {description})")
            
            if not os.path.exists(test_file_path):
                print(f"    Arquivo de teste não encontrado: {test_file}")
                continue
                
            if not os.path.exists(reference_file_path):
                print(f"    Arquivo de referência não encontrado: {reference_file_name}")
                continue
            
            try:
                # Carrega arquivo de referência para este tipo específico
                reference_df = pd.read_csv(reference_file_path, sep='\t')
                if reference_df.empty:
                    print(f"    Arquivo de referência vazio: {reference_file_name}")
                    continue
                    
                # Verifica se as colunas necessárias existem
                if not {'user', 'item'}.issubset(reference_df.columns):
                    print(f"    Arquivo de referência não tem colunas user/item: {reference_file_name}")
                    continue
                    
                valid_pairs = reference_df[['user', 'item']].copy()
                
                # Remove duplicatas e ordena
                valid_pairs = valid_pairs.drop_duplicates().sort_values(['user', 'item']).reset_index(drop=True)
                
                print(f"    Pares válidos de referência ({description}): {len(valid_pairs)}")
                
                if valid_pairs.empty:
                    print(f"    Nenhum par válido encontrado na referência!")
                    continue
                
                # Carrega arquivo de teste original
                test_df = pd.read_csv(test_file_path)
                original_count = len(test_df)
                
                if test_df.empty:
                    print(f"    {test_file}: arquivo vazio - pulando")
                    continue
                    
                print(f"    {test_file}: {original_count} linhas -> ", end="")
                
                # Verifica se as colunas necessárias existem
                if not {'user', 'item'}.issubset(test_df.columns):
                    print(f"arquivo não tem colunas user/item - pulando")
                    continue
                
                # Filtra para manter apenas pares válidos
                test_filtered = pd.merge(valid_pairs, test_df, on=['user', 'item'], how='inner')
                filtered_count = len(test_filtered)
                
                print(f"{filtered_count} linhas (removidas {original_count - filtered_count})")
                
                # Salva arquivo filtrado na pasta processed (preserva o original)
                processed_file_path = os.path.join(processed_dir, test_file.replace('.csv', '_filtered.csv'))
                test_filtered.to_csv(processed_file_path, index=False)
                
                print(f"    Salvo em: {processed_file_path}")
                
            except Exception as e:
                print(f"    Erro ao processar {test_file}: {e}")
                continue
        
        print(f"Dados de teste para janela {window} filtrados com sucesso!")
    
    print(f"\nTodos os arquivos de teste foram filtrados e salvos em: {processed_dir}")

# Executa o filtro
if __name__ == "__main__":
    print("Iniciando filtragem de pares comuns e dados de teste...")
    
    # Primeiro filtra os arquivos de predição para pares comuns
    print("\n=== PASSO 1: Filtrando arquivos de predição ===")
    filter_to_common_pairs_by_window_and_type()
    
    # Depois filtra os dados de teste para serem consistentes
    # print("\n=== PASSO 2: Filtrando dados de teste ===")
    filter_test_data_to_match_predictions()
    
    # Opcional: corrigir headers dos híbridos se necessário
    # print("\n=== PASSO 3: Corrigindo headers híbridos ===")
    # add_header_to_hybrid()
    
    print("\n✅ Processamento completo finalizado!")