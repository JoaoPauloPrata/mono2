import pandas as pd
import numpy as np
import os

class FairnessDifferenceCalculator:
    """
    Calcula diferenças absolutas entre métricas de grupos masculino e feminino
    """
    
    def __init__(self):
        self.fairness_dir = "data/MetricsForMethods/Fairness"
    
    def calculate_differences_for_window(self, window_count):
        """
        Calcula diferenças absolutas para uma janela específica
        """
        male_file = f"{self.fairness_dir}/MetricsForWindow{window_count}Male.csv"
        female_file = f"{self.fairness_dir}/MetricsForWindow{window_count}Female.csv"
        
        if not os.path.exists(male_file):
            print(f"Arquivo masculino não encontrado: {male_file}")
            return None
            
        if not os.path.exists(female_file):
            print(f"Arquivo feminino não encontrado: {female_file}")
            return None
        
        try:
            # Carrega dados dos grupos
            male_df = pd.read_csv(male_file, index_col='method')
            female_df = pd.read_csv(female_file, index_col='method')
            
            print(f"Processando janela {window_count}:")
            print(f"  Métodos masculinos: {len(male_df)}")
            print(f"  Métodos femininos: {len(female_df)}")
            
            # Encontra métodos comuns
            common_methods = male_df.index.intersection(female_df.index)
            
            if len(common_methods) == 0:
                print("  Nenhum método comum encontrado")
                return None
            
            # Calcula diferenças absolutas
            differences = []
            
            for method in common_methods:
                male_metrics = male_df.loc[method]
                female_metrics = female_df.loc[method]
                
                diff_row = {
                    'method': method,
                    'window': window_count
                }
                
                # Calcula diferença absoluta para cada métrica
                for metric in ['RMSE', 'MAE', 'NDCG', 'F1']:
                    if metric in male_metrics.index and metric in female_metrics.index:
                        male_val = male_metrics[metric]
                        female_val = female_metrics[metric]
                        
                        # Verifica se ambos os valores são válidos
                        if pd.notna(male_val) and pd.notna(female_val):
                            abs_diff = abs(male_val - female_val)
                            
                            diff_row.update({
                                f'{metric}_Male': male_val,
                                f'{metric}_Female': female_val,
                                f'{metric}_AbsDiff': abs_diff
                            })
                        else:
                            diff_row.update({
                                f'{metric}_Male': male_val if pd.notna(male_val) else None,
                                f'{metric}_Female': female_val if pd.notna(female_val) else None,
                                f'{metric}_AbsDiff': None
                            })
                
                differences.append(diff_row)
            
            # Converte para DataFrame
            diff_df = pd.DataFrame(differences)
            
            # Salva resultados
            output_file = f"{self.fairness_dir}/FairnessDifferences_Window{window_count}.csv"
            diff_df.to_csv(output_file, index=False)
            
            print(f"  Diferenças calculadas para {len(differences)} métodos")
            print(f"  Salvo em: {output_file}")
            
            # Mostra resumo das maiores diferenças
            if 'RMSE_AbsDiff' in diff_df.columns:
                max_rmse_diff = diff_df.loc[diff_df['RMSE_AbsDiff'].idxmax()]
                print(f"  Maior diferença RMSE: {max_rmse_diff['method']} = {max_rmse_diff['RMSE_AbsDiff']:.4f}")
            
            return diff_df
            
        except Exception as e:
            print(f"Erro ao processar janela {window_count}: {e}")
            return None
    
    def calculate_differences_all_windows(self, start_window=1, end_window=20):
        """
        Calcula diferenças para todas as janelas e cria resumo consolidado
        """
        print(f"Calculando diferenças de fairness para janelas {start_window}-{end_window}")
        print("="*60)
        
        all_differences = []
        successful_windows = []
        
        for window in range(start_window, end_window + 1):
            diff_df = self.calculate_differences_for_window(window)
            
            if diff_df is not None:
                all_differences.append(diff_df)
                successful_windows.append(window)
                print()
        
        if all_differences:
            # Consolida todos os resultados
            consolidated_df = pd.concat(all_differences, ignore_index=True)
            
            # Salva resultados consolidados
            consolidated_file = f"{self.fairness_dir}/FairnessDifferences_AllWindows.csv"
            consolidated_df.to_csv(consolidated_file, index=False)
            
            print("="*60)
            print(f"RESUMO CONSOLIDADO:")
            print(f"Janelas processadas: {successful_windows}")
            print(f"Total de avaliações: {len(consolidated_df)}")
            print(f"Resultados salvos em: {consolidated_file}")
            
            # Análise estatística básica
            if 'RMSE_AbsDiff' in consolidated_df.columns:
                rmse_stats = consolidated_df['RMSE_AbsDiff'].describe()
                print(f"\nEstatísticas RMSE AbsDiff:")
                print(f"  Média: {rmse_stats['mean']:.4f}")
                print(f"  Mediana: {rmse_stats['50%']:.4f}")
                print(f"  Máximo: {rmse_stats['max']:.4f}")
                
                # Métodos com maiores diferenças médias
                avg_by_method = consolidated_df.groupby('method')['RMSE_AbsDiff'].mean().sort_values(ascending=False)
                print(f"\nMétodos com maiores diferenças médias (RMSE):")
                for method, avg_diff in avg_by_method.head(5).items():
                    print(f"  {method}: {avg_diff:.4f}")
            
            return consolidated_df
        else:
            print("Nenhuma janela foi processada com sucesso")
            return None
    
    def generate_fairness_summary(self):
        """
        Gera resumo executivo de fairness
        """
        consolidated_file = f"{self.fairness_dir}/FairnessDifferences_AllWindows.csv"
        
        if not os.path.exists(consolidated_file):
            print("Arquivo consolidado não encontrado. Execute calculate_differences_all_windows() primeiro.")
            return
        
        df = pd.read_csv(consolidated_file)
        
        summary = {
            'total_evaluations': len(df),
            'unique_methods': df['method'].nunique(),
            'windows_evaluated': df['window'].nunique(),
        }
        
        # Análise por métrica
        metrics = ['RMSE', 'MAE', 'NDCG', 'F1']
        
        for metric in metrics:
            abs_diff_col = f'{metric}_AbsDiff'
            if abs_diff_col in df.columns:
                valid_data = df[abs_diff_col].dropna()
                if len(valid_data) > 0:
                    summary[f'{metric}_mean_diff'] = valid_data.mean()
                    summary[f'{metric}_std_diff'] = valid_data.std()
                    summary[f'{metric}_max_diff'] = valid_data.max()
        
        # Salva resumo
        summary_file = f"{self.fairness_dir}/FairnessSummary.csv"
        summary_df = pd.DataFrame([summary])
        summary_df.to_csv(summary_file, index=False)
        
        print(f"Resumo executivo salvo em: {summary_file}")
        return summary


if __name__ == "__main__":
    calculator = FairnessDifferenceCalculator()
    
    # Exemplo de uso
    print("Calculando diferenças de fairness...")
    
    # Para uma janela específica
    # calculator.calculate_differences_for_window(1)
    
    # Para todas as janelas
    calculator.calculate_differences_all_windows(1, 5)  # Teste com 5 janelas
    
    # Gera resumo
    calculator.generate_fairness_summary()
