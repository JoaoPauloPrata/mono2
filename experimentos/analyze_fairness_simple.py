"""
Análise simples e direta dos dados de fairness existentes
"""

import pandas as pd
import numpy as np

def analyze_fairness_data():
    """
    Analisa dados de fairness da janela 1
    """
    print("🔍 ANÁLISE DE FAIRNESS - JANELA 1")
    print("="*60)
    
    # Carrega dados
    file_path = "data/MetricsForMethods/Fairness/FairnessDifferences_Window1.csv"
    
    try:
        df = pd.read_csv(file_path)
        print(f"✅ Dados carregados: {len(df)} métodos")
        
        # Define grupos de métodos
        simple_methods = ["SVD", "BIASEDMF", "NMF", "StochasticItemKNN"]
        hybrid_methods = ["BayesianRidge", "Tweedie", "Ridge", "RandomForest", 
                         "Bagging", "AdaBoost", "GradientBoosting", "LinearSVR"]
        
        metrics = ["RMSE", "MAE", "NDCG", "F1"]
        
        print("\n📊 ANÁLISE POR MÉTRICA:")
        print("-" * 60)
        
        summary_results = {}
        
        for metric in metrics:
            abs_diff_col = f"{metric}_AbsDiff"
            
            if abs_diff_col in df.columns:
                # Calcula médias por grupo
                simple_data = df[df['method'].isin(simple_methods)][abs_diff_col].dropna()
                hybrid_data = df[df['method'].isin(hybrid_methods)][abs_diff_col].dropna()
                
                simple_avg = simple_data.mean() if len(simple_data) > 0 else np.nan
                hybrid_avg = hybrid_data.mean() if len(hybrid_data) > 0 else np.nan
                improvement = simple_avg - hybrid_avg if not (np.isnan(simple_avg) or np.isnan(hybrid_avg)) else np.nan
                
                print(f"\n🎯 {metric}:")
                print(f"   Métodos Simples  - Média: {simple_avg:.6f} ({len(simple_data)} métodos)")
                print(f"   Métodos Híbridos - Média: {hybrid_avg:.6f} ({len(hybrid_data)} métodos)")
                
                if not np.isnan(improvement):
                    if improvement > 0:
                        print(f"   ✅ MELHORIA: Híbridos são {abs(improvement):.6f} melhores")
                        status = "Híbridos melhores"
                    else:
                        print(f"   ❌ PIORA: Híbridos são {abs(improvement):.6f} piores")
                        status = "Simples melhores"
                else:
                    print(f"   ⚠️  Não foi possível calcular melhoria")
                    status = "Indeterminado"
                
                summary_results[metric] = {
                    'simple_avg': simple_avg,
                    'hybrid_avg': hybrid_avg,
                    'improvement': improvement,
                    'status': status
                }
        
        # Resumo consolidado
        print("\n" + "="*60)
        print("📋 RESUMO CONSOLIDADO:")
        print("-" * 60)
        
        for metric, results in summary_results.items():
            improvement = results['improvement']
            if not np.isnan(improvement):
                print(f"{metric:5}: {results['status']:18} (diferença: {abs(improvement):.4f})")
            else:
                print(f"{metric:5}: Indeterminado")
        
        # Ranking dos métodos por fairness
        print(f"\n🏆 RANKING POR FAIRNESS (menor diferença = melhor):")
        print("-" * 60)
        
        for metric in metrics:
            abs_diff_col = f"{metric}_AbsDiff"
            if abs_diff_col in df.columns:
                print(f"\n{metric}:")
                sorted_methods = df.sort_values(abs_diff_col)
                for i, (_, row) in enumerate(sorted_methods.iterrows(), 1):
                    method_type = "S" if row['method'] in simple_methods else "H"
                    print(f"   {i:2}. {row['method']:15} ({method_type}): {row[abs_diff_col]:.4f}")
        
        # Detalhes completos
        print(f"\n📋 DETALHES COMPLETOS:")
        print("-" * 90)
        print(f"{'Método':15} {'Tipo':8} {'RMSE':8} {'MAE':8} {'NDCG':8} {'F1':8}")
        print("-" * 90)
        
        for _, row in df.iterrows():
            method_type = "Simples" if row['method'] in simple_methods else "Híbrido"
            print(f"{row['method']:15} {method_type:8} {row['RMSE_AbsDiff']:.4f}   "
                  f"{row['MAE_AbsDiff']:.4f}   {row['NDCG_AbsDiff']:.4f}   {row['F1_AbsDiff']:.4f}")
        
        print("\n" + "="*60)
        print("💡 INTERPRETAÇÃO:")
        print("• Diferença absoluta menor = Melhor fairness entre gêneros")
        print("• Se híbridos têm média menor = Híbridos são mais justos")
        print("• Se simples têm média menor = Simples são mais justos")
        print("="*60)
        
    except FileNotFoundError:
        print(f"❌ Arquivo não encontrado: {file_path}")
        print("Execute primeiro a avaliação de fairness!")
    except Exception as e:
        print(f"❌ Erro durante análise: {e}")

if __name__ == "__main__":
    analyze_fairness_data()
