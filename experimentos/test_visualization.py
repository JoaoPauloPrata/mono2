"""
Teste manual da visualização de fairness
"""

from src.Metrics.FairnessVisualization import FairnessVisualization
import pandas as pd

print("🧪 Teste da Visualização de Fairness")
print("="*50)

# Carrega dados existentes
visualizer = FairnessVisualization()

try:
    # Testa carregamento dos dados
    print("1. Carregando dados da janela 1...")
    df = visualizer.load_differences_data(window=1)
    
    if df is not None:
        print(f"   ✅ Dados carregados: {len(df)} métodos")
        print(f"   Colunas: {list(df.columns)}")
        
        # Testa cálculo de médias por grupo
        print("\n2. Calculando médias por grupo...")
        group_stats = visualizer.calculate_group_averages(df)
        
        print("\n📊 RESULTADOS:")
        print("-" * 60)
        
        for metric, stats in group_stats.items():
            print(f"\n🎯 {metric}:")
            print(f"   Métodos Simples  - Média: {stats['simple_avg']:.6f} ({stats['simple_count']} métodos)")
            print(f"   Métodos Híbridos - Média: {stats['hybrid_avg']:.6f} ({stats['hybrid_count']} métodos)")
            
            if not pd.isna(stats['improvement']):
                if stats['improvement'] > 0:
                    print(f"   ✅ MELHORIA: Híbridos são {abs(stats['improvement']):.6f} melhores")
                else:
                    print(f"   ❌ PIORA: Híbridos são {abs(stats['improvement']):.6f} piores")
            else:
                print(f"   ⚠️  Não foi possível calcular melhoria")
        
        print("\n" + "="*60)
        print("🎯 INTERPRETAÇÃO:")
        print("• Valores menores de diferença absoluta = MELHOR fairness")
        print("• Melhoria positiva = Métodos híbridos têm melhor fairness")
        print("• Melhoria negativa = Métodos simples têm melhor fairness")
        
        # Mostra detalhes dos métodos
        print(f"\n📋 DETALHES POR MÉTODO:")
        for _, row in df.iterrows():
            method_type = "Simples" if row['method'] in visualizer.simple_methods else "Híbrido"
            print(f"   {row['method']:15} ({method_type:8}): RMSE={row['RMSE_AbsDiff']:.4f}, "
                  f"MAE={row['MAE_AbsDiff']:.4f}, NDCG={row['NDCG_AbsDiff']:.4f}, F1={row['F1_AbsDiff']:.4f}")
        
        print(f"\n✅ Teste concluído com sucesso!")
        
    else:
        print("   ❌ Não foi possível carregar os dados")
        
except Exception as e:
    print(f"❌ Erro durante o teste: {e}")
    import traceback
    traceback.print_exc()
