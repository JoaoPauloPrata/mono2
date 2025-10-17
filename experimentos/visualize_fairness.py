#!/usr/bin/env python3
"""
Script para visualizar análises de fairness
"""

from src.Metrics.FairnessVisualization import FairnessVisualization
import sys

def main():
    print("🎨 VISUALIZADOR DE FAIRNESS - ANÁLISE DE DIFERENÇAS ABSOLUTAS")
    print("="*70)
    print("Este script gera gráficos e relatórios comparando fairness entre:")
    print("• Métodos Simples: itemKNN, BIAS, userKNN, SVD, BIASEDMF")
    print("• Métodos Híbridos: BayesianRidge, Tweedie, Ridge, RandomForest, etc.")
    print()
    
    visualizer = FairnessVisualization()
    
    try:
        print("Opções disponíveis:")
        print("1. Analisar janela específica")
        print("2. Analisar todas as janelas (consolidado)")
        print("3. Analisar janela 1 (rápido)")
        
        choice = input("\nEscolha uma opção (1-3): ").strip()
        
        if choice == "1":
            window = input("Digite o número da janela: ").strip()
            try:
                window_num = int(window)
                print(f"\n🔍 Analisando janela {window_num}...")
                stats = visualizer.analyze_window(window_num)
                
                if stats:
                    print(f"\n✅ Análise concluída para janela {window_num}")
                    print("📁 Gráficos salvos em: data/MetricsForMethods/Fairness/plots/")
                    print("📄 Relatório salvo em: data/MetricsForMethods/Fairness/")
                else:
                    print(f"❌ Não foi possível analisar janela {window_num}")
                    
            except ValueError:
                print("❌ Por favor, digite um número válido.")
                return 1
                
        elif choice == "2":
            print("\n🔍 Analisando todas as janelas (consolidado)...")
            print("⚠️  Esta operação pode demorar alguns minutos...")
            
            stats = visualizer.analyze_all_windows()
            
            if stats:
                print("\n✅ Análise consolidada concluída!")
                print("📁 Gráficos salvos em: data/MetricsForMethods/Fairness/plots/")
                print("📄 Relatório salvo em: data/MetricsForMethods/Fairness/")
            else:
                print("❌ Não foi possível realizar análise consolidada")
                
        elif choice == "3":
            print("\n🔍 Analisando janela 1 (demonstração rápida)...")
            stats = visualizer.analyze_window(1)
            
            if stats:
                print("\n✅ Análise da janela 1 concluída!")
                print("📁 Gráficos salvos em: data/MetricsForMethods/Fairness/plots/")
                print("📄 Relatório salvo em: data/MetricsForMethods/Fairness/")
                
                # Mostra resumo rápido
                print("\n📊 RESUMO RÁPIDO:")
                for metric, data in stats.items():
                    if not any(pd.isna(val) for val in [data['simple_avg'], data['hybrid_avg']]):
                        improvement = data['improvement']
                        status = "✅ Híbridos melhores" if improvement > 0 else "❌ Simples melhores"
                        print(f"   {metric}: {status} (diferença: {abs(improvement):.4f})")
            else:
                print("❌ Não foi possível analisar janela 1")
        else:
            print("❌ Opção inválida.")
            return 1
        
    except KeyboardInterrupt:
        print("\n❌ Operação interrompida pelo usuário.")
        return 1
    except Exception as e:
        print(f"❌ Erro inesperado: {e}")
        return 1
    
    return 0

if __name__ == "__main__":
    import pandas as pd
    sys.exit(main())
