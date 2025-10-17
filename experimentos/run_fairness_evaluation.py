#!/usr/bin/env python3
"""
Script principal para avaliação de fairness por gênero
"""

from src.Metrics.Fairness import Fairness
from src.Metrics.FairnessDifferenceCalculator import FairnessDifferenceCalculator
import sys

def run_fairness_evaluation_for_window(window):
    """
    Executa avaliação de fairness para uma janela específica
    """
    print(f"🔍 Avaliando fairness para janela {window}")
    print("="*50)
    
    try:
        # Executa avaliação separada por gênero
        evaluator = Fairness()
        evaluator.evaluateAllMetricsForAllMethods(window)
        
        # Calcula diferenças absolutas
        calculator = FairnessDifferenceCalculator()
        differences = calculator.calculate_differences_for_window(window)
        
        if differences is not None:
            print(f"✅ Avaliação de fairness concluída para janela {window}")
            return True
        else:
            print(f"❌ Falha na avaliação de fairness para janela {window}")
            return False
            
    except Exception as e:
        print(f"❌ Erro durante avaliação da janela {window}: {e}")
        return False

def run_fairness_evaluation_all_windows(start=1, end=20):
    """
    Executa avaliação de fairness para todas as janelas
    """
    print(f"🔍 Avaliando fairness para janelas {start}-{end}")
    print("="*60)
    
    successful_windows = []
    failed_windows = []
    
    # Avalia cada janela individualmente
    for window in range(start, end + 1):
        print(f"\n📊 Processando janela {window}/{end}...")
        
        success = run_fairness_evaluation_for_window(window)
        
        if success:
            successful_windows.append(window)
        else:
            failed_windows.append(window)
    
    # Calcula diferenças consolidadas
    if successful_windows:
        print(f"\n🔄 Calculando diferenças consolidadas...")
        calculator = FairnessDifferenceCalculator()
        consolidated = calculator.calculate_differences_all_windows(start, end)
        
        if consolidated is not None:
            calculator.generate_fairness_summary()
    
    # Resumo final
    print(f"\n{'='*60}")
    print(f"RESUMO FINAL:")
    print(f"✅ Janelas processadas com sucesso: {len(successful_windows)}")
    if successful_windows:
        print(f"   Janelas: {successful_windows}")
    
    if failed_windows:
        print(f"❌ Janelas com falha: {len(failed_windows)}")
        print(f"   Janelas: {failed_windows}")
    
    print(f"📁 Resultados salvos em: data/MetricsForMethods/Fairness/")
    
    return len(successful_windows) > 0

def main():
    print("🎯 AVALIADOR DE FAIRNESS POR GÊNERO")
    print("="*60)
    print("Este script avalia fairness dividindo usuários por gênero (M/F)")
    print("e calculando diferenças absolutas entre as métricas dos grupos.")
    print()
    
    try:
        print("Opções disponíveis:")
        print("1. Avaliar janela específica")
        print("2. Avaliar todas as janelas (1-20)")
        print("3. Avaliar intervalo customizado")
        
        choice = input("\nEscolha uma opção (1-3): ").strip()
        
        if choice == "1":
            window = input("Digite o número da janela (1-20): ").strip()
            try:
                window_num = int(window)
                if 1 <= window_num <= 20:
                    success = run_fairness_evaluation_for_window(window_num)
                    return 0 if success else 1
                else:
                    print("❌ Número da janela deve estar entre 1 e 20.")
                    return 1
            except ValueError:
                print("❌ Por favor, digite um número válido.")
                return 1
                
        elif choice == "2":
            print("\n⚠️  Esta operação pode demorar vários minutos...")
            confirm = input("Deseja continuar? (s/n): ").strip().lower()
            if confirm in ['s', 'sim', 'y', 'yes']:
                success = run_fairness_evaluation_all_windows(1, 20)
                return 0 if success else 1
            else:
                print("❌ Operação cancelada.")
                return 1
                
        elif choice == "3":
            start = int(input("Janela inicial (1-20): ").strip())
            end = int(input("Janela final (1-20): ").strip())
            
            if 1 <= start <= end <= 20:
                success = run_fairness_evaluation_all_windows(start, end)
                return 0 if success else 1
            else:
                print("❌ Intervalo inválido.")
                return 1
        else:
            print("❌ Opção inválida.")
            return 1
            
    except KeyboardInterrupt:
        print("\n❌ Operação interrompida pelo usuário.")
        return 1
    except Exception as e:
        print(f"❌ Erro inesperado: {e}")
        return 1

if __name__ == "__main__":
    sys.exit(main())