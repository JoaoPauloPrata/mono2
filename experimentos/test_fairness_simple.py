"""
Teste simples da avaliação de fairness
"""

from src.Metrics.Fairness import Fairness
from src.Metrics.FairnessDifferenceCalculator import FairnessDifferenceCalculator

print("🧪 Teste da Avaliação de Fairness")
print("="*40)

# Testa avaliação para janela 1
evaluator = Fairness()
calculator = FairnessDifferenceCalculator()

print("Executando avaliação de fairness para janela 1...")
try:
    for i in range(1, 21):
        evaluator.evaluateAllMetricsForAllMethods(i)
        calculator.calculate_differences_for_window(i)
    print("✅ Teste concluído!")
except Exception as e:
    print(f"❌ Erro no teste: {e}")
