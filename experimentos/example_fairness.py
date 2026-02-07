"""
Exemplo de uso da classe FairnessEvaluator
"""

from src.Metrics.FairnessEvaluator import FairnessEvaluator

# Criar avaliador de fairness
print("Criando avaliador de fairness...")
fairness_evaluator = FairnessEvaluator()

# Exemplo 1: Avaliar fairness para um método específico
print("\n" + "="*60)
print("EXEMPLO 1: Avaliação para um método específico")
print("="*60)

# Testa com o método BIAS da janela 1
predictions_file = "data/filtered_predictions/window_1_constituent_methods_BIASEDMF.tsv"
truth_file = "data/windows/test_to_get_constituent_methods_1.csv"

result = fairness_evaluator.evaluate_fairness_for_method(
    predictions_file, 
    truth_file, 
    "BIAS_Test"
)

if result:
    print("\nResultados obtidos:")
    for key, value in result.items():
        if isinstance(value, float):
            print(f"{key}: {value:.4f}")
        else:
            print(f"{key}: {value}")

# Exemplo 2: Avaliar fairness para uma janela completa
print("\n" + "="*60)
print("EXEMPLO 2: Avaliação para janela completa")
print("="*60)

print("Avaliando fairness para janela 1...")
window_results = fairness_evaluator.evaluate_fairness_for_window(1)

print(f"\nTotal de métodos avaliados: {len(window_results) if window_results else 0}")

print("\nExemplo concluído! ✅")
