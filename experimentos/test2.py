# test_georisk.py
import numpy as np

# Ajuste o import conforme o nome do seu arquivo onde a classe Evaluator está.
# Exemplo: se você salvou a classe em "evaluator.py", isso funciona:
from src.Metrics.Evaluator import Evaluator


def run_case(name: str, mat: np.ndarray, alpha: float):
    print(f"\n=== {name} ===")
    print("Matriz (queries x sistemas):")
    print(mat)
    grisk = Evaluator.getGeoRisk(mat, alpha)
    print(f"\nalpha = {alpha}")
    print("GeoRisk por sistema (coluna):")
    for i, val in enumerate(grisk):
        print(f"  Sistema {i}: {val:.6f}")
    print(f"Melhor sistema (maior GeoRisk): {int(np.argmax(grisk))}")


def main():
    alpha = 0.5  # teste com 0.5; depois mude para 1.0 e compare

    # 5 queries x 3 sistemas
    mat_boa = np.array([
        [0.90, 0.70, 0.60],
        [0.88, 0.72, 0.58],
        [0.91, 0.69, 0.61],
        [0.87, 0.71, 0.59],
        [0.89, 0.70, 0.60],
    ], dtype=float)

    mat_ruim = np.array([
        [0.95, 0.70, 0.60],
        [0.20, 0.72, 0.58],
        [0.92, 0.69, 0.61],
        [0.15, 0.71, 0.59],
        [0.94, 0.70, 0.60],
    ], dtype=float)

    run_case("CASO BOM (consistente)", mat_boa, alpha)
    run_case("CASO RUIM (instavel)", mat_ruim, alpha)


if __name__ == "__main__":
    main()
