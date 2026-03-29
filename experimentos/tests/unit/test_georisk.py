import numpy as np

from recsys.evaluation.quality_metrics import Evaluator


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


def test_georisk_consistent_system_wins():
    """The consistent system (small variance) should have higher GeoRisk than unstable one."""
    alpha = 0.5

    mat_consistent = np.array([
        [0.90, 0.70, 0.60],
        [0.88, 0.72, 0.58],
        [0.91, 0.69, 0.61],
        [0.87, 0.71, 0.59],
        [0.89, 0.70, 0.60],
    ], dtype=float)

    mat_unstable = np.array([
        [0.95, 0.70, 0.60],
        [0.20, 0.72, 0.58],
        [0.92, 0.69, 0.61],
        [0.15, 0.71, 0.59],
        [0.94, 0.70, 0.60],
    ], dtype=float)

    grisk_consistent = Evaluator.getGeoRisk(mat_consistent, alpha)
    grisk_unstable = Evaluator.getGeoRisk(mat_unstable, alpha)

    # System 0 in consistent matrix should beat system 0 in unstable matrix
    assert grisk_consistent[0] > grisk_unstable[0]


if __name__ == "__main__":
    alpha = 0.5
    mat_consistent = np.array([
        [0.90, 0.70, 0.60],
        [0.88, 0.72, 0.58],
        [0.91, 0.69, 0.61],
        [0.87, 0.71, 0.59],
        [0.89, 0.70, 0.60],
    ], dtype=float)
    mat_unstable = np.array([
        [0.95, 0.70, 0.60],
        [0.20, 0.72, 0.58],
        [0.92, 0.69, 0.61],
        [0.15, 0.71, 0.59],
        [0.94, 0.70, 0.60],
    ], dtype=float)
    run_case("CASO BOM (consistente)", mat_consistent, alpha)
    run_case("CASO RUIM (instavel)", mat_unstable, alpha)
