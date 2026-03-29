"""
Script to run fairness evaluation (kmeans activity groups + gender groups)
across all windows and executions.
"""

from recsys.evaluation.fairness_metrics import kmeansGroupCalculator, genderGroupCalculator


def main():
    print("Running KMeans activity group fairness evaluation...")
    kmeansGroupCalculator()
    print("Running gender group fairness evaluation...")
    genderGroupCalculator()
    print("Fairness evaluation complete.")


if __name__ == "__main__":
    main()
