"""
Post-process constituent predictions before hybrid training.

Filters all prediction files to the common (user, item) pairs that received
valid predictions from every constituent method, then aligns the test splits
to match those pairs.

Run after: run_constituent_predictions.py
Run before: run_pipeline.py
"""

from recsys.data.post_processor import (
    filter_to_common_pairs_by_window_and_type,
    filter_test_data_to_match_predictions,
)


def main():
    print("=== Step 1: Filtering predictions to common user-item pairs ===")
    filter_to_common_pairs_by_window_and_type()

    print("\n=== Step 2: Aligning test splits to filtered pairs ===")
    filter_test_data_to_match_predictions()

    print("\nPost-processing complete.")


if __name__ == "__main__":
    main()
