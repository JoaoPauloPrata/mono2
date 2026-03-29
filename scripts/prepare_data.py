"""Split MovieLens 1M into sliding-window train/test CSV files."""

import os
import sys

sys.path.insert(0, os.path.dirname(__file__))

from run_pipeline import split_data, split_full_windows


def main():
    print("Splitting windows for constituent methods and hybrid training...")
    split_data()
    print("Splitting full windows...")
    split_full_windows()
    print("Done.")


if __name__ == "__main__":
    main()
