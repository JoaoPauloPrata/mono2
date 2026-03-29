"""Generate constituent method predictions for all windows and executions."""

import os
import sys

sys.path.insert(0, os.path.dirname(__file__))

from run_pipeline import load_data_and_run


def main():
    for exec_number in range(1, 6):
        print(f"Execution {exec_number}/5")
        for window_number in range(1, 21):
            print(f"  Window {window_number}/20")
            load_data_and_run(window_number, exec_number)
    print("Done.")


if __name__ == "__main__":
    main()
