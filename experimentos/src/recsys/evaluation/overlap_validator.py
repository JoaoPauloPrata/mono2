import os
import pandas as pd


def _load_window_pair(train_path: str, test_path: str):
    if not (os.path.exists(train_path) and os.path.exists(test_path)):
        return None, None
    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)
    return train_df, test_df


def _check_overlap(train_df: pd.DataFrame, test_df: pd.DataFrame, label: str):
    train_pairs = set(zip(train_df["user"], train_df["item"]))
    test_pairs = set(zip(test_df["user"], test_df["item"]))
    overlap = train_pairs.intersection(test_pairs)
    print(f"{label}: overlap de {len(overlap)} pares.")
    return overlap


def validate_all_windows():
    base = "./data/windows"

    leak_reports = []

    for window in range(1, 21):
        # regression train data
        train_reg = os.path.join(base, f"train_to_get_regression_train_data_{window}.csv")
        test_reg = os.path.join(base, f"test_to_get_regression_train_data_{window}.csv")
        tr_df, te_df = _load_window_pair(train_reg, test_reg)
        if tr_df is not None:
            overlap = _check_overlap(tr_df, te_df, f"regression window {window}")
            if overlap:
                leak_reports.append(("regression", window, len(overlap)))

        # constituent methods
        train_const = os.path.join(base, f"train_to_get_constituent_methods_{window}.csv")
        test_const = os.path.join(base, f"test_to_get_constituent_methods_{window}.csv")
        tc_df, ts_df = _load_window_pair(train_const, test_const)
        if tc_df is not None:
            overlap2 = _check_overlap(tc_df, ts_df, f"constituent window {window}")
            if overlap2:
                leak_reports.append(("constituent", window, len(overlap2)))

    if not leak_reports:
        print("Nenhum vazamento encontrado.")
    else:
        print("Vazamentos detectados:")
        for kind, window, count in leak_reports:
            print(f"- {kind} window {window}: {count} pares em comum")


if __name__ == "__main__":
    validate_all_windows()
