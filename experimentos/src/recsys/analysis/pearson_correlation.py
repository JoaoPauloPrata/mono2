import os
import pandas as pd
import numpy as np
from scipy.stats import pearsonr


def load_predictions(window: int, exec_number: int):
    base_dir = "./data/filtered_predictions"
    methods = ["SVD", "BIASEDMF", "NMF", "StochasticItemKNN"]
    dfs = {}
    for m in methods:
        path = os.path.join(base_dir, f"window_{window}_{exec_number}_constituent_methods_{m}.tsv")
        if os.path.exists(path):
            df = pd.read_csv(path, sep="\t")[["user", "item", "prediction"]]
            df.rename(columns={"prediction": m}, inplace=True)
            dfs[m] = df
    return dfs


def compute_pearson_for_window(window: int, exec_number: int):
    dfs = load_predictions(window, exec_number)
    if len(dfs) < 2:
        print(f"Janela {window} exec {exec_number}: menos de 2 métodos disponíveis.")
        return None

    # Merge sequencial para alinhar pares user-item entre métodos
    merged = None
    for df in dfs.values():
        if merged is None:
            merged = df
        else:
            merged = pd.merge(merged, df, on=["user", "item"], how="inner")

    if merged is None or merged.shape[0] == 0:
        print(f"Janela {window} exec {exec_number}: nenhum par comum.")
        return None

    methods = list(dfs.keys())
    results = []
    for i in range(len(methods)):
        for j in range(i + 1, len(methods)):
            a, b = methods[i], methods[j]
            r, p = pearsonr(merged[a], merged[b])
            results.append(
                {"method_a": a, "method_b": b, "pearson_r": r, "p_value": p, "window": window, "exec": exec_number}
            )
    return results


def run_all():
    rows = []
    for exec_number in range(1, 6):
        for window in range(1, 21):
            res = compute_pearson_for_window(window, exec_number)
            if res:
                rows.extend(res)
    if not rows:
        print("Nenhuma correlação calculada.")
        return

    df = pd.DataFrame(rows)
    agg = (
        df.groupby(["method_a", "method_b"])
        .agg(pearson_r_mean=("pearson_r", "mean"), p_value_mean=("p_value", "mean"))
        .reset_index()
    )

    out_path = "./data/MetricsForMethods/pearson_correlations.csv"
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    agg.to_csv(out_path, index=False)
    print(f"Correlação de Pearson agregada salva em {out_path}")


if __name__ == "__main__":
    run_all()
