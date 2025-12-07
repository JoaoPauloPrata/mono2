from pathlib import Path
import pandas as pd


HYBRID_METHODS = [
    "AdaBoost",
    "Bagging",
    "BayesianRidge",
    "GradientBoosting",
    "LinearSVR",
    "RandomForest",
    "Ridge",
    "Tweedie",
]


def padronize_hybrid(file_path: Path, template_df: pd.DataFrame) -> pd.DataFrame:
    """
    Garante que o arquivo de predições híbrido tenha colunas user, item, prediction.
    Usa o template (user/item) para reconstruir se o arquivo tiver apenas a coluna de scores.
    """
    hyb_df = pd.read_csv(file_path, sep="\t")

    # Já está formatado
    if {"user", "item", "prediction"}.issubset(hyb_df.columns):
        return hyb_df[["user", "item", "prediction"]]

    # Caso mais comum: apenas a coluna de scores
    scores = hyb_df.iloc[:, 0].values

    if len(scores) != len(template_df):
        raise ValueError(
            f"Tamanho diferente entre {file_path.name} ({len(scores)}) e template ({len(template_df)})"
        )

    return pd.DataFrame(
        {
            "user": template_df["user"].values,
            "item": template_df["item"].values,
            "prediction": scores,
        }
    )


def process_window(window: int, base_dir: Path):
    # Mesmo template usado no Evaluator: BIAS, ordenado por user
    template_path = base_dir.parent / "filtered_predictions" / f"window_{window}_constituent_methods_BIAS.tsv"
    template_df = (
        pd.read_csv(template_path, sep="\t")
        .sort_values("user")
        .reset_index(drop=True)[["user", "item"]]
    )

    for method in HYBRID_METHODS:
        hyb_path = base_dir / f"window_{window}_predicted{method}.tsv"
        if not hyb_path.exists():
            continue

        pred_df = padronize_hybrid(hyb_path, template_df)
        pred_df.to_csv(hyb_path, sep="\t", index=False)
        print(f"Atualizado: {hyb_path.name}")


if __name__ == "__main__":
    base_dir = Path(__file__).resolve().parent
    for window_count in range(1, 21):
        try:
            process_window(window_count, base_dir)
        except Exception as exc:
            print(f"[ERRO] window {window_count}: {exc}")
