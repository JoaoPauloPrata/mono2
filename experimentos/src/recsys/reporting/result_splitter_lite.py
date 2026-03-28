import os
from pathlib import Path

import pandas as pd


class FinalResultSplitterLite:
    """
    Versão enxuta do FinalResultSplitter.
    Cria um CSV por combinação (analysis_type, metric) contendo
    apenas: method, mean, ci_lower, ci_upper.
    """

    def __init__(
        self,
        input_path: str = "data/MetricsForMethods/final_results.csv",
        output_dir: str = "data/MetricsForMethods/final_results_split_lite",
    ) -> None:
        base_dir = Path(__file__).resolve().parents[3]
        self.input_path = base_dir / input_path
        self.output_dir = (base_dir / output_dir).resolve()

    @staticmethod
    def _collapse_methods(df: pd.DataFrame) -> pd.DataFrame:
        """
        Consolida múltiplas linhas do mesmo método tirando a média das estatísticas.
        """
        return (
            df.groupby("method", as_index=False)
            .agg(
                mean=("mean", "mean"),
                ci_lower=("ci_lower", "mean"),
                ci_upper=("ci_upper", "mean"),
            )
            .sort_values(by="mean", ascending=False)
        )

    def run(self) -> None:
        if not os.path.exists(self.input_path):
            raise FileNotFoundError(f"Arquivo não encontrado: {self.input_path}")

        df = pd.read_csv(self.input_path)
        required_cols = {"method", "metric", "analysis_type", "mean", "ci_lower", "ci_upper"}
        missing = required_cols.difference(df.columns)
        if missing:
            raise ValueError(f"Colunas faltantes em {self.input_path}: {missing}")

        os.makedirs(self.output_dir, exist_ok=True)

        for (analysis_type, metric), subset in df.groupby(["analysis_type", "metric"]):
            essential = subset[["method", "mean", "ci_lower", "ci_upper"]].copy()
            collapsed = self._collapse_methods(essential)
            filename = f"{analysis_type}_{metric}.csv"
            out_path = os.path.join(self.output_dir, filename)
            collapsed.to_csv(out_path, index=False)
            print(f"Gerado (lite): {out_path} ({len(collapsed)} linhas)")


if __name__ == "__main__":
    splitter = FinalResultSplitterLite()
    splitter.run()
