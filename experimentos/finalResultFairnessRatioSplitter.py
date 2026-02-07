import os
import pandas as pd


class FairnessRatioSplitter:
    """
    Divide o arquivo fairness_ratio_results.csv em um CSV por métrica/tipo de análise,
    atribuindo grupos por sobreposição de intervalos de confiança (IC).
    """

    def __init__(
        self,
        input_path: str = "./data/MetricsForMethods/fairness_ratio_results.csv",
        output_dir: str = "./data/MetricsForMethods/fairness_ratio_results_split",
    ) -> None:
        self.input_path = input_path
        self.output_dir = output_dir.rstrip("/\\")

    @staticmethod
    def _overlaps(ci1, ci2) -> bool:
        l1, u1 = ci1
        l2, u2 = ci2
        if any(pd.isna([l1, u1, l2, u2])):
            return False
        return (l1 <= u2) and (l2 <= u1)

    def _assign_groups(self, df: pd.DataFrame) -> pd.DataFrame:
        if df.empty:
            return df
        df = df.sort_values(by="mean", ascending=False).reset_index(drop=True)
        groups = []
        current_group = 1
        current_cis = []
        for _, row in df.iterrows():
            ci = (row.get("ci_lower"), row.get("ci_upper"))
            if current_cis and any(self._overlaps(ci, gci) for gci in current_cis):
                groups.append(current_group)
                current_cis.append(ci)
            else:
                groups.append(current_group if not current_cis else current_group + 1)
                if current_cis:
                    current_group += 1
                current_cis = [ci]
        df["group"] = groups
        return df

    def run(self) -> None:
        if not os.path.exists(self.input_path):
            raise FileNotFoundError(f"Arquivo não encontrado: {self.input_path}")

        df = pd.read_csv(self.input_path)
        required_cols = {"method", "metric", "analysis_type"}
        missing = required_cols.difference(df.columns)
        if missing:
            raise ValueError(f"Colunas faltantes em {self.input_path}: {missing}")

        os.makedirs(self.output_dir, exist_ok=True)

        for (analysis_type, metric), subset in df.groupby(["analysis_type", "metric"]):
            subset_with_groups = self._assign_groups(subset.copy())
            filename = f"{analysis_type}_{metric}.csv"
            out_path = os.path.join(self.output_dir, filename)
            subset_with_groups.to_csv(out_path, index=False)
            print(f"Gerado: {out_path} ({len(subset_with_groups)} linhas)")


if __name__ == "__main__":
    splitter = FairnessRatioSplitter()
    splitter.run()
