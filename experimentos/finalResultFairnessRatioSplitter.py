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

    def _assign_groups(self, df: pd.DataFrame, ascending: bool) -> pd.DataFrame:
        """
        Agrupa por sobreposição direta (não transitiva). Métodos podem pertencer a múltiplos grupos.
        """
        if df.empty:
            return df

        df = df.sort_values(by="mean", ascending=ascending).reset_index(drop=True)
        grouped_rows = []
        group_id = 1
        grouped_rows.append({**df.iloc[0].to_dict(), "group": group_id})

        for i in range(1, len(df)):
            prev = df.iloc[i - 1]
            curr = df.iloc[i]
            ci_prev = (prev.get("ci_lower"), prev.get("ci_upper"))
            ci_curr = (curr.get("ci_lower"), curr.get("ci_upper"))

            if self._overlaps(ci_prev, ci_curr):
                group_id += 1
                grouped_rows.append({**prev.to_dict(), "group": group_id})
                grouped_rows.append({**curr.to_dict(), "group": group_id})
            else:
                group_id += 1
                grouped_rows.append({**curr.to_dict(), "group": group_id})

        return pd.DataFrame(grouped_rows)

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
            is_fairness = analysis_type.startswith("fairness")
            subset_with_groups = self._assign_groups(subset.copy(), ascending=is_fairness)
            filename = f"{analysis_type}_{metric}.csv"
            out_path = os.path.join(self.output_dir, filename)
            subset_with_groups.to_csv(out_path, index=False)
            print(f"Gerado: {out_path} ({len(subset_with_groups)} linhas)")


if __name__ == "__main__":
    splitter = FairnessRatioSplitter()
    splitter.run()
