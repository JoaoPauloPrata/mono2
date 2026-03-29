import os
import pandas as pd


class FinalResultSplitter:
    """
    Lê o arquivo final_results.csv (gerado por finalResult.py) e cria
    arquivos separados por métrica e tipo de análise.

    Exemplos de saídas:
      - georisk_f1.csv
      - fairness_activity_ndcg.csv
      - fairness_gender_mae.csv
    """

    def __init__(
        self,
        input_path: str = "./data/MetricsForMethods/final_results.csv",
        output_dir: str = "./data/MetricsForMethods/final_results_split",
    ) -> None:
        self.input_path = input_path
        self.output_dir = output_dir.rstrip("/\\")

    @staticmethod
    def _overlaps(ci1, ci2) -> bool:
        """Retorna True se dois ICs [l,u] se sobrepõem."""
        l1, u1 = ci1
        l2, u2 = ci2
        if any(pd.isna([l1, u1, l2, u2])):
            return False
        return (l1 <= u2) and (l2 <= u1)

    def _assign_groups(self, df: pd.DataFrame, ascending: bool) -> pd.DataFrame:
        """
        Atribui grupos por sobreposição de IC sem transitividade.
        Cada par de vizinhos que empata gera um novo grupo contendo ambos.
        Um mesmo método pode aparecer em mais de um grupo.
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
                # cria novo grupo para o par empatado
                group_id += 1
                grouped_rows.append({**prev.to_dict(), "group": group_id})
                grouped_rows.append({**curr.to_dict(), "group": group_id})
            else:
                # próximo grupo (pior resultado)
                group_id += 1
                grouped_rows.append({**curr.to_dict(), "group": group_id})

        # garante que menor group_id corresponde ao melhor (já ordenado por mean desc)
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

        # Agrupa por análise e métrica, salvando um CSV para cada combinação
        for (analysis_type, metric), subset in df.groupby(["analysis_type", "metric"]):
            is_fairness = analysis_type.startswith("fairness")
            subset_with_groups = self._assign_groups(subset.copy(), ascending=is_fairness)
            filename = f"{analysis_type}_{metric}.csv"
            out_path = os.path.join(self.output_dir, filename)
            subset_with_groups.to_csv(out_path, index=False)
            print(f"Gerado: {out_path} ({len(subset_with_groups)} linhas)")


if __name__ == "__main__":
    splitter = FinalResultSplitter()
    splitter.run()
