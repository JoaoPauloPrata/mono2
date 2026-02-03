import os
import pandas as pd


class LatexTableGenerator:
    """
    Gera tabelas LaTeX a partir dos arquivos produzidos pelo finalResultSplitter.py.
    Para cada CSV em final_results_split, salva um .txt contendo o ambiente tabular completo.
    """

    def __init__(
        self,
        input_dir: str = "./data/MetricsForMethods/final_results_split",
        output_dir: str = "./data/MetricsForMethods/latex_tables",
    ) -> None:
        self.input_dir = input_dir.rstrip("/\\")
        self.output_dir = output_dir.rstrip("/\\")
        os.makedirs(self.output_dir, exist_ok=True)

        self.analysis_labels = {
            "fairness_activity": "fairness por atividade",
            "fairness_gender": "fairness por gênero",
            "georisk": "GeoRisk",
        }

        self.metric_labels = {
            "f1": "F1",
            "ndcg": "NDCG",
            "rmse": "RMSE",
            "mae": "MAE",
        }

    @staticmethod
    def _fmt(val) -> str:
        return "" if pd.isna(val) else f"{float(val):.5f}"

    def _caption_and_label(self, analysis: str, metric: str):
        analysis_label = self.analysis_labels.get(analysis, analysis)
        metric_label = self.metric_labels.get(metric, metric.upper())
        caption = f"Resultados de {analysis_label} ({metric_label}) ao longo das janelas temporais."
        label = f"tab:{analysis}-{metric}"
        return caption, label

    def _build_table(self, df: pd.DataFrame, analysis: str, metric: str) -> str:
        # Ordena por grupo e média decrescente para legibilidade
        df = df.sort_values(by=["group", "mean"], ascending=[True, False])

        caption, label = self._caption_and_label(analysis, metric)

        header = (
            "\\begin{table}[H]\n"
            "\\centering\n"
            f"\\caption{{{caption}}}\n"
            f"\\label{{{label}}}\n\n"
            "\\resizebox{\\textwidth}{!}{%\n"
            "\\begin{tabular}{lcccccccc}\n"
            "\\toprule\n"
            "\\textbf{Método} & \\textbf{Média} & \\textbf{DP} & \\textbf{Mediana} & "
            "\\textbf{Mín.} & \\textbf{Máx.} & \\textbf{IC 95\\% (Inf.)} & \\textbf{IC 95\\% (Sup.)} & \\textbf{Grupo} \\\\\n"
            "\\midrule\n"
        )

        lines = []
        for _, row in df.iterrows():
            line = " & ".join(
                [
                    str(row["method"]),
                    self._fmt(row["mean"]),
                    self._fmt(row["std"]),
                    self._fmt(row["median"]),
                    self._fmt(row["min"]),
                    self._fmt(row["max"]),
                    self._fmt(row.get("ci_lower")),
                    self._fmt(row.get("ci_upper")),
                    str(int(row["group"])) if not pd.isna(row["group"]) else "",
                ]
            )
            lines.append(line + " \\\\")

        body = "\n".join(lines)

        footer = (
            "\n\\bottomrule\n"
            "\\end{tabular}\n"
            "}\n"
            "\\end{table}\n"
        )

        return header + body + footer

    def run(self) -> None:
        files = [f for f in os.listdir(self.input_dir) if f.endswith(".csv")]
        if not files:
            print(f"Nenhum CSV encontrado em {self.input_dir}")
            return

        for filename in files:
            path = os.path.join(self.input_dir, filename)
            df = pd.read_csv(path)

            # Deriva analysis e metric do nome do arquivo: ex fairness_activity_f1.csv
            name_without_ext = os.path.splitext(filename)[0]
            parts = name_without_ext.split("_")
            if len(parts) < 2:
                print(f"Ignorando arquivo (nome inesperado): {filename}")
                continue
            analysis = "_".join(parts[:-1])
            metric = parts[-1]

            table_tex = self._build_table(df, analysis, metric)

            out_path = os.path.join(self.output_dir, f"{name_without_ext}.txt")
            with open(out_path, "w", encoding="utf-8") as f:
                f.write(table_tex)

            print(f"Tabela gerada: {out_path}")


if __name__ == "__main__":
    generator = LatexTableGenerator()
    generator.run()
