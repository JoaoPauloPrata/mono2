from pathlib import Path
from typing import Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


class ChartGenerator:
    """Gera graficos de barras com intervalo de confianca para cada CSV gerado pelo FinalResultSplitter."""

    def __init__(
        self,
        split_dir: str = "data/MetricsForMethods/final_results_split",
        output_dir: str = "data/charts",
    ) -> None:
        base_dir = Path(__file__).resolve().parents[3]
        self.split_dir = base_dir / split_dir
        self.output_dir = base_dir / output_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)

    @staticmethod
    def _prepare_dataframe(df: pd.DataFrame) -> pd.DataFrame:
        required_cols = {"method", "mean", "ci_lower", "ci_upper"}
        missing = required_cols.difference(df.columns)
        if missing:
            raise ValueError(f"Colunas faltantes no CSV: {missing}")

        # Consolida linhas repetidas do mesmo algoritmo
        aggregated = (
            df.groupby("method", as_index=False)
            .agg(
                mean=("mean", "mean"),
                ci_lower=("ci_lower", "mean"),
                ci_upper=("ci_upper", "mean"),
            )
            .sort_values(by="mean", ascending=False)
        )

        # Substitui NaN por media para evitar barras sem IC
        aggregated[["ci_lower", "ci_upper"]] = aggregated[
            ["ci_lower", "ci_upper"]
        ].fillna(aggregated["mean"])

        # Garante limites coerentes para o yerr
        aggregated["ci_lower"] = np.minimum(aggregated["ci_lower"], aggregated["mean"])
        aggregated["ci_upper"] = np.maximum(aggregated["ci_upper"], aggregated["mean"])
        return aggregated

    @staticmethod
    def _title_and_ylabel(df: pd.DataFrame, csv_path: Path) -> Tuple[str, str, str]:
        metric = df["metric"].iloc[0] if "metric" in df.columns else csv_path.stem
        analysis = df["analysis_type"].iloc[0] if "analysis_type" in df.columns else ""
        title = f"{analysis} - {metric}" if analysis else csv_path.stem
        ylabel = f"Media ({metric})" if metric else "Media"
        return title, ylabel, analysis

    def _plot(
        self, df: pd.DataFrame, title: str, ylabel: str, analysis: str, out_file: Path
    ) -> None:
        methods = df["method"].to_list()
        means = df["mean"].to_numpy()
        err_lower = (df["mean"] - df["ci_lower"]).clip(lower=0).to_numpy()
        err_upper = (df["ci_upper"] - df["mean"]).clip(lower=0).to_numpy()
        yerr = np.vstack([err_lower, err_upper])

        x = np.arange(len(methods))
        fig, ax = plt.subplots(figsize=(10, 6))
        bars = ax.bar(x, means, yerr=yerr, capsize=6, color="#4c72b0")

        # Remove a borda superior para evitar sobreposicao visual
        ax.spines["top"].set_visible(False)

        ax.set_title(title)
        ax.set_ylabel(ylabel)
        ax.set_xlabel("Algoritmo")
        ax.set_xticks(x)
        ax.set_xticklabels(methods, rotation=45, ha="right")
        ax.grid(axis="y", linestyle="--", alpha=0.3)

        # Valor numerico acima das barras para leitura rapida
        range_y = float(df["ci_upper"].max() - df["ci_lower"].min())
        pad_label = max(range_y * 0.05, float(df["mean"].max()) * 0.005, 0.0005)
        for bar, mean, err_up in zip(bars, means, err_upper):
            height = bar.get_height()
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                height + err_up + pad_label,
                f"{mean:.5f}",
                ha="center",
                va="bottom",
                fontsize=8,
            )

        # Formata escala do eixo Y com 5 casas decimais
        ax.yaxis.set_major_formatter(plt.FormatStrFormatter("%.5f"))

        # Ajusta limites para garantir que rótulos e ICs nao encostem no topo
        y_min_raw = float(df["ci_lower"].min())
        y_max_raw = float(df["ci_upper"].max())
        span = max(y_max_raw - y_min_raw, 1e-9)
        padding = max(span * 0.12, 0.003 * abs(y_max_raw) + 0.0005)

        y_top_needed = (means + err_upper + pad_label * 2).max()
        y_top = max(y_max_raw + padding, y_top_needed)
        y_bottom = min(y_min_raw - padding * 0.5, df["mean"].min() - err_lower.max())

        # Evita limites negativos para metricas sempre positivas
        if y_bottom < 0 and y_min_raw >= 0:
            y_bottom = 0

        ax.set_ylim(y_bottom, y_top)

        fig.tight_layout()
        fig.savefig(out_file, dpi=300)
        plt.close(fig)

    def generate_all(self) -> None:
        csv_files = sorted(self.split_dir.glob("*.csv"))
        if not csv_files:
            raise FileNotFoundError(f"Nenhum CSV encontrado em {self.split_dir}")

        for csv_path in csv_files:
            df_raw = pd.read_csv(csv_path)
            prepared = self._prepare_dataframe(df_raw)
            title, ylabel, analysis = self._title_and_ylabel(df_raw, csv_path)
            out_file = self.output_dir / f"{csv_path.stem}.png"
            self._plot(prepared, title, ylabel, analysis, out_file)
            print(f"Grafico salvo em: {out_file}")


if __name__ == "__main__":
    generator = ChartGenerator()
    generator.generate_all()
