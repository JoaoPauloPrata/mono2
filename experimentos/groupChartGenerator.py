from pathlib import Path
from typing import List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


class GroupChartGenerator:
    """
    Gera graficos de barras comparando os grupos (ex.: high vs low, male vs female)
    para cada algoritmo e metrica, a partir dos CSVs produzidos por groupsQualityResults.py.
    Cada algoritmo aparece no eixo X com barras lado a lado para cada grupo.
    """

    def __init__(
        self,
        split_dir: str = "data/MetricsForMethods/fairness_group_means_split",
        output_dir: str = "data/charts_groups",
    ) -> None:
        base_dir = Path(__file__).resolve().parent
        self.split_dir = base_dir / split_dir
        self.output_dir = base_dir / output_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)

    @staticmethod
    def _prepare_dataframe(df: pd.DataFrame) -> Tuple[pd.DataFrame, List[str]]:
        required_cols = {"method", "group", "mean", "ci_lower", "ci_upper"}
        missing = required_cols.difference(df.columns)
        if missing:
            raise ValueError(f"Colunas faltantes no CSV: {missing}")

        # Consolidar possiveis duplicatas (media dos valores e dos ICs)
        aggregated = (
            df.groupby(["method", "group"], as_index=False)
            .agg(
                mean=("mean", "mean"),
                ci_lower=("ci_lower", "mean"),
                ci_upper=("ci_upper", "mean"),
            )
        )

        # Substituir NaN de IC pela media (barra sem erro)
        aggregated[["ci_lower", "ci_upper"]] = aggregated[
            ["ci_lower", "ci_upper"]
        ].fillna(aggregated["mean"])

        groups = list(aggregated["group"].unique())
        return aggregated, groups

    @staticmethod
    def _title_and_ylabel(df: pd.DataFrame, csv_path: Path) -> Tuple[str, str, str]:
        metric = df["metric"].iloc[0] if "metric" in df.columns else csv_path.stem
        analysis = df["analysis_type"].iloc[0] if "analysis_type" in df.columns else ""

        if "gender" in analysis.lower():
            title = f"Gender results - {metric}"
        elif "activity" in analysis.lower():
            title = f"Activity results - {metric}"
        else:
            title = f"{analysis} - {metric}" if analysis else csv_path.stem

        ylabel = f"Media ({metric})" if metric else "Media"
        return title, ylabel, analysis

    def _plot_grouped(
        self,
        df: pd.DataFrame,
        groups: List[str],
        title: str,
        ylabel: str,
        analysis: str,
        out_file: Path,
    ) -> None:
        # Ordena metodos pelo valor medio (considerando a media das medias dos grupos)
        order = (
            df.groupby("method")["mean"]
            .mean()
            .sort_values(ascending=False)
            .index.tolist()
        )
        df = df.set_index(["method", "group"]).loc[order]

        x = np.arange(len(order))
        width = 0.8 / max(len(groups), 1)

        fig, ax = plt.subplots(figsize=(11, 6))

        for idx, group in enumerate(groups):
            subset = df[df.index.get_level_values("group") == group]
            subset = subset.droplevel("group").reindex(order)

            means = subset["mean"].to_numpy()
            ci_lower = subset["ci_lower"].to_numpy()
            ci_upper = subset["ci_upper"].to_numpy()

            err_lower = np.clip(means - ci_lower, a_min=0, a_max=None)
            err_upper = np.clip(ci_upper - means, a_min=0, a_max=None)
            positions = x + (idx - (len(groups) - 1) / 2) * width

            bars = ax.bar(
                positions,
                means,
                width,
                label=group,
                yerr=np.vstack([err_lower, err_upper]),
                capsize=5,
            )

            # valores acima das barras
            pad = max((err_upper.max() if len(err_upper) else 0) * 0.5, 0.0005)
            for bar, mean in zip(bars, means):
                ax.text(
                    bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + pad,
                    f"{mean:.4f}",
                    ha="center",
                    va="bottom",
                    fontsize=8,
                )

        ax.set_title(title)
        ax.set_ylabel(ylabel)
        ax.set_xlabel("Algoritmo")
        ax.set_xticks(x)
        ax.set_xticklabels(order, rotation=45, ha="right")
        ax.grid(axis="y", linestyle="--", alpha=0.3)
        ax.legend(title="Grupo")

        # Ajusta limites para destacar intervalos de confiança
        y_min_raw = float(df["ci_lower"].min())
        y_max_raw = float(df["ci_upper"].max())
        span = y_max_raw - y_min_raw
        padding = max(span * 0.15, 0.001)
        ax.set_ylim(y_min_raw - padding, y_max_raw + padding)

        fig.tight_layout()
        fig.savefig(out_file, dpi=300)
        plt.close(fig)

    def generate_all(self) -> None:
        csv_files = sorted(self.split_dir.glob("*.csv"))
        if not csv_files:
            raise FileNotFoundError(f"Nenhum CSV encontrado em {self.split_dir}")

        for csv_path in csv_files:
            df_raw = pd.read_csv(csv_path)
            prepared, groups = self._prepare_dataframe(df_raw)
            title, ylabel, analysis = self._title_and_ylabel(df_raw, csv_path)
            out_file = self.output_dir / f"{csv_path.stem}.png"
            self._plot_grouped(prepared, groups, title, ylabel, analysis, out_file)
            print(f"Grafico salvo em: {out_file}")


if __name__ == "__main__":
    generator = GroupChartGenerator()
    generator.generate_all()
