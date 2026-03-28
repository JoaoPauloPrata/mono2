import os
import pandas as pd
from scipy import stats
import json
import numpy as np


class AnovaAnalysis:
    """
    Realiza testes ANOVA (Analysis of Variance) nos resultados de métodos
    coletados de múltiplas janelas e execuções, para descobrir se existem 
    diferenças estatísticas reais entre os métodos.
    
    Coleta dados dos mesmos caminhos que finalResult.py.
    
    Salva os resultados em:
      - anova_results.csv (resumo geral com significância estatística)
      - anova_{analysis_type}_{metric}.csv (detalhes das amostras por método)
    """

    def __init__(
        self,
        georisk_path: str = "./data/MetricsForMethods/GeoRisk",
        fairness_gender_path: str = "./data/MetricsForMethods/Fairness/gender",
        fairness_activity_path: str = "./data/MetricsForMethods/Fairness/kmeans",
        output_dir: str = "./data/MetricsForMethods/anova_results",
        executions: int = 5,
    ) -> None:
        self.georisk_path = georisk_path.rstrip("/\\")
        self.fairness_gender_path = fairness_gender_path.rstrip("/\\")
        self.fairness_activity_path = fairness_activity_path.rstrip("/\\")
        self.output_dir = output_dir.rstrip("/\\")
        self.executions = executions
        
        self.metrics = ["rmse", "f1", "ndcg", "mae"]
        self.algorithms = [
            "StochasticItemKNN",
            "NMF",
            "SVD",
            "BIASEDMF",
            "BayesianRidge",
            "Tweedie",
            "Ridge",
            "RandomForest",
            "Bagging",
            "AdaBoost",
            "GradientBoosting",
            "LinearSVR",
        ]
        self.window_range = range(1, 21)
        self.results = []

    @staticmethod
    def _init_result_dict(metrics):
        return {m: [] for m in metrics}

    def _gather_georisk(self):
        """Coleta dados de GeoRisk de todas as janelas e execuções."""
        acc = {algo: self._init_result_dict(self.metrics) for algo in self.algorithms}
        for window in self.window_range:
            for exec_number in range(1, self.executions + 1):
                for metric in self.metrics:
                    path = f"{self.georisk_path}/window_{window}/execution_{exec_number}/{metric}.csv"
                    if not os.path.exists(path):
                        continue
                    try:
                        df = pd.read_csv(path)
                        if "algorithm" not in df.columns or "georisk" not in df.columns:
                            continue
                        for _, row in df.iterrows():
                            algo = row["algorithm"]
                            if algo not in acc:
                                acc[algo] = self._init_result_dict(self.metrics)
                            val = row["georisk"]
                            if pd.notna(val):
                                acc[algo][metric].append(float(val))
                    except Exception as e:
                        print(f"Erro ao ler {path}: {e}")
                        continue
        return acc

    def _gather_fairness(self, base_path):
        """Coleta dados de fairness de todas as janelas e execuções."""
        acc = {algo: self._init_result_dict(self.metrics) for algo in self.algorithms}
        metric_map = {"rmse": "RMSE", "ndcg": "NDCG", "f1": "F1", "mae": "MAE"}
        for window in self.window_range:
            for exec_number in range(1, self.executions + 1):
                for group in ("constituent", "hybrid"):
                    for algo in self.algorithms:
                        path = f"{base_path}/{group}/window_{window}/execution_{exec_number}/{algo}_absDiff_.csv"
                        if not os.path.exists(path):
                            continue
                        try:
                            df = pd.read_csv(path)
                            if df.empty:
                                continue
                            row = df.iloc[0]
                            for m in self.metrics:
                                col = metric_map[m]
                                if col in row and pd.notna(row[col]):
                                    acc[algo][m].append(float(row[col]))
                        except Exception as e:
                            print(f"Erro ao ler {path}: {e}")
                            continue
        return acc

    def run(self) -> None:
        """Executa testes ANOVA para cada combinação de análise e métrica."""
        os.makedirs(self.output_dir, exist_ok=True)

        # Coleta dados de GeoRisk
        georisk_data = self._gather_georisk()
        self._process_anova(georisk_data, "georisk")

        # Coleta dados de Fairness - gênero
        fairness_gender = self._gather_fairness(self.fairness_gender_path)
        self._process_anova(fairness_gender, "fairness_gender")

        # Coleta dados de Fairness - atividade/grupos
        fairness_activity = self._gather_fairness(self.fairness_activity_path)
        self._process_anova(fairness_activity, "fairness_activity")

        # Salva resumo geral
        self._save_summary()

    def _process_anova(self, data: dict, analysis_type: str) -> None:
        """Processa ANOVA para um tipo de análise (georisk ou fairness).
        
        data: dict com {method: {metric: [list of values]}}
        """
        # Reorganiza: agrupa por métrica, depois aplica ANOVA
        metrics_aggregated = {}
        for metric in self.metrics:
            metrics_aggregated[metric] = {}
            for method, metric_dict in data.items():
                if metric in metric_dict and metric_dict[metric]:
                    metrics_aggregated[metric][method] = metric_dict[metric]
        
        # Executa ANOVA para cada métrica
        for metric, method_data in metrics_aggregated.items():
            if method_data:  # Se há dados para esta métrica
                self._run_anova_for_metric(analysis_type, metric, method_data)

    def _perform_anova(self, method: str, metric: str, analysis_type: str, values: list) -> None:
        """Armazena dados de um método para análise ANOVA posterior."""
        # Este método é chamado uma vez por método por métrica
        # Os dados são acumulados em self._current_anova
        pass

    def _run_anova_for_metric(self, analysis_type: str, metric: str, method_data: dict) -> None:
        """Executa ANOVA para uma combinação específica de analysis_type e metric.
        
        method_data: dict com {method_name: [lista de valores]}
        """
        groups = []
        methods = []
        sample_sizes = []
        
        for method, values in method_data.items():
            if values:
                groups.append(np.array(values))
                methods.append(method)
                sample_sizes.append(len(values))

        # Precisa de pelo menos 2 grupos
        if len(groups) < 2:
            status = "Pulado: menos de 2 métodos com dados"
            f_stat = p_val = h_stat = p_kw = None
            sig_anova = sig_kruskal = None
            num_samples = sum(sample_sizes)
        elif all(len(g) == 1 for g in groups):
            status = "Pulado: cada método tem apenas 1 amostra"
            f_stat = p_val = h_stat = p_kw = None
            sig_anova = sig_kruskal = None
            num_samples = sum(sample_sizes)
        else:
            try:
                # Executa ANOVA
                f_stat, p_val = stats.f_oneway(*groups)
                
                # Executa teste de Kruskal-Wallis (não-paramétrico) como alternativa
                h_stat, p_kw = stats.kruskal(*groups)

                # Verifica significância estatística (α = 0.05)
                sig_anova = bool(p_val < 0.05) if not np.isnan(p_val) else False
                sig_kruskal = bool(p_kw < 0.05) if not np.isnan(p_kw) else False
                
                f_stat = round(float(f_stat), 6) if not np.isnan(f_stat) else None
                p_val = round(float(p_val), 6) if not np.isnan(p_val) else None
                h_stat = round(float(h_stat), 6) if not np.isnan(h_stat) else None
                p_kw = round(float(p_kw), 6) if not np.isnan(p_kw) else None
                
                status = "✓ SIGNIFICANTE" if sig_anova else "✗ Não significante"
                num_samples = sum(sample_sizes)
            except Exception as e:
                status = f"Erro: {str(e)}"
                f_stat = p_val = h_stat = p_kw = None
                sig_anova = sig_kruskal = None
                num_samples = sum(sample_sizes)
        
        result_row = {
            "analysis_type": analysis_type,
            "metric": metric,
            "num_methods": len(methods),
            "methods": ", ".join(methods),
            "total_samples": num_samples,
            "sample_sizes": str(dict(zip(methods, sample_sizes))),
            "f_statistic": f_stat,
            "p_value_anova": p_val,
            "h_statistic": h_stat,
            "p_value_kruskal": p_kw,
            "significant_anova": sig_anova,
            "significant_kruskal": sig_kruskal,
            "status": status,
        }

        self.results.append(result_row)

        # Salva detalhes em CSV (todas as amostras)
        filename = f"anova_{analysis_type}_{metric}.csv"
        out_path = os.path.join(self.output_dir, filename)
        
        detail_rows = []
        for method, values in method_data.items():
            for value in values:
                detail_rows.append({"method": method, "value": value})
        detail_df = pd.DataFrame(detail_rows)
        detail_df.to_csv(out_path, index=False)

        print(f"{analysis_type}_{metric}: {status} (n={num_samples}, métodos={len(methods)})")

    def _save_summary(self) -> None:
        """Salva resumo geral dos resultados ANOVA."""
        results_df = pd.DataFrame(self.results)
        
        # Salva como CSV
        csv_path = os.path.join(self.output_dir, "anova_results.csv")
        results_df.to_csv(csv_path, index=False)
        
        # Salva como JSON (converte tipos numpy para tipos nativos)
        json_path = os.path.join(self.output_dir, "anova_results.json")
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(self.results, f, indent=2, ensure_ascii=False, default=str)

        print(f"\n✓ Resultados salvos em {self.output_dir}")
        print(f"  - {csv_path}")
        print(f"  - {json_path}")


if __name__ == "__main__":
    anova = AnovaAnalysis()
    anova.run()