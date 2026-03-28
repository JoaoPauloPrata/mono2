import os
import pandas as pd
import numpy as np
import json
from scipy import stats
from statsmodels.stats.multicomp import pairwise_tukeyhsd


class PostHocAnalysis:
    """
    Realiza testes post-hoc (Tukey HSD) para identificar quais métodos
    diferem estatisticamente entre si e quais estão empatados.
    
    Salva os resultados em:
      - tukey_{analysis_type}_{metric}.csv (comparações pairwise)
      - grouped_{analysis_type}_{metric}.csv (grupos de métodos empatados)
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
        self.tukey_results = []
        self.grouped_results = []

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
                            continue
        return acc

    def run(self) -> None:
        """Executa testes post-hoc para cada combinação de análise e métrica."""
        os.makedirs(self.output_dir, exist_ok=True)

        print("Executando testes post-hoc (Tukey HSD)...\n")

        # GeoRisk
        georisk_data = self._gather_georisk()
        self._process_tukey(georisk_data, "georisk")

        # Fairness gender
        fairness_gender = self._gather_fairness(self.fairness_gender_path)
        self._process_tukey(fairness_gender, "fairness_gender")

        # Fairness activity
        fairness_activity = self._gather_fairness(self.fairness_activity_path)
        self._process_tukey(fairness_activity, "fairness_activity")

        # Salva resultados
        self._save_results()

    def _process_tukey(self, data: dict, analysis_type: str) -> None:
        """Processa teste de Tukey para um tipo de análise."""
        for metric in self.metrics:
            method_data = {}
            for method, metric_dict in data.items():
                if metric in metric_dict and metric_dict[metric]:
                    method_data[method] = metric_dict[metric]

            if len(method_data) >= 2:
                self._run_tukey(analysis_type, metric, method_data)

    def _run_tukey(self, analysis_type: str, metric: str, method_data: dict) -> None:
        """Executa teste de Tukey HSD para comparações pairwise."""
        # Prepara dados para Tukey
        values = []
        groups = []
        
        for method, vals in method_data.items():
            values.extend(vals)
            groups.extend([method] * len(vals))

        # Executa Tukey HSD
        tukey_result = pairwise_tukeyhsd(endog=values, groups=groups, alpha=0.05)
        
        # Processa resultados
        summary_df = pd.DataFrame(data=tukey_result.summary().data[1:], 
                                  columns=tukey_result.summary().data[0])
        
        # Salva comparações pairwise
        filename_tukey = f"tukey_{analysis_type}_{metric}.csv"
        out_path_tukey = os.path.join(self.output_dir, filename_tukey)
        summary_df.to_csv(out_path_tukey, index=False)

        # Agrupa métodos em grupos empatados
        grouped = self._group_methods(method_data, summary_df)
        
        # Salva grupos
        filename_grouped = f"grouped_{analysis_type}_{metric}.csv"
        out_path_grouped = os.path.join(self.output_dir, filename_grouped)
        grouped_df = pd.DataFrame(grouped)
        grouped_df.to_csv(out_path_grouped, index=False)

        print(f"✓ {analysis_type}_{metric}: {len(grouped)} grupo(s)")
        for i, group in enumerate(grouped, 1):
            methods = group['methods'].split(", ")
            mean_val = group['mean_value']
            print(f"  Grupo {i} (média={mean_val:.6f}): {', '.join(methods)}")

    def _group_methods(self, method_data: dict, tukey_df: pd.DataFrame) -> list:
        """Agrupa métodos que são estatisticamente similares (não rejeitam H0)."""
        # Cria grafo de similaridade
        n_methods = len(method_data)
        methods_list = list(method_data.keys())
        
        # Matriz de similaridade: True se métodos não diferem significativamente
        similar = np.ones((n_methods, n_methods), dtype=bool)
        
        for _, row in tukey_df.iterrows():
            group1 = row['group1']
            group2 = row['group2']
            reject = row['reject']  # True se diferem significativamente
            
            idx1 = methods_list.index(group1)
            idx2 = methods_list.index(group2)
            
            if reject:
                similar[idx1, idx2] = False
                similar[idx2, idx1] = False

        # Agrupa métodos similares usando componentes conexas
        groups = []
        unvisited = set(range(n_methods))
        
        while unvisited:
            current = unvisited.pop()
            component = {current}
            queue = [current]
            
            while queue:
                node = queue.pop(0)
                for neighbor in range(n_methods):
                    if neighbor in unvisited and similar[node, neighbor]:
                        component.add(neighbor)
                        queue.append(neighbor)
                        unvisited.remove(neighbor)
            
            # Extrai métodos do componente
            component_methods = [methods_list[i] for i in component]
            component_values = np.concatenate([method_data[m] for m in component_methods])
            mean_value = float(np.mean(component_values))
            
            groups.append({
                "group": len(groups) + 1,
                "methods": ", ".join(component_methods),
                "num_methods": len(component_methods),
                "mean_value": round(mean_value, 6),
            })

        return groups

    def _save_results(self) -> None:
        """Salva resumo dos resultados post-hoc."""
        json_path = os.path.join(self.output_dir, "posthoc_summary.json")
        
        summary = {
            "description": "Análise post-hoc (Tukey HSD) identificando grupos de métodos empatados",
            "alpha": 0.05,
            "generated_files": [
                "tukey_*.csv - Comparações pairwise detalhadas",
                "grouped_*.csv - Grupos de métodos empatados"
            ]
        }
        
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)

        print(f"\n✓ Resultados salvos em {self.output_dir}")


if __name__ == "__main__":
    posthoc = PostHocAnalysis()
    posthoc.run()