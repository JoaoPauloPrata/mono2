import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os
from glob import glob

class FairnessVisualization:
    """
    Classe para visualizar e analisar diferenças de fairness entre métodos
    """
    
    def __init__(self):
        self.fairness_dir = "../../data/MetricsForMethods/Fairness"
        self.simple_methods = ["itemKNN", "BIAS", "userKNN", "SVD", "BIASEDMF"]
        self.hybrid_methods = ["BayesianRidge", "Tweedie", "Ridge", "RandomForest", 
                              "Bagging", "AdaBoost", "GradientBoosting", "LinearSVR"]
        self.metrics = ["RMSE", "MAE", "NDCG", "F1"]
        
        # Configurações de estilo
        plt.style.use('default')
        sns.set_palette("husl")
        
    def load_differences_data(self, window=None):
        """
        Carrega dados de diferenças para uma janela específica ou todas
        """
        if window is not None:
            file_path = f"{self.fairness_dir}/FairnessDifferences_Window{window}.csv"
            if os.path.exists(file_path):
                return pd.read_csv(file_path)
            else:
                print(f"Arquivo não encontrado: {file_path}")
                return None
        else:
            # Carrega todas as janelas
            pattern = f"{self.fairness_dir}/FairnessDifferences_Window*.csv"
            files = glob(pattern)
            
            if not files:
                print(f"Nenhum arquivo encontrado em: {pattern}")
                return None
                
            all_data = []
            for file in files:
                df = pd.read_csv(file)
                all_data.append(df)
            
            return pd.concat(all_data, ignore_index=True)
    
    def calculate_group_averages(self, df):
        """
        Calcula médias das diferenças absolutas por grupo (simples vs híbridos)
        """
        results = {}
        
        for metric in self.metrics:
            abs_diff_col = f"{metric}_AbsDiff"
            
            if abs_diff_col in df.columns:
                # Médias para métodos simples
                simple_data = df[df['method'].isin(self.simple_methods)][abs_diff_col].dropna()
                simple_avg = simple_data.mean() if len(simple_data) > 0 else np.nan
                
                # Médias para métodos híbridos
                hybrid_data = df[df['method'].isin(self.hybrid_methods)][abs_diff_col].dropna()
                hybrid_avg = hybrid_data.mean() if len(hybrid_data) > 0 else np.nan
                
                results[metric] = {
                    'simple_avg': simple_avg,
                    'hybrid_avg': hybrid_avg,
                    'improvement': simple_avg - hybrid_avg if not (np.isnan(simple_avg) or np.isnan(hybrid_avg)) else np.nan,
                    'simple_count': len(simple_data),
                    'hybrid_count': len(hybrid_data)
                }
        
        return results
    
    def plot_abs_differences_by_method(self, df, window=None, save_path=None):
        """
        Gráfico de barras das diferenças absolutas por método
        """
        # Prepara dados para o gráfico
        plot_data = []
        
        for metric in self.metrics:
            abs_diff_col = f"{metric}_AbsDiff"
            if abs_diff_col in df.columns:
                for _, row in df.iterrows():
                    if pd.notna(row[abs_diff_col]):
                        plot_data.append({
                            'Method': row['method'],
                            'Metric': metric,
                            'AbsDiff': row[abs_diff_col],
                            'Type': 'Simples' if row['method'] in self.simple_methods else 'Híbrido'
                        })
        
        if not plot_data:
            print("Nenhum dado válido para plotar")
            return
        
        plot_df = pd.DataFrame(plot_data)
        
        # Cria subplots para cada métrica
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle(f'Diferenças Absolutas de Fairness por Método{" - Janela " + str(window) if window else " - Todas as Janelas"}', 
                    fontsize=16, fontweight='bold')
        
        axes = axes.flatten()
        
        for i, metric in enumerate(self.metrics):
            metric_data = plot_df[plot_df['Metric'] == metric]
            
            if not metric_data.empty:
                # Ordena por diferença absoluta
                method_order = metric_data.groupby('Method')['AbsDiff'].mean().sort_values().index
                
                sns.barplot(data=metric_data, x='Method', y='AbsDiff', hue='Type', 
                           order=method_order, ax=axes[i])
                
                axes[i].set_title(f'{metric} - Diferença Absoluta', fontweight='bold')
                axes[i].set_xlabel('Método')
                axes[i].set_ylabel('Diferença Absoluta')
                axes[i].tick_params(axis='x', rotation=45)
                axes[i].grid(axis='y', alpha=0.3)
                
                # Adiciona linha da média
                mean_val = metric_data['AbsDiff'].mean()
                axes[i].axhline(y=mean_val, color='red', linestyle='--', alpha=0.7, 
                               label=f'Média: {mean_val:.4f}')
                axes[i].legend()
            else:
                axes[i].set_title(f'{metric} - Sem dados')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Gráfico salvo em: {save_path}")
        
        #plt.show()
    
    def plot_group_comparison(self, group_stats, save_path=None):
        """
        Gráfico comparando médias entre grupos (simples vs híbridos)
        """
        # Prepara dados
        metrics = []
        simple_avgs = []
        hybrid_avgs = []
        improvements = []
        
        for metric, stats in group_stats.items():
            if not np.isnan(stats['simple_avg']) and not np.isnan(stats['hybrid_avg']):
                metrics.append(metric)
                simple_avgs.append(stats['simple_avg'])
                hybrid_avgs.append(stats['hybrid_avg'])
                improvements.append(stats['improvement'])
        
        if not metrics:
            print("Nenhum dado válido para comparação de grupos")
            return
        
        # Cria gráfico
        x = np.arange(len(metrics))
        width = 0.35
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # Gráfico 1: Comparação das médias
        bars1 = ax1.bar(x - width/2, simple_avgs, width, label='Métodos Simples', alpha=0.8)
        bars2 = ax1.bar(x + width/2, hybrid_avgs, width, label='Métodos Híbridos', alpha=0.8)
        
        ax1.set_xlabel('Métrica')
        ax1.set_ylabel('Diferença Absoluta Média')
        ax1.set_title('Comparação: Métodos Simples vs Híbridos', fontweight='bold')
        ax1.set_xticks(x)
        ax1.set_xticklabels(metrics)
        ax1.legend()
        ax1.grid(axis='y', alpha=0.3)
        
        # Adiciona valores nas barras
        for bar in bars1:
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2., height + 0.001,
                    f'{height:.4f}', ha='center', va='bottom', fontsize=10)
        
        for bar in bars2:
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2., height + 0.001,
                    f'{height:.4f}', ha='center', va='bottom', fontsize=10)
        
        # Gráfico 2: Melhoria (valores positivos = híbridos são melhores)
        colors = ['green' if imp > 0 else 'red' for imp in improvements]
        bars3 = ax2.bar(metrics, improvements, color=colors, alpha=0.7)
        
        ax2.set_xlabel('Métrica')
        ax2.set_ylabel('Melhoria (Simples - Híbridos)')
        ax2.set_title('Melhoria dos Métodos Híbridos\n(Valores positivos = Híbridos melhores)', fontweight='bold')
        ax2.axhline(y=0, color='black', linestyle='-', alpha=0.3)
        ax2.grid(axis='y', alpha=0.3)
        
        # Adiciona valores nas barras
        for bar, imp in zip(bars3, improvements):
            height = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width()/2., height + (0.001 if height > 0 else -0.001),
                    f'{imp:.4f}', ha='center', va='bottom' if height > 0 else 'top', fontsize=10)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Gráfico salvo em: {save_path}")
        
        #plt.show()
    
    def plot_distribution_comparison(self, df, save_path=None):
        """
        Boxplot comparando distribuições entre grupos
        """
        plot_data = []
        
        for metric in self.metrics:
            abs_diff_col = f"{metric}_AbsDiff"
            if abs_diff_col in df.columns:
                for _, row in df.iterrows():
                    if pd.notna(row[abs_diff_col]):
                        plot_data.append({
                            'Metric': metric,
                            'AbsDiff': row[abs_diff_col],
                            'Type': 'Simples' if row['method'] in self.simple_methods else 'Híbrido'
                        })
        
        if not plot_data:
            print("Nenhum dado válido para distribuições")
            return
        
        plot_df = pd.DataFrame(plot_data)
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle('Distribuição das Diferenças Absolutas: Simples vs Híbridos', 
                    fontsize=16, fontweight='bold')
        
        axes = axes.flatten()
        
        for i, metric in enumerate(self.metrics):
            metric_data = plot_df[plot_df['Metric'] == metric]
            
            if not metric_data.empty:
                sns.boxplot(data=metric_data, x='Type', y='AbsDiff', ax=axes[i])
                axes[i].set_title(f'{metric}', fontweight='bold')
                axes[i].set_xlabel('Tipo de Método')
                axes[i].set_ylabel('Diferença Absoluta')
                axes[i].grid(axis='y', alpha=0.3)
            else:
                axes[i].set_title(f'{metric} - Sem dados')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Gráfico salvo em: {save_path}")

        #plt.show()

    def generate_summary_report(self, group_stats, window=None):
        """
        Gera relatório resumo da análise
        """
        print("="*80)
        print(f"📊 RELATÓRIO DE FAIRNESS{' - JANELA ' + str(window) if window else ' - CONSOLIDADO'}")
        print("="*80)
        
        for metric, stats in group_stats.items():
            print(f"\n🎯 {metric}:")
            print(f"   Métodos Simples  - Média: {stats['simple_avg']:.6f} ({stats['simple_count']} métodos)")
            print(f"   Métodos Híbridos - Média: {stats['hybrid_avg']:.6f} ({stats['hybrid_count']} métodos)")
            
            if not np.isnan(stats['improvement']):
                if stats['improvement'] > 0:
                    print(f"   ✅ MELHORIA: {stats['improvement']:.6f} (Híbridos {abs(stats['improvement']):.6f} melhores)")
                else:
                    print(f"   ❌ PIORA: {stats['improvement']:.6f} (Híbridos {abs(stats['improvement']):.6f} piores)")
            else:
                print(f"   ⚠️  Não foi possível calcular melhoria")
        
        print("\n" + "="*80)
        
        # Salva relatório em arquivo
        report_file = f"{self.fairness_dir}/FairnessReport{'_Window' + str(window) if window else '_Consolidated'}.txt"
        
        with open(report_file, 'w', encoding='utf-8') as f:
            f.write(f"RELATÓRIO DE FAIRNESS{' - JANELA ' + str(window) if window else ' - CONSOLIDADO'}\n")
            f.write("="*80 + "\n\n")
            
            for metric, stats in group_stats.items():
                f.write(f"{metric}:\n")
                f.write(f"   Métodos Simples  - Média: {stats['simple_avg']:.6f} ({stats['simple_count']} métodos)\n")
                f.write(f"   Métodos Híbridos - Média: {stats['hybrid_avg']:.6f} ({stats['hybrid_count']} métodos)\n")
                
                if not np.isnan(stats['improvement']):
                    status = "MELHORIA" if stats['improvement'] > 0 else "PIORA"
                    f.write(f"   {status}: {stats['improvement']:.6f}\n")
                else:
                    f.write(f"   Não foi possível calcular melhoria\n")
                f.write("\n")
        
        print(f"📄 Relatório salvo em: {report_file}")
    
    def analyze_window(self, window):
        """
        Análise completa para uma janela específica
        """
        print(f"🔍 Analisando fairness para janela {window}")
        
        df = self.load_differences_data(window)
        if df is None:
            return
        
        # Calcula estatísticas por grupo
        group_stats = self.calculate_group_averages(df)
        
        # Gera gráficos
        output_dir = f"{self.fairness_dir}/plots"
        os.makedirs(output_dir, exist_ok=True)
        
        self.plot_abs_differences_by_method(df, window, 
                                          f"{output_dir}/differences_by_method_window{window}.png")
        
        self.plot_group_comparison(group_stats, 
                                 f"{output_dir}/group_comparison_window{window}.png")
        
        self.plot_distribution_comparison(df, 
                                        f"{output_dir}/distribution_comparison_window{window}.png")
        
        # Gera relatório
        self.generate_summary_report(group_stats, window)
        
        return group_stats
    
    def analyze_all_windows(self):
        """
        Análise consolidada de todas as janelas
        """
        print("🔍 Analisando fairness consolidado (todas as janelas)")
        
        df = self.load_differences_data()
        if df is None:
            return
        
        # Calcula estatísticas por grupo
        group_stats = self.calculate_group_averages(df)
        
        # Gera gráficos
        output_dir = f"{self.fairness_dir}/plots"
        os.makedirs(output_dir, exist_ok=True)
        
        self.plot_abs_differences_by_method(df, None, 
                                          f"{output_dir}/differences_by_method_all_windows.png")
        
        self.plot_group_comparison(group_stats, 
                                 f"{output_dir}/group_comparison_all_windows.png")
        
        self.plot_distribution_comparison(df, 
                                        f"{output_dir}/distribution_comparison_all_windows.png")
        
        # Gera relatório
        self.generate_summary_report(group_stats)
        
        return group_stats


def main():
    """
    Exemplo de uso da classe de visualização
    """
    visualizer = FairnessVisualization()
    
    print("🎨 VISUALIZADOR DE FAIRNESS")
    print("="*50)
    
    try:
        for i in range(1, 21):
            # Análise para janela i
            print(f"\n1️⃣ Análise da Janela {i}:")
            visualizer.analyze_window(i)
            
            # Análise consolidada (se houver mais janelas)
            print("\n📊 Análise Consolidada:")
            visualizer.analyze_all_windows()
        
    except Exception as e:
        print(f"❌ Erro durante análise: {e}")


if __name__ == "__main__":
    main()
