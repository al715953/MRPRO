import json
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from rich.console import Console
from rich.table import Table
from src.data_access.config import CSV_FILE_PATH, DATA_FOLDER_PATH

console = Console()


class DecadeVarianceAnalyzer:
    """
    Analizador V1.4: Mapeo Oficial para Melate Retro.
    Mapea CONCURSO y F1-F6 para el análisis de los 200 sorteos.
    """

    def __init__(
        self,
        results_path=None,
        history_path=None,
    ):
        self.results_path = str(results_path or DATA_FOLDER_PATH / "backtest_results.json")
        self.history_path = str(history_path or CSV_FILE_PATH)

    def run_analysis(self):
        # 1. Carga de Resultados del Backtest
        if not os.path.exists(self.results_path):
            console.print(
                "[bold red]❌ Error: No se encontró 'backtest_results.json'.[/]"
            )
            return

        with open(self.results_path, "r") as f:
            df_res = pd.DataFrame(json.load(f))

        # 2. Carga de Histórico (Mapeo F1-F6)
        if not os.path.exists(self.history_path):
            console.print(f"[bold red]❌ Error: No se encontró {self.history_path}[/]")
            return

        df_hist_raw = pd.read_csv(self.history_path)

        # Mapeo directo basado en la inspección del archivo
        try:
            df_hist = df_hist_raw[
                ["CONCURSO", "F1", "F2", "F3", "F4", "F5", "F6"]
            ].copy()
            df_hist.columns = ["draw_id", "n1", "n2", "n3", "n4", "n5", "n6"]
            console.print(
                "[bold green]✅ Mapeo oficial (CONCURSO, F1-F6) aplicado correctamente.[/]"
            )
        except KeyError as e:
            console.print(f"[bold red]❌ Error de columnas: {e}[/]")
            console.print(f"Columnas disponibles: {list(df_hist_raw.columns)}")
            return

        # 3. Cruce de Datos
        df = pd.merge(df_res, df_hist, on="draw_id")

        # 4. Cálculo de Perfiles de Décadas (Ignorando F7)
        def get_profile(row):
            nums = [row["n1"], row["n2"], row["n3"], row["n4"], row["n5"], row["n6"]]
            d = [0, 0, 0, 0]  # [1-9, 10-19, 20-29, 30-39]
            for n in nums:
                if 1 <= n <= 9:
                    d[0] += 1
                elif 10 <= n <= 19:
                    d[1] += 1
                elif 20 <= n <= 29:
                    d[2] += 1
                elif 30 <= n <= 39:
                    d[3] += 1
            return f"{d[0]}-{d[1]}-{d[2]}-{d[3]}"

        df["profile"] = df.apply(get_profile, axis=1)

        # 5. Agrupación y Reporte (N=200)
        analysis = (
            df.groupby("profile")
            .agg(
                {
                    "rank": ["mean", "min", "count"],
                    "ai_score": "mean",
                    "proximity": "mean",
                }
            )
            .reset_index()
        )

        analysis.columns = [
            "profile",
            "rank_avg",
            "rank_min",
            "samples",
            "ai_avg",
            "prox_avg",
        ]
        analysis = analysis.sort_values("rank_avg")

        table = Table(
            title=f"📊 Análisis de Varianza por Décadas (N={len(df)} sorteos)"
        )
        table.add_column("Perfil (1-10-20-30)", style="cyan")
        table.add_column("Sorteos", justify="center")
        table.add_column("AI Avg", justify="right")
        table.add_column("Rank Avg", style="magenta", justify="right")
        table.add_column("Best Rank", style="green", justify="right")
        table.add_column("Prox Avg", style="yellow", justify="right")

        for _, row in analysis.iterrows():
            table.add_row(
                row["profile"],
                str(int(row["samples"])),
                f"{row['ai_avg']:.4f}",
                f"{row['rank_avg']:.0f}",
                f"{row['rank_min']:.0f}",
                f"{row['prox_avg']:.1f}",
            )
        console.print(table)

        # 6. Gráfico de Estabilidad
        plt.figure(figsize=(14, 7))
        counts = df["profile"].value_counts()
        df_f = df[df["profile"].isin(counts[counts > 1].index)]
        sns.boxplot(
            data=df_f,
            x="profile",
            y="rank",
            palette="viridis",
            hue="profile",
            legend=False,
        )
        plt.yscale("log")
        plt.title("Varianza de Rank por Perfil (Escala Log)", fontsize=15)
        plt.grid(True, which="both", ls="-", alpha=0.2)
        plt.tight_layout()
        output_path = DATA_FOLDER_PATH / "full_variance_analysis.png"
        plt.savefig(output_path)
        console.print(
            f"\n[bold green]✅ Gráfico guardado en '{output_path}'[/]"
        )


if __name__ == "__main__":
    DecadeVarianceAnalyzer().run_analysis()
