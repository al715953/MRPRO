import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from rich.console import Console
from rich.table import Table
from src.data_access.config import MASTER_LOG_PATH


class AlphaComparator:
    """
    Motor de Comparativa V1.0.
    Analiza el rendimiento histórico de las versiones de MRPRO
    basado en el archivo master_performance.csv.
    """

    def __init__(self):
        self.console = Console()
        self.path = MASTER_LOG_PATH

    def run_comparison(self):
        """Ejecuta el análisis completo y genera el Dashboard de Evolución."""
        if not os.path.exists(self.path):
            self.console.print(
                f"\n[bold red]❌ Error:[/] No se encontró la bitácora en {self.path}"
            )
            return

        # 1. CARGA Y LIMPIEZA DE DATOS
        df = pd.read_csv(self.path)
        if df.empty:
            self.console.print(
                "\n[bold yellow]⚠ Advertencia:[/] La bitácora está vacía."
            )
            return

        # 2. RENDERIZADO DE TABLA EN CONSOLA
        self._show_text_summary(df)

        # 3. GENERACIÓN DE DASHBOARD VISUAL
        self._generate_visual_dashboard(df)

    def _show_text_summary(self, df):
        """Muestra un resumen ejecutivo en la terminal."""
        table = Table(title="📋 HISTORIAL DE VERSIONES - ALPHA GLOBAL")

        table.add_column("Tag de Versión", style="cyan", no_wrap=True)
        table.add_column("Sorteos", justify="right")
        table.add_column("ROI (%)", justify="right")
        table.add_column("Balance ($)", justify="right")
        table.add_column("4/6", style="white")
        table.add_column("5/6", style="green")
        table.add_column("6/6", style="bold cyan")

        # Agrupamos por tag para ver el mejor resultado de cada versión
        summary = (
            df.groupby("tag")
            .agg(
                {
                    "draws": "max",
                    "ROI": "mean",
                    "balance": "sum",
                    "hits_4": "sum",
                    "hits_5": "sum",
                    "hits_6": "sum",
                }
            )
            .reset_index()
        )

        for _, row in summary.iterrows():
            roi_style = "bold green" if row["ROI"] > 0 else "bold red"
            table.add_row(
                str(row["tag"]),
                str(int(row["draws"])),
                f"[{roi_style}]{row['ROI']:.2f}%[/]",
                f"${row['balance']:.2f}",
                str(int(row["hits_4"])),
                str(int(row["hits_5"])),
                str(int(row["hits_6"])),
            )

        self.console.print(table)

    def _generate_visual_dashboard(self, df):
        """Crea gráficas de barras y líneas para comparar ROI y Eficiencia."""
        sns.set_theme(style="darkgrid")
        fig, axes = plt.subplots(1, 2, figsize=(18, 8))
        fig.suptitle(
            "Comparativa de Estrategias: Evolución de MRPRO",
            fontsize=18,
            fontweight="bold",
        )

        # Gráfico 1: ROI por Versión
        sns.barplot(data=df, x="tag", y="ROI", palette="viridis", ax=axes[0])
        axes[0].axhline(0, color="red", linestyle="--", alpha=0.6)
        axes[0].set_title("Eficiencia de Inversión (ROI %)", fontsize=14)
        axes[0].set_xticklabels(axes[0].get_xticklabels(), rotation=45, ha="right")

        # Gráfico 2: Acumulado de Hits (4/6 y 5/6)
        # Transformamos los datos para un gráfico apilado
        hits_df = df.melt(
            id_vars=["tag"],
            value_vars=["hits_4", "hits_5"],
            var_name="Tipo_Hit",
            value_name="Cantidad",
        )

        sns.barplot(
            data=hits_df,
            x="tag",
            y="Cantidad",
            hue="Tipo_Hit",
            palette={"hits_4": "silver", "hits_5": "gold"},
            ax=axes[1],
        )
        axes[1].set_title("Volumen de Aciertos Críticos", fontsize=14)
        axes[1].set_xticklabels(axes[1].get_xticklabels(), rotation=45, ha="right")

        plt.tight_layout(rect=[0, 0.03, 1, 0.95])

        # Guardar comparativa
        out_path = os.path.join("data", "evolution_comparison.png")
        plt.savefig(out_path, dpi=300)
        self.console.print(
            f"\n[bold green]✅ Dashboard comparativo guardado en:[/] {out_path}"
        )
        plt.show()


# Función de conveniencia para llamar desde el main
def run_master_comparison():
    comparator = AlphaComparator()
    comparator.run_comparison()
