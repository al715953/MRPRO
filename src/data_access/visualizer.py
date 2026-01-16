import json
import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


def run_forensic_visualization():
    """
    Lee el archivo de auditoría generado por el backtester
    y despliega el dashboard de rendimiento estadístico V9.8.
    """
    # Ruta relativa desde la raíz del proyecto
    path = "data/backtest_results.json"

    if not os.path.exists(path):
        print("\n❌ Error: No se detectó el archivo 'data/backtest_results.json'.")
        print("   Asegúrate de ejecutar un Backtest (Opción 6) primero.")
        return

    try:
        with open(path, "r") as f:
            data = json.load(f)

        df = pd.DataFrame(data)

        if df.empty:
            print("⚠️ El archivo de resultados está vacío.")
            return

        # Configuración del entorno visual
        sns.set_theme(style="darkgrid")
        fig, axes = plt.subplots(1, 3, figsize=(18, 6))
        fig.suptitle(
            "Dashboard de Precisión MRPRO V9.8 - Ghost Sniper Edition", fontsize=16
        )

        # Panel 1: Histograma de Ranks
        sns.histplot(df["rank"], bins=20, kde=True, ax=axes[0], color="skyblue")
        axes[0].set_title("Distribución de Ranks Ganadores")
        axes[0].set_xlabel("Posición en el Ranking")

        # --- PALETA DE ALTO CONTRASTE REVISADA V9.8.1 ---
        # 3 hits: Negro
        # 4 hits: Morado Claro (Medium Purple)
        # 5 hits: Azul Marino (Navy)
        # 6 hits: Dorado (Gold) - Preparado para el impacto directo
        custom_palette = {3: "#BB2309", 4: "#6B42BE", 5: "#32A151", 6: "#FFD700"}

        # Panel 2: Correlación Score vs Rank
        sns.scatterplot(
            data=df,
            x="ai_score",
            y="rank",
            hue="hits",
            palette=custom_palette,  # Aplicamos la paleta con soporte para 6 hits
            s=130,  # Un poco más grandes para resaltar el morado y dorado
            edgecolor="white",  # Mantenemos el borde para separar puntos encimados
            alpha=0.9,  # Mayor opacidad para que los colores sean sólidos
            ax=axes[1],
        )
        axes[1].set_title("Consistencia: Score vs Rank")
        axes[1].invert_yaxis()  # El Rank #1 debe estar en la parte superior

        # Panel 3: Análisis de Proximidad (Zancada)
        sns.boxplot(y=df["proximity"], ax=axes[2], color="salmon")
        axes[2].set_title("Auditoría de Distancia (Proximity)")
        axes[2].set_ylabel("Posiciones de desfase")

        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        plt.show()

    except Exception as e:
        print(f"❌ Error al generar la visualización: {e}")
