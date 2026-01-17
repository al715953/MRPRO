import json
import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


def run_forensic_visualization():
    """Visualizador V3.3: Mapeo de colores Amarillito-Verde-Diamante."""
    path = "data/backtest_results.json"

    if not os.path.exists(path):
        print(
            "\n❌ Error: No se detectó 'data/backtest_results.json'. Corre el Backtest primero."
        )
        return

    with open(path, "r") as f:
        data = json.load(f)

    df = pd.DataFrame(data)
    if df.empty:
        return

    # --- CONFIGURACIÓN DE TUS COLORES (Sincronizados con el Log) ---
    # 3 hits: Gris (Neutro)
    # 4 hits: Dorado/Amarillito (#FFD700)
    # 5 hits: Verde Intenso (#228B22)
    # 6 hits: Azul Diamante/Cian (#00FFFF)
    custom_palette = {3: "#A9A9A9", 4: "#FFD700", 5: "#228B22", 6: "#00FFFF"}

    sns.set_theme(style="darkgrid")
    fig, axes = plt.subplots(1, 3, figsize=(20, 7))
    fig.suptitle("Dashboard de Precisión MRPRO V9.8.7 - Zero-Gap Edition", fontsize=18)

    # 1. DISTRIBUCIÓN DE RANKS
    sns.histplot(df["rank"], bins=30, kde=True, ax=axes[0], color="skyblue")
    axes[0].set_title("Distribución de Ranks Ganadores")

    # 2. CONSISTENCIA: SCORE VS RANK (Con tus colores)
    sns.scatterplot(
        data=df[df["hits"] >= 3],
        x="ai_score",
        y="rank",
        hue="hits",
        palette=custom_palette,
        s=150,
        edgecolor="white",
        alpha=0.9,
        ax=axes[1],
    )
    axes[1].invert_yaxis()
    axes[1].set_title("Consistencia: AI Score vs Rank")

    # 3. AUDITORÍA DE DISTANCIA (PROXIMITY)
    sns.boxplot(y=df["proximity"], ax=axes[2], color="salmon")
    sns.stripplot(y=df["proximity"], color="black", alpha=0.3, ax=axes[2])
    axes[2].set_title("Auditoría de Distancia (Proximity)")

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.show()
