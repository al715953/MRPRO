import json
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


def run_forensic_visualization():
    """Visualizador V3.4: Dashboard de Saturación Neural (2x2)."""
    path = "data/backtest_results.json"

    if not os.path.exists(path):
        print("\n❌ Error: No se detectó 'data/backtest_results.json'.")
        return

    with open(path, "r") as f:
        data = json.load(f)

    df = pd.DataFrame(data)
    if df.empty:
        return

    # --- CONFIGURACIÓN DE COLORES SNIPER ---
    custom_palette = {3: "#A9A9A9", 4: "#FFD700", 5: "#228B22", 6: "#00FFFF"}
    sns.set_theme(style="darkgrid")

    # Cambiamos a 2x2 para acomodar la nueva métrica
    fig, axes = plt.subplots(2, 2, figsize=(18, 12))
    fig.suptitle(
        "Dashboard de Precisión MRPRO V9.9.1 - Neural Mesh Edition",
        fontsize=20,
        fontweight="bold",
    )

    # 1. DISTRIBUCIÓN DE RANKS (Arriba-Izquierda)
    sns.histplot(df["rank"], bins=30, kde=True, ax=axes[0, 0], color="skyblue")
    axes[0, 0].set_title("Distribución Global de Ranks Ganadores", fontsize=14)
    axes[0, 0].set_xlabel("Rank (IA Score)")

    # 2. CONSISTENCIA: SCORE VS RANK (Arriba-Derecha)
    sns.scatterplot(
        data=df[df["hits"] >= 3],
        x="ai_score",
        y="rank",
        hue="hits",
        palette=custom_palette,
        s=120,
        edgecolor="white",
        alpha=0.8,
        ax=axes[0, 1],
    )
    axes[0, 1].invert_yaxis()
    axes[0, 1].set_title("Consistencia: AI Score vs Rank Real", fontsize=14)

    # 3. AUDITORÍA DE DISTANCIA (Abajo-Izquierda)
    sns.boxplot(y=df["proximity"], ax=axes[1, 0], color="salmon", width=0.4)
    sns.stripplot(y=df["proximity"], color="black", alpha=0.3, ax=axes[1, 0])
    axes[1, 0].set_title("Desviación de Malla (Proximity)", fontsize=14)
    axes[1, 0].set_ylabel("Distancia al Ticket más cercano")

    # 4. NUEVO: MAPA DE CALOR DE SATURACIÓN (Abajo-Derecha)
    # Visualizamos los 20 Ranks seleccionados vs el Rank ganador por sorteo
    draws = []
    ranks_selected = []

    # Aplanamos los datos para el heatmap de puntos
    for i, row in df.iterrows():
        d_id = row.get("draw_id", i)
        winner_r = row["rank"]
        selected_rs = row.get("selected_ranks", [])

        for r in selected_rs:
            draws.append(d_id)
            ranks_selected.append(r)

    # Graficamos la malla (los 20 tickets)
    axes[1, 1].scatter(
        draws, ranks_selected, s=5, color="gray", alpha=0.3, label="Malla (20 Tkt)"
    )

    # Graficamos el ganador (Highlight)
    sns.scatterplot(
        data=df,
        x="draw_id",
        y="rank",
        hue="hits",
        palette=custom_palette,
        s=100,
        marker="D",
        ax=axes[1, 1],
        legend=False,
    )

    axes[1, 1].set_yscale("log")  # Escala logarítmica para ver mejor el Top 100
    axes[1, 1].invert_yaxis()
    axes[1, 1].set_title(
        "Mapa de Saturación: Malla vs Ganador (Log Scale)", fontsize=14
    )
    axes[1, 1].set_xlabel("Sorteo ID")
    axes[1, 1].set_ylabel("Rank")

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])

    # Guardar para inspección
    plt.savefig("data/precision_dashboard_v9.9.png", dpi=300)
    plt.show()
