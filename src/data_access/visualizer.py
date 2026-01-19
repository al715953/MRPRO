import json
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


def run_forensic_visualization():
    """
    Visualizador V3.5: Dashboard de Precisión Alpha Sniper (3x2).
    Añade análisis de correlación de confianza y tendencia de proximidad.
    """
    path = "data/backtest_results.json"

    if not os.path.exists(path):
        print("\n❌ Error: No se detectó 'data/backtest_results.json'.")
        return

    with open(path, "r") as f:
        data = json.load(f)

    df = pd.DataFrame(data)
    if df.empty:
        return

    # --- CONFIGURACIÓN ESTÉTICA ---
    custom_palette = {3: "#A9A9A9", 4: "#FFD700", 5: "#228B22", 6: "#00FFFF"}
    sns.set_theme(style="darkgrid")

    # Expandimos a 3 filas y 2 columnas
    fig, axes = plt.subplots(3, 2, figsize=(18, 18))
    fig.suptitle(
        "Dashboard de Misión Alpha Global - Sniper V3.5",
        fontsize=22,
        fontweight="bold",
        color="#1f77b4",
    )

    # 1. DISTRIBUCIÓN DE RANKS (0,0)
    sns.histplot(df["rank"], bins=30, kde=True, ax=axes[0, 0], color="skyblue")
    axes[0, 0].set_title("1. Distribución Global de Ranks Ganadores", fontsize=14)
    axes[0, 0].set_xlabel("Rank (IA Score)")

    # 2. CONSISTENCIA: SCORE VS RANK (0,1)
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
    axes[0, 1].set_title("2. Consistencia: AI Score vs Rank Real", fontsize=14)

    # 3. AUDITORÍA DE DISTANCIA (1,0)
    sns.boxplot(y=df["proximity"], ax=axes[1, 0], color="salmon", width=0.4)
    sns.stripplot(y=df["proximity"], color="black", alpha=0.3, ax=axes[1, 0])
    axes[1, 0].set_title("3. Desviación de Malla (Proximity)", fontsize=14)
    axes[1, 0].set_ylabel("Distancia al Ticket más cercano")

    # 4. MAPA DE SATURACIÓN (1,1)
    draws, ranks_selected = [], []
    for i, row in df.iterrows():
        d_id = row.get("draw_id", i)
        for r in row.get("selected_ranks", []):
            draws.append(d_id)
            ranks_selected.append(r)

    axes[1, 1].scatter(
        draws, ranks_selected, s=5, color="gray", alpha=0.2, label="Malla"
    )
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
    axes[1, 1].set_yscale("log")
    axes[1, 1].invert_yaxis()
    axes[1, 1].set_title("4. Saturación: Malla vs Ganador (Log Scale)", fontsize=14)

    # --- NUEVAS MÉTRICAS ALPHA GLOBAL ---

    # 5. CORRELACIÓN: CONFIANZA VS PROXIMIDAD (2,0)
    # Aquí vemos si la repulsión dinámica funciona: A mayor Score, menor Proximidad.
    sns.regplot(
        data=df,
        x="ai_score",
        y="proximity",
        scatter_kws={"alpha": 0.5, "s": 80, "color": "purple"},
        line_kws={"color": "red", "label": "Tendencia de Colapso"},
        ax=axes[2, 0],
    )
    axes[2, 0].set_title(
        "5. Correlación: Confianza (AI) vs Distancia Crítica", fontsize=14
    )
    axes[2, 0].set_xlabel("AI Confidence Score")
    axes[2, 0].set_ylabel("Distancia (Proximity)")

    # 6. EVOLUCIÓN TEMPORAL DE PROXIMIDAD (2,1)
    # Nos dice si el Forensic Loop está mejorando el sistema sorteo tras sorteo.
    axes[2, 1].plot(
        df["draw_id"],
        df["proximity"],
        marker="o",
        linestyle="-",
        color="teal",
        alpha=0.7,
    )
    axes[2, 1].axhline(y=10, color="red", linestyle="--", label="Zona de Éxito (<10)")
    axes[2, 1].set_title("6. Evolución de la Distancia Crítica", fontsize=14)
    axes[2, 1].set_xlabel("Sorteo ID")
    axes[2, 1].set_ylabel("Distancia")
    axes[2, 1].legend()

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])

    # Guardamos la versión mejorada
    output_path = "data/alpha_global_dashboard_v3.5.png"
    plt.savefig(output_path, dpi=300)
    print(f"✅ Dashboard Alpha Sniper guardado en: {output_path}")
    plt.show()
