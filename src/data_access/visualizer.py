import json
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


def run_forensic_visualization():
    """
    Visualizador V4.0: Dashboard de Precisión Alpha Sniper (3x2).
    Optimizado para detectar 5/6 y 6/6 mediante escala logarítmica y Heatmap de Fusión.
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

    fig, axes = plt.subplots(3, 2, figsize=(20, 20))
    fig.suptitle(
        "Dashboard de Misión Alpha Global - Sniper V4.0 (Entropy Mirroring)",
        fontsize=24,
        fontweight="bold",
        color="#1f77b4",
    )

    # 1. DISTRIBUCIÓN DE RANKS (Zoom Quirúrgico al Top 2000)
    sns.histplot(
        df[df["rank"] <= 2000]["rank"],
        bins=50,
        kde=True,
        ax=axes[0, 0],
        color="skyblue",
    )
    axes[0, 0].set_title("1. Distribución de Ranks (Zoom Top 2000)", fontsize=16)

    # 2. CONSISTENCIA: SCORE VS RANK (Escala Logarítmica Crítica)
    # Permite ver la diferencia real entre el Rank #10 y el #100
    sns.scatterplot(
        data=df[df["hits"] >= 3],
        x="ai_score",
        y="rank",
        hue="hits",
        palette=custom_palette,
        s=150,
        edgecolor="black",
        alpha=0.9,
        ax=axes[0, 1],
    )
    axes[0, 1].set_yscale("log")
    axes[0, 1].invert_yaxis()
    axes[0, 1].axhline(
        y=20, color="gold", linestyle="--", label="Zona de Ataque (Top 20)"
    )
    axes[0, 1].set_title("2. Consistencia: AI Score vs Rank (Log Scale)", fontsize=16)
    axes[0, 1].legend(title="Aciertos")

    # 3. HEATMAP DE FUSIÓN: AI VS GEO (NUEVO SENSOR)
    # Aquí validamos si el Jackpot cae en la zona de alta resonancia
    sns.scatterplot(
        data=df,
        x="ai_score",
        y="geo_score",
        hue="hits",
        size="rank",
        sizes=(20, 200),
        palette=custom_palette,
        alpha=0.6,
        ax=axes[1, 0],
    )
    axes[1, 0].set_title("3. Heatmap: AI Confidence vs Geo Resonance", fontsize=16)
    axes[1, 0].set_xlabel("AI Score")
    axes[1, 0].set_ylabel("Geo Score")

    # 4. SATURACIÓN DE MALLA (Log Scale + Record Jackpot)
    # Comparamos contra nuestro récord anterior de Rank #289
    axes[1, 1].scatter(
        df["draw_id"], [10] * len(df), s=2, color="gray", alpha=0.1
    )  # Simulación de malla
    sns.scatterplot(
        data=df,
        x="draw_id",
        y="rank",
        hue="hits",
        palette=custom_palette,
        s=120,
        marker="D",
        ax=axes[1, 1],
        legend=False,
    )
    axes[1, 1].set_yscale("log")
    axes[1, 1].invert_yaxis()
    axes[1, 1].axhline(
        y=289, color="cyan", linestyle=":", label="Record V33.4 (Rank #289)"
    )
    axes[1, 1].set_title("4. Saturación: Malla vs Ganador (Log Scale)", fontsize=16)

    # 5. CORRELACIÓN: CONFIANZA VS PROXIMIDAD (Validación de IA)
    sns.regplot(
        data=df,
        x="ai_score",
        y="proximity",
        scatter_kws={"alpha": 0.4, "s": 100, "color": "purple"},
        line_kws={"color": "red", "label": "Tendencia Sniper"},
        ax=axes[2, 0],
    )
    axes[2, 0].set_title(
        "5. Correlación: Confianza (AI) vs Distancia Crítica", fontsize=16
    )

    # 6. EVOLUCIÓN DE PROXIMIDAD (Zoom a la Zona de Éxito)
    axes[2, 1].plot(df["draw_id"], df["proximity"], marker="o", color="teal", alpha=0.8)
    axes[2, 1].axhline(y=10, color="red", linestyle="--", label="Zona de Éxito (<10)")
    axes[2, 1].set_ylim(-500, 10000)  # Zoom a la zona de interés
    axes[2, 1].set_title("6. Evolución de la Distancia Crítica", fontsize=16)
    axes[2, 1].legend()

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    output_path = "data/alpha_global_dashboard_v4.0.png"
    plt.savefig(output_path, dpi=300)
    print(f"✅ Dashboard Alpha Sniper V4.0 guardado en: {output_path}")
