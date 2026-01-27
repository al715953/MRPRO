# src/data_access/visualizer.py

import json
import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


def run_forensic_visualization(json_path="data/backtest_results.json"):
    """
    VISUALIZER V6.3: Auditor de Resonancia (Fix de Palette & Hue).
    Optimizado para evitar ValueErrors por tipos de datos.
    """
    if not os.path.exists(json_path):
        print(f"❌ Error: Archivo no encontrado en {json_path}")
        return

    try:
        with open(json_path, "r") as f:
            data = json.load(f)
        df = pd.DataFrame(data)
    except Exception as e:
        print(f"❌ Error al cargar JSON: {e}")
        return

    # Aseguramos que 'hits' sea entero para consistencia
    df["hits"] = df["hits"].astype(int)

    # 1. Configuración de Estética Forense
    plt.style.use("dark_background")
    fig, axes = plt.subplots(2, 2, figsize=(18, 12))
    fig.suptitle(
        "MRPRO V6.3: AUDITORÍA DE RESONANCIA Y SUCCIÓN",
        fontsize=22,
        color="#00ffcc",
        fontweight="bold",
    )

    # Diccionario blindado: Llaves en int y str para evitar errores de mapeo
    palette = {
        0: "#1a1a1a",
        "0": "#1a1a1a",
        1: "#2a2a2a",
        "1": "#2a2a2a",
        2: "#3a3a3a",
        "2": "#3a3a3a",
        3: "#0055ff",
        "3": "#0055ff",
        4: "#ffcc00",
        "4": "#ffcc00",
        5: "#ff0055",
        "5": "#ff0055",
        6: "#00ffff",
        "6": "#00ffff",
    }

    # PANEL A: Conflict Map (IA vs Geometría)
    sns.scatterplot(
        data=df,
        x="ai_score",
        y="geo_score",
        hue="hits",
        palette=palette,
        s=100,
        alpha=0.6,
        ax=axes[0, 0],
    )
    axes[0, 0].set_title("1. Mapa de Conflicto: IA vs Geo Score", color="cyan")
    axes[0, 0].axhline(
        df["geo_score"].median(), color="white", linestyle="--", alpha=0.3
    )

    # PANEL B: Distribución de Hits (Frecuencia)
    # Corregido: Asignamos x a hue y desactivamos la leyenda
    counts = df["hits"].value_counts().sort_index()
    sns.barplot(
        x=counts.index,
        y=counts.values,
        hue=counts.index,
        palette=palette,
        ax=axes[0, 1],
        legend=False,
    )
    axes[0, 1].set_title("2. Distribución de Éxitos (108 Sorteos)", color="cyan")
    for i, v in enumerate(counts.values):
        axes[0, 1].text(i, v + 0.2, str(v), ha="center", color="white")

    # PANEL C: Succión de Rango (Escala Logarítmica Invertida)
    # Corregido: Asignamos hue="hits" para cumplir con v0.14.0
    sns.boxenplot(
        data=df,
        x="hits",
        y="rank",
        hue="hits",
        palette=palette,
        ax=axes[1, 0],
        legend=False,
    )
    axes[1, 0].set_yscale("log")
    axes[1, 0].invert_yaxis()
    axes[1, 0].set_title("3. Presión de Succión (Rango Invertido)", color="cyan")

    # PANEL D: Proximidad del 'Outlier'
    sns.stripplot(
        data=df,
        x="hits",
        y="proximity",
        hue="hits",
        palette=palette,
        ax=axes[1, 1],
        alpha=0.5,
        legend=False,
    )
    axes[1, 1].set_yscale("log")
    axes[1, 1].invert_yaxis()
    axes[1, 1].set_title("4. Proximidad al Jackpot", color="cyan")

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    output_path = "data/forensic_v6_3.png"
    plt.savefig(output_path, dpi=300)
    print(f"✅ Visualización generada exitosamente en: {output_path}")
