# src/data_access/visualizer.py

import json
import os
import pandas as pd
from src.data_access.config import DATA_FOLDER


def run_forensic_visualization(json_path=None):
    """
    VISUALIZER V7.21: Tablero Táctico Magneto (Clean UI).
    Corregido para procesar la estructura de datos Omega Stride V15.
    """
    try:
        import matplotlib.pyplot as plt
        import seaborn as sns
    except Exception as e:
        print(f"❌ Dependencias de visualización no disponibles: {e}")
        print("ℹ️ Instala/actualiza matplotlib y seaborn para usar esta opción.")
        return

    if json_path is None:
        json_path = os.path.join(DATA_FOLDER, "backtest_results.json")

    print(f"\n📊 GENERANDO TABLERO TACTICO (V7.21)")

    if not os.path.exists(json_path):
        print(f"❌ Error: No se encontró {json_path}")
        return

    try:
        with open(json_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        # --- FIX V15: Extracción de la lista de detalles ---
        if "forensic_details" in data:
            df = pd.DataFrame(data["forensic_details"])
        else:
            # Fallback si el JSON no tiene la estructura esperada
            df = pd.DataFrame(data)

    except Exception as e:
        print(f"❌ Error leyendo JSON: {e}")
        return

    if df.empty:
        print("⚠️ El JSON de resultados está vacío o no contiene 'forensic_details'.")
        return

    # --- LIMPIEZA Y NORMALIZACIÓN DE TIPOS ---
    numeric_cols = ["ai_score", "geo_score", "rank", "hits", "univ_size"]
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0)

    # Configuración de estilo
    sns.set_theme(style="darkgrid")
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    fig.suptitle(f"Reporte de Inteligencia: {data.get('version', 'V15')}", fontsize=16)

    # --- PANEL 1: RADAR DE IMPACTO ---
    sns.scatterplot(
        data=df,
        x="ai_score",
        y="geo_score",
        hue="hits",
        palette="viridis",
        size="hits",
        sizes=(50, 200),
        ax=axes[0],
    )
    axes[0].set_title("Radar de Impacto: AI vs Geo", color="navy")
    axes[0].set_xlabel("Inteligencia Artificial (0-1)")
    axes[0].set_ylabel("Resonancia Geometrica (0-1)")
    axes[0].axhline(0.5, color="red", linestyle="--", alpha=0.3)
    axes[0].axvline(0.5, color="red", linestyle="--", alpha=0.3)

    # --- PANEL 2: DINÁMICA DEL UNIVERSO ---
    # Usamos el índice como eje X si draw_id no es único o no está presente
    x_axis = "draw_id" if "draw_id" in df.columns else df.index

    if "univ_size" in df.columns and df["univ_size"].sum() > 0:
        sns.lineplot(
            data=df, x=x_axis, y="univ_size", color="green", marker="o", ax=axes[1]
        )
        axes[1].set_title("Dinamica del Universo (Sniper)", color="green")
        axes[1].set_ylabel("Tickets en Juego")
    else:
        axes[1].text(0.5, 0.5, "Datos de Universo no disponibles", ha="center")

    # --- PANEL 3: PROFUNDIDAD DEL RANKING ---
    sns.barplot(
        data=df,
        x=x_axis,
        y="rank",
        hue="hits",
        dodge=False,
        palette="magma",
        ax=axes[2],
    )
    axes[2].set_yscale("log")
    axes[2].set_title("Profundidad del Ganador (Rank Log)", color="purple")
    axes[2].set_ylabel("Ranking (Escala Log)")

    plt.tight_layout()

    # Persistencia del tablero
    output_plot = os.path.join(DATA_FOLDER, "tactical_dashboard.png")
    plt.savefig(output_plot)
    print(f"✅ TABLERO GUARDADO EN: {output_plot}")
    plt.show()
