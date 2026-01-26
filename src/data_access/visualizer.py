import json
import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from src.data_access.config import DATA_FOLDER


def run_forensic_visualization():
    """
    VISUALIZER V5.1: Conflict & Dilution Auditor.
    Evolucionado para detectar por qué los expertos anulan la señal del Jackpot.
    Resuelve el ValueError de mapeo de tipos en la paleta de Seaborn.
    """
    # 1. LOCALIZACIÓN DE TELEMETRÍA
    log_path = "data/detailed_forensic_log.csv"
    json_path = "data/backtest_results.json"

    if os.path.exists(log_path):
        df = pd.read_csv(log_path)
        source = "CSV (Detallado)"
    elif os.path.exists(json_path):
        with open(json_path, "r") as f:
            data = json.load(f)
            df = pd.DataFrame(data)
        source = "JSON (Resumen)"
    else:
        print("❌ Error: No se encontraron archivos de telemetría en la carpeta data.")
        return

    # 2. LIMPIEZA Y BLINDAJE DE DATOS
    df.columns = [c.lower().strip() for c in df.columns]

    # CRÍTICO: Convertir hits a entero para que coincida con las llaves de la paleta
    if "hits" in df.columns:
        df["hits"] = pd.to_numeric(df["hits"], errors="coerce").fillna(0).astype(int)
    else:
        print("❌ Error: La columna 'hits' no existe en los logs.")
        return

    # 3. CONFIGURACIÓN ESTÉTICA (Dark Mode para Ingeniería)
    plt.style.use("dark_background")
    fig, axes = plt.subplots(2, 3, figsize=(24, 16))
    fig.suptitle(
        f"AUDITORÍA MRPRO V5.1 - Diagnóstico de Señal\nFuente: {source} | Foco: Ruptura del Muro de Energía",
        fontsize=28,
        color="#00d4ff",
        fontweight="bold",
    )

    # Paleta técnica con llaves enteras
    palette = {
        0: "#2a2a2a",
        1: "#3a3a3a",
        2: "#4a4a4a",
        3: "#5a5a5a",
        4: "#ffcc00",
        5: "#ff3300",
        6: "#00ffff",
    }

    # --- PANEL 1: MAPA DE CONFLICTO (ALPHA VS OMEGA) ---
    # Detecta si el modelo ancla está matando la señal del cazador
    x_col = "score_alpha" if "score_alpha" in df.columns else "ai_score"
    y_col = "score_omega" if "score_omega" in df.columns else "geo_score"

    sns.scatterplot(
        data=df,
        x=x_col,
        y=y_col,
        hue="hits",
        palette=palette,
        s=150,
        alpha=0.8,
        ax=axes[0, 0],
    )
    axes[0, 0].set_title(
        f"1. Espacio de Conflicto: {x_col} vs {y_col}", fontsize=16, color="cyan"
    )
    axes[0, 0].grid(True, alpha=0.1)

    # --- PANEL 2: ESPECTRO DE ENERGÍA (IDENTIFICADOR DE MUROS) ---
    sns.histplot(
        data=df,
        x="ai_score",
        hue="hits",
        multiple="stack",
        bins=40,
        palette=palette,
        ax=axes[0, 1],
    )
    axes[0, 1].axvline(
        0.15, color="#00ffff", linestyle="--", label="Bono Mutante (0.15)"
    )
    axes[0, 1].axvline(
        0.1028, color="#ff00ff", linestyle=":", label="Muro de Residuos (0.1028)"
    )
    axes[0, 1].set_title(
        "2. Espectro de Energía: Identificación de Muros", fontsize=16, color="cyan"
    )
    axes[0, 1].legend()

    # --- PANEL 3: EFICIENCIA DE RANKING (BOXENPLOT CORREGIDO) ---
    # Usamos hue='hits' para evitar el Warning de Seaborn v0.13
    sns.boxenplot(
        data=df,
        x="hits",
        y="rank",
        hue="hits",
        palette=palette,
        legend=False,
        ax=axes[0, 2],
    )
    axes[0, 2].set_yscale("log")
    axes[0, 2].invert_yaxis()
    axes[0, 2].set_title(
        "3. Profundidad de Captura (Rank Log Scale)", fontsize=16, color="cyan"
    )

    # --- PANEL 4: CRONOLOGÍA DE RESONANCIA ---
    if "draw_id" in df.columns:
        sns.lineplot(
            data=df, x="draw_id", y="ai_score", color="white", alpha=0.2, ax=axes[1, 0]
        )
        sns.scatterplot(
            data=df,
            x="draw_id",
            y="ai_score",
            hue="hits",
            palette=palette,
            s=100,
            ax=axes[1, 0],
        )
        axes[1, 0].set_title(
            "4. Evolución de Energía por Sorteo", fontsize=16, color="cyan"
        )

    # --- PANEL 5: ENERGÍA VS PROXIMIDAD ---
    if "proximity" in df.columns:
        sns.regplot(
            data=df,
            x="ai_score",
            y="proximity",
            scatter=False,
            color="#00d4ff",
            ax=axes[1, 1],
        )
        sns.scatterplot(
            data=df,
            x="ai_score",
            y="proximity",
            hue="hits",
            palette=palette,
            s=100,
            ax=axes[1, 1],
        )
        axes[1, 1].set_title(
            "5. Correlación: Energía vs Proximidad Real", fontsize=16, color="cyan"
        )

    # --- PANEL 6: HISTOGRAMA DE RENDIMIENTO ---
    hits_count = df["hits"].value_counts().sort_index()
    hits_count.plot(kind="bar", color="#00d4ff", ax=axes[1, 2])
    axes[1, 2].set_title(
        "6. Distribución Total de Aciertos (Hits)", fontsize=16, color="cyan"
    )
    for i, v in enumerate(hits_count):
        axes[1, 2].text(
            i, v + 0.1, str(v), ha="center", color="white", fontweight="bold"
        )

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])

    # Guardado de alta fidelidad para el equipo
    output_img = os.path.join(DATA_FOLDER, "diagnostico_v5_conflict.png")
    plt.savefig(output_img, dpi=300)
    print(f"✅ Reporte Visual V5.1 generado con éxito: {output_img}")


if __name__ == "__main__":
    run_forensic_visualization()
