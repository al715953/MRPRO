# src/data_access/visualizer.py

import json
import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
# Importamos la ruta centralizada para asegurar compatibilidad en Mac
from src.data_access.config import DATA_FOLDER

def run_forensic_visualization(json_path=None):
    """
    VISUALIZER V6.3: Auditor de Resonancia (Mac Optimized).
    Sincronizado con PerformanceTracker para localizar los datos de backtest.
    """
    # Si no se provee ruta, usamos la ubicación centralizada en la carpeta data
    if json_path is None:
        json_path = os.path.join(DATA_FOLDER, "backtest_results.json")

    print(f"\n[bold cyan]📊 INICIANDO ESTACIÓN DE VISUALIZACIÓN[/bold cyan]")

    if not os.path.exists(json_path):
        print(f"❌ Error: Archivo de resultados no encontrado.")
        print(f"   Se buscó en: {json_path}")
        print("   Asegúrate de ejecutar un Backtest (Opción 6) primero para generar los datos.")
        return

    try:
        with open(json_path, "r") as f:
            data = json.load(f)
        df = pd.DataFrame(data)
    except Exception as e:
        print(f"❌ Error al cargar JSON en Mac: {e}")
        return

    if df.empty:
        print("⚠️ El archivo de resultados está vacío.")
        return

    # Aseguramos que 'hits' sea entero para consistencia en el mapeo de colores
    df["hits"] = df["hits"].astype(int)

    # 1. Configuración de Estética Forense (Modo Oscuro)
    plt.style.use("dark_background")
    fig, axes = plt.subplots(2, 2, figsize=(18, 12))
    fig.suptitle(
        "MRPRO V6.3: AUDITORÍA DE RESONANCIA Y SUCCIÓN",
        fontsize=22,
        color="#00ffcc",
        fontweight="bold",
    )

    # Diccionario de colores blindado para los niveles de aciertos (0 a 6)
    palette = {
        0: "#1a1a1a", 1: "#2a2a2a", 2: "#3a3a3a",
        3: "#0055ff", 4: "#ffcc00", 5: "#ff0055", 6: "#00ffff"
    }

    # PANEL A: Conflict Map (IA Score vs Geometría)
    sns.scatterplot(
        data=df, x="ai_score", y="geo_score", hue="hits",
        palette=palette, s=100, alpha=0.6, ax=axes[0, 0]
    )
    axes[0, 0].set_title("1. Mapa de Conflicto: IA vs Geo Score", color="cyan")
    axes[0, 0].axhline(df["geo_score"].median(), color="white", linestyle="--", alpha=0.3)

    # PANEL B: Distribución de Hits (Frecuencia de aciertos)
    counts = df["hits"].value_counts().sort_index()
    sns.barplot(
        x=counts.index, y=counts.values, hue=counts.index,
        palette=palette, ax=axes[0, 1], legend=False
    )
    axes[0, 1].set_title(f"2. Distribución de Éxitos ({len(df)} Sorteos)", color="cyan")
    for i, v in enumerate(counts.values):
        axes[0, 1].text(i, v + 0.2, str(v), ha="center", color="white")

    # PANEL C: Succión de Rango (Escala Logarítmica Invertida)
    # Analiza qué tan cerca del Top 1 estuvieron los ganadores reales
    sns.boxenplot(
        data=df, x="hits", y="rank", hue="hits",
        palette=palette, ax=axes[1, 0], legend=False
    )
    axes[1, 0].set_yscale("log")
    axes[1, 0].invert_yaxis()
    axes[1, 0].set_title("3. Presión de Succión (Rango Invertido)", color="cyan")

    # PANEL D: Proximidad al Jackpot
    sns.stripplot(
        data=df, x="hits", y="proximity", hue="hits",
        palette=palette, ax=axes[1, 1], alpha=0.5, legend=False
    )
    axes[1, 1].set_yscale("log")
    axes[1, 1].invert_yaxis()
    axes[1, 1].set_title("4. Proximidad al Jackpot", color="cyan")

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    
    # Guardamos el resultado en la carpeta data centralizada
    output_path = os.path.join(DATA_FOLDER, "forensic_analysis.png")
    plt.savefig(output_path, dpi=300)
    
    print(f"✅ Visualización generada exitosamente en: {output_path}")
    
    # Intentar abrir la imagen automáticamente en Mac
    try:
        os.system(f"open {output_path}")
    except:
        pass

if __name__ == "__main__":
    run_forensic_visualization()