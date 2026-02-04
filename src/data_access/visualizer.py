# src/data_access/visualizer.py

import json
import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from src.data_access.config import DATA_FOLDER

def run_forensic_visualization(json_path=None):
    """
    VISUALIZER V7.21: Tablero Táctico Magneto (Clean UI).
    Versión optimizada para Mac (sin emojis en títulos para evitar Glyph Warnings).
    """
    if json_path is None:
        json_path = os.path.join(DATA_FOLDER, "backtest_results.json")

    print(f"\n📊 GENERANDO TABLERO TACTICO (V7.21)")

    if not os.path.exists(json_path):
        print(f"❌ Error: No se encontró {json_path}")
        return

    try:
        with open(json_path, "r") as f:
            data = json.load(f)
        df = pd.DataFrame(data)
    except Exception as e:
        print(f"❌ Error leyendo JSON: {e}")
        return

    if df.empty:
        print("⚠️ El JSON de resultados está vacío.")
        return

    # --- LIMPIEZA DE DATOS ---
    numeric_cols = ['ai_score', 'geo_score', 'rank', 'hits', 'univ_size']
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
        else:
            df[col] = 0 

    # --- CONFIGURACIÓN DEL LIENZO ---
    sns.set_theme(style="darkgrid")
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    
    run_tag = df['tag'].iloc[0] if 'tag' in df.columns else "Unknown Mission"
    fig.suptitle(f'Tablero Tactico MRPRO - Mision: {run_tag}', fontsize=14, color='#1f77b4', weight='bold')

    # --- PANEL 1: RADAR DE CAZA (AI vs GEO) ---
    sns.scatterplot(
        data=df, 
        x='ai_score', 
        y='geo_score', 
        hue='hits', 
        palette="viridis",
        size='hits',
        sizes=(20, 200),
        style='hits',
        ax=axes[0]
    )
    # Títulos limpios (Sin emojis para evitar Warnings)
    axes[0].set_title('Radar de Impacto: AI vs Geo', color='navy')
    axes[0].set_xlabel('Inteligencia Artificial (0-1)')
    axes[0].set_ylabel('Resonancia Geometrica (0-1)')
    axes[0].axhline(0.5, color='red', linestyle='--', alpha=0.3)
    axes[0].axvline(0.5, color='red', linestyle='--', alpha=0.3)

    # --- PANEL 2: DINÁMICA DEL UNIVERSO ---
    if 'univ_size' in df.columns and df['univ_size'].sum() > 0:
        sns.lineplot(data=df, x='draw_id', y='univ_size', color='green', marker='o', ax=axes[1])
        axes[1].set_title('Dinamica del Universo (Sniper)', color='green')
        axes[1].set_ylabel('Tickets en Juego')
    else:
        axes[1].text(0.5, 0.5, "Datos de Universo no disponibles", ha='center')

    # --- PANEL 3: PROFUNDIDAD DEL RANKING ---
    sns.barplot(data=df, x='draw_id', y='rank', hue='hits', dodge=False, palette="magma", ax=axes[2])
    axes[2].set_yscale('log') 
    axes[2].set_title('Profundidad del Ganador (Rank Log)', color='purple')
    axes[2].set_ylabel('Ranking (Escala Log)')
    axes[2].axhline(5000, color='red', linestyle='--', label='Limite de Compra (5k)')
    axes[2].legend(loc='upper right', bbox_to_anchor=(1, 1))

    plt.tight_layout()
    
    output_filename = "tactical_dashboard_v7.png"
    output_path = os.path.join(DATA_FOLDER, output_filename)
    plt.savefig(output_path, dpi=150)
    print(f"🖼️  Tablero generado: {output_path}")
    plt.close()

if __name__ == "__main__":
    run_forensic_visualization()