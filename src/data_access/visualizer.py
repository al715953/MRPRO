import json
import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from src.data_access.config import DATA_FOLDER


def run_forensic_visualization():
    """
    DASHBOARD V4.5.2: Sniper X-Ray.
    Optimizado para detectar colapsos de energía y validar la Trifecta.
    """
    # Ruta maestra del JSON generado por el PerformanceTracker
    path = os.path.join(DATA_FOLDER, "backtest_results.json")

    if not os.path.exists(path):
        # Fallback para rutas absolutas en entornos UHPC
        path = r"D:\Python\MRPro\data\backtest_results.json"

    if not os.path.exists(path):
        print(f"❌ Error: Archivo de resultados no encontrado en {path}")
        return

    with open(path, "r") as f:
        data = json.load(f)

    df = pd.DataFrame(data)
    if df.empty:
        print("⚠️ El JSON está vacío. Corra el backtest nuevamente.")
        return

    # Limpieza de columnas para compatibilidad total
    df.columns = [c.lower().strip() for c in df.columns]

    # Mapeo de compatibilidad: Si no hay 'hybrid_score', usamos 'ai_score'
    if "hybrid_score" not in df.columns and "ai_score" in df.columns:
        df["hybrid_score"] = df["ai_score"]

    # --- CONFIGURACIÓN SNIPER V4.5 ---
    sns.set_theme(style="darkgrid")
    custom_palette = {3: "#A9A9A9", 4: "#FFD700", 5: "#228B22", 6: "#00FFFF"}
    fig, axes = plt.subplots(3, 2, figsize=(22, 22))
    fig.suptitle(
        f"Dashboard Sniper V4.5.2 - Análisis de Fuga de Energía\nArchivo: {os.path.basename(path)}",
        fontsize=26,
        fontweight="bold",
        color="#1f77b4",
        y=0.98,
    )

    # 1. DISTRIBUCIÓN DE RANKS (Zoom Zona de Captura)
    sns.histplot(
        df[df["rank"] <= 5000]["rank"],
        bins=50,
        kde=True,
        ax=axes[0, 0],
        color="#7fcdbb",
    )
    axes[0, 0].set_title("1. Distribución de Ranks (Foco Top 5000)", fontsize=16)

    # 2. ANÁLISIS DEL COLAPSO (0.15 Diagnostic)
    sns.scatterplot(
        data=df,
        x="hybrid_score",
        y="rank",
        hue="hits",
        palette=custom_palette,
        s=150,
        edgecolor="black",
        alpha=0.7,
        ax=axes[0, 1],
    )
    axes[0, 1].set_yscale("log")
    axes[0, 1].invert_yaxis()
    # Línea crítica del bono de alerta
    axes[0, 1].axvline(x=0.15, color="red", linestyle="--", label="Bono Mutante (0.15)")
    axes[0, 1].set_title("2. Energía Híbrida vs Rank (Muro de los 0.15)", fontsize=16)
    axes[0, 1].legend()

    # 3. DESGLOSE DE LA TRIFECTA (Alpha, Beta, Omega)
    # Buscamos si el JSON tiene el ADN desglosado
    cols_expert = ["score_alpha", "score_beta", "score_omega"]
    if all(c in df.columns for c in cols_expert) and df["score_alpha"].sum() > 0:
        recent = df.tail(15)
        recent[cols_expert].plot(
            kind="bar",
            stacked=True,
            ax=axes[1, 0],
            color=["#3498db", "#e67e22", "#2ecc71"],
        )
        axes[1, 0].set_title(
            "3. Desglose de Energía: Alpha | Beta | Omega", fontsize=16
        )
        axes[1, 0].set_xticklabels(recent["draw_id"], rotation=45)
    else:
        axes[1, 0].text(
            0.5,
            0.5,
            "SIN DATOS DE TRIFECTA\n(Verifique snapshot en Selector)",
            ha="center",
            va="center",
            fontsize=14,
            color="red",
        )
        axes[1, 0].set_title("3. Desglose de Energía (Inactivo)", fontsize=16)

    # 4. SATURACIÓN TEMPORAL
    sns.scatterplot(
        data=df,
        x="draw_id",
        y="rank",
        hue="hits",
        palette=custom_palette,
        s=120,
        marker="X",
        ax=axes[1, 1],
    )
    axes[1, 1].set_yscale("log")
    axes[1, 1].invert_yaxis()
    axes[1, 1].set_title("4. Saturación: Evolución de Ranks", fontsize=16)

    # 5. CORRELACIÓN IA VS PROXIMIDAD (Order 2)
    sns.regplot(
        data=df,
        x="hybrid_score",
        y="proximity",
        order=2,
        ax=axes[2, 0],
        scatter_kws={"alpha": 0.4, "s": 100},
        line_kws={"color": "red"},
    )
    axes[2, 0].set_title("5. Correlación: Confianza vs Distancia Crítica", fontsize=16)

    # 6. MAPA DE CALOR DE HITS (AI vs GEO)
    if "geo_score" in df.columns:
        sns.scatterplot(
            data=df,
            x="hybrid_score",
            y="geo_score",
            hue="hits",
            size="rank",
            sizes=(20, 200),
            palette=custom_palette,
            ax=axes[2, 1],
        )
        axes[2, 1].set_title(
            "6. Resonancia Híbrida: AI Score vs Geo Score", fontsize=16
        )
    else:
        sns.lineplot(data=df, x="draw_id", y="proximity", marker="o", ax=axes[2, 1])
        axes[2, 1].set_title("6. Evolución de Proximidad", fontsize=16)

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])

    # Guardar reporte de alta fidelidad
    output_img = os.path.join(DATA_FOLDER, "diagnostico_sniper_v4.5.jpg")
    plt.savefig(output_img, dpi=300)
    print(f"✅ Dashboard V4.5.2 generado: {output_img}")
    plt.show()


if __name__ == "__main__":
    run_forensic_visualization()
