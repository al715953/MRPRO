import pandas as pd
import numpy as np


def run_forensics():
    print("--- INICIANDO AUTOPSIA FORENSE V9 ---")
    try:
        # Carga del log
        df = pd.read_csv("src\data\detailed_forensic_log.csv")

        # 1. Auditoría del Ranking Engine (¿El premio está realmente ahí?)
        print("\n[1] CALIDAD DEL MOTOR DE RANKING (Resonance Engine)")
        mean_rank = df["Rank_Winner"].mean()
        median_rank = df["Rank_Winner"].median()
        top_100_hits = (df["Rank_Winner"] <= 100).sum()
        top_500_hits = (df["Rank_Winner"] <= 500).sum()
        total_draws = len(df)

        print(f"   Total Sorteos Analizados: {total_draws}")
        print(f"   Rank Promedio del Ganador: #{mean_rank:.2f}")
        print(f"   Rank Mediano del Ganador:  #{median_rank:.0f}")
        print(
            f"   Ganadores en Top 100:      {top_100_hits} ({top_100_hits/total_draws:.1%})"
        )
        print(
            f"   Ganadores en Top 500:      {top_500_hits} ({top_500_hits/total_draws:.1%})"
        )

        if median_rank > 500:
            print(
                "   >>> CONCLUSIÓN: EL MOTOR FALLA. El premio no está en el Top 500. El Selector es inocente."
            )
        else:
            print(
                "   >>> CONCLUSIÓN: EL MOTOR FUNCIONA. El premio está ahí. El Selector V9 es el culpable."
            )

        # 2. Auditoría del Selector (¿Por qué no lo atrapamos?)
        # Asumimos que 'Selected_Ranks' es una cadena o lista. Si es cadena, la parseamos.
        # Calcularemos la distancia mínima entre lo que jugamos y el ganador real.

        print("\n[2] PRECISIÓN DEL SELECTOR (Holo-Cover)")
        # Filtramos solo los casos donde el Ranking Engine hizo su trabajo (Premio en Top 500)
        valid_cases = df[df["Rank_Winner"] <= 500].copy()

        if len(valid_cases) == 0:
            print("   No hay casos válidos en Top 500 para analizar el selector.")
            return

        # Función para encontrar el Rank jugado más cercano al Rank Ganador
        def get_min_dist(row):
            winner = row["Rank_Winner"]
            # Limpieza básica de la cadena de lista si es necesario
            try:
                selected = (
                    eval(str(row["Selected_Ranks"]))
                    if isinstance(row["Selected_Ranks"], str)
                    else row["Selected_Ranks"]
                )
                if not isinstance(selected, list):
                    return 9999
                # Distancia absoluta mínima
                dists = [abs(s - winner) for s in selected if s > 0]
                return min(dists) if dists else 9999
            except:
                return 9999

        valid_cases["Min_Dist"] = valid_cases.apply(get_min_dist, axis=1)
        avg_dist = valid_cases["Min_Dist"].mean()

        print(f"   En los casos donde el premio estaba en Top 500:")
        print(
            f"   Distancia Promedio (Rank Jugado vs Rank Ganador): {avg_dist:.2f} puestos"
        )

        if avg_dist < 20:
            print(
                "   >>> FALLO DE PRECISIÓN FINA: Estamos cerca, pero no exactos (Falta 'Jitter')."
            )
        else:
            print(
                "   >>> FALLO ESTRUCTURAL: Estamos buscando en el vecindario equivocado del Top 500."
            )

    except Exception as e:
        print(f"ERROR CRÍTICO: {e}")
        print("Asegúrate de que 'detailed_forensic_log.csv' está en la carpeta data.")


if __name__ == "__main__":
    run_forensics()
