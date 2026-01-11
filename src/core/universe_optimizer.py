import numpy as np
import time
from collections import Counter
import itertools
from typing import List, Dict
from colorama import Fore, Style

from src.domain.dtos import DrawHistoryDTO, PredictionConfigDTO
from src.strategies.universe_reduction import UniverseReductionStrategy

# Reutilizamos lógica interna de la estrategia para no repetir código,
# pero la adaptamos para hacer barridos rápidos.


class UniverseOptimizer:
    def __init__(self, history: DrawHistoryDTO):
        self.history = history

    def optimize(self, base_config: PredictionConfigDTO, lookback: int = 15):
        """
        Busca el mejor 'Quality Percentile' para maximizar cobertura
        minimizando el tamaño del archivo.
        """
        print(
            f"\n{Fore.MAGENTA}🧠 INICIANDO OPTIMIZACIÓN DE UNIVERSO...{Style.RESET_ALL}"
        )
        print(
            f"Analizando los últimos {lookback} sorteos para calibrar la Red de Pesca."
        )

        # Parametros a probar (Niveles de Exigencia)
        # 80 = Muy laxo (Universo grande), 98 = Muy estricto (Universo pequeño)
        percentile_grid = [80, 85, 88, 90, 92, 94, 96]

        # Resultados: {percentil: {'hits': 0, 'total_size': 0}}
        results = {
            p: {"hits": 0, "misses": 0, "total_size": 0, "jackpots": 0}
            for p in percentile_grid
        }

        # Preparar datos históricos
        full_data = list(
            zip(
                self.history.dates, self.history.winning_numbers, self.history.concursos
            )
        )
        # Ordenamos cronológicamente
        full_data.sort(key=lambda x: x[2])

        start_index = len(full_data) - lookback
        if start_index < 0:
            start_index = 0

        # Bucle de Sorteos
        for i in range(start_index, len(full_data)):
            target_date, target_draw, target_id = full_data[i]
            target_set = set(target_draw[:6])  # Set del ganador real

            # Contexto histórico para ese momento (Viaje en el tiempo)
            past_winning_numbers = [x[1] for x in full_data[:i]]

            # --- SIMULACIÓN LIGERA DE LA ESTRATEGIA ---
            # 1. Pesos y Clusters (Entrenamiento con datos pasados)
            freq_counter = Counter()
            cluster_counter = Counter()
            for draw in past_winning_numbers:
                freq_counter.update(draw[:6])
                for pair in itertools.combinations(sorted(draw[:6]), 2):
                    cluster_counter[pair] += 1

            total_balls = base_config.total_balls
            weights = [freq_counter.get(n, 1) + 1 for n in range(1, total_balls + 1)]
            weights_np = np.array(weights, dtype=float)
            weights_np /= weights_np.sum()
            clusters_dict = dict(cluster_counter)

            # 2. Generación Cruda (Una sola vez masiva)
            # Generamos un pool grande para luego filtrarlo con distintos percentiles
            # Usamos menos cantidad que la real para que la optimización sea rápida (simulada)
            sim_size = 500000
            pool_nums = np.arange(1, total_balls + 1)
            raw_batch = np.random.choice(
                pool_nums,
                size=(sim_size, base_config.ticket_size),
                replace=True,
                p=weights_np,
            )

            # 3. Calcular Scores de ese lote
            batch_scores = []
            batch_tickets = []

            for row in raw_batch:
                uniques = np.unique(row)
                if len(uniques) != base_config.ticket_size:
                    continue
                ticket = tuple(sorted(uniques.tolist()))

                # Scoring rápido
                score = 0
                for pair in itertools.combinations(ticket, 2):
                    if pair in clusters_dict:
                        score += clusters_dict[pair]

                if score > 0:
                    batch_scores.append(score)
                    batch_tickets.append(ticket)

            # Convertir a numpy para percentiles rápidos
            scores_np = np.array(batch_scores)

            print(
                f"  > Sorteo {target_id}: Evaluando {len(batch_tickets)} candidatos base..."
            )

            # --- PROBAR CADA PERCENTIL ---
            for p in percentile_grid:
                # Calcular corte
                threshold = np.percentile(scores_np, p)

                # Filtrar ganadores virtuales
                # Verificamos si el ganador real hubiera pasado el corte
                # Para saber esto, calculamos el score del GANADOR REAL
                real_winner_score = 0
                for pair in itertools.combinations(sorted(tuple(target_set)), 2):
                    if pair in clusters_dict:
                        real_winner_score += clusters_dict[pair]

                # El ganador real estaría en el universo si:
                # A) Su score es mayor al umbral del percentil
                # B) (Simulado) Asumimos que si cumple el score, el generador masivo eventualmente lo produciría.

                # Estimamos tamaño del universo resultante
                passing_count = np.sum(scores_np >= threshold)
                projected_universe_size = (
                    passing_count / len(scores_np)
                ) * 2500000  # Proyección a escala real

                # Check de Cobertura:
                # ¿El ticket ganador tenía calidad suficiente para entrar en este percentil?
                hit = False
                jackpot = False
                if real_winner_score >= threshold:
                    hit = True
                    jackpot = True  # En teoría entró en la red

                # Guardar métricas
                results[p]["total_size"] += projected_universe_size
                if hit:
                    results[p]["hits"] += 1
                    results[p]["jackpots"] += 1
                else:
                    results[p]["misses"] += 1

        # --- REPORTE FINAL ---
        print("\n" + "=" * 60)
        print(f"📊 RESULTADOS DE OPTIMIZACIÓN DE RED (Promedio {lookback} Sorteos)")
        print("=" * 60)
        print(
            f"{'PERCENTIL':<10} | {'COBERTURA':<10} | {'TAMAÑO PROM.':<15} | {'CALIFICACIÓN':<10}"
        )
        print("-" * 60)

        best_p = 88
        best_score = -float("inf")

        for p in percentile_grid:
            hits = results[p]["hits"]
            coverage = (hits / lookback) * 100
            avg_size = results[p]["total_size"] / lookback

            # Fórmula de Score: Queremos cobertura alta, penalizando tamaño excesivo
            # Prioridad absoluta a cobertura > 90%
            score = coverage * 100 - (avg_size / 5000)

            color = Fore.WHITE
            if coverage >= 90:
                color = Fore.GREEN
            if coverage < 80:
                color = Fore.RED

            print(
                f"{p:<10} | {color}{coverage:5.1f}%{Style.RESET_ALL}    | {avg_size:10,.0f} tkts  | {score:6.1f}"
            )

            if score > best_score:
                best_score = score
                best_p = p

        print("=" * 60)
        print(f"🏆 MEJOR CONFIGURACIÓN RECOMENDADA: PERCENTIL {best_p}")
        print(f"   (Equilibrio ideal entre atrapar al ganador y no generar basura)")

        return best_p
