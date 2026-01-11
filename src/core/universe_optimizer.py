import numpy as np
import itertools
from collections import Counter
from typing import List, Tuple
from colorama import Fore, Style

from src.domain.dtos import DrawHistoryDTO, PredictionConfigDTO, CandidateCombination

# --- IMPORTAMOS LOS FILTROS PARA QUE LA SIMULACIÓN SEA REAL ---
from src.core.filters.pipeline import FilterPipeline
from src.core.filters.implementations.geometric import SumRangeFilter
from src.core.filters.implementations.probabilistic import ParityFilter, PrimeFilter
from src.core.filters.implementations.arithmetic import ACValueFilter

try:
    from src.core.filters.implementations.structural import (
        ConsecutiveFilter,
        QuadrantFilter,
        LastDigitFilter,
    )
except ImportError:
    ConsecutiveFilter = None
    QuadrantFilter = None
    LastDigitFilter = None


class UniverseOptimizer:
    def __init__(self, history: DrawHistoryDTO):
        self.history = history

    def optimize(self, base_config: PredictionConfigDTO, lookback: int = 15):
        """
        Busca el mejor 'Quality Percentile' simulando EXACTAMENTE
        lo que hace la estrategia UniverseReduction (Filtros + Score).
        """
        print(
            f"\n{Fore.MAGENTA}🧠 INICIANDO OPTIMIZACIÓN (REALISTA)...{Style.RESET_ALL}"
        )
        print(
            f"Calibrando con los últimos {lookback} sorteos y filtros estructurales activados."
        )

        percentile_grid = [75, 78, 80, 82, 85, 88, 90]
        results = {p: {"hits": 0, "total_size": 0} for p in percentile_grid}

        # Preparar historia
        full_data = list(
            zip(
                self.history.dates, self.history.winning_numbers, self.history.concursos
            )
        )
        full_data.sort(key=lambda x: x[2])

        start_index = max(0, len(full_data) - lookback)

        # Configuración de filtros (Simulamos los defaults del sistema)
        # Nota: Usamos valores "seguros" para la simulación
        filter_cfg = {
            "sum_min": 90,
            "sum_max": 200,
            "even_min": 2,
            "even_max": 4,
            "ac_min": 5,
        }

        for i in range(start_index, len(full_data)):
            target_date, target_draw, target_id = full_data[i]
            target_set = set(target_draw[:6])
            past_data = [x[1] for x in full_data[:i]]

            # 1. Análisis de Frecuencia y Clusters (Entrenamiento)
            freq_counter = Counter()
            cluster_counter = Counter()
            for draw in past_data:
                freq_counter.update(draw[:6])
                for pair in itertools.combinations(sorted(draw[:6]), 2):
                    cluster_counter[pair] += 1

            total_balls = base_config.total_balls
            weights = [freq_counter.get(n, 1) + 1 for n in range(1, total_balls + 1)]
            weights_np = np.array(weights, dtype=float)
            weights_np /= weights_np.sum()
            clusters_dict = dict(cluster_counter)

            # 2. Construcción del Pipeline (IGUAL QUE EN LA ESTRATEGIA REAL)
            pipeline = FilterPipeline()
            pipeline.add_filter(
                SumRangeFilter(filter_cfg["sum_min"], filter_cfg["sum_max"])
            )
            pipeline.add_filter(
                ParityFilter(filter_cfg["even_min"], filter_cfg["even_max"])
            )
            pipeline.add_filter(ACValueFilter(filter_cfg["ac_min"]))
            pipeline.add_filter(PrimeFilter(min_primes=1, max_primes=4))

            if ConsecutiveFilter and QuadrantFilter and LastDigitFilter:
                pipeline.add_filter(ConsecutiveFilter(max_consecutive_pairs=2))
                pipeline.add_filter(QuadrantFilter())
                pipeline.add_filter(LastDigitFilter(max_same_ending=3))

            # 3. Generación y Filtrado (Simulación reducida pero representativa)
            sim_size = 200_000  # Muestra estadística
            pool_nums = np.arange(1, total_balls + 1)
            raw_batch = np.random.choice(
                pool_nums,
                size=(sim_size, base_config.ticket_size),
                replace=True,
                p=weights_np,
            )

            valid_scores = []

            for row in raw_batch:
                uniques = np.unique(row)
                if len(uniques) != base_config.ticket_size:
                    continue

                # A. Validación Estructural (El filtro real)
                cand_obj = CandidateCombination(tuple(sorted(uniques.tolist())))
                if not pipeline.validate(cand_obj):
                    continue

                # B. Scoring
                score = 0
                for pair in itertools.combinations(cand_obj.numbers, 2):
                    if pair in clusters_dict:
                        score += clusters_dict[pair]

                if score > 0:
                    valid_scores.append(score)

            if not valid_scores:
                continue

            scores_np = np.array(valid_scores)

            # Score del Ganador Real
            winner_score = 0
            for pair in itertools.combinations(sorted(tuple(target_set)), 2):
                if pair in clusters_dict:
                    winner_score += clusters_dict[pair]

            # Proyección del Universo Total
            # (Ratio de paso * Universo Total Generado en Estrategia Real ~5M)
            pass_ratio = len(valid_scores) / sim_size
            projected_universe_raw = pass_ratio * 5_000_000

            # --- EVALUACIÓN DE PERCENTILES ---
            for p in percentile_grid:
                threshold = np.percentile(scores_np, p)

                # Tamaño final tras corte de percentil
                count_above = np.sum(scores_np >= threshold)
                percentile_pass_ratio = count_above / len(scores_np)
                final_size = projected_universe_raw * percentile_pass_ratio

                results[p]["total_size"] += final_size

                # ¿Atrapamos al ganador?
                # Debe superar el umbral Y (implícitamente) pasar los filtros estructurales.
                # Como es un sorteo real, asumimos que pasa los filtros estructurales (son patrones naturales).
                if winner_score >= threshold:
                    results[p]["hits"] += 1

        # --- REPORTE ---
        print("\n" + "=" * 65)
        print(f"📊 RESULTADOS DE OPTIMIZACIÓN (Con Filtros Estructurales)")
        print("=" * 65)
        print(
            f"{'PERCENTIL':<10} | {'COBERTURA':<10} | {'TAMAÑO EST.':<15} | {'CALIFICACIÓN':<10}"
        )
        print("-" * 65)

        best_p = 80
        best_score = -float("inf")

        for p in percentile_grid:
            hits = results[p]["hits"]
            coverage = (hits / lookback) * 100
            avg_size = results[p]["total_size"] / lookback

            # Score: Bonifica cobertura, penaliza tamaño > 300k
            score = coverage * 20 - (avg_size / 20000)

            color = Fore.WHITE
            if coverage > 10:
                color = Fore.GREEN
            if coverage == 0:
                color = Fore.RED

            print(
                f"{p:<10} | {color}{coverage:5.1f}%{Style.RESET_ALL}    | {avg_size:10,.0f} tkts  | {score:6.1f}"
            )

            if score > best_score:
                best_score = score
                best_p = p

        print("=" * 65)
        print(f"🏆 MEJOR CONFIGURACIÓN: PERCENTIL {best_p}")
        print(f"   (Ajusta QUALITY_PERCENTILE en universe_reduction.py)")

        return best_p
