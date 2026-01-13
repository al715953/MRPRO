import numpy as np
import pandas as pd
import os
import itertools
import time
from multiprocessing import Pool, cpu_count
from collections import Counter
from typing import List, Tuple, Dict, Any

from src.domain.interfaces import ILotteryStrategy
from src.domain.dtos import (
    DrawHistoryDTO,
    PredictionConfigDTO,
    PredictionResultDTO,
    CandidateCombination,
)

# --- Filtros de la Arquitectura ---
from src.core.filters.pipeline import FilterPipeline

# Nota: SumRangeFilter y ParityFilter ya no se importan para el pipeline
# porque se aplican vectorialmente, pero los mantenemos si se usan en otro lado.
from src.core.filters.implementations.probabilistic import PrimeFilter
from src.core.filters.implementations.arithmetic import ACValueFilter
from src.core.filters.implementations.physical import InertiaFilter

# Importación segura de filtros estructurales
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

# --- CONFIGURACIÓN ---
RAW_GENERATION_SIZE = 5_000_000
QUALITY_PERCENTILE = 75
BATCH_BUFFER_RATE = 1.5  # Generamos 50% extra para compensar el filtrado vectorial


def worker_weighted_generation_optimized(
    args: Tuple,
) -> List[Tuple[float, Tuple[int, ...]]]:
    """
    WORKER OPTIMIZADO (V2):
    Utiliza vectorización de Numpy para pre-filtrar candidatos antes de
    instanciar objetos costosos de Python.
    """
    (
        target_batch_size,
        ticket_size,
        total_balls,
        weights_array,
        top_clusters_set,
        filter_cfg,
    ) = args

    # --- FASE 1: GENERACIÓN VECTORIAL MASIVA (Velocidad C) ---
    pool_nums = np.arange(1, total_balls + 1)

    # Generamos un exceso (buffer) porque el filtrado vectorial eliminará muchos
    raw_size = int(target_batch_size * BATCH_BUFFER_RATE)

    # Generación probabilística rápida
    raw_batch = np.random.choice(
        pool_nums, size=(raw_size, ticket_size), replace=True, p=weights_array
    )

    # --- FASE 2: PRE-FILTRADO MATRICIAL (Numpy) ---
    # 1. Unicidad (Eliminar filas con números repetidos ej. [5,5,...])
    raw_batch.sort(axis=1)  # Ordenamiento in-place (muy rápido)
    diffs = np.diff(raw_batch, axis=1)
    # Si la diferencia mínima es > 0, todos son distintos
    mask_unique = np.min(diffs, axis=1) > 0

    # Aplicamos máscara de unicidad primero para limpiar
    candidates = raw_batch[mask_unique]

    # 2. Filtro de Suma (Vectorizado)
    sums = candidates.sum(axis=1)
    mask_sum = (sums >= filter_cfg["sum_min"]) & (sums <= filter_cfg["sum_max"])

    # 3. Filtro de Pares (Vectorizado)
    # (n % 2 == 0) genera matriz booleana, sumamos True como 1
    evens_count = (candidates % 2 == 0).sum(axis=1)
    mask_even = (evens_count >= filter_cfg["even_min"]) & (
        evens_count <= filter_cfg["even_max"]
    )

    # Aplicamos filtros matemáticos duros de golpe
    final_mask = mask_sum & mask_even
    survivors_np = candidates[final_mask]

    # --- FASE 3: FILTRADO FINO Y ESTRUCTURAL (Python Objects) ---
    # Solo llegamos aquí con candidatos que ya cumplen Suma, Pares y Unicidad.

    valid_candidates = []

    # Construimos el pipeline solo con los filtros complejos (no vectorizables fácilmente)
    pipeline = FilterPipeline()

    pipeline.add_filter(ACValueFilter(filter_cfg["ac_min"]))
    pipeline.add_filter(
        PrimeFilter(min_primes=1, max_primes=4)
    )  # Primos se queda aquí por ahora

    if ConsecutiveFilter and QuadrantFilter and LastDigitFilter:
        pipeline.add_filter(ConsecutiveFilter(max_consecutive_pairs=2))
        pipeline.add_filter(QuadrantFilter())
        pipeline.add_filter(LastDigitFilter(max_same_ending=3))

    if filter_cfg.get("inertia_min", 0) > 0 and filter_cfg.get("previous_draw"):
        pipeline.add_filter(
            InertiaFilter(filter_cfg["previous_draw"], filter_cfg["inertia_min"])
        )

    # Iteramos solo sobre los sobrevivientes (muchos menos que el inicio)
    for row in survivors_np:
        # Convertimos a tupla nativa de Python
        candidate_tuple = tuple(row.tolist())

        # Validación Estructural Compleja
        candidate_obj = CandidateCombination(candidate_tuple)

        if not pipeline.validate(candidate_obj):
            continue

        # --- FASE 4: SCORING (ADN Histórico) ---
        score = 0
        for pair in itertools.combinations(candidate_tuple, 2):
            if pair in top_clusters_set:
                score += top_clusters_set[pair]

        if score > 0:
            valid_candidates.append((score, candidate_tuple))

        # Micro-optimización: Si ya llenamos el cupo del batch, salimos
        if len(valid_candidates) >= target_batch_size:
            break

    return valid_candidates


class UniverseReductionStrategy(ILotteryStrategy):
    """
    Estrategia 'Red de Pesca de Élite' (Optimizada con Numpy Vectorization).
    """

    def predict(
        self, history: DrawHistoryDTO, config: PredictionConfigDTO
    ) -> PredictionResultDTO:
        overrides = getattr(config, "filter_overrides", {})
        verbose = overrides.get("verbose", True)

        start_time = time.time()

        if verbose:
            print(f"🌌 Iniciando Generador de Universo (Motor Numpy V2)...")

        # --- FASE 1: ANÁLISIS DE PESOS Y CLUSTERS ---
        freq_counter = Counter()
        for draw in history.winning_numbers:
            freq_counter.update(draw[:6])

        weights = [freq_counter.get(n, 1) + 1 for n in range(1, config.total_balls + 1)]
        weights_np = np.array(weights, dtype=float)
        weights_np /= weights_np.sum()

        cluster_counter = Counter()
        for draw in history.winning_numbers:
            main_draw = sorted(draw[:6])
            for pair in itertools.combinations(main_draw, 2):
                cluster_counter[pair] += 1

        clusters_dict = dict(cluster_counter)

        # --- FASE 2: PREPARACIÓN DE CONFIGURACIÓN ---
        filter_config = {
            "sum_min": overrides.get("sum_min", 108),
            "sum_max": overrides.get("sum_max", 180),
            "even_min": overrides.get("even_min", 2),
            "even_max": overrides.get("even_max", 4),
            "ac_min": overrides.get("ac_min", 5),
            "inertia_min": overrides.get("inertia_min", 0),
            "previous_draw": (
                history.winning_numbers[-1] if history.winning_numbers else []
            ),
        }

        # --- FASE 3: GENERACIÓN MASIVA PARALELA ---
        num_cores = cpu_count()
        # Dividimos el trabajo
        chunk_size = RAW_GENERATION_SIZE // num_cores

        if verbose:
            print(f"🔥 Procesando objetivo de {RAW_GENERATION_SIZE:,} candidatos...")
            print(f"🚀 Vectorización activada en {num_cores} núcleos.")

        args = [
            (
                chunk_size,
                config.ticket_size,
                config.total_balls,
                weights_np,
                clusters_dict,
                filter_config,
            )
            for _ in range(num_cores)
        ]

        global_pool = []
        with Pool(processes=num_cores) as pool:
            # Usamos la nueva función optimizada
            results = pool.map(worker_weighted_generation_optimized, args)
            for res in results:
                global_pool.extend(res)

        total_valid = len(global_pool)

        if total_valid == 0:
            if verbose:
                print("⚠ ALERTA: Filtros demasiado estrictos. Universo vacío.")
            return PredictionResultDTO("Empty", [])

        # --- FASE 4: CORTE DE CALIDAD (PERCENTIL) ---
        if verbose:
            print(
                f"💎 Aplicando Corte de Excelencia (Top {100 - QUALITY_PERCENTILE}%)..."
            )

        unique_pool = {}
        for score, ticket in global_pool:
            if ticket not in unique_pool:
                unique_pool[ticket] = score

        pool_list = [(score, ticket) for ticket, score in unique_pool.items()]

        if not pool_list:
            return PredictionResultDTO("Empty", [])

        all_scores = np.array([x[0] for x in pool_list])
        score_threshold = np.percentile(all_scores, QUALITY_PERCENTILE)

        if verbose:
            print(f"   📊 Umbral de Score: {score_threshold:.2f}")

        final_universe = [
            ticket for score, ticket in pool_list if score >= score_threshold
        ]

        # --- FASE 5: EXPORTACIÓN ---
        output_folder = "data"
        filename = os.path.join(output_folder, "universo_reducido.csv")
        os.makedirs(output_folder, exist_ok=True)

        df = pd.DataFrame(final_universe, columns=[f"B{i}" for i in range(1, 7)])
        df.to_csv(filename, index=False)

        elapsed_time = time.time() - start_time

        if verbose:
            print("-" * 50)
            print(f"📊 RESUMEN UNIVERSO REDUCIDO (OPTIMIZADO)")
            print(f"⏱️  Tiempo Total: {elapsed_time:.2f} segundos")
            print(f"🔢 Válidos Pre-Corte: {len(pool_list):,}")
            print(f"🌌 TAMAÑO FINAL UNIVERSO: {len(final_universe):,}")
            print(f"📂 Guardado en: {filename}")
            print("-" * 50)

        return PredictionResultDTO(
            strategy_name="Elite Universe Reduction V2",
            tickets=[list(t) for t in final_universe],
        )
