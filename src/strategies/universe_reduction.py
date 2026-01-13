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
)

# --- CONFIGURACIÓN ---
RAW_GENERATION_SIZE = 5_000_000
QUALITY_PERCENTILE = 75
BATCH_BUFFER_RATE = 1.8  # Aumentamos buffer porque los filtros nuevos son agresivos


def calculate_ac_fast(nums: Tuple[int, ...]) -> int:
    """Calcula AC Value sin overhead de objetos."""
    diffs = {b - a for a, b in itertools.combinations(nums, 2)}
    return len(diffs) - (len(nums) - 1)


def worker_weighted_generation_optimized(
    args: Tuple,
) -> List[Tuple[float, Tuple[int, ...]]]:
    """
    WORKER HIPER-OPTIMIZADO (V3):
    Vectoriza el 95% de la lógica de filtrado.
    """
    (
        target_batch_size,
        ticket_size,
        total_balls,
        weights_array,
        top_clusters_set,
        filter_cfg,
    ) = args

    # --- FASE 1: GENERACIÓN VECTORIAL ---
    pool_nums = np.arange(1, total_balls + 1)
    raw_size = int(target_batch_size * BATCH_BUFFER_RATE)

    # Generación probabilística (reemplazo=True es mucho más rápido, luego filtramos)
    # Nota: np.random.choice con p=weights es lento en bucles, pero aquí hacemos batch gigante
    raw_batch = np.random.choice(
        pool_nums, size=(raw_size, ticket_size), replace=True, p=weights_array
    )

    # Ordenar filas (necesario para diffs y lógica posterior)
    raw_batch.sort(axis=1)

    # 1. Filtro Unicidad (Eliminar [5,5,...])
    # diff > 0 implica estrictamente creciente (sin repetidos)
    diffs = np.diff(raw_batch, axis=1)
    mask_unique = np.min(diffs, axis=1) > 0
    candidates = raw_batch[mask_unique]

    # Actualizamos diffs para candidatos únicos (se usa en consecutivos)
    diffs = diffs[mask_unique]

    # --- FASE 2: FILTROS MATEMÁTICOS (VECTORIZADOS) ---

    # A. Suma
    sums = candidates.sum(axis=1)
    mask_sum = (sums >= filter_cfg["sum_min"]) & (sums <= filter_cfg["sum_max"])

    # B. Pares
    evens_count = (candidates % 2 == 0).sum(axis=1)
    mask_even = (evens_count >= filter_cfg["even_min"]) & (
        evens_count <= filter_cfg["even_max"]
    )

    # C. Consecutivos (Usamos diffs calculado arriba)
    # diffs==1 significa números seguidos. Sumamos cuántos hay por fila.
    cons_count = (diffs == 1).sum(axis=1)
    mask_cons = cons_count <= filter_cfg["max_consecutive"]

    # D. Primos (Lookup Table Vectorizado)
    # Máscara booleana de primos para 1..39
    primes_lookup = np.array([False] * (total_balls + 2))
    primes_list = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37]
    primes_lookup[primes_list] = True

    # Indexado fantasía: verifica primalidad de toda la matriz de golpe
    primes_count = primes_lookup[candidates].sum(axis=1)
    mask_prime = (primes_count >= filter_cfg["prime_min"]) & (
        primes_count <= filter_cfg["prime_max"]
    )

    # E. Terminaciones (Last Digits)
    # Queremos evitar >3 números con misma terminación.
    # Estrategia: Modulo 10 -> Sort -> Chequear saltos de índice
    last_digits = candidates % 10
    last_digits.sort(axis=1)
    # Si col[i] == col[i+3], significa que hay 4 números iguales (índices i, i+1, i+2, i+3)
    # Para ticket de 6, chequeamos índices: (0 vs 3), (1 vs 4), (2 vs 5)
    # Si alguno coincide, rechazamos.
    has_4_same = (
        (last_digits[:, 0] == last_digits[:, 3])
        | (last_digits[:, 1] == last_digits[:, 4])
        | (last_digits[:, 2] == last_digits[:, 5])
    )
    mask_last_digit = ~has_4_same  # Negamos

    # F. Cuadrantes
    # Q1: 1-9, Q2: 10-19, Q3: 20-29, Q4: 30-39
    # np.digitize devuelve índices de bins. Bins: [10, 20, 30]
    # 1-9 -> 0, 10-19 -> 1, etc.
    quad_bins = np.array([10, 20, 30])
    quads = np.digitize(candidates, quad_bins)

    # Contar cuadrantes únicos por fila.
    # Truco: Ordenamos quads por fila (ya deberían estarlo por candidates ordenados)
    # y usamos np.diff != 0 para contar cambios + 1
    # O mejor: Scikit-learn tiene row-wise unique, pero numpy puro no fácil.
    # Aproximación rápida: Chequear si el rango cubre cuadrantes o usar lógica Python fina luego.
    # Dado que es solo "al menos 2 cuadrantes", podemos hacerlo simple:
    # Q_min vs Q_max. Si estan todos en el mismo cuadrante, Q_max == Q_min.
    # Eso elimina tickets concentrados en 1 solo cuadrante (ej. 1,2,3,4,5,6).
    q_min = quads.min(axis=1)
    q_max = quads.max(axis=1)
    mask_quad = q_max > q_min  # Al menos 2 cuadrantes distintos

    # --- APLICAR MÁSCARA TOTAL ---
    final_mask = (
        mask_sum & mask_even & mask_cons & mask_prime & mask_last_digit & mask_quad
    )
    survivors_np = candidates[final_mask]

    # --- FASE 3: FILTRADO FINO (Python) & SCORING ---
    # Solo llegamos aquí con candidatos muy prometedores
    valid_candidates = []

    # Pre-cálculo config
    min_ac = filter_cfg["ac_min"]

    for row in survivors_np:
        # Tuple nativa
        candidate_tuple = tuple(row.tolist())

        # 1. Filtro AC Value (Costoso, no vectorizado aun)
        if calculate_ac_fast(candidate_tuple) < min_ac:
            continue

        # 2. Scoring (Clusters)
        score = 0
        for pair in itertools.combinations(candidate_tuple, 2):
            if pair in top_clusters_set:
                score += top_clusters_set[pair]

        # Guardar si tiene algún valor genético
        if score > 0:
            valid_candidates.append((score, candidate_tuple))

        if len(valid_candidates) >= target_batch_size:
            break

    return valid_candidates


class UniverseReductionStrategy(ILotteryStrategy):
    """
    Estrategia 'Red de Pesca de Élite' V3 (Full Vectorized).
    """

    def predict(
        self, history: DrawHistoryDTO, config: PredictionConfigDTO
    ) -> PredictionResultDTO:
        overrides = getattr(config, "filter_overrides", {})
        verbose = overrides.get("verbose", True)

        start_time = time.time()

        if verbose:
            print(f"🌌 Iniciando Generador de Universo (Numpy Accelerated V3)...")

        # --- PREPARACIÓN DE DATOS ---
        freq_counter = Counter()
        for draw in history.winning_numbers:
            freq_counter.update(draw[:6])

        # Pesos con suavizado (+1) para evitar probabilidad 0
        weights = [freq_counter.get(n, 1) + 1 for n in range(1, config.total_balls + 1)]
        weights_np = np.array(weights, dtype=float)
        weights_np /= weights_np.sum()

        cluster_counter = Counter()
        for draw in history.winning_numbers:
            for pair in itertools.combinations(sorted(draw[:6]), 2):
                cluster_counter[pair] += 1
        clusters_dict = dict(cluster_counter)

        # Configuración compacta para Workers
        filter_config = {
            "sum_min": overrides.get("sum_min", 108),
            "sum_max": overrides.get("sum_max", 180),
            "even_min": overrides.get("even_min", 2),
            "even_max": overrides.get("even_max", 4),
            "prime_min": overrides.get("prime_min", 1),
            "prime_max": overrides.get("prime_max", 4),
            "max_consecutive": 2,  # Max pares consecutivos (1,2,3 -> 2 pares)
            "ac_min": overrides.get("ac_min", 5),
        }

        # --- PARALELISMO ---
        num_cores = max(1, cpu_count() - 1)  # Dejar 1 core libre para el sistema
        chunk_size = RAW_GENERATION_SIZE // num_cores

        if verbose:
            print(f"🔥 Objetivo: {RAW_GENERATION_SIZE:,} candidatos brutos.")
            print(f"🚀 Motores encendidos: {num_cores} núcleos.")

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
            results = pool.map(worker_weighted_generation_optimized, args)
            for res in results:
                global_pool.extend(res)

        # --- POST-PROCESADO ---
        if not global_pool:
            print("⚠ Universo vacío. Relaja los filtros.")
            return PredictionResultDTO("Empty", [])

        # Deduplicación y Corte
        unique_pool = {}
        for score, ticket in global_pool:
            # Quedarse con el ticket si ya existe (mismo score)
            unique_pool[ticket] = score

        pool_list = list(unique_pool.items())

        # Percentil de Calidad
        scores = np.array([x[1] for x in pool_list])
        threshold = np.percentile(scores, QUALITY_PERCENTILE)

        final_universe = [t for t, s in pool_list if s >= threshold]

        # Guardado
        output_folder = "data"
        os.makedirs(output_folder, exist_ok=True)
        filename = os.path.join(output_folder, "universo_reducido.csv")

        df = pd.DataFrame(final_universe, columns=[f"B{i}" for i in range(1, 7)])
        df.to_csv(filename, index=False)

        elapsed = time.time() - start_time
        if verbose:
            print(
                f"⏱️  Tiempo: {elapsed:.2f}s | 📥 Generados: {len(pool_list):,} | 📤 Final: {len(final_universe):,}"
            )
            print(f"✅ Universo guardado en '{filename}'")

        return PredictionResultDTO("Universe V3", [list(t) for t in final_universe])
