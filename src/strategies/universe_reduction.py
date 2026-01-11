import numpy as np
import pandas as pd
import os
import itertools
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

# --- Filtros de la Arquitectura Nueva ---
from src.core.filters.pipeline import FilterPipeline
from src.core.filters.implementations.geometric import SumRangeFilter
from src.core.filters.implementations.probabilistic import ParityFilter
from src.core.filters.implementations.arithmetic import ACValueFilter
from src.core.filters.implementations.physical import InertiaFilter

# --- CONFIGURACIÓN ---
RAW_GENERATION_SIZE = 5000000
# Solo aceptamos tickets cuyo puntaje sea superior al del 88% de la población generada
QUALITY_PERCENTILE = 80


def worker_weighted_generation(args: Tuple) -> List[Tuple[float, Tuple[int, ...]]]:
    """
    Worker que genera y filtra, retornando TODO lo que sea válido matemáticamente.
    El filtrado de calidad (Score) se hará globalmente después.
    """
    (
        batch_size,
        ticket_size,
        total_balls,
        weights_array,
        top_clusters_set,
        filter_cfg,
    ) = args

    # A. Pipeline (Filtros Duros: Suma, AC, Pares)
    pipeline = FilterPipeline()
    pipeline.add_filter(SumRangeFilter(filter_cfg["sum_min"], filter_cfg["sum_max"]))
    pipeline.add_filter(ParityFilter(filter_cfg["even_min"], filter_cfg["even_max"]))
    pipeline.add_filter(ACValueFilter(filter_cfg["ac_min"]))

    if filter_cfg.get("inertia_min", 0) > 0 and filter_cfg.get("previous_draw"):
        pipeline.add_filter(
            InertiaFilter(filter_cfg["previous_draw"], filter_cfg["inertia_min"])
        )

    # B. Generación Vectorizada Ponderada
    pool_nums = np.arange(1, total_balls + 1)

    # replace=True es más rápido en generación, luego validamos unicidad
    raw_batch = np.random.choice(
        pool_nums, size=(batch_size, ticket_size), replace=True, p=weights_array
    )

    valid_candidates = []

    # C. Procesamiento y Scoring
    for row in raw_batch:
        unique_nums = np.unique(row)
        if len(unique_nums) != ticket_size:
            continue

        candidate_tuple = tuple(sorted(unique_nums.tolist()))
        candidate_obj = CandidateCombination(candidate_tuple)

        # 1. Filtro Duro (Matemático) - Si no pasa, se descarta.
        if not pipeline.validate(candidate_obj):
            continue

        # 2. Scoring de Clústers (ADN Histórico)
        score = 0
        for pair in itertools.combinations(candidate_tuple, 2):
            if pair in top_clusters_set:
                score += top_clusters_set[pair]

        # Solo guardamos candidatos que tengan al menos un rastro histórico (Score > 0)
        if score > 0:
            valid_candidates.append((score, candidate_tuple))

    return valid_candidates


class UniverseReductionStrategy(ILotteryStrategy):
    """
    Estrategia 'Red de Pesca de Élite'.
    Genera millones de candidatos ponderados, filtra matemáticamente
    y selecciona el percentil superior basado en clústers históricos.
    """

    def predict(
        self, history: DrawHistoryDTO, config: PredictionConfigDTO
    ) -> PredictionResultDTO:
        # Detectar modo silencioso (para Backtests)
        overrides = getattr(config, "filter_overrides", {})
        verbose = overrides.get("verbose", True)

        if verbose:
            print(f"🌌 Iniciando Generador de Universo (Modo Élite)...")

        # --- FASE 1: ANÁLISIS DE PESOS Y CLUSTERS ---
        freq_counter = Counter()
        for draw in history.winning_numbers:
            # Usamos solo los primeros 6 naturales para el análisis
            freq_counter.update(draw[:6])

        # Suavizado (+1) para evitar probabilidad cero
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
            "sum_min": overrides.get("sum_min", 90),
            "sum_max": overrides.get("sum_max", 200),
            "even_min": overrides.get("even_min", 2),
            "even_max": overrides.get("even_max", 4),
            "ac_min": overrides.get("ac_min", 4),
            "inertia_min": overrides.get("inertia_min", 0),
            "previous_draw": (
                history.winning_numbers[-1] if history.winning_numbers else []
            ),
        }

        # --- FASE 3: GENERACIÓN MASIVA PARALELA ---
        num_cores = cpu_count()
        chunk_size = RAW_GENERATION_SIZE // num_cores

        if verbose:
            print(
                f"🔥 Procesando {RAW_GENERATION_SIZE:,} candidatos en {num_cores} núcleos..."
            )
            print(f"⚙️  Filtros Base: {filter_config}")

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
            results = pool.map(worker_weighted_generation, args)
            for res in results:
                global_pool.extend(res)

        total_valid = len(global_pool)

        if total_valid == 0:
            if verbose:
                print(
                    "⚠ ALERTA: Filtros demasiado estrictos. No pasó ningún candidato."
                )
            return PredictionResultDTO("Empty", [])

        # --- FASE 4: CORTE DE CALIDAD (PERCENTIL) ---
        if verbose:
            print(
                f"💎 Aplicando Corte de Excelencia (Top {100 - QUALITY_PERCENTILE}%)..."
            )

        # Extraemos scores para calcular umbral
        all_scores = np.array([x[0] for x in global_pool])

        # Calculamos el puntaje de corte dinámico
        score_threshold = np.percentile(all_scores, QUALITY_PERCENTILE)

        if verbose:
            print(f"   📊 Umbral de Score calculado: {score_threshold:.2f}")

        # Filtramos: Solo pasan los que superan el umbral
        final_universe = [
            ticket for score, ticket in global_pool if score >= score_threshold
        ]

        # --- FASE 5: EXPORTACIÓN (Solo si no es test masivo, o siempre se sobrescribe) ---
        output_folder = "data"
        filename = os.path.join(output_folder, "universo_reducido.csv")

        # Crear carpeta si no existe
        os.makedirs(output_folder, exist_ok=True)

        # Guardamos CSV (Útil para análisis manual posterior)
        df = pd.DataFrame(final_universe, columns=[f"B{i}" for i in range(1, 7)])
        df.to_csv(filename, index=False)

        if verbose:
            print("-" * 50)
            print(f"📊 RESUMEN FINAL DEL UNIVERSO")
            print(f"🔢 Total Matemáticamente Válidos: {total_valid:,}")
            print(
                f"✂️  Descartados por Baja Calidad: {total_valid - len(final_universe):,}"
            )
            print(f"💎 Universo Élite Final: {len(final_universe):,}")
            print(f"📂 Archivo guardado: {filename}")
            print("-" * 50)

        # Retornamos TODO el universo para que el Backtester pueda medir la cobertura real
        return PredictionResultDTO(
            strategy_name="Elite Universe Reduction",
            tickets=[list(t) for t in final_universe],
        )
