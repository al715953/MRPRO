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

# --- Filtros de la Arquitectura ---
from src.core.filters.pipeline import FilterPipeline
from src.core.filters.implementations.geometric import SumRangeFilter
from src.core.filters.implementations.probabilistic import ParityFilter, PrimeFilter
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
# Generamos 5 millones de candidatos crudos para filtrar y quedarnos con la "crema y nata"
RAW_GENERATION_SIZE = 5_000_000
# Solo aceptamos tickets cuyo puntaje histórico sea superior al del 80% de la población
QUALITY_PERCENTILE = 75


def worker_weighted_generation(args: Tuple) -> List[Tuple[float, Tuple[int, ...]]]:
    """
    Worker que genera candidatos masivos y aplica TODOS los filtros (Duros + Estructurales).
    """
    (
        batch_size,
        ticket_size,
        total_balls,
        weights_array,
        top_clusters_set,
        filter_cfg,
    ) = args

    # --- 1. CONSTRUCCIÓN DEL PIPELINE (Sincronizado con MonteCarlo) ---
    pipeline = FilterPipeline()

    # A. Filtros Matemáticos
    pipeline.add_filter(SumRangeFilter(filter_cfg["sum_min"], filter_cfg["sum_max"]))
    pipeline.add_filter(ParityFilter(filter_cfg["even_min"], filter_cfg["even_max"]))
    pipeline.add_filter(ACValueFilter(filter_cfg["ac_min"]))
    pipeline.add_filter(PrimeFilter(min_primes=1, max_primes=4))

    # B. Filtros Estructurales (Para limpiar el universo de combinaciones "feas")
    if ConsecutiveFilter and QuadrantFilter and LastDigitFilter:
        pipeline.add_filter(ConsecutiveFilter(max_consecutive_pairs=2))
        pipeline.add_filter(QuadrantFilter())
        pipeline.add_filter(LastDigitFilter(max_same_ending=3))

    # C. Inercia (Opcional en generación masiva, a veces es mejor desactivarlo aquí para dar variedad)
    if filter_cfg.get("inertia_min", 0) > 0 and filter_cfg.get("previous_draw"):
        pipeline.add_filter(
            InertiaFilter(filter_cfg["previous_draw"], filter_cfg["inertia_min"])
        )

    # --- 2. GENERACIÓN VECTORIZADA PONDERADA ---
    pool_nums = np.arange(1, total_balls + 1)

    # Generamos matriz gigante de números aleatorios basados en frecuencia (pesos)
    raw_batch = np.random.choice(
        pool_nums, size=(batch_size, ticket_size), replace=True, p=weights_array
    )

    valid_candidates = []

    # --- 3. PROCESAMIENTO Y FILTRADO ---
    for row in raw_batch:
        unique_nums = np.unique(row)
        if len(unique_nums) != ticket_size:
            continue

        candidate_tuple = tuple(sorted(unique_nums.tolist()))
        candidate_obj = CandidateCombination(candidate_tuple)

        # A. Filtro Duro (El gran colador)
        if not pipeline.validate(candidate_obj):
            continue

        # B. Scoring de Clústers (ADN Histórico)
        # Solo guardamos si tiene cierta resonancia con pares históricos
        score = 0
        for pair in itertools.combinations(candidate_tuple, 2):
            if pair in top_clusters_set:
                score += top_clusters_set[pair]

        # Guardamos todo lo que sea válido y tenga score positivo
        if score > 0:
            valid_candidates.append((score, candidate_tuple))

    return valid_candidates


class UniverseReductionStrategy(ILotteryStrategy):
    """
    Estrategia 'Red de Pesca de Élite'.
    Genera millones, filtra estructuralmente y guarda el CSV.
    """

    def predict(
        self, history: DrawHistoryDTO, config: PredictionConfigDTO
    ) -> PredictionResultDTO:
        overrides = getattr(config, "filter_overrides", {})
        verbose = overrides.get("verbose", True)

        if verbose:
            print(f"🌌 Iniciando Generador de Universo (Sincronizado)...")

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
        # Usamos rangos "seguros" por defecto, o los que vengan del config
        filter_config = {
            "sum_min": overrides.get("sum_min", 90),
            "sum_max": overrides.get("sum_max", 200),
            "even_min": overrides.get("even_min", 2),
            "even_max": overrides.get("even_max", 4),
            "ac_min": overrides.get(
                "ac_min", 5
            ),  # Un poco más permisivo para el universo
            "inertia_min": overrides.get("inertia_min", 0),
            "previous_draw": (
                history.winning_numbers[-1] if history.winning_numbers else []
            ),
        }

        # --- FASE 3: GENERACIÓN MASIVA PARALELA ---
        num_cores = cpu_count()
        chunk_size = RAW_GENERATION_SIZE // num_cores

        if verbose:
            print(f"🔥 Procesando {RAW_GENERATION_SIZE:,} candidatos crudos...")
            print(f"⚙️  Aplicando Pipeline Estructural...")

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
                print("⚠ ALERTA: Filtros demasiado estrictos. Universo vacío.")
            return PredictionResultDTO("Empty", [])

        # --- FASE 4: CORTE DE CALIDAD (PERCENTIL) ---
        if verbose:
            print(
                f"💎 Aplicando Corte de Excelencia (Top {100 - QUALITY_PERCENTILE}%)..."
            )

        # Eliminamos duplicados antes de filtrar por score
        # (Esto es importante para reducir el universo real)
        unique_pool = {}
        for score, ticket in global_pool:
            if ticket not in unique_pool:
                unique_pool[ticket] = score
            else:
                # Si sale repetido, podríamos sumar score o quedarnos con el max (aquí es igual)
                pass

        pool_list = [(score, ticket) for ticket, score in unique_pool.items()]

        # Calculamos percentil sobre únicos
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

        if verbose:
            print("-" * 50)
            print(f"📊 RESUMEN UNIVERSO REDUCIDO")
            print(f"🔢 Matemáticamente Válidos (Únicos): {len(pool_list):,}")
            print(
                f"✂️  Umbral de Calidad (Percentil {QUALITY_PERCENTILE}): Score >= {score_threshold:.2f}"
            )
            print(f"🌌 TAMAÑO FINAL UNIVERSO: {len(final_universe):,}")
            print(f"📂 Guardado en: {filename}")
            print("-" * 50)

        return PredictionResultDTO(
            strategy_name="Elite Universe Reduction",
            tickets=[list(t) for t in final_universe],
        )
