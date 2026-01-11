import random
import heapq
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

# --- Imports de la nueva Arquitectura de Filtros ---
from src.core.filters.pipeline import FilterPipeline
from src.core.filters.implementations.geometric import SumRangeFilter
from src.core.filters.implementations.probabilistic import ParityFilter
from src.core.filters.implementations.arithmetic import ACValueFilter
from src.core.filters.implementations.physical import InertiaFilter

# ==========================================
# CONFIGURACIÓN DE ALTO RENDIMIENTO
# ==========================================
NUM_SIMULACIONES_CALIB = 200000  # Fase 1
NUM_CANDIDATOS_TOTAL = 20000000  # Fase 2
TOP_PER_CORE = 1000  # Fase 4 (Límite por núcleo)


def worker_monte_carlo_pipeline(args: Tuple) -> List[Tuple[float, List[int]]]:
    """
    TRABAJADOR (FASES 2 Y 3) - VERSIÓN PIPELINE
    Se ejecuta en cada núcleo de forma aislada.
    """
    count, ticket_size, total_balls, heat_map, filter_cfg = args

    # --- Construcción del Pipeline Local ---
    # Instanciamos los filtros aquí para evitar problemas de serialización
    pipeline = FilterPipeline()

    # 1. Capa Física (Inercia): ¿Trae números del sorteo anterior?
    if filter_cfg.get("previous_draw"):
        pipeline.add_filter(
            InertiaFilter(
                previous_draw=filter_cfg["previous_draw"],
                min_matches=filter_cfg.get(
                    "inertia_min", 0
                ),  # Por defecto 0 si no se exige
            )
        )

    # 2. Capa Geométrica (Gauss): Suma total
    pipeline.add_filter(
        SumRangeFilter(
            min_val=filter_cfg.get("sum_min", 80),
            max_val=filter_cfg.get("sum_max", 220),
        )
    )

    # 3. Capa Probabilística: Pares e Impares
    pipeline.add_filter(
        ParityFilter(
            min_even=filter_cfg.get("even_min", 2),
            max_even=filter_cfg.get("even_max", 4),
        )
    )

    # 4. Capa Aritmética: Complejidad (AC Value)
    # Se pone al final porque es computacionalmente más costoso
    pipeline.add_filter(ACValueFilter(min_ac=filter_cfg.get("ac_min", 7)))

    # --- Inicio de Generación ---
    local_heap = []
    all_numbers = list(range(1, total_balls + 1))
    get_score = heat_map.get

    for _ in range(count):
        # FASE 2: Generación Aleatoria
        nums = sorted(random.sample(all_numbers, ticket_size))

        # Envolvemos en el "Ticket Rico" (Lazy evaluation)
        candidate = CandidateCombination(tuple(nums))

        # FASE 3: Validación por Pipeline (Fail-Fast)
        # Si validate retorna False, saltamos al siguiente ciclo inmediatamente
        if not pipeline.validate(candidate):
            continue

        # Si pasa los filtros, calculamos Score (Inercia Histórica)
        score = sum(get_score(n, 0) for n in nums)

        # Mantenimiento del Top Local (Heap)
        if len(local_heap) < TOP_PER_CORE:
            heapq.heappush(local_heap, (score, nums))
        else:
            if score > local_heap[0][0]:
                heapq.heapreplace(local_heap, (score, nums))

    return sorted(local_heap, key=lambda x: x[0], reverse=True)


class MonteCarloStrategy(ILotteryStrategy):
    """
    Estrategia de Simulación Monte Carlo Evolucionada con Pipeline de 5 Capas.
    """

    def predict(
        self, history: DrawHistoryDTO, config: PredictionConfigDTO
    ) -> PredictionResultDTO:
        all_numbers = list(range(1, config.total_balls + 1))

        # --- FASE 1: CALIBRACIÓN (MAPA DE CALOR) ---
        real_freq = Counter()
        for nums in history.winning_numbers:
            real_freq.update(nums)

        weights = [real_freq.get(n, 1) for n in all_numbers]

        # Simulación rápida para proyectar tendencias
        sim_heat_map = Counter()
        for _ in range(NUM_SIMULACIONES_CALIB):
            sim_draw = random.choices(
                all_numbers, weights=weights, k=config.ticket_size
            )
            sim_heat_map.update(set(sim_draw))

        heat_map_dict = dict(sim_heat_map)

        # --- PREPARACIÓN DE FILTROS ---
        # Obtenemos el último sorteo real para la Capa Física (Inercia)
        last_draw = history.winning_numbers[-1] if history.winning_numbers else []

        # Configuración de Filtros (Estos valores podrían venir de un Config o ML en el futuro)
        filter_config = {
            "previous_draw": (
                history.winning_numbers[-1] if history.winning_numbers else []
            ),
            "inertia_min": 0,  # Opcional: forzar al menos 1 del anterior
            "sum_min": 100,  # Rango ajustado para Melate Retro (aprox)
            "sum_max": 180,
            "even_min": 2,
            "even_max": 4,
            "ac_min": 5,  # Valor conservador, 7 es muy estricto para 6/39
        }
        # 2. 👇 FUSIÓN MÁGICA: Sobrescribimos con lo que diga el config (si existe)
        # Esto permite que el Optimizador cambie 'ac_min' de 5 a 7 dinámicamente.
        overrides = getattr(config, "filter_overrides", {})
        filter_config = {**filter_config, **overrides}

        
        # --- FASE 2: FUERZA BRUTA PARALELA ---
        num_cores = cpu_count()
        chunk = NUM_CANDIDATOS_TOTAL // num_cores

        print(f"🚀 Saturando {num_cores} núcleos con Pipeline de Filtrado...")
        print(f"⚙️ Configuración: {filter_config}")

        tasks_args = [
            (
                chunk,
                config.ticket_size,
                config.total_balls,
                heat_map_dict,
                filter_config,
            )
            for _ in range(num_cores)
        ]

        global_elite_pool = []

        with Pool(processes=num_cores) as pool:
            # Usamos el nuevo worker con pipeline
            results = pool.map(worker_monte_carlo_pipeline, tasks_args)

            # --- FASE 4: CONSOLIDACIÓN ---
            seen_tickets = set()
            for res in results:
                for score, ticket in res:
                    ticket_tuple = tuple(ticket)
                    if ticket_tuple not in seen_tickets:
                        global_elite_pool.append((score, ticket))
                        seen_tickets.add(ticket_tuple)

        # Selección final
        global_elite_pool.sort(key=lambda x: x[0], reverse=True)
        selected_tickets = [t for s, t in global_elite_pool[: config.num_tickets]]

        return PredictionResultDTO(
            strategy_name="Monte Carlo Pipeline (5-Layers)",
            tickets=selected_tickets,
        )
