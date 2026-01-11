import random
import heapq
import numpy as np
from multiprocessing import Pool, cpu_count
from collections import Counter
from typing import List, Tuple

from src.domain.interfaces import ILotteryStrategy
from src.domain.dtos import (
    DrawHistoryDTO,
    PredictionConfigDTO,
    PredictionResultDTO,
    CandidateCombination,
)
from src.core.filters.pipeline import FilterPipeline
from src.core.filters.implementations.geometric import SumRangeFilter
from src.core.filters.implementations.arithmetic import ACValueFilter
from src.core.filters.implementations.physical import InertiaFilter
from src.core.filters.implementations.probabilistic import ParityFilter, PrimeFilter

# Importamos con seguridad (por si acaso falla la estructura de archivos)
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

# ==========================================
# CONFIGURACIÓN
# ==========================================
NUM_CANDIDATOS_TOTAL = 15_000_000
TOP_PER_CORE = 2000


def worker_monte_carlo_smart(args: Tuple) -> List[Tuple[float, List[int]]]:
    (
        count,
        ticket_size,
        total_balls,
        heat_map_weights,
        seeds_pool,
        filter_cfg,
    ) = args

    pipeline = FilterPipeline()

    # 1. Filtros Básicos
    pipeline.add_filter(SumRangeFilter(filter_cfg["sum_min"], filter_cfg["sum_max"]))
    pipeline.add_filter(ParityFilter(filter_cfg["even_min"], filter_cfg["even_max"]))
    pipeline.add_filter(ACValueFilter(filter_cfg["ac_min"]))
    pipeline.add_filter(PrimeFilter(min_primes=1, max_primes=4))

    # 2. Capa Estructural (RELAJADA PARA RECUPERAR HITS)
    if ConsecutiveFilter and QuadrantFilter and LastDigitFilter:
        # Antes era 1, subimos a 2 para permitir combinaciones como [1,2... 10,11...]
        pipeline.add_filter(ConsecutiveFilter(max_consecutive_pairs=2))
        pipeline.add_filter(QuadrantFilter())
        # Antes era 2, subimos a 3. Es común ver 3 números con la misma terminación (ej. 3, 13, 33)
        pipeline.add_filter(LastDigitFilter(max_same_ending=3))

    if filter_cfg.get("previous_draw"):
        pipeline.add_filter(InertiaFilter(filter_cfg["previous_draw"], min_matches=0))

    local_heap = []
    pool_numbers = list(range(1, total_balls + 1))
    get_weight = heat_map_weights.get
    use_seeds = len(seeds_pool) > 0

    for _ in range(count):
        current_seeds = []
        # Mantenemos el "Truco" del 80% de probabilidad de usar semilla
        if use_seeds and random.random() < 0.80:
            seed = random.choice(seeds_pool)
            current_seeds.append(seed)

        needed = ticket_size - len(current_seeds)
        available = [n for n in pool_numbers if n not in current_seeds]

        try:
            random_part = random.sample(available, needed)
            nums = sorted(current_seeds + random_part)
        except ValueError:
            continue

        candidate = CandidateCombination(tuple(nums))

        if not pipeline.validate(candidate):
            continue

        # Scoring simple: Suma de pesos de frecuencia
        base_score = sum(get_weight(n, 1) for n in nums)

        # Boost a las semillas para que suban al Top
        if current_seeds:
            base_score += 50

        if len(local_heap) < TOP_PER_CORE:
            heapq.heappush(local_heap, (base_score, nums))
        else:
            if base_score > local_heap[0][0]:
                heapq.heapreplace(local_heap, (base_score, nums))

    return sorted(local_heap, key=lambda x: x[0], reverse=True)


class MonteCarloStrategy(ILotteryStrategy):
    def predict(
        self, history: DrawHistoryDTO, config: PredictionConfigDTO
    ) -> PredictionResultDTO:
        print(
            "🧠 MonteCarlo: Motor Híbrido (Lógica Fase 1 + Filtros Estructurales V2)..."
        )

        all_draws = history.winning_numbers
        if not all_draws:
            return PredictionResultDTO("Error", [])

        # Análisis Histórico
        flat_history = [n for draw in all_draws for n in draw[:6]]
        weights_dict = dict(Counter(flat_history))

        # Zombies y Hot Numbers
        current_draw_idx = len(all_draws)
        last_appearance = {n: 999 for n in range(1, config.total_balls + 1)}

        for idx, draw in enumerate(reversed(all_draws)):
            for num in draw[:6]:
                if last_appearance[num] == 999:
                    last_appearance[num] = idx

        zombies = [n for n, gap in last_appearance.items() if gap > 18]
        recent_flat = [n for draw in all_draws[-50:] for n in draw[:6]]
        hot_nums = [n for n, _ in Counter(recent_flat).most_common(10)]
        seeds_pool = list(set(zombies + hot_nums))

        print(f"🧬 Semillas Activas ({len(seeds_pool)}): {seeds_pool}")

        # Rangos Dinámicos
        sums = [sum(d[:6]) for d in all_draws[-100:]]
        mean_sum = np.mean(sums)
        std_sum = np.std(sums)

        dynamic_sum_min = int(mean_sum - 1.6 * std_sum)
        dynamic_sum_max = int(mean_sum + 1.6 * std_sum)

        filter_config = {
            "previous_draw": all_draws[-1] if all_draws else [],
            "sum_min": dynamic_sum_min,
            "sum_max": dynamic_sum_max,
            "even_min": 2,
            "even_max": 4,
            "ac_min": 6,
        }
        filter_config.update(getattr(config, "filter_overrides", {}))

        # Ejecución Paralela
        num_cores = cpu_count()
        chunk = NUM_CANDIDATOS_TOTAL // num_cores

        tasks = [
            (
                chunk,
                config.ticket_size,
                config.total_balls,
                weights_dict,
                seeds_pool,
                filter_config,
            )
            for _ in range(num_cores)
        ]

        global_elite = []
        with Pool(processes=num_cores) as pool:
            results = pool.map(worker_monte_carlo_smart, tasks)
            for res in results:
                global_elite.extend(res)

        global_elite.sort(key=lambda x: x[0], reverse=True)

        final_tickets = []
        seen_tuples = set()

        for _, nums in global_elite:
            if len(final_tickets) >= config.num_tickets:
                break
            t_tuple = tuple(nums)
            if t_tuple not in seen_tuples:
                final_tickets.append(nums)
                seen_tuples.add(t_tuple)

        return PredictionResultDTO("Monte Carlo V2", final_tickets)
