import random
import multiprocessing
from multiprocessing import Pool, cpu_count
from collections import Counter
from typing import List, Tuple, Dict, Any
from src.domain.interfaces import ILotteryStrategy
from src.domain.dtos import DrawHistoryDTO, PredictionConfigDTO, PredictionResultDTO

NUM_SIMULACIONES = 200000
NUM_CANDIDATOS = 20000000

# Monte Carlo con Filtros Biométricos y Diversificación


def worker_monte_carlo_batch(args: Tuple) -> List[Tuple[int, List[int]]]:
    count, ticket_size, all_numbers, heat_map = args
    local_candidates = []
    get_score = heat_map.get

    for _ in range(count):
        candidate = sorted(random.sample(all_numbers, ticket_size))

        # 1. Filtro de Paridad (Existente)
        evens = sum(1 for n in candidate if n % 2 == 0)
        if evens < 2 or evens > 4:
            continue

        # 2. NUEVO: Filtro de Suma (Melate Retro 39 nms: media ~120)
        suma = sum(candidate)
        if not (80 <= suma <= 160):
            continue

        # 3. NUEVO: Filtro de Consecutivos (No más de 2 seguidos)
        consecutivos = 0
        for i in range(len(candidate) - 1):
            if candidate[i + 1] - candidate[i] == 1:
                consecutivos += 1
        if consecutivos > 2:
            continue

        score = sum(get_score(num, 0) for num in candidate)
        local_candidates.append((score, candidate))

    local_candidates.sort(key=lambda x: x[0], reverse=True)
    return local_candidates[:2000]  # Aumentamos el pool de candidatos


class MonteCarloStrategy(ILotteryStrategy):
    def predict(
        self, history: DrawHistoryDTO, config: PredictionConfigDTO
    ) -> PredictionResultDTO:
        all_numbers = list(range(1, config.total_balls + 1))

        # 1. Mapa de Calor
        real_freq = Counter()
        for nums in history.winning_numbers:
            real_freq.update(nums)
        weights = [real_freq.get(n, 1) for n in all_numbers]

        sim_heat_map = Counter()
        for _ in range(NUM_SIMULACIONES):
            sim_draw = random.choices(
                all_numbers, weights=weights, k=config.ticket_size
            )
            sim_heat_map.update(set(sim_draw))

        heat_map_dict = dict(sim_heat_map)

        # 2. Multiprocesamiento
        num_cores = cpu_count()
        chunk = NUM_CANDIDATOS // num_cores

        tasks_args = [
            (chunk, config.ticket_size, all_numbers, heat_map_dict)
            for _ in range(num_cores)
        ]
        final_candidates = []

        with Pool(processes=num_cores) as pool:
            results = pool.map(worker_monte_carlo_batch, tasks_args)
            for res in results:
                final_candidates.extend(res)

        # 3. Selección
        final_candidates.sort(key=lambda x: x[0], reverse=True)
        selected_tickets = []
        for score, ticket in final_candidates:
            if len(selected_tickets) >= config.num_tickets:
                break

            # Solo añadir si no comparte más de 4 números con un ticket ya seleccionado
            # Esto garantiza que cubras más combinaciones posibles
            if not any(len(set(ticket) & set(st)) > 4 for st in selected_tickets):
                selected_tickets.append(ticket)

        return PredictionResultDTO(
            strategy_name=f"Monte Carlo Extreme ({NUM_CANDIDATOS/1_000_000:.0f}M)",
            tickets=selected_tickets,
        )
