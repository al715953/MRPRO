import random
import multiprocessing
from multiprocessing import Pool, cpu_count
from collections import Counter
from typing import List, Tuple, Dict, Any
from src.domain.interfaces import ILotteryStrategy
from src.domain.dtos import DrawHistoryDTO, PredictionConfigDTO, PredictionResultDTO

NUM_SIMULACIONES = 200_000
NUM_CANDIDATOS = 10_000_000

def worker_monte_carlo_batch(args: Tuple) -> List[Tuple[int, List[int]]]:
    count, ticket_size, all_numbers, heat_map = args
    local_candidates = []
    get_score = heat_map.get
    
    for _ in range(count):
        candidate = sorted(random.sample(all_numbers, ticket_size))
        # Filtro paridad
        evens = sum(1 for n in candidate if n % 2 == 0)
        if evens < 2 or evens > 4: continue 
            
        score = sum(get_score(num, 0) for num in candidate)
        local_candidates.append((score, candidate))
    
    local_candidates.sort(key=lambda x: x[0], reverse=True)
    return local_candidates[:1000]

class MonteCarloStrategy(ILotteryStrategy):
    def predict(self, history: DrawHistoryDTO, config: PredictionConfigDTO) -> PredictionResultDTO:
        all_numbers = list(range(1, config.total_balls + 1))
        
        # 1. Mapa de Calor
        real_freq = Counter()
        for nums in history.winning_numbers: real_freq.update(nums)
        weights = [real_freq.get(n, 1) for n in all_numbers]

        sim_heat_map = Counter()
        for _ in range(NUM_SIMULACIONES):
            sim_draw = random.choices(all_numbers, weights=weights, k=config.ticket_size)
            sim_heat_map.update(set(sim_draw))
            
        heat_map_dict = dict(sim_heat_map)

        # 2. Multiprocesamiento
        num_cores = cpu_count()
        chunk = NUM_CANDIDATOS // num_cores
        
        tasks_args = [(chunk, config.ticket_size, all_numbers, heat_map_dict) for _ in range(num_cores)]
        final_candidates = []

        with Pool(processes=num_cores) as pool:
            results = pool.map(worker_monte_carlo_batch, tasks_args)
            for res in results: final_candidates.extend(res)

        # 3. Selección (Se eliminó el print de consolidación)
        final_candidates.sort(key=lambda x: x[0], reverse=True)
        selected_tickets = [t for s, t in final_candidates[:config.num_tickets]]
        
        return PredictionResultDTO(
            strategy_name=f"Monte Carlo Extreme ({NUM_CANDIDATOS/1_000_000:.0f}M)",
            tickets=selected_tickets
        )