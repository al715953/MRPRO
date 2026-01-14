import pandas as pd
import numpy as np
import os
import itertools
from collections import Counter
from typing import List, Tuple
from rich.console import Console
from rich.progress import track

from src.domain.interfaces import ILotteryStrategy
from src.domain.dtos import DrawHistoryDTO, PredictionConfigDTO, PredictionResultDTO

console = Console()

class HeuristicSelectorStrategy(ILotteryStrategy):
    """
    ESTRATEGIA 'CLÁSICA' (DINÁMICA).
    Ahora permite inyección de pesos para optimización.
    """

    def predict(
        self, history: DrawHistoryDTO, config: PredictionConfigDTO
    ) -> PredictionResultDTO:
        # Configuración Dinámica (Defaults clásicos si no se envían overrides)
        overrides = config.filter_overrides or {}
        verbose = overrides.get("verbose", True)
        
        # Pesos (Deben sumar aprox 1.0)
        w_cluster = overrides.get("w_cluster", 0.5)
        w_hotness = overrides.get("w_hotness", 0.3)
        w_balance = overrides.get("w_balance", 0.2)

        if verbose:
            console.print(
                f"\n[bold cyan]📐 HEURÍSTICA (W_Cluster={w_cluster:.2f}, W_Hot={w_hotness:.2f})...[/bold cyan]"
            )

        # 1. CARGAR UNIVERSO
        csv_path = os.path.join("data", "universo_reducido.csv")
        if not os.path.exists(csv_path):
            return PredictionResultDTO("Error", [])

        try:
            df = pd.read_csv(csv_path)
            candidates = [tuple(x) for x in df.iloc[:, :6].values.astype(int)]
        except Exception:
            return PredictionResultDTO("Error", [])

        if not candidates:
            return PredictionResultDTO("Empty Universe", [])

        # 2. PREPARAR CAPAS (Igual que antes)
        cluster_counts = Counter()
        for draw in history.winning_numbers:
            for pair in itertools.combinations(sorted(draw[:6]), 2):
                cluster_counts[pair] += 1
        max_cluster = max(cluster_counts.values()) if cluster_counts else 1

        recent_nums = [n for d in history.winning_numbers[-15:] for n in d[:6]]
        freq_map = Counter(recent_nums)
        max_freq = max(freq_map.values()) if freq_map else 1

        last_app = {}
        for idx, draw in enumerate(reversed(history.winning_numbers)):
            for n in draw[:6]:
                if n not in last_app:
                    last_app[n] = idx

        # 3. SCORING DETERMINISTA
        ranked_candidates = []
        
        # Si NO es verbose (modo optimizador), no mostramos la barra de progreso
        iterator = candidates
        if verbose:
             iterator = track(candidates, description="   📐 Calculando Geometría...")

        for ticket in iterator:
            # Layer 1: Clusters
            c_score = sum(cluster_counts.get(pair, 0) for pair in itertools.combinations(ticket, 2))
            norm_cluster = c_score / (15 * max_cluster)

            # Layer 2: Hotness
            h_score = sum(freq_map.get(n, 0) for n in ticket)
            norm_hot = h_score / (6 * max_freq)

            # Layer 3: Balance
            zombies = sum(1 for n in ticket if last_app.get(n, 0) > 18)
            repeats = sum(1 for n in ticket if last_app.get(n, 0) <= 1)
            
            balance_penalty = 0
            if zombies > 2: balance_penalty += 0.1
            if repeats > 1: balance_penalty += 0.1
            
            # Puntuación usando los pesos inyectados
            final_score = (
                (norm_cluster * w_cluster) + 
                (norm_hot * w_hotness) + 
                (w_balance * (1.0 - balance_penalty)) # Balance penaliza sobre el peso
            )

            ranked_candidates.append((final_score, ticket))

        # 4. SELECCIÓN
        ranked_candidates.sort(key=lambda x: x[0], reverse=True)
        selection = []
        seen = []

        for score, ticket in ranked_candidates:
            if len(selection) >= config.num_tickets:
                break
            ticket_set = set(ticket)
            if not any(len(ticket_set & set(p)) >= 5 for p in seen):
                selection.append(ticket)
                seen.append(ticket)

        return PredictionResultDTO("Heuristic Optimized", selection)