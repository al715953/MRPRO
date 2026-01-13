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
    ESTRATEGIA 'CLÁSICA' (BASELINE SIN IA).

    Recupera la lógica pura de capas heurísticas:
    1. Cluster Score (Fuerza de Pares) - 50%
    2. Hotness Score (Frecuencia Reciente) - 30%
    3. Probabilistic Balance (Ley del Retorno/Zombies) - 20%
    """

    def predict(
        self, history: DrawHistoryDTO, config: PredictionConfigDTO
    ) -> PredictionResultDTO:
        console.print(
            f"\n[bold cyan]📐 INICIANDO SELECTOR HEURÍSTICO (Modo Clásico)...[/bold cyan]"
        )

        # 1. CARGAR UNIVERSO (La misma red que usa la IA)
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

        # 2. PREPARAR CAPAS DE ANÁLISIS

        # A. Mapa de Calor (Topografía)
        # Usamos TODA la historia para los pares (Estructura Rígida)
        cluster_counts = Counter()
        for draw in history.winning_numbers:
            for pair in itertools.combinations(sorted(draw[:6]), 2):
                cluster_counts[pair] += 1
        max_cluster = max(cluster_counts.values()) if cluster_counts else 1

        # B. Frecuencia Reciente (Tendencia)
        # Últimos 15 sorteos
        recent_nums = [n for d in history.winning_numbers[-15:] for n in d[:6]]
        freq_map = Counter(recent_nums)
        max_freq = max(freq_map.values()) if freq_map else 1

        # C. Ley del Retorno (Zombies)
        last_app = {}
        for idx, draw in enumerate(reversed(history.winning_numbers)):
            for n in draw[:6]:
                if n not in last_app:
                    last_app[n] = idx

        # 3. SCORING DETERMINISTA
        ranked_candidates = []

        # Analizamos TODOS los candidatos (sin filtro previo de IA)
        # Esto nos dirá si la IA estaba recortando opciones buenas.
        for ticket in track(candidates, description="   📐 Calculando Geometría..."):

            # Layer 1: Clusters (50 pts)
            c_score = sum(
                cluster_counts.get(pair, 0)
                for pair in itertools.combinations(ticket, 2)
            )
            norm_cluster = c_score / (15 * max_cluster)

            # Layer 2: Hotness (30 pts)
            h_score = sum(freq_map.get(n, 0) for n in ticket)
            norm_hot = h_score / (6 * max_freq)

            # Layer 3: Balance (20 pts)
            # Penalizamos extremos: Ni muy rezagados, ni muy repetidos
            zombies = sum(1 for n in ticket if last_app.get(n, 0) > 18)
            repeats = sum(
                1 for n in ticket if last_app.get(n, 0) <= 1
            )  # Salieron hace 1 sorteo

            balance_penalty = 0
            if zombies > 2:
                balance_penalty += 0.1
            if repeats > 1:
                balance_penalty += 0.1

            # Score Final
            final_score = (
                (norm_cluster * 0.5) + (norm_hot * 0.3) + (0.2 - balance_penalty)
            )

            ranked_candidates.append((final_score, ticket))

        # 4. SELECCIÓN
        ranked_candidates.sort(key=lambda x: x[0], reverse=True)

        selection = []
        seen = []

        for score, ticket in ranked_candidates:
            if len(selection) >= config.num_tickets:
                break

            # Mismo filtro de diversidad
            ticket_set = set(ticket)
            if not any(len(ticket_set & set(p)) >= 5 for p in seen):
                selection.append(ticket)
                seen.append(ticket)

        return PredictionResultDTO("Heuristic V1 (No AI)", selection)
