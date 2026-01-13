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
from src.core.ai_scorer import LotteryAIModel

console = Console()


class GeneticSelectorStrategy(ILotteryStrategy):
    """
    ESTRATEGIA 'CENTAURO' V7 (Hybrid Ensemble).

    Fusión de Inteligencia:
    1. BASE HEURÍSTICA (60%): Aporta estabilidad estructural (Clusters + Hotness).
       Probado: Genera más tickets de 3 y 2 aciertos.
    2. REFINAMIENTO IA (40%): Aporta visión temporal y patrones no lineales.
       Probado: Detecta tendencias de decaimiento que la heurística ignora.
    """

    def predict(
        self, history: DrawHistoryDTO, config: PredictionConfigDTO
    ) -> PredictionResultDTO:
        console.print(
            f"\n[bold yellow]🧬 INICIANDO PROTOCOLO CENTAURO V7 (Hybrid AI + Heuristic)...[/bold yellow]"
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

        # --- COMPONENTE 1: INTELIGENCIA ARTIFICIAL (CEREBRO) ---
        console.print("🤖 [bold]FASE 1:[/bold] Ejecutando Red Neuronal Temporal...")
        ai_engine = LotteryAIModel()
        ai_engine.train(history.winning_numbers, config.total_balls)

        # Obtenemos probabilidades puras (0.0 - 1.0)
        ai_scores = ai_engine.score_tickets(candidates)

        # --- COMPONENTE 2: HEURÍSTICA CLÁSICA (CUERPO) ---
        console.print("📐 [bold]FASE 2:[/bold] Calculando Estructura Geométrica...")

        # A. Clusters (Pares Históricos)
        cluster_counts = Counter()
        for draw in history.winning_numbers:
            for pair in itertools.combinations(sorted(draw[:6]), 2):
                cluster_counts[pair] += 1
        max_cluster = max(cluster_counts.values()) if cluster_counts else 1

        # B. Hotness (Tendencia Reciente - Últimos 20)
        recent_nums = [n for d in history.winning_numbers[-20:] for n in d[:6]]
        freq_map = Counter(recent_nums)
        max_freq = max(freq_map.values()) if freq_map else 1

        # C. Zombies (Para penalización ligera)
        last_app = {}
        for idx, draw in enumerate(reversed(history.winning_numbers)):
            for n in draw[:6]:
                if n not in last_app:
                    last_app[n] = idx

        # --- FASE 3: FUSIÓN DE SCORES (ENSAMBLE) ---
        hybrid_candidates = []

        # Iteramos una sola vez combinando todo
        # Usamos zip para iterar ticket y su score de IA al mismo tiempo
        for i, (ticket, ai_prob) in enumerate(zip(candidates, ai_scores)):

            # --- CALCULO HEURÍSTICO ---
            # Score Clusters
            c_val = sum(
                cluster_counts.get(pair, 0)
                for pair in itertools.combinations(ticket, 2)
            )
            norm_cluster = c_val / (15 * max_cluster)  # 0.0 a 1.0 aprox

            # Score Hotness
            h_val = sum(freq_map.get(n, 0) for n in ticket)
            norm_hot = h_val / (6 * max_freq)  # 0.0 a 1.0 aprox

            # Score Heurístico Total (La fórmula ganadora del round anterior)
            heuristic_score = (norm_cluster * 0.6) + (norm_hot * 0.4)

            # Penalización suave de Zombies (solo extremos)
            zombies = sum(1 for n in ticket if last_app.get(n, 0) > 20)
            penalty = 0.1 if zombies > 2 else 0

            # --- ECUACIÓN MAESTRA V7 ---
            # Heurística (Estabilidad): 60%
            # IA (Tendencia): 40%
            final_score = (heuristic_score * 0.60) + (ai_prob * 0.40) - penalty

            hybrid_candidates.append((final_score, ticket))

        # --- SELECCIÓN FINAL ---
        hybrid_candidates.sort(key=lambda x: x[0], reverse=True)

        selection = []
        seen_tickets = []

        console.print(f"   ⚖️  Fusionando {len(hybrid_candidates):,} candidatos...")

        for score, ticket in hybrid_candidates:
            if len(selection) >= config.num_tickets:
                break

            # Filtro Diversidad
            is_diverse = True
            ticket_set = set(ticket)
            for picked in seen_tickets:
                if len(ticket_set & set(picked)) >= 5:
                    is_diverse = False
                    break

            if is_diverse:
                selection.append(ticket)
                seen_tickets.append(ticket)

        console.print(
            f"[bold green]✅ CENTAURO COMPLETADO: {len(selection)} tickets híbridos generados.[/]"
        )
        return PredictionResultDTO("Centaur V7 (Hybrid)", selection)
