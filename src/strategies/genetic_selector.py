import pandas as pd
import numpy as np
import os
import itertools
from collections import Counter
from typing import List, Tuple, Dict
from colorama import Fore, Style

from src.domain.interfaces import ILotteryStrategy
from src.domain.dtos import DrawHistoryDTO, PredictionConfigDTO, PredictionResultDTO


class GeneticSelectorStrategy(ILotteryStrategy):
    """
    ESTRATEGIA 'EL FRANCOTIRADOR' (V2 - Aumentada).
    Selecciona los mejores tickets basándose en:
    1. ADN Histórico (Clústers de pares).
    2. Rescate de Zombies (Números rezagados).
    3. Números Calientes (Tendencia individual).
    """

    def predict(
        self, history: DrawHistoryDTO, config: PredictionConfigDTO
    ) -> PredictionResultDTO:
        print(
            f"\n{Fore.MAGENTA}🧬 INICIANDO SELECTOR GENÉTICO (Sniper V2)...{Style.RESET_ALL}"
        )

        # 1. CARGAR EL UNIVERSO
        csv_path = os.path.join("data", "universo_reducido.csv")
        if not os.path.exists(csv_path):
            print(f"{Fore.RED}❌ ERROR: No se encontró '{csv_path}'.")
            print(
                "Ejecuta primero la Opción 5 para generar el universo.{Style.RESET_ALL}"
            )
            return PredictionResultDTO("Error", [])

        print(f"📂 Cargando universo desde: {csv_path}...")
        try:
            df = pd.read_csv(csv_path)
            candidates = df.iloc[:, :6].values.tolist()
            print(f"✅ Candidatos cargados: {len(candidates):,}")
        except Exception as e:
            print(f"{Fore.RED}❌ Error leyendo CSV: {e}{Style.RESET_ALL}")
            return PredictionResultDTO("Error", [])

        # 2. ANÁLISIS DE INTELIGENCIA (Recalculando métricas clave)
        print("microscopio🔬 Analizando prioridades estratégicas...")

        # A. Detectar Zombies (Igual que en MonteCarlo)
        last_appearance = {n: 999 for n in range(1, config.total_balls + 1)}
        all_draws = history.winning_numbers
        current_draw_idx = len(all_draws)

        for idx, draw in enumerate(reversed(all_draws)):
            for num in draw[:6]:
                if last_appearance[num] == 999:
                    last_appearance[num] = idx

        zombies = {n for n, gap in last_appearance.items() if gap > 18}
        print(f"   🧟 Zombies detectados: {len(zombies)}")

        # B. Detectar Hot Numbers (Top 10 frecuencia reciente)
        recent_flat = [n for draw in all_draws[-20:] for n in draw[:6]]
        hot_counts = Counter(recent_flat)
        hot_numbers = {n for n, _ in hot_counts.most_common(10)}
        print(f"   🔥 Hot Numbers detectados: {len(hot_numbers)}")

        # C. Mapa de Clústers (ADN de Pares)
        global_clusters = Counter()
        for draw in all_draws:
            for pair in itertools.combinations(sorted(draw[:6]), 2):
                global_clusters[pair] += 1

        # Mapa de Puntuación Base
        score_map = dict(global_clusters)

        # 3. TORNEO DE SELECCIÓN (SCORING MULTI-FACTOR)
        print("🏆 Calculando puntajes evolutivos...")

        scored_candidates = []

        for ticket in candidates:
            ticket = sorted(ticket)
            ticket_set = set(ticket)
            score = 0

            # Factor 1: Fuerza de Pares (La base estructural)
            for pair in itertools.combinations(ticket, 2):
                if pair in score_map:
                    score += score_map[pair]

            # Factor 2: Bonus Zombie (Vital para alinear con Fase 1)
            # Si tiene al menos un zombie, damos un empujón fuerte
            zombie_count = len(ticket_set.intersection(zombies))
            if zombie_count > 0:
                score += 150  # Bonus significativo para rescatarlo del fondo

            # Factor 3: Bonus Hot (Sincronía con tendencia)
            hot_count = len(ticket_set.intersection(hot_numbers))
            score += hot_count * 50

            scored_candidates.append((score, ticket))

        # Ordenar por Score Descendente
        scored_candidates.sort(key=lambda x: x[0], reverse=True)

        # 4. SELECCIÓN CON DIVERSIDAD
        final_selection = []
        seen_tickets = []

        print(f"⚔️  Seleccionando los {config.num_tickets} boletos de élite...")

        for score, ticket in scored_candidates:
            if len(final_selection) >= config.num_tickets:
                break

            # Filtro de Diversidad: Evitar boletos "gemelos"
            is_diverse = True
            ticket_set = set(ticket)

            for picked in seen_tickets:
                picked_set = set(picked)
                # Si comparten 5 números, son demasiado parecidos -> Descartar
                if len(ticket_set & picked_set) >= 5:
                    is_diverse = False
                    break

            if is_diverse:
                final_selection.append(ticket)
                seen_tickets.append(ticket)

        print(f"{Fore.GREEN}✅ SELECCIÓN COMPLETADA.{Style.RESET_ALL}")

        # Reporte rápido de lo seleccionado
        zombie_presence = sum(
            1 for t in final_selection if set(t).intersection(zombies)
        )
        print(
            f"   📊 Resumen: {zombie_presence} de {len(final_selection)} boletos incluyen Zombies."
        )

        return PredictionResultDTO(
            strategy_name="Genetic Selector V2 (Zombie Aware)", tickets=final_selection
        )
