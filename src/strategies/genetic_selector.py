import pandas as pd
import numpy as np
import os
import itertools
from collections import Counter
from typing import List, Tuple
from colorama import Fore, Style

from src.domain.interfaces import ILotteryStrategy
from src.domain.dtos import DrawHistoryDTO, PredictionConfigDTO, PredictionResultDTO
from src.core.ai_scorer import LotteryAIModel  # <--- NUEVA IMPORTACIÓN


class GeneticSelectorStrategy(ILotteryStrategy):
    """
    ESTRATEGIA 'EL FRANCOTIRADOR' V3 (AI ENHANCED).
    Añade una capa de Inteligencia Artificial al Scoring Genético.
    """

    def predict(
        self, history: DrawHistoryDTO, config: PredictionConfigDTO
    ) -> PredictionResultDTO:
        print(
            f"\n{Fore.MAGENTA}🧬 INICIANDO SELECTOR GENÉTICO V3 (AI-Powered)...{Style.RESET_ALL}"
        )

        # 1. CARGAR UNIVERSO
        csv_path = os.path.join("data", "universo_reducido.csv")
        if not os.path.exists(csv_path):
            print(
                f"{Fore.RED}❌ ERROR: Falta 'universo_reducido.csv'. Ejecuta Opción 5.{Style.RESET_ALL}"
            )
            return PredictionResultDTO("Error", [])

        try:
            # Optimizamos lectura especificando tipos
            df = pd.read_csv(csv_path)
            candidates = [tuple(x) for x in df.iloc[:, :6].values]
            print(f"📂 Universo cargado: {len(candidates):,} tickets")
        except Exception as e:
            print(f"❌ Error CSV: {e}")
            return PredictionResultDTO("Error", [])

        # 2. ENTRENAR IA EN TIEMPO REAL
        ai_engine = LotteryAIModel()
        ai_engine.train(history.winning_numbers, config.total_balls)

        print("🤖 Consultando Oráculo Digital (Scoring AI)...")
        # Obtenemos score de IA para TODO el universo (vectorizado es rápido)
        ai_scores = ai_engine.score_tickets(candidates)

        # 3. ANÁLISIS DE INTELIGENCIA (Clásico)
        print("microscopio🔬 Calculando factores genéticos...")

        # A. Zombies (>18 sorteos)
        last_app = {n: 999 for n in range(1, config.total_balls + 1)}
        for idx, draw in enumerate(reversed(history.winning_numbers)):
            for n in draw[:6]:
                if last_app[n] == 999:
                    last_app[n] = idx
        zombies = {n for n, gap in last_app.items() if gap > 18}

        # B. Hot Numbers (Top 10 últimos 20)
        recent = [n for d in history.winning_numbers[-20:] for n in d[:6]]
        hot_nums = {n for n, _ in Counter(recent).most_common(10)}

        # C. Mapa de Clústers
        cluster_counts = Counter()
        for draw in history.winning_numbers:
            for pair in itertools.combinations(sorted(draw[:6]), 2):
                cluster_counts[pair] += 1

        # Normalizamos scores de clústers para que no eclipsen a la IA
        max_cluster_score = max(cluster_counts.values()) if cluster_counts else 1

        # 4. FUSIÓN DE SCORES (HÍBRIDO)
        scored_candidates = []

        print(f"⚔️  Torneo de Selección (Genética + IA)...")

        for i, ticket in enumerate(candidates):
            ticket_set = set(ticket)

            # --- Score Genético ---
            g_score = 0
            # Pares
            for pair in itertools.combinations(ticket, 2):
                g_score += cluster_counts.get(pair, 0)

            # Normalizar score genético base (0-100 aprox)
            g_score = (g_score / (15 * max_cluster_score)) * 100

            # Bonus Zombie & Hot
            if len(ticket_set & zombies) > 0:
                g_score += 50
            if len(ticket_set & hot_nums) > 0:
                g_score += 30

            # --- Score IA ---
            # ai_scores[i] es probabilidad 0.0-1.0. Lo escalamos a 0-100
            ai_factor = ai_scores[i] * 100

            # --- FORMULA FINAL ---
            # Damos 60% peso a la Estructura (Genética) y 40% a la IA
            final_score = (g_score * 0.6) + (ai_factor * 0.4)

            scored_candidates.append((final_score, ticket))

        # 5. SELECCIÓN FINAL (DIVERSIDAD)
        scored_candidates.sort(key=lambda x: x[0], reverse=True)

        final_selection = []
        seen_tickets = []

        for score, ticket in scored_candidates:
            if len(final_selection) >= config.num_tickets:
                break

            # Filtro Diversidad (No tickets gemelos)
            is_diverse = True
            ticket_set = set(ticket)
            for picked in seen_tickets:
                if len(ticket_set & set(picked)) >= 5:  # Si coinciden 5 números
                    is_diverse = False
                    break

            if is_diverse:
                final_selection.append(ticket)
                seen_tickets.append(ticket)

        print(f"{Fore.GREEN}✅ SELECCIÓN COMPLETADA.{Style.RESET_ALL}")
        return PredictionResultDTO("Genetic Sniper V3 + AI", final_selection)
