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
    ESTRATEGIA 'EL FRANCOTIRADOR'.
    No genera números. Lee el 'universo_reducido.csv' y selecciona
    quirúrgicamente los mejores tickets basándose en ADN Histórico (Clústers).
    """

    def predict(
        self, history: DrawHistoryDTO, config: PredictionConfigDTO
    ) -> PredictionResultDTO:
        print(
            f"\n{Fore.MAGENTA}🧬 INICIANDO SELECTOR GENÉTICO (Refinamiento Final)...{Style.RESET_ALL}"
        )

        # 1. CARGAR EL UNIVERSO (LAGO DE PESCA)
        csv_path = os.path.join("data", "universo_reducido.csv")
        if not os.path.exists(csv_path):
            print(f"{Fore.RED}❌ ERROR: No se encontró '{csv_path}'.")
            print(
                "Ejecuta primero la Opción 5/6 para generar el universo.{Style.RESET_ALL}"
            )
            return PredictionResultDTO("Error", [])

        print(f"📂 Cargando universo desde: {csv_path}...")
        df = pd.read_csv(csv_path)

        # Convertir a listas de enteros para procesar
        # Asumimos columnas B1, B2... o simplemente las primeras 6 columnas
        candidates = df.iloc[:, :6].values.tolist()
        print(f"✅ Cargados {len(candidates):,} candidatos para análisis.")

        # 2. CONSTRUIR EL MAPA DE CLÚSTERS (ADN GANADOR)
        print("microscopio🔬 Analizando ADN histórico...")

        # A. Clusters Globales (Toda la historia)
        global_clusters = Counter()
        for draw in history.winning_numbers:
            draw_set = sorted(draw[:6])  # Solo naturales
            for pair in itertools.combinations(draw_set, 2):
                global_clusters[pair] += 1

        # B. Clusters Calientes (Últimos 20 sorteos - Tendencia Reciente)
        recent_clusters = Counter()
        recent_history = (
            history.winning_numbers[-20:]
            if len(history.winning_numbers) > 20
            else history.winning_numbers
        )
        for draw in recent_history:
            draw_set = sorted(draw[:6])
            for pair in itertools.combinations(draw_set, 2):
                recent_clusters[pair] += 3  # ¡Valen triple!

        # Fusionamos los mapas para el Scoring
        # Convertimos a dict para acceso rápido
        score_map = dict(global_clusters)
        for pair, val in recent_clusters.items():
            score_map[pair] = score_map.get(pair, 0) + val

        # 3. TORNEO DE SELECCIÓN (SCORING)
        print("🏆 Calculando puntajes de evolución...")

        scored_candidates = []

        for ticket in candidates:
            ticket = sorted(ticket)  # Asegurar orden
            score = 0

            # Sumar puntos por cada par conocido en la historia
            for pair in itertools.combinations(ticket, 2):
                if pair in score_map:
                    score += score_map[pair]

            scored_candidates.append((score, ticket))

        # Ordenar: Los de mayor puntaje arriba
        scored_candidates.sort(key=lambda x: x[0], reverse=True)

        # 4. SELECCIÓN CON DIVERSIDAD (EVITAR CLONES)
        # No queremos 15 tickets que sean casi iguales (ej. 1,2,3,4,5,6 y 1,2,3,4,5,7)
        # Aplicamos una regla: El siguiente ticket debe diferir en al menos 2 números del anterior.

        final_selection = []
        seen_tickets = []

        print(f"⚔️  Seleccionando los {config.num_tickets} guerreros más fuertes...")

        for score, ticket in scored_candidates:
            if len(final_selection) >= config.num_tickets:
                break

            # Chequeo de diversidad
            is_diverse = True
            ticket_set = set(ticket)

            for picked in seen_tickets:
                picked_set = set(picked)
                # Si comparten 5 o más números, son "hermanos gemelos", lo saltamos
                if len(ticket_set & picked_set) >= 5:
                    is_diverse = False
                    break

            if is_diverse:
                final_selection.append(ticket)
                seen_tickets.append(ticket)

        # --- RESULTADO FINAL ---
        print(f"{Fore.GREEN}✅ SELECCIÓN COMPLETADA.{Style.RESET_ALL}")

        return PredictionResultDTO(
            strategy_name="Genetic Selector (Sniper)", tickets=final_selection
        )
