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
    ESTRATEGIA 'CENTAURO' V7 (Hybrid Ensemble) - ULTIMATE EDITION.

    Arquitectura de 3 Capas de Blindaje:
    1. VELOCIDAD: Cálculo Heurístico Vectorizado (Numpy).
    2. ROBUSTEZ: Protocolo de Consenso de Estabilidad (5 Expertos IA).
    3. DIVERSIDAD: Filtro 'Hard Cap Anti-Monopolio' para evitar fijación numérica.
    """

    def predict(
        self, history: DrawHistoryDTO, config: PredictionConfigDTO
    ) -> PredictionResultDTO:
        # --- CONFIGURACIÓN DE ROBUSTEZ ---
        N_ROUNDS = 5  # Número de "expertos" (IAs independientes)
        MIN_CONSENSUS = 3  # Votos mínimos para considerar un ticket "Fuerte"
        TOP_CANDIDATES_PER_ROUND = 60  # Cuántos tickets nomina cada experto

        console.print(
            f"\n[bold yellow]🧬 INICIANDO PROTOCOLO CENTAURO V7 (Full Spectrum)...[/bold yellow]"
        )

        # 1. CARGAR UNIVERSO (Modo Vectorial)
        csv_path = os.path.join("data", "universo_reducido.csv")
        if not os.path.exists(csv_path):
            console.print(
                "[red]❌ No se encontró el universo reducido (Ejecuta Fase 1 primero).[/red]"
            )
            return PredictionResultDTO("Error", [])

        try:
            # Cargamos directamente como matriz de Numpy (Int32 es suficiente y rápido)
            df = pd.read_csv(csv_path)
            candidates_np = df.iloc[:, :6].values.astype(int)
            num_candidates = len(candidates_np)

            if num_candidates == 0:
                return PredictionResultDTO("Empty Universe", [])

            console.print(f"   📥 Universo Cargado: {num_candidates:,} tickets.")

        except Exception as e:
            console.print(f"[red]❌ Error leyendo CSV: {e}[/red]")
            return PredictionResultDTO("Error", [])

        # --- FASE PREVIA: CÁLCULO HEURÍSTICO VECTORIZADO (Determinista) ---
        # Calculamos esto UNA sola vez porque los datos históricos no cambian entre rondas.
        console.print(
            "📐 [bold]Pre-calculando Estructura Geométrica (Vectorizado)...[/bold]"
        )

        # A. PREPARAR TABLAS DE BÚSQUEDA (LOOKUP TABLES)

        # 1. Hotness Lookup (Frecuencia Reciente - Últimos 20)
        recent_nums = [n for d in history.winning_numbers[-20:] for n in d[:6]]
        freq_map = Counter(recent_nums)
        max_freq = max(freq_map.values()) if freq_map else 1

        hotness_lookup = np.zeros(config.total_balls + 1, dtype=float)
        for ball, count in freq_map.items():
            if ball <= config.total_balls:
                hotness_lookup[ball] = count

        # 2. Cluster Matrix (Matriz de Adyacencia para Pares)
        cluster_counts = Counter()
        for draw in history.winning_numbers:
            for pair in itertools.combinations(sorted(draw[:6]), 2):
                cluster_counts[pair] += 1
        max_cluster = max(cluster_counts.values()) if cluster_counts else 1

        # Matriz simétrica (40x40)
        cluster_matrix = np.zeros(
            (config.total_balls + 1, config.total_balls + 1), dtype=float
        )
        for (a, b), count in cluster_counts.items():
            if a <= config.total_balls and b <= config.total_balls:
                cluster_matrix[a, b] = count
                cluster_matrix[b, a] = count  # Simetría

        # 3. Zombie Lookup (Antigüedad)
        last_app = {}
        for idx, draw in enumerate(reversed(history.winning_numbers)):
            for n in draw[:6]:
                if n not in last_app:
                    last_app[n] = idx

        zombie_lookup = np.zeros(config.total_balls + 1, dtype=int)
        for ball, age in last_app.items():
            if ball <= config.total_balls:
                zombie_lookup[ball] = age

        # B. CÁLCULO MASIVO DE SCORES

        # Score Hotness: Suma de frecuencias normalizada
        h_vals = hotness_lookup[candidates_np].sum(axis=1)
        norm_hot = h_vals / (6 * max_freq)

        # Score Clusters: Suma de pesos de pares
        # Iteramos sobre las 15 combinaciones de columnas (indices 0..5)
        c_vals = np.zeros(num_candidates, dtype=float)
        for i in range(6):
            for j in range(i + 1, 6):
                col_i = candidates_np[:, i]
                col_j = candidates_np[:, j]
                c_vals += cluster_matrix[col_i, col_j]

        norm_cluster = c_vals / (15 * max_cluster)

        # Penalización Zombie (>20 sorteos sin salir)
        ages = zombie_lookup[candidates_np]
        zombie_counts = (ages > 20).sum(axis=1)
        penalties = np.where(zombie_counts > 2, 0.1, 0.0)

        # Heurística Base (60% Cluster, 40% Hotness)
        # Este score es fijo para todos los rounds
        base_heuristic_score = (norm_cluster * 0.6) + (norm_hot * 0.4)

        # --- BUCLE DE ESTABILIDAD (CONSENSUS LOOP) ---
        ticket_votes = Counter()

        # Preparamos lista de tuplas una sola vez para pasarla al scorer (interfaz compatibilidad)
        candidates_tuples = [tuple(row) for row in candidates_np]

        # Iteramos N veces para eliminar el factor suerte del ruido de la IA
        for round_i in track(
            range(N_ROUNDS), description="🗳️  Consultando Expertos (AI Consensus)..."
        ):

            # 1. Nueva IA (Nuevo Ruido Inteligente)
            ai_engine = LotteryAIModel()
            ai_engine.train(history.winning_numbers, config.total_balls)

            # 2. Score IA
            ai_scores = ai_engine.score_tickets(candidates_tuples)

            # 3. Fusión Híbrida de esta ronda
            # Heurística (60%) + IA (40%) - Penalización
            final_scores = (
                (base_heuristic_score * 0.60) + (ai_scores * 0.40) - penalties
            )

            # 4. Votación de esta ronda
            # Obtenemos los índices de los mejores candidatos
            sorted_indices = np.argsort(final_scores)[::-1]  # Descendente
            top_indices = sorted_indices[:TOP_CANDIDATES_PER_ROUND]

            for idx in top_indices:
                t_tuple = candidates_tuples[idx]
                ticket_votes[t_tuple] += 1

        # --- SELECCIÓN FINAL (HARD CAP DIVERSITY) ---
        console.print(f"   ⚖️  Filtrando por Consenso y Diversidad Forzada...")

        # 1. Obtener TODOS los candidatos votados (no solo los del consenso estricto)
        # Esto asegura que tengamos una piscina profunda para pescar si los tops se agotan por el hard cap.
        all_candidates = [t for t, votes in ticket_votes.most_common()]

        final_selection = []
        seen_tickets = []

        # --- HARD CAP: Límite estricto de apariciones ---
        # 4 es un buen número para 15 tickets. Ningún número dominará más del 25-30%
        MAX_USAGES = 4
        global_number_usage = Counter()

        # Primera Pasada: Búsqueda Estricta (Respetando el Hard Cap)
        for ticket in all_candidates:
            if len(final_selection) >= config.num_tickets:
                break

            # A. Filtro de Diversidad (Ticket vs Ticket) - Evitar clones
            is_diverse = True
            ticket_set = set(ticket)
            for picked in seen_tickets:
                if len(ticket_set & set(picked)) >= 5:  # Muy parecidos
                    is_diverse = False
                    break
            if not is_diverse:
                continue

            # B. Filtro Anti-Monopolio (HARD CAP)
            # Si ALGUNA bola del ticket ya excedió su uso, DESCARTAMOS el ticket.
            has_exhausted_ball = False
            for ball in ticket:
                if global_number_usage[ball] >= MAX_USAGES:
                    has_exhausted_ball = True
                    break

            if has_exhausted_ball:
                continue  # Sin piedad: salta al siguiente candidato para buscar variedad

            # C. Aprobado
            final_selection.append(ticket)
            seen_tickets.append(ticket)
            for ball in ticket:
                global_number_usage[ball] += 1

        # Fallback: Si fuimos demasiado estrictos y no llenamos los 15, rellenamos con lo mejor disponible
        # (Relajando el Anti-Monopolio, pero manteniendo diversidad de tickets)
        if len(final_selection) < config.num_tickets:
            console.print(
                f"[dim]   ⚠️ Rellenando {config.num_tickets - len(final_selection)} cupos con reservas (Relajando Hard Cap)...[/dim]"
            )
            for ticket in all_candidates:
                if len(final_selection) >= config.num_tickets:
                    break

                if ticket in seen_tickets:
                    continue

                # Solo chequeo de clonación, ignoramos monopolio aquí
                ticket_set = set(ticket)
                is_diverse = True
                for picked in seen_tickets:
                    if len(ticket_set & set(picked)) >= 5:
                        is_diverse = False
                        break

                if is_diverse:
                    final_selection.append(ticket)
                    seen_tickets.append(ticket)

        console.print(
            f"[bold green]✅ CENTAURO V7 COMPLETADO: {len(final_selection)} tickets generados (Hard Diversity).[/]"
        )
        return PredictionResultDTO("Centaur V7 (Hard Cap)", final_selection)
