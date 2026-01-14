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
    
    Características:
    - Ensamble 70% Heurística / 30% IA.
    - Cooling Protocol: Penaliza excesos de 'Hotness' para evitar sobreajuste.
    - Tiered Selection: Selecciona candidatos de 3 estratos (Elite, Mid, Low) para maximizar cobertura.
    """

    def __init__(self):
        self.ai_model = LotteryAIModel()
        self._last_trained_date = None

    def _train_model(self, history: DrawHistoryDTO, total_balls: int):
        """Gestiona el entrenamiento para no repetirlo innecesariamente."""
        last_date = history.dates[-1] if history.dates else "None"
        if self._last_trained_date != last_date:
            self.ai_model.train(history.winning_numbers, total_balls)
            self._last_trained_date = last_date

    def predict(
        self, history: DrawHistoryDTO, config: PredictionConfigDTO
    ) -> PredictionResultDTO:
        console.print(
            f"\n[bold yellow]🧬 INICIANDO PROTOCOLO CENTAURO V7 (Tiered Selection)...[/bold yellow]"
        )

        # 1. CARGAR UNIVERSO
        csv_path = os.path.join("data", "universo_reducido.csv")
        if not os.path.exists(csv_path):
            return PredictionResultDTO("Error: No Universe", [])

        try:
            df = pd.read_csv(csv_path)
            candidates = [tuple(x) for x in df.iloc[:, :6].values.astype(int)]
        except Exception:
            return PredictionResultDTO("Error: CSV Bad Format", [])

        if not candidates:
            return PredictionResultDTO("Empty Universe", [])

        # 2. ENTRENAMIENTO IA (On Demand)
        self._train_model(history, config.total_balls)

        # 3. PREPARAR HEURÍSTICA
        cluster_counts = Counter()
        for draw in history.winning_numbers:
            for pair in itertools.combinations(sorted(draw[:6]), 2):
                cluster_counts[pair] += 1
        max_cluster = max(cluster_counts.values()) if cluster_counts else 1

        recent_nums = [n for d in history.winning_numbers[-15:] for n in d[:6]]
        freq_map = Counter(recent_nums)
        max_freq = max(freq_map.values()) if freq_map else 1
        
        # 4. SCORING MASIVO
        ai_scores = self.ai_model.score_tickets(candidates)
        hybrid_candidates = []

        # Pesos V7 (Ajustados por Forense)
        w_heu = 0.70
        w_ai = 0.30

        for i, ticket in enumerate(candidates):
            # A. Heurística
            c_score = sum(cluster_counts.get(pair, 0) for pair in itertools.combinations(ticket, 2))
            norm_cluster = c_score / (15 * max_cluster)
            
            h_score = sum(freq_map.get(n, 0) for n in ticket)
            norm_hot = h_score / (6 * max_freq)
            
            # --- COOLING CAP (Protocolo V7) ---
            # Penalizamos tickets excesivamente calientes (>0.75)
            if norm_hot > 0.75:
                norm_hot *= 0.80

            heur_val = (norm_cluster * 0.6) + (norm_hot * 0.4)
            
            # B. Fusión
            ai_val = ai_scores[i]
            final_score = (heur_val * w_heu) + (ai_val * w_ai)
            
            hybrid_candidates.append((final_score, ticket))

        # 5. SELECCIÓN POR ESTRATOS (TIERED SELECTION)
        # Objetivo: Romper el "Techo de Cristal" y forzar entrada de tickets 0.50-0.70
        hybrid_candidates.sort(key=lambda x: x[0], reverse=True)
        
        selection = []
        seen_tickets = []
        
        # Cuotas por Estrato (Total 15)
        quota_elite = 4
        quota_mid = 5
        quota_low = 6
        
        c_elite = 0
        c_mid = 0
        c_low = 0
        
        console.print(f"   ⚖️  Aplicando Selección Estratificada (Elite/Mid/Low)...")

        for score, ticket in hybrid_candidates:
            if len(selection) >= config.num_tickets:
                break

            # Filtro Diversidad Endurecido (>=4 clones prohibidos)
            ticket_set = set(ticket)
            if any(len(ticket_set & set(p)) >= 4 for p in seen_tickets):
                continue

            # Clasificación del Candidato
            added = False
            
            # 1. TIER ELITE (Score >= 0.70)
            if score >= 0.70:
                if c_elite < quota_elite:
                    selection.append(ticket)
                    seen_tickets.append(ticket)
                    c_elite += 1
                    added = True
            
            # 2. TIER MID (0.60 <= Score < 0.70)
            elif 0.60 <= score < 0.70:
                if c_mid < quota_mid:
                    selection.append(ticket)
                    seen_tickets.append(ticket)
                    c_mid += 1
                    added = True
                    
            # 3. TIER LOW (0.50 <= Score < 0.60)
            elif 0.50 <= score < 0.60:
                if c_low < quota_low:
                    selection.append(ticket)
                    seen_tickets.append(ticket)
                    c_low += 1
                    added = True

        # RELLENO DE EMERGENCIA (Si faltaron candidatos en algún tier)
        if len(selection) < config.num_tickets:
            for score, ticket in hybrid_candidates:
                if len(selection) >= config.num_tickets: break
                if ticket not in seen_tickets:
                    if not any(len(set(ticket) & set(p)) >= 4 for p in seen_tickets):
                        selection.append(ticket)
                        seen_tickets.append(ticket)

        console.print(
            f"[bold green]✅ CENTAURO COMPLETADO: {len(selection)} tickets (E:{c_elite}, M:{c_mid}, L:{c_low}).[/]"
        )
        return PredictionResultDTO("Centaur V7 (Tiered)", selection)

    def audit_winner(self, history: DrawHistoryDTO, config: PredictionConfigDTO, winning_ticket: List[int]) -> str:
        """
        MÉTODO FORENSE V3 (Consistente con Protocolo de Enfriamiento).
        Calcula el score del ganador aplicando las mismas penalizaciones que la predicción real.
        """
        target = tuple(sorted(winning_ticket[:6]))

        # 1. Verificar Universo
        csv_path = os.path.join("data", "universo_reducido.csv")
        try:
            df = pd.read_csv(csv_path)
            candidates_set = set(tuple(x) for x in df.iloc[:, :6].values.astype(int))
        except:
            return "[red]Error leyendo universo[/]"

        if target not in candidates_set:
            return f"[bold red]❌ El ganador {target} NO estaba en el Universo (Fase 1 falló).[/]"

        # 2. Preparar Datos
        self._train_model(history, config.total_balls)

        cluster_counts = Counter()
        for draw in history.winning_numbers:
            for pair in itertools.combinations(sorted(draw[:6]), 2):
                cluster_counts[pair] += 1
        max_cluster = max(cluster_counts.values()) if cluster_counts else 1

        recent_nums = [n for d in history.winning_numbers[-15:] for n in d[:6]]
        freq_map = Counter(recent_nums)
        max_freq = max(freq_map.values()) if freq_map else 1

        # --- FUNCIÓN INTERNA DE SCORING ---
        def calculate_score_v7(t):
            # A. Heurística
            c = sum(cluster_counts.get(pair, 0) for pair in itertools.combinations(t, 2))
            nc = c / (15 * max_cluster)
            
            h = sum(freq_map.get(n, 0) for n in t)
            nh = h / (6 * max_freq)
            
            # >>> COOLING CAP <<<
            if nh > 0.75:
                nh *= 0.80

            heur = (nc * 0.6) + (nh * 0.4)
            
            # B. IA
            ai = self.ai_model.score_tickets([t])[0]
            
            # C. Fusión 70/30
            final = (heur * 0.70) + (ai * 0.30)
            return final, heur, ai

        # 3. Calcular Score del Ganador Real
        winner_score, w_heur, w_ai = calculate_score_v7(target)

        # 4. Obtener el "Score de Corte" (Simulación)
        old_overrides = config.filter_overrides or {}
        config.filter_overrides = {**old_overrides, "verbose": False}
        
        result = self.predict(history, config)
        config.filter_overrides = old_overrides
        
        if not result.tickets:
            return "Error: No se generaron tickets en la simulación."
            
        # Tomamos el último seleccionado (aprox. el corte inferior)
        last_selected = tuple(result.tickets[-1])
        cutoff_score, _, _ = calculate_score_v7(last_selected)
        
        gap = cutoff_score - winner_score
        
        # 5. Reporte
        msg = f"\n   🕵️‍♂️  [bold cyan]REPORTE FORENSE (Protocolo Enfriamiento):[/bold cyan]\n"
        msg += f"   🎯 [bold]Ganador:[/bold] {target}\n"
        msg += f"   📊 [bold]Score Ganador: {winner_score:.5f}[/] (Heur: {w_heur:.2f}, IA: {w_ai:.2f})\n"
        msg += f"   🚪 [bold]Score de Corte (aprox): {cutoff_score:.5f}[/] (Ticket #15)\n"
        
        if gap > 0:
            msg += f"   ❌ [bold red]Brecha: -{gap:.5f}[/] (Nos faltó esto para entrar)\n"
        else:
            msg += f"   ✅ [bold green]¡ADENTRO! Superamos el corte por +{abs(gap):.5f}[/]\n"
            
        return msg