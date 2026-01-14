import pandas as pd
import numpy as np
import os
import itertools
from collections import Counter
from typing import List, Tuple, Dict, Optional
from rich.console import Console

from src.domain.interfaces import ILotteryStrategy
from src.domain.dtos import DrawHistoryDTO, PredictionConfigDTO, PredictionResultDTO
from src.core.ai_scorer import LotteryAIModel

console = Console()


class GeneticSelectorStrategy(ILotteryStrategy):
    """
    ESTRATEGIA 'CENTAURO' V11 (Floating Thresholds).

    MEJORAS V11:
    - Umbrales Dinámicos: Los cortes Elite/Mid ya no son fijos (0.70/0.60).
      Se leen de la configuración para adaptarse a la "realidad del mercado".
      Si los ganadores promedian 0.62, bajamos el Elite a 0.65 para descongestionar Mid.
    """

    def __init__(self):
        self.ai_model = LotteryAIModel()
        self._last_trained_date = None
        self._cache_metrics = {
            "cluster_counts": None,
            "max_cluster": 1,
            "freq_map": None,
            "max_freq": 1,
        }

    def _train_model(self, history: DrawHistoryDTO, total_balls: int):
        last_date = history.dates[-1] if history.dates else "None"
        if self._last_trained_date != last_date:
            self.ai_model.train(history.winning_numbers, total_balls)
            self._last_trained_date = last_date
            self._update_heuristic_metrics(history)

    def _update_heuristic_metrics(self, history: DrawHistoryDTO):
        cluster_counts = Counter()
        for draw in history.winning_numbers:
            for pair in itertools.combinations(sorted(draw[:6]), 2):
                cluster_counts[pair] += 1

        recent_nums = [n for d in history.winning_numbers[-15:] for n in d[:6]]
        freq_map = Counter(recent_nums)

        self._cache_metrics["cluster_counts"] = cluster_counts
        self._cache_metrics["max_cluster"] = (
            max(cluster_counts.values()) if cluster_counts else 1
        )
        self._cache_metrics["freq_map"] = freq_map
        self._cache_metrics["max_freq"] = max(freq_map.values()) if freq_map else 1

    def _compute_v7_score(
        self, ticket: Tuple[int, ...], ai_score: float, weights: Dict[str, float]
    ) -> Tuple[float, float, float]:
        cc = self._cache_metrics["cluster_counts"]
        mc = self._cache_metrics["max_cluster"]
        fm = self._cache_metrics["freq_map"]
        mf = self._cache_metrics["max_freq"]

        # A. Heurística
        c_score = sum(cc.get(pair, 0) for pair in itertools.combinations(ticket, 2))
        norm_cluster = c_score / (15 * mc)

        h_score = sum(fm.get(n, 0) for n in ticket)
        norm_hot = h_score / (6 * mf)

        if norm_hot > 0.75:
            norm_hot *= 0.80

        w_cluster = weights.get("w_cluster", 0.6)
        w_hotness = weights.get("w_hotness", 0.4)
        w_ai_global = weights.get("w_ai", 0.3)
        w_heur_global = 1.0 - w_ai_global

        if w_ai_global == 0.3:
            w_heur_global = 0.7

        heur_val = (norm_cluster * w_cluster) + (norm_hot * w_hotness)
        final_score = (heur_val * w_heur_global) + (ai_score * w_ai_global)

        return final_score, heur_val, ai_score

    def predict(
        self, history: DrawHistoryDTO, config: PredictionConfigDTO
    ) -> PredictionResultDTO:
        overrides = getattr(config, "filter_overrides", {})
        verbose = overrides.get("verbose", True)

        # --- LECTURA DE UMBRALES DINÁMICOS ---
        th_elite = overrides.get("threshold_elite", 0.70)  # Default histórico
        th_mid = overrides.get("threshold_mid", 0.60)  # Default histórico

        if verbose:
            console.print(
                f"\n[bold yellow]🧬 CENTAURO V11 (Floating Thresholds E:{th_elite} M:{th_mid})...[/]"
            )

        # Cargar Universo
        csv_path = os.path.join("data", "universo_reducido.csv")
        if not os.path.exists(csv_path):
            return PredictionResultDTO("Error: No Universe", [])
        try:
            df = pd.read_csv(csv_path)
            candidates = [tuple(x) for x in df.iloc[:, :6].values.astype(int)]
        except:
            return PredictionResultDTO("Error: CSV Bad Format", [])
        if not candidates:
            return PredictionResultDTO("Empty Universe", [])

        self._train_model(history, config.total_balls)

        raw_ai_scores = self.ai_model.score_tickets(candidates)
        scored_candidates = []

        scoring_weights = {
            "w_cluster": overrides.get("w_cluster", 0.6),
            "w_hotness": overrides.get("w_hotness", 0.4),
            "w_ai": overrides.get("w_ai", 0.3),
        }

        for i, ticket in enumerate(candidates):
            final, _, _ = self._compute_v7_score(
                ticket, raw_ai_scores[i], scoring_weights
            )
            scored_candidates.append((final, ticket))

        # --- BUCKETING DINÁMICO ---
        bucket_elite = []
        bucket_mid = []
        bucket_low = []

        for item in scored_candidates:
            s = item[0]
            # Usamos las variables en lugar de hardcode
            if s >= th_elite:
                bucket_elite.append(item)
            elif th_mid <= s < th_elite:
                bucket_mid.append(item)
            elif (th_mid - 0.10) <= s < th_mid:  # Low es Mid - 0.10
                bucket_low.append(item)

        bucket_elite.sort(key=lambda x: x[0], reverse=True)
        bucket_mid.sort(key=lambda x: x[0], reverse=True)
        bucket_low.sort(key=lambda x: x[0], reverse=True)

        q_elite = overrides.get("quota_elite", 2)
        q_mid = overrides.get("quota_mid", 6)
        q_low = overrides.get("quota_low", 7)

        final_selection = []
        seen_tickets = []

        quotas = [
            (bucket_elite, q_elite, "Elite"),
            (bucket_mid, q_mid, "Mid"),
            (bucket_low, q_low, "Low"),
        ]

        counts = {"Elite": 0, "Mid": 0, "Low": 0}

        for bucket, quota, tier_name in quotas:
            for score, ticket in bucket:
                if counts[tier_name] >= quota:
                    break

                ticket_set = set(ticket)
                is_clone = any(
                    len(ticket_set.intersection(set(existing))) >= 5
                    for existing in seen_tickets
                )

                if not is_clone:
                    final_selection.append(ticket)
                    seen_tickets.append(ticket)
                    counts[tier_name] += 1

        if len(final_selection) < config.num_tickets:
            if verbose:
                console.print("   ⚠ Relleno de Emergencia...")
            all_remaining = sorted(scored_candidates, key=lambda x: x[0], reverse=True)
            for score, ticket in all_remaining:
                if len(final_selection) >= config.num_tickets:
                    break
                if ticket not in seen_tickets:
                    if not any(
                        len(set(ticket).intersection(set(e))) >= 5 for e in seen_tickets
                    ):
                        final_selection.append(ticket)
                        seen_tickets.append(ticket)

        if verbose:
            console.print(
                f"[bold green]✅ CENTAURO COMPLETADO: {len(final_selection)} tickets.[/]"
            )
        return PredictionResultDTO("Centaur V11", final_selection)

    def _analyze_tier_distribution(
        self,
        universe_candidates: List[Tuple[int, ...]],
        winning_set: set,
        scores: List[float],
        th_elite: float,
        th_mid: float,
    ) -> str:
        """Radar de Estratos Dinámico."""
        hits_map = {4: [], 5: [], 6: []}

        for i, ticket in enumerate(universe_candidates):
            hits = len(set(ticket) & winning_set)
            if hits >= 4:
                hits_map[hits].append(scores[i])

        msg = "\n   📡 [bold purple]RADAR DE ESTRATOS (Zona de Impacto):[/]\n"

        found_any = False
        for h in [6, 5, 4]:
            scores_list = hits_map[h]
            if not scores_list:
                continue
            found_any = True

            avg_score = sum(scores_list) / len(scores_list)
            min_s = min(scores_list)
            max_s = max(scores_list)

            # Clasificación visual basada en los umbrales actuales
            tier = "Low"
            if avg_score >= th_elite:
                tier = f"Elite (>={th_elite})"
            elif avg_score >= th_mid:
                tier = f"Mid ({th_mid}-{th_elite})"

            color = (
                "green" if "Elite" in tier else ("yellow" if "Mid" in tier else "cyan")
            )

            msg += f"      🔹 [bold]{h} Aciertos:[/bold] {len(scores_list)} tickets.\n"
            msg += f"          Avg Score: [{color}]{avg_score:.5f}[/] (Rango: {min_s:.4f} - {max_s:.4f})\n"
            msg += f"          🎯 Zona: [bold {color}]{tier}[/]\n"

        if not found_any:
            msg += "      (No se encontraron tickets con 4+ aciertos en el universo filtrado)\n"
        return msg

    def audit_winner(
        self,
        history: DrawHistoryDTO,
        config: PredictionConfigDTO,
        winning_ticket: List[int],
    ) -> str:
        """MÉTODO FORENSE V7 (Dynamic Thresholds Support)."""
        target_set = set(winning_ticket[:6])
        target_tuple = tuple(sorted(winning_ticket[:6]))
        overrides = getattr(config, "filter_overrides", {})

        th_elite = overrides.get("threshold_elite", 0.70)
        th_mid = overrides.get("threshold_mid", 0.60)

        csv_path = os.path.join("data", "universo_reducido.csv")
        try:
            df = pd.read_csv(csv_path)
            universe_candidates = [tuple(x) for x in df.iloc[:, :6].values.astype(int)]
        except:
            return "[red]Error leyendo universo[/]"

        scoring_weights = {
            "w_cluster": overrides.get("w_cluster", 0.6),
            "w_hotness": overrides.get("w_hotness", 0.4),
            "w_ai": overrides.get("w_ai", 0.3),
        }

        self._train_model(history, config.total_balls)
        raw_ai_scores = self.ai_model.score_tickets(universe_candidates)
        universe_final_scores = []
        for i, t in enumerate(universe_candidates):
            s, _, _ = self._compute_v7_score(t, raw_ai_scores[i], scoring_weights)
            universe_final_scores.append(s)

        subject_ticket = None
        subject_hits = 0
        if target_tuple in set(universe_candidates):
            subject_ticket = target_tuple
            subject_hits = 6
            status_msg = "[bold green]PRESENTE (Jackpot 6/6)[/]"
        else:
            best_ticket = None
            max_hits = -1
            for t in universe_candidates:
                h = len(set(t) & target_set)
                if h > max_hits:
                    max_hits = h
                    best_ticket = t
            subject_ticket = best_ticket
            subject_hits = max_hits
            status_msg = f"[bold yellow]AUSENTE (Mejor: {max_hits} hits)[/]"

        if not subject_ticket:
            return "[red]Universo vacío.[/]"
        clean_ticket = tuple(int(x) for x in subject_ticket)
        idx = universe_candidates.index(subject_ticket)
        subject_score = universe_final_scores[idx]

        radar_msg = self._analyze_tier_distribution(
            universe_candidates, target_set, universe_final_scores, th_elite, th_mid
        )

        old_verbose = overrides.get("verbose", True)
        overrides["verbose"] = False
        result = self.predict(history, config)
        overrides["verbose"] = old_verbose

        if not result.tickets:
            return "Error sim."
        selected_tuples = [tuple(t) for t in result.tickets]

        # Clasificación Bucket Real
        target_bucket = "Low"
        if subject_score >= th_elite:
            target_bucket = "Elite"
        elif subject_score >= th_mid:
            target_bucket = "Mid"

        assassin = None
        ai_scores_sel = self.ai_model.score_tickets(selected_tuples)
        min_sel_score = 1.0

        for i, sel in enumerate(selected_tuples):
            s_final, _, _ = self._compute_v7_score(
                sel, ai_scores_sel[i], scoring_weights
            )
            if s_final < min_sel_score:
                min_sel_score = s_final
            if s_final > subject_score and len(set(subject_ticket) & set(sel)) >= 5:
                assassin = sel
                break

        msg = f"\n   🕵️‍♂️ [bold cyan]REPORTE FORENSE V7 (Dynamic):[/]\n"
        msg += f"   🎯 [bold]Sujeto:[/bold] {clean_ticket} ({subject_hits} Hits)\n"
        msg += f"   📊 [bold]Score:[/bold] {subject_score:.5f} | [bold]Bucket:[/bold] {target_bucket}\n"
        msg += f"   📏 [bold]Umbrales:[/bold] E>={th_elite}, M>={th_mid}\n"
        msg += f"   🌌 [bold]Status Universo:[/bold] {status_msg}\n"
        msg += radar_msg + "\n"

        if subject_ticket in selected_tuples:
            msg += f"   🎉 [bold green]RESULTADO: ¡CAPTURADO![/]"
        elif assassin:
            msg += f"   💀 [bold red]CAUSA: Canibalismo[/]"
        elif min_sel_score > subject_score:
            msg += f"   📉 [bold red]CAUSA: Score Insuficiente (Corte: {min_sel_score:.5f})[/]"
        else:
            msg += f"   📉 [bold red]CAUSA: Desplazamiento ({target_bucket} Lleno)[/]"

        return msg
