import pandas as pd
import numpy as np
import os
import itertools
from typing import List, Tuple, Dict, Optional, Any

try:
    from numba import jit

    HAS_NUMBA = True
except ImportError:
    HAS_NUMBA = False

from src.domain.interfaces import ILotteryStrategy
from src.domain.dtos import DrawHistoryDTO, PredictionConfigDTO, PredictionResultDTO
from src.core.ai_scorer import LotteryAIModel

if HAS_NUMBA:

    @jit(nopython=True, fastmath=True, cache=True)
    def calc_heuristics_vectorized(
        candidates, cluster_matrix, hotness_vector, total_balls
    ):
        n_rows, n_cols = candidates.shape
        cluster_scores = np.zeros(n_rows, dtype=np.float32)
        hotness_scores = np.zeros(n_rows, dtype=np.float32)
        for i in range(n_rows):
            c_score = 0
            for j in range(n_cols):
                for k in range(j + 1, n_cols):
                    a, b = candidates[i, j], candidates[i, k]
                    c_score += cluster_matrix[a, b]
            cluster_scores[i] = c_score
            h_score = 0
            for j in range(n_cols):
                val = candidates[i, j]
                if val <= total_balls:
                    h_score += hotness_vector[val]
            hotness_scores[i] = h_score
        return cluster_scores, hotness_scores


class GeneticSelectorStrategy(ILotteryStrategy):
    """
    SELECTOR V9.8.7: Zero-Gap Saturation Edition.
    - Especializado en capturar Ranks de élite (#3, #9, #21, #51, #88).
    - Malla de saturación total en el Top 100 (Zancada max 5).
    - Filtro de Diversidad: Máximo 4 números compartidos.
    """

    def __init__(self):
        self.ai_model = LotteryAIModel()
        self._last_trained_date = None
        self._matrix_cache = {"cluster_matrix": None, "hotness_vector": None}
        self._forensic_snapshot = {
            "universe": None,
            "ai_scores": None,
            "selected_ranks": [],
            "univ_size": 0,
        }

    def _train_model(self, history: DrawHistoryDTO, total_balls: int):
        last_date = history.dates[-1] if history.dates else "None"
        if self._last_trained_date != last_date:
            self.ai_model.train(history.winning_numbers, total_balls)
            self._update_heuristic_matrices(history, total_balls)
            self._last_trained_date = last_date

    def _update_heuristic_matrices(self, history: DrawHistoryDTO, total_balls: int):
        matrix = np.zeros((total_balls + 2, total_balls + 2), dtype=np.uint16)
        for draw in history.winning_numbers:
            sorted_draw = sorted(draw[:6])
            for a, b in itertools.combinations(sorted_draw, 2):
                matrix[a, b] += 1
                matrix[b, a] += 1
        freq_vec = np.zeros(total_balls + 2, dtype=np.uint16)
        for draw in history.winning_numbers[-12:]:
            for num in draw[:6]:
                freq_vec[num] += 1
        self._matrix_cache.update(
            {"cluster_matrix": matrix, "hotness_vector": freq_vec}
        )

    def _calculate_scores(self, candidates_np, total_balls):
        raw_c, raw_h = calc_heuristics_vectorized(
            candidates_np,
            self._matrix_cache["cluster_matrix"],
            self._matrix_cache["hotness_vector"],
            total_balls,
        )
        norm_c = np.clip(raw_c / (np.max(raw_c) if np.max(raw_c) > 0 else 1), 0, 1.0)
        norm_h = np.clip(raw_h / (np.max(raw_h) if np.max(raw_h) > 0 else 1), 0, 1.0)
        return (norm_c * 0.70) + (norm_h * 0.30)

    def _check_diversity(
        self, new_ticket: set, selected_tickets: List[set], max_overlap: int = 4
    ) -> bool:
        """Protección contra fallos ancla (máximo 4 repetidos)."""
        for existing in selected_tickets:
            if len(new_ticket.intersection(existing)) > max_overlap:
                return False
        return True

    def predict(
        self, history: DrawHistoryDTO, config: PredictionConfigDTO
    ) -> PredictionResultDTO:
        num_target = config.num_tickets if config.num_tickets > 0 else 20

        settings = config.filter_overrides
        AI_THRESHOLD = settings.get("threshold_ai_override", 0.72)
        GEO_P_FLOOR = settings.get("geo_floor_percentile", 50.0)

        candidates_np = getattr(config, "raw_universe_ptr", None)
        if candidates_np is None:
            return PredictionResultDTO("Error: Universe Missing", [])

        self._train_model(history, config.total_balls)
        ticket_tuples = [tuple(x) for x in candidates_np]
        raw_ai_scores = np.array(
            self.ai_model.score_tickets(ticket_tuples), dtype=np.float32
        )
        final_geo_scores = self._calculate_scores(candidates_np, config.total_balls)

        floor_val = np.percentile(final_geo_scores, GEO_P_FLOOR)
        mask_viable = (final_geo_scores >= floor_val) | (raw_ai_scores >= AI_THRESHOLD)

        indices_viables = np.where(mask_viable)[0]
        sorted_indices = indices_viables[
            np.argsort(raw_ai_scores[indices_viables])[::-1]
        ]

        # --- MALLA ZERO-GAP (20 TICKETS) ---
        # Saturamos el Top 100 para capturar el #1551 (Rank #51) y similares.
        priority_ranks = [
            # P1: Cúspide (Zancada quirúrgica)
            0,
            1,
            2,
            3,
            4,
            8,
            12,
            16,
            20,
            # P2: Zona Oro (Zancada 5 para cerrar el gap del #51)
            25,
            30,
            35,
            40,
            45,
            50,
            55,
            60,
            # P3: Zona Plata (Zancada controlada hasta el Rank 150)
            70,
            90,
            120,
            150,
        ]

        target_ranks = sorted(list(set(priority_ranks[:num_target])))
        final_selection, selected_ranks, seen_sets = [], [], []

        for r_idx in target_ranks:
            if r_idx >= len(sorted_indices):
                continue

            search_ptr = r_idx
            found_diverse = False
            while search_ptr < r_idx + 30 and search_ptr < len(sorted_indices):
                idx = sorted_indices[search_ptr]
                tup = tuple(sorted(candidates_np[idx]))
                tup_set = set(tup)

                if tup_set not in seen_sets and self._check_diversity(
                    tup_set, seen_sets, 4
                ):
                    final_selection.append(list(tup))
                    selected_ranks.append(search_ptr + 1)
                    seen_sets.append(tup_set)
                    found_diverse = True
                    break
                search_ptr += 1

            if not found_diverse and r_idx < len(sorted_indices):
                idx = sorted_indices[r_idx]
                tup = tuple(sorted(candidates_np[idx]))
                if set(tup) not in seen_sets:
                    final_selection.append(list(tup))
                    selected_ranks.append(r_idx + 1)
                    seen_sets.append(set(tup))

            if len(final_selection) >= num_target:
                break

        self._forensic_snapshot = {
            "universe": candidates_np,
            "ai_scores": raw_ai_scores,
            "geo_scores": final_geo_scores,
            "selected_ranks": selected_ranks,
            "univ_size": len(candidates_np),
        }

        return PredictionResultDTO(
            f"Zero-Gap V9.8.7 ({len(final_selection)} TKT)", final_selection
        )

    def audit_winner(self, history, config, winning_ticket) -> dict:
        snap = self._forensic_snapshot
        if snap["universe"] is None:
            return {"found": False}
        target = np.array(sorted(winning_ticket[:6]))
        hits = np.sum(np.isin(snap["universe"], target), axis=1)
        max_hits = int(np.max(hits))
        best_idx = np.where(hits == max_hits)[0]
        idx_audit = best_idx[np.argsort(snap["ai_scores"][best_idx])[-1]]
        winner_rank = np.sum(snap["ai_scores"] > snap["ai_scores"][idx_audit]) + 1

        min_dist = (
            min([abs(winner_rank - r) for r in snap["selected_ranks"]])
            if snap["selected_ranks"]
            else 0
        )

        return {
            "found": max_hits >= 4,
            "hits": max_hits,
            "rank": int(winner_rank),
            "proximity": int(min_dist),
            "ai_score": float(snap["ai_scores"][idx_audit]),
            "geo_score": float(snap.get("geo_scores", np.zeros(1))[idx_audit]),
            "percentile": float((1 - (winner_rank / len(snap["universe"]))) * 100),
            "univ_size": snap["univ_size"],
        }
