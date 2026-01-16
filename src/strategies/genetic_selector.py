import pandas as pd
import numpy as np
import os
import itertools
import json
from typing import List, Tuple, Dict, Optional, Any

# --- CAPA HPC (Numba JIT) ---
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
    SELECTOR V9.7: Dynamic Strided Selection.
    Especializado en cerrar brechas de proximidad mediante muestreo bifocal.
    """

    def __init__(self):
        self.ai_model = LotteryAIModel()
        self._last_trained_date = None
        self._matrix_cache = {"cluster_matrix": None, "hotness_vector": None}
        self._forensic_snapshot = {
            "universe": None,
            "ai_scores": None,
            "selected_ranks": [],
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

    def predict(
        self, history: DrawHistoryDTO, config: PredictionConfigDTO
    ) -> PredictionResultDTO:
        settings = config.filter_overrides
        AI_THRESHOLD = settings.get("threshold_ai_override", 0.72)
        GEO_P_FLOOR = settings.get("geo_floor_percentile", 50.0)

        TOTAL_TICKETS = config.num_tickets

        candidates_np = getattr(config, "raw_universe_ptr", None)
        if candidates_np is None:
            try:
                csv_path = os.path.join("data", "universo_reducido.csv")
                candidates_np = pd.read_csv(csv_path).values[:, :6].astype(np.uint8)
            except:
                return PredictionResultDTO("Error Data", [])

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

        final_selection, selected_ranks, seen_sets = [], [], []

        # --- TIER 1: ALFA-FOCUS (5 Tkts - Stride 50) ---
        # Objetivo: Capturar el Rank #1 al #250 con alta resolución
        for i in range(5):
            idx_in_rank = i * 50
            if idx_in_rank < len(sorted_indices):
                idx = sorted_indices[idx_in_rank]
                tup = tuple(sorted(candidates_np[idx]))
                final_selection.append(list(tup))
                selected_ranks.append(idx_in_rank + 1)
                seen_sets.append(set(tup))

        # --- TIER 2: ALFA-SWEEP (5 Tkts - Stride 1000) ---
        # Objetivo: Barrer la zona de hombros (Rank 1000 a 5000)
        for i in range(5):
            idx_in_rank = 1000 + (i * 1000)
            if idx_in_rank < len(sorted_indices):
                idx = sorted_indices[idx_in_rank]
                tup = tuple(sorted(candidates_np[idx]))
                if set(tup) not in seen_sets:
                    final_selection.append(list(tup))
                    selected_ranks.append(idx_in_rank + 1)
                    seen_sets.append(set(tup))

        # --- TIER 3: BETA-DIVERSITY (Resto hasta 20 - Overlap 2) ---
        for idx_rank, idx in enumerate(sorted_indices):
            if len(final_selection) >= TOTAL_TICKETS:
                break
            current_set = set(candidates_np[idx])
            if any(len(current_set & s) > 2 for s in seen_sets):
                continue
            final_selection.append(list(sorted(candidates_np[idx])))
            selected_ranks.append(idx_rank + 1)
            seen_sets.append(current_set)

        self._forensic_snapshot = {
            "universe": candidates_np,
            "ai_scores": raw_ai_scores,
            "geo_scores": final_geo_scores,
            "selected_ranks": selected_ranks,
            "thresholds": {"ai_limit": AI_THRESHOLD, "geo_floor": floor_val},
        }

        return PredictionResultDTO("Sniper V9.7", final_selection)

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
            "geo_score": (
                float(snap.get("geo_scores", np.zeros(1))[idx_audit])
                if "geo_scores" in snap
                else 0
            ),
            "percentile": float((1 - (winner_rank / len(snap["universe"]))) * 100),
        }
