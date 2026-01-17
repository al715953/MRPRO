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

try:
    import cupy as cp

    HAS_CUPY = True
except ImportError:
    HAS_CUPY = False

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
    SELECTOR V9.9.1: Neural Mesh Expansion (Telemetry Fixed).
    - Malla de saturación K-Medoids en GPU RTX 4070 Ti.
    - Telemetría completa para Dashboard (AI, Geo, Percentile).
    """

    def __init__(self):
        self.ai_model = LotteryAIModel()
        self._last_trained_date = None
        self._matrix_cache = {"cluster_matrix": None, "hotness_vector": None}
        self._forensic_snapshot = {}

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

    def _gpu_kmedoids_selection(
        self, top_candidates_np: np.ndarray, num_clusters: int = 20
    ) -> List[int]:
        if not HAS_CUPY:
            return list(range(num_clusters))

        X = cp.asarray(top_candidates_np, dtype=cp.float32)
        n_samples = X.shape[0]
        initial_idx = cp.random.choice(n_samples, num_clusters, replace=False)
        medoids = X[initial_idx]

        for _ in range(10):
            distances = cp.sum(cp.abs(X[:, cp.newaxis, :] - medoids), axis=2)
            labels = cp.argmin(distances, axis=1)
            new_medoids_idx = cp.zeros(num_clusters, dtype=cp.int32)
            for k in range(num_clusters):
                mask = labels == k
                if cp.any(mask):
                    cluster_points = X[mask]
                    dist_in_cluster = cp.sum(
                        cp.abs(cluster_points[:, cp.newaxis, :] - cluster_points),
                        axis=(1, 2),
                    )
                    rel_idx = cp.argmin(dist_in_cluster)
                    new_medoids_idx[k] = cp.where(mask)[0][rel_idx]
                else:
                    new_medoids_idx[k] = cp.random.randint(0, n_samples)

            if cp.all(initial_idx == new_medoids_idx):
                break
            initial_idx = new_medoids_idx
            medoids = X[initial_idx]

        return initial_idx.get().tolist()

    def predict(
        self, history: DrawHistoryDTO, config: PredictionConfigDTO
    ) -> PredictionResultDTO:
        num_target = config.num_tickets if config.num_tickets > 0 else 20
        candidates_np = getattr(config, "raw_universe_ptr", None)
        if candidates_np is None:
            return PredictionResultDTO("Error: Universe Missing", [])

        last_date = history.dates[-1] if history.dates else "None"
        if self._last_trained_date != last_date:
            self.ai_model.train(history.winning_numbers, config.total_balls)
            self._update_heuristic_matrices(history, config.total_balls)
            self._last_trained_date = last_date

        # Scoring Dual (AI + Geométrico)
        ticket_tuples = [tuple(x) for x in candidates_np]
        raw_ai_scores = self.ai_model.score_tickets(ticket_tuples)
        final_geo_scores = self._calculate_scores(candidates_np, config.total_balls)

        # Selección Geométrica en el Top 1000
        TOP_N_POOL = 1000
        sorted_indices = np.argsort(raw_ai_scores)[::-1]
        pool_indices = sorted_indices[:TOP_N_POOL]
        pool_data = candidates_np[pool_indices]

        centroid_rel_indices = self._gpu_kmedoids_selection(pool_data, num_target)
        final_indices = pool_indices[centroid_rel_indices]

        final_selection = [list(candidates_np[idx]) for idx in final_indices]
        selected_ranks = [
            int(np.where(sorted_indices == idx)[0][0] + 1) for idx in final_indices
        ]

        self._forensic_snapshot = {
            "universe": candidates_np,
            "ai_scores": raw_ai_scores,
            "geo_scores": final_geo_scores,
            "selected_ranks": sorted(selected_ranks),
            "univ_size": len(candidates_np),
        }

        return PredictionResultDTO(f"Neural Mesh V9.9.1 (GPU)", final_selection)

    def audit_winner(self, history, config, winning_ticket) -> dict:
        snap = self._forensic_snapshot
        target = np.array(sorted(winning_ticket[:6]))
        hits = np.sum(np.isin(snap["universe"], target), axis=1)
        max_hits = int(np.max(hits))
        best_idx_group = np.where(hits == max_hits)[0]

        # Encontrar el mejor índice basado en AI score dentro del grupo de aciertos
        idx_audit = best_idx_group[np.argsort(snap["ai_scores"][best_idx_group])[-1]]
        winner_rank = np.sum(snap["ai_scores"] > snap["ai_scores"][idx_audit]) + 1
        min_dist = min([abs(winner_rank - r) for r in snap["selected_ranks"]])

        return {
            "found": max_hits >= 4,
            "hits": max_hits,
            "rank": int(winner_rank),
            "proximity": int(min_dist),
            "ai_score": float(snap["ai_scores"][idx_audit]),
            "geo_score": float(snap["geo_scores"][idx_audit]),
            "percentile": float((1 - (winner_rank / snap["univ_size"])) * 100),
            "univ_size": snap["univ_size"],
            "selected_ranks": snap["selected_ranks"],
        }
