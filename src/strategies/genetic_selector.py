import pandas as pd
import numpy as np
import os
import itertools
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
    SELECTOR V31.2: Estrategia de Caza de Alta Densidad.
    Implementa Selección de Élite Ampliada y Muestreo Estocástico Power 10.
    """

    def __init__(self):
        self.ai_model = LotteryAIModel()
        self._last_trained_date = None
        self._matrix_cache = {
            "cluster_matrix": None,
            "hotness_vector": None,
            "max_cluster": 1.0,
            "max_hotness": 1.0,
        }
        self._forensic_snapshot = {
            "universe": None,
            "ai_scores": None,
            "geo_scores": None,
            "thresholds": {},
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
        flat_matrix = matrix.flatten()
        max_cluster_val = (
            np.percentile(flat_matrix[flat_matrix > 0], 99) if np.sum(matrix) > 0 else 1
        )
        freq_vec = np.zeros(total_balls + 2, dtype=np.uint16)
        for draw in history.winning_numbers[-12:]:
            for num in draw[:6]:
                freq_vec[num] += 1
        self._matrix_cache.update(
            {
                "cluster_matrix": matrix,
                "hotness_vector": freq_vec,
                "max_cluster": float(max_cluster_val),
                "max_hotness": float(np.max(freq_vec)) if np.max(freq_vec) > 0 else 1,
            }
        )

    def _calculate_scores(self, candidates_np, total_balls):
        raw_c, raw_h = calc_heuristics_vectorized(
            candidates_np,
            self._matrix_cache["cluster_matrix"],
            self._matrix_cache["hotness_vector"],
            total_balls,
        )
        norm_c = np.clip(raw_c / (15 * self._matrix_cache["max_cluster"]), 0, 1.0)
        norm_h = np.clip(raw_h / (6 * self._matrix_cache["max_hotness"]), 0, 1.0)
        return (norm_c * 0.70) + (norm_h * 0.30)

    def predict(
        self, history: DrawHistoryDTO, config: PredictionConfigDTO
    ) -> PredictionResultDTO:
        AI_THRESHOLD, GEO_P_FLOOR = 0.84, 35.0

        # Cargamos el universo reducido generado por la Fase 1
        csv_path = os.path.join("data", "universo_reducido.csv")
        try:
            df = pd.read_csv(csv_path)
            candidates_np = df.iloc[:, :6].values.astype(np.uint8)
        except:
            return PredictionResultDTO("Error CSV", [])

        self._train_model(history, config.total_balls)

        # Scoring de IA y Geometría
        raw_ai_scores = np.array(
            self.ai_model.score_tickets([tuple(x) for x in candidates_np]),
            dtype=np.float32,
        )
        final_geo_scores = self._calculate_scores(candidates_np, config.total_balls)

        global_lower = np.percentile(final_geo_scores, GEO_P_FLOOR)
        mid_point = (global_lower + np.max(final_geo_scores)) / 2

        # Guardamos Snapshot para Auditoría de Rank
        self._forensic_snapshot = {
            "universe": candidates_np,
            "ai_scores": raw_ai_scores,
            "geo_scores": final_geo_scores,
            "thresholds": {
                "ai_limit": AI_THRESHOLD,
                "geo_low": global_lower,
                "geo_mid": mid_point,
            },
        }

        final_selection, seen_tickets = [], []
        indices = np.arange(len(final_geo_scores))
        strat_counts = {"stars": 0, "stochastic_high": 0, "stochastic_low": 0}

        def add_ticket(idx, category):
            tup = tuple(sorted(candidates_np[idx]))
            t_set = set(tup)
            # Filtro de diversidad: Evitamos tickets demasiado similares
            if any(len(t_set.intersection(s)) >= 5 for s in seen_tickets):
                return False
            final_selection.append(list(tup))
            seen_tickets.append(t_set)
            strat_counts[category] += 1
            return True

        # --- 1. SELECCIÓN DE ÉLITE (AMPLIADA A 10) ---
        mask_stars = raw_ai_scores >= AI_THRESHOLD
        idx_stars = indices[mask_stars]
        if len(idx_stars) > 0:
            # Ordenamos por los mejores AI Scores
            sorted_stars = idx_stars[np.argsort(raw_ai_scores[mask_stars])[::-1]]
            for idx in sorted_stars[:10]:  # Aseguramos el Top 10 absoluto
                add_ticket(idx, "stars")

        # --- 2. MUESTREO ESTOCÁSTICO (POWER 10) ---
        def weighted_sample(mask, quota, category):
            available_idx = indices[mask]
            if len(available_idx) == 0:
                return

            # ELEVACIÓN DE PESO POWER 10: Concentración masiva en Ranks altos
            weights = raw_ai_scores[mask] ** 12
            prob = weights / weights.sum()

            # Tomamos un pool de muestreo para filtrar por diversidad
            sample_size = min(len(available_idx), quota * 5)
            chosen = np.random.choice(
                available_idx, size=sample_size, replace=False, p=prob
            )

            for idx in chosen:
                if strat_counts[category] >= quota:
                    break
                add_ticket(idx, category)

        remaining = config.num_tickets - len(final_selection)
        if remaining > 0:
            quota_high = int(remaining * 0.60)
            quota_low = remaining - quota_high

            # Banda Alta (Muestreo ponderado sobre el punto medio geométrico)
            weighted_sample(
                (final_geo_scores >= mid_point), quota_high, "stochastic_high"
            )

            # Banda Baja (Muestreo ponderado sobre el piso geométrico)
            weighted_sample(
                (final_geo_scores >= global_lower) & (final_geo_scores < mid_point),
                quota_low,
                "stochastic_low",
            )

        result = PredictionResultDTO("Sniper V31.2", final_selection)
        result.metadata = {
            "total_candidates": int(len(candidates_np)),
            "floor_p35": float(global_lower),
            "ai_threshold": float(AI_THRESHOLD),
            "stratification": strat_counts,
        }
        return result

    def audit_winner(self, history, config, winning_ticket) -> dict:
        """Auditoría forense para localizar al ganador dentro del ranking de la IA."""
        snap = self._forensic_snapshot
        if snap["universe"] is None:
            return {"found": False}

        target = np.array(sorted(winning_ticket[:6]))
        hits = np.sum(np.isin(snap["universe"], target), axis=1)
        max_hits = int(np.max(hits))

        # Localizamos el mejor ticket posible del universo
        best_indices = np.where(hits == max_hits)[0]
        if len(best_indices) == 0:
            return {"found": False, "hits": 0}

        idx_audit = best_indices[np.argsort(snap["ai_scores"][best_indices])[-1]]
        ai_val = snap["ai_scores"][idx_audit]
        rank = np.sum(snap["ai_scores"] > ai_val) + 1

        return {
            "found": max_hits >= 4,
            "is_jackpot": max_hits == 6,
            "hits": max_hits,
            "ai_score": float(ai_val),
            "geo_score": float(snap["geo_scores"][idx_audit]),
            "rank": int(rank),
            "percentile": float((1 - (rank / len(snap["universe"]))) * 100),
            "thresholds": {k: float(v) for k, v in snap["thresholds"].items()},
        }
