import numpy as np
import os
import itertools
from typing import List, Tuple

try:
    import cupy as cp

    HAS_CUPY = True
except ImportError:
    HAS_CUPY = False

from src.domain.interfaces import ILotteryStrategy
from src.domain.dtos import DrawHistoryDTO, PredictionConfigDTO, PredictionResultDTO
from src.core.ai_scorer import LotteryAIModel


class GeneticSelectorStrategy(ILotteryStrategy):
    """
    SELECTOR V10.3.1: Quantum Dynamic Mesh (Null-Safety Edition).
    - Alpha Diversificado: Evita redundancia en el Top 3.
    - Mesh Adaptativo: Resolución basada en la confianza del XGBoost.
    - Blindaje de Auditoría: Previene errores 'U: 0'.
    """

    def __init__(self):
        self.ai_model = LotteryAIModel()
        self._last_trained_date = None
        self._matrix_cache = {"cluster_matrix": None}
        self._forensic_snapshot = {}

    def _update_heuristics(self, history: DrawHistoryDTO, total_balls: int):
        """Cálculo de matriz de co-ocurrencia para Telemetría Geo."""
        matrix = np.zeros((total_balls + 2, total_balls + 2), dtype=np.uint16)
        for draw in history.winning_numbers:
            for a, b in itertools.combinations(sorted(draw[:6]), 2):
                matrix[a, b] += 1
                matrix[b, a] += 1
        self._matrix_cache["cluster_matrix"] = matrix

    def _calculate_geo_score(self, candidates_np):
        """Genera el score geométrico para auditoría visual."""
        matrix = self._matrix_cache["cluster_matrix"]
        if matrix is None:
            return np.zeros(len(candidates_np))
        scores = np.zeros(len(candidates_np), dtype=np.float32)
        for i in range(len(candidates_np)):
            row = candidates_np[i]
            s = 0
            for j in range(6):
                for k in range(j + 1, 6):
                    s += matrix[row[j], row[k]]
            scores[i] = s
        return scores / (np.max(scores) if np.max(scores) > 0 else 1)

    def _gpu_dynamic_kmedoids(self, top_data, scores, n_clusters, expansion_factor):
        """Clustering adaptativo optimizado para RTX 4070 Ti."""
        if not HAS_CUPY:
            return list(range(n_clusters))

        X = cp.asarray(top_data, dtype=cp.float32)
        S = cp.asarray(scores, dtype=cp.float32)
        n_samples = X.shape[0]

        # Potencia de probabilidad según factor de expansión
        p_power = 2.0 if expansion_factor < 1.4 else 1.0
        probs = cp.asnumpy(S**p_power)
        probs /= probs.sum()

        initial_idx = cp.asarray(
            np.random.choice(n_samples, n_clusters, replace=False, p=probs)
        )
        medoids = X[initial_idx]

        for _ in range(15):
            distances = cp.sum(cp.abs(X[:, cp.newaxis, :] - medoids), axis=2)
            # Factor de Atracción Gravitacional Dinámico
            weighted_dist = distances / (S[:, cp.newaxis] ** expansion_factor + 1e-6)
            labels = cp.argmin(weighted_dist, axis=1)

            new_idx = cp.zeros(n_clusters, dtype=cp.int32)
            for k in range(n_clusters):
                mask = labels == k
                if cp.any(mask):
                    cluster_pts = X[mask]
                    d_int = cp.sum(
                        cp.abs(cluster_pts[:, cp.newaxis, :] - cluster_pts), axis=(1, 2)
                    )
                    new_idx[k] = cp.where(mask)[0][cp.argmin(d_int)]
                else:
                    new_idx[k] = cp.asarray(np.random.choice(n_samples, p=probs))

            if cp.all(initial_idx == new_idx):
                break
            initial_idx, medoids = new_idx, X[new_idx]

        return initial_idx.get().tolist()

    def predict(
        self, history: DrawHistoryDTO, config: PredictionConfigDTO
    ) -> PredictionResultDTO:
        candidates_np = getattr(config, "raw_universe_ptr", None)
        if candidates_np is None or len(candidates_np) == 0:
            return PredictionResultDTO("Error: Empty Universe", [])

        # Entrenamiento y Heurística
        if self._last_trained_date != history.dates[-1]:
            self.ai_model.train(history.winning_numbers, config.total_balls)
            self._update_heuristics(history, config.total_balls)
            self._last_trained_date = history.dates[-1]

        # Inferencia y Ranking
        raw_ai_scores = self.ai_model.score_tickets([tuple(x) for x in candidates_np])
        geo_scores = self._calculate_geo_score(candidates_np)
        sorted_idx = np.argsort(raw_ai_scores)[::-1]

        # 1. EVALUACIÓN DE CONFIANZA IA
        top_10_mean = np.mean(raw_ai_scores[sorted_idx[:10]])
        expansion_factor = np.clip(2.5 - (top_10_mean * 2), 0.8, 1.8)

        # 2. ALPHA DIVERSIFICADO (Tickets #1, #2, #3 con control de solapamiento)
        alpha_indices = [sorted_idx[0]]
        for i in range(1, 150):
            if len(alpha_indices) >= 3:
                break
            cand_idx = sorted_idx[i]
            # Solo agregamos si el ticket no comparte más de 3 números con los ya elegidos
            if all(
                len(set(candidates_np[cand_idx]) & set(candidates_np[a])) <= 3
                for a in alpha_indices
            ):
                alpha_indices.append(cand_idx)

        # 3. MALLA DINÁMICA (17 Tickets en Pool de 1000)
        pool_limit = min(1000, len(sorted_idx) - 3)
        pool_indices = sorted_idx[3 : 3 + pool_limit]

        mesh_rel_idx = self._gpu_dynamic_kmedoids(
            candidates_np[pool_indices],
            raw_ai_scores[pool_indices],
            17,
            expansion_factor,
        )

        final_indices = alpha_indices + pool_indices[mesh_rel_idx].tolist()
        selected_ranks = [
            int(np.where(sorted_idx == idx)[0][0] + 1) for idx in final_indices
        ]

        # Snapshot para Auditoría V10.3.1
        self._forensic_snapshot = {
            "universe": candidates_np,
            "ai_scores": raw_ai_scores,
            "geo_scores": geo_scores,
            "selected_ranks": sorted(selected_ranks),
            "univ_size": len(candidates_np),
        }

        return PredictionResultDTO(
            "V10.3.1 Dynamic Mesh", [list(candidates_np[idx]) for idx in final_indices]
        )

    def audit_winner(self, history, config, winning_ticket) -> dict:
        """Auditoría blindada contra universos vacíos o fallos de reducción."""
        snap = self._forensic_snapshot

        if (
            "universe" not in snap
            or snap["universe"] is None
            or len(snap["universe"]) == 0
        ):
            return {
                "found": False,
                "hits": 0,
                "rank": 0,
                "proximity": 0,
                "ai_score": 0.0,
                "geo_score": 0.0,
                "percentile": 0.0,
                "univ_size": 0,
                "selected_ranks": [],
            }

        target = np.array(sorted(winning_ticket[:6]))
        hits = np.sum(np.isin(snap["universe"], target), axis=1)
        max_h = int(np.max(hits)) if len(hits) > 0 else 0

        if max_h == 0:
            return {
                "found": False,
                "hits": 0,
                "rank": 0,
                "proximity": 0,
                "ai_score": 0.0,
                "geo_score": 0.0,
                "percentile": 0.0,
                "univ_size": snap.get("univ_size", 0),
                "selected_ranks": [],
            }

        # Búsqueda del mejor representante en el universo
        best_idx_group = np.where(hits == max_h)[0]
        idx_audit = best_idx_group[np.argsort(snap["ai_scores"][best_idx_group])[-1]]
        w_rank = np.sum(snap["ai_scores"] > snap["ai_scores"][idx_audit]) + 1

        return {
            "found": max_h >= 4,
            "hits": max_h,
            "rank": int(w_rank),
            "proximity": (
                int(min([abs(w_rank - r) for r in snap["selected_ranks"]]))
                if snap.get("selected_ranks")
                else 0
            ),
            "ai_score": float(snap["ai_scores"][idx_audit]),
            "geo_score": (
                float(snap["geo_scores"][idx_audit]) if "geo_scores" in snap else 0.0
            ),
            "percentile": float((1 - (w_rank / snap["univ_size"])) * 100),
            "univ_size": snap["univ_size"],
            "selected_ranks": snap["selected_ranks"],
        }
