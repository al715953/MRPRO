import numpy as np
import itertools
from src.core.ai_scorer import LotteryAIModel
from src.domain.dtos import PredictionResultDTO

try:
    import cupy as cp

    HAS_CUPY = True
except ImportError:
    HAS_CUPY = False


class GeneticSelectorStrategy:
    """
    SELECTOR V34.0: Mesh Sniper Edition.
    Implementa Redundancia N-4, Gradient Mesh (Sectores) y Fusión de Confianza Suave.
    Optimizado para rescatar el Jackpot #1535 y maximizar cobertura Top 20.
    """

    def __init__(self):
        self.ai_model = LotteryAIModel()
        self._matrix_cache = {"cluster_matrix": None}
        self._forensic_snapshot = {}

    def predict(self, history, config) -> PredictionResultDTO:
        univ = config.raw_universe_ptr
        if univ is None or len(univ) == 0:
            return PredictionResultDTO("Empty", [])

        if not self.ai_model.is_trained:
            self.ai_model.train(history.winning_numbers, config.total_balls)

        xp = cp if (HAS_CUPY and hasattr(univ, "get")) else np
        u_xp = xp.asarray(univ)

        # 1. GENERACIÓN DE SCORES BASE
        ai_scores_raw = self.ai_model.score_tickets(univ)
        ai_scores = xp.asarray(ai_scores_raw)
        ai_norm = (ai_scores - ai_scores.min()) / (
            ai_scores.max() - ai_scores.min() + 1e-10
        )

        # 2. GEO RESONANCE
        if self._matrix_cache["cluster_matrix"] is None:
            m = np.zeros(
                (config.total_balls + 2, config.total_balls + 2), dtype=np.uint16
            )
            for d in history.winning_numbers:
                for a, b in itertools.combinations(sorted(d[:6]), 2):
                    m[a, b] += 1
                    m[b, a] += 1
            self._matrix_cache["cluster_matrix"] = m

        m_xp = xp.asarray(self._matrix_cache["cluster_matrix"])
        geo_scores = xp.zeros(len(u_xp))
        for i, j in itertools.combinations(range(6), 2):
            geo_scores += m_xp[u_xp[:, i], u_xp[:, j]]
        geo_norm = (geo_scores - geo_scores.min()) / (
            geo_scores.max() - geo_scores.min() + 1e-10
        )

        # 3. FUSIÓN GRADIENTE V34.0
        # Evitamos matar al Jackpot con una sigmoide tan agresiva (k=5)
        ai_w = 1 / (1 + xp.exp(-5 * (ai_norm - 0.5)))
        hybrid_scores = (ai_w * ai_norm) + ((1.0 - ai_w) * geo_norm)

        # 4. SELECCIÓN CON FILTRO DE MALLA (Redundancia N-4)
        sorted_indices = xp.argsort(hybrid_scores)[::-1]
        selected_indices = []

        # Escaneamos el Top 1000 para encontrar diversidad real
        for idx in sorted_indices[:1000]:
            idx_int = int(idx)
            ticket = (
                u_xp[idx_int].get().tolist()
                if hasattr(u_xp[idx_int], "get")
                else u_xp[idx_int].tolist()
            )

            # Tu idea "Fuera de la Caja": Descarte desde 4 números iguales
            is_redundant = False
            for s_idx in selected_indices:
                selected_ticket = (
                    u_xp[s_idx].get().tolist()
                    if hasattr(u_xp[s_idx], "get")
                    else u_xp[s_idx].tolist()
                )
                overlap = len(set(ticket) & set(selected_ticket))
                if overlap >= 4:
                    is_redundant = True
                    break

            if not is_redundant:
                selected_indices.append(idx_int)

            if len(selected_indices) >= config.num_tickets:
                break

        # SNAPSHOT TELEMÉTRICO (Fix: Geo visibility)
        self._forensic_snapshot = {
            "universe": univ,
            "ai_scores": ai_scores.get() if hasattr(ai_scores, "get") else ai_scores,
            "geo_scores": (
                geo_scores.get() if hasattr(geo_scores, "get") else geo_scores
            ),
            "hybrid_scores": (
                hybrid_scores.get() if hasattr(hybrid_scores, "get") else hybrid_scores
            ),
            "selected_ranks": list(range(1, len(selected_indices) + 1)),
            "univ_size": len(univ),
        }

        tickets = [
            u_xp[i].get().tolist() if hasattr(u_xp[i], "get") else u_xp[i].tolist()
            for i in selected_indices
        ]
        return PredictionResultDTO("Mesh Sniper V34.0", tickets)

    def audit_winner(self, history, config, winning_ticket) -> dict:
        """Auditoría de precisión con fix de mapeo Geo."""
        snap = self._forensic_snapshot
        if "universe" not in snap or len(snap["universe"]) == 0:
            return {"hits": 0, "rank": 0, "proximity": 999, "univ_size": 0}

        xp = cp if (HAS_CUPY and hasattr(snap["universe"], "get")) else np
        target = xp.asarray(sorted(winning_ticket[:6]), dtype=xp.uint8)
        hits_vec = xp.sum(xp.isin(snap["universe"], target), axis=1)
        max_h = int(xp.max(hits_vec))

        if max_h == 0:
            return {
                "hits": 0,
                "rank": 0,
                "proximity": 999,
                "univ_size": snap["univ_size"],
            }

        best_indices = xp.where(hits_vec == max_h)[0]
        scores_for_rank = snap["hybrid_scores"]

        # Encontramos el índice absoluto con mayor score híbrido
        best_idx_cpu = (
            best_indices.get() if hasattr(best_indices, "get") else best_indices
        )
        idx_f_abs = int(best_idx_cpu[np.argsort(scores_for_rank[best_idx_cpu])[-1]])

        rank = int(np.sum(scores_for_rank > scores_for_rank[idx_f_abs]) + 1)
        proximity = int(min([abs(rank - r) for r in snap["selected_ranks"]]))

        return {
            "hits": max_h,
            "rank": rank,
            "proximity": proximity,
            "ai_score": float(snap["ai_scores"][idx_f_abs]),
            "geo_score": float(snap["geo_scores"][idx_f_abs]),
            "hybrid_score": float(scores_for_rank[idx_f_abs]),
            "univ_size": snap["univ_size"],
        }
