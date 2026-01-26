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
    """SELECTOR V11.0: Independiente y Blindado."""

    def __init__(self):
        self.ai_model = LotteryAIModel()
        self._matrix_cache = {"cluster_matrix": None}
        self._forensic_snapshot = {}

    def predict(self, history, config) -> PredictionResultDTO:
        """La estrategia ahora gestiona su propio entrenamiento y heurística."""
        univ = config.raw_universe_ptr
        if univ is None or len(univ) == 0:
            return PredictionResultDTO("Empty", [])

        # MEJORA: El selector decide cuándo entrenar, liberando al Backtester
        if not self.ai_model.is_trained:
            self.ai_model.train(history.winning_numbers, config.total_balls)

        # MEJORA: La heurística Geo se inicializa siempre que sea necesario
        if self._matrix_cache["cluster_matrix"] is None:
            m = np.zeros(
                (config.total_balls + 2, config.total_balls + 2), dtype=np.uint16
            )
            for d in history.winning_numbers:
                for a, b in itertools.combinations(sorted(d[:6]), 2):
                    m[a, b] += 1
                    m[b, a] += 1
            self._matrix_cache["cluster_matrix"] = m

        ai_scores = self.ai_model.score_tickets(univ)

        # Geo-Scores con protección contra división por cero
        u_cpu = univ.get() if hasattr(univ, "get") else univ
        geo_scores = np.zeros(len(u_cpu))
        m = self._matrix_cache["cluster_matrix"]
        for i, j in itertools.combinations(range(6), 2):
            geo_scores += m[u_cpu[:, i], u_cpu[:, j]]
        geo_scores = (
            geo_scores / np.max(geo_scores) if np.max(geo_scores) > 0 else geo_scores
        )

        # Selección Top Rank (IA)
        idx = np.argsort(ai_scores)[::-1][: config.num_tickets].tolist()

        self._forensic_snapshot = {
            "universe": univ,
            "ai_scores": ai_scores,
            "geo_scores": geo_scores,
            "selected_ranks": list(range(1, config.num_tickets + 1)),
            "univ_size": len(univ),
        }

        tickets = [
            univ[i].get().tolist() if hasattr(univ[i], "get") else univ[i].tolist()
            for i in idx
        ]
        return PredictionResultDTO("Genetic V11.0", tickets)

    def audit_winner(self, history, config, winning_ticket) -> dict:
        """Función de auditoría blindada contra IndexError y fallos de Pipe."""
        snap = self._forensic_snapshot
        if (
            "universe" not in snap
            or len(snap["universe"]) == 0
            or len(snap.get("ai_scores", [])) == 0
        ):
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
        best_idx_cpu = (
            best_indices.get() if hasattr(best_indices, "get") else best_indices
        )

        # Selección segura del mejor índice absoluto
        idx_f_abs = int(best_idx_cpu[np.argsort(snap["ai_scores"][best_idx_cpu])[-1]])

        rank = int(np.sum(snap["ai_scores"] > snap["ai_scores"][idx_f_abs]) + 1)
        proximity = int(min([abs(rank - r) for r in snap["selected_ranks"]]))

        # Métricas defensivas para telemetría
        ai_s = float(snap["ai_scores"][idx_f_abs])
        geo_s = (
            float(snap["geo_scores"][idx_f_abs])
            if len(snap["geo_scores"]) > idx_f_abs
            else 0.0
        )

        return {
            "hits": max_h,
            "rank": rank,
            "proximity": proximity,
            "ai_score": ai_s,
            "geo_score": geo_s,
            "univ_size": snap["univ_size"],
        }
