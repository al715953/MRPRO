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
    SELECTOR V35.1: Budget Controlled Edition (Fixed).
    Alinea la potencia de la IA V5.0 con restricciones de num_tickets.
    """

    def __init__(self):
        self.ai_model = LotteryAIModel()
        self._matrix_cache = {"cluster_matrix": None}

    def predict(self, history, config) -> PredictionResultDTO:
        univ = config.raw_universe_ptr
        if univ is None or len(univ) == 0:
            return PredictionResultDTO("Empty", [])

        # 1. ENTRENAMIENTO Y SCORING
        if not self.ai_model.is_trained:
            self.ai_model.train(history.winning_numbers, config.total_balls)

        xp = cp if (HAS_CUPY and hasattr(univ, "get")) else np
        u_xp = xp.asarray(univ)

        ai_scores, geo_scores = self._compute_hybrid_scores(u_xp, history, config, xp)

        # 2. NORMALIZACIÓN DE ENERGÍA
        ai_min, ai_max = ai_scores.min(), ai_scores.max()
        ai_norm = (ai_scores - ai_min) / (ai_max - ai_min + 1e-10)

        geo_min, geo_max = geo_scores.min(), geo_scores.max()
        geo_norm = (geo_scores - geo_min) / (geo_max - geo_min + 1e-10)

        ai_w = 0.5 * (1 / (1 + xp.exp(-4 * (ai_norm - 0.5)))) + 0.5 * ai_norm
        hybrid_scores = (ai_w * ai_norm) + ((1.0 - ai_w) * geo_norm)

        # 3. MALLA DE SELECCIÓN CONTROLADA (CORRECCIÓN DE VARIABLE)
        # Cambiado config.n_tickets -> config.num_tickets
        selected_indices = self._apply_mutant_mesh(
            u_xp, hybrid_scores, ai_norm, xp, config.num_tickets
        )

        # 4. EXTRACCIÓN DE TICKETS
        tickets = [
            u_xp[i].get().tolist() if hasattr(u_xp[i], "get") else u_xp[i].tolist()
            for i in selected_indices
        ]

        snapshot = {
            "universe": univ,
            "ai_scores": ai_scores,
            "geo_scores": geo_scores,
            "hybrid_scores": hybrid_scores,
            "selected_ranks": list(range(1, len(selected_indices) + 1)),
        }

        return PredictionResultDTO("Mutant Hunter V35.1", tickets), snapshot

    def _compute_hybrid_scores(self, u_xp, history, config, xp):
        ai_scores = xp.asarray(self.ai_model.score_tickets(u_xp))

        if self._matrix_cache["cluster_matrix"] is None:
            m = np.zeros(
                (config.total_balls + 1, config.total_balls + 1), dtype=np.uint16
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

        return ai_scores, geo_scores

    def _apply_mutant_mesh(self, u_xp, hybrid_scores, ai_norm, xp, n_target):
        """
        MALLA V35.1: Distribución Proporcional de Presupuesto.
        Garantiza cumplimiento estricto del n_target (num_tickets).
        """
        sorted_indices = xp.argsort(hybrid_scores)[::-1]
        selected_indices = []

        # Reparto de cuotas dinámico
        q1 = max(1, int(n_target * 0.5))
        q2 = max(1, int(n_target * 0.3))
        q3 = max(1, int(n_target * 0.2))

        zones = [(0, 50, q1), (51, 500, q2), (501, 2000, q3)]

        for start, end, quota in zones:
            zone_count = 0
            for idx in sorted_indices[start:end]:
                if len(selected_indices) >= n_target:
                    break

                idx_int = int(idx)
                ticket_val = (
                    u_xp[idx_int].get()
                    if hasattr(u_xp[idx_int], "get")
                    else u_xp[idx_int]
                )
                ticket_set = set(ticket_val.tolist())

                if float(ai_norm[idx_int]) > 0.85:
                    if idx_int not in selected_indices:
                        selected_indices.append(idx_int)
                        zone_count += 1
                        if zone_count >= quota:
                            break
                    continue

                is_redundant = False
                for s_idx in selected_indices:
                    sel_val = (
                        u_xp[s_idx].get()
                        if hasattr(u_xp[s_idx], "get")
                        else u_xp[s_idx]
                    )
                    sel_ticket = set(sel_val.tolist())

                    if len(ticket_set & sel_ticket) >= 5:
                        is_redundant = True
                        break

                if not is_redundant:
                    selected_indices.append(idx_int)
                    zone_count += 1

                if zone_count >= quota:
                    break

        return selected_indices[:n_target]
