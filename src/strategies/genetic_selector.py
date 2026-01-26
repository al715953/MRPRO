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
    SELECTOR V5.9.7: Triple-Threat Resonance.
    Fusión No-Lineal (Power-Boost) para capturar el 5/6 sin perder el 4/6.
    Optimizado para RTX 4070 Ti y flujos de trabajo en Mac.
    """

    def __init__(self):
        self.ai_model = LotteryAIModel()
        self._matrix_cache = {"cluster_matrix": None}

    def predict(self, history, config) -> PredictionResultDTO:
        univ = config.raw_universe_ptr
        if univ is None or len(univ) == 0:
            return PredictionResultDTO("Empty", [])

        if not self.ai_model.is_trained:
            self.ai_model.train(history.winning_numbers, config.total_balls)

        xp = cp if (HAS_CUPY and hasattr(univ, "get")) else np
        u_xp = xp.asarray(univ)

        # --- FASE 1: EXTRACCIÓN ALPHA (STABILITY ANCHOR) ---
        ai_scores, geo_scores = self._compute_hybrid_scores(u_xp, history, config, xp)
        ai_norm = (ai_scores - ai_scores.min()) / (
            ai_scores.max() - ai_scores.min() + 1e-10
        )

        # --- FASE 2: OMEGA DEEP SQUEEZE (RADAR 20,000) ---
        candidate_indices = xp.argsort(ai_norm)[-20000:][::-1]
        u_reduced = u_xp[candidate_indices]

        omega_scores, breakdown = self.ai_model.score_tickets(
            u_reduced, return_breakdown=True
        )
        omega_signal = xp.asarray(breakdown.get("omega_hunter", omega_scores))
        omega_norm = (omega_signal - omega_signal.min()) / (
            omega_signal.max() - omega_signal.min() + 1e-10
        )

        # --- FASE 3: FUSIÓN NO-LINEAL (POWER BOOST) ---
        # Elevamos Omega al cuadrado para que solo los picos de alta resonancia influyan
        omega_boosted = xp.power(omega_norm, 2.0)

        final_scores_global = xp.copy(ai_norm)
        # Aplicamos el boost solo al radar de 20,000
        # Mezcla: 60% Alpha + 40% Omega Potenciado
        final_scores_global[candidate_indices] = (ai_norm[candidate_indices] * 0.6) + (
            omega_boosted * 0.4
        )

        m_xp = xp.asarray(self._matrix_cache["cluster_matrix"])
        final_tickets = self._apply_unified_mesh_v597(
            u_xp, final_scores_global, xp, config.num_tickets, m_xp
        )

        snapshot = {
            "universe": univ,
            "ai_scores": ai_norm,  # Monitor Alpha
            "hybrid_scores": final_scores_global,  # Monitor Fusionado
            "geo_scores": geo_scores,
            "selected_ranks": list(range(1, len(final_tickets) + 1)),
        }

        return (
            PredictionResultDTO(
                "ENGINE V5.9.7: Triple-Threat Resonance", final_tickets
            ),
            snapshot,
        )

    def _apply_unified_mesh_v597(self, u_xp, scores, xp, n_target, m_xp):
        """Malla de alta resonancia con expansión dinámica en el Top 5."""
        sorted_idx = xp.argsort(scores)[::-1]
        final_tickets = []
        seen_sets = []

        for idx in sorted_idx:
            if len(final_tickets) >= n_target:
                break
            t_val = self._to_flat_list(u_xp[int(idx)])
            t_set = set(t_val)

            # Filtro N-5 Estricto
            if not any(len(t_set & s) >= 5 for s in seen_sets):
                final_tickets.append(t_val)
                seen_sets.append(t_set)

                # Inyección de Nube Omega: Ahora expandimos el Top 5
                if len(final_tickets) <= 5:
                    cloud = self._generate_omega_cloud(t_val, m_xp, xp)
                    for neighbor in cloud:
                        if len(final_tickets) < n_target:
                            n_set = set(neighbor)
                            if not any(len(n_set & s) >= 5 for s in seen_sets):
                                final_tickets.append(neighbor)
                                seen_sets.append(n_set)
        return final_tickets

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

    def _to_flat_list(self, arr):
        if hasattr(arr, "get"):
            arr = arr.get()
        return [int(x) for x in np.asarray(arr).ravel()]

    def _generate_omega_cloud(self, base_ticket, m_xp, xp):
        ticket_arr = xp.array(base_ticket)
        scores = xp.zeros(len(ticket_arr))
        for i in range(len(ticket_arr)):
            for j in range(len(ticket_arr)):
                if i != j:
                    scores[i] += m_xp[ticket_arr[i], ticket_arr[j]]
        weak_idx = int(xp.argmin(scores))
        variants = []
        base_cpu = self._to_flat_list(ticket_arr)
        # Vecindad +/- 1 en el eslabón más débil
        for shift in [-1, 1]:
            nv = base_cpu[weak_idx] + shift
            if 1 <= nv <= 39 and nv not in base_cpu:
                v = list(base_cpu)
                v[weak_idx] = nv
                variants.append(sorted(v))
        return variants
