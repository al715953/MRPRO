import numpy as np
import itertools
from src.core.ai_scorer import LotteryAIModel


class ResonanceEngine:
    """
    Motor V7.16: Boundary-Flex & Elite-Liberation.
    Implementa relajación dinámica de restricciones físicas para candidatos
    de alta confianza (AI Score > 0.94) para permitir cierres no convencionales.
    """

    def __init__(self):
        self.ai_model = LotteryAIModel()
        self._matrix_cache = {"cluster_matrix": None}

    def calculate_resonance(self, u_xp, history, config, xp):
        raw_history = history.winning_numbers
        if not self.ai_model.is_trained:
            self.ai_model.train(raw_history, config.total_balls)

        if u_xp is None or len(u_xp) == 0:
            return None

        # COLD-ANALYSIS V7.16
        recent_draws = raw_history[-5:]
        flat_recent = set([num for draw in recent_draws for num in draw[:6]])
        thermal_numbers = sorted(
            list(set(range(1, config.total_balls + 1)) - flat_recent)
        )

        ai_scores, geo_scores_raw = self._compute_base_scores(u_xp, history, config, xp)

        if len(ai_scores) == 0:
            return None

        ai_norm = (ai_scores - ai_scores.min()) / (
            ai_scores.max() - ai_scores.min() + 1e-10
        )
        geo_norm = (geo_scores_raw - geo_scores_raw.min()) / (
            geo_scores_raw.max() - geo_scores_raw.min() + 1e-10
        )

        radar_limit = int(len(u_xp) * 0.45)
        radar_indices = xp.argsort((ai_norm * 0.85) + (geo_norm * 0.15))[::-1][
            :radar_limit
        ]
        u_reduced = u_xp[radar_indices]
        ai_subset = ai_norm[radar_indices]

        # --- FASE V7.16: ELITE-LIBERATION BOOST ---
        sweet_spot = xp.ones_like(ai_subset)
        # Incrementamos el boost para el rango de élite absoluto
        sweet_spot[ai_subset > 0.94] = 2.8  # Antes penalizado, ahora potenciado
        sweet_spot[(ai_subset >= 0.50) & (ai_subset <= 0.93)] = 2.5

        # --- FASE V7.16: BOUNDARY-FLEX ---
        f1, f6 = u_reduced[:, 0], u_reduced[:, 5]
        boundary_penalty = xp.ones(len(u_reduced))

        # Regla estándar: penaliza f1 > 18 o f6 < 25
        standard_penalty_mask = (f1 > 18) | (f6 < 25)

        # LIBERACIÓN: Si el AI Score es > 0.94, el penalty se reduce de 0.01 a 0.85 (Flexión)
        # Esto permite que combinaciones "raras" pero probables sobrevivan.
        boundary_penalty[standard_penalty_mask] = 0.01  # Penalty base
        boundary_penalty[standard_penalty_mask & (ai_subset > 0.94)] = 0.85  # FLEX

        k, x0 = 12.0, 0.70
        disruption_signal = ai_subset * sweet_spot * boundary_penalty
        sigmoidal_boost = 1.0 / (1.0 + xp.exp(-k * (disruption_signal - x0)))

        final_scores_reduced = sigmoidal_boost * 10.0 + (geo_norm[radar_indices] * 0.02)

        return {
            "u_reduced": u_reduced,
            "final_scores_reduced": final_scores_reduced,
            "radar_indices": radar_indices,
            "ai_norm": ai_norm,
            "geo_scores": geo_norm,
            "geo_matrix_xp": xp.asarray(self._matrix_cache["cluster_matrix"]),
            "boundary_applied": boundary_penalty,
            "thermal_numbers": thermal_numbers,
        }

    def _compute_base_scores(self, u_xp, history, config, xp):
        ai_scores = xp.asarray(self.ai_model.score_tickets(u_xp))
        if self._matrix_cache["cluster_matrix"] is None:
            m = np.zeros(
                (config.total_balls + 1, config.total_balls + 1), dtype=np.uint16
            )
            for draw in history.winning_numbers:
                for a, b in itertools.combinations(sorted(draw[:6]), 2):
                    m[a, b] += 1
                    m[b, a] += 1
            self._matrix_cache["cluster_matrix"] = m
        m_xp = xp.asarray(self._matrix_cache["cluster_matrix"])
        t_xp = u_xp.astype(xp.int32)
        geo_scores = xp.zeros(len(u_xp), dtype=xp.float32)
        for i in range(6):
            for j in range(i + 1, 6):
                geo_scores += m_xp[t_xp[:, i], t_xp[:, j]]
        return ai_scores, geo_scores
