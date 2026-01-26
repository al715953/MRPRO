# src/strategies/genetic/resonance.py

import numpy as np
import itertools
from src.core.ai_scorer import LotteryAIModel


class ResonanceEngine:
    """
    Motor encargado del cálculo de puntajes (Scoring).
    Maneja los modelos Alpha/Omega, la matriz geométrica y la fusión no-lineal.
    """

    def __init__(self):
        self.ai_model = LotteryAIModel()
        self._matrix_cache = {"cluster_matrix": None}

    def calculate_resonance(self, u_xp, history, config, xp):
        """Ejecuta el flujo completo de scoring y fusión V5.9.8."""
        if not self.ai_model.is_trained:
            self.ai_model.train(history.winning_numbers, config.total_balls)

        # --- FASE 1: SCORES BASE Y GEOMETRÍA ---
        ai_scores, geo_scores = self._compute_base_scores(u_xp, history, config, xp)
        ai_norm = (ai_scores - ai_scores.min()) / (
            ai_scores.max() - ai_scores.min() + 1e-10
        )

        # --- FASE 2: DEFINICIÓN DEL RADAR (Top 20,000) ---
        radar_indices = xp.argsort(ai_norm)[-20000:][::-1]
        u_reduced = u_xp[radar_indices]

        # --- FASE 3: OMEGA POWER SQUEEZE (Fusión V5.9.8) ---
        omega_scores, breakdown = self.ai_model.score_tickets(
            u_reduced, return_breakdown=True
        )
        omega_signal = xp.asarray(breakdown.get("omega_hunter", omega_scores))
        omega_norm = (omega_signal - omega_signal.min()) / (
            omega_signal.max() - omega_signal.min() + 1e-10
        )

        # Boost No-Lineal (Power 2.0)
        omega_boosted = xp.power(omega_norm, 2.0)

        # Fusión 60/40
        final_scores_reduced = (ai_norm[radar_indices] * 0.6) + (omega_boosted * 0.4)

        return {
            "u_reduced": u_reduced,
            "final_scores_reduced": final_scores_reduced,
            "radar_indices": radar_indices,
            "ai_norm": ai_norm,
            "geo_scores": geo_scores,
            "geo_matrix_xp": xp.asarray(self._matrix_cache["cluster_matrix"]),
        }

    def _compute_base_scores(self, u_xp, history, config, xp):
        """Helper para cálculos vectorizados iniciales."""
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
