# src/strategies/genetic/resonance.py

import numpy as np
import itertools
from src.core.ai_scorer import LotteryAIModel


class ResonanceEngine:
    """
    Motor V6.3: Resonance Recovery.
    Utiliza un Radar Híbrido (IA + Geometría) para asegurar que los
    outliers de 5/6 entren en la fase de selección competitiva.
    """

    def __init__(self):
        self.ai_model = LotteryAIModel()
        self._matrix_cache = {"cluster_matrix": None}

    def calculate_resonance(self, u_xp, history, config, xp):
        """Ejecuta el flujo de radar híbrido y fusión V6.3."""
        if not self.ai_model.is_trained:
            self.ai_model.train(history.winning_numbers, config.total_balls)

        # --- FASE 1: SCORES BASE Y GEOMETRÍA ---
        ai_scores, geo_scores = self._compute_base_scores(u_xp, history, config, xp)

        # Normalización para balanceo de radar
        ai_norm = (ai_scores - ai_scores.min()) / (
            ai_scores.max() - ai_scores.min() + 1e-10
        )
        geo_norm = (geo_scores - geo_scores.min()) / (
            geo_scores.max() - geo_scores.min() + 1e-10
        )

        # --- FASE 2: RADAR HÍBRIDO (V6.3) ---
        # Definimos quién entra al radar basado en un balance 40/60
        # Damos prioridad a la Geometría porque detectó mejor los 5/6 en el log forense.
        hybrid_radar_score = (ai_norm * 0.4) + (geo_norm * 0.6)

        # Seleccionamos los mejores 30,000 del universo recuperado (112-128)
        radar_indices = xp.argsort(hybrid_radar_score)[-30000:][::-1]
        u_reduced = u_xp[radar_indices]

        # --- FASE 3: OMEGA POWER SQUEEZE ---
        omega_scores, breakdown = self.ai_model.score_tickets(
            u_reduced, return_breakdown=True
        )
        omega_signal = xp.asarray(breakdown.get("omega_hunter", omega_scores))
        omega_norm = (omega_signal - omega_signal.min()) / (
            omega_signal.max() - omega_signal.min() + 1e-10
        )

        # Recuperación de parámetros desde el DTO
        alpha_val = getattr(config, "hybrid_alpha", 0.50)
        beta_val = 1.0 - alpha_val

        # Boost No-Lineal Suavizado para mantener la consistencia en 5/6
        omega_boosted = xp.power(omega_norm, 1.8)

        # Fusión Final (En el radar reducido)
        final_scores_reduced = (ai_norm[radar_indices] * alpha_val) + (
            omega_boosted * beta_val
        )

        return {
            "u_reduced": u_reduced,
            "final_scores_reduced": final_scores_reduced,
            "radar_indices": radar_indices,
            "ai_norm": ai_norm,
            "geo_scores": geo_scores,
            "geo_matrix_xp": xp.asarray(self._matrix_cache["cluster_matrix"]),
        }

    def _compute_base_scores(self, u_xp, history, config, xp):
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
