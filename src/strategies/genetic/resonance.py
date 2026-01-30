# src/strategies/genetic/resonance.py

import numpy as np
import itertools
from src.core.ai_scorer import LotteryAIModel


class ResonanceEngine:
    """
    Motor V7.5: Quantum Slot-Mapping & Residue Fusion.
    Corregido para eliminar KeyErrors e inconsistencias de IDE.
    """

    def __init__(self):
        self.ai_model = LotteryAIModel()
        self._matrix_cache = {"cluster_matrix": None}

    def calculate_resonance(self, u_xp, history, config, xp):
        # 1. Recuperación de historial (Fix winning_numbers)
        raw_history = history.winning_numbers

        if not self.ai_model.is_trained:
            self.ai_model.train(raw_history, config.total_balls)

        # --- FASE 1: SCORES BASE ---
        ai_scores, geo_scores_raw = self._compute_base_scores(u_xp, history, config, xp)

        ai_norm = (ai_scores - ai_scores.min()) / (
            ai_scores.max() - ai_scores.min() + 1e-10
        )
        geo_norm = (geo_scores_raw - geo_scores_raw.min()) / (
            geo_scores_raw.max() - geo_scores_raw.min() + 1e-10
        )

        # --- FASE 2: RADAR HÍBRIDO ---
        hybrid_radar_score = (ai_norm * 0.35) + (geo_norm * 0.65)
        radar_indices = xp.argsort(hybrid_radar_score)[-30000:][::-1]
        u_reduced = u_xp[radar_indices]

        # --- FASE 3: EXTRACCIÓN DE SEÑALES V7.5 ---
        # 3.1 Señal Omega (Fix omega_boosted visibility)
        omega_scores, breakdown = self.ai_model.score_tickets(
            u_reduced, return_breakdown=True
        )
        omega_signal = xp.asarray(breakdown.get("omega_hunter", omega_scores))
        omega_norm = (omega_signal - omega_signal.min()) / (
            omega_signal.max() - omega_signal.min() + 1e-10
        )
        omega_boosted = xp.power(omega_norm, 4.5)

        # 3.2 Slot-Mapping (Frecuencia por Posición)
        hist_arr = xp.asarray(raw_history)
        slot_bonus = xp.zeros(len(u_reduced))
        for pos in range(6):
            counts = xp.bincount(hist_arr[:, pos], minlength=40)
            freqs = counts / (len(hist_arr) + 1e-10)
            slot_bonus += freqs[u_reduced[:, pos]]

        slot_norm = (slot_bonus - slot_bonus.min()) / (
            slot_bonus.max() - slot_bonus.min() + 1e-10
        )

        # 3.3 Bias Posicional (Dispersión std)
        u_std = xp.std(u_reduced, axis=1)
        std_bias = xp.exp(-xp.power(u_std - 10.5, 2) / (2 * xp.power(1.2, 2)))

        # --- FASE 4: FUSIÓN FINAL ---
        final_scores_reduced = (
            (ai_norm[radar_indices] * 0.10)
            + (omega_boosted * 0.50)
            + (geo_norm[radar_indices] * 0.25)
            + (std_bias * 0.05)
            + (slot_norm * 0.10)
        )

        return {
            "u_reduced": u_reduced,
            "final_scores_reduced": final_scores_reduced,
            "radar_indices": radar_indices,
            "ai_norm": ai_norm,
            "geo_scores": geo_norm,  # <--- FIX: Nombre clave para genetic_selector.py
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
