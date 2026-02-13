# src/strategies/genetic/resonance.py
import os
import numpy as np
import xgboost as xgb
import itertools
from src.data_access.config import BEST_SETTINGS, MODEL_FILE_PATH


class ResonanceEngine:
    """
    Motor de Resonancia V8.2 (Bulletproof).
    - Prevención total de AI=0.0000 / Rank #0.
    - Carga robusta de modelo estático.
    - Fallback a Random inteligente si el cerebro falla, para no detener la simulación.
    """

    def __init__(self):
        # 1. Búsqueda de Modelo (Ruta Absoluta Dinámica)
        self.model_path = MODEL_FILE_PATH
        self.bst = None
        self._matrix_cache = {"cluster_matrix": None}

        # Intentamos cargar
        if os.path.exists(self.model_path):
            try:
                self.bst = xgb.Booster()
                self.bst.load_model(self.model_path)
                # print("✅ V8 Brain Loaded.")
            except Exception as e:
                print(f"⚠️ Error loading V8: {e}")
                self.bst = None
        else:
            # Fallback a búsqueda local simple
            if os.path.exists("mrpro_model_v8_static.json"):
                self.bst = xgb.Booster()
                self.bst.load_model("mrpro_model_v8_static.json")
            else:
                print(f"❌ FATAL: Model not found at {self.model_path} or local.")

    def calculate_resonance(self, u_xp, history, config, xp):
        # 0. Validación Básica
        if u_xp is None or len(u_xp) == 0:
            return None

        # 1. Preparación de Datos
        raw_history = (
            history.winning_numbers if hasattr(history, "winning_numbers") else history
        )
        n_balls = config.total_balls

        # 2. Matriz Nexus
        if self._matrix_cache["cluster_matrix"] is None:
            self._build_nexus_matrix(raw_history, n_balls)
        geo_matrix_xp = xp.asarray(self._matrix_cache["cluster_matrix"])

        # 3. Geo Score (Siempre calculable)
        geo_scores = self._compute_geo_score(u_xp, raw_history, xp)

        # 4. AI Score (Con Red de Seguridad para Ceros)
        n_candidates = len(u_xp)

        if self.bst:
            try:
                # Conversión segura a CPU uint8
                if hasattr(u_xp, "get"):
                    candidates_cpu = u_xp.get().astype(np.uint8)
                else:
                    candidates_cpu = np.asarray(u_xp).astype(np.uint8)

                dtest = xgb.DMatrix(candidates_cpu)
                ai_scores_cpu = self.bst.predict(dtest)
                ai_scores = xp.asarray(ai_scores_cpu)
            except Exception as e:
                print(f"⚠️ AI Prediction Failed: {e}")
                ai_scores = xp.zeros(n_candidates, dtype=xp.float32)
        else:
            # Si no hay cerebro, AI score es 0
            ai_scores = xp.zeros(n_candidates, dtype=xp.float32)

        # 5. Normalización Segura (Evita división por cero)
        ai_min, ai_max = ai_scores.min(), ai_scores.max()
        div = ai_max - ai_min
        if div == 0:
            div = 1.0
        ai_norm = (ai_scores - ai_min) / div

        # 6. Lógica V8.1 (Safety Net + Hybrid Cutoff)
        hybrid_signal = (ai_norm + geo_scores) / 2.0

        # CORRECCIÓN V8.2: Si todo es cero (Hybrid=0), inyectamos ruido minúsculo
        # para que el sorteo tenga orden y no Rank #0
        if xp.max(hybrid_signal) == 0:
            # Ruido epsilon aleatorio para desempatar
            hybrid_signal += xp.random.rand(n_candidates) * 0.0001
            # print("⚠️ Zero Signal Detected -> Injecting Noise to prevent Rank #0")

        cutoff = xp.percentile(hybrid_signal, 50)
        radar_indices = xp.where(hybrid_signal >= cutoff)[0]

        # Fallback de cantidad mínima
        if len(radar_indices) < 100:
            radar_indices = xp.argsort(hybrid_signal)[-100:]  # Top 100 forzoso

        u_reduced = u_xp[radar_indices]
        ai_subset = ai_norm[radar_indices]
        geo_subset = geo_scores[radar_indices]

        # Safety Net Logic (Si AI < 0.15, Geo manda 90%)
        is_ai_confused = ai_subset < 0.15
        is_geo_strong = geo_subset > 0.4

        w_ai_std = xp.where(is_geo_strong, 0.40, 0.80)
        w_geo_std = xp.where(is_geo_strong, 0.60, 0.20)

        w_ai = xp.where(is_ai_confused, 0.10, w_ai_std)
        w_geo = xp.where(is_ai_confused, 0.90, w_geo_std)

        final_scores_reduced = (ai_subset * w_ai) + (geo_subset * w_geo)

        # Thermal numbers (Compatibility)
        recent_draws = raw_history[-5:]
        flat_recent = set([num for draw in recent_draws for num in draw[:6]])
        thermal_numbers = sorted(list(set(range(1, n_balls + 1)) - flat_recent))

        return {
            "u_reduced": u_reduced,
            "final_scores_reduced": final_scores_reduced,
            "radar_indices": radar_indices,
            "ai_norm": ai_norm,
            "geo_scores": geo_scores,
            "geo_matrix_xp": geo_matrix_xp,
            "thermal_numbers": thermal_numbers,
        }

    def _build_nexus_matrix(self, history, n_balls):
        m = np.zeros((n_balls + 1, n_balls + 1), dtype=np.uint16)
        for draw in history:
            for a, b in itertools.combinations(draw[:6], 2):
                if a <= n_balls and b <= n_balls:
                    m[a, b] += 1
                    m[b, a] += 1
        self._matrix_cache["cluster_matrix"] = m

    def _compute_geo_score(self, u_xp, history, xp):
        magnets = xp.array([d[:6] for d in history[-10:]], dtype=xp.uint8)
        geo_scores = xp.zeros(len(u_xp), dtype=xp.float32)
        weights = xp.array([0.0, 0.1, 0.3, 0.9, 0.5, 0.1, 0.0], dtype=xp.float32)
        for magnet in magnets:
            matches = xp.sum(xp.isin(u_xp, magnet), axis=1)
            geo_scores += weights[matches]
        g_min, g_max = geo_scores.min(), geo_scores.max()
        div = (g_max - g_min) if (g_max - g_min) > 0 else 1.0
        return (geo_scores - g_min) / div
