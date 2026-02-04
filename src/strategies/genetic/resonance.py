# src/strategies/genetic/resonance.py
import os
import numpy as np
import xgboost as xgb
import itertools
from src.data_access.config import BEST_SETTINGS

class ResonanceEngine:
    """
    Motor de Resonancia V7.21 (Gold Master - Adaptive Fusion).
    - Corte Top 50% (Equilibrio perfecto).
    - Fusión Adaptativa: La IA toma el control si el Geo-Score es débil.
    - Sin ruido térmico.
    """

    def __init__(self):
        self.model_path = "mrpro_model_v7.json"
        self.bst = None
        self._matrix_cache = {"cluster_matrix": None}
        
        if os.path.exists(self.model_path):
            try:
                self.bst = xgb.Booster()
                self.bst.load_model(self.model_path)
            except Exception as e:
                print(f"⚠️ Alerta: Modelo corrupto. ({e})")
                self.bst = None
        else:
            print("⚠️ Modo 'Entrenamiento en Vivo' Activado.")

    def calculate_resonance(self, u_xp, history, config, xp):
        # 0. Validación
        if u_xp is None or len(u_xp) == 0: return None

        # 1. Preparación
        raw_history = history.winning_numbers if hasattr(history, 'winning_numbers') else history
        n_balls = config.total_balls

        if self.bst is None: self._train_jit_model(raw_history, n_balls)

        # Cálculo estructural para el Selector
        recent_draws = raw_history[-5:]
        flat_recent = set([num for draw in recent_draws for num in draw[:6]])
        thermal_numbers = sorted(list(set(range(1, n_balls + 1)) - flat_recent))

        if self._matrix_cache["cluster_matrix"] is None: self._build_nexus_matrix(raw_history, n_balls)
        geo_matrix_xp = xp.asarray(self._matrix_cache["cluster_matrix"])

        # 2. AI Score
        candidates_cpu = np.ascontiguousarray(u_xp.get()) if hasattr(u_xp, "get") else np.ascontiguousarray(u_xp)
        dtest = xgb.DMatrix(candidates_cpu)
        ai_scores = xp.asarray(self.bst.predict(dtest))

        # 3. Geo Score
        geo_scores = self._compute_geo_score(u_xp, raw_history, xp)

        # 4. Normalización
        ai_min, ai_max = ai_scores.min(), ai_scores.max()
        div = (ai_max - ai_min) if (ai_max - ai_min) > 0 else 1.0
        ai_norm = (ai_scores - ai_min) / div

        # --- ESTRATEGIA: CORTE EQUILIBRADO (Top 50%) ---
        cutoff = xp.percentile(ai_norm, 50) 
        radar_indices = xp.where(ai_norm >= cutoff)[0]
        
        if len(radar_indices) < 100: radar_indices = xp.arange(len(u_xp))

        u_reduced = u_xp[radar_indices]
        ai_subset = ai_norm[radar_indices]
        geo_subset = geo_scores[radar_indices]

        # --- ESTRATEGIA: FUSIÓN ADAPTATIVA (Gold Standard) ---
        # Si Geo > 0.4, peso 60% Geo. Si no, peso 20% Geo.
        is_geo_strong = (geo_subset > 0.4)
        
        w_ai = xp.where(is_geo_strong, 0.40, 0.80)
        w_geo = xp.where(is_geo_strong, 0.60, 0.20)
        
        final_scores_reduced = (ai_subset * w_ai) + (geo_subset * w_geo)

        return {
            "u_reduced": u_reduced,
            "final_scores_reduced": final_scores_reduced,
            "radar_indices": radar_indices,
            "ai_norm": ai_norm,
            "geo_scores": geo_scores,
            "geo_matrix_xp": geo_matrix_xp,
            "thermal_numbers": thermal_numbers
        }

    def _train_jit_model(self, history, n_balls):
        X_pos = np.array([d[:6] for d in history], dtype=np.uint8)
        n_neg = len(X_pos)
        X_neg = np.random.randint(1, n_balls + 1, size=(n_neg, 6)).astype(np.uint8)
        X_neg.sort(axis=1)
        X = np.vstack([X_pos, X_neg])
        y = np.hstack([np.ones(len(X_pos)), np.zeros(len(X_neg))])
        
        dtrain = xgb.DMatrix(X, label=y)
        params = {
            "objective": "binary:logistic",
            "max_depth": 7, "eta": 0.2, 
            "tree_method": "hist", "eval_metric": "logloss"
        }
        self.bst = xgb.train(params, dtrain, num_boost_round=30)

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