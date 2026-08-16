# src/strategies/genetic/resonance.py
import os
import numpy as np
import xgboost as xgb
import itertools
from src.data_access.config import (
    BACKTEST_MODEL_FILE_PATH,
    BACKTEST_NUMBER_MODEL_FILE_PATH,
    BEST_SETTINGS,
    MODEL_FILE_PATH,
    NUMBER_MODEL_FILE_PATH,
)
from src.core.melate_features import (
    FEATURE_NAMES as MELATE_FEATURE_NAMES,
    FEATURE_SCHEMA as MELATE_FEATURE_SCHEMA,
    build_candidate_features,
)
from src.core.melate_number_model import (
    predict_number_probabilities,
    score_tickets_from_number_probs,
)


class ResonanceEngine:
    """
    Motor de Resonancia V8.2 (Bulletproof).
    - Prevención total de AI=0.0000 / Rank #0.
    - Carga robusta de modelo estático.
    - Fallback a Random inteligente si el cerebro falla, para no detener la simulación.
    """

    def __init__(self, model_path=None, number_model_path=None):
        # 1. Búsqueda de Modelo (Ruta Absoluta Dinámica)
        self.model_path = model_path or MODEL_FILE_PATH
        self.bst = None
        inferred_backtest = os.path.abspath(self.model_path) == os.path.abspath(
            BACKTEST_MODEL_FILE_PATH
        )
        self.number_model_path = number_model_path or (
            BACKTEST_NUMBER_MODEL_FILE_PATH if inferred_backtest else NUMBER_MODEL_FILE_PATH
        )
        self.number_bst = None
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
            print(f"❌ FATAL: Model not found at {self.model_path}.")

        if os.path.exists(self.number_model_path):
            try:
                self.number_bst = xgb.Booster()
                self.number_bst.load_model(self.number_model_path)
            except Exception as e:
                print(f"⚠️ Error loading number model: {e}")
                self.number_bst = None

    @property
    def training_cutoff_contest(self):
        """Last contest visible during training, when the model records it."""
        if self.bst is None:
            return None
        raw = self.bst.attr("trained_through_concurso")
        try:
            return int(raw) if raw is not None else None
        except (TypeError, ValueError):
            return None

    @property
    def temporal_holdout_auc(self):
        if self.bst is None:
            return None
        raw = self.bst.attr("temporal_holdout_auc")
        try:
            return float(raw) if raw is not None else None
        except (TypeError, ValueError):
            return None

    @property
    def feature_schema(self):
        if self.bst is None:
            return None
        return self.bst.attr("feature_schema")

    @property
    def number_temporal_holdout_auc(self):
        if self.number_bst is None:
            return None
        raw = self.number_bst.attr("temporal_holdout_auc")
        try:
            return float(raw) if raw is not None else None
        except (TypeError, ValueError):
            return None

    @property
    def ai_signal_enabled(self):
        """A loaded model remains active; validation is reported separately."""
        return self.bst is not None

    @property
    def ai_signal_validated(self):
        """Whether the model beat the minimum out-of-sample ranking threshold."""
        auc = self.temporal_holdout_auc
        return auc is None or auc >= 0.51

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
        candidates_cpu = (
            u_xp.get().astype(np.uint8)
            if hasattr(u_xp, "get")
            else np.asarray(u_xp).astype(np.uint8)
        )

        ai_prediction_ok = False
        if self.bst:
            try:
                if self.feature_schema == MELATE_FEATURE_SCHEMA:
                    model_features = build_candidate_features(
                        candidates_cpu,
                        raw_history,
                    )
                    dtest = xgb.DMatrix(
                        model_features,
                        feature_names=list(MELATE_FEATURE_NAMES),
                    )
                else:
                    model_features = candidates_cpu
                    dtest = xgb.DMatrix(model_features)
                ai_scores_cpu = self.bst.predict(dtest)
                ai_scores = xp.asarray(ai_scores_cpu)
                ai_prediction_ok = True
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
        contextual_norm = (ai_scores - ai_min) / div

        number_prediction_ok = False
        number_norm = xp.zeros(n_candidates, dtype=xp.float32)
        if self.number_bst is not None:
            try:
                number_probs = predict_number_probabilities(
                    self.number_bst,
                    raw_history,
                )
                number_scores_cpu = score_tickets_from_number_probs(
                    candidates_cpu,
                    number_probs,
                )
                number_scores = xp.asarray(number_scores_cpu)
                number_min, number_max = number_scores.min(), number_scores.max()
                number_div = number_max - number_min
                if number_div == 0:
                    number_div = 1.0
                number_norm = (number_scores - number_min) / number_div
                number_prediction_ok = True
            except Exception as e:
                print(f"⚠️ Number AI Prediction Failed: {e}")

        overrides = getattr(config, "filter_overrides", None) or BEST_SETTINGS
        context_weight = max(0.0, float(overrides.get("ai_context_weight", 1.0)))
        number_weight = max(0.0, float(overrides.get("ai_number_weight", 0.0)))
        if not number_prediction_ok:
            context_weight, number_weight = 1.0, 0.0
        weight_total = context_weight + number_weight
        if weight_total <= 0:
            context_weight, number_weight, weight_total = 1.0, 0.0, 1.0
        ai_norm = (
            contextual_norm * (context_weight / weight_total)
            + number_norm * (number_weight / weight_total)
        )
        ai_prediction_ok = bool(ai_prediction_ok or number_prediction_ok)
        ai_active = bool(ai_prediction_ok and self.ai_signal_enabled)
        ai_effective = ai_norm if ai_active else xp.zeros_like(ai_norm)

        # 6. Lógica V8.1 (Safety Net + Hybrid Cutoff)
        blend_mode = str(overrides.get("resonance_blend_mode", "adaptive")).lower()
        fixed_blend = blend_mode == "fixed"
        if fixed_blend:
            hybrid_alpha = max(0.0, float(overrides.get("hybrid_alpha", 0.5)))
            hybrid_beta = max(0.0, float(overrides.get("hybrid_beta", 0.5)))
            hybrid_weight_total = hybrid_alpha + hybrid_beta
            if hybrid_weight_total <= 0:
                hybrid_alpha, hybrid_beta, hybrid_weight_total = 0.5, 0.5, 1.0
            hybrid_alpha /= hybrid_weight_total
            hybrid_beta /= hybrid_weight_total
            hybrid_signal = (
                ai_effective * hybrid_alpha + geo_scores * hybrid_beta
            )
        else:
            # Conserva exactamente el comportamiento histórico para producción.
            hybrid_alpha, hybrid_beta = 0.5, 0.5
            hybrid_signal = (ai_effective + geo_scores) / 2.0

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
        ai_subset = ai_effective[radar_indices]
        geo_subset = geo_scores[radar_indices]

        if fixed_blend:
            final_scores_reduced = (
                ai_subset * hybrid_alpha + geo_subset * hybrid_beta
            )
        else:
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
            "ai_signal_enabled": ai_active,
            "ai_signal_validated": self.ai_signal_validated,
            "temporal_holdout_auc": self.temporal_holdout_auc,
            "feature_schema": self.feature_schema or "legacy_ticket_numbers",
            "number_ai_scores": number_norm,
            "number_model_enabled": number_prediction_ok,
            "number_model_applied": bool(number_prediction_ok and number_weight > 0),
            "number_temporal_holdout_auc": self.number_temporal_holdout_auc,
            "ai_context_weight": float(context_weight / weight_total),
            "ai_number_weight": float(number_weight / weight_total),
            "resonance_blend_mode": "fixed" if fixed_blend else "adaptive",
            "hybrid_alpha": float(hybrid_alpha),
            "hybrid_beta": float(hybrid_beta),
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
