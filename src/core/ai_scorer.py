import numpy as np
from sklearn.preprocessing import StandardScaler
from xgboost import XGBRegressor
from src.data_access.config import BEST_SETTINGS
import warnings

warnings.filterwarnings("ignore", category=UserWarning)

try:
    import cupy as cp

    HAS_GPU = True
except ImportError:
    HAS_GPU = False


class LotteryAIModel:
    """
    ENGINE V4.6: Contrast Injection Edition.
    Introduce rasgos de paridad y deltas para romper el bloqueo de energía 0.15.
    """

    def __init__(self):
        self.config = BEST_SETTINGS
        self.experts = {}
        self.scaler = StandardScaler()
        self.is_trained = False
        self._build_ensemble()

    def _build_ensemble(self):
        """Instancia la Trifecta basándose exclusivamente en config.py."""
        # Configuración base para alto rendimiento en NVIDIA RTX 4070 Ti
        base_params = {
            "n_estimators": self.config.get("n_estimators", 2000),
            "learning_rate": self.config.get("learning_rate", 0.015),
            "subsample": 0.8,
            "colsample_bytree": 0.8,
            "objective": "reg:squarederror",
            "device": "cuda" if HAS_GPU else "cpu",
            "tree_method": "hist" if HAS_GPU else "auto",
            "gamma": self.config.get("gamma", 4.0),
        }

        # Carga dinámica de expertos desde BEST_SETTINGS
        ensemble_cfg = self.config.get("ensemble_config", {})
        for name, cfg in ensemble_cfg.items():
            params = base_params.copy()
            params["max_depth"] = cfg["depth"]
            self.experts[name] = {
                "model": XGBRegressor(**params),
                "weight": cfg["weight"],
            }

    def _extract_features(self, combinations):
        """EXTRACTOR DE ADN: Añade Paridad y Saltos (Deltas)."""
        data = (
            combinations.get()
            if hasattr(combinations, "get")
            else np.array(combinations)
        )
        if data.ndim == 1:
            data = data.reshape(1, -1)
        data = data.astype(np.float32)

        # 1. Rasgos Básicos (Mantenemos por consistencia)
        sums = data.sum(axis=1)
        stds = data.std(axis=1)
        ranges = data.max(axis=1) - data.min(axis=1)

        # 2. NUEVO: RASGOS DE CONTRASTE SNIPER
        # Paridad: ¿Cuántos pares hay? (Jackpots suelen tener balance 3/3 o 2/4)
        evens = np.sum(data % 2 == 0, axis=1)

        # Deltas: Saltos entre números (Detección de 'Clústeres' vs 'Dispersión')
        deltas = np.diff(data, axis=1)
        avg_delta = np.mean(deltas, axis=1)
        max_delta = np.max(deltas, axis=1)

        # 3. Armonía Cuántica Mejorada
        harmony = (sums / 120.0) * (stds / 11.0) * (evens / 3.0)

        return np.column_stack(
            (sums, stds, ranges, harmony, evens, avg_delta, max_delta)
        )

    def train(self, history_draws, total_balls):
        """Entrenamiento con Soft-Labeling Agresivo."""
        if len(history_draws) < 50:
            return

        winners = [sorted(d[:6]) for d in history_draws]
        X_pos = self._extract_features(winners)

        # Subimos la vara del Labeling para forzar contraste
        y_pos = np.linspace(0.85, 1.0, len(winners), dtype=np.float32)

        # Aumentamos el ruido negativo para que la IA aprenda a decir "NO"
        n_neg = len(winners) * 8
        raw_noise = np.sort(
            np.random.choice(np.arange(1, total_balls + 1), (n_neg, 6)), axis=1
        )
        X_neg = self._extract_features(raw_noise)
        y_neg = np.zeros(len(X_neg), dtype=np.float32)

        X_train = self.scaler.fit_transform(np.vstack((X_pos, X_neg)))
        y_train = np.concatenate((y_pos, y_neg))

        for name, expert in self.experts.items():
            expert["model"].fit(X_train, y_train)

        self.is_trained = True

        self.is_trained = True

    def score_tickets(self, candidates, return_breakdown=False):
        """
        Calcula la Energía de Proximidad sin valores harcodeados.
        Opcionalmente devuelve el desglose de cada experto para el log forense.
        """
        if not self.is_trained or candidates is None or len(candidates) == 0:
            return np.array([]) if not return_breakdown else (np.array([]), {})

        X_scaled = self.scaler.transform(self._extract_features(candidates))

        # 1. SUMA SINÉRGICA DINÁMICA
        final_energy = np.zeros(len(candidates))
        breakdown = {}

        # Iteramos sobre los expertos definidos en config.py
        for name, expert in self.experts.items():
            expert_score = expert["model"].predict(X_scaled)
            # Aplicamos el peso dinámico del config
            final_energy += expert_score * expert["weight"]

            if return_breakdown:
                breakdown[name] = expert_score

        # 2. BOOST DE ALERTA DINÁMICO
        # Umbral tomado de BEST_SETTINGS
        threshold = self.config.get("mutant_threshold_omega", 0.92)

        # Buscamos el score del modelo Omega si existe
        s_omega = breakdown.get("omega_hunter")
        if s_omega is None and "omega_hunter" in self.experts:
            s_omega = self.experts["omega_hunter"]["model"].predict(X_scaled)

        if s_omega is not None:
            alert_bonus = np.where(s_omega > threshold, 0.15, 0.0)
            final_energy = np.clip(final_energy + alert_bonus, 0, 1.0)

        if return_breakdown:
            return final_energy, breakdown

        return final_energy
