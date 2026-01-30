import numpy as np
import warnings
from sklearn.preprocessing import StandardScaler
from xgboost import XGBRegressor
from src.data_access.config import BEST_SETTINGS

warnings.filterwarnings("ignore", category=UserWarning)

try:
    import cupy as cp

    HAS_GPU = True
except ImportError:
    HAS_GPU = False
    cp = None


class LotteryAIModel:
    """
    ENGINE V5.3: Gamma Flexible Edition.
    Ajusta la curvatura de aprendizaje para capturar 'vecindarios' ganadores.
    """

    def __init__(self):
        self.config = BEST_SETTINGS
        self.experts = {}
        self.scaler = StandardScaler()
        self.is_trained = False
        self._build_ensemble()

    def _build_ensemble(self):
        # CONFIGURACIÓN V5.3: Gamma 1.5 para mayor flexibilidad
        base_params = {
            "n_estimators": self.config.get("n_estimators", 2200),
            "learning_rate": 0.03,
            "subsample": 0.85,
            "colsample_bytree": 0.9,
            "objective": "reg:pseudohubererror",
            "device": "cuda" if HAS_GPU else "cpu",
            "tree_method": "hist" if HAS_GPU else "auto",
            "gamma": 0.8,  # Reducción táctica para evitar rigidez
        }

        ensemble_cfg = self.config.get("ensemble_config", {})
        for name, cfg in ensemble_cfg.items():
            params = base_params.copy()
            params["max_depth"] = cfg["depth"]
            self.experts[name] = {
                "model": XGBRegressor(**params),
                "weight": cfg["weight"],
            }

    def _extract_features(self, data):
        if isinstance(data, list):
            data = np.asarray(data)
        xp = cp.get_array_module(data) if HAS_GPU else np

        sums = xp.sum(data, axis=1)
        stds = xp.std(data, axis=1)
        evens = xp.sum(data % 2 == 0, axis=1)

        primes_arr = xp.array([2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37])
        prime_count = xp.sum(xp.isin(data, primes_arr), axis=1)

        # Aceleración de segundo orden
        d1 = xp.diff(data, axis=1)
        avg_acceleration = xp.mean(xp.abs(xp.diff(d1, axis=1)), axis=1)

        # RASGO DE SEGURIDAD: Armonía Estructural
        harmony = (sums / 120.0) * (stds / 11.0) * ((evens + 1) / 4.0)

        features = xp.column_stack(
            [sums, stds, evens, prime_count, avg_acceleration, harmony]
        )
        return (features.get() if hasattr(features, "get") else features), harmony

    def train(self, history_draws, total_balls):
        if len(history_draws) < 50:
            return
        winners = [sorted(d[:6]) for d in history_draws]
        X_pos, _ = self._extract_features(winners)
        y_pos = np.linspace(0.9, 1.0, len(winners), dtype=np.float32)

        # Ruido controlado para el Láser
        raw_noise = np.sort(
            np.random.choice(np.arange(1, total_balls + 1), (len(winners) * 12, 6)),
            axis=1,
        )
        X_neg, _ = self._extract_features(raw_noise)
        y_neg = np.zeros(len(X_neg), dtype=np.float32)

        X_train = self.scaler.fit_transform(np.vstack((X_pos, X_neg)))
        y_train = np.concatenate((y_pos, y_neg))

        for name, expert in self.experts.items():
            expert["model"].fit(X_train, y_train)
        self.is_trained = True

    def score_tickets(self, candidates, return_breakdown=False):
        if not self.is_trained or candidates is None:
            return np.array([])

        features, harmony = self._extract_features(candidates)
        X_scaled = self.scaler.transform(features)

        breakdown = {}
        resonance_pool = []

        for name, expert in self.experts.items():
            score = np.maximum(expert["model"].predict(X_scaled), 1e-6)
            breakdown[name] = score
            # Fusión Logarítmica para expandir micro-señales
            resonance_pool.append(np.log1p(score) * expert["weight"])

        final_energy = np.sum(resonance_pool, axis=0)

        # Seguro de Armonía Estructural (Anti-Singularidad)
        harmony_val = harmony.get() if hasattr(harmony, "get") else harmony
        final_energy = np.maximum(final_energy, harmony_val * 0.05)

        # Normalización Dinámica
        f_max = final_energy.max() + 1e-10
        final_energy = final_energy / f_max

        return (final_energy, breakdown) if return_breakdown else final_energy
