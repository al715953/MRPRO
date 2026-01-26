import numpy as np
from sklearn.preprocessing import StandardScaler
from xgboost import XGBRegressor
import warnings

warnings.filterwarnings("ignore", category=UserWarning)

try:
    import cupy as cp

    HAS_GPU = True
except ImportError:
    HAS_GPU = False


class LotteryAIModel:
    def __init__(self):
        # Configuración para alto rendimiento en RTX 4070 Ti
        gpu_params = (
            {"device": "cuda", "tree_method": "hist"}
            if HAS_GPU
            else {"tree_method": "auto"}
        )

        self.model = XGBRegressor(
            n_estimators=2000,
            max_depth=9,  # Captura la topología del 5/6
            learning_rate=0.015,
            subsample=0.8,
            colsample_bytree=0.8,
            objective="reg:squarederror",
            **gpu_params
        )
        self.scaler = StandardScaler()
        self.is_trained = False

    def _extract_features(self, combinations):
        """Blindaje de Handshake: Conversión segura de GPU a CPU."""
        # Verificamos si los datos vienen de CuPy antes de procesar
        if hasattr(combinations, "get"):
            data = combinations.get().astype(np.int32)
        else:
            data = np.array(combinations, dtype=np.int32)

        if data.ndim == 1:
            data = data.reshape(1, -1)

        # Lógica de extracción de características de MRPRO
        sums = data.sum(axis=1)
        stds = data.std(axis=1)
        ranges = data.max(axis=1) - data.min(axis=1)

        # Métrica de consistencia para el modelo de regresión
        return np.column_stack((sums, stds, ranges, (sums / 120.0) * (stds / 11.0)))

    def train(self, history_draws, total_balls):
        if len(history_draws) < 50:
            return

        winners = [sorted(d[:6]) for d in history_draws]
        X_pos = self._extract_features(winners)

        # Soft-Labeling para detectar proximidad al éxito
        y_pos = np.linspace(0.8, 1.0, len(winners), dtype=np.float32)

        n_neg = len(winners) * 5
        raw_noise = np.sort(
            np.random.choice(np.arange(1, total_balls + 1), (n_neg, 6)), axis=1
        )
        X_neg = self._extract_features(raw_noise)
        y_neg = np.zeros(len(X_neg), dtype=np.float32)

        self.model.fit(
            self.scaler.fit_transform(np.vstack((X_pos, X_neg))),
            np.concatenate((y_pos, y_neg)),
        )
        self.is_trained = True

    def score_tickets(self, candidates):
        """Calcula la Energía de Proximidad sin romper el bus de datos."""
        if not self.is_trained or candidates is None or len(candidates) == 0:
            return np.array([])

        # El escalador y el modelo procesan en CPU tras el .get()
        X_scaled = self.scaler.transform(self._extract_features(candidates))
        return self.model.predict(X_scaled)
