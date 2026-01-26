import numpy as np
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier
import warnings

warnings.filterwarnings("ignore", category=UserWarning)

try:
    import cupy as cp

    HAS_GPU = True
except ImportError:
    HAS_GPU = False


class LotteryAIModel:
    def __init__(self):
        # FIX: Inicialización robusta para evitar AttributeError
        gpu_params = (
            {"device": "cuda", "tree_method": "hist", "predictor": "gpu_predictor"}
            if HAS_GPU
            else {"tree_method": "auto", "predictor": "cpu_predictor"}
        )

        self.model = XGBClassifier(
            n_estimators=2000,
            max_depth=9,
            learning_rate=0.01,
            subsample=0.85,
            colsample_bytree=0.85,
            gamma=3.0,
            reg_alpha=1.0,
            reg_lambda=2.0,
            objective="binary:logistic",
            eval_metric="logloss",
            n_jobs=-1,
            scale_pos_weight=7.5,
            **gpu_params
        )
        self.scaler = StandardScaler()
        self.is_trained = False

    def _extract_features(self, combinations):
        if hasattr(combinations, "get"):
            data = combinations.get().astype(np.int32)
        else:
            data = np.array(combinations, dtype=np.int32)
        if data.ndim == 1:
            data = data.reshape(1, -1)

        sums = data.sum(axis=1)
        stds = data.std(axis=1)
        ranges = data.max(axis=1) - data.min(axis=1)
        return np.column_stack((sums, stds, ranges, (sums / 120.0) * (stds / 11.0)))

    def train(self, history_draws, total_balls):
        if len(history_draws) < 50:
            return
        winners = [tuple(sorted(d[:6])) for d in history_draws]
        X_pos = self._extract_features(winners)
        y_pos = np.ones(len(winners), dtype=np.int32)  # FIX: Labels como int32

        # Generación de ruido balanceado para evitar ValueError de clases únicas
        n_neg = len(winners) * 5
        raw_noise = np.sort(
            np.random.choice(np.arange(1, total_balls + 1), (n_neg, 6)), axis=1
        )
        X_neg = self._extract_features(raw_noise)
        y_neg = np.zeros(len(X_neg), dtype=np.int32)

        self.model.fit(
            self.scaler.fit_transform(np.vstack((X_pos, X_neg))),
            np.concatenate((y_pos, y_neg)),
        )
        self.is_trained = True

    def score_tickets(self, candidates):
        if not self.is_trained or candidates is None or len(candidates) == 0:
            return np.array([])
        X_scaled = self.scaler.transform(self._extract_features(candidates))
        return self.model.predict_proba(X_scaled)[:, 1]
