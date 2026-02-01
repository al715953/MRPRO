# src/core/ai_scorer.py
import numpy as np
from sklearn.preprocessing import StandardScaler
from xgboost import XGBRegressor
from src.data_access.config import BEST_SETTINGS

try:
    import cupy as cp

    HAS_GPU = True
except ImportError:
    HAS_GPU = False


class LotteryAIModel:
    """
    Motor V7.14: Neural-Precision & Deep-Ensemble.
    Incrementa la profundidad de análisis y ajusta la tasa de aprendizaje
    para capturar desviaciones de precisión en sorteos de alta complejidad.
    """

    def __init__(self):
        self.config = BEST_SETTINGS
        self.experts = {}
        self.scaler = StandardScaler()
        self.is_trained = False
        self._build_ensemble()

    def _build_ensemble(self):
        """
        Construye el ensamble con mayor profundidad de árboles (Neural-Precision).
        """
        # Ajustes de hiperparámetros para V7.14
        # Reducimos learning_rate para una convergencia más lenta pero precisa
        fine_learning_rate = 0.008  # Antes 0.012
        # Aumentamos n_estimators para compensar el learning_rate más bajo
        deep_estimators = 4500  # Antes 3000

        cfg = self.config.get(
            "ensemble_config",
            {
                "alpha_ancla": {
                    "depth": 12,  # Incrementado de 6-8 a 12 para mayor detalle
                    "weight": 1.2,  # Aumento de peso en el modelo principal
                    "objective": "reg:pseudohubererror",
                },
                "beta_trend": {
                    "depth": 10,
                    "weight": 0.8,
                    "objective": "reg:squarederror",
                },
            },
        )

        for name, params in cfg.items():
            self.experts[name] = {
                "model": XGBRegressor(
                    n_estimators=deep_estimators,
                    learning_rate=fine_learning_rate,
                    max_depth=params.get("depth", 10),
                    objective=params.get("objective", "reg:squarederror"),
                    # Configuración de regularización para evitar overfit por profundidad
                    reg_alpha=0.1,
                    reg_lambda=1.5,
                    device="cuda" if HAS_GPU else "cpu",
                    tree_method="hist" if HAS_GPU else "auto",
                    random_state=42,
                ),
                "weight": params.get("weight", 0.5),
            }

    def train(self, winning_numbers, total_balls):
        if not winning_numbers:
            return

        # Preparación de muestras
        X_pos = np.array([sorted(draw[:6]) for draw in winning_numbers])
        y_pos = np.ones(len(X_pos), dtype=np.float32)

        # Generar ruido sintético con mayor diversidad para el entrenamiento profundo
        raw_noise = np.random.randint(1, total_balls + 1, (len(X_pos) * 2, 6))
        X_neg = np.sort(raw_noise, axis=1)
        # Eliminar duplicados exactos del set negativo que podrían estar en el positivo
        y_neg = np.zeros(len(X_neg), dtype=np.float32)

        X_train = np.vstack((X_pos, X_neg))
        y_train = np.concatenate((y_pos, y_neg))

        # El escalador es vital para la sensibilidad del modelo profundo
        X_scaled = self.scaler.fit_transform(X_train)

        for name, expert in self.experts.items():
            expert["model"].fit(X_scaled, y_train)

        self.is_trained = True

    def score_tickets(self, candidates, return_breakdown=False):
        if not self.is_trained or candidates is None or len(candidates) == 0:
            return np.array([], dtype=np.float32)

        # Sincronización CPU/GPU para el escalador
        cand_cpu = candidates.get() if hasattr(candidates, "get") else candidates
        X_scaled = self.scaler.transform(cand_cpu)

        if HAS_GPU:
            X_input = cp.asarray(X_scaled)
        else:
            X_input = X_scaled

        scores_pool = []
        for name, expert in self.experts.items():
            pred = expert["model"].predict(X_input)
            # Aplicamos el peso del experto al puntaje crudo
            scores_pool.append(np.maximum(pred, 1e-6) * expert["weight"])

        return np.sum(scores_pool, axis=0)
