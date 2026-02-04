# src/core/ai_scorer.py
import numpy as np
from sklearn.preprocessing import StandardScaler
from xgboost import XGBRegressor
from src.data_access.config import BEST_SETTINGS

# En macOS (Apple Silicon), forzamos HAS_GPU a False para evitar conflictos con CUDA
HAS_GPU = False 

class LotteryAIModel:
    """
    Motor V7.16: Neural-Precision & Deep-Ensemble (Mac Optimized).
    Ajustado para procesar en CPU aprovechando la arquitectura del chip Apple.
    """

    def __init__(self):
        self.config = BEST_SETTINGS
        self.experts = {}
        self.scaler = StandardScaler()
        self.is_trained = False
        self._build_ensemble()

    def _build_ensemble(self):
        """
        Construye el ensamble con parámetros de Neural-Precision.
        """
        fine_learning_rate = 0.008  
        deep_estimators = 4500  

        # Configuración de los expertos desde BEST_SETTINGS
        cfg = self.config.get(
            "ensemble_config",
            {
                "alpha_ancla": {
                    "depth": 12,
                    "weight": 1.2,
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
                    reg_alpha=0.1,
                    reg_lambda=1.5,
                    # Forzamos CPU para compatibilidad con MacBook Air
                    device="cpu",
                    tree_method="auto",
                    n_jobs=-1,  # Usa todos los núcleos (M1/M2/M3)
                    random_state=42,
                ),
                "weight": params.get("weight", 0.5),
            }

    def train(self, winning_numbers, total_balls):
        """
        Entrenamiento del ensamble usando muestras reales y ruido sintético.
        """
        if not winning_numbers:
            return

        # Preparación de muestras positivas
        X_pos = np.array([sorted(draw[:6]) for draw in winning_numbers])
        y_pos = np.ones(len(X_pos), dtype=np.float32)

        # Generar set negativo (ruido)
        raw_noise = np.random.randint(1, total_balls + 1, (len(X_pos) * 2, 6))
        X_neg = np.sort(raw_noise, axis=1)
        y_neg = np.zeros(len(X_neg), dtype=np.float32)

        X_train = np.vstack((X_pos, X_neg))
        y_train = np.concatenate((y_pos, y_neg))

        # Ajuste del escalador (vital para la sensibilidad)
        X_scaled = self.scaler.fit_transform(X_train)

        for name, expert in self.experts.items():
            expert["model"].fit(X_scaled, y_train)

        self.is_trained = True

    def score_tickets(self, candidates, return_breakdown=False):
        """
        Evalúa candidatos asignando un puntaje de confianza IA.
        """
        if not self.is_trained or candidates is None or len(candidates) == 0:
            return np.array([], dtype=np.float32)

        # Sincronización para procesamiento en CPU
        cand_cpu = candidates.get() if hasattr(candidates, "get") else candidates
        X_scaled = self.scaler.transform(cand_cpu)

        # Cálculo de scores por experto
        scores_pool = []
        for name, expert in self.experts.items():
            pred = expert["model"].predict(X_scaled)
            scores_pool.append(np.maximum(pred, 1e-6) * expert["weight"])

        return np.sum(scores_pool, axis=0)