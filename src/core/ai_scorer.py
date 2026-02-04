# src/core/ai_scorer.py
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from xgboost import XGBRegressor
# Importamos la configuración de hardware centralizada
from src.data_access.config import BEST_SETTINGS, GPU_ENABLED

# La señal de GPU ahora es gobernada exclusivamente por config.py
HAS_GPU = GPU_ENABLED 

class ShadowModel:
    """
    V7.17 Shadow-Model: Aprende patrones de 'Falsos Positivos' para evitar 
    tickets que parecen ganadores pero históricamente fallan.
    """
    def __init__(self):
        self.model = XGBRegressor(
            n_estimators=1000, 
            max_depth=6, 
            # Adaptación dinámica de hardware
            device="cuda" if HAS_GPU else "cpu",
            tree_method="hist" if HAS_GPU else "auto",
            n_jobs=-1,
            random_state=42
        )
        self.is_trained = False

    def train_from_forensics(self, forensic_history):
        """Entrena el modelo usando los datos de detailed_forensic_log.csv."""
        if len(forensic_history) < 20: 
            return 
        
        df = pd.DataFrame(forensic_history)
        X = df[['ai_score', 'geo_score', 'proximity']].values
        y = (df['hits'] < 3).astype(float) 
        
        self.model.fit(X, y)
        self.is_trained = True

    def get_risk_score(self, ai_score, geo_score, proximity):
        if not self.is_trained: 
            return 0.0
        X_input = np.array([[ai_score, geo_score, proximity]])
        return float(self.model.predict(X_input)[0])


class LotteryAIModel:
    """
    Motor V7.17: Neural-Precision & Deep-Ensemble.
    Arquitectura híbrida compatible con CUDA (Windows) y CPU (Mac).
    """

    def __init__(self):
        self.config = BEST_SETTINGS
        self.experts = {}
        self.scaler = StandardScaler()
        self.is_trained = False
        self._build_ensemble()

    def _build_ensemble(self):
        """Construye el ensamble basado en la disponibilidad de hardware."""
        fine_learning_rate = 0.008  
        deep_estimators = 4500  

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
                    # Configuración dinámica:
                    device="cuda" if HAS_GPU else "cpu",
                    tree_method="hist" if HAS_GPU else "auto",
                    n_jobs=-1,
                    random_state=42,
                ),
                "weight": params.get("weight", 0.5),
            }

    def train(self, winning_numbers, total_balls):
        if not winning_numbers:
            return

        X_pos = np.array([sorted(draw[:6]) for draw in winning_numbers])
        y_pos = np.ones(len(X_pos), dtype=np.float32)

        raw_noise = np.random.randint(1, total_balls + 1, (len(X_pos) * 2, 6))
        X_neg = np.sort(raw_noise, axis=1)
        y_neg = np.zeros(len(X_neg), dtype=np.float32)

        X_train = np.vstack((X_pos, X_neg))
        y_train = np.concatenate((y_pos, y_neg))

        X_scaled = self.scaler.fit_transform(X_train)

        for name, expert in self.experts.items():
            expert["model"].fit(X_scaled, y_train)

        self.is_trained = True

    def score_tickets(self, candidates, return_breakdown=False):
        if not self.is_trained or candidates is None or len(candidates) == 0:
            return np.array([], dtype=np.float32)

        # Sincronización automática de punteros (CPU/GPU)
        cand_cpu = candidates.get() if hasattr(candidates, "get") else candidates
        X_scaled = self.scaler.transform(cand_cpu)

        scores_pool = []
        for name, expert in self.experts.items():
            pred = expert["model"].predict(X_scaled)
            scores_pool.append(np.maximum(pred, 1e-6) * expert["weight"])

        return np.sum(scores_pool, axis=0)