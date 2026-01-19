import numpy as np
import pandas as pd
from collections import Counter
from sklearn.preprocessing import StandardScaler
from typing import List, Tuple, Optional, Any, Dict
from xgboost import XGBClassifier
import warnings

# Bloqueo de advertencias para una telemetría limpia en consola
warnings.filterwarnings("ignore", category=UserWarning)

# Detección de aceleración por hardware (NVIDIA CUDA)
try:
    import cupy as cp

    HAS_GPU = True
except ImportError:
    HAS_GPU = False


class LotteryAIModel:
    """
    Motor V6.9.3: Arquitectura de Consistencia Operativa.
    Calibrado para maximizar la frecuencia de aciertos 5/6 y 4/6.
    """

    def __init__(self):
        # Configuración agresiva para forzar la elevación de candidatos en el ranking
        gpu_params = (
            {"device": "cuda", "tree_method": "hist", "predictor": "gpu_predictor"}
            if HAS_GPU
            else {"tree_method": "auto", "predictor": "cpu_predictor"}
        )

        self.model = XGBClassifier(
            n_estimators=2000,  # Resolución profunda para patrones de 4/6 y 5/6
            max_depth=9,  # Captura de interacciones complejas de décadas
            learning_rate=0.01,  # Aprendizaje estable para evitar el sobreajuste
            subsample=0.85,
            colsample_bytree=0.85,
            gamma=3.0,
            reg_alpha=1.0,
            reg_lambda=2.0,
            objective="binary:logistic",
            eval_metric="logloss",
            n_jobs=-1,
            scale_pos_weight=7.5,  # Aumentado para forzar la consistencia en el Top
            **gpu_params
        )
        self.scaler = StandardScaler()
        self.is_trained = False
        self.heat_vector_ = None
        self.recency_vector_ = None

    def _extract_features(self, combinations: Any) -> np.ndarray:
        """Ingeniería de Características: Topología y Resonancia Aritmética."""
        data = np.array(combinations, dtype=np.int32)
        if data.ndim == 1:
            data = data.reshape(1, -1)

        # 1. Análisis Aritmético
        sums = data.sum(axis=1)
        stds = data.std(axis=1)
        evens = (data % 2 == 0).sum(axis=1)
        ranges = data.max(axis=1) - data.min(axis=1)

        # 2. Espaciado (Deltas)
        diffs = np.diff(data, axis=1)
        avg_diff = diffs.mean(axis=1)
        consecutive_count = (diffs == 1).sum(axis=1)
        max_gap = diffs.max(axis=1)

        # 3. Estructura de Décadas (Melate Retro 6/39)
        d1 = ((data >= 1) & (data <= 9)).sum(axis=1)
        d2 = ((data >= 10) & (data <= 19)).sum(axis=1)
        d3 = ((data >= 20) & (data <= 29)).sum(axis=1)
        d4 = ((data >= 30) & (data <= 39)).sum(axis=1)

        # 4. Telemetría de Frecuencia
        t_global = np.zeros(len(data))
        t_recency = np.zeros(len(data))
        if self.heat_vector_ is not None:
            for i in range(len(data)):
                t_global[i] = self.heat_vector_[data[i]].sum()
                t_recency[i] = self.recency_vector_[data[i]].sum()

        recency_impact = (t_recency * 0.50) - (t_global / 5.0)

        # Consolidación con Factor de Resonancia (Brújula de Éxito)
        X = np.column_stack(
            (
                sums,
                stds,
                evens,
                ranges,
                avg_diff,
                consecutive_count,
                max_gap,
                d1,
                d2,
                d3,
                d4,
                t_global,
                t_recency,
                recency_impact,
                (sums / 120.0) * (stds / 11.0),  # Factor de Resonancia
            )
        )
        return X

    def train(
        self,
        history_draws: List[List[int]],
        total_balls: int,
        feedback_loop: Optional[List[Dict]] = None,
    ):
        """Entrenamiento con Refuerzo Prioritario para Aciertos 4/6 y 5/6."""
        if len(history_draws) < 50:
            return

        # Calibración de Espectro
        long_h = [n for d in history_draws[-65:] for n in d[:6]]
        self.heat_vector_ = np.zeros(total_balls + 2, dtype=np.float32)
        for b, f in Counter(long_h).items():
            if b <= total_balls:
                self.heat_vector_[b] = f / 65.0

        short_h = [n for d in history_draws[-15:] for n in d[:6]]
        self.recency_vector_ = np.zeros(total_balls + 2, dtype=np.float32)
        for b, f in Counter(short_h).items():
            if b <= total_balls:
                self.recency_vector_[b] = f / 15.0

        # Dataset de Entrenamiento (Clase Positiva)
        winners = [tuple(sorted(d[:6])) for d in history_draws]
        X_pos = self._extract_features(winners)
        y_pos = np.ones(len(winners))
        w_pos = np.exp(np.linspace(0, 3.0, len(winners)))

        # REFUERZO DE CONSISTENCIA (Sugerencia 3 Recalibrada)
        if feedback_loop:
            for entry in feedback_loop:
                hits = entry.get("hits", 0)
                dist = entry.get("proximity", 999)
                draw_idx = entry.get("internal_idx")

                if draw_idx is not None and draw_idx < len(w_pos):
                    # PRIORIDAD: Premiamos los aciertos de 5/6 y 4/6
                    if hits == 5:
                        w_pos[draw_idx] *= 4.0  # Máximo refuerzo para 5 aciertos
                    elif hits == 4:
                        w_pos[draw_idx] *= 2.5  # Refuerzo alto para 4 aciertos

                    # Refuerzo por Distancia (Cerrando el gap)
                    if 0 < dist < 50:
                        w_pos[draw_idx] *= 1.5

        # Generación de Hard Negatives (Ruido Estructural)
        n_neg = len(winners) * 10
        probs_freq = np.maximum(self.heat_vector_[1 : total_balls + 1], 0.1)
        probs_freq /= probs_freq.sum()
        raw_noise = np.sort(
            np.random.choice(
                np.arange(1, total_balls + 1), (n_neg * 5, 6), p=probs_freq
            ),
            axis=1,
        )

        noise_sums = raw_noise.sum(axis=1)
        hard_mask = (noise_sums >= 100) & (noise_sums <= 145)
        hard_noise = raw_noise[hard_mask][:n_neg]
        if len(hard_noise) < n_neg:
            hard_noise = raw_noise[:n_neg]

        X_neg = self._extract_features(hard_noise)
        y_neg = np.zeros(len(X_neg))
        w_neg = np.full(len(y_neg), 0.6)

        # Consolidación y Entrenamiento
        X_final = np.vstack((X_pos, X_neg))
        y_final = np.concatenate((y_pos, y_neg))
        w_final = np.concatenate((w_pos / w_pos.mean(), w_neg))

        X_scaled = self.scaler.fit_transform(X_final)
        self.model.fit(X_scaled, y_final, sample_weight=w_final)
        self.is_trained = True

    def score_tickets(self, candidates: List[Tuple[int, ...]]) -> np.ndarray:
        """Inferencia de Probabilidad Cuántica."""
        if not self.is_trained or not candidates:
            return np.full(len(candidates), 0.5)

        X_feats = self._extract_features(candidates)
        X_scaled = self.scaler.transform(X_feats)
        return self.model.predict_proba(X_scaled)[:, 1]
