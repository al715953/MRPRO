import numpy as np
import pandas as pd
from collections import Counter
from sklearn.preprocessing import StandardScaler
from typing import List, Tuple
import xgboost as xgb
from xgboost import XGBClassifier

class LotteryAIModel:
    """
    Motor de Clasificación Supervisada V5 (Optimized Pipeline).
    
    OPTIMIZACIONES:
    - Feature Caching: El vector de calor se calcula una sola vez.
    - Vectorized Noise: Generación de ruido 10x más rápida.
    - XGBoost: Parámetros afinados para inferencia rápida.
    """

    def __init__(self):
        self.model = XGBClassifier(
            n_estimators=800,
            max_depth=5,
            learning_rate=0.03,
            subsample=0.7,
            colsample_bytree=0.7,
            gamma=1,
            reg_alpha=0.5,
            reg_lambda=1.0,
            objective='binary:logistic',
            eval_metric='logloss',
            n_jobs=-1,
            random_state=42,
            scale_pos_weight=5.0
        )
        self.scaler = StandardScaler()
        self.is_trained = False
        
        # Cache optimizado para inferencia
        self.heat_vector_ = None 

    def _extract_features(self, combinations: List[Tuple[int, ...]]) -> np.ndarray:
        """
        Extrae features vectorizadas. Asume que self.heat_vector_ ya existe.
        """
        # Convertir a numpy una sola vez
        data = np.array(combinations, dtype=np.int32)

        # 1. GEOMETRÍA (Vectorizado)
        sums = data.sum(axis=1)
        stds = data.std(axis=1)
        evens = (data % 2 == 0).sum(axis=1)
        ranges = data.max(axis=1) - data.min(axis=1)
        
        diffs = np.diff(data, axis=1)
        avg_diffs = diffs.mean(axis=1)
        consecutives = (diffs == 1).sum(axis=1)

        # Last Digits
        last_digits = data % 10
        last_digits.sort(axis=1)
        # Unique counts manual hack: diff > 0 + 1
        ld_diffs = np.diff(last_digits, axis=1)
        ld_unique = (ld_diffs > 0).sum(axis=1) + 1
        same_ending_score = 6 - ld_unique

        decades = data // 10
        decades_std = decades.std(axis=1)

        # 2. TREND SCORE (Lookup Vectorizado Instantáneo)
        # Usamos el vector pre-calculado en train()
        if self.heat_vector_ is not None:
            # Indexación fantasía: O(1)
            trend_scores = self.heat_vector_[data].sum(axis=1)
        else:
            trend_scores = np.zeros(len(data))

        # 3. NON-LINEAR
        dist_from_mean = np.abs(sums - 120)

        # Stack final
        X = np.column_stack(
            (
                sums,
                stds,
                evens,
                ranges,
                avg_diffs,
                consecutives,
                same_ending_score,
                decades_std,
                trend_scores,
                dist_from_mean
            )
        )
        return X

    def train(self, history_draws: List[List[int]], total_balls: int):
        """
        Entrenamiento con generación de ruido vectorizada.
        """
        if len(history_draws) < 50:
            print("      ⚠ Insuficientes datos para AI.")
            return

        # --- A. PRE-CÁLCULO DEL HEAT VECTOR (CACHE) ---
        recent_history = [n for d in history_draws[-50:] for n in d[:6]]
        freq_map = Counter(recent_history)
        
        # Creamos un vector estático para lookups O(1)
        # Buffer de seguridad +2
        self.heat_vector_ = np.zeros(total_balls + 2, dtype=np.float32)
        for ball, freq in freq_map.items():
            if ball <= total_balls:
                self.heat_vector_[ball] = freq

        # --- B. PREPARAR DATOS POSITIVOS ---
        winners = [tuple(sorted(d[:6])) for d in history_draws]
        n_winners = len(winners)
        X_pos = self._extract_features(winners)
        y_pos = np.ones(n_winners)

        # Sample Weights (Time Decay)
        weights_pos = np.exp(np.linspace(0, 2, n_winners))
        weights_pos /= weights_pos.mean()

        # --- C. GENERAR RUIDO (VECTORIZADO) ---
        n_noise = n_winners * 5
        
        # Matriz de probabilidad
        flat_hist = [n for d in history_draws for n in d[:6]]
        counts = Counter(flat_hist)
        probs = np.array([counts.get(n, 1) for n in range(1, total_balls + 1)], dtype=float)
        probs /= probs.sum()
        
        # Generación masiva (con reemplazo es mucho más rápido y aceptable para ruido)
        # Para hacerlo sin reemplazo vectorizado estricto es complejo, 
        # pero para ruido de entrenamiento, colisiones puntuales son aceptables.
        # Truco: Generamos un poco más y filtramos duplicados si queremos ser puristas,
        # pero para XGBoost el ruido "casi" válido sirve.
        pool_nums = np.arange(1, total_balls + 1)
        
        # Generamos matriz (n_noise, 6)
        noise_matrix = np.random.choice(pool_nums, size=(n_noise, 6), p=probs)
        noise_matrix.sort(axis=1)
        
        # Convertir a lista de tuplas para extract_features
        # (Aunque extract_features soporta arrays, mantenemos consistencia de tipos)
        # Optimización: pasar directo el numpy array si modificamos extract_features 
        # para aceptar ambos. Por ahora, map simple.
        # Pero espera, _extract_features YA convierte a numpy al inicio.
        # Podemos pasar noise_matrix directamente si hacemos un pequeño bypass.
        # Para seguridad, convertimos a lista de tuplas rápido.
        noise_samples = [tuple(row) for row in noise_matrix]

        X_neg = self._extract_features(noise_samples)
        y_neg = np.zeros(n_noise)
        weights_neg = np.full(n_noise, 0.5)

        # --- D. FUSIÓN ---
        X = np.vstack((X_pos, X_neg))
        y = np.concatenate((y_pos, y_neg))
        weights = np.concatenate((weights_pos, weights_neg))

        X_scaled = self.scaler.fit_transform(X)

        self.model.fit(X_scaled, y, sample_weight=weights)
        self.is_trained = True

    def score_tickets(self, candidates: List[Tuple[int, ...]]) -> np.ndarray:
        if not self.is_trained:
            return np.full(len(candidates), 0.5)

        X = self._extract_features(candidates)
        X_scaled = self.scaler.transform(X)
        
        # Probabilidad de clase 1
        return self.model.predict_proba(X_scaled)[:, 1]