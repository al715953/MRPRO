import numpy as np
import pandas as pd
from collections import Counter
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from typing import List, Tuple


class LotteryAIModel:
    """
    Motor de Clasificación Supervisada V3 (Temporal & Trend Aware).

    MEJORAS CRÍTICAS:
    1. Feature 'Trend Score': La IA ahora 've' si los números son calientes o fríos.
    2. Sample Weighting: Entrena dando más importancia a los sorteos recientes (Memoria a Corto Plazo).
    """

    def __init__(self):
        # Aumentamos un poco la profundidad para que aprenda las sutilezas de la tendencia
        self.model = RandomForestClassifier(
            n_estimators=200,
            max_depth=14,
            min_samples_split=4,
            random_state=42,
            n_jobs=-1,
            class_weight="balanced",
        )
        self.scaler = StandardScaler()
        self.is_trained = False
        self.freq_map = (
            {}
        )  # Memoria de qué estaba caliente en el momento del entrenamiento

    def _extract_features(self, combinations: List[Tuple[int, ...]]) -> np.ndarray:
        """
        Extrae huellas digitales matemáticas + TENDENCIA DE CALOR.
        """
        data = np.array(combinations)

        # --- 1. ESTADÍSTICAS BÁSICAS (GEOMETRÍA) ---
        sums = data.sum(axis=1)
        stds = data.std(axis=1)
        evens = (data % 2 == 0).sum(axis=1)
        ranges = data.max(axis=1) - data.min(axis=1)
        diffs_matrix = np.diff(data, axis=1)
        avg_diffs = diffs_matrix.mean(axis=1)

        # --- 2. ESTRUCTURA AVANZADA ---
        consecutives = (diffs_matrix == 1).sum(axis=1)
        last_digits = data % 10
        ld_sorted = np.sort(last_digits, axis=1)
        ld_unique_counts = (np.diff(ld_sorted, axis=1) > 0).sum(axis=1) + 1
        same_ending_score = 6 - ld_unique_counts
        decades_std = (data // 10).std(axis=1)

        # --- 3. NUEVO: TREND SCORE (La clave faltante) ---
        # Calculamos la "temperatura" total del ticket sumando la frecuencia de sus bolas
        # Vectorización: Usamos un array de lookup para velocidad
        # self.freq_map debe estar lleno. Si no (predicción sin entreno), usamos ceros.
        if self.freq_map:
            # Crear array de pesos donde el índice es la bola
            max_ball = max(max(self.freq_map.keys()), data.max())
            heat_lookup = np.zeros(max_ball + 1)
            for ball, freq in self.freq_map.items():
                heat_lookup[ball] = freq

            # Sumar calor de las 6 bolas
            # heat_lookup[data] crea una matriz (N, 6) con los calores
            trend_scores = heat_lookup[data].sum(axis=1)
        else:
            trend_scores = np.zeros(len(data))

        # --- CONCATENACIÓN FINAL ---
        # Ahora la IA tiene 9 sentidos en lugar de 8
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
                trend_scores,  # <--- NUEVO SENSOR
            )
        )

        return X

    def train(self, history_draws: List[List[int]], total_balls: int):
        """
        Entrena con ponderación temporal (Time Decay).
        Lo reciente vale más que lo antiguo.
        """
        if len(history_draws) < 20:
            print("      ⚠ Insuficientes datos para entrenar IA.")
            return

        # 1. MAPA DE CALOR (Contexto para las features)
        # Usamos los últimos 50 sorteos para definir qué es "Caliente" AHORA
        recent_history = [n for d in history_draws[-50:] for n in d[:6]]
        self.freq_map = Counter(recent_history)

        # 2. PREPARAR DATASET
        winners = [tuple(sorted(d[:6])) for d in history_draws]

        # --- SAMPLE WEIGHTS (DECAIMIENTO TEMPORAL) ---
        # Queremos que los últimos sorteos pesen 5 veces más que los primeros
        n_samples = len(winners)
        # Linspace genera una rampa de 0.2 a 1.0
        sample_weights_pos = np.linspace(0.2, 1.0, n_samples)

        # Extraer features (incluyendo el Trend Score basado en freq_map)
        X_pos = self._extract_features(winners)
        y_pos = np.ones(n_samples)

        # 3. GENERAR RUIDO (Smart Noise)
        # Usamos distribución ponderada para el ruido también
        counts_total = Counter([n for d in history_draws for n in d[:6]])
        weights = np.array(
            [counts_total.get(n, 1) for n in range(1, total_balls + 1)], dtype=float
        )
        weights /= weights.sum()

        n_noise = n_samples * 4
        noise_samples = []
        pool_nums = np.arange(1, total_balls + 1)

        for _ in range(n_noise):
            try:
                t = np.random.choice(pool_nums, size=6, replace=False, p=weights)
                noise_samples.append(tuple(sorted(t)))
            except:
                t = np.random.choice(pool_nums, size=6, replace=False)
                noise_samples.append(tuple(sorted(t)))

        X_neg = self._extract_features(noise_samples)
        y_neg = np.zeros(n_noise)
        # El ruido tiene peso estándar (0.5) para no eclipsar a los ganadores recientes
        sample_weights_neg = np.full(n_noise, 0.5)

        # 4. FUSIÓN
        X = np.vstack((X_pos, X_neg))
        y = np.concatenate((y_pos, y_neg))
        sample_weights = np.concatenate((sample_weights_pos, sample_weights_neg))

        # 5. ENTRENAMIENTO
        self.model.fit(X, y, sample_weight=sample_weights)
        self.is_trained = True

        # Diagnóstico de importancia (Opcional, para ver si usa el Trend Score)
        # print(f"Importancia Trend: {self.model.feature_importances_[-1]:.4f}")

    def score_tickets(self, candidates: List[Tuple[int, ...]]) -> np.ndarray:
        if not self.is_trained:
            return np.full(len(candidates), 0.5)

        X = self._extract_features(candidates)
        # Devolver probabilidad de Clase 1 (Ganador)
        return self.model.predict_proba(X)[:, 1]
