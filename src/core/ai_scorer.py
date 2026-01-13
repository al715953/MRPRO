import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from typing import List, Tuple


class LotteryAIModel:
    """
    Motor de Clasificación Supervisada.
    Aprende a distinguir 'Estructuras Ganadoras' de 'Ruido Aleatorio'.
    """

    def __init__(self):
        # RandomForest es robusto a outliers y no requiere escalado perfecto
        self.model = RandomForestClassifier(
            n_estimators=100,
            max_depth=10,
            random_state=42,
            n_jobs=-1,  # Usar todos los cores
        )
        self.scaler = StandardScaler()
        self.is_trained = False

    def _extract_features(self, combinations: List[Tuple[int, ...]]) -> np.ndarray:
        """
        Convierte una lista de tickets en una matriz de características numéricas.
        Features: [Suma, DesviaciónStd, Pares, Primos, AC_Value, Rango(Max-Min)]
        """
        # Vectorización manual para velocidad
        data = np.array(combinations)

        # 1. Suma
        sums = data.sum(axis=1)

        # 2. Desviación Estándar (Dispersión)
        stds = data.std(axis=1)

        # 3. Pares
        evens = (data % 2 == 0).sum(axis=1)

        # 4. Rango
        ranges = data.max(axis=1) - data.min(axis=1)

        # 5. Distancias promedio (Compactación)
        diffs = np.diff(data, axis=1).mean(axis=1)

        # Concatenamos todo en una matriz X
        X = np.column_stack((sums, stds, evens, ranges, diffs))
        return X

    def train(self, history_draws: List[List[int]], total_balls: int):
        """
        Entrena el modelo contrastando Histórico (Clase 1) vs Ruido (Clase 0).
        """
        print("      🧠 Entrenando IA (RandomForest) con patrones históricos...")

        # CLASE 1: Ganadores Reales
        winners = [tuple(sorted(d[:6])) for d in history_draws]
        X_pos = self._extract_features(winners)
        y_pos = np.ones(len(winners))

        # CLASE 0: Ruido Aleatorio (Generamos 3x la cantidad de ganadores)
        n_noise = len(winners) * 3
        noise = []
        pool = range(1, total_balls + 1)
        for _ in range(n_noise):
            ticket = sorted(np.random.choice(pool, 6, replace=False))
            noise.append(tuple(ticket))

        X_neg = self._extract_features(noise)
        y_neg = np.zeros(n_noise)

        # Juntar y entrenar
        X = np.vstack((X_pos, X_neg))
        y = np.concatenate((y_pos, y_neg))

        self.model.fit(X, y)
        self.is_trained = True

        # Feature Importance (Diagnóstico)
        importances = self.model.feature_importances_
        # print(f"      📊 Importancia: Sum={importances[0]:.2f}, Std={importances[1]:.2f}, Pares={importances[2]:.2f}")

    def score_tickets(self, candidates: List[Tuple[int, ...]]) -> np.ndarray:
        """
        Devuelve la probabilidad (0.0 - 1.0) de que cada ticket se parezca a un ganador.
        """
        if not self.is_trained:
            raise Exception("Modelo no entrenado.")

        X = self._extract_features(candidates)
        # Probabilidad de la clase 1 (Ganador)
        probs = self.model.predict_proba(X)[:, 1]
        return probs
