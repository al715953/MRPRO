import numpy as np
import pandas as pd
from collections import Counter
from sklearn.preprocessing import StandardScaler
from typing import List, Tuple
from xgboost import XGBClassifier
import warnings

warnings.filterwarnings("ignore", category=UserWarning)


class LotteryAIModel:
    """Motor V6.7.1: Balance Espectral con Blindaje de Precisión."""

    def __init__(self):
        self.model = XGBClassifier(
            n_estimators=1000,
            max_depth=6,
            learning_rate=0.02,
            subsample=0.8,
            colsample_bytree=0.8,
            gamma=1.5,
            reg_alpha=0.5,
            reg_lambda=1.2,
            objective="binary:logistic",
            eval_metric="logloss",
            n_jobs=-1,
            device="cuda",
            tree_method="hist",
            scale_pos_weight=4.0,
        )
        self.scaler = StandardScaler()
        self.is_trained = False
        self.heat_vector_ = None
        self.recency_vector_ = None

    def _extract_features(self, combinations: List[Tuple[int, ...]]) -> np.ndarray:
        data = np.array(combinations, dtype=np.int32)
        sums, stds = data.sum(axis=1), data.std(axis=1)
        evens, ranges = (data % 2 == 0).sum(axis=1), data.max(axis=1) - data.min(axis=1)
        diffs = np.diff(data, axis=1)

        t_global = (
            self.heat_vector_[data].sum(axis=1)
            if self.heat_vector_ is not None
            else np.zeros(len(data))
        )
        t_recency = (
            self.recency_vector_[data].sum(axis=1)
            if self.recency_vector_ is not None
            else np.zeros(len(data))
        )

        # Impacto Balanceado: 40% Reciente / 60% Global
        recency_impact = (t_recency * 0.4) - (t_global / 5.0)

        X = np.column_stack(
            (
                sums,
                stds,
                evens,
                ranges,
                diffs.mean(axis=1),
                (diffs == 1).sum(axis=1),
                t_global,
                t_recency,
                recency_impact,
                (sums / 120.0) * (stds / 10.0),
            )
        )
        return X

    def train(self, history_draws: List[List[int]], total_balls: int):
        if len(history_draws) < 50:
            return

        long_h = [n for d in history_draws[-50:] for n in d[:6]]
        self.heat_vector_ = np.zeros(total_balls + 2, dtype=np.float32)
        for b, f in Counter(long_h).items():
            self.heat_vector_[b] = f

        short_h = [n for d in history_draws[-10:] for n in d[:6]]
        self.recency_vector_ = np.zeros(total_balls + 2, dtype=np.float32)
        for b, f in Counter(short_h).items():
            self.recency_vector_[b] = f

        winners = [tuple(sorted(d[:6])) for d in history_draws]
        X_pos = self._extract_features(winners)
        y_pos, w_pos = np.ones(len(winners)), np.exp(np.linspace(0, 2.0, len(winners)))

        # INYECCIÓN DE RUIDO ESPECTRAL (Fix V6.7.1)
        n_noise = len(winners) * 6
        probs_freq = np.maximum(self.heat_vector_[1 : total_balls + 1], 0.1)
        probs_freq /= probs_freq.sum()
        probs_unif = np.full(total_balls, 1.0 / total_balls)

        # Mezcla 70/30 con re-normalización forzada
        final_probs = (probs_freq * 0.7) + (probs_unif * 0.3)
        final_probs /= final_probs.sum()

        noise = np.sort(
            np.random.choice(
                np.arange(1, total_balls + 1), (n_noise, 6), p=final_probs
            ),
            axis=1,
        )

        X_neg = self._extract_features([tuple(row) for row in noise])
        X = np.vstack((X_pos, X_neg))
        y = np.concatenate((y_pos, np.zeros(n_noise)))
        w = np.concatenate((w_pos / w_pos.mean(), np.full(n_noise, 0.4)))

        self.model.fit(self.scaler.fit_transform(X), y, sample_weight=w)
        self.is_trained = True

    def score_tickets(self, candidates: List[Tuple[int, ...]]) -> np.ndarray:
        if not self.is_trained:
            return np.full(len(candidates), 0.5)
        return self.model.predict_proba(
            self.scaler.transform(self._extract_features(candidates))
        )[:, 1]
