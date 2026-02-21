from __future__ import annotations

from typing import List, Sequence, Tuple

import numpy as np


def _safe_digit(value) -> int:
    try:
        return int(float(value))
    except Exception:
        return 0


def _safe_multiplier(value) -> int:
    if value is None:
        return 0
    if isinstance(value, str):
        token = value.strip().upper()
        if token in ("SI", "SÍ", "YES", "Y", "TRUE"):
            return 1
        if token in ("NO", "N", "FALSE"):
            return 0
    try:
        return int(float(value))
    except Exception:
        return 0


def _extract_tris_series(history) -> Tuple[List[List[int]], List[int]]:
    """
    Devuelve series cronologicas (por concurso ascendente) para Tris.
    - digits_list: [ [d1..d5], ... ]
    - mult_list: [0/1/...]
    """
    triples = sorted(
        zip(history.concursos, history.winning_numbers),
        key=lambda x: x[0],
    )
    digits_list: List[List[int]] = []
    mult_list: List[int] = []

    for _, draw in triples:
        if not draw or len(draw) < 5:
            continue
        digits = [_safe_digit(draw[i]) for i in range(5)]
        digits = [d if 0 <= d <= 9 else (d % 10) for d in digits]
        mult = _safe_multiplier(draw[5]) if len(draw) > 5 else 0
        digits_list.append(digits)
        mult_list.append(mult)

    return digits_list, mult_list


class BayesMixtureModel:
    def __init__(
        self,
        alpha: float = 0.5,
        short_window: int = 200,
        long_window: int = 2000,
        mix_lambda: float = 0.7,
    ):
        self.alpha = float(alpha)
        self.short_window = int(short_window)
        self.long_window = int(long_window)
        self.mix_lambda = float(mix_lambda)
        self.short_counts = np.zeros((5, 10), dtype=np.float64)
        self.long_counts = np.zeros((5, 10), dtype=np.float64)

    @staticmethod
    def _count_positions(rows: Sequence[Sequence[int]]) -> np.ndarray:
        counts = np.zeros((5, 10), dtype=np.float64)
        for row in rows:
            if len(row) < 5:
                continue
            for pos in range(5):
                d = int(row[pos])
                if 0 <= d <= 9:
                    counts[pos, d] += 1.0
        return counts

    def fit(self, digits_list: Sequence[Sequence[int]]) -> None:
        if not digits_list:
            self.short_counts.fill(0.0)
            self.long_counts.fill(0.0)
            return

        short_rows = digits_list[-min(len(digits_list), self.short_window) :]
        long_rows = digits_list[-min(len(digits_list), self.long_window) :]
        self.short_counts = self._count_positions(short_rows)
        self.long_counts = self._count_positions(long_rows)

    def _smoothed_probs(self, counts: np.ndarray) -> np.ndarray:
        numer = counts + self.alpha
        denom = numer.sum(axis=1, keepdims=True)
        return numer / np.clip(denom, 1e-12, None)

    def predict_pos_probs(self, context_last_digits: Sequence[int]) -> np.ndarray:
        short_probs = self._smoothed_probs(self.short_counts)
        long_probs = self._smoothed_probs(self.long_counts)
        probs = self.mix_lambda * short_probs + (1.0 - self.mix_lambda) * long_probs
        probs /= np.clip(probs.sum(axis=1, keepdims=True), 1e-12, None)
        return probs


class MarkovPositionalModel:
    def __init__(self, alpha: float = 0.2, window: int = 2000):
        self.alpha = float(alpha)
        self.window = int(window)
        self.trans_counts = np.zeros((5, 10, 10), dtype=np.float64)

    def fit(self, digits_list: Sequence[Sequence[int]]) -> None:
        self.trans_counts.fill(0.0)
        if len(digits_list) < 2:
            return

        rows = digits_list[-min(len(digits_list), self.window) :]
        for i in range(1, len(rows)):
            prev_row = rows[i - 1]
            curr_row = rows[i]
            if len(prev_row) < 5 or len(curr_row) < 5:
                continue
            for pos in range(5):
                d_prev = int(prev_row[pos])
                d_curr = int(curr_row[pos])
                if 0 <= d_prev <= 9 and 0 <= d_curr <= 9:
                    self.trans_counts[pos, d_prev, d_curr] += 1.0

    def predict_pos_probs(self, context_last_digits: Sequence[int]) -> np.ndarray:
        probs = np.zeros((5, 10), dtype=np.float64)
        for pos in range(5):
            if pos < len(context_last_digits):
                prev_d = int(context_last_digits[pos])
            else:
                prev_d = 0
            prev_d = prev_d if 0 <= prev_d <= 9 else (prev_d % 10)
            row = self.trans_counts[pos, prev_d, :]
            numer = row + self.alpha
            denom = numer.sum()
            probs[pos] = numer / max(denom, 1e-12)
        return probs


class TrisV1AModel:
    def __init__(
        self,
        blend_markov: float = 0.35,
        uniform_mix: float = 0.0,
        uniform_floor_mu: float = 0.35,
        peak_max_prob: float = 0.22,
        peak_mu_boost: float = 0.20,
        temperature: float = 1.4,
        bayes_params: dict | None = None,
        markov_params: dict | None = None,
    ):
        self.blend_markov = float(blend_markov)
        self.uniform_mix = float(uniform_mix)
        self.uniform_floor_mu = float(uniform_floor_mu)
        self.peak_max_prob = float(peak_max_prob)
        self.peak_mu_boost = float(peak_mu_boost)
        self.temperature = float(temperature)
        self.bayes = BayesMixtureModel(**(bayes_params or {}))
        self.markov = MarkovPositionalModel(**(markov_params or {}))
        self.p_multiplier_short = 0.5
        self.p_multiplier_long = 0.5
        self.p_multiplier = 0.5

    @staticmethod
    def _beta_rate(values: Sequence[int]) -> float:
        n = len(values)
        if n == 0:
            return 0.5
        positives = float(sum(1 for v in values if v))
        return (positives + 1.0) / (n + 2.0)

    def fit(self, digits_list: Sequence[Sequence[int]], mult_list: Sequence[int]) -> None:
        self.bayes.fit(digits_list)
        self.markov.fit(digits_list)

        short_n = min(len(mult_list), self.bayes.short_window)
        long_n = min(len(mult_list), self.bayes.long_window)
        short_vals = mult_list[-short_n:] if short_n > 0 else []
        long_vals = mult_list[-long_n:] if long_n > 0 else []

        self.p_multiplier_short = self._beta_rate(short_vals)
        self.p_multiplier_long = self._beta_rate(long_vals)
        lam = self.bayes.mix_lambda
        self.p_multiplier = lam * self.p_multiplier_short + (1.0 - lam) * self.p_multiplier_long

    def predict(
        self, context_last_digits: Sequence[int]
    ) -> Tuple[np.ndarray, float, np.ndarray, float, dict]:
        bayes_probs = self.bayes.predict_pos_probs(context_last_digits)
        markov_probs = self.markov.predict_pos_probs(context_last_digits)

        w = self.blend_markov
        pos_probs = (1.0 - w) * bayes_probs + w * markov_probs
        pos_probs = np.clip(pos_probs, 1e-12, None)
        pos_probs /= np.clip(pos_probs.sum(axis=1, keepdims=True), 1e-12, None)

        max_probs = np.max(pos_probs, axis=1)
        mu = min(max(self.uniform_floor_mu, 0.0), 0.8)
        if np.any(max_probs > float(self.peak_max_prob)):
            mu = min(0.8, mu + max(float(self.peak_mu_boost), 0.0))

        uniform = np.full((5, 10), 0.1, dtype=np.float64)
        pos_probs = (1.0 - mu) * pos_probs + mu * uniform
        pos_probs /= np.clip(pos_probs.sum(axis=1, keepdims=True), 1e-12, None)

        temp = max(self.temperature, 1e-6)
        logits = np.log(np.maximum(pos_probs, 1e-12))
        logits = logits / temp
        logits = logits - np.max(logits, axis=1, keepdims=True)
        exp_logits = np.exp(logits)
        pos_probs = exp_logits / np.clip(
            exp_logits.sum(axis=1, keepdims=True), 1e-12, None
        )

        pos_probs = np.clip(pos_probs, 1e-12, None)
        pos_probs /= np.clip(pos_probs.sum(axis=1, keepdims=True), 1e-12, None)

        entropy_pos = -np.sum(pos_probs * np.log(pos_probs), axis=1)
        entropy_mean = float(np.mean(entropy_pos))
        guardrail_meta = {
            "mu_used": float(mu),
            "max_probs": max_probs.tolist(),
        }
        return pos_probs, float(self.p_multiplier), entropy_pos, entropy_mean, guardrail_meta
