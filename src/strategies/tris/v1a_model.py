from __future__ import annotations

import re
from typing import Any, Dict, List, Sequence, Tuple

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


def _normalize_slot_token(slot_value: Any) -> str:
    if slot_value is None:
        return "unknown"
    token = str(slot_value).strip().lower()
    if not token:
        return "unknown"
    token = (
        token.replace("á", "a")
        .replace("é", "e")
        .replace("í", "i")
        .replace("ó", "o")
        .replace("ú", "u")
    )
    compact = token.replace(" ", "").replace("_", "").replace("-", "")
    if "mediodia" in compact or "midday" in compact:
        return "mediodia"
    if "clasico" in compact or "classic" in compact or "vespertino" in compact or "noche" in compact:
        return "clasico"
    if compact in {"md"}:
        return "mediodia"
    if compact in {"cl"}:
        return "clasico"
    return "unknown"


def _slot_from_date_value(raw_date: Any) -> str:
    if raw_date is None:
        return "unknown"
    hour = getattr(raw_date, "hour", None)
    if hour is not None:
        try:
            h = int(hour)
            return "mediodia" if h < 15 else "clasico"
        except Exception:
            pass

    token = str(raw_date).strip().lower()
    if not token:
        return "unknown"
    norm = _normalize_slot_token(token)
    if norm != "unknown":
        return norm

    match = re.search(r"(\d{1,2}):(\d{2})", token)
    if match is None:
        return "unknown"
    try:
        h = int(match.group(1))
    except Exception:
        return "unknown"
    return "mediodia" if h < 15 else "clasico"


def _extract_tris_series_with_context(
    history,
) -> tuple[List[List[int]], List[int], List[Dict[str, Any]]]:
    """
    Devuelve series cronologicas (por concurso ascendente) para Tris con contexto por sorteo.
    - digits_list: [ [d1..d5], ... ]
    - mult_list: [0/1/...]
    - ctx_list: [{concurso,date,slot}, ...]
    """
    concursos = list(getattr(history, "concursos", []) or [])
    dates = list(getattr(history, "dates", []) or [])
    draws = list(getattr(history, "winning_numbers", []) or [])
    slots = list(getattr(history, "slots", []) or [])

    triples = []
    for idx, draw in enumerate(draws):
        if idx < len(concursos):
            concurso = concursos[idx]
        else:
            concurso = idx
        raw_date = dates[idx] if idx < len(dates) else None
        raw_slot = slots[idx] if idx < len(slots) else None
        if raw_slot is None and isinstance(draw, (list, tuple)) and len(draw) > 6:
            raw_slot = draw[6]
        triples.append((concurso, idx, draw, raw_date, raw_slot))

    triples.sort(key=lambda x: (x[0], x[1]))

    digits_list: List[List[int]] = []
    mult_list: List[int] = []
    ctx_list: List[Dict[str, Any]] = []

    for concurso, _, draw, raw_date, raw_slot in triples:
        if not draw or len(draw) < 5:
            continue
        digits = [_safe_digit(draw[i]) for i in range(5)]
        digits = [d if 0 <= d <= 9 else (d % 10) for d in digits]
        mult = _safe_multiplier(draw[5]) if len(draw) > 5 else 0

        slot = _normalize_slot_token(raw_slot)
        if slot == "unknown":
            slot = _slot_from_date_value(raw_date)

        digits_list.append(digits)
        mult_list.append(mult)
        ctx_list.append(
            {
                "concurso": concurso,
                "date": raw_date,
                "slot": slot,
            }
        )

    return digits_list, mult_list, ctx_list


def _extract_tris_series(history) -> Tuple[List[List[int]], List[int]]:
    """
    Devuelve series cronologicas (por concurso ascendente) para Tris.
    - digits_list: [ [d1..d5], ... ]
    - mult_list: [0/1/...]
    """
    digits_list, mult_list, _ = _extract_tris_series_with_context(history)
    return digits_list, mult_list


def analyze_slot_drift(
    digits_list,
    ctx_list,
    slot_a: str = "mediodia",
    slot_b: str = "clasico",
) -> dict:
    """
    Diagnostico de deriva por slot temporal para Tris (sin afectar scoring).
    """
    rows = list(digits_list or [])
    contexts = list(ctx_list or [])
    n = min(len(rows), len(contexts))
    slot_a_norm = _normalize_slot_token(slot_a)
    slot_b_norm = _normalize_slot_token(slot_b)

    counts_a = np.zeros((5, 10), dtype=np.float64)
    counts_b = np.zeros((5, 10), dtype=np.float64)
    n_a = 0
    n_b = 0

    for i in range(n):
        row = rows[i]
        if row is None or len(row) < 5:
            continue
        ctx = contexts[i] if isinstance(contexts[i], dict) else {}
        slot = _normalize_slot_token(ctx.get("slot", "unknown"))
        if slot == slot_a_norm:
            for pos in range(5):
                counts_a[pos, int(row[pos]) % 10] += 1.0
            n_a += 1
        elif slot == slot_b_norm:
            for pos in range(5):
                counts_b[pos, int(row[pos]) % 10] += 1.0
            n_b += 1

    alpha = 1.0
    probs_a = (counts_a + alpha) / np.clip(
        np.sum(counts_a + alpha, axis=1, keepdims=True), 1e-12, None
    )
    probs_b = (counts_b + alpha) / np.clip(
        np.sum(counts_b + alpha, axis=1, keepdims=True), 1e-12, None
    )

    l1_by_pos = np.sum(np.abs(probs_a - probs_b), axis=1)
    chi2_like_by_pos = np.sum(
        ((probs_a - probs_b) ** 2) / np.clip(probs_a + probs_b, 1e-12, None),
        axis=1,
    )

    return {
        "slot_a": slot_a_norm,
        "slot_b": slot_b_norm,
        "sample_size_a": int(n_a),
        "sample_size_b": int(n_b),
        "counts_a": counts_a.astype(np.int64).tolist(),
        "counts_b": counts_b.astype(np.int64).tolist(),
        "l1_by_pos": l1_by_pos.astype(np.float64).tolist(),
        "l1_mean": float(np.mean(l1_by_pos)) if l1_by_pos.size > 0 else 0.0,
        "chi2_like_by_pos": chi2_like_by_pos.astype(np.float64).tolist(),
        "chi2_like_mean": float(np.mean(chi2_like_by_pos))
        if chi2_like_by_pos.size > 0
        else 0.0,
    }


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

    def apply_positional_logit_bias(
        self, pos_probs: np.ndarray, bias_logits: np.ndarray | None
    ) -> np.ndarray:
        probs = np.asarray(pos_probs, dtype=np.float64)
        if probs.shape != (5, 10):
            raise ValueError("pos_probs debe tener shape (5,10).")
        probs = np.clip(probs, 1e-12, None)
        probs = probs / np.clip(np.sum(probs, axis=1, keepdims=True), 1e-12, None)

        if bias_logits is None:
            return probs

        bias = np.asarray(bias_logits, dtype=np.float64)
        if bias.shape != (5, 10):
            raise ValueError("bias_logits debe tener shape (5,10).")

        logits = np.log(probs) + bias
        logits = logits - np.max(logits, axis=1, keepdims=True)
        exp_logits = np.exp(logits)
        out = exp_logits / np.clip(exp_logits.sum(axis=1, keepdims=True), 1e-12, None)
        return out

    def predict(
        self, context_last_digits: Sequence[int], bias_logits: np.ndarray | None = None
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
        pos_probs = self.apply_positional_logit_bias(pos_probs, bias_logits)

        entropy_pos = -np.sum(pos_probs * np.log(pos_probs), axis=1)
        entropy_mean = float(np.mean(entropy_pos))
        guardrail_meta = {
            "mu_used": float(mu),
            "max_probs": max_probs.tolist(),
        }
        return pos_probs, float(self.p_multiplier), entropy_pos, entropy_mean, guardrail_meta
