from __future__ import annotations

from typing import Sequence

import numpy as np
import xgboost as xgb

from src.core.melate_features import PRIMES, TOTAL_BALLS, build_context_state


NUMBER_FEATURE_SCHEMA = "melate_number_context_v1"
NUMBER_FEATURE_NAMES = (
    "number",
    "long_rate",
    "rate10",
    "rate20",
    "rate50",
    "rate100",
    "gap",
    "recent5_rate",
    "in_last_draw",
    "trend10_long",
    "trend20_long",
    "trend50_long",
    "last_pair_affinity_mean",
    "last_pair_affinity_max",
    "is_even",
    "is_prime",
    "decade_1",
    "decade_2",
    "decade_3",
    "decade_4",
)


def _window_rates(draws: np.ndarray, window: int) -> np.ndarray:
    selected = draws[-int(window) :]
    counts = np.bincount(
        selected.ravel(), minlength=TOTAL_BALLS + 1
    ).astype(np.float32)
    return counts / float(max(1, len(selected)))


def build_number_features(
    history: Sequence[Sequence[int]] | np.ndarray,
) -> np.ndarray:
    """Build one feature row for each number 1..39 from past draws only."""
    rows = [list(draw[:6]) for draw in history if len(draw) >= 6]
    draws = (
        np.sort(np.asarray(rows, dtype=np.int16), axis=1)
        if rows
        else np.empty((0, 6), dtype=np.int16)
    )
    state = build_context_state(draws)
    numbers = np.arange(1, TOTAL_BALLS + 1, dtype=np.int16)
    rate10 = _window_rates(draws, 10)[numbers]
    rate100 = _window_rates(draws, 100)[numbers]
    long_rate = state.long_rates[numbers]
    rate20 = state.rates20[numbers]
    rate50 = state.rates50[numbers]

    last_numbers = np.flatnonzero(state.last_draw_mask)
    if len(last_numbers):
        affinities = state.pair_rates[numbers[:, None], last_numbers[None, :]]
        pair_mean = np.mean(affinities, axis=1)
        pair_max = np.max(affinities, axis=1)
    else:
        pair_mean = np.zeros(TOTAL_BALLS, dtype=np.float32)
        pair_max = np.zeros(TOTAL_BALLS, dtype=np.float32)

    decades = (numbers - 1) // 10
    features = np.column_stack(
        (
            numbers.astype(np.float32) / float(TOTAL_BALLS),
            long_rate,
            rate10,
            rate20,
            rate50,
            rate100,
            state.gaps[numbers],
            state.recent5_rates[numbers],
            state.last_draw_mask[numbers],
            rate10 - long_rate,
            rate20 - long_rate,
            rate50 - long_rate,
            pair_mean,
            pair_max,
            (numbers % 2 == 0).astype(np.float32),
            np.isin(numbers, PRIMES).astype(np.float32),
            (decades == 0).astype(np.float32),
            (decades == 1).astype(np.float32),
            (decades == 2).astype(np.float32),
            (decades == 3).astype(np.float32),
        )
    ).astype(np.float32, copy=False)
    if features.shape != (TOTAL_BALLS, len(NUMBER_FEATURE_NAMES)):
        raise RuntimeError(f"Esquema por número inválido: {features.shape}")
    return features


def build_number_walk_forward_dataset(
    draws: np.ndarray,
    *,
    start_idx: int,
    end_idx: int,
) -> tuple[np.ndarray, np.ndarray]:
    ordered = np.sort(np.asarray(draws, dtype=np.uint8), axis=1)
    start = max(1, int(start_idx))
    end = min(int(end_idx), len(ordered))
    feature_parts = []
    label_parts = []
    for target_idx in range(start, end):
        features = build_number_features(ordered[:target_idx])
        labels = np.zeros(TOTAL_BALLS, dtype=np.float32)
        labels[ordered[target_idx].astype(np.int16) - 1] = 1.0
        feature_parts.append(features)
        label_parts.append(labels)
    if not feature_parts:
        return (
            np.empty((0, len(NUMBER_FEATURE_NAMES)), dtype=np.float32),
            np.empty(0, dtype=np.float32),
        )
    return np.vstack(feature_parts), np.concatenate(label_parts)


def predict_number_probabilities(
    model: xgb.Booster,
    history: Sequence[Sequence[int]] | np.ndarray,
) -> np.ndarray:
    features = build_number_features(history)
    probs = model.predict(
        xgb.DMatrix(features, feature_names=list(NUMBER_FEATURE_NAMES))
    )
    return np.clip(np.asarray(probs, dtype=np.float32), 1e-6, 1.0 - 1e-6)


def score_tickets_from_number_probs(
    candidates: np.ndarray,
    number_probs: np.ndarray,
) -> np.ndarray:
    tickets = np.asarray(candidates, dtype=np.int16)
    probs = np.asarray(number_probs, dtype=np.float64)
    logits = np.log(probs / (1.0 - probs))
    return np.sum(logits[tickets - 1], axis=1).astype(np.float32)


def number_topk_metrics(
    scores: np.ndarray,
    labels: np.ndarray,
) -> dict:
    score_groups = np.asarray(scores).reshape(-1, TOTAL_BALLS)
    label_groups = np.asarray(labels).reshape(-1, TOTAL_BALLS)
    metrics = {}
    for k in (6, 10):
        top_idx = np.argpartition(-score_groups, k - 1, axis=1)[:, :k]
        hits = np.take_along_axis(label_groups, top_idx, axis=1).sum(axis=1)
        metrics[f"recall_at_{k}"] = float(np.mean(hits / 6.0))
        metrics[f"mean_hits_at_{k}"] = float(np.mean(hits))
    return metrics
