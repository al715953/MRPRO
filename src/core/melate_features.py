from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Sequence

import numpy as np


FEATURE_SCHEMA = "melate_context_v1"
TOTAL_BALLS = 39
TICKET_SIZE = 6
PRIMES = np.asarray([2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37])

FEATURE_NAMES = (
    "n1",
    "n2",
    "n3",
    "n4",
    "n5",
    "n6",
    "sum",
    "span",
    "std",
    "even_ratio",
    "prime_ratio",
    "consecutive_ratio",
    "long_freq_mean",
    "long_freq_min",
    "long_freq_max",
    "freq20_mean",
    "freq20_min",
    "freq20_max",
    "freq50_mean",
    "freq50_min",
    "freq50_max",
    "gap_mean",
    "gap_min",
    "gap_max",
    "pair_rate_mean",
    "pair_rate_max",
    "recent5_freq_mean",
    "recent5_freq_max",
    "last_draw_overlap",
)


@dataclass(frozen=True)
class MelateContextState:
    n_draws: int
    long_rates: np.ndarray
    rates20: np.ndarray
    rates50: np.ndarray
    gaps: np.ndarray
    pair_rates: np.ndarray
    recent5_rates: np.ndarray
    last_draw_mask: np.ndarray


def _coerce_draws(history: Sequence[Sequence[int]] | np.ndarray) -> np.ndarray:
    rows = [list(draw[:TICKET_SIZE]) for draw in history if len(draw) >= TICKET_SIZE]
    if not rows:
        return np.empty((0, TICKET_SIZE), dtype=np.int16)
    draws = np.sort(np.asarray(rows, dtype=np.int16), axis=1)
    if np.any(draws < 1) or np.any(draws > TOTAL_BALLS):
        raise ValueError("El historial Melate contiene números fuera del rango 1..39.")
    return draws


def _number_rates(draws: np.ndarray, window: int | None = None) -> np.ndarray:
    selected = draws if window is None else draws[-int(window) :]
    denominator = max(1, len(selected))
    counts = np.bincount(selected.ravel(), minlength=TOTAL_BALLS + 1).astype(
        np.float32
    )
    return counts / float(denominator)


def build_context_state(
    history: Sequence[Sequence[int]] | np.ndarray,
) -> MelateContextState:
    """Summarize only the draws supplied as past context."""
    draws = _coerce_draws(history)
    n_draws = len(draws)

    long_rates = _number_rates(draws)
    rates20 = _number_rates(draws, 20)
    rates50 = _number_rates(draws, 50)
    recent5_rates = _number_rates(draws, 5)

    gaps = np.full(TOTAL_BALLS + 1, max(1, n_draws), dtype=np.float32)
    if n_draws:
        for offset, draw in enumerate(draws[::-1]):
            unseen = gaps[draw] == max(1, n_draws)
            gaps[draw[unseen]] = float(offset)
    gaps /= float(max(1, n_draws))

    pair_rates = np.zeros(
        (TOTAL_BALLS + 1, TOTAL_BALLS + 1), dtype=np.float32
    )
    for draw in draws:
        for left in range(TICKET_SIZE):
            for right in range(left + 1, TICKET_SIZE):
                a, b = int(draw[left]), int(draw[right])
                pair_rates[a, b] += 1.0
                pair_rates[b, a] += 1.0
    pair_rates /= float(max(1, n_draws))

    last_draw_mask = np.zeros(TOTAL_BALLS + 1, dtype=np.float32)
    if n_draws:
        last_draw_mask[draws[-1]] = 1.0

    return MelateContextState(
        n_draws=n_draws,
        long_rates=long_rates,
        rates20=rates20,
        rates50=rates50,
        gaps=gaps,
        pair_rates=pair_rates,
        recent5_rates=recent5_rates,
        last_draw_mask=last_draw_mask,
    )


def build_candidate_features(
    candidates: Iterable[Iterable[int]] | np.ndarray,
    history: Sequence[Sequence[int]] | np.ndarray,
) -> np.ndarray:
    """Build candidate features using no information beyond ``history``."""
    tickets = np.sort(np.asarray(candidates, dtype=np.int16), axis=1)
    if tickets.ndim != 2 or tickets.shape[1] != TICKET_SIZE:
        raise ValueError("Los candidatos Melate deben tener forma (N, 6).")
    if np.any(tickets < 1) or np.any(tickets > TOTAL_BALLS):
        raise ValueError("Los candidatos Melate deben usar números entre 1 y 39.")

    state = build_context_state(history)
    ticket_float = tickets.astype(np.float32)
    normalized_numbers = ticket_float / float(TOTAL_BALLS)

    sums = np.sum(ticket_float, axis=1) / float(TOTAL_BALLS * TICKET_SIZE)
    spans = (ticket_float[:, -1] - ticket_float[:, 0]) / float(TOTAL_BALLS - 1)
    stds = np.std(ticket_float, axis=1) / float(TOTAL_BALLS)
    even_ratio = np.mean((tickets % 2) == 0, axis=1)
    prime_ratio = np.mean(np.isin(tickets, PRIMES), axis=1)
    consecutive_ratio = np.sum(np.diff(tickets, axis=1) == 1, axis=1) / float(
        TICKET_SIZE - 1
    )

    def aggregate_number_signal(signal: np.ndarray) -> tuple[np.ndarray, ...]:
        values = signal[tickets]
        return (
            np.mean(values, axis=1),
            np.min(values, axis=1),
            np.max(values, axis=1),
        )

    long_agg = aggregate_number_signal(state.long_rates)
    freq20_agg = aggregate_number_signal(state.rates20)
    freq50_agg = aggregate_number_signal(state.rates50)
    gap_agg = aggregate_number_signal(state.gaps)

    pair_values = []
    for left in range(TICKET_SIZE):
        for right in range(left + 1, TICKET_SIZE):
            pair_values.append(state.pair_rates[tickets[:, left], tickets[:, right]])
    pair_matrix = np.column_stack(pair_values)

    recent5 = state.recent5_rates[tickets]
    last_overlap = np.mean(state.last_draw_mask[tickets], axis=1)

    features = np.column_stack(
        (
            normalized_numbers,
            sums,
            spans,
            stds,
            even_ratio,
            prime_ratio,
            consecutive_ratio,
            *long_agg,
            *freq20_agg,
            *freq50_agg,
            *gap_agg,
            np.mean(pair_matrix, axis=1),
            np.max(pair_matrix, axis=1),
            np.mean(recent5, axis=1),
            np.max(recent5, axis=1),
            last_overlap,
        )
    ).astype(np.float32, copy=False)

    if features.shape[1] != len(FEATURE_NAMES):
        raise RuntimeError(
            f"Esquema contextual inválido: {features.shape[1]} != {len(FEATURE_NAMES)}"
        )
    return features
