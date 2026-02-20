from __future__ import annotations

from typing import Iterable, List, Sequence, Tuple

import numpy as np


def beam_search(
    pos_probs: np.ndarray,
    k: int = 1000,
    per_pos_topm: int = 6,
    beam_width: int = 2000,
) -> List[Tuple[List[int], float]]:
    """
    Beam search sobre 5 posiciones con log-probabilidades.
    """
    probs = np.asarray(pos_probs, dtype=np.float64)
    probs = np.clip(probs, 1e-12, None)
    probs /= np.clip(probs.sum(axis=1, keepdims=True), 1e-12, None)

    beam: List[Tuple[List[int], float]] = [([], 0.0)]
    n_pos = probs.shape[0]

    for pos in range(n_pos):
        row = probs[pos]
        top_digits = np.argsort(row)[::-1][: max(1, int(per_pos_topm))]
        expansions: List[Tuple[List[int], float]] = []
        for partial, lp in beam:
            for d in top_digits:
                new_lp = lp + float(np.log(row[int(d)]))
                expansions.append((partial + [int(d)], new_lp))

        expansions.sort(key=lambda x: x[1], reverse=True)
        beam = expansions[: max(1, int(beam_width))]

    beam.sort(key=lambda x: x[1], reverse=True)
    return beam[: max(1, int(k))]


def _hamming(a: Sequence[int], b: Sequence[int]) -> int:
    return sum(1 for x, y in zip(a, b) if x != y)


def select_diverse(
    candidates: Iterable[Tuple[List[int], float]],
    n: int,
    min_hamming: int = 2,
) -> List[List[int]]:
    ranked = list(candidates)
    if n <= 0:
        return []
    if not ranked:
        return []

    selected: List[List[int]] = []
    target_n = int(n)

    for threshold in (int(min_hamming), 1, 0):
        for digits, _ in ranked:
            if len(selected) >= target_n:
                break
            if digits in selected:
                continue
            if all(_hamming(digits, prev) >= threshold for prev in selected):
                selected.append(digits)
        if len(selected) >= target_n:
            break

    if len(selected) < target_n:
        for digits, _ in ranked:
            if len(selected) >= target_n:
                break
            if digits not in selected:
                selected.append(digits)

    return selected[:target_n]
