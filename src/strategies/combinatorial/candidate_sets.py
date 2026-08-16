from __future__ import annotations

from collections import Counter
from typing import Sequence

import numpy as np


def _validate_candidate_set(numbers: Sequence[int], v: int, total_balls: int):
    result = tuple(int(number) for number in numbers)
    if len(result) != int(v) or len(set(result)) != int(v):
        raise ValueError("El conjunto candidato debe contener exactamente v números únicos")
    if any(number < 1 or number > int(total_balls) for number in result):
        raise ValueError("El conjunto candidato contiene números fuera del juego")
    return result


def oracle_candidate_set(
    winning_numbers: Sequence[int],
    *,
    v: int,
    total_balls: int,
    rng: np.random.Generator,
) -> tuple[int, ...]:
    """Controlled non-predictive set that always contains the winning ticket."""

    winner = tuple(sorted(set(int(number) for number in winning_numbers)))
    if len(winner) > int(v):
        raise ValueError("v es menor que la cantidad de números ganadores")
    remaining = np.asarray(
        [number for number in range(1, int(total_balls) + 1) if number not in winner],
        dtype=np.int16,
    )
    fill = rng.choice(remaining, size=int(v) - len(winner), replace=False)
    return _validate_candidate_set(sorted((*winner, *(int(x) for x in fill))), v, total_balls)


def random_candidate_set(
    *,
    v: int,
    total_balls: int,
    rng: np.random.Generator,
) -> tuple[int, ...]:
    values = rng.choice(
        np.arange(1, int(total_balls) + 1, dtype=np.int16),
        size=int(v),
        replace=False,
    )
    return _validate_candidate_set(sorted(int(value) for value in values), v, total_balls)


def explicit_candidate_set(
    values: Sequence[int],
    *,
    v: int,
    total_balls: int,
) -> tuple[int, ...]:
    return _validate_candidate_set(sorted(int(value) for value in values), v, total_balls)


def mrpro_candidate_set_from_snapshot(
    snapshot: dict,
    *,
    v: int,
    total_balls: int,
    rank_depth: int = 500,
) -> tuple[int, ...]:
    """Rank numbers by weighted presence in MRPRO's highest-scored tickets."""

    universe = snapshot.get("universe")
    scores = snapshot.get("hybrid_scores")
    if universe is None or scores is None:
        raise ValueError("La predicción MRPRO no contiene universo/hybrid_scores")
    if hasattr(universe, "get"):
        universe = universe.get()
    if hasattr(scores, "get"):
        scores = scores.get()
    tickets = np.asarray(universe, dtype=np.int16)
    score_array = np.asarray(scores, dtype=np.float64).reshape(-1)
    if tickets.ndim != 2 or tickets.shape[0] != score_array.size:
        raise ValueError("Snapshot MRPRO inconsistente")

    depth = min(max(int(v), int(rank_depth)), len(tickets))
    order = np.argsort(-score_array, kind="stable")[:depth]
    number_scores = Counter()
    number_frequency = Counter(int(number) for row in tickets for number in row)
    for rank, ticket_idx in enumerate(order, start=1):
        weight = 1.0 / np.log2(rank + 1.0)
        for number in tickets[int(ticket_idx)]:
            number_scores[int(number)] += float(weight)

    ranking = sorted(
        range(1, int(total_balls) + 1),
        key=lambda number: (
            -number_scores[number],
            -number_frequency[number],
            number,
        ),
    )
    # Keep rank order: truncated designs may legitimately exploit MRPRO's ordering.
    return _validate_candidate_set(ranking[: int(v)], v, total_balls)


def candidate_hit_count(
    candidate_numbers: Sequence[int], winning_numbers: Sequence[int]
) -> int:
    return len(set(int(number) for number in candidate_numbers).intersection(
        int(number) for number in winning_numbers
    ))
