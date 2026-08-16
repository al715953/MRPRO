from __future__ import annotations

from collections import Counter
from typing import Sequence

import numpy as np

from .covering import CombinatorialProblem


def coverage_metrics(
    problem: CombinatorialProblem,
    ticket_indices: Sequence[int],
) -> dict:
    selected = np.asarray(ticket_indices, dtype=np.int64)
    counts = np.zeros(problem.n_targets, dtype=np.int32)
    if selected.size:
        np.add.at(counts, problem.ticket_target_indices[selected].ravel(), 1)
    covered = int(np.count_nonzero(counts))
    ticket_count = int(selected.size)
    histogram = {
        str(int(key)): int(value)
        for key, value in sorted(Counter(counts.tolist()).items())
    }
    coverage = float(covered / problem.n_targets) if problem.n_targets else 0.0
    return {
        "ticket_count": ticket_count,
        "targets_covered": covered,
        "targets_total": int(problem.n_targets),
        "coverage_t": coverage,
        "redundancy_mean": float(np.mean(counts)) if counts.size else 0.0,
        "redundancy_min": int(np.min(counts)) if counts.size else 0,
        "redundancy_max": int(np.max(counts)) if counts.size else 0,
        "redundancy_std": float(np.std(counts)) if counts.size else 0.0,
        "redundancy_histogram": histogram,
        "efficiency_t": float(coverage / ticket_count) if ticket_count else 0.0,
    }


def max_ticket_hits(tickets: np.ndarray, winning_numbers: Sequence[int]) -> int:
    if tickets is None or len(tickets) == 0:
        return 0
    winning = np.asarray(list(winning_numbers), dtype=np.int16)
    hits = np.sum(np.isin(np.asarray(tickets), winning), axis=1)
    return int(np.max(hits))


def validate_ticket_matrix(
    tickets: np.ndarray,
    candidate_numbers: Sequence[int],
    ticket_size: int,
) -> None:
    matrix = np.asarray(tickets)
    if matrix.ndim != 2 or matrix.shape[1] != int(ticket_size):
        raise ValueError("Matriz de boletos con forma inválida")
    allowed = set(int(number) for number in candidate_numbers)
    canonical = set()
    for row in matrix:
        ticket = tuple(int(number) for number in row)
        if tuple(sorted(ticket)) != ticket:
            raise ValueError("Los boletos deben estar ordenados")
        if len(set(ticket)) != int(ticket_size):
            raise ValueError("Un boleto contiene números duplicados")
        if not set(ticket).issubset(allowed):
            raise ValueError("Un boleto contiene números fuera del candidato")
        if ticket in canonical:
            raise ValueError("La cartera contiene boletos duplicados")
        canonical.add(ticket)
