from __future__ import annotations

from typing import Iterator

import numpy as np

from .covering import CombinatorialProblem
from .metrics import coverage_metrics


def random_same_size_indices(
    n_candidates: int,
    ticket_count: int,
    *,
    rng: np.random.Generator,
) -> np.ndarray:
    size = min(max(0, int(ticket_count)), int(n_candidates))
    return np.sort(rng.choice(int(n_candidates), size=size, replace=False))


def iter_random_same_size(
    n_candidates: int,
    ticket_count: int,
    trials: int,
    seed: int,
) -> Iterator[np.ndarray]:
    rng = np.random.default_rng(int(seed))
    for _ in range(int(trials)):
        yield random_same_size_indices(n_candidates, ticket_count, rng=rng)


def summarize_distribution(values) -> dict:
    array = np.asarray(values, dtype=np.float64)
    if array.size == 0:
        return {
            key: None
            for key in (
                "mean",
                "median",
                "std",
                "p05",
                "p25",
                "p50",
                "p75",
                "p95",
                "min",
                "max",
            )
        }
    p05, p25, p50, p75, p95 = np.percentile(array, [5, 25, 50, 75, 95])
    return {
        "mean": float(np.mean(array)),
        "median": float(np.median(array)),
        "std": float(np.std(array)),
        "p05": float(p05),
        "p25": float(p25),
        "p50": float(p50),
        "p75": float(p75),
        "p95": float(p95),
        "min": float(np.min(array)),
        "max": float(np.max(array)),
    }


def random_coverage_distribution(
    problem: CombinatorialProblem,
    ticket_count: int,
    trials: int,
    seed: int,
) -> dict:
    coverages = []
    redundancies = []
    for indices in iter_random_same_size(
        problem.n_tickets, ticket_count, trials, seed
    ):
        metrics = coverage_metrics(problem, indices)
        coverages.append(metrics["coverage_t"])
        redundancies.append(metrics["redundancy_mean"])
    return {
        "trials": int(trials),
        "ticket_count": min(int(ticket_count), problem.n_tickets),
        "coverage": summarize_distribution(coverages),
        "redundancy_mean": summarize_distribution(redundancies),
        "raw_coverage": [float(value) for value in coverages],
    }


def score_against_random(value: float, random_values) -> dict:
    array = np.asarray(random_values, dtype=np.float64)
    if array.size == 0:
        return {"percentile_rank": None, "z_score": None}
    std = float(np.std(array))
    return {
        "percentile_rank": float(100.0 * np.mean(array <= float(value))),
        "z_score": (
            float((float(value) - float(np.mean(array))) / std) if std > 0 else None
        ),
    }
