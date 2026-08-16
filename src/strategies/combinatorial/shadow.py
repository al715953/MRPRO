from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from typing import Any

import numpy as np

from .candidate_sets import mrpro_candidate_set_from_snapshot
from .covering import CombinatorialProblem
from .metrics import coverage_metrics, validate_ticket_matrix
from .multiobjective import (
    improve_weighted_local_search,
    weighted_greedy_maximum_coverage,
)


@dataclass(frozen=True)
class CoveringShadowSpec:
    key: str
    label: str
    candidate_pool_size: int
    ticket_budget: int = 300
    rank_depth: int = 500
    primary_target_size: int = 5
    secondary_target_size: int = 4
    primary_weight: float = 0.5
    secondary_weight: float = 0.5
    local_search_iterations: int = 10


PROMOTED_COVERING_SHADOWS = (
    CoveringShadowSpec(
        key="cover_mixed_v20_m300",
        label="Sombra Cover mixto V20 / 300",
        candidate_pool_size=20,
    ),
    CoveringShadowSpec(
        key="cover_mixed_v18_m300",
        label="Sombra Cover mixto V18 / 300",
        candidate_pool_size=18,
    ),
)


@lru_cache(maxsize=16)
def _cached_design(
    v: int,
    ticket_size: int,
    primary_t: int,
    secondary_t: int,
    primary_weight: float,
    secondary_weight: float,
    ticket_budget: int,
    local_iterations: int,
) -> tuple[np.ndarray, dict[str, float], float]:
    canonical = tuple(range(1, int(v) + 1))
    problems = {
        int(primary_t): CombinatorialProblem.build(
            canonical, int(ticket_size), int(primary_t)
        ),
        int(secondary_t): CombinatorialProblem.build(
            canonical, int(ticket_size), int(secondary_t)
        ),
    }
    weight_total = float(primary_weight + secondary_weight)
    weights = {
        int(primary_t): float(primary_weight / weight_total),
        int(secondary_t): float(secondary_weight / weight_total),
    }
    greedy = weighted_greedy_maximum_coverage(
        problems,
        weights,
        int(ticket_budget),
    )
    local = improve_weighted_local_search(
        problems,
        weights,
        greedy,
        max_iterations=int(local_iterations),
    )
    primary_problem = problems[int(primary_t)]
    positions = primary_problem.ticket_positions[
        np.asarray(local.solution.ticket_indices, dtype=np.int64)
    ].copy()
    coverage_by_t = {
        str(target_size): float(
            coverage_metrics(problem, local.solution.ticket_indices)["coverage_t"]
        )
        for target_size, problem in problems.items()
    }
    positions.setflags(write=False)
    return positions, coverage_by_t, float(local.weighted_coverage)


def build_covering_shadow_variant(
    snapshot: dict[str, Any],
    spec: CoveringShadowSpec,
    *,
    total_balls: int,
    ticket_size: int,
) -> dict[str, Any]:
    """Build one prospective portfolio from a pre-draw MRPRO snapshot."""

    candidates = mrpro_candidate_set_from_snapshot(
        snapshot,
        v=spec.candidate_pool_size,
        total_balls=total_balls,
        rank_depth=spec.rank_depth,
    )
    positions, coverage_by_t, weighted_coverage = _cached_design(
        spec.candidate_pool_size,
        ticket_size,
        spec.primary_target_size,
        spec.secondary_target_size,
        spec.primary_weight,
        spec.secondary_weight,
        spec.ticket_budget,
        spec.local_search_iterations,
    )
    candidate_array = np.asarray(candidates, dtype=np.int16)
    tickets = np.sort(candidate_array[positions], axis=1)
    validate_ticket_matrix(tickets, candidates, ticket_size)
    settings = {
        "shadow_family": "combinatorial_covering",
        "candidate_method": "mrpro_candidate_set",
        "candidate_pool_size": int(spec.candidate_pool_size),
        "candidate_rank_depth": int(spec.rank_depth),
        "ticket_budget": int(spec.ticket_budget),
        "coverage_algorithm": "weighted_greedy_local",
        "target_subset_sizes": [
            int(spec.primary_target_size),
            int(spec.secondary_target_size),
        ],
        "target_weights": {
            str(spec.primary_target_size): float(spec.primary_weight),
            str(spec.secondary_target_size): float(spec.secondary_weight),
        },
        "local_search_iterations": int(spec.local_search_iterations),
    }
    return {
        "key": spec.key,
        "label": spec.label,
        "official": False,
        "settings": settings,
        "tickets": tickets.tolist(),
        "metadata": {
            **settings,
            "candidate_numbers": [int(number) for number in candidates],
            "coverage_by_t": coverage_by_t,
            "weighted_coverage": float(weighted_coverage),
            "ticket_count": int(len(tickets)),
        },
    }


def build_promoted_covering_shadows(
    snapshot: dict[str, Any],
    *,
    total_balls: int,
    ticket_size: int,
) -> list[dict[str, Any]]:
    return [
        build_covering_shadow_variant(
            snapshot,
            spec,
            total_balls=total_balls,
            ticket_size=ticket_size,
        )
        for spec in PROMOTED_COVERING_SHADOWS
    ]
