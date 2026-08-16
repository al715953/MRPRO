import math

import numpy as np
import pytest

from src.strategies.combinatorial.baselines import random_same_size_indices
from src.strategies.combinatorial.candidate_sets import (
    oracle_candidate_set,
    random_candidate_set,
)
from src.strategies.combinatorial.covering import (
    CombinatorialProblem,
    ProblemTooLargeError,
    estimate_problem,
)
from src.strategies.combinatorial.greedy import greedy_maximum_coverage
from src.strategies.combinatorial.local_search import improve_by_local_search
from src.strategies.combinatorial.metrics import (
    coverage_metrics,
    validate_ticket_matrix,
)


@pytest.fixture(scope="module")
def toy_problem():
    return CombinatorialProblem.build(range(1, 9), k=4, t=3)


def test_toy_problem_has_exact_enumeration(toy_problem):
    assert toy_problem.n_tickets == math.comb(8, 4) == 70
    assert toy_problem.n_targets == math.comb(8, 3) == 56
    assert toy_problem.ticket_target_indices.shape == (70, 4)


def test_exhaustive_coverage_is_one(toy_problem):
    metrics = coverage_metrics(toy_problem, np.arange(toy_problem.n_tickets))

    assert metrics["coverage_t"] == 1.0
    assert metrics["targets_covered"] == 56
    assert metrics["redundancy_min"] == 5
    assert metrics["redundancy_max"] == 5


@pytest.mark.parametrize("target_size", [1, 2, 3, 4])
def test_exhaustive_coverage_is_one_for_every_t_le_k(target_size):
    problem = CombinatorialProblem.build(range(1, 9), k=4, t=target_size)
    metrics = coverage_metrics(problem, np.arange(problem.n_tickets))

    assert metrics["coverage_t"] == 1.0


def test_greedy_coverage_trace_is_monotonic(toy_problem):
    solution = greedy_maximum_coverage(toy_problem, ticket_budget=12)

    assert len(solution.ticket_indices) == 12
    assert len(set(solution.ticket_indices)) == 12
    assert all(
        right >= left
        for left, right in zip(solution.coverage_trace, solution.coverage_trace[1:])
    )
    assert solution.targets_covered == solution.coverage_trace[-1]


def test_local_search_never_reduces_greedy_coverage(toy_problem):
    greedy = greedy_maximum_coverage(toy_problem, ticket_budget=10)
    improved = improve_by_local_search(
        toy_problem,
        greedy,
        max_iterations=30,
    )

    assert improved.targets_covered >= greedy.targets_covered
    assert len(improved.ticket_indices) <= len(greedy.ticket_indices)
    assert len(set(improved.ticket_indices)) == len(improved.ticket_indices)


def test_random_baseline_has_exact_same_size_and_no_duplicates(toy_problem):
    indices = random_same_size_indices(
        toy_problem.n_tickets,
        17,
        rng=np.random.default_rng(7),
    )

    assert len(indices) == 17
    assert len(set(indices.tolist())) == 17
    validate_ticket_matrix(
        toy_problem.selected_tickets(indices),
        toy_problem.candidate_numbers,
        toy_problem.k,
    )


def test_oracle_always_contains_winning_ticket():
    winning = [2, 5, 11, 18, 27, 34]
    candidates = oracle_candidate_set(
        winning,
        v=15,
        total_balls=39,
        rng=np.random.default_rng(123),
    )

    assert len(candidates) == 15
    assert set(winning).issubset(candidates)


def test_random_candidate_set_is_reproducible_and_valid():
    first = random_candidate_set(v=15, total_balls=39, rng=np.random.default_rng(9))
    second = random_candidate_set(v=15, total_balls=39, rng=np.random.default_rng(9))

    assert first == second
    assert len(first) == len(set(first)) == 15
    assert min(first) >= 1 and max(first) <= 39


def test_guardrail_rejects_problem_before_enumeration():
    estimate = estimate_problem(20, 6, 4)
    assert estimate.candidate_tickets == math.comb(20, 6)

    with pytest.raises(ProblemTooLargeError, match="Configuración omitida"):
        CombinatorialProblem.build(
            range(1, 21),
            k=6,
            t=4,
            max_candidate_tickets=1_000,
        )
