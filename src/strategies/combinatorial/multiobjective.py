from __future__ import annotations

from dataclasses import dataclass
import heapq
from typing import Mapping, Sequence

import numpy as np

from .covering import CombinatorialProblem
from .greedy import DesignSolution
from .metrics import coverage_metrics


@dataclass(frozen=True)
class MultiObjectiveResult:
    """A ticket portfolio scored against several target subset sizes."""

    solution: DesignSolution
    coverage_by_t: dict[int, float]
    weighted_coverage: float
    objective_trace: tuple[float, ...]


def _validate_problems(
    problems: Mapping[int, CombinatorialProblem],
    weights: Mapping[int, float],
) -> tuple[tuple[int, ...], dict[int, float]]:
    target_sizes = tuple(sorted(int(value) for value in problems))
    if not target_sizes:
        raise ValueError("Se requiere al menos un objetivo de cobertura")
    first = problems[target_sizes[0]]
    for target_size in target_sizes:
        problem = problems[target_size]
        if problem.n_tickets != first.n_tickets or not np.array_equal(
            problem.ticket_positions, first.ticket_positions
        ):
            raise ValueError("Los objetivos deben compartir el mismo catálogo de boletos")
    normalized = {target_size: float(weights[target_size]) for target_size in target_sizes}
    if any(value <= 0 for value in normalized.values()):
        raise ValueError("Los pesos multiobjetivo deben ser positivos")
    total = sum(normalized.values())
    return target_sizes, {
        target_size: value / total for target_size, value in normalized.items()
    }


def _inverse_incidence(incidence: np.ndarray, n_targets: int) -> list[list[int]]:
    inverse = [[] for _ in range(int(n_targets))]
    for ticket_idx, targets in enumerate(incidence):
        for target_idx in targets:
            inverse[int(target_idx)].append(ticket_idx)
    return inverse


def _result(
    problems: Mapping[int, CombinatorialProblem],
    weights: Mapping[int, float],
    indices: Sequence[int],
    *,
    method: str,
    trace: Sequence[float],
    iterations: int,
    initial_targets_covered: int = 0,
) -> MultiObjectiveResult:
    primary_t = int(next(iter(problems)))
    metrics = {
        target_size: coverage_metrics(problem, indices)
        for target_size, problem in problems.items()
    }
    coverage_by_t = {
        int(target_size): float(value["coverage_t"])
        for target_size, value in metrics.items()
    }
    weighted = float(
        sum(weights[target_size] * coverage_by_t[target_size] for target_size in problems)
    )
    primary = metrics[primary_t]
    solution = DesignSolution(
        method=method,
        ticket_indices=tuple(int(value) for value in indices),
        coverage_trace=(),
        targets_covered=int(primary["targets_covered"]),
        total_targets=int(primary["targets_total"]),
        iterations=int(iterations),
        initial_targets_covered=int(initial_targets_covered),
    )
    return MultiObjectiveResult(
        solution=solution,
        coverage_by_t=coverage_by_t,
        weighted_coverage=weighted,
        objective_trace=tuple(float(value) for value in trace),
    )


def weighted_greedy_maximum_coverage(
    problems: Mapping[int, CombinatorialProblem],
    weights: Mapping[int, float],
    ticket_budget: int,
    coverage_target: float = 1.0,
) -> MultiObjectiveResult:
    """Greedy maximum coverage over normalized, weighted t-coverages."""

    target_sizes, normalized_weights = _validate_problems(problems, weights)
    first = problems[target_sizes[0]]
    budget = min(max(0, int(ticket_budget)), first.n_tickets)
    selected_mask = np.zeros(first.n_tickets, dtype=bool)
    covered = {
        target_size: np.zeros(problems[target_size].n_targets, dtype=bool)
        for target_size in target_sizes
    }
    inverse = {
        target_size: _inverse_incidence(
            problems[target_size].ticket_target_indices,
            problems[target_size].n_targets,
        )
        for target_size in target_sizes
    }
    gains = np.zeros(first.n_tickets, dtype=np.float64)
    for target_size in target_sizes:
        problem = problems[target_size]
        gains += (
            normalized_weights[target_size]
            * problem.ticket_target_indices.shape[1]
            / problem.n_targets
        )
    heap = [(-float(gain), idx) for idx, gain in enumerate(gains)]
    heapq.heapify(heap)
    selected: list[int] = []
    trace: list[float] = []

    goal = min(max(float(coverage_target), 0.0), 1.0)
    while len(selected) < budget and (not trace or trace[-1] < goal):
        best_idx = None
        while heap:
            negative_gain, candidate_idx = heapq.heappop(heap)
            if selected_mask[candidate_idx]:
                continue
            current_gain = float(gains[candidate_idx])
            if not np.isclose(-negative_gain, current_gain, rtol=0.0, atol=1e-15):
                heapq.heappush(heap, (-current_gain, candidate_idx))
                continue
            best_idx = int(candidate_idx)
            break
        if best_idx is None or gains[best_idx] <= 1e-15:
            break

        selected.append(best_idx)
        selected_mask[best_idx] = True
        for target_size in target_sizes:
            problem = problems[target_size]
            contribution = normalized_weights[target_size] / problem.n_targets
            for target_idx_raw in problem.ticket_target_indices[best_idx]:
                target_idx = int(target_idx_raw)
                if covered[target_size][target_idx]:
                    continue
                covered[target_size][target_idx] = True
                for candidate_idx in inverse[target_size][target_idx]:
                    if not selected_mask[candidate_idx]:
                        gains[candidate_idx] -= contribution
                        heapq.heappush(
                            heap, (-float(gains[candidate_idx]), candidate_idx)
                        )
        gains[best_idx] = -1.0
        trace.append(
            float(
                sum(
                    normalized_weights[target_size]
                    * np.mean(covered[target_size])
                    for target_size in target_sizes
                )
            )
        )

    return _result(
        problems,
        normalized_weights,
        selected,
        method="COVER_GREEDY_MIXED",
        trace=trace,
        iterations=len(selected),
    )


def improve_weighted_local_search(
    problems: Mapping[int, CombinatorialProblem],
    weights: Mapping[int, float],
    initial: MultiObjectiveResult,
    *,
    max_iterations: int = 30,
    removal_candidates: int = 8,
) -> MultiObjectiveResult:
    """Deterministic 1-for-1 swaps; ticket count remains exactly fixed."""

    target_sizes, normalized_weights = _validate_problems(problems, weights)
    selected = list(initial.solution.ticket_indices)
    selected_set = set(selected)
    counts = {}
    for target_size in target_sizes:
        problem = problems[target_size]
        value = np.zeros(problem.n_targets, dtype=np.int32)
        if selected:
            np.add.at(
                value,
                problem.ticket_target_indices[np.asarray(selected)].ravel(),
                1,
            )
        counts[target_size] = value

    current_score = float(initial.weighted_coverage)
    trace = list(initial.objective_trace)
    iterations = 0
    for _ in range(max(0, int(max_iterations))):
        iterations += 1
        losses = []
        for selected_idx in selected:
            loss = 0.0
            for target_size in target_sizes:
                problem = problems[target_size]
                loss += (
                    normalized_weights[target_size]
                    * np.sum(counts[target_size][problem.ticket_target_indices[selected_idx]] == 1)
                    / problem.n_targets
                )
            losses.append(float(loss))
        removal_order = np.argsort(losses, kind="stable")[: min(removal_candidates, len(selected))]
        best = None
        best_score = current_score
        for position_raw in removal_order:
            position = int(position_raw)
            removed_idx = selected[position]
            gains = np.zeros(problems[target_sizes[0]].n_tickets, dtype=np.float64)
            after_counts = {}
            base_score = 0.0
            for target_size in target_sizes:
                problem = problems[target_size]
                target_counts = counts[target_size].copy()
                target_counts[problem.ticket_target_indices[removed_idx]] -= 1
                after_counts[target_size] = target_counts
                base_score += (
                    normalized_weights[target_size]
                    * np.count_nonzero(target_counts)
                    / problem.n_targets
                )
                gains += (
                    normalized_weights[target_size]
                    * np.sum(target_counts[problem.ticket_target_indices] == 0, axis=1)
                    / problem.n_targets
                )
            gains[np.fromiter(selected_set, dtype=np.int64)] = -1.0
            add_idx = int(np.argmax(gains))
            candidate_score = float(base_score + gains[add_idx])
            if candidate_score > best_score + 1e-15:
                best_score = candidate_score
                best = position, removed_idx, add_idx, after_counts
        if best is None:
            iterations -= 1
            break
        position, removed_idx, add_idx, after_counts = best
        selected[position] = add_idx
        selected_set.remove(removed_idx)
        selected_set.add(add_idx)
        counts = after_counts
        for target_size in target_sizes:
            counts[target_size][problems[target_size].ticket_target_indices[add_idx]] += 1
        current_score = best_score
        trace.append(current_score)

    return _result(
        problems,
        normalized_weights,
        selected,
        method="COVER_GREEDY_LOCAL_MIXED",
        trace=trace,
        iterations=iterations,
        initial_targets_covered=initial.solution.targets_covered,
    )
