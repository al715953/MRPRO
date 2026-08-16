from __future__ import annotations

from dataclasses import dataclass
import heapq
import math

import numpy as np

from .covering import CombinatorialProblem


@dataclass(frozen=True)
class DesignSolution:
    method: str
    ticket_indices: tuple[int, ...]
    coverage_trace: tuple[int, ...]
    targets_covered: int
    total_targets: int
    iterations: int
    initial_targets_covered: int = 0

    @property
    def coverage(self) -> float:
        if self.total_targets <= 0:
            return 0.0
        return float(self.targets_covered / self.total_targets)


def _inverse_incidence(incidence: np.ndarray, n_targets: int) -> list[list[int]]:
    inverse = [[] for _ in range(int(n_targets))]
    for ticket_idx, targets in enumerate(incidence):
        for target_idx in targets:
            inverse[int(target_idx)].append(ticket_idx)
    return inverse


def greedy_maximum_coverage(
    problem: CombinatorialProblem,
    ticket_budget: int,
    coverage_target: float = 1.0,
) -> DesignSolution:
    """Greedy maximum coverage with compact incidence and lazy heap updates."""

    budget = min(max(0, int(ticket_budget)), problem.n_tickets)
    if budget == 0:
        return DesignSolution("COVER_GREEDY", (), (), 0, problem.n_targets, 0)

    goal = int(math.ceil(float(coverage_target) * problem.n_targets))
    incidence = problem.ticket_target_indices
    inverse = _inverse_incidence(incidence, problem.n_targets)
    gains = np.full(problem.n_tickets, incidence.shape[1], dtype=np.int32)
    selected_mask = np.zeros(problem.n_tickets, dtype=bool)
    covered = np.zeros(problem.n_targets, dtype=bool)
    heap = [(-int(gain), idx) for idx, gain in enumerate(gains)]
    heapq.heapify(heap)
    selected = []
    trace = []
    covered_count = 0

    while len(selected) < budget and covered_count < goal:
        best_idx = None
        while heap:
            neg_gain, candidate_idx = heapq.heappop(heap)
            if selected_mask[candidate_idx]:
                continue
            current_gain = int(gains[candidate_idx])
            if -neg_gain != current_gain:
                heapq.heappush(heap, (-current_gain, candidate_idx))
                continue
            best_idx = candidate_idx
            break
        if best_idx is None or int(gains[best_idx]) <= 0:
            break

        selected.append(best_idx)
        selected_mask[best_idx] = True
        new_targets = [
            int(target)
            for target in incidence[best_idx]
            if not covered[int(target)]
        ]
        for target_idx in new_targets:
            covered[target_idx] = True
            covered_count += 1
            for candidate_idx in inverse[target_idx]:
                if selected_mask[candidate_idx]:
                    continue
                gains[candidate_idx] -= 1
                heapq.heappush(heap, (-int(gains[candidate_idx]), candidate_idx))
        gains[best_idx] = -1
        trace.append(covered_count)

    return DesignSolution(
        method="COVER_GREEDY",
        ticket_indices=tuple(int(idx) for idx in selected),
        coverage_trace=tuple(int(value) for value in trace),
        targets_covered=int(covered_count),
        total_targets=problem.n_targets,
        iterations=len(selected),
    )
