from __future__ import annotations

import math

import numpy as np

from .covering import CombinatorialProblem
from .greedy import DesignSolution


def _coverage_counts(incidence: np.ndarray, selected: list[int], n_targets: int):
    counts = np.zeros(int(n_targets), dtype=np.int32)
    if selected:
        np.add.at(counts, incidence[np.asarray(selected, dtype=np.int64)].ravel(), 1)
    return counts


def improve_by_local_search(
    problem: CombinatorialProblem,
    initial: DesignSolution,
    *,
    max_iterations: int = 100,
    coverage_target: float = 1.0,
) -> DesignSolution:
    """Improve greedy by deterministic 1-for-1 swaps and redundancy pruning."""

    incidence = problem.ticket_target_indices
    selected = list(initial.ticket_indices)
    selected_set = set(selected)
    counts = _coverage_counts(incidence, selected, problem.n_targets)
    covered_count = int(np.count_nonzero(counts))
    goal = int(math.ceil(float(coverage_target) * problem.n_targets))
    iterations = 0

    while iterations < int(max_iterations) and selected:
        iterations += 1
        unique_loss = np.asarray(
            [int(np.sum(counts[incidence[idx]] == 1)) for idx in selected],
            dtype=np.int32,
        )
        removal_order = np.argsort(unique_loss, kind="stable")[: min(20, len(selected))]
        best_swap = None
        best_coverage = covered_count

        for removal_pos in removal_order:
            remove_idx = selected[int(removal_pos)]
            after_counts = counts.copy()
            after_counts[incidence[remove_idx]] -= 1
            base_coverage = int(np.count_nonzero(after_counts))
            gains = np.sum(after_counts[incidence] == 0, axis=1).astype(np.int32)
            if selected_set:
                gains[np.fromiter(selected_set, dtype=np.int64)] = -1
            add_idx = int(np.argmax(gains))
            candidate_coverage = base_coverage + int(gains[add_idx])
            if candidate_coverage > best_coverage:
                best_coverage = candidate_coverage
                best_swap = (int(removal_pos), remove_idx, add_idx, after_counts)

        if best_swap is None:
            break
        removal_pos, remove_idx, add_idx, after_counts = best_swap
        selected[removal_pos] = add_idx
        selected_set.remove(remove_idx)
        selected_set.add(add_idx)
        counts = after_counts
        counts[incidence[add_idx]] += 1
        covered_count = best_coverage

    # If the requested target is already met, remove tickets that add no required cover.
    if covered_count >= goal:
        changed = True
        while changed and selected:
            changed = False
            unique_loss = [int(np.sum(counts[incidence[idx]] == 1)) for idx in selected]
            for removal_pos in np.argsort(unique_loss, kind="stable"):
                remove_idx = selected[int(removal_pos)]
                loss = int(unique_loss[int(removal_pos)])
                if covered_count - loss < goal:
                    continue
                counts[incidence[remove_idx]] -= 1
                covered_count -= loss
                selected.pop(int(removal_pos))
                selected_set.remove(remove_idx)
                changed = True
                break

    return DesignSolution(
        method="COVER_GREEDY_LOCAL",
        ticket_indices=tuple(int(idx) for idx in selected),
        coverage_trace=initial.coverage_trace,
        targets_covered=int(covered_count),
        total_targets=problem.n_targets,
        iterations=int(iterations),
        initial_targets_covered=int(initial.targets_covered),
    )
