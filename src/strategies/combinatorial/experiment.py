from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from typing import Any, Sequence

import numpy as np

from src.data_access.config import (
    BACKTEST_MODEL_FILE_PATH,
    BEST_SETTINGS,
    CSV_FILE_PATH,
)
from src.data_access.dataset_version import compute_dataset_version
from src.domain.dtos import DrawHistoryDTO, PredictionConfigDTO, sort_history_chronologically
from src.strategies.genetic_selector import GeneticSelectorStrategy
from src.strategies.universe_reduction import UniverseReductionStrategy

from .baselines import (
    iter_random_same_size,
    random_coverage_distribution,
    random_same_size_indices,
    score_against_random,
    summarize_distribution,
)
from .candidate_sets import (
    candidate_hit_count,
    explicit_candidate_set,
    mrpro_candidate_set_from_snapshot,
    oracle_candidate_set,
    random_candidate_set,
)
from .config import CoveringExperimentConfig
from .covering import CombinatorialProblem
from .greedy import DesignSolution, greedy_maximum_coverage
from .local_search import improve_by_local_search
from .metrics import coverage_metrics, max_ticket_hits, validate_ticket_matrix
from .multiobjective import (
    improve_weighted_local_search,
    weighted_greedy_maximum_coverage,
)
from .statistics import (
    exact_mcnemar,
    paired_block_bootstrap_ci,
    paired_sign_permutation_test,
)


METHOD_GREEDY = "COVER_GREEDY"
METHOD_LOCAL = "COVER_GREEDY_LOCAL"
METHOD_RANDOM = "RANDOM_SAME_SIZE"
METHOD_EXHAUSTIVE = "EXHAUSTIVE_REDUCED"
METHOD_CURRENT = "CURRENT_MRPRO"
METHOD_CURRENT_SAME_M = "CURRENT_MRPRO_SAME_M"
METHOD_CURRENT_RESTRICTED = "CURRENT_MRPRO_RESTRICTED"


@dataclass(frozen=True)
class DesignBundle:
    problem: CombinatorialProblem
    problems_by_t: dict[int, CombinatorialProblem]
    greedy: DesignSolution
    local: DesignSolution
    metrics: dict[str, dict[str, Any]]
    random: dict[str, Any]
    multiobjective: dict[str, Any] | None = None


def build_design_bundle(
    config: CoveringExperimentConfig,
    *,
    ticket_size: int,
) -> DesignBundle:
    explicit_max = max((int(value) for value in (config.explicit_candidates or ())), default=0)
    config.validate(
        total_balls=max(config.candidate_pool_size, ticket_size, explicit_max),
        ticket_size=ticket_size,
    )
    v = int(config.candidate_pool_size)
    t = config.resolved_target_size(ticket_size)
    canonical_numbers = tuple(range(1, v + 1))
    problem = CombinatorialProblem.build(
        canonical_numbers,
        ticket_size,
        t,
        max_candidate_tickets=config.max_candidate_tickets,
        max_target_subsets=config.max_target_subsets,
        max_incidences=config.max_incidences,
    )
    problems_by_t = {int(t): problem}
    secondary_t = config.secondary_target_subset_size
    weights = {int(t): 1.0}
    mixed_greedy = None
    mixed_local = None
    if secondary_t is not None:
        secondary_t = int(secondary_t)
        problems_by_t[secondary_t] = CombinatorialProblem.build(
            canonical_numbers,
            ticket_size,
            secondary_t,
            max_candidate_tickets=config.max_candidate_tickets,
            max_target_subsets=config.max_target_subsets,
            max_incidences=config.max_incidences,
        )
        weight_total = float(
            config.primary_target_weight + config.secondary_target_weight
        )
        weights = {
            int(t): float(config.primary_target_weight / weight_total),
            secondary_t: float(config.secondary_target_weight / weight_total),
        }
        mixed_greedy = weighted_greedy_maximum_coverage(
            problems_by_t,
            weights,
            config.ticket_budget,
            config.coverage_target,
        )
        mixed_local = improve_weighted_local_search(
            problems_by_t,
            weights,
            mixed_greedy,
            max_iterations=config.local_search_iterations,
        )
        greedy = mixed_greedy.solution
        local = mixed_local.solution
    else:
        greedy = greedy_maximum_coverage(
            problem,
            config.ticket_budget,
            config.coverage_target,
        )
        local = improve_by_local_search(
            problem,
            greedy,
            max_iterations=config.local_search_iterations,
            coverage_target=config.coverage_target,
        )

    def metrics_for(indices) -> dict[str, Any]:
        primary = coverage_metrics(problem, indices)
        coverage_by_t = {
            str(target_size): coverage_metrics(target_problem, indices)["coverage_t"]
            for target_size, target_problem in problems_by_t.items()
        }
        primary["coverage_by_t"] = coverage_by_t
        primary["weighted_coverage"] = float(
            sum(weights[target_size] * coverage_by_t[str(target_size)] for target_size in problems_by_t)
        )
        return primary

    metrics = {
        METHOD_GREEDY: metrics_for(greedy.ticket_indices),
        METHOD_LOCAL: metrics_for(local.ticket_indices),
        METHOD_EXHAUSTIVE: metrics_for(np.arange(problem.n_tickets, dtype=np.int64)),
    }
    random = {}
    for method in (METHOD_GREEDY, METHOD_LOCAL):
        ticket_count = metrics[method]["ticket_count"]
        distribution = random_coverage_distribution(
            problem,
            ticket_count,
            config.random_trials,
            config.random_seed + (0 if method == METHOD_GREEDY else 1),
        )
        if len(problems_by_t) > 1:
            raw_by_t = {str(target_size): [] for target_size in problems_by_t}
            raw_weighted = []
            for indices in iter_random_same_size(
                problem.n_tickets,
                ticket_count,
                config.random_trials,
                config.random_seed + (0 if method == METHOD_GREEDY else 1),
            ):
                coverage_by_t = {
                    target_size: coverage_metrics(target_problem, indices)["coverage_t"]
                    for target_size, target_problem in problems_by_t.items()
                }
                for target_size, value in coverage_by_t.items():
                    raw_by_t[str(target_size)].append(float(value))
                raw_weighted.append(
                    float(
                        sum(
                            weights[target_size] * coverage_by_t[target_size]
                            for target_size in problems_by_t
                        )
                    )
                )
            distribution["coverage_by_t"] = {
                target_size: summarize_distribution(values)
                for target_size, values in raw_by_t.items()
            }
            distribution["raw_coverage_by_t"] = raw_by_t
            distribution["weighted_coverage"] = summarize_distribution(raw_weighted)
            distribution["raw_weighted_coverage"] = raw_weighted
            distribution["optimized_comparison_weighted"] = score_against_random(
                metrics[method]["weighted_coverage"], raw_weighted
            )
        optimized_coverage = metrics[method]["coverage_t"]
        random_mean = distribution["coverage"]["mean"] or 0.0
        distribution["optimized_comparison"] = {
            **score_against_random(
                optimized_coverage,
                distribution["raw_coverage"],
            ),
            "coverage_gain": float(optimized_coverage - random_mean),
            "coverage_lift": (
                float(optimized_coverage / random_mean) if random_mean > 0 else None
            ),
        }
        random[method] = distribution
    multiobjective = None
    if mixed_greedy is not None and mixed_local is not None:
        multiobjective = {
            "target_weights": {str(key): value for key, value in weights.items()},
            "greedy_weighted_coverage": mixed_greedy.weighted_coverage,
            "local_weighted_coverage": mixed_local.weighted_coverage,
            "greedy_objective_trace": list(mixed_greedy.objective_trace),
            "local_objective_trace": list(mixed_local.objective_trace),
        }
    return DesignBundle(
        problem,
        problems_by_t,
        greedy,
        local,
        metrics,
        random,
        multiobjective,
    )


def _map_solution_tickets(
    problem: CombinatorialProblem,
    solution: DesignSolution,
    candidate_numbers: Sequence[int],
) -> np.ndarray:
    positions = problem.ticket_positions[np.asarray(solution.ticket_indices, dtype=np.int64)]
    numbers = np.asarray(candidate_numbers, dtype=np.int16)
    tickets = np.sort(numbers[positions], axis=1)
    validate_ticket_matrix(tickets, candidate_numbers, problem.k)
    return tickets


def _map_all_tickets(
    problem: CombinatorialProblem,
    candidate_numbers: Sequence[int],
) -> np.ndarray:
    numbers = np.asarray(candidate_numbers, dtype=np.int16)
    return np.sort(numbers[problem.ticket_positions], axis=1)


def _candidate_rng(seed: int, contest: int, stream: int) -> np.random.Generator:
    return np.random.default_rng(np.random.SeedSequence([int(seed), int(contest), int(stream)]))


def _choose_candidates(
    config: CoveringExperimentConfig,
    *,
    winning: Sequence[int],
    contest: int,
    total_balls: int,
    mrpro_snapshot: dict | None,
) -> tuple[int, ...]:
    v = int(config.candidate_pool_size)
    method = config.candidate_method
    if method == "oracle_candidate_set":
        return oracle_candidate_set(
            winning,
            v=v,
            total_balls=total_balls,
            rng=_candidate_rng(config.random_seed, contest, 101),
        )
    if method == "random_candidate_set":
        return random_candidate_set(
            v=v,
            total_balls=total_balls,
            rng=_candidate_rng(config.random_seed, contest, 102),
        )
    if method == "explicit_candidate_set":
        return explicit_candidate_set(
            config.explicit_candidates or (),
            v=v,
            total_balls=total_balls,
        )
    if method == "mrpro_candidate_set":
        if mrpro_snapshot is None:
            raise ValueError("mrpro_candidate_set requiere snapshot MRPRO")
        return mrpro_candidate_set_from_snapshot(
            mrpro_snapshot,
            v=v,
            total_balls=total_balls,
            rank_depth=config.candidate_rank_depth,
        )
    raise ValueError(f"candidate_method desconocido: {method}")


def _summarize_max_hits(values, *, ticket_count: int, ticket_size: int) -> dict:
    array = np.asarray(values, dtype=np.int16)
    distribution = {
        str(hit): int(np.sum(array == hit)) for hit in range(int(ticket_size) + 1)
    }
    draws = int(array.size)
    summary = {
        "draws": draws,
        "ticket_count": int(ticket_count),
        "max_hits_distribution": distribution,
        "avg_max_hits": float(np.mean(array)) if draws else None,
    }
    thresholds = sorted(
        {
            max(1, int(ticket_size) - 3),
            max(1, int(ticket_size) - 2),
            max(1, int(ticket_size) - 1),
        }
    )
    for threshold in thresholds:
        rate = float(np.mean(array >= threshold)) if draws else None
        summary[f"hit_rate_ge_{threshold}"] = rate
        summary[f"hit_rate_ge_{threshold}_per_ticket"] = (
            float(rate / ticket_count) if rate is not None and ticket_count else None
        )
    jackpot_rate = float(np.mean(array == ticket_size)) if draws else None
    summary[f"hit_rate_eq_{ticket_size}"] = jackpot_rate
    summary[f"hit_rate_eq_{ticket_size}_per_ticket"] = (
        float(jackpot_rate / ticket_count)
        if jackpot_rate is not None and ticket_count
        else None
    )
    return summary


def _random_historical_summary(
    random_matrix: np.ndarray,
    *,
    ticket_count: int,
    ticket_size: int,
) -> dict:
    thresholds = sorted(
        {
            max(1, int(ticket_size) - 3),
            max(1, int(ticket_size) - 2),
            max(1, int(ticket_size) - 1),
        }
    )
    trial_metrics: dict[str, list[float]] = {"avg_max_hits": []}
    trial_metrics.update({f"hit_rate_ge_{value}": [] for value in thresholds})
    trial_metrics[f"hit_rate_eq_{ticket_size}"] = []
    for trial in random_matrix:
        trial_metrics["avg_max_hits"].append(float(np.mean(trial)))
        for threshold in thresholds:
            trial_metrics[f"hit_rate_ge_{threshold}"].append(
                float(np.mean(trial >= threshold))
            )
        trial_metrics[f"hit_rate_eq_{ticket_size}"].append(
            float(np.mean(trial == ticket_size))
        )
    return {
        "draws": int(random_matrix.shape[1]) if random_matrix.ndim == 2 else 0,
        "trials": int(random_matrix.shape[0]) if random_matrix.ndim == 2 else 0,
        "ticket_count": int(ticket_count),
        "metric_distributions": {
            metric: summarize_distribution(values)
            for metric, values in trial_metrics.items()
        },
        "raw_trial_metrics": trial_metrics,
    }


def _conditional_summary(
    candidate_hits: np.ndarray,
    deterministic: dict[str, np.ndarray],
    random_matrix: np.ndarray,
    ticket_size: int,
) -> dict:
    output = {}
    near = max(1, int(ticket_size) - 1)
    secondary = max(1, int(ticket_size) - 2)
    for hit_count in sorted(set(int(value) for value in candidate_hits)):
        mask = candidate_hits == hit_count
        row = {"draws": int(np.sum(mask)), "methods": {}}
        for method, values in deterministic.items():
            selected = values[mask]
            row["methods"][method] = {
                f"rate_eq_{ticket_size}": float(np.mean(selected == ticket_size)),
                f"rate_ge_{near}": float(np.mean(selected >= near)),
                f"rate_ge_{secondary}": float(np.mean(selected >= secondary)),
                "avg_max_hits": float(np.mean(selected)),
            }
        random_selected = random_matrix[:, mask]
        row["methods"][METHOD_RANDOM] = {
            f"rate_eq_{ticket_size}": float(np.mean(random_selected == ticket_size)),
            f"rate_ge_{near}": float(np.mean(random_selected >= near)),
            f"rate_ge_{secondary}": float(np.mean(random_selected >= secondary)),
            "avg_max_hits": float(np.mean(random_selected)),
        }
        output[str(hit_count)] = row
    return output


def _statistical_comparisons(
    deterministic: dict[str, np.ndarray],
    random_matrix: np.ndarray,
    config: CoveringExperimentConfig,
    ticket_size: int,
) -> dict:
    comparisons = {}
    reference_methods = [
        method
        for method in (
            METHOD_CURRENT_RESTRICTED,
            METHOD_CURRENT_SAME_M,
            METHOD_CURRENT,
        )
        if method in deterministic
    ]
    for optimized in (METHOD_GREEDY, METHOD_LOCAL):
        left = deterministic[optimized]
        near = max(1, int(ticket_size) - 1)
        secondary = max(1, int(ticket_size) - 2)
        comparisons[optimized] = {"vs_random_distribution": {}}
        random_metrics = {
            "avg_max_hits": np.mean(random_matrix, axis=1),
            f"hit_rate_ge_{secondary}": np.mean(random_matrix >= secondary, axis=1),
            f"hit_rate_ge_{near}": np.mean(random_matrix >= near, axis=1),
            f"hit_rate_eq_{ticket_size}": np.mean(
                random_matrix == ticket_size, axis=1
            ),
        }
        optimized_metrics = {
            "avg_max_hits": float(np.mean(left)),
            f"hit_rate_ge_{secondary}": float(np.mean(left >= secondary)),
            f"hit_rate_ge_{near}": float(np.mean(left >= near)),
            f"hit_rate_eq_{ticket_size}": float(np.mean(left == ticket_size)),
        }
        for metric, value in optimized_metrics.items():
            comparisons[optimized]["vs_random_distribution"][metric] = {
                "optimized": value,
                **score_against_random(value, random_metrics[metric]),
            }
        comparisons[optimized]["vs_random_trial_0"] = {
            f"mcnemar_ge_{near}": exact_mcnemar(
                left >= near, random_matrix[0] >= near
            ),
            "permutation_max_hits": paired_sign_permutation_test(
                left,
                random_matrix[0],
                trials=config.permutation_trials,
                seed=config.random_seed + 300,
            ),
        }
        for reference in reference_methods:
            right = deterministic[reference]
            comparisons[optimized][f"vs_{reference}"] = {
                f"mcnemar_ge_{secondary}": exact_mcnemar(
                    left >= secondary, right >= secondary
                ),
                f"mcnemar_ge_{near}": exact_mcnemar(left >= near, right >= near),
                "permutation_max_hits": paired_sign_permutation_test(
                    left,
                    right,
                    trials=config.permutation_trials,
                    seed=config.random_seed + 301,
                ),
                "bootstrap_max_hits": paired_block_bootstrap_ci(
                    left,
                    right,
                    seed=config.random_seed + 302,
                ),
            }
    return comparisons


def _temporal_fold_summaries(
    *,
    per_draw: list[dict[str, Any]],
    candidate_hits: np.ndarray,
    deterministic: dict[str, np.ndarray],
    method_counts: dict[str, list[int]],
    random_matrix: np.ndarray,
    folds: int,
    ticket_size: int,
) -> list[dict[str, Any]]:
    """Summarize contiguous evaluation folds without retuning on any fold."""

    if not per_draw:
        return []
    split_count = min(max(1, int(folds)), len(per_draw))
    output = []
    for fold_idx, indices in enumerate(np.array_split(np.arange(len(per_draw)), split_count)):
        if indices.size == 0:
            continue
        fold_methods = {}
        for method, values in deterministic.items():
            counts = np.asarray(method_counts[method], dtype=np.int32)[indices]
            fold_methods[method] = _summarize_max_hits(
                values[indices],
                ticket_count=int(round(float(np.mean(counts)))),
                ticket_size=ticket_size,
            )
        fold_candidate_hits = candidate_hits[indices]
        output.append(
            {
                "fold": int(fold_idx + 1),
                "role": (
                    "holdout_test" if fold_idx == split_count - 1 else "walk_forward_evaluation"
                ),
                "draw_range": [
                    int(per_draw[int(indices[0])]["draw_id"]),
                    int(per_draw[int(indices[-1])]["draw_id"]),
                ],
                "draws": int(indices.size),
                "candidate_hit_distribution": {
                    str(hit): int(np.sum(fold_candidate_hits == hit))
                    for hit in range(ticket_size + 1)
                },
                "methods": fold_methods,
                "random_same_size": _random_historical_summary(
                    random_matrix[:, indices],
                    ticket_count=int(
                        round(
                            float(
                                np.mean(
                                    np.asarray(method_counts[METHOD_GREEDY], dtype=np.int32)[
                                        indices
                                    ]
                                )
                            )
                        )
                    ),
                    ticket_size=ticket_size,
                ),
            }
        )
    return output


def run_historical_experiment(
    history: DrawHistoryDTO,
    config: CoveringExperimentConfig,
    *,
    total_balls: int,
    ticket_size: int,
    design_bundle: DesignBundle | None = None,
) -> dict[str, Any]:
    """Walk-forward comparison; every predictive draw sees only earlier history."""

    config.validate(total_balls, ticket_size)
    bundle = design_bundle or build_design_bundle(config, ticket_size=ticket_size)
    chronological = sort_history_chronologically(history)
    rows = list(
        zip(
            chronological.dates,
            chronological.winning_numbers,
            chronological.concursos,
        )
    )
    start = max(0, len(rows) - int(config.backtest_draws))
    needs_mrpro = bool(
        config.include_current_mrpro or config.candidate_method == "mrpro_candidate_set"
    )
    selector = GeneticSelectorStrategy(model_path=BACKTEST_MODEL_FILE_PATH) if needs_mrpro else None
    reducer = UniverseReductionStrategy() if needs_mrpro else None
    training_cutoff = selector.training_cutoff_contest if selector is not None else None
    if training_cutoff is not None and needs_mrpro:
        start = max(
            start,
            next(
                (
                    idx
                    for idx, (_, _, contest) in enumerate(rows)
                    if int(contest) > int(training_cutoff)
                ),
                len(rows),
            ),
        )
    if start >= len(rows):
        raise ValueError("No hay sorteos fuera de muestra disponibles para el experimento")

    method_values: dict[str, list[int]] = {
        METHOD_GREEDY: [],
        METHOD_LOCAL: [],
        METHOD_EXHAUSTIVE: [],
    }
    if config.include_current_mrpro:
        method_values[METHOD_CURRENT] = []
        method_values[METHOD_CURRENT_SAME_M] = []
        method_values[METHOD_CURRENT_RESTRICTED] = []
    method_counts_values: dict[str, list[int]] = {
        method: [] for method in method_values
    }
    candidate_hits_values = []
    per_draw = []
    random_by_draw = []
    greedy_count = len(bundle.greedy.ticket_indices)

    for row_idx in range(start, len(rows)):
        _, target_full, contest = rows[row_idx]
        target = [int(number) for number in target_full[:ticket_size]]
        past_rows = rows[:row_idx]
        past = DrawHistoryDTO(
            dates=[row[0] for row in past_rows],
            winning_numbers=[row[1] for row in past_rows],
            concursos=[row[2] for row in past_rows],
        )
        native_prediction = None
        same_m_prediction = None
        native_snapshot = None
        if needs_mrpro:
            settings = dict(BEST_SETTINGS)
            settings["seed"] = int(config.random_seed)
            current_config = PredictionConfigDTO(
                total_balls,
                ticket_size,
                int(greedy_count),
                filter_overrides=settings,
            )
            reduced = reducer.predict(past, current_config, verbose=False)
            current_config.raw_universe_ptr = reduced.metadata.get("raw_ndarray")
            same_m_prediction = selector.predict(past, current_config)
            native_snapshot = same_m_prediction.metadata
            if int(config.current_mrpro_ticket_count) == int(greedy_count):
                native_prediction = same_m_prediction
            else:
                native_config = PredictionConfigDTO(
                    total_balls,
                    ticket_size,
                    int(config.current_mrpro_ticket_count),
                    filter_overrides=settings,
                )
                native_config.raw_universe_ptr = current_config.raw_universe_ptr
                native_prediction = selector.predict(past, native_config)

        candidates = _choose_candidates(
            config,
            winning=target,
            contest=int(contest),
            total_balls=total_balls,
            mrpro_snapshot=native_snapshot,
        )
        candidate_hits = candidate_hit_count(candidates, target)
        candidate_hits_values.append(candidate_hits)
        greedy_tickets = _map_solution_tickets(bundle.problem, bundle.greedy, candidates)
        local_tickets = _map_solution_tickets(bundle.problem, bundle.local, candidates)
        all_reduced_tickets = _map_all_tickets(bundle.problem, candidates)

        draw_methods = {
            METHOD_GREEDY: max_ticket_hits(greedy_tickets, target),
            METHOD_LOCAL: max_ticket_hits(local_tickets, target),
            METHOD_EXHAUSTIVE: min(candidate_hits, ticket_size),
        }
        draw_ticket_counts = {
            METHOD_GREEDY: int(len(greedy_tickets)),
            METHOD_LOCAL: int(len(local_tickets)),
            METHOD_EXHAUSTIVE: int(bundle.problem.n_tickets),
        }
        if config.include_current_mrpro and native_prediction is not None:
            draw_methods[METHOD_CURRENT] = max_ticket_hits(
                np.asarray(native_prediction.tickets, dtype=np.int16), target
            )
            draw_methods[METHOD_CURRENT_SAME_M] = max_ticket_hits(
                np.asarray(same_m_prediction.tickets, dtype=np.int16), target
            )
            draw_ticket_counts[METHOD_CURRENT] = int(len(native_prediction.tickets))
            draw_ticket_counts[METHOD_CURRENT_SAME_M] = int(
                len(same_m_prediction.tickets)
            )
            restricted_settings = dict(BEST_SETTINGS)
            restricted_settings["seed"] = int(config.random_seed)
            restricted_config = PredictionConfigDTO(
                total_balls,
                ticket_size,
                int(greedy_count),
                filter_overrides=restricted_settings,
            )
            restricted_config.raw_universe_ptr = all_reduced_tickets
            restricted_prediction = selector.predict(past, restricted_config)
            draw_methods[METHOD_CURRENT_RESTRICTED] = max_ticket_hits(
                np.asarray(restricted_prediction.tickets, dtype=np.int16), target
            )
            draw_ticket_counts[METHOD_CURRENT_RESTRICTED] = int(
                len(restricted_prediction.tickets)
            )

        for method, value in draw_methods.items():
            method_values[method].append(int(value))
            method_counts_values[method].append(int(draw_ticket_counts[method]))

        trial_hits = np.empty(int(config.random_trials), dtype=np.int8)
        rng = _candidate_rng(config.random_seed, int(contest), 201)
        for trial in range(int(config.random_trials)):
            indices = random_same_size_indices(
                bundle.problem.n_tickets,
                greedy_count,
                rng=rng,
            )
            random_tickets = all_reduced_tickets[indices]
            trial_hits[trial] = max_ticket_hits(random_tickets, target)
        random_by_draw.append(trial_hits)
        per_draw.append(
            {
                "draw_id": int(contest),
                "history_rows_visible": int(row_idx),
                "candidate_numbers": [int(number) for number in candidates],
                "candidate_hits": int(candidate_hits),
                "max_hits": {method: int(value) for method, value in draw_methods.items()},
                "ticket_counts": {
                    method: int(value) for method, value in draw_ticket_counts.items()
                },
            }
        )

    deterministic_arrays = {
        method: np.asarray(values, dtype=np.int8)
        for method, values in method_values.items()
    }
    random_matrix = np.asarray(random_by_draw, dtype=np.int8).T
    method_summary = {}
    for method, values in deterministic_arrays.items():
        observed_counts = np.asarray(method_counts_values[method], dtype=np.int32)
        representative_count = int(round(float(np.mean(observed_counts))))
        summary = _summarize_max_hits(
            values,
            ticket_count=representative_count,
            ticket_size=ticket_size,
        )
        summary.update(
            {
                "ticket_count_mean": float(np.mean(observed_counts)),
                "ticket_count_min": int(np.min(observed_counts)),
                "ticket_count_max": int(np.max(observed_counts)),
            }
        )
        method_summary[method] = summary
    random_summary = _random_historical_summary(
        random_matrix,
        ticket_count=greedy_count,
        ticket_size=ticket_size,
    )
    candidate_array = np.asarray(candidate_hits_values, dtype=np.int8)
    candidate_distribution = {
        str(hit): int(np.sum(candidate_array == hit))
        for hit in range(ticket_size + 1)
    }
    dataset = compute_dataset_version(CSV_FILE_PATH)

    temporal_folds = _temporal_fold_summaries(
        per_draw=per_draw,
        candidate_hits=candidate_array,
        deterministic=deterministic_arrays,
        method_counts=method_counts_values,
        random_matrix=random_matrix,
        folds=config.temporal_folds,
        ticket_size=ticket_size,
    )

    return {
        "candidate_method": config.candidate_method,
        "predictive_claim_allowed": config.candidate_method not in {
            "oracle_candidate_set"
        },
        "oracle_warning": (
            "CONTROL NO PREDICTIVO: el ganador fue insertado en V."
            if config.candidate_method == "oracle_candidate_set"
            else None
        ),
        "dataset": dataset,
        "training_cutoff_contest": training_cutoff,
        "draw_range": [int(rows[start][2]), int(rows[-1][2])],
        "draws": len(per_draw),
        "candidate_hit_distribution": candidate_distribution,
        "methods": method_summary,
        "random_same_size": random_summary,
        "conditional_by_candidate_hits": _conditional_summary(
            candidate_array,
            deterministic_arrays,
            random_matrix,
            ticket_size,
        ),
        "statistical_comparison": _statistical_comparisons(
            deterministic_arrays,
            random_matrix,
            config,
            ticket_size,
        ),
        "temporal_folds": temporal_folds,
        "temporal_fold_warning": (
            "Folds contiguos de evaluación; no se ajustaron pesos ni hiperparámetros "
            "con sus resultados."
        ),
        "per_draw": per_draw,
    }


def design_bundle_to_dict(bundle: DesignBundle) -> dict[str, Any]:
    payload = {
        "problem": bundle.problem.estimate.to_dict(),
        "methods": bundle.metrics,
        "local_search": {
            "initial_targets_covered": bundle.local.initial_targets_covered,
            "final_targets_covered": bundle.local.targets_covered,
            "coverage_improvement": float(bundle.local.coverage - bundle.greedy.coverage),
            "ticket_reduction": int(
                len(bundle.greedy.ticket_indices) - len(bundle.local.ticket_indices)
            ),
            "iterations": bundle.local.iterations,
        },
        "greedy_coverage_trace": [int(value) for value in bundle.greedy.coverage_trace],
        "random_same_size": bundle.random,
    }
    if bundle.multiobjective is not None:
        payload["multiobjective"] = bundle.multiobjective
        payload["problems_by_t"] = {
            str(target_size): problem.estimate.to_dict()
            for target_size, problem in bundle.problems_by_t.items()
        }
    return payload
