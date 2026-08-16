import itertools

import numpy as np

from src.domain.dtos import DrawHistoryDTO, PredictionResultDTO
from src.strategies.combinatorial import experiment as experiment_module
from src.strategies.combinatorial.config import CoveringExperimentConfig
from src.strategies.combinatorial.experiment import (
    METHOD_EXHAUSTIVE,
    METHOD_GREEDY,
    METHOD_LOCAL,
    build_design_bundle,
    run_historical_experiment,
)


def _toy_history():
    return DrawHistoryDTO(
        dates=[f"d{idx}" for idx in range(6)],
        concursos=[101, 102, 103, 104, 105, 106],
        winning_numbers=[
            [1, 2, 3, 4],
            [2, 3, 4, 5],
            [3, 4, 5, 6],
            [4, 5, 6, 7],
            [1, 3, 5, 8],
            [2, 4, 6, 8],
        ],
    )


def test_oracle_historical_experiment_is_explicit_and_walk_forward():
    config = CoveringExperimentConfig(
        candidate_pool_size=6,
        target_subset_size=3,
        ticket_budget=5,
        random_trials=12,
        random_seed=11,
        local_search_iterations=10,
        candidate_method="oracle_candidate_set",
        backtest_draws=3,
        include_current_mrpro=False,
        permutation_trials=100,
    )
    bundle = build_design_bundle(config, ticket_size=4)

    result = run_historical_experiment(
        _toy_history(),
        config,
        total_balls=8,
        ticket_size=4,
        design_bundle=bundle,
    )

    assert result["oracle_warning"].startswith("CONTROL NO PREDICTIVO")
    assert result["predictive_claim_allowed"] is False
    assert result["draw_range"] == [104, 106]
    assert result["candidate_hit_distribution"]["4"] == 3
    assert [row["history_rows_visible"] for row in result["per_draw"]] == [3, 4, 5]
    assert all(
        set(_toy_history().winning_numbers[idx]).issubset(
            result["per_draw"][idx - 3]["candidate_numbers"]
        )
        for idx in range(3, 6)
    )
    assert result["methods"][METHOD_EXHAUSTIVE]["hit_rate_eq_4"] == 1.0
    assert result["random_same_size"]["trials"] == 12


def test_random_candidates_covering_and_random_use_same_ticket_count():
    config = CoveringExperimentConfig(
        candidate_pool_size=6,
        target_subset_size=2,
        ticket_budget=7,
        random_trials=8,
        random_seed=22,
        candidate_method="random_candidate_set",
        backtest_draws=2,
        include_current_mrpro=False,
        permutation_trials=50,
    )
    bundle = build_design_bundle(config, ticket_size=4)

    result = run_historical_experiment(
        _toy_history(),
        config,
        total_balls=8,
        ticket_size=4,
        design_bundle=bundle,
    )

    greedy_count = result["methods"][METHOD_GREEDY]["ticket_count"]
    assert 0 < greedy_count <= 7
    assert result["methods"][METHOD_LOCAL]["ticket_count"] <= greedy_count
    assert result["random_same_size"]["ticket_count"] == greedy_count
    assert result["random_same_size"]["draws"] == 2
    assert result["predictive_claim_allowed"] is True


def test_mrpro_adapter_never_receives_target_or_future_history(monkeypatch):
    histories_seen = []
    universe = np.asarray(list(itertools.combinations(range(1, 9), 4)), dtype=np.int16)

    class ReducerStub:
        def predict(self, history, config, verbose=False):
            histories_seen.append(("reducer", tuple(history.concursos)))
            return PredictionResultDTO(
                strategy_name="reducer",
                tickets=[],
                metadata={"raw_ndarray": universe},
            )

    class SelectorStub:
        def __init__(self, model_path=None):
            self.training_cutoff_contest = 103

        def predict(self, history, config):
            histories_seen.append(("selector", tuple(history.concursos)))
            raw = np.asarray(config.raw_universe_ptr)
            count = min(int(config.num_tickets), len(raw))
            return PredictionResultDTO(
                strategy_name="selector",
                tickets=raw[:count].tolist(),
                metadata={
                    "universe": raw,
                    "hybrid_scores": np.linspace(0.0, 1.0, len(raw)),
                },
            )

    monkeypatch.setattr(experiment_module, "UniverseReductionStrategy", ReducerStub)
    monkeypatch.setattr(experiment_module, "GeneticSelectorStrategy", SelectorStub)
    config = CoveringExperimentConfig(
        candidate_pool_size=6,
        target_subset_size=3,
        ticket_budget=5,
        current_mrpro_ticket_count=2,
        random_trials=3,
        random_seed=33,
        candidate_method="mrpro_candidate_set",
        backtest_draws=6,
        include_current_mrpro=True,
        permutation_trials=20,
    )
    bundle = build_design_bundle(config, ticket_size=4)

    result = run_historical_experiment(
        _toy_history(),
        config,
        total_balls=8,
        ticket_size=4,
        design_bundle=bundle,
    )

    assert result["draw_range"] == [104, 106]
    expected_histories = [
        (101, 102, 103),
        (101, 102, 103, 104),
        (101, 102, 103, 104, 105),
    ]
    reducer_histories = [history for kind, history in histories_seen if kind == "reducer"]
    selector_histories = [history for kind, history in histories_seen if kind == "selector"]
    assert reducer_histories == expected_histories
    assert selector_histories == [
        history for history in expected_histories for _ in range(3)
    ]
