import itertools

import numpy as np

from src.strategies.genetic.fitness import (
    DeepDispersionConfig,
    EliteCoverageDeepConfig,
    select_core_plus_deep_tickets,
    select_elite_coverage_deep_tickets,
    select_tickets_v16,
)
from src.strategies.genetic_selector import GeneticSelectorStrategy


def test_selector_defaults_preserve_current_official_rank_plan():
    fitness, strata = GeneticSelectorStrategy._selection_configs({})

    assert fitness.focus_max_rank == 200
    assert fitness.candidate_max_rank == 500
    assert fitness.bucket_plan[-1] == (201, 500, 1)
    assert strata.rank_edges == (10, 30, 60, 100, 150, 200, 500)


def test_ticket_subset_coverage_counts_unique_pairs_triples_and_quads():
    metrics = GeneticSelectorStrategy._ticket_subset_coverage(
        [[1, 2, 3, 4, 5, 6], [1, 2, 3, 4, 5, 7]]
    )

    assert metrics == {
        "selected_unique_pairs": 20,
        "selected_unique_triples": 30,
        "selected_unique_quads": 25,
    }


def test_selector_accepts_reproducible_deep_rank_plan():
    fitness, strata = GeneticSelectorStrategy._selection_configs(
        {
            "fitness_focus_max_rank": 5000,
            "fitness_candidate_max_rank": 5000,
            "fitness_rank_edges": [5, 20, 100, 5000],
            "fitness_bucket_plan": [[6, 20, 2], [3001, 5000, 2]],
        }
    )

    assert fitness.focus_max_rank == 5000
    assert fitness.candidate_max_rank == 5000
    assert fitness.bucket_plan == ((6, 20, 2), (3001, 5000, 2))
    assert strata.rank_edges == (5, 20, 100, 5000)


def test_deep_rank_plan_keeps_budget_unique_and_selects_beyond_rank_500():
    candidates = np.asarray(
        list(itertools.islice(itertools.combinations(range(1, 40), 6), 600)),
        dtype=np.uint8,
    )
    scores = np.linspace(1.0, 0.0, len(candidates), dtype=np.float32)
    fitness, strata = GeneticSelectorStrategy._selection_configs(
        {
            "fitness_focus_max_rank": 600,
            "fitness_candidate_max_rank": 600,
            "fitness_rank_edges": [5, 20, 100, 300, 500, 600],
            "fitness_bucket_plan": [
                [6, 20, 4],
                [21, 100, 4],
                [101, 300, 4],
                [301, 500, 4],
                [501, 600, 3],
            ],
        }
    )

    tickets, debug = select_tickets_v16(
        candidates,
        scores,
        n_tickets=24,
        xp=np,
        cfg=fitness,
        strata=strata,
    )

    assert len(tickets) == 24
    assert len({tuple(ticket) for ticket in tickets}) == 24
    assert max(debug["selected_ranks"]) > 500


def test_core_plus_deep_keeps_native_20_and_spreads_10_equal_population_bands():
    rng = np.random.default_rng(20260821)
    candidates = np.unique(
        np.asarray(
            [
                sorted(rng.choice(np.arange(1, 40), size=6, replace=False))
                for _ in range(3500)
            ],
            dtype=np.uint8,
        ),
        axis=0,
    )[:3000]
    scores = np.linspace(1.0, 0.0, len(candidates), dtype=np.float32)
    fitness, strata = GeneticSelectorStrategy._selection_configs({})
    native_core, _ = select_tickets_v16(
        candidates,
        scores,
        n_tickets=20,
        xp=np,
        cfg=fitness,
        strata=strata,
    )

    first, debug = select_core_plus_deep_tickets(
        candidates,
        scores,
        n_tickets=30,
        xp=np,
        cfg=fitness,
        strata=strata,
        deep_cfg=DeepDispersionConfig(),
    )
    second, second_debug = select_core_plus_deep_tickets(
        candidates,
        scores,
        n_tickets=30,
        xp=np,
        cfg=fitness,
        strata=strata,
        deep_cfg=DeepDispersionConfig(),
    )

    assert first == second
    assert debug == second_debug
    assert first[:20] == native_core
    assert len(first) == len({tuple(ticket) for ticket in first}) == 30
    assert len(debug["core_selected_ranks"]) == 20
    assert len(debug["deep_selected_ranks"]) == 10
    assert min(debug["deep_selected_ranks"]) >= 501
    assert max(debug["deep_selected_ranks"]) <= len(candidates)
    assert len(debug["deep_rank_bands"]) == 10
    for band in debug["deep_rank_bands"]:
        assert band["rank_min"] <= band["chosen_rank"] <= band["rank_max"]


def test_deep_dispersion_overrides_are_validated():
    config = GeneticSelectorStrategy._deep_dispersion_config(
        {
            "deep_dispersion_core_tickets": 20,
            "deep_dispersion_tickets": 10,
            "deep_dispersion_min_rank": 501,
            "deep_dispersion_max_overlap": 99,
        }
    )

    assert config.core_tickets == 20
    assert config.deep_tickets == 10
    assert config.min_deep_rank == 501
    assert config.max_overlap_preferred == 6


def test_elite_coverage_deep_forces_elites_and_preserves_three_zones():
    rng = np.random.default_rng(20260822)
    candidates = np.unique(
        np.asarray(
            [
                sorted(rng.choice(np.arange(1, 40), size=6, replace=False))
                for _ in range(4000)
            ],
            dtype=np.uint8,
        ),
        axis=0,
    )[:3000]
    scores = np.linspace(1.0, 0.0, len(candidates), dtype=np.float32)
    config = EliteCoverageDeepConfig(
        elite_tickets=10,
        coverage_tickets=10,
        deep_tickets=10,
    )

    first, debug = select_elite_coverage_deep_tickets(
        candidates, scores, n_tickets=30, xp=np, portfolio_cfg=config
    )
    second, second_debug = select_elite_coverage_deep_tickets(
        candidates, scores, n_tickets=30, xp=np, portfolio_cfg=config
    )

    assert first == second
    assert debug == second_debug
    assert len(first) == len({tuple(ticket) for ticket in first}) == 30
    assert debug["elite_selected_ranks"] == list(range(1, 11))
    assert debug["phase_by_ticket"] == ["elite"] * 10 + ["coverage"] * 10 + [
        "deep"
    ] * 10
    assert min(debug["deep_selected_ranks"]) >= 501
    assert len(debug["deep_rank_bands"]) == 10
    assert debug["coverage_unique_pairs"] > 0
    assert debug["coverage_unique_triples"] > 0
    assert debug["coverage_unique_quads"] > 0


def test_elite_coverage_deep_overrides_are_validated():
    config = GeneticSelectorStrategy._elite_coverage_deep_config(
        {
            "portfolio_elite_tickets": 15,
            "portfolio_coverage_tickets": 10,
            "portfolio_deep_tickets": 5,
            "portfolio_max_overlap": 99,
            "portfolio_quad_novelty_weight": -2,
        }
    )

    assert config.elite_tickets == 15
    assert config.coverage_tickets == 10
    assert config.deep_tickets == 5
    assert config.max_overlap_preferred == 6
    assert config.w_quad_novelty == 0.0
