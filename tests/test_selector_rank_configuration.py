import itertools

import numpy as np

from src.strategies.genetic.fitness import select_tickets_v16
from src.strategies.genetic_selector import GeneticSelectorStrategy


def test_selector_defaults_preserve_current_official_rank_plan():
    fitness, strata = GeneticSelectorStrategy._selection_configs({})

    assert fitness.focus_max_rank == 200
    assert fitness.candidate_max_rank == 500
    assert fitness.bucket_plan[-1] == (201, 500, 1)
    assert strata.rank_edges == (10, 30, 60, 100, 150, 200, 500)


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
