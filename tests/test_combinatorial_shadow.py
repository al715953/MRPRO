import itertools

import numpy as np

from src.strategies.combinatorial.shadow import (
    CoveringShadowSpec,
    build_covering_shadow_variant,
)


def test_covering_shadow_is_reproducible_valid_and_non_official():
    universe = np.asarray(
        list(itertools.combinations(range(1, 11), 6)), dtype=np.int16
    )
    snapshot = {
        "universe": universe,
        "hybrid_scores": np.linspace(1.0, 0.0, len(universe)),
    }
    spec = CoveringShadowSpec(
        key="toy_cover",
        label="Toy",
        candidate_pool_size=8,
        ticket_budget=10,
        rank_depth=30,
        primary_target_size=5,
        secondary_target_size=4,
        local_search_iterations=3,
    )

    first = build_covering_shadow_variant(
        snapshot, spec, total_balls=10, ticket_size=6
    )
    second = build_covering_shadow_variant(
        snapshot, spec, total_balls=10, ticket_size=6
    )

    assert first == second
    assert first["official"] is False
    assert len(first["tickets"]) == 10
    assert len({tuple(ticket) for ticket in first["tickets"]}) == 10
    candidates = set(first["metadata"]["candidate_numbers"])
    assert all(len(ticket) == len(set(ticket)) == 6 for ticket in first["tickets"])
    assert all(set(ticket).issubset(candidates) for ticket in first["tickets"])
    assert set(first["metadata"]["coverage_by_t"]) == {"4", "5"}
