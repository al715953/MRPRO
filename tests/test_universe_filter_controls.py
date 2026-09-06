import itertools

import numpy as np

from src.data_access.config import BEST_SETTINGS
from src.strategies.universe.filters import VectorizedFilters


def test_max_delta_is_a_real_adjacent_gap_limit():
    filters = VectorizedFilters(np)
    candidates = np.asarray(
        [[1, 2, 3, 4, 5, 20], [1, 2, 3, 4, 5, 21]], dtype=np.uint8
    )
    config = {
        "even_min": 0,
        "even_max": 6,
        "prime_min": 0,
        "prime_max": 6,
        "max_contig": 5,
        "max_delta": 15,
    }

    result = filters.apply_structure(candidates, config)

    assert result.tolist() == [[1, 2, 3, 4, 5, 20]]


def test_max_per_decade_controls_spatial_filter():
    filters = VectorizedFilters(np)
    candidates = np.asarray(
        [[1, 2, 3, 4, 15, 25], [1, 2, 3, 14, 15, 25]], dtype=np.uint8
    )

    strict, _ = filters.apply_spatial(candidates, {"max_per_decade": 3})
    relaxed, _ = filters.apply_spatial(candidates, {"max_per_decade": 4})

    assert strict.tolist() == [[1, 2, 3, 14, 15, 25]]
    assert relaxed.tolist() == candidates.tolist()


def test_positional_and_spatial_filters_can_be_disabled_explicitly():
    filters = VectorizedFilters(np)
    candidates = np.asarray(
        [[15, 16, 17, 18, 19, 20], [1, 2, 3, 4, 5, 6]], dtype=np.uint8
    )

    positional = filters.apply_positional_limits(
        candidates,
        {
            "positional_filter_enabled": False,
            "f1_max": 1,
            "f6_min": 39,
        },
    )
    spatial, vectors = filters.apply_spatial(
        candidates,
        {"spatial_filter_enabled": False, "max_per_decade": 1},
    )

    assert positional.tolist() == candidates.tolist()
    assert spatial.tolist() == candidates.tolist()
    assert [vector.tolist() for vector in vectors] == [
        [0, 6],
        [6, 0],
        [0, 0],
        [0, 0],
    ]


def test_std_filter_is_opt_in_and_compensation_reaches_target():
    filters = VectorizedFilters(np)
    candidates = np.asarray(
        list(itertools.islice(itertools.combinations(range(1, 16), 6), 20)),
        dtype=np.uint8,
    )
    disabled = filters.apply_standard_deviation(
        candidates,
        {"std_min": 0.0, "std_max": 0.1, "std_filter_enabled": False},
    )
    compensated = filters.apply_standard_deviation(
        candidates,
        {
            "std_min": 0.0,
            "std_max": 0.1,
            "std_filter_enabled": True,
            "auto_std_compensation": True,
            "target_universe_size": 7,
        },
    )

    assert len(disabled) == len(candidates)
    assert len(compensated) == 7
    assert len({tuple(row) for row in compensated.tolist()}) == 7


def test_entropy_upper_bound_is_clamped_to_theoretical_maximum():
    filters = VectorizedFilters(np)
    uniform_gaps = np.asarray([[1, 2, 3, 4, 5, 6]], dtype=np.uint8)

    accepted = filters.apply_entropy_shannon(
        uniform_gaps, {"entropy_min": 0.0, "entropy_max": 99.0}
    )
    rejected = filters.apply_entropy_shannon(
        uniform_gaps, {"entropy_min": 0.0, "entropy_max": 2.30}
    )

    assert accepted.tolist() == uniform_gaps.tolist()
    assert len(rejected) == 0


def test_production_disables_unvalidated_hard_geo_filters():
    assert BEST_SETTINGS["sniper_mode"] == "soft"
    assert BEST_SETTINGS["candidate_selection_mode"] == "balanced_mixed"
    assert BEST_SETTINGS["radar_percentile"] == 0.0

    hard_flags = (
        "positional_filter_enabled",
        "sum_filter_enabled",
        "structure_filter_enabled",
        "terminal_filter_enabled",
        "spatial_filter_enabled",
        "decade_profile_filter_enabled",
        "entropy_filter_enabled",
        "digital_root_filter_enabled",
        "ac_filter_enabled",
    )
    assert all(BEST_SETTINGS[name] is False for name in hard_flags)


def test_disabled_geo_filters_preserve_candidates_that_violate_legacy_limits():
    filters = VectorizedFilters(np)
    candidates = np.asarray([[1, 2, 3, 4, 5, 6]], dtype=np.uint8)
    cfg = dict(BEST_SETTINGS)

    assert (
        filters.apply_positional_limits(candidates, cfg).tolist()
        == candidates.tolist()
    )
    assert filters.apply_aggregation(candidates, cfg).tolist() == candidates.tolist()
    assert filters.apply_structure(candidates, cfg).tolist() == candidates.tolist()
    terminal, _ = filters.apply_terminal_poda(candidates, cfg)
    assert terminal.tolist() == candidates.tolist()
    spatial, decades = filters.apply_spatial(candidates, cfg)
    assert spatial.tolist() == candidates.tolist()
    assert (
        filters.apply_profile_poda(spatial, decades, cfg).tolist()
        == candidates.tolist()
    )
    assert (
        filters.apply_entropy_shannon(candidates, cfg).tolist()
        == candidates.tolist()
    )
    assert (
        filters.apply_digital_root_sum(candidates, cfg).tolist()
        == candidates.tolist()
    )
    assert filters.apply_ac_complexity(candidates, cfg).tolist() == candidates.tolist()
