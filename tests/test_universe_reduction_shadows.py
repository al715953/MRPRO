from __future__ import annotations

import numpy as np

from src.domain.dtos import DrawHistoryDTO, PredictionConfigDTO, PredictionResultDTO
from src.strategies.genetic_selector import GeneticSelectorStrategy
from src.strategies.universe import shadow as shadow_module
from src.strategies.universe.shadow import (
    PROFILE_OOS_SET,
    PROMOTED_UNIVERSE_SHADOWS,
    build_universe_shadow_variant,
)
from src.strategies.universe_reduction import UniverseReductionStrategy


class _FakeFilters:
    def __init__(self):
        self.generated_with = None

    def get_sniper_exclusion(self, *_args, **_kwargs):
        return [7], "Sniper:-7(0.95)"

    def generate_universe(self, excluded_pool=None):
        self.generated_with = list(excluded_pool or [])
        return np.tile(np.arange(1, 7, dtype=np.uint8), (20, 1))

    def apply_positional_limits(self, universe, _cfg):
        return universe

    def apply_aggregation(self, universe, _cfg):
        return universe

    def apply_structure(self, universe, _cfg):
        return universe

    def apply_terminal_poda(self, universe, _cfg):
        return universe, np.ones(len(universe), dtype=bool)

    def apply_spatial(self, universe, _cfg):
        zeros = np.zeros(len(universe), dtype=np.int8)
        return universe, (zeros, zeros, zeros, zeros)

    def apply_profile_poda(self, universe, _vectors, _cfg):
        return universe

    def apply_entropy_shannon(self, universe, _cfg):
        return universe

    def apply_digital_root_sum(self, universe, _cfg):
        return universe

    def apply_ac_complexity(self, universe, _cfg):
        return universe


def _history():
    return DrawHistoryDTO(
        dates=["2026-01-01"],
        winning_numbers=[[1, 2, 3, 4, 5, 6, 7]],
        concursos=[1],
    )


def test_reducer_honors_configurable_limit_and_reports_every_stage(monkeypatch):
    reducer = UniverseReductionStrategy()
    reducer.filters = _FakeFilters()
    monkeypatch.setattr(
        reducer,
        "_density_penalized_selection",
        lambda universe, _cfg, target: universe[:target],
    )
    config = PredictionConfigDTO(
        39,
        6,
        4,
        filter_overrides={
            "sniper_mode": "hard",
            "universe_ticket_limit": 10,
        },
    )

    result = reducer.predict(_history(), config, verbose=False)

    stats = result.metadata["reduction_stage_stats"]
    assert len(result.tickets) == 10
    assert reducer.filters.generated_with == [7]
    assert stats["topk_applied"] is True
    assert stats["universe_ticket_limit"] == 10
    assert stats["hard_excluded_numbers"] == [7]
    assert [row["stage"] for row in stats["stages"]] == [
        "positional",
        "sum",
        "structure",
        "terminal",
        "spatial",
        "decade_profile",
        "entropy",
        "digital_root",
        "ac_complexity",
        "density_topk",
    ]


def test_soft_sniper_keeps_number_and_exports_transient_signal():
    reducer = UniverseReductionStrategy()
    reducer.filters = _FakeFilters()
    config = PredictionConfigDTO(
        39,
        6,
        4,
        filter_overrides={"sniper_mode": "soft"},
    )

    result = reducer.predict(_history(), config, verbose=False)

    assert reducer.filters.generated_with == []
    assert config.filter_overrides["sniper_soft_numbers"] == [7]
    assert result.metadata["sniper_candidates"] == [7]
    assert result.metadata["hard_excluded_numbers"] == []


def test_profile_shadow_specs_are_non_official_and_use_expected_profiles(monkeypatch):
    class _Reducer:
        def predict(self, _history, config, verbose=False):
            assert verbose is False
            return PredictionResultDTO(
                "reducer",
                [],
                metadata={
                    "raw_ndarray": np.asarray([[1, 2, 3, 4, 5, 6]]),
                    "final_size": 1,
                    "sniper_mode": config.filter_overrides.get(
                        "sniper_mode", "hard"
                    ),
                    "sniper_candidates": [],
                    "hard_excluded_numbers": [],
                    "universe_ticket_limit": config.filter_overrides.get(
                        "universe_ticket_limit", 45000
                    ),
                    "reduction_stage_stats": {"stages": []},
                },
            )

    class _Selector:
        def predict(self, _history, config):
            assert config.raw_universe_ptr is not None
            return PredictionResultDTO("selector", [[1, 2, 3, 4, 5, 6]])

    monkeypatch.setattr(shadow_module, "UniverseReductionStrategy", _Reducer)
    profile_spec = PROMOTED_UNIVERSE_SHADOWS[0]
    variant = build_universe_shadow_variant(
        _history(),
        _Selector(),
        profile_spec,
        total_balls=39,
        ticket_size=6,
        ticket_count=24,
    )

    assert variant["official"] is False
    assert variant["settings"]["valid_decade_profiles"] == list(PROFILE_OOS_SET)
    assert variant["metadata"]["raw_universe_size"] == 1
    assert variant["metadata"]["universe_variant"] == "profile_oos_43k"


def test_soft_reserve_replaces_non_soft_ticket_without_duplicates():
    selected = [
        [1, 2, 3, 4, 5, 6],
        [1, 2, 3, 4, 5, 7],
        [1, 2, 3, 4, 6, 7],
        [1, 2, 3, 5, 6, 7],
    ]
    candidates = np.asarray(
        selected + [[1, 2, 3, 4, 5, 9], [2, 3, 4, 5, 6, 9]], dtype=np.uint8
    )
    scores = np.asarray([1.0, 0.9, 0.8, 0.7, 0.95, 0.85])

    result, debug = GeneticSelectorStrategy._ensure_soft_veto_reserve(
        selected, candidates, scores, [9], 0.25
    )

    assert debug == {"target": 1, "actual": 1, "replacements": 1}
    assert sum(9 in ticket for ticket in result) == 1
    assert len({tuple(ticket) for ticket in result}) == len(result) == 4
