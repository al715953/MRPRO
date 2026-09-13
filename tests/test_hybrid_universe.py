from itertools import combinations, islice
from types import SimpleNamespace

import numpy as np

from src.domain.dtos import PredictionResultDTO
from src.strategies.genetic_selector import GeneticSelectorStrategy
from src.strategies.universe.hybrid import HybridUniverseReductionStrategy


class _ReducerStub:
    def __init__(self, rows):
        self.rows = np.asarray(rows, dtype=np.uint8)
        self.seen_overrides = None

    def predict(self, _history, config, verbose=False):
        self.seen_overrides = dict(config.filter_overrides)
        return PredictionResultDTO(
            "stub",
            self.rows.tolist(),
            metadata={
                "raw_ndarray": self.rows,
                "sniper_log": "stub",
                "reduction_stage_stats": {"final_size": len(self.rows)},
            },
        )


def test_hybrid_universe_unions_lanes_without_duplicates_and_restores_config():
    primary = _ReducerStub([[1, 2, 3, 4, 5, 6], [2, 3, 4, 5, 6, 7]])
    topology = _ReducerStub([[2, 3, 4, 5, 6, 7], [7, 8, 9, 10, 11, 12]])
    strategy = HybridUniverseReductionStrategy(primary, topology)
    original = {"fitness_selector_mode": "hybrid_dual_lane"}
    config = SimpleNamespace(filter_overrides=original)

    result = strategy.predict(SimpleNamespace(), config, verbose=False)

    assert config.filter_overrides is original
    assert len(result.metadata["raw_ndarray"]) == 3
    assert result.metadata["hybrid_primary_size"] == 2
    assert result.metadata["hybrid_topology_size"] == 2
    assert result.metadata["hybrid_lane_overlap"] == 1
    assert len(config.hybrid_primary_universe_ptr) == 2
    assert len(config.hybrid_topology_universe_ptr) == 2
    assert primary.seen_overrides["sum_filter_enabled"] is False
    assert primary.seen_overrides["candidate_selection_mode"] == "nested_balanced"
    assert primary.seen_overrides["universe_ticket_limit"] == 30000
    assert topology.seen_overrides["sum_filter_enabled"] is True
    assert topology.seen_overrides["universe_ticket_limit"] == 45000
    assert topology.seen_overrides["sniper_mode"] == "soft"
    assert topology.seen_overrides["radar_percentile"] == 0.0


class _ResonanceStub:
    def calculate_resonance(self, universe, _history, _config, _xp):
        count = len(universe)
        scores = np.linspace(1.0, 0.0, count, endpoint=False)
        return {
            "u_reduced": np.asarray(universe, dtype=np.uint8),
            "final_scores_reduced": scores,
            "radar_indices": np.arange(count, dtype=np.int64),
            "ai_norm": scores,
            "geo_scores": np.zeros(count, dtype=np.float64),
            "sniper_soft_numbers": [],
        }


def test_hybrid_selector_honors_18_plus_6_as_24_unique_tickets():
    union = np.asarray(
        list(islice(combinations(range(1, 40), 6), 1200)), dtype=np.uint8
    )
    primary = union[:800]
    topology = union[400:]
    strategy = GeneticSelectorStrategy.__new__(GeneticSelectorStrategy)
    strategy.resonance_engine = _ResonanceStub()
    config = SimpleNamespace(
        raw_universe_ptr=union,
        hybrid_primary_universe_ptr=primary,
        hybrid_topology_universe_ptr=topology,
        num_tickets=24,
        total_balls=39,
        filter_overrides={
            "fitness_selector_mode": "hybrid_dual_lane",
            "hybrid_primary_tickets": 18,
            "hybrid_topology_tickets": 6,
            "hybrid_topology_min_rank": 501,
            "sniper_soft_reserve_fraction": 0.0,
        },
    )

    result = strategy.predict(SimpleNamespace(), config)

    keys = [tuple(ticket) for ticket in result.tickets]
    primary_keys = {tuple(row) for row in primary.tolist()}
    topology_keys = {tuple(row) for row in topology.tolist()}
    assert len(keys) == 24
    assert len(set(keys)) == 24
    assert all(key in primary_keys for key in keys[:18])
    assert all(key in topology_keys for key in keys[18:])
    assert result.metadata["hybrid_primary_selected"] == 18
    assert result.metadata["hybrid_topology_selected"] == 6
    assert len(result.metadata["hybrid_topology_lane_ranks"]) == 6
