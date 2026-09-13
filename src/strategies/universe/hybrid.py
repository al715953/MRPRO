"""Hybrid Melate universe preserving soft 5/6 and topological 6/6 lanes."""

from __future__ import annotations

from typing import Any

import numpy as np

from src.domain.dtos import PredictionResultDTO
from src.strategies.universe.shadow import LEGACY_HARD_FILTER_OVERRIDES
from src.strategies.universe_reduction import UniverseReductionStrategy


_HARD_FILTER_FLAGS = {
    key: value
    for key, value in LEGACY_HARD_FILTER_OVERRIDES.items()
    if key.endswith("_filter_enabled")
}

_DISABLED_HARD_FILTER_FLAGS = {
    "sum_filter_enabled": False,
    "structure_filter_enabled": False,
    "parity_filter_enabled": False,
    "prime_filter_enabled": False,
    "consecutive_filter_enabled": False,
    "max_delta_filter_enabled": False,
    "terminal_filter_enabled": False,
    "decade_profile_filter_enabled": False,
    "entropy_filter_enabled": False,
    "digital_root_filter_enabled": False,
    "ac_filter_enabled": False,
    "positional_filter_enabled": False,
    "spatial_filter_enabled": False,
    "std_filter_enabled": False,
}


class HybridUniverseReductionStrategy:
    """Build a deduplicated union of the V17.2 and legacy-topology universes.

    The primary lane keeps the production soft/balanced construction. The
    topological lane enables only the old structural gates; Sniper remains soft
    and radar remains open so the experiment isolates universe membership.
    Lane arrays are attached transiently to ``config`` for the dual-lane
    selector. They are rebuilt from history on every draw and never inspect the
    target draw.
    """

    def __init__(
        self,
        primary_reducer: UniverseReductionStrategy | None = None,
        topology_reducer: UniverseReductionStrategy | None = None,
    ):
        self.primary_reducer = primary_reducer or UniverseReductionStrategy()
        self.topology_reducer = topology_reducer or UniverseReductionStrategy()

    @staticmethod
    def _cpu(value: Any) -> np.ndarray:
        if hasattr(value, "get"):
            value = value.get()
        result = np.asarray(value, dtype=np.uint8)
        if result.ndim != 2:
            return np.empty((0, 6), dtype=np.uint8)
        return result

    def predict(self, history, config, verbose: bool = True) -> PredictionResultDTO:
        original_overrides = getattr(config, "filter_overrides", None)
        base = dict(original_overrides or {})

        primary_overrides = {
            **base,
            **_DISABLED_HARD_FILTER_FLAGS,
            "sniper_mode": "soft",
            "candidate_selection_mode": str(
                base.get("hybrid_primary_selection_mode", "nested_balanced")
            ),
            "universe_ticket_limit": int(
                base.get("hybrid_primary_universe_limit", 30000)
            ),
            "universe_exploration_fraction": float(
                base.get("hybrid_primary_exploration_fraction", 0.50)
            ),
            "radar_percentile": 0.0,
        }
        topology_overrides = {
            **base,
            **_DISABLED_HARD_FILTER_FLAGS,
            **_HARD_FILTER_FLAGS,
            "sniper_mode": "soft",
            "candidate_selection_mode": "density",
            "universe_ticket_limit": int(
                base.get("hybrid_topology_universe_limit", 45000)
            ),
            "radar_percentile": 0.0,
        }

        try:
            config.filter_overrides = primary_overrides
            primary_result = self.primary_reducer.predict(
                history, config, verbose=False
            )
            config.filter_overrides = topology_overrides
            topology_result = self.topology_reducer.predict(
                history, config, verbose=False
            )
        finally:
            config.filter_overrides = original_overrides

        primary = self._cpu(primary_result.metadata.get("raw_ndarray"))
        topology = self._cpu(topology_result.metadata.get("raw_ndarray"))
        if primary.size and topology.size:
            union = np.unique(np.concatenate((primary, topology), axis=0), axis=0)
        elif primary.size:
            union = primary.copy()
        else:
            union = topology.copy()

        primary_keys = {tuple(int(value) for value in row) for row in primary}
        overlap = sum(
            tuple(int(value) for value in row) in primary_keys for row in topology
        )

        # The backtester passes ``raw_ndarray`` to the scorer. These two lane
        # pointers are consumed only by the hybrid selector during this draw.
        config.hybrid_primary_universe_ptr = primary
        config.hybrid_topology_universe_ptr = topology

        primary_stats = primary_result.metadata.get("reduction_stage_stats", {})
        topology_stats = topology_result.metadata.get("reduction_stage_stats", {})
        stage_stats = {
            "final_size": int(len(union)),
            "primary_size": int(len(primary)),
            "topology_size": int(len(topology)),
            "lane_overlap": int(overlap),
            "primary": primary_stats,
            "topology": topology_stats,
        }
        return PredictionResultDTO(
            strategy_name="Universe V17.3 Hybrid Soft + Topology",
            tickets=union.tolist(),
            metadata={
                "raw_ndarray": union,
                "final_size": int(len(union)),
                "sniper_log": primary_result.metadata.get("sniper_log", ""),
                "reduction_stage_stats": stage_stats,
                "candidate_selection_mode": "hybrid_soft_topology_union",
                "selection_core_count": int(len(primary)),
                "selection_exploration_count": int(len(topology) - overlap),
                "hybrid_primary_size": int(len(primary)),
                "hybrid_topology_size": int(len(topology)),
                "hybrid_lane_overlap": int(overlap),
            },
        )
