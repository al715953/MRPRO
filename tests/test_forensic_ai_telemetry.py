from io import StringIO

import numpy as np
from rich.console import Console

from src.core.backtester import BacktestEngine
from src.core.forensics import LotteryForensics


def _snapshot():
    return {
        "universe": np.asarray(
            [
                [1, 2, 3, 4, 5, 6],
                [1, 2, 3, 7, 8, 9],
                [10, 11, 12, 13, 14, 15],
            ],
            dtype=np.uint8,
        ),
        "hybrid_scores": np.asarray([0.65, 0.80, 0.30], dtype=np.float32),
        "ai_scores": np.asarray([0.90, 1.00, 0.10], dtype=np.float32),
        "geo_scores": np.asarray([0.50, 0.20, 0.30], dtype=np.float32),
        "selected_ranks": [1],
        "selected_stable_ranks": [2],
        "tickets": [[1, 2, 3, 4, 5, 6]],
        "ai_signal_enabled": True,
        "ai_signal_validated": False,
        "ai_validation_scope": "model",
        "temporal_holdout_auc": 0.49487,
        "resonance_blend_mode": "adaptive",
    }


def test_forensics_reports_relative_ai_percentile_and_effective_mix():
    audit = LotteryForensics.audit_winner(
        _snapshot(), [1, 2, 3, 4, 5, 6, 7], np
    )

    assert audit["rank"] == 2
    assert audit["proximity"] == 1
    assert audit["ai_score_kind"] == "relative_minmax"
    assert audit["ai_percentile_rank"] == 50.0
    assert audit["ai_weight_effective"] == 0.40
    assert audit["geo_weight_effective"] == 0.60
    assert audit["ai_signal_validated"] is False
    assert audit["winner_in_universe"] == 1
    assert audit["winner_selected_max_overlap"] == 6
    assert audit["winner_selected_min_missing"] == 0
    assert audit["winner_selected_count_ge_5"] == 1
    assert audit["winner_selected_overlap_counts"]["6"] == 1
    assert audit["winner_selected_best_ranks"] == [1]
    assert audit["winner_stable_rank"] == 2
    assert audit["winner_score_tie_size"] == 1
    assert audit["winner_selected_best_stable_ranks"] == [2]


def test_forensics_separates_score_ties_from_stable_selector_rank():
    snapshot = _snapshot()
    snapshot["hybrid_scores"] = np.asarray([0.80, 0.80, 0.30], dtype=np.float32)
    snapshot["selected_ranks"] = [1]
    snapshot["selected_stable_ranks"] = [1]

    audit = LotteryForensics.audit_winner(
        snapshot, [1, 2, 3, 4, 5, 6, 7], np
    )

    assert audit["rank"] == 1
    assert audit["winner_stable_rank"] == 1
    assert audit["winner_score_tie_size"] == 2
    assert audit["winner_stable_rank_proximity"] == 0


def test_forensics_ranks_exact_winner_signals_inside_topology_deep_band():
    universe = np.asarray(
        [
            [1, 2, 3, 4, 5, 6],
            [1, 2, 3, 7, 8, 9],
            [10, 11, 12, 13, 14, 15],
            [16, 17, 18, 19, 20, 21],
            [22, 23, 24, 25, 26, 27],
        ],
        dtype=np.uint8,
    )
    snapshot = {
        **_snapshot(),
        "universe": universe,
        "_hybrid_topology_universe": universe,
        "_pred_tickets": [],
        "hybrid_primary_selected": 0,
        "hybrid_topology_lane_phases": ["deep"],
        "hybrid_topology_deep_bands": [{"rank_min": 1, "rank_max": 5}],
        "hybrid_scores": np.asarray([0.6, 0.9, 0.8, 0.7, 0.5]),
        "ai_scores": np.asarray([0.9, 0.1, 0.2, 0.3, 0.4]),
        "number_ai_scores": np.asarray([0.1, 0.9, 0.8, 0.7, 0.6]),
        "geo_scores": np.asarray([0.5, 0.5, 0.4, 0.3, 0.2]),
        "tickets": [[1, 2, 3, 7, 8, 9]],
    }

    audit = LotteryForensics.audit_winner(
        snapshot, [1, 2, 3, 4, 5, 6, 7], np
    )

    assert audit["winner_topology_deep_band_rank"] == 4
    assert audit["winner_topology_deep_band_size"] == 5
    assert audit["winner_topology_deep_hybrid_rank"] == 4
    assert audit["winner_topology_deep_ai_rank"] == 1
    assert audit["winner_topology_deep_number_rank"] == 5
    assert audit["winner_topology_deep_geo_rank"] == 1
    assert audit["winner_topology_deep_geo_tie_size"] == 2
    assert audit["winner_topology_neighbor_band_count"] == 1
    assert audit["winner_topology_neighbor_ai_best_rank"] == 1
    assert audit["winner_topology_neighbor_number_best_rank"] == 5
    assert audit["winner_topology_neighbor_geo_best_rank"] == 1
    assert audit["winner_topology_neighbor_hybrid_best_overlap"] == 6
    assert audit["winner_topology_deep_oracle_max_overlap"] == 6
    assert audit["winner_topology_deep_oracle_overlap_sum"] == 6
    assert audit["winner_topology_deep_hybrid_top1_max_overlap"] == 3
    assert audit["winner_topology_deep_hybrid_top10_max_overlap"] == 6
    assert audit["winner_topology_deep_ai_top1_max_overlap"] == 6
    assert audit["winner_topology_deep_number_top1_max_overlap"] == 3
    assert audit["winner_topology_deep_geo_top1_max_overlap"] == 3
    assert audit["winner_topology_deep_geo_top_tie_mean"] == 2.0
    assert audit["winner_topology_deep_selected_max_overlap"] == 3
    assert audit["winner_topology_deep_selected_overlap_sum"] == 3
    assert audit["winner_topology_deep_selected_count"] == 1


def test_console_does_not_repeat_model_validation_on_every_draw():
    audit = LotteryForensics.audit_winner(
        _snapshot(), [1, 2, 3, 4, 5, 6, 7], np
    )
    output = StringIO()
    engine = BacktestEngine.__new__(BacktestEngine)
    engine.console = Console(file=output, force_terminal=False, width=180)

    engine._render_telemetry(audit, 1605, 0.0, 6)

    rendered = output.getvalue()
    assert "AIr:" in rendered
    assert "p50" in rendered
    assert "NV" not in rendered
    assert "Mix: 40/60" in rendered


def test_console_keeps_validation_when_scope_can_change_by_draw():
    snapshot = _snapshot()
    snapshot["ai_validation_scope"] = "draw"
    audit = LotteryForensics.audit_winner(
        snapshot, [1, 2, 3, 4, 5, 6, 7], np
    )
    output = StringIO()
    engine = BacktestEngine.__new__(BacktestEngine)
    engine.console = Console(file=output, force_terminal=False, width=180)

    engine._render_telemetry(audit, 1605, 0.0, 6)

    assert "p50 NV" in output.getvalue()
