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
        "ai_signal_enabled": True,
        "ai_signal_validated": False,
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


def test_console_labels_ai_as_relative_and_unvalidated():
    audit = LotteryForensics.audit_winner(
        _snapshot(), [1, 2, 3, 4, 5, 6, 7], np
    )
    output = StringIO()
    engine = BacktestEngine.__new__(BacktestEngine)
    engine.console = Console(file=output, force_terminal=False, width=180)

    engine._render_telemetry(audit, 1605, 0.0, 6)

    rendered = output.getvalue()
    assert "AIr:" in rendered
    assert "p50 NV" in rendered
    assert "Mix: 40/60" in rendered
