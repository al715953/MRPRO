import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.core.analytics import PerformanceTracker


@pytest.fixture(autouse=True)
def isolate_performance_tracker_files(monkeypatch, tmp_path):
    """Prevent backtester tests from writing into the user's real data ledger."""
    original_init = PerformanceTracker.__init__

    def isolated_init(self, *args, **kwargs):
        kwargs.setdefault("log_path", tmp_path / "detailed_forensic_log.csv")
        kwargs.setdefault("json_path", tmp_path / "backtest_results.json")
        kwargs.setdefault("archive_directory", tmp_path / "forensic_log_archive")
        original_init(self, *args, **kwargs)

    monkeypatch.setattr(PerformanceTracker, "__init__", isolated_init)
