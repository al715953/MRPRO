import gzip
import json
from pathlib import Path

import pandas as pd

from src.core.analytics import PerformanceTracker


def _tracker(tmp_path, *, max_bytes=1024 * 1024, keep=12):
    return PerformanceTracker(
        log_path=tmp_path / "detailed_forensic_log.csv",
        json_path=tmp_path / "backtest_results.json",
        archive_directory=tmp_path / "archive",
        max_log_bytes=max_bytes,
        archive_keep=keep,
    )


def _rows(event_id, start, count=2):
    return [
        {
            "draw_id": start + index,
            "hits": index % 7,
            "event_id": event_id,
            "profile_code": "melate_retro",
            "split_id": "bt_test",
            "ai_score": 0.5,
            "geo_score": 0.4,
            "metrics_json": {"payload": "x" * 100},
        }
        for index in range(count)
    ]


def _read_gzip_csv(path):
    with gzip.open(path, mode="rt", encoding="utf-8") as source:
        return pd.read_csv(source)


def test_rotation_archives_complete_previous_run_and_keeps_new_run_together(tmp_path):
    tracker = _tracker(tmp_path)
    tracker._save_to_csv("v1", _rows("first", 1, 3))
    tracker.max_log_bytes = Path(tracker.log_path).stat().st_size + 1

    tracker._save_to_csv("v2", _rows("second", 10, 4))

    archives = list(tracker.archive_directory.glob("*.csv.gz"))
    assert len(archives) == 1
    archived = _read_gzip_csv(archives[0])
    active = pd.read_csv(tracker.log_path)
    assert archived["event_id"].tolist() == ["first"] * 3
    assert active["event_id"].tolist() == ["second"] * 4
    assert active["draw_id"].tolist() == [10, 11, 12, 13]
    assert active.columns.tolist() == list(PerformanceTracker.COLUMNS_ORDER)
    assert tracker.get_summary()["event_id"].tolist() == [
        "first",
        "first",
        "first",
        "second",
        "second",
        "second",
        "second",
    ]
    assert tracker.get_summary(include_archives=False)["event_id"].tolist() == [
        "second",
        "second",
        "second",
        "second",
    ]


def test_archive_retention_removes_only_oldest_files(tmp_path):
    tracker = _tracker(tmp_path, max_bytes=1, keep=2)

    for index in range(4):
        tracker._save_to_csv(f"v{index}", _rows(f"event-{index}", index, 1))

    archives = sorted(tracker.archive_directory.glob("*.csv.gz"))
    active = pd.read_csv(tracker.log_path)
    archived_events = {
        str(_read_gzip_csv(path)["event_id"].iloc[0]) for path in archives
    }
    assert len(archives) == 2
    assert archived_events == {"event-1", "event-2"}
    assert active["event_id"].tolist() == ["event-3"]


def test_non_positive_limit_disables_rotation(tmp_path):
    tracker = _tracker(tmp_path, max_bytes=0)

    tracker._save_to_csv("v1", _rows("first", 1, 2))
    tracker._save_to_csv("v2", _rows("second", 10, 2))

    active = pd.read_csv(tracker.log_path)
    assert active["event_id"].tolist() == ["first", "first", "second", "second"]
    assert not tracker.archive_directory.exists()


def test_archive_failure_leaves_active_log_unchanged(tmp_path, monkeypatch):
    tracker = _tracker(tmp_path)
    tracker._save_to_csv("v1", _rows("first", 1, 2))
    tracker.max_log_bytes = Path(tracker.log_path).stat().st_size + 1
    original = Path(tracker.log_path).read_bytes()

    def fail_archive():
        raise OSError("simulated archive failure")

    monkeypatch.setattr(tracker, "_archive_current_log", fail_archive)
    tracker._save_to_csv("v2", _rows("second", 10, 2))

    assert Path(tracker.log_path).read_bytes() == original


def test_json_report_is_replaced_atomically(tmp_path):
    tracker = _tracker(tmp_path)
    tracker._save_to_json(
        {"total_draws_tested": 1},
        "test-version",
        [{"draw_id": 42, "hits": 4}],
    )

    payload = json.loads(Path(tracker.json_path).read_text(encoding="utf-8"))
    temporary_files = list(tmp_path.glob(".backtest_results.json.*.tmp"))

    assert payload["version"] == "test-version"
    assert payload["summary"]["total_draws_tested"] == 1
    assert payload["forensic_details"] == [{"draw_id": 42, "hits": 4}]
    assert temporary_files == []


def test_json_serialization_failure_preserves_previous_report(tmp_path):
    tracker = _tracker(tmp_path)
    tracker._save_to_json({"total_draws_tested": 1}, "valid", [])
    original = Path(tracker.json_path).read_bytes()

    tracker._save_to_json({"invalid": 1 + 2j}, "invalid", [])

    assert Path(tracker.json_path).read_bytes() == original
    assert list(tmp_path.glob(".backtest_results.json.*.tmp")) == []
