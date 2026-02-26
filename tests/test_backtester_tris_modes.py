from src.core.backtester import BacktestEngine
from src.domain.dtos import DrawHistoryDTO, PredictionConfigDTO, PredictionResultDTO
from src.strategies.tris.tris_forecast import TrisForecastV1A
import numpy as np


class _DummyRules:
    max_hits = 5
    ticket_cost = 0.0

    def validate_ticket(self, ticket, target):
        return 0, False

    def calculate_prize(self, hits, has_additional):
        return 0.0


class _UniverseStrategyNoVerbose:
    def predict(self, history, config):
        k = 7
        raw = [[0, 0, 0, 0, 0] for _ in range(k)]
        return PredictionResultDTO(
            strategy_name="dummy_universe_strategy",
            tickets=[],
            metadata={"raw_ndarray": raw},
        )


class _CompareUniverseStrategyNoVerbose:
    def predict(self, history, config):
        ov = (
            config.filter_overrides
            if hasattr(config, "filter_overrides")
            and isinstance(config.filter_overrides, dict)
            else {}
        )
        score_model = str(ov.get("score_model", "feature_lr")).lower()
        k = int(ov.get("universe_topk_k", 5))
        prev = [int(d) for d in history.winning_numbers[-1][:5]]
        if score_model == "feature_lr":
            row = prev
        else:
            row = [9, 9, 9, 9, 9]
        raw = [row[:] for _ in range(max(0, k))]
        return PredictionResultDTO(
            strategy_name="dummy_compare_universe_strategy",
            tickets=[],
            metadata={"raw_ndarray": raw},
        )


class _SelectorUniformPosProbsStrategy:
    def predict(self, history, config):
        return PredictionResultDTO(
            strategy_name="dummy_selector_uniform",
            tickets=[[0, 0, 0, 0, 0]],
            metadata={"pos_probs": [[0.1] * 10 for _ in range(5)]},
        )


class _SelectorMissingMetadataStrategy:
    def predict(self, history, config):
        return PredictionResultDTO(
            strategy_name="dummy_selector_missing_metadata",
            tickets=[[0, 0, 0, 0, 0]],
            metadata={},
        )


def test_backtester_universe_strategy_uses_raw_ndarray_size_for_audit():
    draws = []
    concursos = []
    dates = []
    for i in range(8):
        draws.append([i % 10, (i + 1) % 10, (i + 2) % 10, (i + 3) % 10, (i + 4) % 10, 0])
        concursos.append(7000 + i)
        dates.append(f"2025-07-{(i % 28) + 1:02d}")

    history = DrawHistoryDTO(dates=dates, winning_numbers=draws, concursos=concursos)
    config = PredictionConfigDTO(
        total_balls=10,
        ticket_size=5,
        num_tickets=1,
        backtest_size=4,
        filter_overrides={
            "tris_backtest_mode": "universe_strategy",
            "structural_enabled": False,
        },
    )

    engine = BacktestEngine(rules=_DummyRules())
    engine.run(_UniverseStrategyNoVerbose(), history, config, verbose=False)

    assert len(engine.forensic_data) == 4
    assert all(int(row.get("univ_size", -1)) == 7 for row in engine.forensic_data)


def test_build_tris_structural_config_maps_structural_override_names_explicitly():
    cfg = BacktestEngine._build_tris_structural_config(
        {
            "structural_enable_global_sum_filter": False,
            "structural_enable_global_parity_filter": False,
            "structural_immediate_repeat_mode": "per_position",
            "structural_immediate_repeat_disallow_positions": [1, 0, 1, 0, 0],
            "structural_positional_limits": [{"forbidden_digits": [9]}, {}, {}, {}, {}],
            "structural_camera_entropy_rules": [{"camera": 1, "min_entropy": 1.5}],
        }
    )

    assert bool(cfg.enable_global_sum_filter) is False
    assert bool(cfg.enable_global_parity_filter) is False
    assert str(cfg.immediate_repeat_mode) == "per_position"
    assert tuple(cfg.immediate_repeat_disallow_positions) == (
        True,
        False,
        True,
        False,
        False,
    )
    assert isinstance(cfg.positional_limits, list)
    assert isinstance(cfg.camera_entropy_rules, list)


def test_backtester_universe_strategy_compare_models_registers_per_draw_metrics():
    draws = []
    concursos = []
    dates = []
    for i in range(10):
        draws.append([1, 1, 1, 1, 1, 0])
        concursos.append(8000 + i)
        dates.append(f"2025-08-{(i % 28) + 1:02d}")

    history = DrawHistoryDTO(dates=dates, winning_numbers=draws, concursos=concursos)
    config = PredictionConfigDTO(
        total_balls=10,
        ticket_size=5,
        num_tickets=1,
        backtest_size=5,
        filter_overrides={
            "tris_backtest_mode": "universe_strategy",
            "compare_models": True,
            "universe_topk_k": 5,
            "structural_enabled": False,
        },
    )

    engine = BacktestEngine(rules=_DummyRules())
    engine.run(_CompareUniverseStrategyNoVerbose(), history, config, verbose=False)

    assert len(engine.forensic_data) == 5
    assert all(int(row.get("univ_size", -1)) == 5 for row in engine.forensic_data)
    assert all(int(row.get("u_lr", -1)) == 5 for row in engine.forensic_data)
    assert all(int(row.get("u_rand", -1)) == 5 for row in engine.forensic_data)
    assert all(int(row.get("in_lr", 0)) == 1 for row in engine.forensic_data)
    assert all(int(row.get("in_rand", 1)) == 0 for row in engine.forensic_data)


def test_backtester_tris_positional_metrics_with_uniform_probs_does_not_break():
    rng = np.random.default_rng(99)
    draws = []
    concursos = []
    dates = []
    for i in range(14):
        row = rng.integers(0, 10, size=5, endpoint=False).tolist()
        draws.append([int(row[0]), int(row[1]), int(row[2]), int(row[3]), int(row[4]), 0])
        concursos.append(9000 + i)
        dates.append(f"2025-09-{(i % 28) + 1:02d}")

    history = DrawHistoryDTO(dates=dates, winning_numbers=draws, concursos=concursos)
    config = PredictionConfigDTO(
        total_balls=10,
        ticket_size=5,
        num_tickets=1,
        backtest_size=6,
        filter_overrides={
            "run_baseline": False,
        },
    )

    engine = BacktestEngine(rules=_DummyRules())
    engine.run(_SelectorUniformPosProbsStrategy(), history, config, verbose=False)

    assert len(engine.forensic_data) == 6
    assert all(str(row.get("logloss", "")) != "" for row in engine.forensic_data)


def test_backtester_tris_missing_metadata_skips_positional_metrics_without_error():
    draws = []
    concursos = []
    dates = []
    for i in range(12):
        draws.append([i % 10, (i + 1) % 10, (i + 2) % 10, (i + 3) % 10, (i + 4) % 10, 0])
        concursos.append(9100 + i)
        dates.append(f"2025-10-{(i % 28) + 1:02d}")

    history = DrawHistoryDTO(dates=dates, winning_numbers=draws, concursos=concursos)
    config = PredictionConfigDTO(
        total_balls=10,
        ticket_size=5,
        num_tickets=1,
        backtest_size=5,
        filter_overrides={
            "run_baseline": False,
        },
    )

    engine = BacktestEngine(rules=_DummyRules())
    engine.run(_SelectorMissingMetadataStrategy(), history, config, verbose=False)

    assert len(engine.forensic_data) == 5
    assert all(str(row.get("logloss", "")) == "" for row in engine.forensic_data)


def test_backtester_camera_mech_masked_universe_smoke_topm2_structural_off():
    rng = np.random.default_rng(20260226)
    draws = []
    concursos = []
    dates = []
    for i in range(120):
        row = rng.integers(0, 10, size=5, endpoint=False).tolist()
        mult = 1 if i % 3 == 0 else 0
        draws.append([int(row[0]), int(row[1]), int(row[2]), int(row[3]), int(row[4]), mult])
        concursos.append(9200 + i)
        dates.append(f"2025-11-{(i % 28) + 1:02d}")

    history = DrawHistoryDTO(dates=dates, winning_numbers=draws, concursos=concursos)
    config = PredictionConfigDTO(
        total_balls=10,
        ticket_size=5,
        num_tickets=1,
        backtest_size=4,
        filter_overrides={
            "tris_backtest_mode": "universe_strategy",
            "gate_margin": -1.0,
            "universe_mode": "topk_scored_universe",
            "score_model": "camera_mech_v1",
            "camera_masked_universe": True,
            "camera_topm_per_position": 2,
            "structural_enable_global_sum_filter": False,
            "structural_enable_global_parity_filter": False,
            "structural_enabled": False,
            "universe_topk_k": 32,
            "run_baseline": False,
            "diversity_min_hamming": 0,
        },
    )

    engine = BacktestEngine(rules=_DummyRules())
    strategy = TrisForecastV1A()
    engine.run(strategy, history, config, verbose=False)

    assert len(engine.forensic_data) == 4
    univ_sizes = [int(row.get("univ_size", -1)) for row in engine.forensic_data]
    assert max(univ_sizes) <= 32
    assert float(np.mean(univ_sizes)) <= 32.0

    mask_present_count = 0
    for row in engine.forensic_data:
        metrics = row.get("metrics_json", {})
        assert bool(metrics.get("camera_mask_present", False))
        mask_present_count += 1 if bool(metrics.get("camera_mask_present", False)) else 0
        pos_unique_final = metrics.get("camera_pos_unique_digits_final")
        assert isinstance(pos_unique_final, list)
        assert len(pos_unique_final) == 5
        assert all(int(v) <= 2 for v in pos_unique_final)

    assert mask_present_count == 4

    pred = strategy.predict(history, config)
    assert isinstance(pred.metadata.get("camera_positional_mask"), list)


def test_backtester_run_context_reflects_structural_repeat_mode(capsys):
    draws = []
    concursos = []
    dates = []
    for i in range(8):
        draws.append([i % 10, (i + 1) % 10, (i + 2) % 10, (i + 3) % 10, (i + 4) % 10, 0])
        concursos.append(9400 + i)
        dates.append(f"2025-12-{(i % 28) + 1:02d}")

    history = DrawHistoryDTO(dates=dates, winning_numbers=draws, concursos=concursos)
    config = PredictionConfigDTO(
        total_balls=10,
        ticket_size=5,
        num_tickets=1,
        backtest_size=2,
        filter_overrides={
            "run_baseline": False,
            "structural_enabled": False,
            "structural_immediate_repeat_mode": "per_position",
            "structural_enable_global_sum_filter": False,
            "structural_enable_global_parity_filter": False,
        },
    )

    engine = BacktestEngine(rules=_DummyRules())
    engine.run(_SelectorMissingMetadataStrategy(), history, config, verbose=True)
    out = capsys.readouterr().out

    assert "TRIS RUN CONTEXT" in out
    assert out.count("TRIS RUN CONTEXT") == 1
    assert "structural_immediate_repeat_mode=per_position" in out
