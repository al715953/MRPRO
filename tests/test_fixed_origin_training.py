from pathlib import Path

import pytest
import xgboost as xgb

from src.core.backtester import BacktestEngine
from src.core.fixed_origin_training import prepare_fixed_origin_models
from src.domain.dtos import DrawHistoryDTO, PredictionConfigDTO


def _history(rows: int = 140) -> DrawHistoryDTO:
    contests = list(range(1001, 1001 + rows))
    draws = []
    for idx in range(rows):
        draws.append(
            sorted((((idx * 7) + (offset * 5)) % 39) + 1 for offset in range(6))
        )
    return DrawHistoryDTO(
        dates=[f"d{idx}" for idx in range(rows)],
        concursos=contests,
        winning_numbers=draws,
    )


def test_fixed_origin_trains_one_before_window_and_reuses_cache(tmp_path):
    artifacts = prepare_fixed_origin_models(
        _history(),
        20,
        cache_directory=tmp_path,
    )

    assert artifacts.training_rows == 120
    assert artifacts.training_cutoff_contest == 1120
    assert artifacts.test_start_contest == 1121
    assert artifacts.test_end_contest == 1140
    assert artifacts.requested_backtest_size == 20
    assert artifacts.reused_cache is False
    assert Path(artifacts.context_model_path).exists()
    assert Path(artifacts.number_model_path).exists()
    assert Path(artifacts.manifest_path).exists()

    context = xgb.Booster()
    context.load_model(artifacts.context_model_path)
    number = xgb.Booster()
    number.load_model(artifacts.number_model_path)
    assert context.attr("model_role") == "fixed_origin_backtest"
    assert context.attr("trained_through_concurso") == "1120"
    assert context.attr("holdout_start_concurso") == "1121"
    assert context.attr("requested_backtest_size") == "20"
    assert number.attr("model_role") == "fixed_origin_backtest_number"
    assert number.attr("trained_through_concurso") == "1120"

    reused = prepare_fixed_origin_models(
        _history(),
        20,
        cache_directory=tmp_path,
    )
    assert reused.reused_cache is True
    assert reused.context_model_path == artifacts.context_model_path
    assert reused.number_model_path == artifacts.number_model_path


def test_backtester_rejects_model_from_another_fixed_origin_window():
    class WrongCutoffStrategy:
        training_cutoff_contest = 110
        temporal_holdout_auc = None
        ai_signal_enabled = True
        ai_signal_validated = True
        number_temporal_holdout_auc = None

    history = _history(140)
    config = PredictionConfigDTO(
        39,
        6,
        24,
        backtest_size=20,
        filter_overrides={
            "fixed_origin_training_cutoff": 1120,
            "fixed_origin_test_start": 1121,
        },
    )

    with pytest.raises(ValueError, match="no coincide con el corte solicitado"):
        BacktestEngine().run(
            WrongCutoffStrategy(),
            history,
            config,
            verbose=False,
        )
