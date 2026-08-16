import numpy as np
import xgboost as xgb
from types import SimpleNamespace

from src.core.train_static_model import (
    _set_model_metadata,
    _temporal_auc,
    generate_valid_negative_draws,
)
from src.domain.dtos import DrawHistoryDTO
from src.strategies.genetic.resonance import ResonanceEngine


def test_negative_draws_are_valid_unique_sorted_and_reproducible():
    forbidden = {(1, 2, 3, 4, 5, 6)}

    first = generate_valid_negative_draws(
        100,
        rng=np.random.default_rng(42),
        forbidden=forbidden,
    )
    second = generate_valid_negative_draws(
        100,
        rng=np.random.default_rng(42),
        forbidden=forbidden,
    )

    assert np.array_equal(first, second)
    assert first.shape == (100, 6)
    assert np.all(np.diff(first, axis=1) > 0)
    assert int(first.min()) >= 1
    assert int(first.max()) <= 39
    assert len({tuple(row) for row in first.tolist()}) == 100
    assert (1, 2, 3, 4, 5, 6) not in {tuple(row) for row in first.tolist()}


def test_temporal_auc_has_random_and_perfect_reference_values():
    assert _temporal_auc(np.array([0.5]), np.array([0.5])) == 0.5
    assert _temporal_auc(np.array([0.9, 0.8]), np.array([0.2, 0.1])) == 1.0


def test_model_training_cutoff_is_read_from_saved_metadata(tmp_path):
    features = np.array([[1, 2, 3, 4, 5, 6], [2, 3, 4, 5, 6, 7]])
    labels = np.array([1.0, 0.0])
    model = xgb.train(
        {"objective": "binary:logistic", "max_depth": 1},
        xgb.DMatrix(features, label=labels),
        num_boost_round=1,
    )
    _set_model_metadata(
        model,
        role="temporal_backtest",
        trained_through=123,
        training_rows=100,
    )
    model_path = tmp_path / "brain.json"
    model.save_model(model_path)

    engine = ResonanceEngine(model_path=str(model_path))

    assert engine.training_cutoff_contest == 123


def test_temporal_model_stays_active_but_unvalidated_when_auc_is_low(tmp_path):
    features = np.array([[1, 2, 3, 4, 5, 6], [2, 3, 4, 5, 6, 7]])
    labels = np.array([1.0, 0.0])
    model = xgb.train(
        {"objective": "binary:logistic", "max_depth": 1},
        xgb.DMatrix(features, label=labels),
        num_boost_round=1,
    )
    model.set_attr(temporal_holdout_auc="0.4932")
    model_path = tmp_path / "weak-brain.json"
    model.save_model(model_path)

    engine = ResonanceEngine(model_path=str(model_path))

    assert engine.temporal_holdout_auc == 0.4932
    assert engine.ai_signal_enabled is True
    assert engine.ai_signal_validated is False


def test_unvalidated_ai_scores_remain_active():
    class BoosterStub:
        def attr(self, name):
            return "0.4932" if name == "temporal_holdout_auc" else None

        def predict(self, matrix):
            return np.linspace(0.1, 0.9, matrix.num_row(), dtype=np.float32)

    engine = ResonanceEngine(
        model_path="missing-for-test.json",
        number_model_path="missing-number-for-test.json",
    )
    engine.bst = BoosterStub()
    universe = np.array(
        [
            [1, 2, 3, 4, 5, 6],
            [3, 8, 13, 18, 23, 28],
            [7, 12, 17, 22, 27, 32],
        ],
        dtype=np.uint8,
    )
    history = DrawHistoryDTO(
        dates=[1, 2],
        winning_numbers=[
            [1, 7, 13, 19, 25, 31, 0],
            [2, 8, 14, 20, 26, 32, 0],
        ],
        concursos=[1, 2],
    )

    result = engine.calculate_resonance(
        universe,
        history,
        SimpleNamespace(total_balls=39),
        np,
    )

    assert result["ai_signal_enabled"] is True
    assert result["ai_signal_validated"] is False
    assert np.ptp(result["ai_norm"]) > 0.0
    assert np.all(result["final_scores_reduced"] >= 0.0)
