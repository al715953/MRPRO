import itertools

import numpy as np

from src.domain.dtos import PredictionConfigDTO
from src.strategies.genetic.resonance import ResonanceEngine


class _FakeBooster:
    def attr(self, _name):
        return None

    def predict(self, dmatrix):
        # Señal deliberadamente opuesta a Geo para comprobar el aislamiento.
        return np.linspace(1.0, 0.0, dmatrix.num_row(), dtype=np.float32)


def test_fixed_geo_only_blend_ignores_ai_scores(monkeypatch):
    candidates = np.asarray(
        list(itertools.islice(itertools.combinations(range(1, 40), 6), 240)),
        dtype=np.uint8,
    )
    history = [
        [1, 7, 13, 19, 25, 31],
        [2, 8, 14, 20, 26, 32],
    ] * 5
    geo = np.linspace(0.0, 1.0, len(candidates), dtype=np.float32)

    engine = ResonanceEngine(model_path="/nonexistent/context.json")
    engine.bst = _FakeBooster()
    monkeypatch.setattr(engine, "_compute_geo_score", lambda *_args: geo)

    config = PredictionConfigDTO(
        total_balls=39,
        ticket_size=6,
        num_tickets=24,
        filter_overrides={
            "resonance_blend_mode": "fixed",
            "hybrid_alpha": 0.0,
            "hybrid_beta": 1.0,
        },
    )
    result = engine.calculate_resonance(candidates, history, config, np)

    radar = np.asarray(result["radar_indices"])
    np.testing.assert_allclose(result["final_scores_reduced"], geo[radar])
    assert result["resonance_blend_mode"] == "fixed"
    assert result["hybrid_alpha"] == 0.0
    assert result["hybrid_beta"] == 1.0


def test_unknown_blend_mode_preserves_adaptive_default(monkeypatch):
    candidates = np.asarray(
        list(itertools.islice(itertools.combinations(range(1, 40), 6), 240)),
        dtype=np.uint8,
    )
    history = [[1, 7, 13, 19, 25, 31]] * 10

    engine = ResonanceEngine(model_path="/nonexistent/context.json")
    engine.bst = _FakeBooster()
    monkeypatch.setattr(
        engine,
        "_compute_geo_score",
        lambda *_args: np.linspace(0.0, 1.0, len(candidates), dtype=np.float32),
    )
    config = PredictionConfigDTO(
        total_balls=39,
        ticket_size=6,
        num_tickets=24,
        filter_overrides={"resonance_blend_mode": "not-a-mode"},
    )

    result = engine.calculate_resonance(candidates, history, config, np)

    assert result["resonance_blend_mode"] == "adaptive"
    assert result["hybrid_alpha"] == 0.5
    assert result["hybrid_beta"] == 0.5


def test_soft_sniper_penalty_is_applied_without_hard_removal(monkeypatch):
    candidates = np.asarray(
        list(itertools.islice(itertools.combinations(range(1, 14), 6), 240)),
        dtype=np.uint8,
    )
    history = [[1, 7, 13, 19, 25, 31]] * 10
    geo = np.linspace(0.2, 1.0, len(candidates), dtype=np.float32)
    engine = ResonanceEngine(model_path="/nonexistent/context.json")
    engine.bst = _FakeBooster()
    monkeypatch.setattr(engine, "_compute_geo_score", lambda *_args: geo)
    config = PredictionConfigDTO(
        total_balls=39,
        ticket_size=6,
        num_tickets=24,
        filter_overrides={
            "resonance_blend_mode": "fixed",
            "hybrid_alpha": 0.0,
            "hybrid_beta": 1.0,
            "sniper_soft_numbers": [1],
            "sniper_soft_penalty": 0.25,
        },
    )

    result = engine.calculate_resonance(candidates, history, config, np)

    radar = np.asarray(result["radar_indices"])
    multiplier = np.where(np.any(candidates[radar] == 1, axis=1), 0.75, 1.0)
    np.testing.assert_allclose(
        result["final_scores_reduced"], geo[radar] * multiplier
    )
    assert result["sniper_soft_numbers"] == [1]
    assert result["sniper_soft_penalty"] == 0.25
    assert result["sniper_soft_candidate_count"] > 0


def test_zero_radar_percentile_keeps_the_whole_candidate_universe(monkeypatch):
    candidates = np.asarray(
        list(itertools.islice(itertools.combinations(range(1, 40), 6), 240)),
        dtype=np.uint8,
    )
    history = [[1, 7, 13, 19, 25, 31]] * 10
    engine = ResonanceEngine(model_path="/nonexistent/context.json")
    engine.bst = _FakeBooster()
    monkeypatch.setattr(
        engine,
        "_compute_geo_score",
        lambda *_args: np.linspace(0.0, 1.0, len(candidates), dtype=np.float32),
    )
    config = PredictionConfigDTO(
        total_balls=39,
        ticket_size=6,
        num_tickets=24,
        filter_overrides={"radar_percentile": 0.0},
    )

    result = engine.calculate_resonance(candidates, history, config, np)

    assert len(result["radar_indices"]) == len(candidates)
    assert result["radar_percentile"] == 0.0
