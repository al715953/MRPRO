import pandas as pd
import numpy as np

from src.core.rules import TrisMultiplicadorRules
from src.data_access.loader import LotteryLoader
from src.data_access.config import get_lottery_profile
from src.domain.dtos import DrawHistoryDTO, PredictionConfigDTO
from src.strategies.tris.tris_forecast import TrisForecastV1A


def test_tris_loader_reads_digit_columns():
    df = pd.DataFrame(
        {
            "CONCURSO": [1001],
            "FECHA": ["01/01/2025"],
            "R1": [1],
            "R2": [2],
            "R3": [3],
            "R4": [4],
            "R5": [5],
            "Multiplicador": ["SI"],
        }
    )

    profile = get_lottery_profile("tris_multiplicador")
    loader = LotteryLoader(profile)
    history = loader._process_tris(df)

    assert history.concursos == [1001]
    assert history.winning_numbers == [[1, 2, 3, 4, 5, 1]]


def test_tris_rules_exact_match_prize_and_profile_exists():
    rules = TrisMultiplicadorRules(base_prize=500)
    assert rules.validate_ticket([1, 2, 3, 4, 5], [1, 2, 3, 4, 5, 2]) == (5, True)
    assert rules.calculate_prize(5, True) == 1000
    assert get_lottery_profile("tris_multiplicador").ticket_size == 5


def test_tris_forecast_v1a_predict_output_shape():
    draws = []
    concursos = []
    dates = []
    for i in range(100):
        d1 = i % 10
        d2 = (i + 3) % 10
        d3 = (i * 2) % 10
        d4 = (i * 3 + 1) % 10
        d5 = (i * 4 + 2) % 10
        mult = 1 if i % 4 == 0 else 0
        draws.append([d1, d2, d3, d4, d5, mult])
        concursos.append(1000 + i)
        dates.append(f"2025-01-{(i % 28) + 1:02d}")

    history = DrawHistoryDTO(dates=dates, winning_numbers=draws, concursos=concursos)
    config = PredictionConfigDTO(total_balls=10, ticket_size=5, num_tickets=12)

    pred = TrisForecastV1A().predict(history, config)

    assert len(pred.tickets) == config.num_tickets
    for t in pred.tickets:
        assert len(t) == 5
        assert all(0 <= d <= 9 for d in t)

    pos_probs = pred.metadata["pos_probs"]
    assert len(pos_probs) == 5
    assert all(len(row) == 10 for row in pos_probs)
    for row in pos_probs:
        assert abs(sum(row) - 1.0) < 1e-6


def test_rank_universe_digits_ticket_score_order_is_reproducible():
    strategy = TrisForecastV1A()
    universe_digits = np.array(
        [
            [0, 0, 0, 0, 0],
            [1, 1, 1, 1, 1],
            [2, 2, 2, 2, 2],
            [3, 3, 3, 3, 3],
        ],
        dtype=np.uint8,
    )
    ticket_scores = np.array([0.20, 0.95, 0.10, -1.00], dtype=np.float64)

    ranked_a = strategy._rank_universe_digits(
        universe_digits,
        pos_probs=None,
        ticket_scores=ticket_scores,
        score_mode="ticket_score",
    )
    ranked_b = strategy._rank_universe_digits(
        universe_digits,
        pos_probs=None,
        ticket_scores=ticket_scores,
        score_mode="ticket_score",
    )

    assert ranked_a == ranked_b
    assert [digits for digits, _ in ranked_a] == [
        [1, 1, 1, 1, 1],
        [0, 0, 0, 0, 0],
        [2, 2, 2, 2, 2],
        [3, 3, 3, 3, 3],
    ]


def test_rank_universe_digits_default_matches_legacy_positional_logp():
    strategy = TrisForecastV1A()
    universe_digits = np.array(
        [
            [0, 1, 2, 3, 4],
            [1, 2, 3, 4, 5],
            [2, 3, 4, 5, 6],
            [3, 4, 5, 6, 7],
        ],
        dtype=np.uint8,
    )
    pos_probs = np.array(
        [
            [0.25, 0.20, 0.15, 0.10, 0.08, 0.07, 0.05, 0.04, 0.035, 0.025],
            [0.05, 0.10, 0.18, 0.20, 0.12, 0.10, 0.08, 0.07, 0.06, 0.04],
            [0.08, 0.09, 0.10, 0.12, 0.13, 0.14, 0.11, 0.09, 0.08, 0.06],
            [0.10, 0.09, 0.11, 0.14, 0.15, 0.13, 0.10, 0.08, 0.06, 0.04],
            [0.06, 0.07, 0.08, 0.09, 0.12, 0.13, 0.15, 0.14, 0.10, 0.06],
        ],
        dtype=np.float64,
    )

    ranked = strategy._rank_universe_digits(
        universe_digits,
        pos_probs=pos_probs,
        ticket_scores=None,
        score_mode="positional_logp",
    )

    eps = 1e-12
    logits = np.log(np.clip(pos_probs, eps, None))
    legacy_scores = (
        logits[0, universe_digits[:, 0]]
        + logits[1, universe_digits[:, 1]]
        + logits[2, universe_digits[:, 2]]
        + logits[3, universe_digits[:, 3]]
        + logits[4, universe_digits[:, 4]]
    )
    legacy_order = np.argsort(legacy_scores)[::-1]
    legacy_ranked = [
        (
            [int(d) for d in universe_digits[idx].tolist()],
            float(legacy_scores[idx]),
        )
        for idx in legacy_order.tolist()
    ]

    assert ranked == legacy_ranked


def test_tris_forecast_ticket_ngram_scores_full_filtered_universe():
    draws = []
    concursos = []
    dates = []
    for i in range(30):
        mult = 1 if i % 3 == 0 else 0
        draws.append([0, 1, 2, 3, 4, mult])
        concursos.append(2000 + i)
        dates.append(f"2025-02-{(i % 28) + 1:02d}")

    history = DrawHistoryDTO(dates=dates, winning_numbers=draws, concursos=concursos)
    config = PredictionConfigDTO(
        total_balls=10,
        ticket_size=5,
        num_tickets=1,
        filter_overrides={
            "gate_margin": -1.0,
            "score_model": "ticket_ngram",
            "universe_mode": "full_filtered_universe",
            "selection_mode": "ranked",
            "structural_enabled": False,
            "diversity_min_hamming": 0,
            "topk_preview": 5,
        },
    )

    pred = TrisForecastV1A().predict(history, config)

    assert pred.tickets[0] == [0, 1, 2, 3, 4]
    assert pred.metadata.get("score_model") == "ticket_ngram"
    score_preview = pred.metadata.get("score_preview", [])
    assert isinstance(score_preview, list)
    assert len(score_preview) > 0
    assert score_preview[0]["digits"] == [0, 1, 2, 3, 4]


def test_tris_forecast_topk_scored_universe_feature_lr_metadata_and_ranking():
    rng = np.random.default_rng(123)
    even_digits = np.array([0, 2, 4, 6, 8], dtype=np.int16)
    draws = []
    concursos = []
    dates = []
    for i in range(120):
        row = rng.choice(even_digits, size=5, replace=True).tolist()
        mult = 1 if i % 3 == 0 else 0
        draws.append([int(row[0]), int(row[1]), int(row[2]), int(row[3]), int(row[4]), mult])
        concursos.append(3000 + i)
        dates.append(f"2025-03-{(i % 28) + 1:02d}")

    history = DrawHistoryDTO(dates=dates, winning_numbers=draws, concursos=concursos)
    config = PredictionConfigDTO(
        total_balls=10,
        ticket_size=5,
        num_tickets=1,
        filter_overrides={
            "gate_margin": -1.0,
            "universe_mode": "topk_scored_universe",
            "score_model": "feature_lr",
            "universe_topk_k": 200,
            "selection_mode": "ranked",
            "structural_enabled": False,
            "diversity_min_hamming": 0,
            "topk_preview": 10,
        },
    )

    pred = TrisForecastV1A().predict(history, config)
    t0 = pred.tickets[0]

    assert all(int(d) % 2 == 0 for d in t0)
    assert pred.metadata.get("universe_mode") == "topk_scored_universe"
    assert pred.metadata.get("score_model") == "feature_lr"
    assert int(pred.metadata.get("universe_topk_k", -1)) == 200
    score_stats = pred.metadata.get("score_stats", {})
    assert isinstance(score_stats, dict)
    assert score_stats.get("min") is not None
    assert score_stats.get("mean") is not None
    assert score_stats.get("max") is not None


def test_tris_forecast_topk_gate_false_falls_back_to_full_filtered_universe():
    rng = np.random.default_rng(321)
    draws = []
    concursos = []
    dates = []
    for i in range(140):
        row = rng.integers(0, 10, size=5, endpoint=False).tolist()
        mult = 1 if i % 4 == 0 else 0
        draws.append([int(row[0]), int(row[1]), int(row[2]), int(row[3]), int(row[4]), mult])
        concursos.append(4000 + i)
        dates.append(f"2025-04-{(i % 28) + 1:02d}")

    history = DrawHistoryDTO(dates=dates, winning_numbers=draws, concursos=concursos)
    config = PredictionConfigDTO(
        total_balls=10,
        ticket_size=5,
        num_tickets=1,
        filter_overrides={
            "gate_margin": -1.0,
            "universe_mode": "topk_scored_universe",
            "score_model": "feature_lr",
            "universe_topk_k": 200,
            "use_topk_gate": True,
            "topk_gate_threshold_z": 9999.0,
            "selection_mode": "ranked",
            "structural_enabled": False,
            "diversity_min_hamming": 0,
        },
    )

    pred = TrisForecastV1A().predict(history, config)
    gate_meta = pred.metadata.get("topk_gate", {})

    assert pred.metadata.get("universe_mode") == "full_filtered_universe"
    assert isinstance(gate_meta, dict)
    assert bool(gate_meta.get("enabled")) is True
    assert bool(gate_meta.get("pass")) is False


def test_tris_forecast_topk_scored_universe_ticket_ngram():
    draws = []
    concursos = []
    dates = []
    for i in range(80):
        mult = 1 if i % 2 == 0 else 0
        draws.append([0, 1, 2, 3, 4, mult])
        concursos.append(5000 + i)
        dates.append(f"2025-05-{(i % 28) + 1:02d}")

    history = DrawHistoryDTO(dates=dates, winning_numbers=draws, concursos=concursos)
    config = PredictionConfigDTO(
        total_balls=10,
        ticket_size=5,
        num_tickets=1,
        filter_overrides={
            "gate_margin": -1.0,
            "universe_mode": "topk_scored_universe",
            "score_model": "ticket_ngram",
            "universe_topk_k": 100,
            "selection_mode": "ranked",
            "structural_enabled": False,
            "diversity_min_hamming": 0,
        },
    )

    pred = TrisForecastV1A().predict(history, config)

    assert pred.tickets[0] == [0, 1, 2, 3, 4]
    assert pred.metadata.get("universe_mode") == "topk_scored_universe"
    assert pred.metadata.get("score_model") == "ticket_ngram"
    assert int(pred.metadata.get("universe_topk_k", -1)) == 100


def test_tris_forecast_topk_scored_universe_size_respects_k():
    draws = []
    concursos = []
    dates = []
    for i in range(30):
        draws.append([0, 0, 0, 0, 0, 1 if i % 2 == 0 else 0])
        concursos.append(6000 + i)
        dates.append(f"2025-06-{(i % 28) + 1:02d}")

    history = DrawHistoryDTO(dates=dates, winning_numbers=draws, concursos=concursos)
    config = PredictionConfigDTO(
        total_balls=10,
        ticket_size=5,
        num_tickets=1,
        filter_overrides={
            "universe_mode": "topk_scored_universe",
            "score_model": "feature_lr",
            "universe_topk_k": 1000,
            "selection_mode": "ranked",
            "structural_enabled": False,
            "diversity_min_hamming": 0,
        },
    )

    pred = TrisForecastV1A().predict(history, config)
    assert int(pred.metadata.get("universe_size", -1)) == 1000


def test_tris_forecast_topk_scored_universe_random_topk_metadata_and_raw_universe():
    rng = np.random.default_rng(77)
    draws = []
    concursos = []
    dates = []
    for i in range(90):
        row = rng.integers(0, 10, size=5, endpoint=False).tolist()
        draws.append([int(row[0]), int(row[1]), int(row[2]), int(row[3]), int(row[4]), i % 2])
        concursos.append(7000 + i)
        dates.append(f"2025-07-{(i % 28) + 1:02d}")

    history = DrawHistoryDTO(dates=dates, winning_numbers=draws, concursos=concursos)
    config = PredictionConfigDTO(
        total_balls=10,
        ticket_size=5,
        num_tickets=1,
        filter_overrides={
            "gate_margin": -1.0,
            "universe_mode": "topk_scored_universe",
            "score_model": "random_topk",
            "universe_topk_k": 321,
            "random_topk_seed": 12345,
            "selection_mode": "ranked",
            "structural_enabled": False,
            "diversity_min_hamming": 0,
        },
    )

    pred = TrisForecastV1A().predict(history, config)
    raw = pred.metadata.get("raw_ndarray")

    assert pred.metadata.get("universe_mode") == "topk_scored_universe"
    assert pred.metadata.get("score_model") == "random_topk"
    assert int(pred.metadata.get("universe_size", -1)) == 321
    assert isinstance(raw, np.ndarray)
    assert raw.shape == (321, 5)
    assert raw.dtype == np.uint8


def _build_camera_history(n: int = 120):
    rng = np.random.default_rng(20260226)
    draws = []
    concursos = []
    dates = []
    for i in range(n):
        row = rng.integers(0, 10, size=5, endpoint=False).tolist()
        mult = 1 if i % 3 == 0 else 0
        draws.append([int(row[0]), int(row[1]), int(row[2]), int(row[3]), int(row[4]), mult])
        concursos.append(8000 + i)
        dates.append(f"2025-08-{(i % 28) + 1:02d}")
    return DrawHistoryDTO(dates=dates, winning_numbers=draws, concursos=concursos)


def test_tris_forecast_camera_mech_topm10_does_not_reduce_full_universe():
    history = _build_camera_history(120)
    config = PredictionConfigDTO(
        total_balls=10,
        ticket_size=5,
        num_tickets=1,
        filter_overrides={
            "gate_margin": -1.0,
            "score_model": "camera_mech_v1",
            "universe_mode": "full_filtered_universe",
            "camera_masked_universe": True,
            "camera_topm_per_position": 10,
            "camera_mech_blend_with_v1a": 0.5,
            "structural_enabled": False,
            "diversity_min_hamming": 0,
        },
    )

    pred = TrisForecastV1A().predict(history, config)
    raw = pred.metadata.get("raw_ndarray")

    assert pred.metadata.get("score_model") == "camera_mech_v1"
    assert int(pred.metadata.get("universe_size", -1)) == 100000
    assert isinstance(raw, np.ndarray)
    assert raw.shape == (100000, 5)


def test_tris_forecast_camera_mech_topm2_limits_masked_universe_size_to_32_or_less():
    history = _build_camera_history(120)
    config = PredictionConfigDTO(
        total_balls=10,
        ticket_size=5,
        num_tickets=1,
        filter_overrides={
            "gate_margin": -1.0,
            "score_model": "camera_mech_v1",
            "universe_mode": "full_filtered_universe",
            "camera_masked_universe": True,
            "camera_topm_per_position": 2,
            "camera_mech_blend_with_v1a": 0.5,
            "structural_enabled": False,
            "diversity_min_hamming": 0,
        },
    )

    pred = TrisForecastV1A().predict(history, config)
    raw = pred.metadata.get("raw_ndarray")

    assert isinstance(raw, np.ndarray)
    assert raw.shape[0] <= 32
    assert int(pred.metadata.get("universe_size", -1)) <= 32


def test_tris_forecast_camera_mech_metadata_contains_camera_fields():
    history = _build_camera_history(120)
    config = PredictionConfigDTO(
        total_balls=10,
        ticket_size=5,
        num_tickets=1,
        filter_overrides={
            "gate_margin": -1.0,
            "score_model": "camera_mech_v1",
            "universe_mode": "topk_scored_universe",
            "camera_masked_universe": True,
            "camera_topm_per_position": 2,
            "universe_topk_k": 10,
            "structural_enabled": False,
            "diversity_min_hamming": 0,
        },
    )

    pred = TrisForecastV1A().predict(history, config)
    pmf = pred.metadata.get("camera_pmf")
    mask = pred.metadata.get("camera_positional_mask")

    assert isinstance(pmf, list)
    assert len(pmf) == 5
    assert all(len(row) == 10 for row in pmf)
    assert isinstance(mask, list)
    assert len(mask) == 5
    assert all(len(row) == 10 for row in mask)
    assert pred.metadata.get("camera_masked_universe") is True
    assert int(pred.metadata.get("camera_topm_per_position", -1)) == 2
    camera_debug = pred.metadata.get("camera_debug", {})
    assert isinstance(camera_debug, dict)
    assert int(camera_debug.get("post_topk_size", -1)) <= 10
    pos_unique_final = camera_debug.get("pos_unique_digits_final")
    assert isinstance(pos_unique_final, list)
    assert len(pos_unique_final) == 5
    assert all(int(v) <= 2 for v in pos_unique_final)
    assert pred.metadata.get("score_model_requested") == "camera_mech_v1"
    assert pred.metadata.get("score_model_effective") == "camera_mech_v1"


def test_tris_forecast_random_topk_with_camera_masked_universe_emits_no_invalid_warning(capsys):
    TrisForecastV1A._reset_warn_once()
    history = _build_camera_history(120)
    config = PredictionConfigDTO(
        total_balls=10,
        ticket_size=5,
        num_tickets=1,
        filter_overrides={
            "gate_margin": -1.0,
            "score_model": "random_topk",
            "universe_mode": "topk_scored_universe",
            "camera_masked_universe": True,
            "camera_topm_per_position": 2,
            "universe_topk_k": 10,
            "structural_enabled": False,
            "diversity_min_hamming": 0,
        },
    )

    TrisForecastV1A().predict(history, config)
    out = capsys.readouterr().out
    assert "camera_masked_universe=True but score_model" not in out


def test_tris_forecast_invalid_camera_masked_universe_warning_is_once(capsys):
    TrisForecastV1A._reset_warn_once()
    history = _build_camera_history(120)
    config = PredictionConfigDTO(
        total_balls=10,
        ticket_size=5,
        num_tickets=1,
        filter_overrides={
            "gate_margin": -1.0,
            "score_model": "feature_lr",
            "universe_mode": "topk_scored_universe",
            "camera_masked_universe": True,
            "camera_topm_per_position": 2,
            "universe_topk_k": 10,
            "structural_enabled": False,
            "diversity_min_hamming": 0,
        },
    )

    strategy = TrisForecastV1A()
    strategy.predict(history, config)
    strategy.predict(history, config)

    out = capsys.readouterr().out
    assert out.count("camera_masked_universe=True but score_model='feature_lr'") == 1


def test_tris_forecast_score_model_alias_positional_mech_maps_to_camera_mech_v1():
    history = _build_camera_history(120)
    config = PredictionConfigDTO(
        total_balls=10,
        ticket_size=5,
        num_tickets=1,
        filter_overrides={
            "gate_margin": -1.0,
            "score_model": "positional_mech",
            "universe_mode": "topk_scored_universe",
            "camera_masked_universe": True,
            "camera_topm_per_position": 2,
            "universe_topk_k": 10,
            "structural_enabled": False,
            "diversity_min_hamming": 0,
        },
    )

    pred = TrisForecastV1A().predict(history, config)
    assert pred.metadata.get("score_model_requested") == "positional_mech"
    assert pred.metadata.get("score_model_effective") == "camera_mech_v1"
