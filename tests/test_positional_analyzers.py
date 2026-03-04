import numpy as np

from src.strategies.tris.positional_analyzers import PositionalAnalyzers


def test_latency_updates_by_position():
    history = [
        [1, 2, 3, 4, 5],
        [1, 2, 3, 4, 5],
        [9, 8, 7, 6, 5],
        [1, 2, 3, 4, 5],
    ]
    model = PositionalAnalyzers().fit(history)

    # Pos0: digito 1 aparece al final => latencia 0; digito 9 aparecio en t=2 => latencia 1
    assert int(model.latency[0, 1]) == 0
    assert int(model.latency[0, 9]) == 1
    # Pos1: digito 8 aparecio por ultima vez en t=2 => latencia 1
    assert int(model.latency[1, 8]) == 1


def test_predict_pmf_sums_to_one_per_position():
    rng = np.random.default_rng(7)
    history = rng.integers(0, 10, size=(250, 5), endpoint=False).tolist()
    model = PositionalAnalyzers(
        alpha=1.0,
        short_window=80,
        long_window=200,
        mix_lambda=0.4,
    ).fit(history)
    out = model.predict()

    pmf = out["pmf"]
    assert pmf.shape == (5, 10)
    np.testing.assert_allclose(np.sum(pmf, axis=1), np.ones(5), atol=1e-12)


def test_topm_per_position_three_is_deterministic_and_exact():
    # Sin historial: PMF uniforme y empates completos.
    model = PositionalAnalyzers(topm_per_position=3).fit([])
    out = model.predict()
    mask = out["positional_mask"]

    assert mask.shape == (5, 10)
    assert np.all(np.sum(mask, axis=1) == 3)
    assert out["favored_digits_by_pos"] == [[0, 1, 2]] * 5


def test_immediate_repeat_penalty_reduces_prev_digit_probability():
    history = [[4, 4, 4, 4, 4] for _ in range(120)] + [[1, 1, 1, 1, 1] for _ in range(120)]
    prev_digits = [1, 1, 1, 1, 1]

    base = PositionalAnalyzers(
        alpha=1.0,
        short_window=120,
        long_window=240,
        mix_lambda=0.5,
        immediate_repeat_penalty=0.0,
    ).fit(history)
    penalized = PositionalAnalyzers(
        alpha=1.0,
        short_window=120,
        long_window=240,
        mix_lambda=0.5,
        immediate_repeat_penalty=2.0,
    ).fit(history)

    pmf_base = base.predict(prev_digits=prev_digits)["pmf"]
    pmf_pen = penalized.predict(prev_digits=prev_digits)["pmf"]

    assert float(pmf_pen[0, 1]) < float(pmf_base[0, 1])


def test_coverage_mask_uniform_target_070_selects_7_digits_per_position():
    model = PositionalAnalyzers(
        mask_mode="coverage",
        target_coverage_per_position=0.70,
        min_digits_per_position=1,
        max_digits_per_position=10,
    ).fit([])

    out = model.predict()
    mask = out["positional_mask"]
    diag = out["diagnostics"]

    assert mask.shape == (5, 10)
    assert np.all(np.sum(mask, axis=1) == 7)
    assert np.all(np.asarray(diag["mask_digits_per_pos"]) == 7)
    np.testing.assert_allclose(
        np.asarray(diag["mask_coverage_empirical_per_pos"], dtype=np.float64),
        np.full(5, 0.7, dtype=np.float64),
        atol=1e-12,
    )


def test_coverage_mask_concentrated_can_use_fewer_digits_respecting_min_digits():
    history = [[0, 0, 0, 0, 0] for _ in range(200)]
    model = PositionalAnalyzers(
        alpha=0.01,
        short_window=200,
        long_window=200,
        mix_lambda=1.0,
        mask_mode="coverage",
        target_coverage_per_position=0.70,
        min_digits_per_position=2,
        max_digits_per_position=10,
    ).fit(history)

    out = model.predict()
    mask = out["positional_mask"]
    diag = out["diagnostics"]

    assert np.all(np.sum(mask, axis=1) == 2)
    assert np.all(np.asarray(diag["mask_digits_per_pos"]) == 2)
    assert np.all(np.asarray(diag["mask_coverage_empirical_per_pos"]) >= 0.70)


def test_coverage_mask_accepts_per_position_targets():
    targets = [0.55, 0.60, 0.65, 0.70, 0.75]
    model = PositionalAnalyzers(
        mask_mode="coverage",
        target_coverage_per_position=targets,
        min_digits_per_position=1,
        max_digits_per_position=10,
    ).fit([])

    out = model.predict()
    diag = out["diagnostics"]

    np.testing.assert_allclose(
        np.asarray(diag["target_coverage_per_pos_effective"], dtype=np.float64),
        np.asarray(targets, dtype=np.float64),
        atol=1e-12,
    )
    np.testing.assert_array_equal(
        np.asarray(diag["mask_digits_per_pos"], dtype=np.int32),
        np.array([6, 6, 7, 7, 8], dtype=np.int32),
    )


def test_adaptive_coverage_expands_more_for_volatile_camera():
    rng = np.random.default_rng(20260302)
    history = []
    for _ in range(240):
        history.append([1, 2, 3, 4, int(rng.integers(0, 10))])

    model = PositionalAnalyzers(
        alpha=0.5,
        short_window=60,
        long_window=240,
        mix_lambda=0.5,
        mask_mode="coverage",
        target_coverage_per_position=0.60,
        adaptive_coverage_enabled=True,
        adaptive_coverage_base=0.60,
        adaptive_coverage_min=0.55,
        adaptive_coverage_max=0.90,
        adaptive_coverage_volatility_gain=0.30,
        min_digits_per_position=1,
        max_digits_per_position=10,
    ).fit(history)

    out = model.predict()
    diag = out["diagnostics"]
    target_cov = np.asarray(diag["target_coverage_per_pos_effective"], dtype=np.float64)
    volatility = np.asarray(diag["volatility_pos"], dtype=np.float64)
    mask_digits = np.asarray(diag["mask_digits_per_pos"], dtype=np.int32)

    assert float(volatility[4]) > float(volatility[0])
    assert float(target_cov[4]) > float(target_cov[0])
    assert int(mask_digits[4]) >= int(mask_digits[0])


def test_coverage_outputs_valid_pmf_mask_and_diagnostics_shapes():
    rng = np.random.default_rng(101)
    history = rng.integers(0, 10, size=(180, 5), endpoint=False).tolist()
    model = PositionalAnalyzers(
        mask_mode="coverage",
        target_coverage_per_position=0.65,
        min_digits_per_position=2,
        max_digits_per_position=6,
    ).fit(history)

    out = model.predict()
    pmf = out["pmf"]
    mask = out["positional_mask"]
    diag = out["diagnostics"]

    assert pmf.shape == (5, 10)
    assert mask.shape == (5, 10)
    assert mask.dtype == np.bool_
    np.testing.assert_allclose(np.sum(pmf, axis=1), np.ones(5), atol=1e-12)
    assert np.asarray(diag["mask_digits_per_pos"]).shape == (5,)
    assert np.asarray(diag["mask_coverage_empirical_per_pos"]).shape == (5,)
    assert diag["positional_mask_mode"] == "coverage"
    assert float(diag["target_coverage_per_position"]) == 0.65


def test_slot_conditioning_blends_towards_slot_distribution():
    history = [[9, 9, 9, 9, 9] for _ in range(60)] + [[0, 0, 0, 0, 0] for _ in range(60)]
    slot_labels = ["mediodia"] * 60 + ["clasico"] * 60

    model = PositionalAnalyzers(
        alpha=0.1,
        short_window=120,
        long_window=120,
        mix_lambda=1.0,
        camera_slot_gamma=1.0,
    ).fit(history, slot_labels=slot_labels)

    out_global = model.predict(slot_context="unknown")
    out_slot = model.predict(slot_context="mediodia")

    pmf_global = np.asarray(out_global["pmf"], dtype=np.float64)
    pmf_slot = np.asarray(out_slot["pmf"], dtype=np.float64)
    diag = out_slot["diagnostics"]

    assert pmf_slot.shape == (5, 10)
    assert float(pmf_slot[0, 9]) > float(pmf_global[0, 9])
    assert float(pmf_slot[0, 9]) > 0.65
    assert diag["slot_context_used"] == "mediodia"
    assert int(diag["slot_sample_size"]) == 60
    assert float(diag["slot_blend_gamma"]) > 0.0
    assert np.asarray(diag["slot_vs_global_l1_by_pos"]).shape == (5,)
    assert float(np.mean(np.asarray(diag["slot_vs_global_l1_by_pos"]))) > 0.0


def test_slot_conditioning_degrades_to_global_without_slot_labels():
    history = [[7, 7, 7, 7, 7] for _ in range(90)] + [[1, 1, 1, 1, 1] for _ in range(90)]
    model = PositionalAnalyzers(
        alpha=0.5,
        short_window=120,
        long_window=180,
        mix_lambda=0.5,
        camera_slot_gamma=1.0,
    ).fit(history)

    out_global = model.predict(slot_context=None)
    out_slot = model.predict(slot_context="mediodia")
    diag = out_slot["diagnostics"]

    np.testing.assert_allclose(
        np.asarray(out_slot["pmf"], dtype=np.float64),
        np.asarray(out_global["pmf"], dtype=np.float64),
        atol=1e-12,
    )
    assert diag["slot_context_used"] == "mediodia"
    assert int(diag["slot_sample_size"]) == 0
    assert float(diag["slot_blend_gamma"]) == 0.0
