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
