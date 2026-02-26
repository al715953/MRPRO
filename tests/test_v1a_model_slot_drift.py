import numpy as np

from src.strategies.tris.v1a_model import TrisV1AModel, analyze_slot_drift


def test_analyze_slot_drift_detects_synthetic_slot_bias():
    rng = np.random.default_rng(20260226)
    digits_list = []
    ctx_list = []
    for i in range(300):
        row = rng.integers(0, 10, size=5, endpoint=False).tolist()
        if i % 2 == 0:
            row[0] = 0
            ctx_list.append({"slot": "mediodia"})
        else:
            row[0] = 9
            ctx_list.append({"slot": "clasico"})
        digits_list.append([int(v) for v in row])

    out = analyze_slot_drift(digits_list, ctx_list, slot_a="mediodia", slot_b="clasico")

    assert int(out["sample_size_a"]) > 0
    assert int(out["sample_size_b"]) > 0
    assert float(out["l1_by_pos"][0]) > 0.0
    assert float(out["chi2_like_by_pos"][0]) > 0.0


def test_apply_positional_logit_bias_keeps_valid_pmf():
    model = TrisV1AModel()
    pos_probs = np.full((5, 10), 0.1, dtype=np.float64)
    bias = np.zeros((5, 10), dtype=np.float64)
    bias[:, 7] = 1.5

    out = model.apply_positional_logit_bias(pos_probs, bias)

    assert out.shape == (5, 10)
    np.testing.assert_allclose(np.sum(out, axis=1), np.ones(5), atol=1e-12)
    assert np.all(out[:, 7] > 0.1)
