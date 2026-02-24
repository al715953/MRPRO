import numpy as np

from src.strategies.tris.universe_gate import should_use_topk


def test_should_use_topk_true_for_repeated_pattern():
    history = [[0, 0, 0, 0, 0] for _ in range(400)]
    ok = should_use_topk(
        history,
        gate_calib_size=200,
        K=1,
        alpha=1.0,
        threshold_z=1.0,
    )
    assert ok is True


def test_should_use_topk_false_for_random_history():
    rng = np.random.default_rng(17)
    history = rng.integers(0, 10, size=(500, 5), endpoint=False).tolist()
    ok = should_use_topk(
        history,
        gate_calib_size=300,
        K=1,
        alpha=1.0,
        threshold_z=1.0,
    )
    assert ok is False
