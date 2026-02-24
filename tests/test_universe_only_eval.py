import numpy as np

from src.strategies.tris.universe_only_eval import (
    block_bootstrap_edge_ci,
    evaluate_universe_only,
)


def test_evaluate_universe_only_mask_and_tickets_outputs():
    history = [
        [0, 1, 2, 3, 4],
        [0, 1, 2, 3, 4],
        [0, 0, 0, 0, 0],
        [1, 1, 1, 1, 1],
    ]

    def build_universe_fn(train, prev):
        idx = prev[0] * 10000 + prev[1] * 1000 + prev[2] * 100 + prev[3] * 10 + prev[4]
        mask = np.zeros(100000, dtype=bool)
        mask[idx] = True
        return mask

    df = evaluate_universe_only(history, build_universe_fn, W_train=100, start_idx=1)

    assert list(df.columns) == ["t", "U", "y", "p", "e", "fail_reason"]
    assert len(df) == 3
    assert df.iloc[0]["y"] == 1
    assert df.iloc[1]["y"] == 0
    assert isinstance(df.iloc[1]["fail_reason"], str)


def test_block_bootstrap_edge_ci_basic_shape():
    edges = np.array([0.1, -0.2, 0.0, 0.3, -0.1], dtype=np.float64)
    out = block_bootstrap_edge_ci(edges, block_size=2, n_boot=200, ci=0.90, seed=7)

    assert "edge_mean" in out and "ci_low" in out and "ci_high" in out
    assert out["n"] == 5
    assert out["ci_low"] <= out["ci_high"]
