import numpy as np

from src.core.prob_metrics import brier_positional, ece_positional, logloss_positional


def _mean_metric_over_samples(metric_fn, pos_probs_batch, y_batch) -> float:
    values = [metric_fn(pos_probs_batch[i], y_batch[i]) for i in range(pos_probs_batch.shape[0])]
    return float(np.mean(values))


def test_prob_metrics_uniform_baseline_sanity():
    rng = np.random.default_rng(20260221)
    n_samples = 5000
    pos_probs = np.full((n_samples, 5, 10), 0.1, dtype=np.float64)
    y_digits = rng.integers(0, 10, size=(n_samples, 5), endpoint=False)

    logloss = _mean_metric_over_samples(logloss_positional, pos_probs, y_digits)
    brier = _mean_metric_over_samples(brier_positional, pos_probs, y_digits)
    # ECE expected to be near 0 with standard definition over a dataset:
    # conf=max(p), pred=argmax(p), correct=(pred==y), binned/weighted by confidence.
    ece = float(ece_positional(pos_probs, y_digits, n_bins=10))

    assert abs(logloss - np.log(10.0)) < 0.02
    assert abs(brier - 0.9) < 0.01
    assert ece < 0.03


def test_prob_metrics_perfect_predictor_sanity():
    rng = np.random.default_rng(17)
    n_samples = 1000
    y_digits = rng.integers(0, 10, size=(n_samples, 5), endpoint=False)
    pos_probs = np.zeros((n_samples, 5, 10), dtype=np.float64)
    sample_idx = np.arange(n_samples)[:, None]
    pos_idx = np.arange(5)[None, :]
    pos_probs[sample_idx, pos_idx, y_digits] = 1.0

    logloss = _mean_metric_over_samples(logloss_positional, pos_probs, y_digits)
    brier = _mean_metric_over_samples(brier_positional, pos_probs, y_digits)
    ece = float(ece_positional(pos_probs, y_digits, n_bins=10))

    assert logloss < 1e-6
    assert brier < 1e-6
    assert ece < 1e-6


def test_prob_metrics_overconfident_wrong_predictor_sanity():
    rng = np.random.default_rng(7)
    n_samples = 3000
    y_digits = rng.integers(0, 10, size=(n_samples, 5), endpoint=False)
    wrong_digits = (y_digits + 1) % 10

    pos_probs = np.full((n_samples, 5, 10), 0.01 / 9.0, dtype=np.float64)
    sample_idx = np.arange(n_samples)[:, None]
    pos_idx = np.arange(5)[None, :]
    pos_probs[sample_idx, pos_idx, wrong_digits] = 0.99

    logloss = _mean_metric_over_samples(logloss_positional, pos_probs, y_digits)
    brier = _mean_metric_over_samples(brier_positional, pos_probs, y_digits)
    ece = float(ece_positional(pos_probs, y_digits, n_bins=10))

    assert logloss > 5.0
    assert brier > 1.9
    assert ece > 0.9
