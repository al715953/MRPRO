import numpy as np

from src.strategies.combinatorial.baselines import (
    score_against_random,
    summarize_distribution,
)
from src.strategies.combinatorial.statistics import (
    exact_mcnemar,
    paired_block_bootstrap_ci,
    paired_sign_permutation_test,
)


def test_distribution_summary_reports_requested_percentiles():
    summary = summarize_distribution([1, 2, 3, 4, 5])

    assert summary["mean"] == 3.0
    assert summary["median"] == 3.0
    assert summary["min"] == 1.0
    assert summary["max"] == 5.0
    assert summary["p50"] == 3.0


def test_percentile_rank_and_z_score_are_reproducible():
    score = score_against_random(4.0, [1.0, 2.0, 3.0, 4.0])

    assert score["percentile_rank"] == 100.0
    assert score["z_score"] > 1.0


def test_exact_mcnemar_counts_discordant_pairs():
    result = exact_mcnemar(
        [True, True, True, False, False],
        [False, False, True, True, False],
    )

    assert result["b_left_only"] == 2
    assert result["c_right_only"] == 1
    assert 0.0 <= result["p_exact"] <= 1.0


def test_paired_tests_return_zero_delta_for_identical_samples():
    values = np.asarray([1, 2, 3, 4, 5], dtype=float)

    permutation = paired_sign_permutation_test(values, values, trials=100, seed=3)
    bootstrap = paired_block_bootstrap_ci(values, values, trials=100, seed=4)

    assert permutation["mean_delta"] == 0.0
    assert permutation["p_two_sided"] == 1.0
    assert bootstrap["mean_delta"] == 0.0
    assert bootstrap["ci_low"] == 0.0
    assert bootstrap["ci_high"] == 0.0
