from __future__ import annotations

import math

import numpy as np


def exact_mcnemar(left_success, right_success) -> dict:
    left = np.asarray(left_success, dtype=bool)
    right = np.asarray(right_success, dtype=bool)
    if left.shape != right.shape:
        raise ValueError("McNemar requiere observaciones pareadas")
    b = int(np.sum(left & ~right))
    c = int(np.sum(~left & right))
    discordant = b + c
    if discordant == 0:
        p_value = 1.0
    else:
        tail = sum(
            math.comb(discordant, value)
            for value in range(0, min(b, c) + 1)
        ) / (2**discordant)
        p_value = min(1.0, 2.0 * tail)
    return {"b_left_only": b, "c_right_only": c, "p_exact": float(p_value)}


def paired_sign_permutation_test(
    left,
    right,
    *,
    trials: int = 10_000,
    seed: int = 20260816,
) -> dict:
    left_array = np.asarray(left, dtype=np.float64)
    right_array = np.asarray(right, dtype=np.float64)
    if left_array.shape != right_array.shape:
        raise ValueError("La permutación requiere observaciones pareadas")
    delta = left_array - right_array
    observed = float(np.mean(delta)) if delta.size else 0.0
    if delta.size == 0:
        return {"mean_delta": observed, "p_two_sided": None, "trials": 0}
    rng = np.random.default_rng(int(seed))
    exceed = 0
    trial_count = max(1, int(trials))
    for _ in range(trial_count):
        signs = rng.choice(np.asarray([-1.0, 1.0]), size=delta.size)
        if abs(float(np.mean(delta * signs))) >= abs(observed):
            exceed += 1
    return {
        "mean_delta": observed,
        "p_two_sided": float((exceed + 1) / (trial_count + 1)),
        "trials": trial_count,
    }


def paired_block_bootstrap_ci(
    left,
    right,
    *,
    block_size: int | None = None,
    trials: int = 5_000,
    seed: int = 20260817,
) -> dict:
    delta = np.asarray(left, dtype=np.float64) - np.asarray(right, dtype=np.float64)
    n = int(delta.size)
    if n == 0:
        return {"mean_delta": None, "ci_low": None, "ci_high": None, "trials": 0}
    block = max(1, min(n, int(block_size or max(2, math.sqrt(n)))))
    blocks_needed = int(math.ceil(n / block))
    rng = np.random.default_rng(int(seed))
    means = np.empty(max(1, int(trials)), dtype=np.float64)
    offsets = np.arange(block, dtype=np.int64)
    for trial in range(means.size):
        starts = rng.integers(0, n, size=blocks_needed)
        indices = np.concatenate([(start + offsets) % n for start in starts])[:n]
        means[trial] = float(np.mean(delta[indices]))
    low, high = np.percentile(means, [2.5, 97.5])
    return {
        "mean_delta": float(np.mean(delta)),
        "ci_low": float(low),
        "ci_high": float(high),
        "block_size": block,
        "trials": int(means.size),
    }
