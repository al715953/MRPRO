from __future__ import annotations

from typing import Any, Dict, Tuple

import numpy as np

_ALL_TICKETS_CACHE: np.ndarray | None = None
_FEATURES_CACHE: Dict[str, np.ndarray] | None = None
_STATIC_MASK_CACHE: Dict[Tuple[Any, ...], np.ndarray] = {}


def _cfg_value(cfg, key: str, default):
    if cfg is None:
        return default
    if isinstance(cfg, dict):
        return cfg.get(key, default)
    return getattr(cfg, key, default)


def _cfg_hash(cfg) -> Tuple[Any, ...]:
    allowed_even = _cfg_value(cfg, "allowed_even_counts", (2, 3))
    if not isinstance(allowed_even, (list, tuple, set)):
        allowed_even = (allowed_even,)
    allowed_even = tuple(sorted(int(v) for v in allowed_even))
    return (
        int(_cfg_value(cfg, "sum_min", 15)),
        int(_cfg_value(cfg, "sum_max", 30)),
        allowed_even,
        int(_cfg_value(cfg, "min_unique_digits", 3)),
        int(_cfg_value(cfg, "max_consecutive_run", 3)),
    )


def build_all_tickets_5d() -> np.ndarray:
    idx = np.arange(100000, dtype=np.int32)
    digits = np.stack(
        [
            idx // 10000,
            (idx // 1000) % 10,
            (idx // 100) % 10,
            (idx // 10) % 10,
            idx % 10,
        ],
        axis=1,
    )
    return digits.astype(np.uint8, copy=False)


def precompute_static_features(all_tickets: np.ndarray) -> Dict[str, np.ndarray]:
    tickets = np.asarray(all_tickets, dtype=np.uint8)
    sum_digits = np.sum(tickets, axis=1, dtype=np.uint16).astype(np.uint8, copy=False)
    even_count = np.sum((tickets % 2) == 0, axis=1, dtype=np.uint8)

    digits_ref = np.arange(10, dtype=np.uint8)
    unique_count = np.sum(
        np.any(tickets[:, :, None] == digits_ref[None, None, :], axis=1),
        axis=1,
        dtype=np.uint8,
    )

    diffs = np.diff(tickets.astype(np.int16), axis=1)
    run_plus = ((diffs[:, 0] == 1) & (diffs[:, 1] == 1) & (diffs[:, 2] == 1)) | (
        (diffs[:, 1] == 1) & (diffs[:, 2] == 1) & (diffs[:, 3] == 1)
    )
    run_minus = ((diffs[:, 0] == -1) & (diffs[:, 1] == -1) & (diffs[:, 2] == -1)) | (
        (diffs[:, 1] == -1) & (diffs[:, 2] == -1) & (diffs[:, 3] == -1)
    )
    consecutive_run_ge4 = run_plus | run_minus

    return {
        "sum_digits": sum_digits,
        "even_count": even_count,
        "unique_count": unique_count,
        "consecutive_run_ge4": consecutive_run_ge4.astype(bool, copy=False),
    }


def build_static_mask(features: Dict[str, np.ndarray], cfg) -> np.ndarray:
    sum_digits = np.asarray(features["sum_digits"])
    even_count = np.asarray(features["even_count"])
    unique_count = np.asarray(features["unique_count"])
    consecutive_run_ge4 = np.asarray(features["consecutive_run_ge4"], dtype=bool)

    sum_min = int(_cfg_value(cfg, "sum_min", 15))
    sum_max = int(_cfg_value(cfg, "sum_max", 30))
    allowed_even = _cfg_value(cfg, "allowed_even_counts", (2, 3))
    if not isinstance(allowed_even, (list, tuple, set)):
        allowed_even = (allowed_even,)
    allowed_even = np.array([int(v) for v in allowed_even], dtype=np.int16)
    min_unique = int(_cfg_value(cfg, "min_unique_digits", 3))
    max_consecutive = int(_cfg_value(cfg, "max_consecutive_run", 3))

    mask = (sum_digits >= sum_min) & (sum_digits <= sum_max)
    mask &= np.isin(even_count.astype(np.int16), allowed_even)
    mask &= unique_count >= min_unique
    if max_consecutive == 3:
        mask &= ~consecutive_run_ge4
    return mask.astype(bool, copy=False)


def get_universe_and_static_mask(cfg):
    global _ALL_TICKETS_CACHE, _FEATURES_CACHE

    if _ALL_TICKETS_CACHE is None:
        _ALL_TICKETS_CACHE = build_all_tickets_5d()
    if _FEATURES_CACHE is None:
        _FEATURES_CACHE = precompute_static_features(_ALL_TICKETS_CACHE)

    cfg_key = _cfg_hash(cfg)
    static_mask = _STATIC_MASK_CACHE.get(cfg_key)
    if static_mask is None:
        static_mask = build_static_mask(_FEATURES_CACHE, cfg)
        _STATIC_MASK_CACHE[cfg_key] = static_mask

    return _ALL_TICKETS_CACHE, _FEATURES_CACHE, static_mask


# Smoke-check reference:
# - total tickets should be exactly 100000.
# - static_mask size is typically in [30000, 70000] depending on cfg.
