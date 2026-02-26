from __future__ import annotations

import hashlib
from typing import Any, Dict, Tuple

import numpy as np

_ALL_TICKETS_CACHE: np.ndarray | None = None
_FEATURES_CACHE: Dict[str, np.ndarray] | None = None
_STATIC_MASK_CACHE: Dict[Tuple[Any, ...], np.ndarray] = {}
_MASKED_UNIVERSE_CACHE: Dict[Tuple[int, ...], np.ndarray] = {}
_MASKED_FEATURES_CACHE: Dict[Tuple[int, ...], Dict[str, np.ndarray]] = {}
_MASKED_STATIC_MASK_CACHE: Dict[Tuple[Tuple[int, ...], Tuple[Any, ...]], np.ndarray] = {}


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
    enable_sum = bool(_cfg_value(cfg, "enable_global_sum_filter", True))
    enable_parity = bool(_cfg_value(cfg, "enable_global_parity_filter", True))
    return (
        int(_cfg_value(cfg, "sum_min", 15)),
        int(_cfg_value(cfg, "sum_max", 30)),
        allowed_even,
        int(_cfg_value(cfg, "min_unique_digits", 3)),
        int(_cfg_value(cfg, "max_consecutive_run", 3)),
        enable_sum,
        enable_parity,
    )


def _normalize_positional_digit_mask(mask_like) -> np.ndarray:
    if mask_like is None:
        raise ValueError("positional_digit_mask no puede ser None en esta funcion.")

    arr = np.asarray(mask_like)
    if arr.shape == (5, 10):
        return arr.astype(bool, copy=False)

    if not isinstance(mask_like, (list, tuple)) or len(mask_like) != 5:
        raise ValueError("positional_digit_mask debe ser shape (5,10) o iterable de longitud 5.")

    out = np.zeros((5, 10), dtype=bool)
    for pos in range(5):
        row = mask_like[pos]
        if row is None:
            continue
        row_arr = np.asarray(row)
        if row_arr.shape == (10,):
            out[pos] = row_arr.astype(bool, copy=False)
            continue
        if not isinstance(row, (list, tuple, set, np.ndarray)):
            raise ValueError("Cada posicion de positional_digit_mask debe ser iterable.")
        for d in row:
            out[pos, int(d) % 10] = True
    return out


def _positional_mask_hash(mask: np.ndarray) -> tuple:
    mask_bool = np.asarray(mask, dtype=bool)
    if mask_bool.shape != (5, 10):
        raise ValueError("mask debe tener shape (5,10).")
    return tuple(int(v) for v in mask_bool.astype(np.uint8, copy=False).reshape(-1).tolist())


def build_masked_tickets_5d(positional_digit_mask) -> np.ndarray:
    mask = _normalize_positional_digit_mask(positional_digit_mask)
    allowed_digits = [np.where(mask[pos])[0].astype(np.uint8, copy=False) for pos in range(5)]
    if any(v.size == 0 for v in allowed_digits):
        return np.empty((0, 5), dtype=np.uint8)

    grids = np.meshgrid(*allowed_digits, indexing="ij")
    tickets = np.stack(grids, axis=-1).reshape(-1, 5)
    return tickets.astype(np.uint8, copy=False)


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


def precompute_features_for_tickets(tickets) -> Dict[str, np.ndarray]:
    tickets = np.asarray(tickets, dtype=np.uint8)
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


def precompute_static_features(all_tickets: np.ndarray) -> Dict[str, np.ndarray]:
    return precompute_features_for_tickets(all_tickets)


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
    enable_sum = bool(_cfg_value(cfg, "enable_global_sum_filter", True))
    enable_parity = bool(_cfg_value(cfg, "enable_global_parity_filter", True))

    mask = np.ones(sum_digits.shape[0], dtype=bool)
    if enable_sum:
        mask &= (sum_digits >= sum_min) & (sum_digits <= sum_max)
    if enable_parity:
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


def get_universe_with_positional_mask(
    cfg,
    positional_digit_mask=None,
    *,
    return_diag: bool = False,
):
    if positional_digit_mask is None:
        tickets, features, static_mask = get_universe_and_static_mask(cfg)
        if return_diag:
            return tickets, features, static_mask, {}
        return tickets, features, static_mask

    norm_mask = _normalize_positional_digit_mask(positional_digit_mask)
    mask_key = _positional_mask_hash(norm_mask)
    allowed_digits_per_pos = []
    empty_positions = []
    for pos in range(5):
        digits = np.where(norm_mask[pos])[0].astype(np.uint8, copy=False).tolist()
        allowed_digits_per_pos.append(
            {
                "position": int(pos),
                "count": int(len(digits)),
                "digits": [int(d) for d in digits],
            }
        )
        if len(digits) == 0:
            empty_positions.append(int(pos))

    diag = {
        "mask_hash": hashlib.sha1(
            np.asarray(norm_mask, dtype=np.uint8).tobytes()
        ).hexdigest(),
        "allowed_digits_per_pos": allowed_digits_per_pos,
        "masked_universe_size_raw": 0,
    }

    if empty_positions:
        print(
            "[TRIS][CameraMask][WARN] positional mask has empty support at positions "
            f"{empty_positions}; returning empty universe."
        )
        tickets = np.empty((0, 5), dtype=np.uint8)
        features = precompute_features_for_tickets(tickets)
        static_mask = np.empty((0,), dtype=bool)
        if return_diag:
            return tickets, features, static_mask, diag
        return tickets, features, static_mask

    tickets = _MASKED_UNIVERSE_CACHE.get(mask_key)
    if tickets is None:
        tickets = build_masked_tickets_5d(norm_mask)
        _MASKED_UNIVERSE_CACHE[mask_key] = tickets
    diag["masked_universe_size_raw"] = int(tickets.shape[0])

    features = _MASKED_FEATURES_CACHE.get(mask_key)
    if features is None:
        features = precompute_features_for_tickets(tickets)
        _MASKED_FEATURES_CACHE[mask_key] = features

    cfg_key = _cfg_hash(cfg)
    static_key = (mask_key, cfg_key)
    static_mask = _MASKED_STATIC_MASK_CACHE.get(static_key)
    if static_mask is None:
        static_mask = build_static_mask(features, cfg)
        _MASKED_STATIC_MASK_CACHE[static_key] = static_mask

    if return_diag:
        return tickets, features, static_mask, diag
    return tickets, features, static_mask


# Smoke-check reference:
# - total tickets should be exactly 100000.
# - static_mask size is typically in [30000, 70000] depending on cfg.
