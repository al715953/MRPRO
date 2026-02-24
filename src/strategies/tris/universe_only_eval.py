from __future__ import annotations

from typing import Any, Callable, Sequence

import numpy as np
import pandas as pd

from src.strategies.tris.structural_filters import StructuralFilterConfig, StructuralFilterEngine


def _coerce_ticket(row: Sequence[int]) -> np.ndarray:
    if row is None or len(row) < 5:
        raise ValueError("Cada ticket debe contener al menos 5 digitos.")
    arr = np.asarray(row[:5], dtype=np.int16).reshape(5)
    return np.mod(arr, 10).astype(np.int16, copy=False)


def _coerce_history(history_digits_list: Sequence[Sequence[int]]) -> np.ndarray:
    rows = []
    for row in history_digits_list or []:
        if row is None or len(row) < 5:
            continue
        rows.append(_coerce_ticket(row))
    if not rows:
        return np.empty((0, 5), dtype=np.int16)
    return np.vstack(rows).astype(np.int16, copy=False)


def _ticket_to_index(ticket: np.ndarray) -> int:
    d0, d1, d2, d3, d4 = [int(v) for v in ticket[:5]]
    return d0 * 10000 + d1 * 1000 + d2 * 100 + d3 * 10 + d4


def _resolve_universe(universe_raw: Any) -> tuple[np.ndarray | None, np.ndarray | None, int]:
    mask = None
    tickets = None

    if isinstance(universe_raw, dict):
        if "mask" in universe_raw:
            mask = np.asarray(universe_raw["mask"], dtype=bool).reshape(-1)
        if "tickets" in universe_raw:
            tickets = np.asarray(universe_raw["tickets"])
        elif "universe" in universe_raw:
            tickets = np.asarray(universe_raw["universe"])
    elif isinstance(universe_raw, (tuple, list)) and len(universe_raw) == 2:
        first = np.asarray(universe_raw[0])
        if first.ndim == 1 and first.dtype == np.bool_:
            mask = np.asarray(universe_raw[0], dtype=bool).reshape(-1)
            tickets = np.asarray(universe_raw[1])
        else:
            tickets = np.asarray(universe_raw)
    else:
        arr = np.asarray(universe_raw)
        if arr.ndim == 1 and arr.dtype == np.bool_:
            mask = arr.astype(bool, copy=False).reshape(-1)
        else:
            tickets = arr

    if tickets is not None and tickets.size > 0:
        if tickets.ndim == 1 and tickets.shape[0] == 5:
            tickets = tickets.reshape(1, 5)
        if tickets.ndim != 2 or tickets.shape[1] < 5:
            raise ValueError("Universo en formato tickets debe tener shape (N, 5).")
        tickets = np.mod(tickets[:, :5].astype(np.int16, copy=False), 10)
        return mask, tickets, int(tickets.shape[0])

    if mask is not None:
        return mask, None, int(np.sum(mask))

    return None, np.empty((0, 5), dtype=np.int16), 0


def _contains_winner(mask: np.ndarray | None, tickets: np.ndarray | None, winner: np.ndarray) -> int:
    if tickets is not None:
        hits = np.all(tickets == winner[None, :], axis=1)
        return int(np.any(hits))

    if mask is None:
        return 0

    if mask.shape[0] == 100000:
        idx = _ticket_to_index(winner)
        return int(bool(mask[idx]))
    return 0


def _infer_engine_cfg(build_universe_fn: Callable) -> tuple[StructuralFilterEngine, Any]:
    owners = [build_universe_fn, getattr(build_universe_fn, "__self__", None)]
    for owner in owners:
        if owner is None:
            continue
        engine = getattr(owner, "structural_engine", None) or getattr(owner, "engine", None)
        if isinstance(engine, StructuralFilterEngine):
            return engine, engine.config
        cfg = getattr(owner, "structural_cfg", None) or getattr(owner, "cfg", None)
        if cfg is not None:
            return StructuralFilterEngine(cfg), cfg

    cfg_default = StructuralFilterConfig()
    return StructuralFilterEngine(cfg_default), cfg_default


def block_bootstrap_edge_ci(
    edge_values: Sequence[float],
    *,
    block_size: int = 20,
    n_boot: int = 2000,
    ci: float = 0.95,
    seed: int | None = 12345,
) -> dict:
    values = np.asarray(edge_values, dtype=np.float64).reshape(-1)
    n = int(values.size)
    if n == 0:
        return {
            "edge_mean": float("nan"),
            "ci_low": float("nan"),
            "ci_high": float("nan"),
            "n": 0,
            "block_size": int(max(1, block_size)),
            "n_boot": int(max(1, n_boot)),
            "ci": float(ci),
        }

    alpha = (1.0 - float(ci)) / 2.0
    alpha = min(max(alpha, 0.0), 0.5)
    bs = int(max(1, min(int(block_size), n)))
    b_n = int(max(1, n_boot))
    n_blocks = int(np.ceil(n / bs))
    rng = np.random.default_rng(seed)

    means = np.empty(b_n, dtype=np.float64)
    for b in range(b_n):
        sample = np.empty(n_blocks * bs, dtype=np.float64)
        for k in range(n_blocks):
            start = int(rng.integers(0, n - bs + 1)) if n > bs else 0
            sample[k * bs : (k + 1) * bs] = values[start : start + bs]
        means[b] = float(np.mean(sample[:n]))

    ci_low, ci_high = np.quantile(means, [alpha, 1.0 - alpha]).tolist()
    return {
        "edge_mean": float(np.mean(values)),
        "ci_low": float(ci_low),
        "ci_high": float(ci_high),
        "n": n,
        "block_size": bs,
        "n_boot": b_n,
        "ci": float(ci),
    }


def evaluate_universe_only(
    history_digits_list: Sequence[Sequence[int]],
    build_universe_fn: Callable[[Sequence[Sequence[int]], Sequence[int]], Any],
    *,
    W_train: int,
    start_idx: int,
) -> pd.DataFrame:
    history = _coerce_history(history_digits_list)
    n = int(history.shape[0])
    if n == 0:
        return pd.DataFrame(columns=["t", "U", "y", "p", "e", "fail_reason"])

    train_window = int(max(1, W_train))
    t0 = int(max(1, start_idx))
    engine, cfg = _infer_engine_cfg(build_universe_fn)

    rows = []
    for t in range(t0, n):
        train_slice = history[:t]
        train_rows = train_slice[-min(train_slice.shape[0], train_window) :]
        prev = history[t - 1]
        winner = history[t]

        universe_raw = build_universe_fn(train_rows.tolist(), prev.tolist())
        mask, tickets, u_size = _resolve_universe(universe_raw)
        y = _contains_winner(mask, tickets, winner)
        p = float(u_size / 100000.0)
        e = float(y - p)

        fail_reason = None
        if y == 0:
            prev_arr = StructuralFilterEngine._coerce_prev(prev, 5)
            violations = engine._violations(winner.tolist(), prev_arr, cfg)
            fail_reason = "|".join(sorted(set(violations))) if violations else "not_in_universe"

        rows.append(
            {
                "t": int(t),
                "U": int(u_size),
                "y": int(y),
                "p": p,
                "e": e,
                "fail_reason": fail_reason,
            }
        )

    return pd.DataFrame(rows, columns=["t", "U", "y", "p", "e", "fail_reason"])

