from __future__ import annotations

from dataclasses import replace
from typing import Sequence

import numpy as np

from src.strategies.tris.structural_filters import StructuralFilterConfig, StructuralFilterEngine
from src.strategies.tris.universe_5d import get_universe_and_static_mask


def _coerce_digits_history(digits_history: Sequence[Sequence[int]]) -> np.ndarray:
    rows = []
    for row in digits_history or []:
        if row is None or len(row) < 5:
            continue
        vals = []
        for i in range(5):
            try:
                d = int(float(row[i]))
            except Exception:
                d = 0
            vals.append(d % 10)
        rows.append(vals)
    if not rows:
        return np.empty((0, 5), dtype=np.int16)
    return np.asarray(rows, dtype=np.int16)


def _ticket_to_index(ticket: np.ndarray) -> int:
    d0, d1, d2, d3, d4 = [int(v) for v in ticket[:5]]
    return d0 * 10000 + d1 * 1000 + d2 * 100 + d3 * 10 + d4


def _build_variants(base_cfg: StructuralFilterConfig) -> list[tuple[str, StructuralFilterConfig]]:
    # "sum=None" is represented as full-range bounds (0..45) to stay compatible with current mask builder.
    no_sum = replace(base_cfg, sum_min=0, sum_max=45)
    # "parity_mode=any" is represented as allowing all even-count values.
    no_parity = replace(base_cfg, allowed_even_counts=(0, 1, 2, 3, 4, 5))
    no_uniques = replace(base_cfg, min_unique_digits=1)
    no_consecutive = replace(base_cfg, max_consecutive_run=5)
    # "max_mirror_matches=5" maps to max_positional_repeats_vs_prev=5.
    no_mirror_prev = replace(base_cfg, max_positional_repeats_vs_prev=5)

    return [
        ("base", base_cfg),
        ("no_sum", no_sum),
        ("no_parity", no_parity),
        ("no_uniques", no_uniques),
        ("no_consecutive", no_consecutive),
        ("no_mirror_prev", no_mirror_prev),
    ]


def _summarize_variant(
    name: str,
    cfg: StructuralFilterConfig,
    history: np.ndarray,
    *,
    start: int,
    end: int,
) -> dict:
    all_tickets, _, static_mask = get_universe_and_static_mask(cfg)

    u_vals = []
    y_vals = []
    p_vals = []
    e_vals = []
    for t in range(start, end):
        prev = history[t - 1]
        winner = history[t]
        final_mask = StructuralFilterEngine.mask_all(all_tickets, prev, static_mask, cfg)

        u = int(np.sum(final_mask))
        y = int(bool(final_mask[_ticket_to_index(winner)]))
        p = float(u / 100000.0)
        e = float(y - p)

        u_vals.append(u)
        y_vals.append(y)
        p_vals.append(p)
        e_vals.append(e)

    u_arr = np.asarray(u_vals, dtype=np.float64)
    y_arr = np.asarray(y_vals, dtype=np.float64)
    p_arr = np.asarray(p_vals, dtype=np.float64)
    e_arr = np.asarray(e_vals, dtype=np.float64)

    n = int(u_arr.size)
    if n == 0:
        return {
            "variant": name,
            "n_eval": 0,
            "AvgU": float("nan"),
            "std_U": float("nan"),
            "FS_pass": float("nan"),
            "fails": 0,
            "edge_mean": float("nan"),
            "z_edge": float("nan"),
        }

    avg_u = float(np.mean(u_arr))
    std_u = float(np.std(u_arr))
    fs_pass = float(np.mean(y_arr))
    fails = int(np.sum(1.0 - y_arr))
    edge_mean = float(np.mean(e_arr))
    var_mean = float(np.mean(p_arr * (1.0 - p_arr)))
    denom = float(np.sqrt(var_mean / n)) if n > 0 else 0.0
    if denom <= 0.0:
        z_edge = float("inf") if edge_mean > 0 else (float("-inf") if edge_mean < 0 else 0.0)
    else:
        z_edge = float(edge_mean / denom)

    return {
        "variant": name,
        "n_eval": n,
        "AvgU": avg_u,
        "std_U": std_u,
        "FS_pass": fs_pass,
        "fails": fails,
        "edge_mean": edge_mean,
        "z_edge": z_edge,
    }


def ablation_study(
    digits_history,
    base_cfg: StructuralFilterConfig,
    *,
    start: int = 50,
    end=None,
) -> list[dict]:
    history = _coerce_digits_history(digits_history)
    n_total = int(history.shape[0])
    if n_total < 2:
        return []

    s = int(max(1, start))
    e = int(n_total if end is None else min(int(end), n_total))
    if s >= e:
        return []

    variants = _build_variants(base_cfg)
    rows = [
        _summarize_variant(name, cfg, history, start=s, end=e) for name, cfg in variants
    ]

    base_row = next((r for r in rows if r["variant"] == "base"), None)
    if base_row is None:
        return rows

    avg_u_base = float(base_row.get("AvgU", np.nan))
    fails_base = int(base_row.get("fails", 0))

    out = []
    for row in rows:
        row_out = dict(row)
        delta_u = float(avg_u_base - float(row_out.get("AvgU", np.nan)))
        delta_fails = int(int(row_out.get("fails", 0)) - fails_base)
        row_out["delta_U"] = delta_u
        row_out["delta_fails"] = delta_fails
        row_out["tickets_removed_per_extra_fail"] = float(
            delta_u / max(delta_fails, 1)
        )
        out.append(row_out)
    return out

