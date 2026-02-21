from __future__ import annotations

import json
import os
import sys
import time
from dataclasses import dataclass
from itertools import product
from typing import Dict, List, Sequence, Tuple

import numpy as np
import pandas as pd

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from src.data_access.loader import LotteryLoader
from src.data_access.config import get_lottery_profile
from src.strategies.tris.structural_filters import StructuralFilterConfig
from src.strategies.tris.universe_5d import get_universe_and_static_mask


POPCOUNT = np.array([bin(i).count("1") for i in range(32)], dtype=np.int8)
MASK_POSITIONS = {
    mask: tuple(i for i in range(5) if (mask >> i) & 1) for mask in range(32)
}
MASK_MULTIPLIERS = {
    mask: (
        (10 ** np.arange(len(MASK_POSITIONS[mask]) - 1, -1, -1)).astype(np.int32)
        if MASK_POSITIONS[mask]
        else np.array([], dtype=np.int32)
    )
    for mask in range(32)
}


@dataclass(frozen=True)
class StaticGridConfig:
    sum_min: int
    sum_max: int
    parity_name: str
    allowed_even_counts: Tuple[int, ...]
    min_unique_digits: int
    max_consecutive_run: int = 3


def _coerce_concurso(value) -> Tuple[int, str]:
    try:
        return (int(value), str(value))
    except Exception:
        return (0, str(value))


def load_tris_draws() -> Tuple[List[str], np.ndarray]:
    profile = get_lottery_profile("tris_multiplicador")
    history = LotteryLoader(profile).load_data()

    triples = sorted(
        zip(history.concursos, history.winning_numbers),
        key=lambda x: _coerce_concurso(x[0]),
    )
    concursos: List[str] = []
    digits_rows: List[List[int]] = []
    for concurso, draw in triples:
        if not draw or len(draw) < 5:
            continue
        digits = [int(draw[i]) % 10 for i in range(5)]
        concursos.append(str(concurso))
        digits_rows.append(digits)

    if not digits_rows:
        raise ValueError("No se encontraron sorteos Tris válidos.")

    return concursos, np.asarray(digits_rows, dtype=np.uint8)


def build_grid() -> List[StaticGridConfig]:
    sum_ranges = [
        (8, 37),
        (10, 35),
        (12, 33),
        (13, 32),
        (14, 31),
        (15, 30),
        (16, 29),
        (17, 28),
    ]
    parity_sets = {
        "allow_all": (0, 1, 2, 3, 4, 5),
        "balanced_only": (2, 3),
        "moderate": (1, 2, 3, 4),
        "relaxed": (1, 2, 3, 4, 5),
    }
    unique_thresholds = [2, 3, 4]

    grid: List[StaticGridConfig] = []
    for (smin, smax), (pname, pevens), min_u in product(
        sum_ranges, parity_sets.items(), unique_thresholds
    ):
        grid.append(
            StaticGridConfig(
                sum_min=int(smin),
                sum_max=int(smax),
                parity_name=str(pname),
                allowed_even_counts=tuple(int(x) for x in pevens),
                min_unique_digits=int(min_u),
                max_consecutive_run=3,
            )
        )
    return grid


def build_subset_tables(static_tickets: np.ndarray) -> Dict[int, np.ndarray]:
    tables: Dict[int, np.ndarray] = {0: np.array([int(static_tickets.shape[0])], dtype=np.int32)}
    if static_tickets.size == 0:
        for mask in range(1, 32):
            tables[mask] = np.zeros(10 ** len(MASK_POSITIONS[mask]), dtype=np.int32)
        return tables

    ints = static_tickets.astype(np.int32, copy=False)
    for mask in range(1, 32):
        positions = MASK_POSITIONS[mask]
        multipliers = MASK_MULTIPLIERS[mask]
        keys = (ints[:, positions] * multipliers[None, :]).sum(axis=1)
        tables[mask] = np.bincount(keys, minlength=10 ** len(positions)).astype(
            np.int32, copy=False
        )
    return tables


def lookup_match_superset_counts(
    prev_digits_eval: np.ndarray, tables: Dict[int, np.ndarray]
) -> np.ndarray:
    n_eval = int(prev_digits_eval.shape[0])
    cvals = np.empty((n_eval, 32), dtype=np.int32)
    cvals[:, 0] = int(tables[0][0])

    prev_i = prev_digits_eval.astype(np.int32, copy=False)
    for mask in range(1, 32):
        positions = MASK_POSITIONS[mask]
        multipliers = MASK_MULTIPLIERS[mask]
        keys = (prev_i[:, positions] * multipliers[None, :]).sum(axis=1)
        cvals[:, mask] = tables[mask][keys]
    return cvals


def mobius_to_exact_match_counts(superset_counts: np.ndarray) -> np.ndarray:
    exact = superset_counts.astype(np.int64, copy=True)
    for bit in range(5):
        b = 1 << bit
        for mask in range(32):
            if (mask & b) == 0:
                exact[:, mask] -= exact[:, mask | b]
    return exact


def winner_feature_arrays(
    prev_digits_eval: np.ndarray, winner_digits_eval: np.ndarray
) -> Dict[str, np.ndarray]:
    winners = winner_digits_eval.astype(np.int16, copy=False)
    prev = prev_digits_eval.astype(np.int16, copy=False)

    sum_digits = np.sum(winners, axis=1, dtype=np.int16)
    even_count = np.sum((winners % 2) == 0, axis=1, dtype=np.int16)
    digits_ref = np.arange(10, dtype=np.int16)
    unique_count = np.sum(
        np.any(winners[:, :, None] == digits_ref[None, None, :], axis=1),
        axis=1,
        dtype=np.int16,
    )
    diffs = np.diff(winners, axis=1)
    consecutive_run_ge4 = ((diffs[:, 0] == 1) & (diffs[:, 1] == 1) & (diffs[:, 2] == 1)) | (
        (diffs[:, 1] == 1) & (diffs[:, 2] == 1) & (diffs[:, 3] == 1)
    ) | ((diffs[:, 0] == -1) & (diffs[:, 1] == -1) & (diffs[:, 2] == -1)) | (
        (diffs[:, 1] == -1) & (diffs[:, 2] == -1) & (diffs[:, 3] == -1)
    )
    mirror_repeats = np.sum(winners == prev, axis=1, dtype=np.int16)

    return {
        "sum_digits": sum_digits,
        "even_count": even_count,
        "unique_count": unique_count,
        "consecutive_run_ge4": consecutive_run_ge4.astype(bool, copy=False),
        "mirror_repeats": mirror_repeats,
    }


def compute_pareto_frontier(rows: Sequence[Dict]) -> List[Dict]:
    points = list(rows)
    frontier: List[Dict] = []
    for i, row_i in enumerate(points):
        fs_i = float(row_i["test_FS_pass"])
        u_i = float(row_i["test_avg_U"])
        dominated = False
        for j, row_j in enumerate(points):
            if i == j:
                continue
            fs_j = float(row_j["test_FS_pass"])
            u_j = float(row_j["test_avg_U"])
            if (fs_j >= fs_i and u_j <= u_i) and (fs_j > fs_i or u_j < u_i):
                dominated = True
                break
        if not dominated:
            frontier.append(row_i)
    frontier.sort(key=lambda r: (-float(r["test_FS_pass"]), float(r["test_avg_U"])))
    return frontier


def _safe_mean(arr: np.ndarray) -> float:
    if arr.size == 0:
        return 0.0
    return float(np.mean(arr))


def main() -> None:
    t0 = time.time()
    concursos, digits = load_tris_draws()
    n_draws = int(digits.shape[0])
    if n_draws < 3:
        raise ValueError("Se requieren al menos 3 sorteos para split temporal.")

    prev_eval = digits[:-1, :]
    winner_eval = digits[1:, :]
    draw_ids_eval = concursos[1:]
    n_eval = int(prev_eval.shape[0])
    split_at = int(np.floor(0.8 * n_eval))
    split_at = min(max(split_at, 1), n_eval - 1)

    train_slice = slice(0, split_at)
    test_slice = slice(split_at, n_eval)
    test_draw_ids = draw_ids_eval[split_at:]

    winner_feat = winner_feature_arrays(prev_eval, winner_eval)
    mirrors = [2, 3, 4]
    grid_static = build_grid()

    base_cfg = StructuralFilterConfig()
    all_tickets, _, _ = get_universe_and_static_mask(base_cfg)
    total_universe = int(all_tickets.shape[0])

    rows: List[Dict] = []
    n_static = len(grid_static)
    for idx, static_cfg in enumerate(grid_static, start=1):
        cfg_for_mask = StructuralFilterConfig(
            enabled=True,
            sum_min=int(static_cfg.sum_min),
            sum_max=int(static_cfg.sum_max),
            allowed_even_counts=tuple(int(x) for x in static_cfg.allowed_even_counts),
            min_unique_digits=int(static_cfg.min_unique_digits),
            max_consecutive_run=int(static_cfg.max_consecutive_run),
            max_positional_repeats_vs_prev=2,
            hard_filter=True,
            soft_penalties=None,
        )
        _, _, static_mask = get_universe_and_static_mask(cfg_for_mask)
        static_tickets = all_tickets[static_mask]

        tables = build_subset_tables(static_tickets)
        superset_counts = lookup_match_superset_counts(prev_eval, tables)
        exact_match_counts = mobius_to_exact_match_counts(superset_counts)

        u_series_by_mirror: Dict[int, np.ndarray] = {}
        for max_rep in mirrors:
            pass_masks = POPCOUNT <= int(max_rep)
            u_series_by_mirror[max_rep] = np.sum(
                exact_match_counts[:, pass_masks], axis=1, dtype=np.int64
            ).astype(np.int64, copy=False)

        sum_ok = (winner_feat["sum_digits"] >= static_cfg.sum_min) & (
            winner_feat["sum_digits"] <= static_cfg.sum_max
        )
        parity_ok = np.isin(
            winner_feat["even_count"],
            np.asarray(static_cfg.allowed_even_counts, dtype=np.int16),
        )
        uniques_ok = winner_feat["unique_count"] >= static_cfg.min_unique_digits
        if static_cfg.max_consecutive_run == 3:
            consecutive_ok = ~winner_feat["consecutive_run_ge4"]
        else:
            consecutive_ok = np.ones(n_eval, dtype=bool)
        static_pass = sum_ok & parity_ok & uniques_ok & consecutive_ok

        for max_rep in mirrors:
            u_series = u_series_by_mirror[max_rep]
            mirror_ok = winner_feat["mirror_repeats"] <= int(max_rep)
            fs_pass = static_pass & mirror_ok

            train_u = u_series[train_slice].astype(np.float64, copy=False)
            test_u = u_series[test_slice].astype(np.float64, copy=False)
            train_fs = fs_pass[train_slice]
            test_fs = fs_pass[test_slice]

            train_avg_u = _safe_mean(train_u)
            test_avg_u = _safe_mean(test_u)
            train_fs_rate = _safe_mean(train_fs.astype(np.float64, copy=False))
            test_fs_rate = _safe_mean(test_fs.astype(np.float64, copy=False))
            train_edge = float(train_fs_rate - (train_avg_u / float(total_universe)))
            test_edge = float(test_fs_rate - (test_avg_u / float(total_universe)))

            failed_test = ~test_fs
            sum_fail = (~sum_ok[test_slice]) & failed_test
            parity_fail = (~parity_ok[test_slice]) & failed_test
            uniques_fail = (~uniques_ok[test_slice]) & failed_test
            consecutive_fail = (~consecutive_ok[test_slice]) & failed_test
            mirror_fail = (winner_feat["mirror_repeats"][test_slice] > int(max_rep)) & failed_test

            cfg_payload = {
                "sum_min": int(static_cfg.sum_min),
                "sum_max": int(static_cfg.sum_max),
                "allowed_even_counts": [int(x) for x in static_cfg.allowed_even_counts],
                "min_unique_digits": int(static_cfg.min_unique_digits),
                "max_consecutive_run": int(static_cfg.max_consecutive_run),
                "max_positional_repeats_vs_prev": int(max_rep),
            }
            rows.append(
                {
                    "config_id": json.dumps(cfg_payload, sort_keys=True, separators=(",", ":")),
                    "parity_name": static_cfg.parity_name,
                    "train_avg_U": float(train_avg_u),
                    "train_min_U": int(np.min(train_u)) if train_u.size else 0,
                    "train_max_U": int(np.max(train_u)) if train_u.size else 0,
                    "train_FS_pass": float(train_fs_rate),
                    "train_edge": float(train_edge),
                    "test_avg_U": float(test_avg_u),
                    "test_min_U": int(np.min(test_u)) if test_u.size else 0,
                    "test_max_U": int(np.max(test_u)) if test_u.size else 0,
                    "test_FS_pass": float(test_fs_rate),
                    "test_edge": float(test_edge),
                    "test_winner_fail_sum": int(np.sum(sum_fail)),
                    "test_winner_fail_parity": int(np.sum(parity_fail)),
                    "test_winner_fail_uniques": int(np.sum(uniques_fail)),
                    "test_winner_fail_consecutive": int(np.sum(consecutive_fail)),
                    "test_winner_fail_mirror_prev": int(np.sum(mirror_fail)),
                    "test_draws": int(test_u.size),
                    "train_draws": int(train_u.size),
                }
            )

        if idx % 12 == 0 or idx == n_static:
            elapsed = time.time() - t0
            print(
                f"[tris-tuner] static_cfg {idx}/{n_static} completado "
                f"(rows={len(rows)}) elapsed={elapsed:.1f}s"
            )

    df = pd.DataFrame(rows)
    df = df.sort_values(
        by=["test_edge", "test_FS_pass", "test_avg_U"],
        ascending=[False, False, True],
        kind="mergesort",
    ).reset_index(drop=True)

    out_dir = os.path.join("artifacts", "tris")
    os.makedirs(out_dir, exist_ok=True)
    out_csv = os.path.join(out_dir, "filter_tuning_results.csv")
    df.to_csv(out_csv, index=False)

    top_n = min(12, len(df))
    print("\n=== Tris Filter Tuner (Top TEST by edge/fs/U) ===")
    for i in range(top_n):
        row = df.iloc[i]
        print(
            f"{i+1:02d}. test_edge={row['test_edge']:.6f} "
            f"test_FS={100.0*row['test_FS_pass']:.2f}% "
            f"test_avg_U={row['test_avg_U']:.1f} "
            f"cfg={row['config_id']}"
        )

    pareto_rows = compute_pareto_frontier(df.to_dict(orient="records"))
    print("\n=== Pareto Frontier (TEST: higher FS_pass, lower avg_U) ===")
    print(f"Pareto configs: {len(pareto_rows)} / {len(df)}")
    for i, row in enumerate(pareto_rows[:20], start=1):
        print(
            f"{i:02d}. FS={100.0*float(row['test_FS_pass']):.2f}% "
            f"avg_U={float(row['test_avg_U']):.1f} "
            f"edge={float(row['test_edge']):.6f} "
            f"cfg={row['config_id']}"
        )

    print("\n=== Summary ===")
    print(f"Draws loaded: {n_draws} | eval draws: {n_eval}")
    print(
        f"Temporal split: train={split_at} ({100.0*split_at/max(1, n_eval):.1f}%), "
        f"test={n_eval-split_at} ({100.0*(n_eval-split_at)/max(1, n_eval):.1f}%)"
    )
    print(f"Test draw id range: {test_draw_ids[0]} .. {test_draw_ids[-1]}")
    print(f"Grid static={n_static}, mirrors={len(mirrors)}, total rows={len(df)}")
    print(f"CSV: {out_csv}")


if __name__ == "__main__":
    main()
