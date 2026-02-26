from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional

import numpy as np


@dataclass
class StructuralFilterConfig:
    enabled: bool = True
    sum_min: int = 15
    sum_max: int = 30
    allowed_even_counts: tuple[int, ...] = (2, 3)
    min_unique_digits: int = 3
    max_consecutive_run: int = 3
    max_positional_repeats_vs_prev: int = 2
    hard_filter: bool = True
    soft_penalties: Optional[Dict[str, float]] = None
    enable_global_sum_filter: bool = True
    enable_global_parity_filter: bool = True
    positional_limits: Optional[list[dict]] = None
    immediate_repeat_mode: str = "global_count"  # {"global_count", "per_position"}
    immediate_repeat_disallow_positions: tuple[bool, ...] = (
        False,
        False,
        False,
        False,
        False,
    )
    camera_entropy_rules: Optional[list[dict]] = None


def digit_sum(ticket: List[int]) -> int:
    return int(sum(int(d) for d in ticket))


def even_count(ticket: List[int]) -> int:
    return int(sum(1 for d in ticket if int(d) % 2 == 0))


def unique_count(ticket: List[int]) -> int:
    return int(len(set(int(d) for d in ticket)))


def has_consecutive_run(ticket: List[int], run_len: int = 4) -> bool:
    if run_len <= 1:
        return False
    if len(ticket) < run_len:
        return False

    values = [int(d) for d in ticket]
    for i in range(0, len(values) - run_len + 1):
        window = values[i : i + run_len]
        diff = [window[j + 1] - window[j] for j in range(run_len - 1)]
        if all(d == 1 for d in diff) or all(d == -1 for d in diff):
            return True
    return False


def positional_repeats(ticket: List[int], prev: List[int]) -> int:
    return int(sum(1 for a, b in zip(ticket, prev) if int(a) == int(b)))


class StructuralFilterEngine:
    def __init__(self, config: StructuralFilterConfig):
        self.config = config

    @staticmethod
    def _coerce_prev(prev_digits, n_pos: int):
        if prev_digits is None:
            return None
        prev_arr = np.asarray(prev_digits, dtype=np.int16).reshape(-1)
        if prev_arr.size < n_pos:
            return None
        return prev_arr[:n_pos]

    @staticmethod
    def _cfg_value(cfg, key: str, default):
        if cfg is None:
            return default
        if isinstance(cfg, dict):
            return cfg.get(key, default)
        return getattr(cfg, key, default)

    @staticmethod
    def _normalize_positional_limits(cfg, n_pos: int) -> list[dict]:
        limits = StructuralFilterEngine._cfg_value(cfg, "positional_limits", None)
        out: list[dict] = [{} for _ in range(n_pos)]
        if not isinstance(limits, (list, tuple)):
            return out
        for pos in range(min(n_pos, len(limits))):
            rule = limits[pos]
            if isinstance(rule, dict):
                out[pos] = rule
        return out

    @staticmethod
    def _normalize_disallow_positions(cfg, n_pos: int) -> np.ndarray:
        raw = StructuralFilterEngine._cfg_value(
            cfg,
            "immediate_repeat_disallow_positions",
            (False, False, False, False, False),
        )
        if not isinstance(raw, (list, tuple, np.ndarray)):
            return np.zeros(n_pos, dtype=bool)
        arr = np.asarray(raw, dtype=bool).reshape(-1)
        if arr.size < n_pos:
            arr = np.pad(arr, (0, n_pos - arr.size), mode="constant", constant_values=False)
        return arr[:n_pos]

    def _violations(self, ticket: List[int], prev_digits, cfg) -> List[str]:
        vals = [int(d) for d in ticket]
        violations = []

        if bool(self._cfg_value(cfg, "enable_global_sum_filter", True)):
            s = digit_sum(vals)
            if s < int(self._cfg_value(cfg, "sum_min", 15)) or s > int(
                self._cfg_value(cfg, "sum_max", 30)
            ):
                violations.append("sum")

        if bool(self._cfg_value(cfg, "enable_global_parity_filter", True)):
            allowed_even = self._cfg_value(cfg, "allowed_even_counts", (2, 3))
            if not isinstance(allowed_even, (list, tuple, set)):
                allowed_even = (allowed_even,)
            allowed_even = tuple(int(v) for v in allowed_even)
            if even_count(vals) not in allowed_even:
                violations.append("parity")

        if unique_count(vals) < int(self._cfg_value(cfg, "min_unique_digits", 3)):
            violations.append("uniques")

        run_len = max(2, int(self._cfg_value(cfg, "max_consecutive_run", 3)) + 1)
        if has_consecutive_run(vals, run_len=run_len):
            violations.append("consecutive")

        positional_limits = self._normalize_positional_limits(cfg, len(vals))
        for pos, rule in enumerate(positional_limits):
            forbidden = rule.get("forbidden_digits", [])
            if isinstance(forbidden, (list, tuple, set, np.ndarray)):
                forbidden_vals = {int(v) for v in forbidden}
                if vals[pos] in forbidden_vals:
                    violations.append(f"pos{pos + 1}_forbidden")

            allowed_parity = rule.get("allowed_parity")
            if isinstance(allowed_parity, str):
                parity_mode = allowed_parity.strip().lower()
                if parity_mode == "even" and (vals[pos] % 2 != 0):
                    violations.append(f"pos{pos + 1}_parity")
                elif parity_mode == "odd" and (vals[pos] % 2 == 0):
                    violations.append(f"pos{pos + 1}_parity")

        if prev_digits is not None:
            prev_vals = np.asarray(prev_digits, dtype=np.int16).reshape(-1)
            repeat_mode = str(self._cfg_value(cfg, "immediate_repeat_mode", "global_count")).lower()
            if repeat_mode == "per_position":
                disallow = self._normalize_disallow_positions(cfg, len(vals))
                for pos in range(len(vals)):
                    if bool(disallow[pos]) and vals[pos] == int(prev_vals[pos]):
                        violations.append(f"pos{pos + 1}_repeat_prev")
            else:
                repeats = positional_repeats(vals, [int(v) for v in prev_vals[: len(vals)]])
                if repeats > int(self._cfg_value(cfg, "max_positional_repeats_vs_prev", 2)):
                    violations.append("mirror_prev")

        return violations

    def passes(self, ticket: List[int], prev_digits: List[int] | None) -> bool:
        prev = self._coerce_prev(prev_digits, len(ticket))
        return len(self._violations(ticket, prev, self.config)) == 0

    @staticmethod
    def mask_all(
        all_tickets,
        prev_digits,
        static_mask,
        cfg,
    ):
        tickets = np.asarray(all_tickets, dtype=np.uint8)
        base_mask = np.asarray(static_mask, dtype=bool)
        if tickets.ndim != 2 or tickets.shape[1] != 5:
            raise ValueError("all_tickets debe tener shape (N, 5).")
        if base_mask.shape[0] != tickets.shape[0]:
            raise ValueError("static_mask debe tener longitud N.")

        final_mask = base_mask.copy()
        n_pos = tickets.shape[1]

        positional_limits = StructuralFilterEngine._normalize_positional_limits(cfg, n_pos)
        for pos, rule in enumerate(positional_limits):
            forbidden = rule.get("forbidden_digits", [])
            if isinstance(forbidden, (list, tuple, set, np.ndarray)):
                forbidden_vals = np.array([int(v) for v in forbidden], dtype=np.int16)
                if forbidden_vals.size > 0:
                    final_mask &= ~np.isin(tickets[:, pos].astype(np.int16), forbidden_vals)

            allowed_parity = rule.get("allowed_parity")
            if isinstance(allowed_parity, str):
                parity_mode = allowed_parity.strip().lower()
                if parity_mode == "even":
                    final_mask &= (tickets[:, pos] % 2) == 0
                elif parity_mode == "odd":
                    final_mask &= (tickets[:, pos] % 2) == 1

        prev = StructuralFilterEngine._coerce_prev(prev_digits, tickets.shape[1])
        repeat_mode = str(StructuralFilterEngine._cfg_value(cfg, "immediate_repeat_mode", "global_count")).lower()

        if prev is None:
            return final_mask

        if repeat_mode == "per_position":
            disallow = StructuralFilterEngine._normalize_disallow_positions(cfg, n_pos)
            for pos in range(n_pos):
                if bool(disallow[pos]):
                    final_mask &= tickets[:, pos].astype(np.int16) != int(prev[pos])
            return final_mask

        max_repeats = int(
            StructuralFilterEngine._cfg_value(cfg, "max_positional_repeats_vs_prev", 2)
        )
        mirror_count = np.sum(tickets.astype(np.int16) == prev[None, :], axis=1)
        mirror_mask = mirror_count <= max_repeats
        return final_mask & mirror_mask

    def apply(
        self,
        candidates: List[List[int]],
        prev_digits: List[int] | None,
        limit: int | None = None,
    ):
        items = [[int(d) for d in ticket] for ticket in (candidates or [])]
        prev = (
            self._coerce_prev(prev_digits, len(items[0]))
            if len(items) > 0
            else None
        )

        reject_reasons = {
            "sum": 0,
            "parity": 0,
            "uniques": 0,
            "consecutive": 0,
            "mirror_prev": 0,
        }
        accepted: List[List[int]] = []
        soft_penalties = self.config.soft_penalties or {}
        soft_total = 0.0
        soft_positive = 0

        for ticket in items:
            violations = self._violations(ticket, prev, self.config)

            if violations:
                for reason in set(violations):
                    reject_reasons[reason] = int(reject_reasons.get(reason, 0)) + 1

            if self.config.hard_filter and violations:
                continue

            if not self.config.hard_filter and violations:
                penalty = 0.0
                for reason in set(violations):
                    if reason == "sum":
                        penalty += float(soft_penalties.get("sum_out", 1.0))
                    elif reason == "parity":
                        penalty += float(soft_penalties.get("parity_out", 1.0))
                    elif reason == "uniques":
                        penalty += float(soft_penalties.get("uniques_low", 1.0))
                    elif reason == "consecutive":
                        penalty += float(soft_penalties.get("run_high", 1.0))
                    elif reason == "mirror_prev":
                        penalty += float(soft_penalties.get("mirror_high", 1.0))
                    else:
                        penalty += float(soft_penalties.get(reason, 0.0))
                soft_total += penalty
                if penalty > 0:
                    soft_positive += 1

            accepted.append(ticket)

        if limit is not None and limit >= 0:
            accepted_out = accepted[: int(limit)]
        else:
            accepted_out = accepted

        total_in = int(len(items))
        accepted_n = int(len(accepted_out))
        total_out = int(total_in - accepted_n)
        acceptance_rate = float((accepted_n / total_in) if total_in else 0.0)

        diagnostics = {
            "total_in": total_in,
            "total_out": total_out,
            "accepted": accepted_n,
            "reject_reasons": reject_reasons,
            "acceptance_rate": acceptance_rate,
        }
        if not self.config.hard_filter:
            diagnostics["soft_penalty_total"] = float(soft_total)
            diagnostics["soft_penalty_avg"] = float(
                (soft_total / accepted_n) if accepted_n else 0.0
            )
            diagnostics["soft_penalty_positive"] = int(soft_positive)

        return accepted_out, diagnostics
