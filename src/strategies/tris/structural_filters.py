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

    def _violations(self, ticket: List[int], prev_digits, cfg) -> List[str]:
        vals = [int(d) for d in ticket]
        violations = []

        s = digit_sum(vals)
        if s < int(self._cfg_value(cfg, "sum_min", 15)) or s > int(
            self._cfg_value(cfg, "sum_max", 30)
        ):
            violations.append("sum")

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

        if prev_digits is not None:
            repeats = positional_repeats(vals, prev_digits.tolist())
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

        prev = StructuralFilterEngine._coerce_prev(prev_digits, tickets.shape[1])
        if prev is None:
            return base_mask

        max_repeats = int(
            StructuralFilterEngine._cfg_value(cfg, "max_positional_repeats_vs_prev", 2)
        )
        mirror_count = np.sum(tickets.astype(np.int16) == prev[None, :], axis=1)
        mirror_mask = mirror_count <= max_repeats
        return base_mask & mirror_mask

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
                    reject_reasons[reason] += 1

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
