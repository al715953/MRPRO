from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional


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
        return True
    if len(ticket) < run_len:
        return False

    values = [int(d) for d in ticket]
    for i in range(0, len(values) - run_len + 1):
        segment = values[i : i + run_len]
        diffs = [segment[j + 1] - segment[j] for j in range(run_len - 1)]
        if all(step == 1 for step in diffs) or all(step == -1 for step in diffs):
            return True
    return False


def positional_repeats(ticket: List[int], prev: List[int]) -> int:
    return int(sum(1 for a, b in zip(ticket, prev) if int(a) == int(b)))


class StructuralFilterEngine:
    def __init__(self, config: StructuralFilterConfig):
        self.config = config

    def apply(
        self,
        candidates: List[List[int]],
        prev_digits: List[int] | None,
        limit: int | None = None,
    ):
        items = [[int(d) for d in ticket] for ticket in (candidates or [])]
        prev = (
            [int(d) for d in prev_digits[: len(items[0])]]
            if (prev_digits is not None and len(items) > 0)
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

        run_len = max(2, int(self.config.max_consecutive_run) + 1)
        allowed_even = tuple(int(v) for v in self.config.allowed_even_counts)

        for ticket in items:
            violations = []

            s = digit_sum(ticket)
            if s < int(self.config.sum_min) or s > int(self.config.sum_max):
                violations.append("sum")

            evens = even_count(ticket)
            if evens not in allowed_even:
                violations.append("parity")

            uniq = unique_count(ticket)
            if uniq < int(self.config.min_unique_digits):
                violations.append("uniques")

            if has_consecutive_run(ticket, run_len=run_len):
                violations.append("consecutive")

            if prev is not None:
                repeats = positional_repeats(ticket, prev)
                if repeats > int(self.config.max_positional_repeats_vs_prev):
                    violations.append("mirror_prev")

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
