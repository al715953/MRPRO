from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from src.data_access.config import BEST_SETTINGS_TRIS
from src.domain.dtos import DrawHistoryDTO, PredictionConfigDTO, PredictionResultDTO
from src.strategies.tris.structural_filters import (
    StructuralFilterConfig,
    StructuralFilterEngine,
)


class RandomWithinStructuralFiltersStrategy:
    def __init__(self):
        self.strategy_name = "Tris Random Within Structural Filters"
        self.model_version = "random_within_filters"

    @staticmethod
    def _get_override(cfg: Dict[str, Any], key: str, default):
        return cfg.get(key, default) if isinstance(cfg, dict) else default

    def _get_structural_override(self, overrides: Dict[str, Any], key: str, default):
        fallback = BEST_SETTINGS_TRIS.get(key, default)
        return self._get_override(overrides, key, fallback)

    @staticmethod
    def _seed_from_overrides(overrides: Dict[str, Any], history_len: int) -> Optional[int]:
        if not isinstance(overrides, dict):
            return None
        raw = overrides.get("seed")
        if raw in (None, ""):
            return None
        try:
            return int(raw) + int(history_len)
        except (TypeError, ValueError):
            return None

    @staticmethod
    def _estimate_multiplier_rate(history: DrawHistoryDTO) -> float:
        draws = history.winning_numbers if history and history.winning_numbers else []
        if not draws:
            return 0.5
        observed = [1 if (len(row) > 5 and bool(row[5])) else 0 for row in draws]
        if not observed:
            return 0.5
        return float(sum(observed) / len(observed))

    def _build_structural_config(
        self, overrides: Dict[str, Any], sum_min: int, sum_max: int
    ) -> StructuralFilterConfig:
        allowed_even = self._get_structural_override(
            overrides, "structural_allowed_even_counts", [2, 3]
        )
        if allowed_even is None:
            allowed_even = [2, 3]
        elif not isinstance(allowed_even, (list, tuple, set)):
            allowed_even = [allowed_even]

        return StructuralFilterConfig(
            enabled=bool(
                self._get_structural_override(overrides, "structural_enabled", True)
            ),
            sum_min=int(sum_min),
            sum_max=int(sum_max),
            allowed_even_counts=tuple(int(v) for v in allowed_even),
            min_unique_digits=int(
                self._get_structural_override(
                    overrides, "structural_min_unique_digits", 3
                )
            ),
            max_consecutive_run=int(
                self._get_structural_override(
                    overrides, "structural_max_consecutive_run", 3
                )
            ),
            max_positional_repeats_vs_prev=int(
                self._get_structural_override(
                    overrides, "structural_max_positional_repeats_vs_prev", 2
                )
            ),
            hard_filter=bool(
                self._get_structural_override(overrides, "structural_hard_filter", True)
            ),
            soft_penalties=self._get_structural_override(
                overrides, "structural_soft_penalties", None
            ),
        )

    def predict(
        self, history: DrawHistoryDTO, config: PredictionConfigDTO
    ) -> PredictionResultDTO:
        overrides = config.filter_overrides or {}
        target_n = max(0, int(config.num_tickets))
        if target_n == 0:
            return PredictionResultDTO(
                strategy_name=self.strategy_name,
                tickets=[],
                metadata={"model_version": self.model_version},
            )

        history_len = len(history.concursos) if history and history.concursos else 0
        rng = np.random.default_rng(self._seed_from_overrides(overrides, history_len))
        prev_digits = (
            [int(d) for d in history.winning_numbers[-1][:5]]
            if history and history.winning_numbers
            else None
        )

        max_unique = 100000
        max_iter = max(
            1000,
            int(self._get_override(overrides, "random_struct_max_iter", 50000)),
        )
        batch_size = max(
            32,
            int(self._get_override(overrides, "random_struct_batch_size", 512)),
        )
        max_relax_steps = max(
            0,
            int(self._get_override(overrides, "random_struct_max_relax_steps", 8)),
        )

        base_sum_min = int(
            self._get_structural_override(overrides, "structural_sum_min", 15)
        )
        base_sum_max = int(
            self._get_structural_override(overrides, "structural_sum_max", 30)
        )

        seen = set()
        tickets: List[List[int]] = []
        reject_reasons = {
            "sum": 0,
            "parity": 0,
            "uniques": 0,
            "consecutive": 0,
            "mirror_prev": 0,
        }
        relax_history = []
        attempts_total = 0
        checked_unique_total = 0
        accepted_from_filters = 0
        fallback_unfiltered = False

        sum_min = int(base_sum_min)
        sum_max = int(base_sum_max)
        for step in range(max_relax_steps + 1):
            cfg = self._build_structural_config(overrides, sum_min=sum_min, sum_max=sum_max)
            engine = StructuralFilterEngine(cfg)
            attempts_round = 0
            accepted_before = len(tickets)

            while (
                len(tickets) < target_n
                and attempts_round < max_iter
                and len(seen) < max_unique
            ):
                sample_n = min(batch_size, max_iter - attempts_round)
                raw_block = rng.integers(0, 10, size=(sample_n, 5), endpoint=False)
                unique_block = []
                for row in raw_block.tolist():
                    key = tuple(int(d) for d in row)
                    if key in seen:
                        continue
                    seen.add(key)
                    unique_block.append([int(d) for d in key])

                if not unique_block:
                    continue

                accepted, diag = engine.apply(unique_block, prev_digits)
                checked = int(len(unique_block))
                attempts_round += checked
                attempts_total += checked
                checked_unique_total += checked
                for reason, count in (diag.get("reject_reasons") or {}).items():
                    if reason in reject_reasons:
                        reject_reasons[reason] += int(count)

                for tkt in accepted:
                    if len(tickets) >= target_n:
                        break
                    tickets.append([int(d) for d in tkt])
                    accepted_from_filters += 1

            relax_history.append(
                {
                    "step": int(step),
                    "sum_min": int(sum_min),
                    "sum_max": int(sum_max),
                    "attempts": int(attempts_round),
                    "accepted_added": int(len(tickets) - accepted_before),
                }
            )

            if len(tickets) >= target_n or len(seen) >= max_unique:
                break
            sum_min -= 2
            sum_max += 2

        if len(tickets) < target_n:
            fallback_unfiltered = True
            while len(tickets) < target_n and len(seen) < max_unique:
                row = [int(x) for x in rng.integers(0, 10, size=5, endpoint=False).tolist()]
                key = tuple(row)
                if key in seen:
                    continue
                seen.add(key)
                tickets.append(row)
            while len(tickets) < target_n:
                tickets.append(
                    [int(x) for x in rng.integers(0, 10, size=5, endpoint=False).tolist()]
                )

        metadata = {
            "model_version": self.model_version,
            "p_multiplier": self._estimate_multiplier_rate(history),
            "pos_probs": np.full((5, 10), 0.1, dtype=np.float64).tolist(),
            "structural_filters": {
                "enabled": bool(
                    self._get_structural_override(overrides, "structural_enabled", True)
                ),
                "reject_reasons": reject_reasons,
                "accepted_from_filters": int(accepted_from_filters),
                "fallback_unfiltered": bool(fallback_unfiltered),
                "relax_history": relax_history,
                "initial_sum_min": int(base_sum_min),
                "initial_sum_max": int(base_sum_max),
                "final_sum_min": int(relax_history[-1]["sum_min"]) if relax_history else int(base_sum_min),
                "final_sum_max": int(relax_history[-1]["sum_max"]) if relax_history else int(base_sum_max),
                "attempts_total": int(attempts_total),
                "checked_unique_total": int(checked_unique_total),
            },
        }

        return PredictionResultDTO(
            strategy_name=self.strategy_name,
            tickets=tickets[:target_n],
            metadata=metadata,
        )
