from __future__ import annotations

from typing import Any, Dict, List, Optional

import numpy as np

from src.domain.dtos import DrawHistoryDTO, PredictionConfigDTO, PredictionResultDTO


class TrisUniformBaselineStrategy:
    def __init__(self):
        self.strategy_name = "Tris Uniform Baseline"
        self.model_version = "tris_uniform_baseline"

    @staticmethod
    def _seed_from_overrides(overrides: Dict[str, Any]) -> Optional[int]:
        if not isinstance(overrides, dict):
            return None
        seed = overrides.get("seed")
        if seed in (None, ""):
            return None
        try:
            return int(seed)
        except (TypeError, ValueError):
            return None

    @staticmethod
    def _ticket_from_index(idx: int) -> List[int]:
        n = int(idx)
        return [
            (n // 10000) % 10,
            (n // 1000) % 10,
            (n // 100) % 10,
            (n // 10) % 10,
            n % 10,
        ]

    def _generate_tickets(self, num_tickets: int, rng: np.random.Generator) -> List[List[int]]:
        if num_tickets <= 0:
            return []

        max_unique = 100000  # 10^5 secuencias posibles.
        if num_tickets <= max_unique:
            indexes = rng.choice(max_unique, size=num_tickets, replace=False)
            return [self._ticket_from_index(idx) for idx in indexes.tolist()]

        tickets = [self._ticket_from_index(i) for i in range(max_unique)]
        extra = num_tickets - max_unique
        extra_block = rng.integers(0, 10, size=(extra, 5), endpoint=False)
        tickets.extend(extra_block.tolist())
        return tickets

    @staticmethod
    def _estimate_multiplier_rate(history: DrawHistoryDTO) -> float:
        draws = history.winning_numbers if history and history.winning_numbers else []
        if not draws:
            return 0.5

        observed = [1 if (len(row) > 5 and bool(row[5])) else 0 for row in draws]
        if not observed:
            return 0.5
        return float(sum(observed) / len(observed))

    def predict(
        self, history: DrawHistoryDTO, config: PredictionConfigDTO
    ) -> PredictionResultDTO:
        overrides = config.filter_overrides or {}
        seed = self._seed_from_overrides(overrides)
        rng = np.random.default_rng(seed)

        tickets = self._generate_tickets(int(config.num_tickets), rng)
        metadata = {
            "pos_probs": np.full((5, 10), 0.1, dtype=np.float64).tolist(),
            "p_multiplier": self._estimate_multiplier_rate(history),
            "model_version": self.model_version,
        }

        return PredictionResultDTO(
            strategy_name=self.strategy_name,
            tickets=tickets,
            metadata=metadata,
        )
