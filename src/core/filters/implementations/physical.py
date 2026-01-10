# src/core/filters/implementations/physical.py
from typing import List
from src.core.filters.base import BaseFilter
from src.domain.dtos import CandidateCombination


class InertiaFilter(BaseFilter):
    """
    Obliga a que la nueva combinación contenga 'n' números del sorteo anterior.
    Basado en la estadística de repetición.
    """

    def __init__(self, previous_draw: List[int], min_matches: int = 1):
        super().__init__(name="InertiaFilter")
        self.previous_draw = previous_draw
        self.min_matches = min_matches

    def validate(self, candidate: CandidateCombination) -> bool:
        if not self.previous_draw:
            return True  # Si no hay historia, pasamos (ej. primer sorteo)

        # Calcula intersección
        matches = sum(1 for n in candidate.numbers if n in self.previous_draw)
        return matches >= self.min_matches
