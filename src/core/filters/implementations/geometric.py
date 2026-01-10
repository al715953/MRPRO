# src/core/filters/implementations/geometric.py
from src.core.filters.base import BaseFilter
from src.domain.dtos import CandidateCombination


class SumRangeFilter(BaseFilter):
    """Verifica si la suma total cae dentro de la Campana de Gauss deseada."""

    def __init__(self, min_val: int, max_val: int):
        super().__init__(name="SumRangeFilter")
        self.min_val = min_val
        self.max_val = max_val

    def validate(self, candidate: CandidateCombination) -> bool:
        s = candidate.total_sum  # Acceso Lazy
        return self.min_val <= s <= self.max_val
