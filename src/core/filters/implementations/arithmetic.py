# src/core/filters/implementations/arithmetic.py
from src.core.filters.base import BaseFilter
from src.domain.dtos import CandidateCombination


class ACValueFilter(BaseFilter):
    """
    Filtra combinaciones demasiado simples o demasiado complejas.
    Para loterías 6/39, un valor >= 7 suele indicar suficiente 'rugosidad'.
    """

    def __init__(self, min_ac: int):
        super().__init__(name="ACValueFilter")
        self.min_ac = min_ac

    def validate(self, candidate: CandidateCombination) -> bool:
        # Costoso computacionalmente, poner al final del pipeline
        return candidate.ac_value >= self.min_ac
