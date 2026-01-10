from typing import List
from src.domain.dtos import CandidateCombination
from .base import BaseFilter


class FilterPipeline:
    """
    Gestor de la cadena de responsabilidad.
    Ejecuta filtros secuencialmente y aplica Short-Circuit (Fail-Fast).
    """

    def __init__(self):
        self.filters: List[BaseFilter] = []

    def add_filter(self, filter_instance: BaseFilter):
        """Agrega un filtro al final de la cadena."""
        self.filters.append(filter_instance)
        return self  # Permite encadenamiento fluido

    def validate(self, candidate: CandidateCombination) -> bool:
        """
        Pasa el candidato por todos los filtros habilitados.
        Si uno falla, retorna False inmediatamente.
        """
        for f in self.filters:
            if f.enabled:
                if not f.validate(candidate):
                    return False
        return True
