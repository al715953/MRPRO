from abc import ABC, abstractmethod
from src.domain.dtos import CandidateCombination


class BaseFilter(ABC):
    """
    Clase abstracta para todos los filtros del Pipeline.
    """

    def __init__(self, name: str, enabled: bool = True):
        self.name = name
        self.enabled = enabled

    @abstractmethod
    def validate(self, candidate: CandidateCombination) -> bool:
        """
        Lógica del filtro.
        Return True: El ticket SOBREVIVE.
        Return False: El ticket es DESCARTADO.
        """
        pass
