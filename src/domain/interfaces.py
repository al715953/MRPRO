from abc import ABC, abstractmethod
from .dtos import DrawHistoryDTO, PredictionConfigDTO, PredictionResultDTO

class ILotteryStrategy(ABC):
    """
    Contrato que deben cumplir todas las estrategias (Monte Carlo, Genética, etc).
    """
    @abstractmethod
    def predict(self, history: DrawHistoryDTO, config: PredictionConfigDTO) -> PredictionResultDTO:
        pass