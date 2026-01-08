from src.domain.interfaces import ILotteryStrategy
from src.strategies.monte_carlo import MonteCarloStrategy

class StrategyFactory:
    @staticmethod
    def get_strategy(strategy_id: str) -> ILotteryStrategy:
        if strategy_id == "MONTE_CARLO":
            return MonteCarloStrategy()
        # elif strategy_id == "GENETIC": return GeneticStrategy()
        else:
            # Default fallback
            return MonteCarloStrategy()