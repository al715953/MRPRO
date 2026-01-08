from dataclasses import dataclass, field
from typing import List, Any, Dict

@dataclass
class DrawHistoryDTO:
    dates: List[Any]
    winning_numbers: List[List[int]]
    concursos: List[int]

@dataclass
class PredictionConfigDTO:
    total_balls: int
    ticket_size: int
    num_tickets: int
    backtest_size: int = 10 

@dataclass
class PredictionResultDTO:
    strategy_name: str
    tickets: List[List[int]]

# --- NUEVA CLASE QUE FALTABA ---
@dataclass
class BacktestResultDTO:
    strategy_name: str
    total_draws_tested: int
    investment: float
    earnings: float
    net_balance: float
    hit_distribution: Dict[int, int]