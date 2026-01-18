import numpy as np
from dataclasses import dataclass, field
from typing import List, Any, Dict, Tuple, Optional


# --- Infraestructura de Datos Base ---


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
    # DECLARACIÓN PARA EL IDE:
    raw_universe_ptr: Optional[np.ndarray] = None
    filter_overrides: Dict[str, Any] = field(default_factory=dict)


@dataclass
class PredictionResultDTO:
    strategy_name: str
    tickets: List[List[int]]
    # Contenedor para evidencia forense (Rank, AI Score, Geo Score, etc.)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class BacktestResultDTO:
    strategy_name: str
    total_draws_tested: int
    investment: float
    earnings: float
    net_balance: float
    hit_distribution: Dict[int, int]


# --- Infraestructura de Filtrado de Alto Rendimiento ---


class CandidateCombination:
    """
    Representa un ticket candidato con evaluación perezosa (Lazy Evaluation).
    Optimizado para manejar millones de instancias con __slots__.
    """

    __slots__ = (
        "numbers",  # La combinación (Tupla inmutable)
        "_total_sum",  # Cache
        "_even_count",  # Cache
        "_ac_value",  # Cache
        "_primes_count",  # Cache
        "_distances",  # Cache para deltas
    )

    def __init__(self, numbers: Tuple[int, ...]):
        self.numbers = numbers
        self._total_sum: Optional[int] = None
        self._even_count: Optional[int] = None
        self._ac_value: Optional[int] = None
        self._primes_count: Optional[int] = None
        self._distances: Optional[List[int]] = None

    @property
    def total_sum(self) -> int:
        if self._total_sum is None:
            self._total_sum = sum(self.numbers)
        return self._total_sum

    @property
    def even_count(self) -> int:
        if self._even_count is None:
            self._even_count = sum(1 for n in self.numbers if n % 2 == 0)
        return self._even_count

    @property
    def primes_count(self) -> int:
        if self._primes_count is None:
            primes = {2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53}
            self._primes_count = sum(1 for n in self.numbers if n in primes)
        return self._primes_count

    @property
    def ac_value(self) -> int:
        if self._ac_value is None:
            diffs = set()
            n_len = len(self.numbers)
            for i in range(n_len):
                for j in range(i + 1, n_len):
                    d = self.numbers[j] - self.numbers[i]
                    diffs.add(d)
            self._ac_value = len(diffs) - (n_len - 1)
        return self._ac_value

    def has_numbers_from(self, other_numbers: List[int]) -> bool:
        return not set(self.numbers).isdisjoint(other_numbers)

    def __repr__(self):
        return f"Ticket({list(self.numbers)})"
