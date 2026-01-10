# src/core/filters/implementations/probabilistic.py
from src.core.filters.base import BaseFilter
from src.domain.dtos import CandidateCombination


class ParityFilter(BaseFilter):
    """Filtra por cantidad de números pares."""

    def __init__(self, min_even: int, max_even: int):
        super().__init__(name="ParityFilter")
        self.min_even = min_even
        self.max_even = max_even

    def validate(self, candidate: CandidateCombination) -> bool:
        evens = candidate.even_count
        return self.min_even <= evens <= self.max_even


class PrimeFilter(BaseFilter):
    """Filtra por cantidad de números primos."""

    def __init__(self, min_primes: int, max_primes: int):
        super().__init__(name="PrimeFilter")
        self.min_primes = min_primes
        self.max_primes = max_primes

    def validate(self, candidate: CandidateCombination) -> bool:
        p = candidate.primes_count
        return self.min_primes <= p <= self.max_primes
