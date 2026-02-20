from typing import List, Tuple


class MelateRetroRules:
    """Reglas de Negocio actualizadas para Melate Retro"""

    def __init__(self):
        self.ticket_cost = 10.0
        self.pay_table = {
            (6, False): 4650000.0,
            (5, True): 30000.0,
            (5, False): 800.0,
            (4, False): 150.0,
            (3, False): 20.0,
            (2, True): 15.0,
            (1, True): 10.0,
        }
        self.max_hits = 6

    def validate_ticket(
        self, ticket: List[int], winning_draw: List[int]
    ) -> Tuple[int, bool]:
        naturales_reales = set(winning_draw[:6])
        adicional_real = winning_draw[6]
        ticket_set = set(ticket)

        hits_naturales = len(ticket_set.intersection(naturales_reales))
        has_adicional = adicional_real in ticket_set

        return hits_naturales, has_adicional

    def calculate_prize(self, hits_naturales: int, has_adicional: bool) -> float:
        prize = self.pay_table.get((hits_naturales, has_adicional))
        if prize is not None:
            return prize

        if has_adicional:
            prize = self.pay_table.get((hits_naturales, False))
            if prize is not None:
                return prize

        return 0.0


class TrisMultiplicadorRules:
    """Reglas base para backtest de Tris con Multiplicador."""

    def __init__(self, ticket_cost: float = 10.0, base_prize: float = 600.0):
        self.ticket_cost = ticket_cost
        self.base_prize = base_prize
        self.max_hits = 5

    def validate_ticket(
        self, ticket: List[int], winning_draw: List[int]
    ) -> Tuple[int, bool]:
        winning_digits = winning_draw[:5]
        hits_pos = sum(
            1 for i in range(5) if int(ticket[i]) == int(winning_digits[i])
        )
        has_multiplier = bool(winning_draw[5]) if len(winning_draw) > 5 else False
        return hits_pos, has_multiplier

    def calculate_prize(self, hits_naturales: int, has_adicional: bool) -> float:
        if hits_naturales < 5:
            return 0.0
        return self.base_prize * (2 if has_adicional else 1)
