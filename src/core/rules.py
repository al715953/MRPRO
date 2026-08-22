from typing import List, Tuple


class MelateRetroRules:
    """Reglas de Negocio actualizadas para Melate Retro"""

    PRIZE_CATEGORY_ORDER = (
        "6",
        "5+AD",
        "5",
        "4",
        "3",
        "2+AD",
        "1+AD",
        "SIN_PREMIO",
    )

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

    def prize_category(self, hits_naturales: int, has_adicional: bool) -> str:
        """Return the paid Melate category without hiding the additional ball.

        Categories whose prize does not change when the ticket also contains the
        additional ball (3 and 4 natural hits) remain in their natural category.
        """
        prize = self.calculate_prize(hits_naturales, has_adicional)
        if prize <= 0:
            return "SIN_PREMIO"
        if has_adicional and (hits_naturales, True) in self.pay_table:
            return f"{int(hits_naturales)}+AD"
        return str(int(hits_naturales))

    def category_from_recorded_result(
        self, hits_naturales: int, prize: float
    ) -> str:
        """Reconstruct a category from legacy telemetry without an AD flag."""
        hits_naturales = int(hits_naturales)
        prize = float(prize)
        if prize <= 0:
            return "SIN_PREMIO"
        additional_prize = self.pay_table.get((hits_naturales, True))
        natural_prize = self.pay_table.get((hits_naturales, False))
        if (
            additional_prize is not None
            and additional_prize != natural_prize
            and prize == float(additional_prize)
        ):
            return f"{hits_naturales}+AD"
        return str(hits_naturales)


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
