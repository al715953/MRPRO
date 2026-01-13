from typing import List, Tuple


class MelateRetroRules:
    """Reglas de Negocio actualizadas para Melate Retro"""

    def __init__(self):
        self.ticket_cost = 10.0
        # Tabla de premios EXACTOS
        self.pay_table = {
            (6, False): 4650000.0,  # 1er Lugar
            (5, True): 30000.0,  # 2do Lugar (Naturales + Adicional)
            (5, False): 800.0,  # 3er Lugar
            (4, False): 150.0,  # 4to Lugar
            (3, False): 20.0,  # 5to Lugar
            (2, True): 15.0,  # 6to Lugar (2 + Adicional)
            (1, True): 10.0,  # 7mo Lugar (1 + Adicional)
        }

    def validate_ticket(
        self, ticket: List[int], winning_draw: List[int]
    ) -> Tuple[int, bool]:
        """
        Calcula aciertos.
        winning_draw debe tener 7 números: [6 Naturales..., 1 Adicional]
        """
        naturales_reales = set(winning_draw[:6])
        adicional_real = winning_draw[6]
        ticket_set = set(ticket)

        hits_naturales = len(ticket_set.intersection(naturales_reales))
        has_adicional = adicional_real in ticket_set

        return hits_naturales, has_adicional

    def calculate_prize(self, hits_naturales: int, has_adicional: bool) -> float:
        """
        Calcula el premio intentando el match exacto y luego el fallback.
        Ej: 3 Naturales + Adicional -> No existe en tabla -> Paga como 3 Naturales.
        """
        # 1. Intento Exacto (Prioridad a premios con Adicional)
        prize = self.pay_table.get((hits_naturales, has_adicional))
        if prize is not None:
            return prize

        # 2. Intento Fallback (Si tiene adicional pero no hay premio especial, paga el base)
        if has_adicional:
            prize = self.pay_table.get((hits_naturales, False))
            if prize is not None:
                return prize

        return 0.0
