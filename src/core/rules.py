from typing import List, Tuple


class MelateRetroRules:
    """Reglas de Negocio actualizadas para Melate Retro"""

    def __init__(self):
        self.ticket_cost = 10.0
        self.max_number = 39
        self.balls_per_ticket = 6  # Se eligen 6 números naturales

        # Tabla de premios completa según categorías oficiales
        # (Aciertos Naturales, Adicional) -> Premio
        self.pay_table = {
            (6, False): 4650000.0,  # 1er Lugar: 6 naturales
            (5, True): 30000.0,  # 2do Lugar: 5 naturales + adicional
            (5, False): 800.0,  # 3er Lugar: 5 naturales
            (4, False): 150.0,  # 4to Lugar: 4 naturales
            (3, False): 20.0,  # 5to Lugar: 3 naturales
            (2, True): 15.0,  # 6to Lugar: 2 naturales + adicional
            (1, True): 10.0,  # 7mo Lugar: 1 natural + adicional
        }

    def validate_ticket(
        self, ticket: List[int], winning_draw: List[int]
    ) -> Tuple[int, bool]:
        """
        Calcula aciertos.
        winning_draw debe contener 7 números: [N1, N2, N3, N4, N5, N6, Adicional]
        """
        naturales_reales = set(winning_draw[:6])
        adicional_real = winning_draw[6]

        ticket_set = set(ticket)

        hits_naturales = len(ticket_set.intersection(naturales_reales))
        has_adicional = adicional_real in ticket_set

        return hits_naturales, has_adicional

    def calculate_prize(self, hits_naturales: int, has_adicional: bool) -> float:
        # Busca la combinación exacta en la tabla
        return self.pay_table.get((hits_naturales, has_adicional), 0.0)
