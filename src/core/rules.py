from typing import List, Dict

class MelateRetroRules:
    """Reglas de Negocio para Melate Retro"""
    def __init__(self):
        self.ticket_cost = 10.0
        self.max_number = 39
        self.balls_per_ticket = 6
        
        # Tabla de premios APROXIMADA (Ajustar según reglas vigentes)
        self.pay_table = {
            6: 5000000.0, # Bolsa garantizada mínima
            5: 50000.0,
            4: 1500.0,
            3: 50.0,
            2: 10.0,      # Reintegro o ganancia mínima
            1: 0.0,
            0: 0.0
        }

    def validate_ticket(self, ticket: List[int], winning_numbers: List[int]) -> int:
        """Devuelve número de aciertos (intersección de conjuntos)"""
        return len(set(ticket).intersection(set(winning_numbers)))

    def calculate_prize(self, hits: int) -> float:
        return self.pay_table.get(hits, 0.0)