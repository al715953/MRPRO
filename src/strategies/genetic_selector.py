# src/strategies/genetic_selector.py

import numpy as np
from src.domain.dtos import PredictionResultDTO
from src.strategies.genetic.resonance import ResonanceEngine
from src.strategies.genetic.diffusion import CloudGenerator
from src.strategies.genetic.mesh import CompetitiveMesh

try:
    import cupy as cp

    HAS_CUPY = True
except ImportError:
    HAS_CUPY = False


class GeneticSelectorStrategy:
    """
    SELECTOR V6.4: Trinity Split Architecture.
    Corregido: Inclusión de 'universe' en metadata para auditoría forense.
    """

    def __init__(self):
        self.resonance_engine = ResonanceEngine()
        self.cloud_generator = CloudGenerator()
        self.competitive_mesh = CompetitiveMesh(self.cloud_generator)

    def predict(self, history, config) -> PredictionResultDTO:
        univ = config.raw_universe_ptr
        if univ is None or len(univ) == 0:
            return PredictionResultDTO("Empty", [])

        xp = cp if (HAS_CUPY and hasattr(univ, "get")) else np
        u_xp = xp.asarray(univ)

        # 1. MOTOR DE RESONANCIA: Calcula scores y matriz geo
        res = self.resonance_engine.calculate_resonance(u_xp, history, config, xp)

        # 2. MOTOR DE MALLA: Selección competitiva con Salto Armónico
        final_tickets = self.competitive_mesh.apply_mesh(
            res["u_reduced"],
            res["final_scores_reduced"],
            config.num_tickets,
            res["geo_matrix_xp"],
            xp,
        )

        # 3. MAPEO DE TELEMETRÍA (Handshake Forense)
        # Creamos un mapa de scores que coincida exactamente con los índices de 'univ'
        full_hybrid_map = xp.zeros(u_xp.shape[0])
        full_hybrid_map[res["radar_indices"]] = res["final_scores_reduced"]

        # IMPORTANTE: Metadata completa para evitar 'Rank #0' en backtester
        return PredictionResultDTO(
            strategy_name="ENGINE V6.4: Trinity Split (Harmonic Leap)",
            tickets=final_tickets,
            metadata={
                "universe": univ,  # <--- OBLIGATORIO: El mapa físico de búsqueda
                "ai_scores": res["ai_norm"],  # Puntaje de IA base
                "hybrid_scores": full_hybrid_map,  # Puntaje final fusionado
                "geo_scores": res["geo_scores"],  # Puntaje geométrico
                "selected_ranks": list(range(1, len(final_tickets) + 1)),
            },
        )
