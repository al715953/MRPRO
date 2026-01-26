# src/strategies/genetic_selector.py

import numpy as np
from src.domain.dtos import PredictionResultDTO

# Importamos los nuevos motores desde la subcarpeta 'genetic'
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
    SELECTOR V6.0: Trinity Split Architecture.
    Fachada que orquesta los motores de Resonancia, Difusión y Malla.
    Mantiene la lógica de V5.9.8 (Omni-Cloud + Power Boost) pero modularizada.
    """

    def __init__(self):
        # Inicializamos la "Trinidad" de motores
        self.resonance_engine = ResonanceEngine()
        self.cloud_generator = CloudGenerator()
        # La malla necesita acceso al generador de nubes
        self.competitive_mesh = CompetitiveMesh(self.cloud_generator)

    def predict(self, history, config) -> PredictionResultDTO:
        univ = config.raw_universe_ptr
        if univ is None or len(univ) == 0:
            return PredictionResultDTO("Empty", [])

        # 0. Selección de dispositivo (Handshake)
        xp = cp if (HAS_CUPY and hasattr(univ, "get")) else np
        u_xp = xp.asarray(univ)

        # 1. MOTOR DE RESONANCIA: Calcula scores, matriz geo y define el radar
        # Se encarga de entrenar la IA si es necesario y aplicar la fusión V5.9.8
        resonance_result = self.resonance_engine.calculate_resonance(
            u_xp, history, config, xp
        )

        # 2. MOTOR DE MALLA: Ejecuta la selección competitiva con difusión
        # Recibe los datos del radar y usa el generador de nubes internamente
        final_tickets = self.competitive_mesh.apply_mesh(
            resonance_result["u_reduced"],
            resonance_result["final_scores_reduced"],
            config.num_tickets,
            resonance_result["geo_matrix_xp"],
            xp,
        )

        # 3. Restauración de Telemetría (Snapshot Facade)
        # Mapeamos los scores del radar al universo completo para el log
        full_hybrid_map = xp.zeros(
            u_xp.shape[0], dtype=resonance_result["ai_norm"].dtype
        )
        full_hybrid_map[resonance_result["radar_indices"]] = resonance_result[
            "final_scores_reduced"
        ]

        snapshot = {
            "universe": univ,
            "ai_scores": resonance_result["ai_norm"],
            "hybrid_scores": full_hybrid_map,
            "geo_scores": resonance_result["geo_scores"],
            "selected_ranks": list(range(1, len(final_tickets) + 1)),
        }

        return (
            PredictionResultDTO(
                "ENGINE V6.0: Trinity Split (Omni-Cloud)", final_tickets
            ),
            snapshot,
        )
