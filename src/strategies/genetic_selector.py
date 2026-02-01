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
    def __init__(self):
        self.resonance_engine = ResonanceEngine()
        self.cloud_generator = CloudGenerator()
        self.competitive_mesh = CompetitiveMesh(self.cloud_generator)

    def predict(self, history, config) -> PredictionResultDTO:
        univ = config.raw_universe_ptr
        if univ is None or len(univ) == 0:
            return PredictionResultDTO("Empty_Input", [])

        xp = cp if (HAS_CUPY and hasattr(univ, "get")) else np
        u_xp = xp.asarray(univ)

        # 1. RESONANCIA
        res = self.resonance_engine.calculate_resonance(u_xp, history, config, xp)
        if res is None:
            return PredictionResultDTO("Resonance_Collapse", [])

        # 2. MALLA COMPETITIVA
        final_tickets = self.competitive_mesh.apply_mesh(
            res["u_reduced"],
            res["final_scores_reduced"],
            config.num_tickets,
            res["geo_matrix_xp"],
            xp,
            thermal_numbers=res.get("thermal_numbers"),  # <--- Inyección térmica
        )

        # 3. FIX DE DISTANCIA Y RANK (Sincronización CPU/GPU)
        # Bajamos datos a CPU para que el motor forense pueda leerlos
        u_cpu = univ.get() if hasattr(univ, "get") else univ

        # Reconstruimos el mapa de scores en CPU para auditoría
        full_hybrid_map = np.zeros(u_cpu.shape[0], dtype=np.float32)
        scores_cpu = (
            res["final_scores_reduced"].get()
            if hasattr(res["final_scores_reduced"], "get")
            else res["final_scores_reduced"]
        )
        idx_cpu = (
            res["radar_indices"].get()
            if hasattr(res["radar_indices"], "get")
            else res["radar_indices"]
        )
        full_hybrid_map[idx_cpu] = scores_cpu

        # --- CÁLCULO DE SELECTED RANKS (Para quitar el 999) ---
        # Identificamos qué posición (Rank) ocupa cada ticket que elegimos
        selected_ranks = []
        # Ordenamos los scores globales para saber la jerarquía
        sorted_global_scores = np.sort(full_hybrid_map)[::-1]

        for ticket in final_tickets:
            # Buscamos el score de este ticket en el pool original
            # Nota: Esto asume que el ticket está en el radar_indices
            ticket_arr = np.array(ticket)
            match_idx = np.where((u_cpu == ticket_arr).all(axis=1))[0]
            if len(match_idx) > 0:
                t_score = full_hybrid_map[match_idx[0]]
                t_rank = np.searchsorted(-sorted_global_scores, -t_score) + 1
                selected_ranks.append(int(t_rank))

        return PredictionResultDTO(
            strategy_name=f"MRPRO {config.filter_overrides.get('VERSION_TAG', 'V7.x2')}",
            tickets=final_tickets,
            metadata={
                "universe": u_cpu,
                "ai_scores": (
                    res.get("ai_norm").get()
                    if hasattr(res.get("ai_norm"), "get")
                    else res.get("ai_norm")
                ),
                "geo_scores": (
                    res.get("geo_scores").get()
                    if hasattr(res.get("geo_scores"), "get")
                    else res.get("geo_scores")
                ),
                "hybrid_scores": full_hybrid_map,
                "selected_ranks": selected_ranks,  # <--- EL FIX PARA DIST
                "radar_indices": idx_cpu,
            },
        )
