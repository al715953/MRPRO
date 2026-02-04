# src/strategies/genetic_selector.py

import numpy as np
from collections import Counter
from src.domain.dtos import PredictionResultDTO
from src.strategies.genetic.resonance import ResonanceEngine
from src.strategies.genetic.diffusion import CloudGenerator
from src.strategies.genetic.mesh import CompetitiveMesh
from src.data_access.config import GPU_ENABLED, BEST_SETTINGS

# Sincronización de hardware con la configuración global (Mac/PC)
HAS_CUPY = GPU_ENABLED
if HAS_CUPY:
    try:
        import cupy as cp
    except ImportError:
        HAS_CUPY = False


class GeneticSelectorStrategy:
    """
    Estrategia V7.17: Selector Genético con Concentración Nexus.
    Evita la dispersión de aciertos agrupando los números con mayor resonancia.
    """
    def __init__(self):
        self.resonance_engine = ResonanceEngine()
        self.cloud_generator = CloudGenerator()
        self.competitive_mesh = CompetitiveMesh(self.cloud_generator)

    def apply_anchor_nexus(self, final_tickets, u_reduced, config):
        """
        Lógica V7.17: Anclaje Nexus.
        Identifica los números con mayor 'anclaje' en el Top 100 de la IA
        y los inyecta en la mayoría de los tickets finales.
        """
        n_anclas = config.filter_overrides.get("anchor_nexus_size", 3)
        density = config.filter_overrides.get("nexus_density", 0.8)
        n_tickets = len(final_tickets)
        
        if n_tickets == 0: return final_tickets

        # 1. Identificar las anclas del Top 100 (Candidatos con mayor AI_Score)
        # u_reduced viene ordenado por score del ResonanceEngine
        top_pool = u_reduced[:100]
        if hasattr(top_pool, "get"): top_pool = top_pool.get() # Sincronización CPU
        
        flat_nums = [n for ticket in top_pool for n in ticket]
        anclas = [num for num, count in Counter(flat_nums).most_common(n_anclas)]
        
        # 2. Re-estructuración de tickets
        nexus_tickets = []
        limit = int(n_tickets * density)
        
        for i, ticket in enumerate(final_tickets):
            if i < limit:
                # Ticket de Concentración: Forzamos las anclas
                new_tkt = set(anclas)
                source_nums = list(ticket)
                # Rellenamos hasta completar 6 números únicos
                for n in source_nums:
                    if len(new_tkt) >= 6: break
                    new_tkt.add(n)
                nexus_tickets.append(sorted(list(new_tkt)))
            else:
                # Ticket de Exploración: Mantenemos la diversidad original
                nexus_tickets.append(ticket)
                
        return nexus_tickets

    def predict(self, history, config) -> PredictionResultDTO:
        univ = config.raw_universe_ptr
        if univ is None or len(univ) == 0:
            return PredictionResultDTO("Empty_Input", [])

        xp = cp if (HAS_CUPY and hasattr(univ, "get")) else np
        u_xp = xp.asarray(univ)

        # 1. RESONANCIA (IA Scorer + Geo Resonance)
        res = self.resonance_engine.calculate_resonance(u_xp, history, config, xp)
        if res is None:
            return PredictionResultDTO("Resonance_Collapse", [])

        # 2. MALLA COMPETITIVA (Evolución Genética)
        # Retorna los tickets optimizados pero dispersos
        raw_tickets = self.competitive_mesh.apply_mesh(
            res["u_reduced"],
            res["final_scores_reduced"],
            config.num_tickets,
            res["geo_matrix_xp"],
            xp,
            thermal_numbers=res.get("thermal_numbers"),
        )

        # 3. APLICACIÓN DE CONCENTRACIÓN NEXUS (Novedad V7.17)
        # Evita que los 6 aciertos queden repartidos en 20 tickets
        final_tickets = self.apply_anchor_nexus(raw_tickets, res["u_reduced"], config)

        # 4. SINCRONIZACIÓN Y TELEMETRÍA PARA BACKTESTER
        u_cpu = univ.get() if hasattr(univ, "get") else univ
        
        # Reconstrucción del mapa de scores para auditoría forense
        full_hybrid_map = np.zeros(u_cpu.shape[0], dtype=np.float32)
        scores_cpu = res["final_scores_reduced"].get() if hasattr(res["final_scores_reduced"], "get") else res["final_scores_reduced"]
        idx_cpu = res["radar_indices"].get() if hasattr(res["radar_indices"], "get") else res["radar_indices"]
        full_hybrid_map[idx_cpu] = scores_cpu

        # Fix de Distancia y Rank para el reporte de Opción 6
        selected_ranks = []
        sorted_global_scores = np.sort(full_hybrid_map)[::-1]

        for ticket in final_tickets:
            ticket_arr = np.array(ticket)
            match_idx = np.where((u_cpu == ticket_arr).all(axis=1))[0]
            if len(match_idx) > 0:
                t_score = full_hybrid_map[match_idx[0]]
                t_rank = np.searchsorted(-sorted_global_scores, -t_score) + 1
                selected_ranks.append(int(t_rank))

        return PredictionResultDTO(
            strategy_name=f"MRPRO V7.17 (Nexus Density: {config.filter_overrides.get('nexus_density', 0.8)})",
            tickets=final_tickets,
            metadata={
                "universe": u_cpu,
                "ai_scores": res.get("ai_norm").get() if hasattr(res.get("ai_norm"), "get") else res.get("ai_norm"),
                "geo_scores": res.get("geo_scores").get() if hasattr(res.get("geo_scores"), "get") else res.get("geo_scores"),
                "hybrid_scores": full_hybrid_map,
                "selected_ranks": selected_ranks,
                "radar_indices": idx_cpu,
            },
        )