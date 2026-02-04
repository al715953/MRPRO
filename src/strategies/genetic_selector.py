# src/strategies/genetic_selector.py

import numpy as np
from src.domain.dtos import PredictionResultDTO
from src.strategies.genetic.resonance import ResonanceEngine
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
    Estrategia V7.22 OMEGA: Selector de Enjambre Diverso (Diversity Swarm).
    
    Filosofía:
    En lugar de concentrar la apuesta en un solo patrón (Nexus), desplegamos 
    un enjambre de tickets de ALTA CALIDAD que mantienen una distancia geométrica 
    mínima entre sí.
    
    Objetivo:
    Mitigar la varianza del modelo JIT. Si el Top 1 falla, el Top 2 (que es 
    estructuralmente diferente) puede capturar el premio.
    """
    def __init__(self):
        self.resonance_engine = ResonanceEngine()

    def apply_diversity_swarm(self, u_pool, scores_pool, n_tickets, xp):
        """
        Algoritmo OMEGA:
        1. Ordena candidatos por Score Descendente.
        2. Selecciona el mejor absoluto (Alpha).
        3. Rellena el resto buscando candidatos de alto score que NO se parezcan
           demasiado a los que ya seleccionamos (Diversidad > 1 número).
        """
        if len(u_pool) == 0: return []
        
        # 1. Ordenamiento Jerárquico (Los mejores arriba)
        # Usamos argsort en reversa
        sort_idx = xp.argsort(scores_pool)[::-1]
        
        # Zona de Caza: Top 2000 (Suficiente para encontrar 20 diversos)
        search_depth = min(len(u_pool), 2000) 
        pool_idx = sort_idx[:search_depth]
        
        # Traemos la data a CPU para iterar rápido (Python loops en GPU son lentos)
        pool_candidates = u_pool[pool_idx]
        if hasattr(pool_candidates, "get"): pool_candidates = pool_candidates.get()
        
        selected_tickets = []
        
        # 2. Selección Greedy con Filtro de Diversidad
        # Max Overlap 4 significa: Aceptamos tickets que compartan hasta 4 números.
        # Rechazamos si comparten 5 o 6 (son clones o casi clones).
        max_overlap = 4 
        
        for candidate in pool_candidates:
            if len(selected_tickets) >= n_tickets:
                break
                
            cand_list = candidate.tolist()
            
            if len(selected_tickets) == 0:
                # El Alpha entra siempre
                selected_tickets.append(cand_list)
                continue
                
            # Chequeo de Diversidad (Hamming Soft)
            is_diverse = True
            cand_set = set(cand_list)
            
            for existing in selected_tickets:
                # Calculamos intersección
                overlap = len(cand_set.intersection(existing))
                if overlap > max_overlap:
                    is_diverse = False
                    break
            
            if is_diverse:
                selected_tickets.append(cand_list)
                
        # 3. Fallback (Relleno de Emergencia)
        # Si fuimos muy estrictos y nos faltan tickets, rellenamos con los mejores
        # disponibles aunque se parezcan, para no entregar menos de 20.
        if len(selected_tickets) < n_tickets:
            # print(f"⚠️ Alerta Swarm: Rellenando {n_tickets - len(selected_tickets)} cupos sin filtro.")
            for candidate in pool_candidates:
                cand_list = candidate.tolist()
                if cand_list not in selected_tickets:
                    selected_tickets.append(cand_list)
                    if len(selected_tickets) >= n_tickets:
                        break
                        
        return selected_tickets

    def predict(self, history, config) -> PredictionResultDTO:
        univ = config.raw_universe_ptr
        if univ is None or len(univ) == 0:
            return PredictionResultDTO("Empty_Input", [])

        xp = cp if (HAS_CUPY and hasattr(univ, "get")) else np
        u_xp = xp.asarray(univ)

        # 1. RESONANCIA (Motor V7.21 Gold Master)
        # Calculamos los scores híbridos (AI + Geo Adaptativo)
        res = self.resonance_engine.calculate_resonance(u_xp, history, config, xp)
        if res is None:
            return PredictionResultDTO("Resonance_Collapse", [])

        # 2. SELECCIÓN OMEGA (Diversity Swarm)
        # Operamos sobre el universo reducido (Top 50%) que ya viene limpio de basura
        final_tickets = self.apply_diversity_swarm(
            res["u_reduced"], 
            res["final_scores_reduced"], 
            config.num_tickets, 
            xp
        )

        # 3. TELEMETRÍA (Auditoría de Ranks)
        # Queremos saber en qué Rank real quedaron los tickets que elegimos
        u_cpu = univ.get() if hasattr(univ, "get") else univ
        
        # Reconstruimos el mapa de scores global para calcular el ranking real
        full_hybrid_map = np.zeros(u_cpu.shape[0], dtype=np.float32)
        scores_cpu = res["final_scores_reduced"].get() if hasattr(res["final_scores_reduced"], "get") else res["final_scores_reduced"]
        idx_cpu = res["radar_indices"].get() if hasattr(res["radar_indices"], "get") else res["radar_indices"]
        full_hybrid_map[idx_cpu] = scores_cpu

        # Ordenamos todos los scores del universo para saber el Rank # exacto
        sorted_global_scores = np.sort(full_hybrid_map)[::-1]
        selected_ranks = []

        for ticket in final_tickets:
            ticket_arr = np.array(ticket)
            # Buscamos el ticket en el universo original para ver su score
            match_idx = np.where((u_cpu == ticket_arr).all(axis=1))[0]
            if len(match_idx) > 0:
                t_score = full_hybrid_map[match_idx[0]]
                # Binary search para encontrar el Rank
                t_rank = np.searchsorted(-sorted_global_scores, -t_score) + 1
                selected_ranks.append(int(t_rank))

        return PredictionResultDTO(
            strategy_name="MRPRO V7.22 OMEGA (Diversity Swarm)",
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