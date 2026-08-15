# src/strategies/genetic_selector.py

import numpy as np
from src.domain.dtos import PredictionResultDTO
from src.strategies.genetic.resonance import ResonanceEngine
from src.data_access.config import GPU_ENABLED
from src.strategies.genetic.fitness import (
    select_tickets_v16,
    FitnessConfig,
    StrataConfig,
)

# Sincronización de hardware
try:
    import cupy as cp

    HAS_CUPY = True
except ImportError:
    HAS_CUPY = False


class GeneticSelectorStrategy:
    """
    Estrategia V15: OMEGA STRIDE (Cobertura Extendida Determinista).

    Diagnóstico:
    El premio cae consistentemente en Top 30 (ej. Rank #27).
    Con 20 tickets fijos (1-20) perdemos el #27.
    Con aleatoriedad (V14) perdemos el #27 por mala suerte.

    Solución:
    Expandimos el radio de acción a 30 usando "Stride" (Pasos).
    Seleccionamos tickets estratégicamente espaciados en el Top 30
    para maximizar la probabilidad de captura por "vecindad".

    Patrón de Selección (20 Tickets):
    - Top 10: FIJOS (1, 2, 3... 10). Asegura premios obvios.
    - Next 10: STRIDE 2 (12, 14, 16... 30). Cubre hasta el Rank 30.
    """

    def __init__(self, model_path=None):
        self.resonance_engine = ResonanceEngine(model_path=model_path)

    @property
    def training_cutoff_contest(self):
        return self.resonance_engine.training_cutoff_contest

    @property
    def temporal_holdout_auc(self):
        return self.resonance_engine.temporal_holdout_auc

    @property
    def ai_signal_enabled(self):
        return self.resonance_engine.ai_signal_enabled

    @property
    def ai_signal_validated(self):
        return self.resonance_engine.ai_signal_validated

    def apply_omega_stride(self, u_pool, scores_pool, n_tickets, xp):
        if len(u_pool) == 0:
            return []

        # Transferencia a CPU
        if hasattr(u_pool, "get"):
            pool_cpu = u_pool.get()
            scores_cpu = scores_pool.get()
        else:
            pool_cpu = u_pool
            scores_cpu = scores_pool

        n_candidates = len(pool_cpu)
        if n_candidates <= n_tickets:
            return pool_cpu.tolist()

        final_selection = []
        selected_indices = set()

        # Ordenamos por score descendente (El Ranking es sagrado)
        sorted_indices = np.argsort(scores_cpu)[::-1]

        # FASE 1: NUCLEO DURO (Top 10)
        # Nadie perdona al Rank 1, 2, 3...
        n_core = 10
        for i in range(n_core):
            idx = sorted_indices[i]
            final_selection.append(pool_cpu[idx].tolist())
            selected_indices.add(idx)

        # FASE 2: EXTENSIÓN (Stride 2 hasta llenar)
        # Saltamos de 2 en 2 para llegar más lejos (hasta Rank 30 aprox)
        current_rank_ptr = n_core  # Empezamos en el 10 (que es el 11vo)

        while len(final_selection) < n_tickets and current_rank_ptr < n_candidates:
            # Tomamos el siguiente disponible
            idx = sorted_indices[current_rank_ptr]

            if idx not in selected_indices:
                final_selection.append(pool_cpu[idx].tolist())
                selected_indices.add(idx)

            # SALTO ESTRATÉGICO
            # Saltamos 1 candidato (Stride 2).
            # Si jugamos el 11, saltamos el 12, jugamos el 13.
            # Esto nos permite cubrir hasta el Rank 30 con los 10 tickets restantes.
            current_rank_ptr += 2

        return final_selection

    def predict(self, history, config) -> PredictionResultDTO:
        univ = config.raw_universe_ptr
        if univ is None or len(univ) == 0:
            return PredictionResultDTO("Empty_Input", [])

        if HAS_CUPY:
            xp = cp.get_array_module(univ)
        else:
            xp = np

        #        xp = cp if (HAS_CUPY and hasattr(univ, "get")) else np
        # Aseguramos que univ sea del tipo correcto para el backend detectado
        u_xp = xp.asarray(univ)

        # 1. RESONANCIA (Motor V11)
        res = self.resonance_engine.calculate_resonance(u_xp, history, config, xp)
        if res is None:
            return PredictionResultDTO("Resonance_Collapse", [])

        # 2. SELECCIÓN V15 (Omega Stride)
        # final_tickets = self.apply_omega_stride(
        #    res["u_reduced"], res["final_scores_reduced"], config.num_tickets, xp
        # )
        # 2 Nueva logica de 200

        final_tickets, dbg = select_tickets_v16(
            res["u_reduced"],
            res["final_scores_reduced"],
            n_tickets=config.num_tickets,
            xp=xp,
            cfg=FitnessConfig(focus_max_rank=200, candidate_max_rank=500),
            strata=StrataConfig(rank_edges=(10, 30, 60, 100, 150, 200, 500)),
        )
        # si quieres, puedes guardar dbg["selected_ranks"] directo

        # 3. TELEMETRÍA
        u_cpu = univ.get() if hasattr(univ, "get") else univ
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

        full_hybrid_map = np.zeros(u_cpu.shape[0], dtype=np.float32)
        full_hybrid_map[idx_cpu] = scores_cpu
        sorted_global_scores = np.sort(full_hybrid_map)[::-1]

        selected_ranks = []
        for ticket in final_tickets:
            ticket_arr = np.array(ticket)
            match_mask = (u_cpu == ticket_arr).all(axis=1)
            if np.any(match_mask):
                match_idx = np.where(match_mask)[0][0]
                t_score = full_hybrid_map[match_idx]
                t_rank = np.searchsorted(-sorted_global_scores, -t_score) + 1
                selected_ranks.append(int(t_rank))
            else:
                selected_ranks.append(-1)

        ai_raw = res.get("ai_norm")
        geo_raw = res.get("geo_scores")

        return PredictionResultDTO(
            strategy_name="MRPRO V15 (Omega Stride)",
            tickets=final_tickets,
            metadata={
                "universe": u_cpu,
                "selected_ranks": selected_ranks,
                "radar_indices": idx_cpu,
                "ai_scores": ai_raw.get() if hasattr(ai_raw, "get") else ai_raw,
                "geo_scores": geo_raw.get() if hasattr(geo_raw, "get") else geo_raw,
                "hybrid_scores": full_hybrid_map,
                "tickets": final_tickets,
                "ai_signal_enabled": res.get("ai_signal_enabled", True),
                "ai_signal_validated": res.get("ai_signal_validated", True),
                "temporal_holdout_auc": res.get("temporal_holdout_auc"),
            },
        )
