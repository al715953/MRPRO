# src/strategies/genetic_selector.py

import itertools
import math

import numpy as np
from src.domain.dtos import PredictionResultDTO
from src.strategies.genetic.resonance import ResonanceEngine
from src.data_access.config import GPU_ENABLED
from src.strategies.genetic.fitness import (
    DeepDispersionConfig,
    EliteCoverageDeepConfig,
    select_tickets_v16,
    select_core_plus_deep_tickets,
    select_elite_coverage_deep_tickets,
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

    def __init__(self, model_path=None, number_model_path=None):
        self.resonance_engine = ResonanceEngine(
            model_path=model_path,
            number_model_path=number_model_path,
        )

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

    @property
    def number_temporal_holdout_auc(self):
        return self.resonance_engine.number_temporal_holdout_auc

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

    @staticmethod
    def _ensure_soft_veto_reserve(
        final_tickets,
        candidate_tickets,
        candidate_scores,
        soft_numbers,
        reserve_fraction,
    ):
        """Keep a small hedge containing soft-veto numbers without duplicates."""
        numbers = {int(number) for number in (soft_numbers or [])}
        requested = int(
            math.ceil(max(0.0, float(reserve_fraction)) * len(final_tickets))
        )
        target = min(len(final_tickets), requested) if numbers else 0
        if target <= 0 or not final_tickets:
            return final_tickets, {
                "target": 0,
                "actual": 0,
                "replacements": 0,
            }

        selected = [list(map(int, ticket)) for ticket in final_tickets]
        selected_keys = {tuple(ticket) for ticket in selected}
        actual = sum(bool(numbers.intersection(ticket)) for ticket in selected)
        if actual >= target:
            return selected, {
                "target": target,
                "actual": actual,
                "replacements": 0,
            }

        pool = (
            candidate_tickets.get()
            if hasattr(candidate_tickets, "get")
            else np.asarray(candidate_tickets)
        )
        scores = (
            candidate_scores.get()
            if hasattr(candidate_scores, "get")
            else np.asarray(candidate_scores)
        )
        order = np.argsort(np.asarray(scores, dtype=np.float64))[::-1]
        replacements = 0
        for idx in order:
            candidate = sorted(int(number) for number in pool[int(idx)])
            key = tuple(candidate)
            if key in selected_keys or not numbers.intersection(candidate):
                continue
            replace_idx = next(
                (
                    pos
                    for pos in range(len(selected) - 1, -1, -1)
                    if not numbers.intersection(selected[pos])
                ),
                None,
            )
            if replace_idx is None:
                break
            selected_keys.remove(tuple(selected[replace_idx]))
            selected[replace_idx] = candidate
            selected_keys.add(key)
            replacements += 1
            actual += 1
            if actual >= target:
                break
        return selected, {
            "target": target,
            "actual": actual,
            "replacements": replacements,
        }

    @staticmethod
    def _ticket_subset_coverage(tickets):
        canonical = [tuple(sorted(int(number) for number in ticket)) for ticket in tickets]
        return {
            f"selected_unique_{label}": len(
                {
                    subset
                    for ticket in canonical
                    for subset in itertools.combinations(ticket, size)
                }
            )
            for size, label in ((2, "pairs"), (3, "triples"), (4, "quads"))
        }

    @staticmethod
    def _selection_configs(overrides):
        """Build validated selector configs while preserving official defaults."""
        source = overrides if isinstance(overrides, dict) else {}
        try:
            focus = max(1, int(source.get("fitness_focus_max_rank", 200)))
            candidate = max(
                focus, int(source.get("fitness_candidate_max_rank", 500))
            )
        except (TypeError, ValueError):
            focus, candidate = 200, 500

        default_edges = (10, 30, 60, 100, 150, 200, 500)
        try:
            edges = tuple(
                sorted(
                    {
                        int(value)
                        for value in source.get("fitness_rank_edges", default_edges)
                        if int(value) > 0
                    }
                )
            )
        except (TypeError, ValueError):
            edges = default_edges
        if not edges:
            edges = default_edges

        default_plan = FitnessConfig().bucket_plan
        plan = []
        try:
            for raw in source.get("fitness_bucket_plan", default_plan):
                lo, hi, count = (int(value) for value in raw)
                if 1 <= lo <= hi and count > 0:
                    plan.append((lo, hi, count))
        except (TypeError, ValueError):
            plan = []
        if not plan:
            plan = list(default_plan)

        return (
            FitnessConfig(
                focus_max_rank=focus,
                candidate_max_rank=candidate,
                bucket_plan=tuple(plan),
            ),
            StrataConfig(rank_edges=edges),
        )

    @staticmethod
    def _deep_dispersion_config(overrides):
        source = overrides if isinstance(overrides, dict) else {}
        defaults = DeepDispersionConfig()

        def _integer(key, default, minimum=0, maximum=None):
            try:
                value = max(minimum, int(source.get(key, default)))
            except (TypeError, ValueError):
                value = int(default)
            return min(value, maximum) if maximum is not None else value

        def _weight(key, default):
            try:
                return max(0.0, float(source.get(key, default)))
            except (TypeError, ValueError):
                return float(default)

        return DeepDispersionConfig(
            core_tickets=_integer(
                "deep_dispersion_core_tickets", defaults.core_tickets
            ),
            deep_tickets=_integer(
                "deep_dispersion_tickets", defaults.deep_tickets
            ),
            min_deep_rank=_integer(
                "deep_dispersion_min_rank", defaults.min_deep_rank, minimum=1
            ),
            max_overlap_preferred=_integer(
                "deep_dispersion_max_overlap",
                defaults.max_overlap_preferred,
                maximum=6,
            ),
            w_pair_novelty=_weight(
                "deep_dispersion_pair_novelty_weight", defaults.w_pair_novelty
            ),
            w_number_rarity=_weight(
                "deep_dispersion_number_rarity_weight", defaults.w_number_rarity
            ),
            w_dissimilarity=_weight(
                "deep_dispersion_dissimilarity_weight", defaults.w_dissimilarity
            ),
            w_local_quality=_weight(
                "deep_dispersion_local_quality_weight", defaults.w_local_quality
            ),
        )

    @staticmethod
    def _elite_coverage_deep_config(overrides):
        source = overrides if isinstance(overrides, dict) else {}
        defaults = EliteCoverageDeepConfig()

        def _integer(key, default, minimum=0, maximum=None):
            try:
                value = max(minimum, int(source.get(key, default)))
            except (TypeError, ValueError):
                value = int(default)
            return min(value, maximum) if maximum is not None else value

        def _weight(key, default):
            try:
                return max(0.0, float(source.get(key, default)))
            except (TypeError, ValueError):
                return float(default)

        return EliteCoverageDeepConfig(
            elite_tickets=_integer(
                "portfolio_elite_tickets", defaults.elite_tickets
            ),
            coverage_tickets=_integer(
                "portfolio_coverage_tickets", defaults.coverage_tickets
            ),
            deep_tickets=_integer(
                "portfolio_deep_tickets", defaults.deep_tickets
            ),
            coverage_max_rank=_integer(
                "portfolio_coverage_max_rank",
                defaults.coverage_max_rank,
                minimum=1,
            ),
            min_deep_rank=_integer(
                "portfolio_min_deep_rank", defaults.min_deep_rank, minimum=1
            ),
            max_overlap_preferred=_integer(
                "portfolio_max_overlap",
                defaults.max_overlap_preferred,
                maximum=6,
            ),
            w_pair_novelty=_weight(
                "portfolio_pair_novelty_weight", defaults.w_pair_novelty
            ),
            w_triple_novelty=_weight(
                "portfolio_triple_novelty_weight", defaults.w_triple_novelty
            ),
            w_quad_novelty=_weight(
                "portfolio_quad_novelty_weight", defaults.w_quad_novelty
            ),
            w_number_rarity=_weight(
                "portfolio_number_rarity_weight", defaults.w_number_rarity
            ),
            w_dissimilarity=_weight(
                "portfolio_dissimilarity_weight", defaults.w_dissimilarity
            ),
            w_local_quality=_weight(
                "portfolio_local_quality_weight", defaults.w_local_quality
            ),
        )

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

        overrides = getattr(config, "filter_overrides", None) or {}
        fitness_config, strata_config = self._selection_configs(overrides)
        selector_mode = str(
            overrides.get("fitness_selector_mode", "native")
        ).lower()
        deep_config = None
        portfolio_config = None
        if selector_mode == "core_plus_deep":
            deep_config = self._deep_dispersion_config(overrides)
            final_tickets, dbg = select_core_plus_deep_tickets(
                res["u_reduced"],
                res["final_scores_reduced"],
                n_tickets=config.num_tickets,
                xp=xp,
                cfg=fitness_config,
                strata=strata_config,
                deep_cfg=deep_config,
            )
        elif selector_mode == "elite_coverage_deep":
            portfolio_config = self._elite_coverage_deep_config(overrides)
            final_tickets, dbg = select_elite_coverage_deep_tickets(
                res["u_reduced"],
                res["final_scores_reduced"],
                n_tickets=config.num_tickets,
                xp=xp,
                cfg=fitness_config,
                strata=strata_config,
                portfolio_cfg=portfolio_config,
            )
        else:
            selector_mode = "native"
            final_tickets, dbg = select_tickets_v16(
                res["u_reduced"],
                res["final_scores_reduced"],
                n_tickets=config.num_tickets,
                xp=xp,
                cfg=fitness_config,
                strata=strata_config,
            )
        final_tickets, soft_reserve = self._ensure_soft_veto_reserve(
            final_tickets,
            res["u_reduced"],
            res["final_scores_reduced"],
            res.get("sniper_soft_numbers", []),
            overrides.get("sniper_soft_reserve_fraction", 0.0),
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
        idx_cpu = np.asarray(idx_cpu, dtype=np.int64)
        scores_cpu = np.asarray(scores_cpu, dtype=np.float64)

        full_hybrid_map = np.zeros(u_cpu.shape[0], dtype=np.float32)
        full_hybrid_map[idx_cpu] = scores_cpu
        sorted_global_scores = np.sort(full_hybrid_map)[::-1]

        selected_ranks = []
        stable_order = np.argsort(-scores_cpu, kind="stable")
        stable_ranks_reduced = np.empty(len(scores_cpu), dtype=np.int32)
        stable_ranks_reduced[stable_order] = np.arange(
            1, len(scores_cpu) + 1, dtype=np.int32
        )
        stable_rank_by_ticket = {
            tuple(int(number) for number in u_cpu[int(full_idx)]): int(
                stable_ranks_reduced[reduced_idx]
            )
            for reduced_idx, full_idx in enumerate(idx_cpu)
        }
        selected_stable_ranks = []
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
            selected_stable_ranks.append(
                int(stable_rank_by_ticket.get(tuple(int(v) for v in ticket), -1))
            )

        ai_raw = res.get("ai_norm")
        geo_raw = res.get("geo_scores")
        subset_coverage = self._ticket_subset_coverage(final_tickets)

        return PredictionResultDTO(
            strategy_name="MRPRO V17 (Balanced Exploration)",
            tickets=final_tickets,
            metadata={
                "universe": u_cpu,
                "selected_ranks": selected_ranks,
                "selected_stable_ranks": selected_stable_ranks,
                "radar_indices": idx_cpu,
                "ai_scores": ai_raw.get() if hasattr(ai_raw, "get") else ai_raw,
                "geo_scores": geo_raw.get() if hasattr(geo_raw, "get") else geo_raw,
                "hybrid_scores": full_hybrid_map,
                "tickets": final_tickets,
                "ai_signal_enabled": res.get("ai_signal_enabled", True),
                "ai_signal_validated": res.get("ai_signal_validated", True),
                "ai_validation_scope": res.get("ai_validation_scope", "model"),
                "temporal_holdout_auc": res.get("temporal_holdout_auc"),
                "feature_schema": res.get("feature_schema"),
                "number_ai_scores": res.get("number_ai_scores"),
                "number_model_enabled": res.get("number_model_enabled", False),
                "number_model_applied": res.get("number_model_applied", False),
                "number_temporal_holdout_auc": res.get(
                    "number_temporal_holdout_auc"
                ),
                "ai_context_weight": res.get("ai_context_weight"),
                "ai_number_weight": res.get("ai_number_weight"),
                "resonance_blend_mode": res.get("resonance_blend_mode"),
                "hybrid_alpha": res.get("hybrid_alpha"),
                "hybrid_beta": res.get("hybrid_beta"),
                "radar_percentile": res.get("radar_percentile", 50.0),
                "fitness_focus_max_rank": fitness_config.focus_max_rank,
                "fitness_candidate_max_rank": fitness_config.candidate_max_rank,
                "fitness_rank_edges": list(strata_config.rank_edges),
                "fitness_bucket_plan": [
                    list(bucket) for bucket in fitness_config.bucket_plan
                ],
                "fitness_selector_mode": selector_mode,
                "selector_debug_ranks": [
                    int(rank) for rank in dbg.get("selected_ranks", [])
                ],
                "deep_dispersion_core_tickets": (
                    int(deep_config.core_tickets) if deep_config else 0
                ),
                "deep_dispersion_tickets": (
                    int(deep_config.deep_tickets) if deep_config else 0
                ),
                "deep_dispersion_min_rank": (
                    int(deep_config.min_deep_rank) if deep_config else None
                ),
                "deep_dispersion_core_ranks": [
                    int(rank) for rank in dbg.get("core_selected_ranks", [])
                ],
                "deep_dispersion_ranks": [
                    int(rank) for rank in dbg.get("deep_selected_ranks", [])
                ],
                "deep_dispersion_bands": dbg.get("deep_rank_bands", []),
                "deep_dispersion_weights": dbg.get(
                    "deep_dispersion_weights", {}
                ),
                "portfolio_elite_tickets": (
                    int(portfolio_config.elite_tickets) if portfolio_config else 0
                ),
                "portfolio_coverage_tickets": (
                    int(portfolio_config.coverage_tickets)
                    if portfolio_config
                    else 0
                ),
                "portfolio_deep_tickets": (
                    int(portfolio_config.deep_tickets) if portfolio_config else 0
                ),
                "portfolio_elite_ranks": [
                    int(rank) for rank in dbg.get("elite_selected_ranks", [])
                ],
                "portfolio_coverage_ranks": [
                    int(rank) for rank in dbg.get("coverage_selected_ranks", [])
                ],
                "portfolio_deep_ranks": [
                    int(rank) for rank in dbg.get("deep_selected_ranks", [])
                ],
                "portfolio_phase_by_ticket": list(
                    dbg.get("phase_by_ticket", [])
                ),
                "portfolio_unique_pairs": dbg.get("coverage_unique_pairs"),
                "portfolio_unique_triples": dbg.get("coverage_unique_triples"),
                "portfolio_unique_quads": dbg.get("coverage_unique_quads"),
                "portfolio_coverage_weights": dbg.get("coverage_weights", {}),
                **subset_coverage,
                "sniper_soft_numbers": res.get("sniper_soft_numbers", []),
                "sniper_soft_penalty": res.get("sniper_soft_penalty", 0.0),
                "sniper_soft_candidate_count": res.get(
                    "sniper_soft_candidate_count", 0
                ),
                "sniper_soft_reserve_target": soft_reserve["target"],
                "sniper_soft_reserve_actual": soft_reserve["actual"],
                "sniper_soft_reserve_replacements": soft_reserve[
                    "replacements"
                ],
            },
        )
