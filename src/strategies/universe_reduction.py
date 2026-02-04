# src/strategies/universe_reduction.py
import time
import numpy as np
from src.domain.dtos import PredictionResultDTO
from .universe.backend import UniverseBackend
from .universe.filters import VectorizedFilters
from src.data_access.config import BEST_SETTINGS

class UniverseReductionStrategy:
    """Coordinador Sniper V14.1 + V7.17 (Poda Dinámica Activa)."""

    def __init__(self):
        self.xp, self.backend_name = UniverseBackend.get_xp()
        self.filters = VectorizedFilters(self.xp)

    def predict(self, history, config, verbose=True) -> PredictionResultDTO:
        universe = self.reduce(history, config, verbose=verbose)

        if universe is None:
            universe = self.xp.array([], dtype=self.xp.uint8)

        res = PredictionResultDTO(
            strategy_name=f"Sniper V14.1 ({self.backend_name})",
            tickets=universe.tolist() if hasattr(universe, "tolist") else universe,
        )
        res.metadata = {"raw_ndarray": universe, "final_size": len(universe)}
        return res

    def reduce(self, history, config, verbose=True):
        """Lógica V7.17: Integra la exclusión dinámica antes de la generación."""
        start_time = time.time()

        cfg = getattr(config, "filter_overrides", None) or BEST_SETTINGS

        if verbose:
            print(f"🚀 Sniper V7.17 [Mac Mode: {self.backend_name}]")

        # --- NUEVA ETAPA V7.17: INFERENCIA DE EXCLUSIÓN ---
        # 1. Identificamos los números a eliminar basados en el historial
        n_exclude = cfg.get("dynamic_exclude_count", 3)
        excluded_pool = self.filters.get_dynamic_exclusion_pool(history, n_exclude)
        
        if verbose:
            print(f"   ✂️ Poda de Raíz: Excluyendo {excluded_pool} por inercia térmica.")

        # --- ETAPA 1: ORIGEN (Pool Reducido) ---
        # 2. Pasamos la lista de excluidos al generador
        universe = self.filters.generate_universe(excluded_pool=excluded_pool)
        
        if verbose:
            print(f"   ├─ Universo Base (Podado): {len(universe):,}")

        # --- ETAPA 2 EN ADELANTE: FILTROS SNIPER ---
        # (El resto del proceso de filtrado se mantiene igual)
        universe = self.filters.apply_positional_limits(universe, cfg)
        universe = self.filters.apply_aggregation(universe, cfg)
        universe = self.filters.apply_structure(universe, cfg)
        
        universe, mask_p = self.filters.apply_terminal_poda(universe, cfg)
        universe, d_vecs = self.filters.apply_spatial(universe, cfg)

        if len(universe) > 0:
            universe = self.filters.apply_profile_poda(universe, d_vecs, cfg)
            universe = self.filters.apply_entropy_shannon(universe, cfg)
            universe = self.filters.apply_digital_root_sum(universe, cfg)
            universe = self.filters.apply_ac_complexity(universe, cfg)

            # Pulido Final (Std)
            stds = self.xp.std(universe.astype(self.xp.float32), axis=1)
            universe = universe[
                (stds >= cfg.get("std_min", 8.2)) & (stds <= cfg.get("std_max", 12.4))
            ]

        elapsed = time.time() - start_time
        if verbose:
            print(f"✅ UNIVERSO DINÁMICO: {len(universe):,} tickets ({elapsed:.2f}s)")

        return universe