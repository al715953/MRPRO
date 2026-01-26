import time
import numpy as np
from src.domain.dtos import PredictionResultDTO
from .universe.backend import UniverseBackend
from .universe.filters import VectorizedFilters

# IMPORTANTE: Vinculación con configuración global
from src.data_access.config import BEST_SETTINGS


class UniverseReductionStrategy:
    """Coordinador Sniper V14.1: Versión Blindada (Handshake Obligatorio)."""

    def __init__(self):
        self.xp, self.backend_name = UniverseBackend.get_xp()
        self.filters = VectorizedFilters(self.xp)

    def predict(self, history, config, verbose=True) -> PredictionResultDTO:
        """Handshake garantizado: El universo fluye sin pérdidas a la Fase 2."""
        universe = self.reduce(history, config, verbose=verbose)

        # Blindaje: Si el universo falla, devolvemos una estructura vacía pero válida
        if universe is None:
            universe = self.xp.array([], dtype=self.xp.uint8)

        res = PredictionResultDTO(
            strategy_name=f"Sniper V14.1 ({self.backend_name})",
            tickets=universe.tolist() if hasattr(universe, "tolist") else universe,
        )

        # CONTRATO: Esta metadata es sagrada para el Backtester y la Fase 2
        res.metadata = {"raw_ndarray": universe, "final_size": len(universe)}
        return res

    def reduce(self, history, config, verbose=True):
        """Tu lógica original V14.1 intacta con todos tus logs de telemetría."""
        start_time = time.time()

        cfg = getattr(config, "filter_overrides", None)
        if not cfg:
            cfg = BEST_SETTINGS

        if verbose:
            print(f"🚀 Sniper V14.1 [Hardware: {self.backend_name}]")

        # --- ETAPA 1: ORIGEN ---
        universe = self.filters.generate_universe()
        if verbose:
            print(f"   ├─ Universo Base: {len(universe):,}")

        # --- ETAPA 2: FRONTERAS ---
        universe = self.filters.apply_positional_limits(universe, cfg)
        if verbose:
            print(f"   ├─ Frontera Posicional: {len(universe):,}")

        # --- ETAPA 3: MASA MATEMÁTICA ---
        universe = self.filters.apply_aggregation(universe, cfg)
        if verbose:
            print(f"   ├─ Agregación (Suma/Raíz): {len(universe):,}")

        universe = self.filters.apply_structure(universe, cfg)
        if verbose:
            print(f"   ├─ Estructura (Par/Prim/Cont): {len(universe):,}")

        # --- ETAPA 4: PODAS TÁCTICAS ---
        universe, _ = self.filters.apply_terminal_poda(universe, cfg)
        if verbose:
            print(f"   ├─ Poda Terminales: {len(universe):,}")

        universe, d_vecs = self.filters.apply_spatial(universe, cfg)
        if verbose:
            print(f"   ├─ Espacial (Décadas): {len(universe):,}")

        if len(universe) > 0:
            universe = self.filters.apply_profile_poda(universe, d_vecs, cfg)
            if verbose:
                print(f"   ├─ Poda Perfiles: {len(universe):,}")

            # --- NUEVA ETAPA: DISSECCIÓN QUIRÚRGICA ---
            universe = self.filters.apply_entropy_shannon(universe, cfg)
            if verbose:
                print(f"   ├─ Entropía Shannon: {len(universe):,}")

            universe = self.filters.apply_digital_root_sum(universe, cfg)
            if verbose:
                print(f"   ├─ Raíz Digital (SDR): {len(universe):,}")

            # --- ETAPA 5: COMPLEXITY ---
            universe = self.filters.apply_ac_complexity(universe, cfg)
            if verbose:
                print(f"   ├─ Complejidad AC: {len(universe):,}")

            # Pulido final original
            stds = self.xp.std(universe.astype(self.xp.float32), axis=1)
            universe = universe[
                (stds >= cfg.get("std_min", 8.0)) & (stds <= cfg.get("std_max", 12.6))
            ]
            if verbose:
                print(f"   ├─ Pulido Final (Std): {len(universe):,}")

        elapsed = time.time() - start_time
        if verbose:
            print(
                f"✅ PUNTO DULCE RESTAURADO: {len(universe):,} tickets ({elapsed:.2f}s)"
            )

        return universe
