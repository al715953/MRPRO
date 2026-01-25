import time
import numpy as np
from src.domain.dtos import PredictionResultDTO
from src.strategies.universe.filters import VectorizedFilters, calculate_ac_values

class UniverseReductionStrategy:
    def __init__(self, xp=np):
        self.xp = xp
        self.filters = VectorizedFilters(xp)

    def predict(self, history, config) -> PredictionResultDTO:
        universe = self.reduce(history, config)
        return PredictionResultDTO(
            strategy_name="Sniper V13.8 (Recovery Engine)",
            tickets=universe.tolist() if hasattr(universe, 'tolist') else universe
        )

    def reduce(self, history, config, verbose=True):
        start_time = time.time()
        cfg = config.filter_overrides if hasattr(config, 'filter_overrides') else config
        
        if verbose: print(f"🚀 Sniper V13.8 [Hardware: {self.xp.__name__}]")

        # 1. Generación y Frontera
        universe = self.filters.generate_universe()
        if verbose: print(f"   ├─ Universo Base: {len(universe):,}")
        
        universe = self.filters.apply_positional_limits(universe, cfg)
        if verbose: print(f"   ├─ Frontera Posicional: {len(universe):,}")

        # 2. Agregación y Estructura
        universe = self.filters.apply_aggregation(universe, cfg)
        if verbose: print(f"   ├─ Agregación (Suma/Raíz): {len(universe):,}")

        universe = self.filters.apply_structure(universe, cfg)
        if verbose: print(f"   ├─ Estructura (Par/Prim/Cont): {len(universe):,}")

        # 3. Poda y Espacial (Sincronizado)
        universe, _ = self.filters.apply_terminal_poda(universe, cfg)
        if verbose: print(f"   ├─ Poda Terminales: {len(universe):,}")

        universe, d_vecs = self.filters.apply_spatial(universe, cfg)
        if verbose: print(f"   ├─ Espacial (Décadas): {len(universe):,}")

        universe = self.filters.apply_profile_poda(universe, d_vecs, cfg)
        if verbose: print(f"   ├─ Poda Perfiles: {len(universe):,}")

        # 4. Complejidad y Pulido
        univ_cpu = universe.get() if hasattr(universe, 'get') else universe
        ac_vals = calculate_ac_values(univ_cpu)
        mask_ac = self.xp.asarray(ac_vals >= cfg.get("ac_min", 7))
        universe = universe[mask_ac]
        if verbose: print(f"   ├─ Complejidad AC: {len(universe):,}")
        
        stds = self.xp.std(universe.astype(float), axis=1)
        universe = universe[(stds >= 7.8) & (stds <= 12.8)]
        if verbose: print(f"   ├─ Pulido Final (Std): {len(universe):,}")

        elapsed = time.time() - start_time
        if verbose: 
            print(f"✅ PUNTO DULCE: {len(universe):,} tickets ({elapsed:.2f}s)")
        
        return universe