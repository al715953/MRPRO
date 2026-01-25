import time
from colorama import Fore, Style
from src.domain.dtos import PredictionResultDTO
from .universe.backend import UniverseBackend
from .universe.filters import VectorizedFilters, calculate_ac_values

class UniverseReductionStrategy:
    def __init__(self):
        self.xp, self.backend_name = UniverseBackend.get_xp()
        self.filters = VectorizedFilters(self.xp)

    def predict(self, history, config) -> PredictionResultDTO:
        start_time = time.time()
        cfg = config.filter_overrides
        verbose = cfg.get("verbose", True)
        
        if verbose: print(f"🚀 Sniper V11.9 [{self.backend_name}]")

        # 1. Generación
        universe = self.filters.generate_universe()
        if verbose: print(f"   ├─ Universo Base: {len(universe):,}")

        # 2. Pipeline de Filtros con Seguimiento
        universe = self.filters.apply_aggregation(universe, cfg)
        if verbose: print(f"   ├─ Agregación (Suma/Raíz): {len(universe):,}")

        universe = self.filters.apply_structure(universe, cfg)
        if verbose: print(f"   ├─ Estructura (Par/Prim/Cont): {len(universe):,}")

        universe, d_vecs = self.filters.apply_spatial(universe, cfg)
        if verbose: print(f"   ├─ Espacial (Décadas): {len(universe):,}")

        universe, mask_sync = self.filters.apply_terminal_poda(universe, cfg)
        d_vecs = tuple(v[mask_sync] for v in d_vecs)
        if verbose: print(f"   ├─ Poda Terminales: {len(universe):,}")

        # 3. Poda Élite
        if len(universe) > 0:
            universe = self.filters.apply_profile_poda(universe, d_vecs, cfg)
            if verbose: print(f"   ├─ Poda Perfiles: {len(universe):,}")

            if len(universe) > 0:
                univ_cpu = universe.get() if hasattr(universe, 'get') else universe
                ac_vals = calculate_ac_values(univ_cpu)
                universe = universe[self.xp.asarray(ac_vals >= cfg.get("ac_min", 7))]
                if verbose: print(f"   ├─ Filtro AC Complexity: {len(universe):,}")

        # 4. Pulido Final
        if len(universe) > 0:
            stds = self.xp.std(universe.astype(self.xp.float32), axis=1)
            universe = universe[(stds >= cfg.get("std_min", 7.5)) & (stds <= cfg.get("std_max", 13.0))]
            
            last_draw = self.xp.array(history.winning_numbers[-1][:6], dtype=self.xp.uint8) if history.winning_numbers else self.xp.array([], dtype=self.xp.uint8)
            if last_draw.size > 0:
                universe = universe[self.xp.sum(self.xp.isin(universe, last_draw), axis=1) <= 1]
            if verbose: print(f"   └─ Pulido Final (Std/Rep): {len(universe):,}")

        elapsed = time.time() - start_time
        if verbose:
            print(f"{Fore.GREEN}✅ PUNTO DULCE: {len(universe):,} tkts ({elapsed:.2f}s){Style.RESET_ALL}")

        return PredictionResultDTO("Universe V11.9 Mod", [tuple(x) for x in universe])