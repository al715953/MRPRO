import time
import itertools
import numpy as np
from colorama import Fore, Style
from src.domain.dtos import PredictionResultDTO
from src.data_access.config import TOTAL_BALLS, TICKET_SIZE
from .backend import UniverseBackend
from .filters import VectorizedFilters, calculate_ac_values

class UniverseReductionStrategy:
    """Motor Sniper V11.9 Modularizado."""
    def __init__(self):
        self.xp, self.backend_name = UniverseBackend.get_xp()
        is_prime_mask = self.xp.array([
            False, False, True, True, False, True, False, True, False, False, 
            False, True, False, True, False, False, False, True, False, True, 
            False, False, False, True, False, False, False, False, False, True, 
            False, True, False, False, False, False, False, True, False, False
        ], dtype=bool)
        self.filters = VectorizedFilters(self.xp, is_prime_mask)

    def predict(self, history, config) -> PredictionResultDTO:
        start_time = time.time()
        cfg = config.filter_overrides
        if cfg.get("verbose", True):
            print(f"🚀 Sniper V11.9 Modular [{self.backend_name}]")

        # 1. Generación
        universe = self.xp.asarray(np.fromiter(
            itertools.chain.from_iterable(itertools.combinations(range(1, TOTAL_BALLS + 1), TICKET_SIZE)),
            dtype=np.uint8).reshape(-1, TICKET_SIZE))

        # 2. Pipeline de Reducción
        universe = self.filters.apply_aggregation(universe, cfg)
        universe = self.filters.apply_structure(universe, cfg)
        universe, d_vecs = self.filters.apply_spatial(universe, cfg)
        
        # Poda de Terminales y Sincronización
        universe, mask_p = self.filters.apply_terminal_poda(universe, cfg)
        d_vecs = tuple(v[mask_p] for v in d_vecs)

        # 3. Poda de Perfiles y AC
        if len(universe) > 0:
            profiles = [f"{int(d_vecs[0][i])}-{int(d_vecs[1][i])}-{int(d_vecs[2][i])}-{int(d_vecs[3][i])}" 
                        for i in range(len(universe))]
            valid = cfg.get("valid_decade_profiles", [])
            mask_prof = self.xp.isin(self.xp.array(profiles), self.xp.array(valid))
            universe = universe[mask_prof]

            if len(universe) > 0:
                univ_cpu = universe.get() if hasattr(universe, 'get') else universe
                ac_vals = calculate_ac_values(univ_cpu)
                universe = universe[self.xp.asarray(ac_vals >= cfg.get("ac_min", 7))]

        # 4. Finalización (Dispersión e Inhibición)
        if len(universe) > 0:
            stds = self.xp.std(universe.astype(self.xp.float32), axis=1)
            universe = universe[(stds >= cfg.get("std_min", 7.5)) & (stds <= cfg.get("std_max", 13.0))]
            
            last_draw = self.xp.array(history.winning_numbers[-1][:6], dtype=self.xp.uint8) if history.winning_numbers else self.xp.array([], dtype=self.xp.uint8)
            if last_draw.size > 0:
                universe = universe[self.xp.sum(self.xp.isin(universe, last_draw), axis=1) <= 1]

        elapsed = time.time() - start_time
        if cfg.get("verbose", True):
            print(f"{Fore.GREEN}✅ PUNTO DULCE: {len(universe):,} tkts ({elapsed:.2f}s){Style.RESET_ALL}")

        res = PredictionResultDTO("Universe V11.9 Mod", [tuple(x) for x in universe])
        res.metadata = {"final_size": len(universe), "execution_time": elapsed}
        return res