import numpy as np
import itertools
import numba
from src.data_access.config import TOTAL_BALLS, TICKET_SIZE

@numba.jit(nopython=True, parallel=True)
def calculate_ac_values(candidates):
    n_rows = candidates.shape[0]
    ac_results = np.empty(n_rows, dtype=np.int8)
    for i in numba.prange(n_rows):
        row = candidates[i]
        diffs = np.zeros(15, dtype=np.int16) 
        count = 0
        for j in range(6):
            for k in range(j + 1, 6):
                d = row[k] - row[j]
                exists = False
                for m in range(count):
                    if diffs[m] == d:
                        exists = True
                        break
                if not exists:
                    diffs[count] = d
                    count += 1
        ac_results[i] = count - 5
    return ac_results

class VectorizedFilters:
    """Librería Sniper V11.9.1: Reglas con Red de Seguridad."""
    def __init__(self, xp):
        self.xp = xp
        self.is_prime = xp.array([
            False, False, True, True, False, True, False, True, False, False, 
            False, True, False, True, False, False, False, True, False, True, 
            False, False, False, True, False, False, False, False, False, True, 
            False, True, False, False, False, False, False, True, False, False
        ], dtype=bool)
        # Perfiles de Élite como respaldo para asegurar el "Punto Dulce"
        self.default_profiles = ["2-1-2-1", "1-2-2-1", "2-2-1-1", "1-1-2-2", "2-1-1-2", "1-2-1-2"]

    def generate_universe(self):
        raw = np.fromiter(itertools.chain.from_iterable(itertools.combinations(range(1, 40), 6)),
                          dtype=np.uint8).reshape(-1, 6)
        return self.xp.asarray(raw)

    def apply_aggregation(self, universe, cfg):
        sums = self.xp.sum(universe, axis=1)
        # 84k Target: 105-135 es el rango ideal
        mask = (sums >= cfg.get("sum_min", 105)) & (sums <= cfg.get("sum_max", 135))
        universe, sums = universe[mask], sums[mask]
        root_mask = self.xp.isin((sums - 1) % 9 + 1, self.xp.array([1, 4, 9]))
        return universe[root_mask]

    def apply_structure(self, universe, cfg):
        evens = self.xp.sum(universe % 2 == 0, axis=1)
        primes = self.xp.sum(self.is_prime[universe], axis=1)
        deltas = self.xp.diff(universe, axis=1)
        mask = (evens >= cfg.get("even_min", 2)) & (evens <= cfg.get("even_max", 4)) & \
               (primes >= cfg.get("prime_min", 1)) & (primes <= cfg.get("prime_max", 3)) & \
               (self.xp.sum(deltas == 1, axis=1) <= 1) & \
               (self.xp.max(deltas, axis=1) <= cfg.get("max_delta", 15))
        return universe[mask]

    def apply_spatial(self, universe, cfg):
        m_decade = cfg.get("max_per_decade", 3)
        lows = self.xp.sum(universe <= 19, axis=1)
        d1 = self.xp.sum((universe >= 1) & (universe <= 10), axis=1)
        d2 = self.xp.sum((universe >= 11) & (universe <= 20), axis=1)
        d3 = self.xp.sum((universe >= 21) & (universe <= 30), axis=1)
        d4 = self.xp.sum((universe >= 31) & (universe <= 39), axis=1)
        mask = (d1 <= m_decade) & (d2 <= m_decade) & (d3 <= m_decade) & (d4 <= m_decade) & \
               (lows >= 2) & (lows <= 4)
        return universe[mask], (d1[mask], d2[mask], d3[mask], d4[mask])

    def apply_terminal_poda(self, universe, cfg):
        if len(universe) == 0: return universe, self.xp.array([], dtype=bool)
        univ_cpu = universe.get() if hasattr(universe, 'get') else universe
        last_digits = univ_cpu % 10
        counts = np.zeros((len(last_digits), 10), dtype=np.uint8)
        for i in range(6):
            counts[np.arange(len(last_digits)), last_digits[:, i]] += 1
        max_term = counts.max(axis=1)
        # Ajuste a 3 para el Golden Ratio (84k)
        mask = self.xp.asarray(max_term <= cfg.get("max_same_last_digit", 3))
        return universe[mask], mask

    def apply_profile_poda(self, universe, d_vecs, cfg):
        # Si config no trae perfiles, usamos el fallback para no perder el Punto Dulce
        valid = cfg.get("valid_decade_profiles", [])
        if not valid: valid = self.default_profiles
        
        profiles = [f"{int(d_vecs[0][i])}-{int(d_vecs[1][i])}-{int(d_vecs[2][i])}-{int(d_vecs[3][i])}" 
                    for i in range(len(universe))]
        mask = self.xp.isin(self.xp.array(profiles), self.xp.array(valid))
        return universe[mask]