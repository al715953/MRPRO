import numpy as np
import itertools
import numba


@numba.jit(nopython=True, parallel=True)
def calculate_ac_numba(candidates):
    # ... (Mantenemos la lógica de AC Numba para CPU fallback) ...
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
    def __init__(self, xp):
        self.xp = xp
        self.is_prime = self.xp.array(
            [
                False,
                False,
                True,
                True,
                False,
                True,
                False,
                True,
                False,
                False,
                False,
                True,
                False,
                True,
                False,
                False,
                False,
                True,
                False,
                True,
                False,
                False,
                False,
                True,
                False,
                False,
                False,
                False,
                False,
                True,
                False,
                True,
                False,
                False,
                False,
                False,
                False,
                True,
                False,
                False,
            ],
            dtype=bool,
        )

        self.default_profiles = self.xp.array(
            [
                2112,
                1212,
                2121,
                1221,
                1122,
                2211,
                1311,
                3111,
                1131,
                1113,
                222,
                2202,
                1230,
                2022,
                231,
                2310,
                1203,
                3021,
                2220,
            ],
            dtype=self.xp.int32,
        )

    def generate_universe(self):
        raw = np.fromiter(
            itertools.chain.from_iterable(itertools.combinations(range(1, 40), 6)),
            dtype=np.uint8,
        ).reshape(-1, 6)
        return self.xp.asarray(raw)

    # --- NUEVAS CAPAS DE DISSECCIÓN ---

    def apply_entropy_shannon(self, universe, cfg):
        """
        Filtro de Entropía: Mide el 'desorden' de los saltos entre números.
        Elimina combinaciones demasiado predecibles o excesivamente caóticas.
        """
        if len(universe) == 0:
            return universe

        # 1. Calculamos deltas (distancias entre números)
        deltas = self.xp.diff(universe, axis=1).astype(self.xp.float32)

        # 2. Normalizamos para obtener una distribución de probabilidad local
        row_sums = self.xp.sum(deltas, axis=1)[:, self.xp.newaxis]
        p = deltas / row_sums

        # 3. Fórmula de Shannon: -sum(p * log2(p))
        # Añadimos epsilon 1e-9 para evitar log(0)
        entropy = -self.xp.sum(p * self.xp.log2(p + 1e-9), axis=1)

        e_min = cfg.get("entropy_min", 2.1)
        e_max = cfg.get("entropy_max", 2.5)

        mask = (entropy >= e_min) & (entropy <= e_max)
        return universe[mask]

    def apply_digital_root_sum(self, universe, cfg):
        """
        Calcula la raíz digital individual de cada número y su suma total.
        Fórmula: dr(n) = 1 + ((n - 1) % 9)
        """
        if len(universe) == 0:
            return universe

        # Cálculo vectorial de raíces digitales para los 6 números
        roots = 1 + ((universe.astype(self.xp.int32) - 1) % 9)
        sdr_total = self.xp.sum(roots, axis=1)

        sdr_min = cfg.get("sdr_min", 20)
        sdr_max = cfg.get("sdr_max", 45)

        mask = (sdr_total >= sdr_min) & (sdr_total <= sdr_max)
        return universe[mask]

    def apply_positional_limits(self, universe, cfg):
        if len(universe) == 0:
            return universe
        mask = (universe[:, 0] <= cfg.get("f1_max", 11)) & (
            universe[:, 5] >= cfg.get("f6_min", 30)
        )
        return universe[mask]

    def apply_aggregation(self, universe, cfg):
        sums = self.xp.sum(universe, axis=1)
        mask = (sums >= cfg.get("sum_min", 108)) & (sums <= cfg.get("sum_max", 132))
        universe, sums = universe[mask], sums[mask]
        root_mask = self.xp.isin((sums - 1) % 9 + 1, self.xp.array([1, 4, 7, 9]))
        return universe[root_mask]

    def apply_structure(self, universe, cfg):
        evens = self.xp.sum(universe % 2 == 0, axis=1)
        primes = self.xp.sum(self.is_prime[universe], axis=1)
        deltas = self.xp.diff(universe, axis=1)
        mask = (
            (evens >= cfg.get("evens_min", 2))
            & (evens <= cfg.get("evens_max", 4))
            & (primes >= cfg.get("primes_min", 1))
            & (primes <= cfg.get("primes_max", 3))
            & (self.xp.sum(deltas == 1, axis=1) <= cfg.get("max_contig", 1))
        )
        return universe[mask]

    def apply_terminal_poda(self, universe, cfg):
        if len(universe) == 0:
            return universe, self.xp.array([], dtype=bool)
        last_digits = (universe % 10).astype(self.xp.int32)
        counts = self.xp.zeros((len(last_digits), 10), dtype=self.xp.int32)
        for i in range(6):
            self.xp.add.at(
                counts, (self.xp.arange(len(last_digits)), last_digits[:, i]), 1
            )
        mask = self.xp.max(counts, axis=1) <= cfg.get("max_terminals", 3)
        return universe[mask], mask

    def apply_spatial(self, universe, cfg):
        d1 = self.xp.sum((universe >= 1) & (universe <= 10), axis=1)
        d2 = self.xp.sum((universe >= 11) & (universe <= 20), axis=1)
        d3 = self.xp.sum((universe >= 21) & (universe <= 30), axis=1)
        d4 = self.xp.sum((universe >= 31) & (universe <= 39), axis=1)
        mask = (d1 <= 3) & (d2 <= 3) & (d3 <= 3) & (d4 <= 3)
        return universe[mask], (d1[mask], d2[mask], d3[mask], d4[mask])

    def apply_profile_poda(self, universe, d_vecs, cfg):
        hashes = (
            d_vecs[0] * 1000 + d_vecs[1] * 100 + d_vecs[2] * 10 + d_vecs[3]
        ).astype(self.xp.int32)
        return universe[self.xp.isin(hashes, self.default_profiles)]

    def apply_ac_complexity(self, universe, cfg):
        ac_min = cfg.get("ac_min", 7)
        if len(universe) == 0:
            return universe
        if self.xp.__name__ == "cupy":
            idx = [
                (0, 1),
                (0, 2),
                (0, 3),
                (0, 4),
                (0, 5),
                (1, 2),
                (1, 3),
                (1, 4),
                (1, 5),
                (2, 3),
                (2, 4),
                (2, 5),
                (3, 4),
                (3, 5),
                (4, 5),
            ]
            diffs = self.xp.column_stack(
                [universe[:, j] - universe[:, i] for i, j in idx]
            )
            diffs.sort(axis=1)
            changes = self.xp.column_stack(
                [
                    self.xp.ones(diffs.shape[0], dtype=bool),
                    diffs[:, 1:] != diffs[:, :-1],
                ]
            )
            return universe[(self.xp.sum(changes, axis=1) - 5) >= ac_min]
        else:
            return universe[calculate_ac_numba(universe) >= ac_min]
