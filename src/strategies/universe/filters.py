# src/strategies/universe/filters.py
import numpy as np
import itertools
import numba
from src.data_access.config import BEST_SETTINGS  # Importamos la configuración


@numba.jit(nopython=True, parallel=True)
def calculate_ac_numba(candidates):
    """Cálculo de complejidad aritmética optimizado para MacBook Air."""
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

    def get_sniper_exclusion(self, history, threshold=0.85, weights=None, n_exclude=1):
        """
        PROTOCOLO E1-SNIPER (Opción B): Exclusión Quirúrgica.
        weights: Tupla (w_gap, w_term, w_freq).
        AHORA RETORNA TUPLA: (lista_excluidos, mensaje_log)
        """
        if not history or not hasattr(history, "winning_numbers"):
            return [], ""
        if n_exclude <= 0:
            return [], ""

        draws = history.winning_numbers
        # w_gap, w_term, w_freq = 0.25, 0.10, 0.60

        if weights:
            w_gap, w_term, w_freq = weights
        else:
            w_gap = BEST_SETTINGS.get("w_gap", 0.25)
            w_term = BEST_SETTINGS.get("w_term", 0.10)
            w_freq = BEST_SETTINGS.get("w_freq", 0.60)

        gaps = self._calculate_gaps(draws)
        max_gap = max(gaps.values()) if gaps else 1

        last_10 = draws[-10:]
        flat_10 = [n % 10 for d in last_10 for n in d[:6]]
        last_50 = draws[-50:]
        flat_50 = [n for d in last_50 for n in d[:6]]

        last_draw = set(draws[-1][:6])

        scores = {n: 0.0 for n in range(1, 40)}
        for n in range(1, 40):
            scores[n] += (gaps[n] / max_gap) * w_gap
            if flat_10.count(n % 10) > 3:
                scores[n] += w_term
            if flat_50.count(n) > 8:
                scores[n] += w_freq

        sorted_candidates = sorted(scores.items(), key=lambda x: x[1], reverse=True)

        excluded = []
        labels = []
        for num, score in sorted_candidates:
            if num in last_draw:
                continue
            if score < threshold:
                break
            excluded.append(num)
            labels.append(f"-{num}({score:.2f})")
            if len(excluded) >= n_exclude:
                break

        if excluded:
            return excluded, f"Sniper:{','.join(labels)}"

        # CORRECCIÓN: Retornamos tupla vacía compatible
        return [], ""

    def get_dynamic_exclusion_pool(self, history, n_exclude=2):
        """Versión legacy para compatibilidad (no usada por Sniper E1)."""
        if not history or not hasattr(history, "winning_numbers"):
            return []

        draws = history.winning_numbers
        juries = [
            {"gap": 0.50, "term": 0.10, "freq": 0.40},
            {"gap": 0.50, "term": 0.20, "freq": 0.30},
            {"gap": 0.50, "term": 0.30, "freq": 0.20},
        ]

        ensemble_votes = {n: 0.0 for n in range(1, 40)}
        gaps = self._calculate_gaps(draws)
        max_gap = max(gaps.values()) if gaps else 1

        last_10 = draws[-10:]
        flat_10 = [n % 10 for d in last_10 for n in d[:6]]
        last_50 = draws[-50:]
        flat_50 = [n for d in last_50 for n in d[:6]]

        for config in juries:
            scores = {n: 0.0 for n in range(1, 40)}
            for n in range(1, 40):
                scores[n] += (gaps[n] / max_gap) * config["gap"]
                if flat_10.count(n % 10) > 3:
                    scores[n] += config["term"]
                if flat_50.count(n) > 8:
                    scores[n] += config["freq"]

            top_voted = sorted(scores.items(), key=lambda x: x[1], reverse=True)[
                :n_exclude
            ]
            for num, _ in top_voted:
                ensemble_votes[num] += 1

        final_exclusion = sorted(
            ensemble_votes.items(), key=lambda x: x[1], reverse=True
        )
        return [final_exclusion[i][0] for i in range(n_exclude)]

    def _calculate_gaps(self, winning_numbers):
        num_draws = len(winning_numbers)
        gaps = {i: num_draws for i in range(1, 40)}
        for gap, draw in enumerate(reversed(winning_numbers)):
            for n in draw[:6]:
                if gaps[n] == num_draws:
                    gaps[n] = gap
        return gaps

    def generate_universe(self, excluded_pool=None):
        full_range = set(range(1, 40))
        active_pool = sorted(list(full_range - set(excluded_pool or [])))

        raw = np.fromiter(
            itertools.chain.from_iterable(itertools.combinations(active_pool, 6)),
            dtype=np.uint8,
        ).reshape(-1, 6)
        return self.xp.asarray(raw)

    def apply_aggregation(self, universe, cfg):
        if len(universe) == 0:
            return universe
        sums = self.xp.sum(universe, axis=1)
        mask = (sums >= cfg.get("sum_min", 90)) & (sums <= cfg.get("sum_max", 150))
        return universe[mask]

    def apply_entropy_shannon(self, universe, cfg):
        if len(universe) == 0:
            return universe
        deltas = self.xp.diff(universe, axis=1).astype(self.xp.float32)
        row_sums = self.xp.sum(deltas, axis=1)[:, self.xp.newaxis]
        p = deltas / (row_sums + 1e-9)
        entropy = -self.xp.sum(p * self.xp.log2(p + 1e-9), axis=1)
        mask = (entropy >= cfg.get("entropy_min", 2.10)) & (
            entropy <= cfg.get("entropy_max", 2.50)
        )
        return universe[mask]

    def apply_digital_root_sum(self, universe, cfg):
        if len(universe) == 0:
            return universe
        roots = 1 + ((universe.astype(self.xp.int32) - 1) % 9)
        sdr_total = self.xp.sum(roots, axis=1)
        mask = (sdr_total >= cfg.get("sdr_min", 20)) & (
            sdr_total <= cfg.get("sdr_max", 42)
        )
        return universe[mask]

    def apply_positional_limits(self, universe, cfg):
        if len(universe) == 0:
            return universe
        mask = (universe[:, 0] <= cfg.get("f1_max", 12)) & (
            universe[:, 5] >= cfg.get("f6_min", 28)
        )
        return universe[mask]

    def apply_structure(self, universe, cfg):
        evens = self.xp.sum(universe % 2 == 0, axis=1)
        primes = self.xp.sum(self.is_prime[universe], axis=1)
        deltas = self.xp.diff(universe, axis=1)
        max_contig = cfg.get("max_contig", cfg.get("max_delta", 1))
        mask = (
            (evens >= cfg.get("even_min", 2))
            & (evens <= cfg.get("even_max", 4))
            & (primes >= cfg.get("prime_min", 1))
            & (primes <= cfg.get("prime_max", 4))
            & (self.xp.sum(deltas == 1, axis=1) <= max_contig)
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
        mask = self.xp.max(counts, axis=1) <= cfg.get("max_same_last_digit", 3)
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
        valid_profiles = [
            int(p.replace("-", "")) for p in cfg.get("valid_decade_profiles", [])
        ]
        if not valid_profiles:
            return universe[self.xp.isin(hashes, self.default_profiles)]
        return universe[self.xp.isin(hashes, self.xp.array(valid_profiles))]

    def apply_ac_complexity(self, universe, cfg):
        ac_min = cfg.get("ac_min", 7)
        if len(universe) == 0:
            return universe
        candidates_np = universe if isinstance(universe, np.ndarray) else universe.get()
        return universe[calculate_ac_numba(candidates_np) >= ac_min]

    # ==============================
    # SURVIVAL SCORING ENGINE V15
    # ==============================

    def compute_survival_scores(self, universe, cfg):
        """
        Calcula score acumulado normalizado [0,1]
        SIN descartar tickets.
        GPU-first.
        """

        if len(universe) == 0:
            return self.xp.zeros(0, dtype=self.xp.float32)

        xp = self.xp
        universe_f = universe.astype(xp.float32)

        # ----------------------------
        # 1. Score Suma (gaussiano)
        # ----------------------------
        sums = xp.sum(universe_f, axis=1)
        mean_sum = (cfg.get("sum_min", 110) + cfg.get("sum_max", 132)) / 2.0
        std_sum = (cfg.get("sum_max", 132) - cfg.get("sum_min", 110)) / 4.0
        score_sum = xp.exp(-((sums - mean_sum) ** 2) / (2 * std_sum**2))

        # ----------------------------
        # 2. Score Desviación Estándar
        # ----------------------------
        stds = xp.std(universe_f, axis=1)
        mean_std = (cfg.get("std_min", 8.0) + cfg.get("std_max", 12.0)) / 2.0
        std_std = (cfg.get("std_max", 12.0) - cfg.get("std_min", 8.0)) / 4.0
        score_std = xp.exp(-((stds - mean_std) ** 2) / (2 * std_std**2))

        # ----------------------------
        # 3. Score Paridad Balance
        # ----------------------------
        evens = xp.sum(universe % 2 == 0, axis=1)
        ideal_even = (cfg.get("even_min", 2) + cfg.get("even_max", 4)) / 2.0
        score_even = 1.0 - xp.abs(evens - ideal_even) / 6.0

        # ----------------------------
        # 4. Score Terminal Distribution
        # ----------------------------
        last_digits = universe % 10
        counts = xp.zeros((len(universe), 10), dtype=xp.int32)
        for i in range(6):
            xp.add.at(counts, (xp.arange(len(universe)), last_digits[:, i]), 1)
        max_rep = xp.max(counts, axis=1)
        score_term = 1.0 - (max_rep - 1) / 5.0

        # ----------------------------
        # Score Total (ponderado)
        # ----------------------------
        total_score = (
            0.30 * score_sum + 0.30 * score_std + 0.20 * score_even + 0.20 * score_term
        )

        return total_score.astype(xp.float32)

    # ============================================
    # CORE SURVIVAL SCORE (V15)
    # ============================================
    def compute_survival_scores(self, universe, cfg):

        xp = self.xp
        universe_f = universe.astype(xp.float32)

        sums = xp.sum(universe_f, axis=1)
        mean_sum = (cfg.get("sum_min", 110) + cfg.get("sum_max", 132)) / 2.0
        std_sum = (cfg.get("sum_max", 132) - cfg.get("sum_min", 110)) / 4.0
        score_sum = xp.exp(-((sums - mean_sum) ** 2) / (2 * std_sum**2))

        stds = xp.std(universe_f, axis=1)
        mean_std = (cfg.get("std_min", 8.0) + cfg.get("std_max", 12.0)) / 2.0
        std_std = (cfg.get("std_max", 12.0) - cfg.get("std_min", 8.0)) / 4.0
        score_std = xp.exp(-((stds - mean_std) ** 2) / (2 * std_std**2))

        evens = xp.sum(universe % 2 == 0, axis=1)
        ideal_even = 3.0
        score_even = 1.0 - xp.abs(evens - ideal_even) / 6.0

        total_score = 0.4 * score_sum + 0.4 * score_std + 0.2 * score_even

        return total_score.astype(xp.float32)

    # ============================================
    # TAIL EXPLORATION SCORE (NUEVO)
    # ============================================
    def compute_tail_scores(self, universe, cfg):

        xp = self.xp
        universe_f = universe.astype(xp.float32)

        stds = xp.std(universe_f, axis=1)
        mean_std = (cfg.get("std_min", 8.0) + cfg.get("std_max", 12.0)) / 2.0

        # Distancia al centro (cola controlada)
        tail_score = xp.abs(stds - mean_std)

        # Normalización
        tail_score = tail_score / (xp.max(tail_score) + 1e-9)

        return tail_score.astype(xp.float32)

    # ============================================
    # DENSITY PENALTY (2D BINNING SUM + STD)
    # ============================================
    def compute_density_penalty(self, universe):

        xp = self.xp

        universe_f = universe.astype(xp.float32)

        sums = xp.sum(universe_f, axis=1)
        stds = xp.std(universe_f, axis=1)

        # Binning
        sum_bins = xp.floor((sums - xp.min(sums)) / 3).astype(xp.int32)
        std_bins = xp.floor((stds - xp.min(stds)) / 0.5).astype(xp.int32)

        # Crear índice 2D compacto
        combined_bin = sum_bins * 100 + std_bins

        unique_bins, counts = xp.unique(combined_bin, return_counts=True)

        # Map bin -> density
        bin_to_density = dict(zip(unique_bins.tolist(), counts.tolist()))

        # Construir vector densidad
        density = xp.array(
            [bin_to_density[int(b)] for b in combined_bin.tolist()],
            dtype=xp.float32,
        )

        # Normalización
        density = density / (xp.max(density) + 1e-9)

        return density
