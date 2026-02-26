from __future__ import annotations

import numpy as np

_N_POS = 5
_N_DIGITS = 10


def _coerce_rows(digits_list: list[list[int]]) -> np.ndarray:
    rows: list[list[int]] = []
    for row in digits_list or []:
        if row is None or len(row) < _N_POS:
            continue
        rows.append([int(row[i]) % _N_DIGITS for i in range(_N_POS)])
    if not rows:
        return np.empty((_N_POS, 0), dtype=np.int16).T
    return np.asarray(rows, dtype=np.int16)


class PositionalAnalyzers:
    """Analiza 5 camaras independientes, donde cada camara es una posicion del Tris."""

    def __init__(
        self,
        alpha: float = 1.0,
        short_window: int = 100,
        long_window: int = 1000,
        mix_lambda: float = 0.3,
        latency_boost: float = 0.0,
        immediate_repeat_penalty: float = 0.0,
        parity_bias_strength: float = 0.0,
        topm_per_position: int | None = None,
        pmf_floor: float = 1e-6,
    ):
        self.alpha = float(alpha)
        self.short_window = int(short_window)
        self.long_window = int(long_window)
        self.mix_lambda = float(mix_lambda)
        self.latency_boost = float(latency_boost)
        self.immediate_repeat_penalty = float(immediate_repeat_penalty)
        self.parity_bias_strength = float(parity_bias_strength)
        self.topm_per_position = None if topm_per_position is None else int(topm_per_position)
        self.pmf_floor = float(pmf_floor)

        self.counts_short = np.zeros((_N_POS, _N_DIGITS), dtype=np.float64)
        self.counts_long = np.zeros((_N_POS, _N_DIGITS), dtype=np.float64)
        self.latency = np.zeros((_N_POS, _N_DIGITS), dtype=np.int32)
        self.parity_counts = np.zeros((_N_POS, 2), dtype=np.float64)
        self.parity_streak_len = np.zeros(_N_POS, dtype=np.int32)
        self.parity_streak_value = np.full(_N_POS, -1, dtype=np.int32)
        self.n_rows = 0

    def fit(self, digits_list: list[list[int]]) -> "PositionalAnalyzers":
        """Ajusta el estado usando solo el historial pasado entregado por el caller."""
        rows = _coerce_rows(digits_list)
        n_rows = int(rows.shape[0])
        self.n_rows = n_rows

        s_n = min(n_rows, max(0, self.short_window))
        l_n = min(n_rows, max(0, self.long_window))
        rows_short = rows[-s_n:] if s_n > 0 else rows[:0]
        rows_long = rows[-l_n:] if l_n > 0 else rows[:0]

        counts_short = np.zeros((_N_POS, _N_DIGITS), dtype=np.float64)
        counts_long = np.zeros((_N_POS, _N_DIGITS), dtype=np.float64)
        for pos in range(_N_POS):
            if rows_short.shape[0] > 0:
                counts_short[pos] = np.bincount(
                    rows_short[:, pos], minlength=_N_DIGITS
                ).astype(np.float64, copy=False)
            if rows_long.shape[0] > 0:
                counts_long[pos] = np.bincount(
                    rows_long[:, pos], minlength=_N_DIGITS
                ).astype(np.float64, copy=False)
        self.counts_short = counts_short
        self.counts_long = counts_long

        parity_counts = np.zeros((_N_POS, 2), dtype=np.float64)
        if rows_short.shape[0] > 0:
            parity = rows_short % 2
            for pos in range(_N_POS):
                odd_count = float(np.sum(parity[:, pos] == 1))
                even_count = float(rows_short.shape[0] - odd_count)
                parity_counts[pos, 0] = even_count
                parity_counts[pos, 1] = odd_count
        self.parity_counts = parity_counts

        latency = np.full((_N_POS, _N_DIGITS), n_rows + 1, dtype=np.int32)
        if n_rows > 0:
            for pos in range(_N_POS):
                for digit in range(_N_DIGITS):
                    idx = np.where(rows[:, pos] == digit)[0]
                    if idx.size > 0:
                        latency[pos, digit] = n_rows - 1 - int(idx[-1])
        self.latency = latency

        streak_len = np.zeros(_N_POS, dtype=np.int32)
        streak_value = np.full(_N_POS, -1, dtype=np.int32)
        if n_rows > 0:
            par_end = (rows[-1] % 2).astype(np.int32)
            for pos in range(_N_POS):
                pv = int(par_end[pos])
                streak_value[pos] = pv
                run = 0
                for i in range(n_rows - 1, -1, -1):
                    if int(rows[i, pos] % 2) == pv:
                        run += 1
                    else:
                        break
                streak_len[pos] = run
        self.parity_streak_len = streak_len
        self.parity_streak_value = streak_value
        return self

    def _smoothed_mix_pmf(self) -> np.ndarray:
        alpha = max(self.alpha, 1e-12)
        sum_short = np.sum(self.counts_short, axis=1, keepdims=True)
        sum_long = np.sum(self.counts_long, axis=1, keepdims=True)
        short_p = (self.counts_short + alpha) / np.clip(
            sum_short + alpha * _N_DIGITS, 1e-12, None
        )
        long_p = (self.counts_long + alpha) / np.clip(
            sum_long + alpha * _N_DIGITS, 1e-12, None
        )
        mix_w = float(np.clip(self.mix_lambda, 0.0, 1.0))
        return mix_w * short_p + (1.0 - mix_w) * long_p

    def _apply_latency_adjustment(self, pmf: np.ndarray) -> np.ndarray:
        if abs(self.latency_boost) <= 0.0:
            return pmf
        lat = self.latency.astype(np.float64, copy=False)
        lat_centered = lat - np.mean(lat, axis=1, keepdims=True)
        scale = np.maximum(np.std(lat_centered, axis=1, keepdims=True), 1.0)
        lat_z = lat_centered / scale
        return pmf * np.exp(self.latency_boost * lat_z)

    def _apply_immediate_repeat_penalty(
        self, pmf: np.ndarray, prev_digits: list[int] | None
    ) -> np.ndarray:
        if prev_digits is None or self.immediate_repeat_penalty <= 0.0:
            return pmf
        out = pmf.copy()
        for pos in range(_N_POS):
            if pos >= len(prev_digits):
                continue
            d = int(prev_digits[pos]) % _N_DIGITS
            out[pos, d] *= np.exp(-self.immediate_repeat_penalty)
        return out

    def _parity_local_prob(self) -> np.ndarray:
        alpha = max(self.alpha, 1e-12)
        totals = np.sum(self.parity_counts, axis=1, keepdims=True)
        return (self.parity_counts + alpha) / np.clip(totals + 2.0 * alpha, 1e-12, None)

    def _apply_parity_bias(self, pmf: np.ndarray, parity_prob: np.ndarray) -> np.ndarray:
        if abs(self.parity_bias_strength) <= 0.0:
            return pmf
        out = pmf.copy()
        even_boost = np.exp(self.parity_bias_strength * (parity_prob[:, 0] - 0.5))
        odd_boost = np.exp(self.parity_bias_strength * (parity_prob[:, 1] - 0.5))
        even_digits = np.array([0, 2, 4, 6, 8], dtype=np.int32)
        odd_digits = np.array([1, 3, 5, 7, 9], dtype=np.int32)
        out[:, even_digits] *= even_boost[:, None]
        out[:, odd_digits] *= odd_boost[:, None]
        return out

    @staticmethod
    def _entropy_rows(pmf: np.ndarray) -> np.ndarray:
        p = np.clip(pmf, 1e-12, None)
        return -np.sum(p * np.log(p), axis=1)

    @staticmethod
    def _topm_mask(pmf: np.ndarray, topm: int) -> np.ndarray:
        m = max(1, int(topm))
        mask = np.zeros_like(pmf, dtype=bool)
        for pos in range(pmf.shape[0]):
            order = np.argsort(-pmf[pos], kind="mergesort")
            mask[pos, order[:m]] = True
        return mask

    def predict(
        self,
        prev_digits: list[int] | None = None,
        slot_labels: list[str] | None = None,
        slot_context: str | None = None,
    ) -> dict:
        """Genera PMF por camara (posicion) y mascara de probabilidad por posicion."""
        pmf = self._smoothed_mix_pmf()
        pmf = self._apply_latency_adjustment(pmf)
        pmf = self._apply_immediate_repeat_penalty(pmf, prev_digits=prev_digits)
        parity_local_prob = self._parity_local_prob()
        pmf = self._apply_parity_bias(pmf, parity_prob=parity_local_prob)

        pmf = np.clip(pmf, max(self.pmf_floor, 1e-12), None)
        pmf = pmf / np.clip(np.sum(pmf, axis=1, keepdims=True), 1e-12, None)

        if self.topm_per_position is None:
            positional_mask = np.ones_like(pmf, dtype=bool)
            favored_digits_by_pos: list[list[int]] = []
            for pos in range(_N_POS):
                thr = float(np.quantile(pmf[pos], 0.8))
                idx = np.where(pmf[pos] >= thr)[0]
                idx_sorted = idx[np.argsort(-pmf[pos, idx], kind="mergesort")]
                favored_digits_by_pos.append([int(v) for v in idx_sorted.tolist()])
        else:
            positional_mask = self._topm_mask(pmf, self.topm_per_position)
            favored_digits_by_pos = [
                [int(d) for d in np.where(positional_mask[pos])[0].tolist()] for pos in range(_N_POS)
            ]

        forbidden_digits_by_pos = [
            [int(d) for d in np.where(~positional_mask[pos])[0].tolist()] for pos in range(_N_POS)
        ]
        entropy_pos = self._entropy_rows(pmf)

        diagnostics = {
            "entropy_pos": entropy_pos,
            "parity_local_prob": parity_local_prob,
            "latency": self.latency.copy(),
            "slot_labels": slot_labels,
            "slot_context": slot_context,
            "parity_streak_len": self.parity_streak_len.copy(),
            "parity_streak_value": self.parity_streak_value.copy(),
        }
        return {
            "pmf": pmf,
            "pmf_pos": pmf,
            "positional_mask": positional_mask,
            "forbidden_digits_by_pos": forbidden_digits_by_pos,
            "favored_digits_by_pos": favored_digits_by_pos,
            "diagnostics": diagnostics,
        }
