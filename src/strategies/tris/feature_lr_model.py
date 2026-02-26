from __future__ import annotations

from typing import Sequence

import numpy as np

_N_SUM = 46
_N_EVEN = 6
_N_UNIQ = 5
_N_CONSEC = 2
_N_MIRROR = 6

_N_BINS_STATIC = _N_SUM * _N_EVEN * _N_UNIQ * _N_CONSEC
_N_BINS_FULL = _N_BINS_STATIC * _N_MIRROR

# Cache global para no recalcular static_code_all en cada llamada.
_STATIC_CODE_ALL_CACHE: dict[tuple[int, int, int, int, int], tuple[np.ndarray, int]] = {}


def _coerce_digit(v) -> int:
    try:
        d = int(float(v))
    except Exception:
        d = 0
    return d % 10


def _coerce_rows(digits_list: Sequence[Sequence[int]]) -> np.ndarray:
    rows = []
    for row in digits_list or []:
        if row is None or len(row) < 5:
            continue
        rows.append([_coerce_digit(row[i]) for i in range(5)])
    if not rows:
        return np.empty((0, 5), dtype=np.int16)
    return np.asarray(rows, dtype=np.int16)


def _has_consecutive_run_ge4(rows: np.ndarray) -> np.ndarray:
    if rows.shape[0] == 0:
        return np.zeros(0, dtype=np.uint8)
    diffs = np.diff(rows.astype(np.int16, copy=False), axis=1)
    run_plus = ((diffs[:, 0] == 1) & (diffs[:, 1] == 1) & (diffs[:, 2] == 1)) | (
        (diffs[:, 1] == 1) & (diffs[:, 2] == 1) & (diffs[:, 3] == 1)
    )
    run_minus = ((diffs[:, 0] == -1) & (diffs[:, 1] == -1) & (diffs[:, 2] == -1)) | (
        (diffs[:, 1] == -1) & (diffs[:, 2] == -1) & (diffs[:, 3] == -1)
    )
    return (run_plus | run_minus).astype(np.uint8, copy=False)


def _unique_count(rows: np.ndarray) -> np.ndarray:
    if rows.shape[0] == 0:
        return np.zeros(0, dtype=np.int16)
    out = np.empty(rows.shape[0], dtype=np.int16)
    for i in range(rows.shape[0]):
        out[i] = int(len(set(int(v) for v in rows[i].tolist())))
    return out


def _static_code_from_features(
    sum_digits: np.ndarray,
    even_count: np.ndarray,
    unique_count: np.ndarray,
    consecutive_run_ge4: np.ndarray,
) -> np.ndarray:
    s = np.clip(np.asarray(sum_digits, dtype=np.int16), 0, 45).astype(np.int32, copy=False)
    e = np.clip(np.asarray(even_count, dtype=np.int16), 0, 5).astype(np.int32, copy=False)
    uniq_idx = np.clip(np.asarray(unique_count, dtype=np.int16) - 1, 0, 4).astype(
        np.int32, copy=False
    )
    c = np.asarray(consecutive_run_ge4, dtype=np.uint8).astype(np.int32, copy=False)
    return s + 46 * e + 46 * 6 * uniq_idx + 46 * 6 * 5 * c


def _counts_from_codes(codes: np.ndarray, n_bins: int) -> np.ndarray:
    if codes.size == 0:
        return np.zeros(n_bins, dtype=np.float64)
    return np.bincount(codes.astype(np.int32, copy=False), minlength=n_bins).astype(
        np.float64, copy=False
    )


def _smoothed_prob(counts: np.ndarray, alpha: float, n_bins: int) -> np.ndarray:
    a = max(float(alpha), 1e-12)
    numer = np.asarray(counts, dtype=np.float64) + a
    denom = float(np.sum(counts) + a * n_bins)
    return numer / max(denom, 1e-12)


class FeatureLRModel:
    def __init__(
        self,
        alpha: float = 1.0,
        short_window: int = 200,
        long_window: int = 2000,
        mix_lambda: float = 0.7,
        use_mirror: bool = True,
        shrink_c: float = 3000.0,
    ):
        self.alpha = float(alpha)
        self.short_window = int(short_window)
        self.long_window = int(long_window)
        self.mix_lambda = float(mix_lambda)
        self.use_mirror = bool(use_mirror)
        self.shrink_c = float(shrink_c)

        self.n_bins_train = _N_BINS_FULL if self.use_mirror else _N_BINS_STATIC
        self.real_prob = np.full(self.n_bins_train, 1.0 / self.n_bins_train, dtype=np.float64)
        self.n_eff = 0.0

    def _codes_for_history(self, rows: np.ndarray, use_mirror: bool) -> tuple[np.ndarray, int]:
        if rows.shape[0] == 0:
            n_bins = _N_BINS_FULL if use_mirror else _N_BINS_STATIC
            return np.zeros(0, dtype=np.int32), n_bins

        sum_digits = np.sum(rows, axis=1, dtype=np.int16)
        even_count = np.sum((rows % 2) == 0, axis=1, dtype=np.int16)
        uniq = _unique_count(rows)
        consec = _has_consecutive_run_ge4(rows)
        static_code = _static_code_from_features(sum_digits, even_count, uniq, consec)

        if not use_mirror:
            return static_code.astype(np.int32, copy=False), _N_BINS_STATIC

        mirror_count = np.zeros(rows.shape[0], dtype=np.int16)
        if rows.shape[0] > 1:
            mirror_count[1:] = np.sum(rows[1:] == rows[:-1], axis=1, dtype=np.int16)
        mirror_count = np.clip(mirror_count, 0, 5).astype(np.int32, copy=False)
        code = static_code + _N_BINS_STATIC * mirror_count
        return code.astype(np.int32, copy=False), _N_BINS_FULL

    def fit(self, digits_list: list[list[int]]) -> "FeatureLRModel":
        rows = _coerce_rows(digits_list)
        use_mirror_train = self.use_mirror
        codes_all, n_bins = self._codes_for_history(rows, use_mirror=use_mirror_train)

        if codes_all.size == 0:
            self.n_bins_train = n_bins
            self.real_prob = np.full(n_bins, 1.0 / n_bins, dtype=np.float64)
            return self

        s_n = min(codes_all.shape[0], max(0, self.short_window))
        l_n = min(codes_all.shape[0], max(0, self.long_window))
        codes_short = codes_all[-s_n:] if s_n > 0 else codes_all[:0]
        codes_long = codes_all[-l_n:] if l_n > 0 else codes_all[:0]

        counts_short = _counts_from_codes(codes_short, n_bins=n_bins)
        counts_long = _counts_from_codes(codes_long, n_bins=n_bins)

        short_prob = _smoothed_prob(counts_short, self.alpha, n_bins)
        long_prob = _smoothed_prob(counts_long, self.alpha, n_bins)
        mix_w = min(max(self.mix_lambda, 0.0), 1.0)
        real_prob_raw = mix_w * short_prob + (1.0 - mix_w) * long_prob
        real_prob_raw = real_prob_raw / max(float(np.sum(real_prob_raw)), 1e-12)
        n_eff = mix_w * float(s_n) + (1.0 - mix_w) * float(l_n)

        self.n_bins_train = n_bins
        self.real_prob = real_prob_raw.astype(np.float64, copy=False)
        self.n_eff = max(0.0, float(n_eff))
        return self

    @staticmethod
    def _static_code_all_from_cache(
        all_tickets: np.ndarray, features_cache: dict[str, np.ndarray]
    ) -> tuple[np.ndarray, int]:
        sum_digits = np.asarray(features_cache["sum_digits"])
        even_count = np.asarray(features_cache["even_count"])
        unique_count = np.asarray(features_cache["unique_count"])
        consecutive_run_ge4 = np.asarray(features_cache["consecutive_run_ge4"], dtype=np.uint8)

        key = (
            int(id(all_tickets)),
            int(id(sum_digits)),
            int(id(even_count)),
            int(id(unique_count)),
            int(id(consecutive_run_ge4)),
        )
        cached = _STATIC_CODE_ALL_CACHE.get(key)
        if cached is not None:
            return cached

        static_code = _static_code_from_features(
            sum_digits=sum_digits,
            even_count=even_count,
            unique_count=unique_count,
            consecutive_run_ge4=consecutive_run_ge4,
        ).astype(np.int32, copy=False)
        _STATIC_CODE_ALL_CACHE[key] = (static_code, _N_BINS_STATIC)
        return _STATIC_CODE_ALL_CACHE[key]

    def _adapt_real_prob(self, score_use_mirror: bool) -> np.ndarray:
        real = np.asarray(self.real_prob, dtype=np.float64)
        trained_use_mirror = bool(self.n_bins_train == _N_BINS_FULL)

        if trained_use_mirror == score_use_mirror:
            out = real
        elif (not trained_use_mirror) and score_use_mirror:
            out = np.tile(real, _N_MIRROR).astype(np.float64, copy=False)
            out = out / float(_N_MIRROR)
        else:
            out = real.reshape(_N_MIRROR, _N_BINS_STATIC).sum(axis=0)

        total = float(np.sum(out))
        if total <= 0.0:
            n_bins = _N_BINS_FULL if score_use_mirror else _N_BINS_STATIC
            return np.full(n_bins, 1.0 / n_bins, dtype=np.float64)
        return out / total

    def score_all(
        self,
        all_tickets: np.ndarray,
        features_cache: dict[str, np.ndarray],
        prev_digits: list[int] | None,
    ) -> np.ndarray:
        tickets = np.asarray(all_tickets)
        if tickets.ndim != 2 or tickets.shape[1] < 5:
            raise ValueError("all_tickets debe tener shape (N, 5).")
        tickets = np.mod(tickets[:, :5].astype(np.int16, copy=False), 10)

        static_code_all, n_bins_static = self._static_code_all_from_cache(tickets, features_cache)

        score_use_mirror = bool(self.use_mirror and prev_digits is not None)
        if score_use_mirror:
            prev = np.asarray(prev_digits, dtype=np.int16).reshape(-1)[:5]
            if prev.shape[0] < 5:
                prev = np.pad(prev, (0, 5 - prev.shape[0]), mode="constant")
            mirror_count = np.sum(
                tickets.astype(np.int16, copy=False) == prev[None, :],
                axis=1,
                dtype=np.int16,
            )
            mirror_count = np.clip(mirror_count, 0, 5).astype(np.int32, copy=False)
            code = static_code_all + n_bins_static * mirror_count
            n_bins = _N_BINS_FULL
        else:
            code = static_code_all
            n_bins = _N_BINS_STATIC

        unif_counts = np.bincount(code.astype(np.int32, copy=False), minlength=n_bins).astype(
            np.float64, copy=False
        )
        unif_prob = _smoothed_prob(unif_counts, self.alpha, n_bins)
        real_prob_raw = self._adapt_real_prob(score_use_mirror)
        n_eff = max(0.0, float(getattr(self, "n_eff", 0.0)))
        shrink_c = max(0.0, float(getattr(self, "shrink_c", 3000.0)))
        if shrink_c <= 0.0:
            lam = 1.0
        else:
            lam = n_eff / max(n_eff + shrink_c, 1e-12)
        lam = min(max(float(lam), 0.0), 1.0)
        real_prob = lam * real_prob_raw + (1.0 - lam) * unif_prob

        score = np.log(np.clip(real_prob[code], 1e-12, None)) - np.log(
            np.clip(unif_prob[code], 1e-12, None)
        )
        return score.astype(np.float64, copy=False)
