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


def _normalize_slot_label(slot_value) -> str:
    if slot_value is None:
        return "unknown"
    token = str(slot_value).strip().lower()
    if not token:
        return "unknown"
    token = (
        token.replace("á", "a")
        .replace("é", "e")
        .replace("í", "i")
        .replace("ó", "o")
        .replace("ú", "u")
    )
    compact = token.replace(" ", "").replace("_", "").replace("-", "")
    if "mediodia" in compact or "midday" in compact or compact == "md":
        return "mediodia"
    if (
        "clasico" in compact
        or "classic" in compact
        or "vespertino" in compact
        or "noche" in compact
        or compact == "cl"
    ):
        return "clasico"
    return "unknown"


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
        target_coverage_per_position: float | list[float] | None = None,
        adaptive_coverage_enabled: bool = False,
        adaptive_coverage_base: float = 0.70,
        adaptive_coverage_min: float = 0.55,
        adaptive_coverage_max: float = 0.90,
        adaptive_coverage_volatility_gain: float = 0.20,
        min_digits_per_position: int = 1,
        max_digits_per_position: int = 10,
        mask_mode: str = "topm",
        camera_slot_gamma: float = 0.0,
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
        self.target_coverage_per_position = self._normalize_target_coverage_config(
            target_coverage_per_position
        )
        self._target_coverage_vector = self._coerce_target_coverage_vector(
            self.target_coverage_per_position
        )
        self.adaptive_coverage_enabled = bool(adaptive_coverage_enabled)
        self.adaptive_coverage_base = float(adaptive_coverage_base)
        self.adaptive_coverage_min = float(adaptive_coverage_min)
        self.adaptive_coverage_max = float(adaptive_coverage_max)
        self.adaptive_coverage_volatility_gain = float(adaptive_coverage_volatility_gain)
        self.min_digits_per_position = int(min_digits_per_position)
        self.max_digits_per_position = int(max_digits_per_position)
        self.mask_mode = str(mask_mode or "topm").strip().lower()
        if self.mask_mode not in {"topm", "coverage"}:
            self.mask_mode = "topm"
        self.camera_slot_gamma = float(np.clip(camera_slot_gamma, 0.0, 1.0))
        self.min_digits_per_position = int(
            np.clip(self.min_digits_per_position, 1, _N_DIGITS)
        )
        self.max_digits_per_position = int(
            np.clip(self.max_digits_per_position, 1, _N_DIGITS)
        )
        if self.max_digits_per_position < self.min_digits_per_position:
            self.max_digits_per_position = self.min_digits_per_position

        self.counts_short = np.zeros((_N_POS, _N_DIGITS), dtype=np.float64)
        self.counts_long = np.zeros((_N_POS, _N_DIGITS), dtype=np.float64)
        self.latency = np.zeros((_N_POS, _N_DIGITS), dtype=np.int32)
        self.parity_counts = np.zeros((_N_POS, 2), dtype=np.float64)
        self.parity_streak_len = np.zeros(_N_POS, dtype=np.int32)
        self.parity_streak_value = np.full(_N_POS, -1, dtype=np.int32)
        self.slot_counts_short: dict[str, np.ndarray] = {}
        self.slot_counts_long: dict[str, np.ndarray] = {}
        self.slot_sample_size_short: dict[str, int] = {}
        self.slot_sample_size_long: dict[str, int] = {}
        self.n_rows = 0

    @staticmethod
    def _normalize_target_coverage_config(
        values: float | list[float] | None,
    ) -> float | list[float] | None:
        if values is None:
            return None
        if isinstance(values, (list, tuple, np.ndarray)):
            try:
                arr = np.asarray(values, dtype=np.float64).reshape(-1)
            except Exception:
                return None
            if arr.size == 0:
                return None
            arr = np.clip(arr, 0.0, 1.0)
            if arr.size == 1:
                return float(arr[0])
            if arr.size < _N_POS:
                arr = np.pad(arr, (0, _N_POS - arr.size), mode="edge")
            return [float(v) for v in arr[:_N_POS].tolist()]
        try:
            return float(np.clip(float(values), 0.0, 1.0))
        except (TypeError, ValueError):
            return None

    @staticmethod
    def _coerce_target_coverage_vector(
        values: float | list[float] | None,
    ) -> np.ndarray | None:
        if values is None:
            return None
        try:
            arr = np.asarray(values, dtype=np.float64).reshape(-1)
        except Exception:
            return None
        if arr.size == 0:
            return None
        if arr.size == 1:
            arr = np.full(_N_POS, float(arr[0]), dtype=np.float64)
        elif arr.size < _N_POS:
            arr = np.pad(arr, (0, _N_POS - arr.size), mode="edge")
        return np.clip(arr[:_N_POS], 0.0, 1.0).astype(np.float64, copy=False)

    @staticmethod
    def _count_positions(rows: np.ndarray) -> np.ndarray:
        counts = np.zeros((_N_POS, _N_DIGITS), dtype=np.float64)
        if rows.shape[0] == 0:
            return counts
        for pos in range(_N_POS):
            counts[pos] = np.bincount(rows[:, pos], minlength=_N_DIGITS).astype(
                np.float64, copy=False
            )
        return counts

    @staticmethod
    def _count_positions_by_slot(
        rows: np.ndarray, slot_labels: list[str]
    ) -> tuple[dict[str, np.ndarray], dict[str, int]]:
        counts_by_slot: dict[str, np.ndarray] = {}
        sample_size_by_slot: dict[str, int] = {}
        n = int(rows.shape[0])
        if n == 0:
            return counts_by_slot, sample_size_by_slot
        for i in range(n):
            slot = (
                _normalize_slot_label(slot_labels[i])
                if i < len(slot_labels)
                else "unknown"
            )
            if slot not in counts_by_slot:
                counts_by_slot[slot] = np.zeros((_N_POS, _N_DIGITS), dtype=np.float64)
                sample_size_by_slot[slot] = 0
            sample_size_by_slot[slot] += 1
            row = rows[i]
            for pos in range(_N_POS):
                counts_by_slot[slot][pos, int(row[pos])] += 1.0
        return counts_by_slot, sample_size_by_slot

    def fit(
        self,
        digits_list: list[list[int]],
        slot_labels: list[str] | None = None,
    ) -> "PositionalAnalyzers":
        """Ajusta el estado usando solo el historial pasado entregado por el caller."""
        rows = _coerce_rows(digits_list)
        n_rows = int(rows.shape[0])
        self.n_rows = n_rows

        s_n = min(n_rows, max(0, self.short_window))
        l_n = min(n_rows, max(0, self.long_window))
        rows_short = rows[-s_n:] if s_n > 0 else rows[:0]
        rows_long = rows[-l_n:] if l_n > 0 else rows[:0]
        labels = list(slot_labels or [])
        labels_aligned = (
            [_normalize_slot_label(v) for v in labels[-n_rows:]]
            if n_rows > 0 and labels
            else ["unknown"] * n_rows
        )
        labels_short = labels_aligned[-rows_short.shape[0] :] if rows_short.shape[0] else []
        labels_long = labels_aligned[-rows_long.shape[0] :] if rows_long.shape[0] else []

        counts_short = self._count_positions(rows_short)
        counts_long = self._count_positions(rows_long)
        self.counts_short = counts_short
        self.counts_long = counts_long
        self.slot_counts_short, self.slot_sample_size_short = self._count_positions_by_slot(
            rows_short, labels_short
        )
        self.slot_counts_long, self.slot_sample_size_long = self._count_positions_by_slot(
            rows_long, labels_long
        )

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

    def _smoothed_mix_pmf_from_counts(
        self, counts_short: np.ndarray, counts_long: np.ndarray
    ) -> np.ndarray:
        short_p = self._smoothed_pmf_from_counts(counts_short)
        long_p = self._smoothed_pmf_from_counts(counts_long)
        mix_w = float(np.clip(self.mix_lambda, 0.0, 1.0))
        return mix_w * short_p + (1.0 - mix_w) * long_p

    def _smoothed_mix_pmf(self) -> np.ndarray:
        return self._smoothed_mix_pmf_from_counts(self.counts_short, self.counts_long)

    def _smoothed_pmf_from_counts(self, counts: np.ndarray) -> np.ndarray:
        alpha = max(self.alpha, 1e-12)
        totals = np.sum(counts, axis=1, keepdims=True)
        return (counts + alpha) / np.clip(totals + alpha * _N_DIGITS, 1e-12, None)

    def _slot_conditioned_pmf(
        self, pmf_global: np.ndarray, slot_context: str | None
    ) -> tuple[np.ndarray, dict]:
        slot = _normalize_slot_label(slot_context)
        sample_short = int(self.slot_sample_size_short.get(slot, 0))
        sample_long = int(self.slot_sample_size_long.get(slot, 0))
        sample_size = sample_long
        pmf_slot = pmf_global
        gamma_eff = 0.0

        counts_short = self.slot_counts_short.get(slot)
        counts_long = self.slot_counts_long.get(slot)
        if (
            counts_short is not None
            and counts_long is not None
            and counts_short.shape == (_N_POS, _N_DIGITS)
            and counts_long.shape == (_N_POS, _N_DIGITS)
            and sample_long > 0
        ):
            pmf_slot = self._smoothed_mix_pmf_from_counts(counts_short, counts_long)
            base_gamma = float(np.clip(self.camera_slot_gamma, 0.0, 1.0))
            # Suaviza gamma con evidencia disponible para evitar sobreajuste en slots escasos.
            n_eff = float(max(0, sample_long))
            gamma_eff = base_gamma * (n_eff / (n_eff + 20.0))

        pmf_out = (1.0 - gamma_eff) * pmf_global + gamma_eff * pmf_slot
        slot_vs_global_l1_by_pos = np.sum(np.abs(pmf_slot - pmf_global), axis=1)
        diag = {
            "slot_context_used": slot,
            "slot_sample_size": int(sample_size),
            "slot_sample_size_short": int(sample_short),
            "slot_sample_size_long": int(sample_long),
            "slot_blend_gamma": float(gamma_eff),
            "slot_vs_global_l1_by_pos": slot_vs_global_l1_by_pos.astype(np.float64),
        }
        return pmf_out, diag

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

    def _compute_volatility_per_position(self) -> np.ndarray:
        if int(self.n_rows) <= 0:
            return np.zeros(_N_POS, dtype=np.float64)

        short_totals = np.sum(self.counts_short, axis=1)
        long_totals = np.sum(self.counts_long, axis=1)
        short_p = self._smoothed_pmf_from_counts(self.counts_short)
        long_p = self._smoothed_pmf_from_counts(self.counts_long)

        entropy_norm = self._entropy_rows(short_p) / np.log(float(_N_DIGITS))
        entropy_norm = np.clip(entropy_norm, 0.0, 1.0)
        entropy_norm = np.where(short_totals > 0.0, entropy_norm, 0.0)

        shift = 0.5 * np.sum(np.abs(short_p - long_p), axis=1)
        shift = np.clip(shift, 0.0, 1.0)
        shift = np.where((short_totals > 0.0) & (long_totals > 0.0), shift, 0.0)

        return np.clip(0.5 * (entropy_norm + shift), 0.0, 1.0).astype(
            np.float64, copy=False
        )

    def _resolve_target_coverage_per_position(
        self, volatility_pos: np.ndarray
    ) -> np.ndarray:
        configured = self._target_coverage_vector
        if configured is None:
            configured = np.ones(_N_POS, dtype=np.float64)
        if not self.adaptive_coverage_enabled:
            return np.clip(configured, 0.0, 1.0).astype(np.float64, copy=False)

        lower = min(self.adaptive_coverage_min, self.adaptive_coverage_max)
        upper = max(self.adaptive_coverage_min, self.adaptive_coverage_max)
        derived = self.adaptive_coverage_base + (
            self.adaptive_coverage_volatility_gain * np.asarray(volatility_pos, dtype=np.float64)
        )
        return np.clip(derived, lower, upper).astype(np.float64, copy=False)

    def _coverage_mask(self, pmf: np.ndarray, target_cov_by_pos: np.ndarray) -> np.ndarray:
        min_d = int(np.clip(self.min_digits_per_position, 1, _N_DIGITS))
        max_d = int(np.clip(self.max_digits_per_position, 1, _N_DIGITS))
        if max_d < min_d:
            max_d = min_d

        digits = np.arange(_N_DIGITS, dtype=np.int32)
        mask = np.zeros_like(pmf, dtype=bool)
        for pos in range(pmf.shape[0]):
            target_cov = float(np.clip(target_cov_by_pos[pos], 0.0, 1.0))
            # Deterministic tie-break: pmf desc, digit asc.
            order = np.lexsort((digits, -pmf[pos]))
            chosen = np.zeros(_N_DIGITS, dtype=bool)
            cum = 0.0
            count = 0
            for d in order.tolist():
                if count >= max_d:
                    break
                chosen[d] = True
                cum += float(pmf[pos, d])
                count += 1
                if count >= min_d and cum >= target_cov:
                    break
            if count < min_d:
                for d in order.tolist():
                    if count >= min_d:
                        break
                    if not chosen[d]:
                        chosen[d] = True
                        count += 1
            mask[pos] = chosen
        return mask

    def predict(
        self,
        prev_digits: list[int] | None = None,
        slot_labels: list[str] | None = None,
        slot_context: str | None = None,
    ) -> dict:
        """Genera PMF por camara (posicion) y mascara de probabilidad por posicion."""
        pmf_global = self._smoothed_mix_pmf()
        pmf, slot_diag = self._slot_conditioned_pmf(pmf_global, slot_context)
        pmf = self._apply_latency_adjustment(pmf)
        pmf = self._apply_immediate_repeat_penalty(pmf, prev_digits=prev_digits)
        parity_local_prob = self._parity_local_prob()
        pmf = self._apply_parity_bias(pmf, parity_prob=parity_local_prob)

        pmf = np.clip(pmf, max(self.pmf_floor, 1e-12), None)
        pmf = pmf / np.clip(np.sum(pmf, axis=1, keepdims=True), 1e-12, None)
        volatility_pos = self._compute_volatility_per_position()
        target_coverage_effective = self._resolve_target_coverage_per_position(
            volatility_pos
        )

        if self.mask_mode == "coverage":
            positional_mask = self._coverage_mask(pmf, target_coverage_effective)
            favored_digits_by_pos = [
                [int(d) for d in np.where(positional_mask[pos])[0].tolist()]
                for pos in range(_N_POS)
            ]
        else:
            if self.topm_per_position is None:
                positional_mask = np.ones_like(pmf, dtype=bool)
                favored_digits_by_pos = []
                for pos in range(_N_POS):
                    thr = float(np.quantile(pmf[pos], 0.8))
                    idx = np.where(pmf[pos] >= thr)[0]
                    idx_sorted = idx[np.argsort(-pmf[pos, idx], kind="mergesort")]
                    favored_digits_by_pos.append([int(v) for v in idx_sorted.tolist()])
            else:
                positional_mask = self._topm_mask(pmf, self.topm_per_position)
                favored_digits_by_pos = [
                    [int(d) for d in np.where(positional_mask[pos])[0].tolist()]
                    for pos in range(_N_POS)
                ]

        forbidden_digits_by_pos = [
            [int(d) for d in np.where(~positional_mask[pos])[0].tolist()] for pos in range(_N_POS)
        ]
        entropy_pos = self._entropy_rows(pmf)
        mask_digits_per_pos = np.sum(positional_mask.astype(np.int32), axis=1).astype(np.int32)
        mask_coverage_empirical_per_pos = np.sum(
            pmf * positional_mask.astype(np.float64), axis=1
        )

        diagnostics = {
            "entropy_pos": entropy_pos,
            "parity_local_prob": parity_local_prob,
            "latency": self.latency.copy(),
            "slot_labels": slot_labels,
            "slot_context": slot_context,
            "slot_context_used": slot_diag["slot_context_used"],
            "slot_sample_size": int(slot_diag["slot_sample_size"]),
            "slot_sample_size_short": int(slot_diag["slot_sample_size_short"]),
            "slot_sample_size_long": int(slot_diag["slot_sample_size_long"]),
            "slot_blend_gamma": float(slot_diag["slot_blend_gamma"]),
            "slot_vs_global_l1_by_pos": slot_diag["slot_vs_global_l1_by_pos"].copy(),
            "parity_streak_len": self.parity_streak_len.copy(),
            "parity_streak_value": self.parity_streak_value.copy(),
            "volatility_pos": volatility_pos.copy(),
            "target_coverage_per_pos_effective": target_coverage_effective.copy(),
            "mask_digits_per_pos": mask_digits_per_pos.copy(),
            "mask_coverage_empirical_per_pos": mask_coverage_empirical_per_pos.copy(),
            "positional_mask_mode": str(self.mask_mode),
            "target_coverage_per_position": self.target_coverage_per_position,
        }
        return {
            "pmf": pmf,
            "pmf_pos": pmf,
            "positional_mask": positional_mask,
            "forbidden_digits_by_pos": forbidden_digits_by_pos,
            "favored_digits_by_pos": favored_digits_by_pos,
            "diagnostics": diagnostics,
        }
