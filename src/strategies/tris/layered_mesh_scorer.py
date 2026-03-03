from __future__ import annotations

from math import comb
from typing import Any

import numpy as np

_N_POS = 5
_N_DIGITS = 10
_N_HAMMING = 6

_NULL_HAMMING_PMF = np.array(
    [comb(_N_POS, d) * (0.9**d) * (0.1 ** (_N_POS - d)) for d in range(_N_HAMMING)],
    dtype=np.float64,
)
_NULL_HAMMING_PMF = _NULL_HAMMING_PMF / np.clip(
    float(np.sum(_NULL_HAMMING_PMF)), 1e-12, None
)

_NULL_LOWCOUNT_PMF = np.array(
    [comb(_N_POS, d) * (0.5**_N_POS) for d in range(_N_HAMMING)],
    dtype=np.float64,
)
_NULL_LOWCOUNT_PMF = _NULL_LOWCOUNT_PMF / np.clip(
    float(np.sum(_NULL_LOWCOUNT_PMF)), 1e-12, None
)

# Placeholder funcional (penaliza extremos low/high) hasta meter "simetria de masa".
_DEFAULT_CROSS_EMP_PMF = np.array([0.01, 0.12, 0.37, 0.37, 0.12, 0.01], dtype=np.float64)
_DEFAULT_CROSS_EMP_PMF = _DEFAULT_CROSS_EMP_PMF / np.clip(
    float(np.sum(_DEFAULT_CROSS_EMP_PMF)), 1e-12, None
)


def _safe_prob_vector(values: Any, n: int) -> np.ndarray | None:
    try:
        arr = np.asarray(values, dtype=np.float64).reshape(-1)
    except Exception:
        return None
    if arr.size != n:
        return None
    arr = np.clip(arr, 0.0, None)
    s = float(np.sum(arr))
    if s <= 0.0 or not np.isfinite(s):
        return None
    return arr / s


def _safe_counts_to_prob(
    values: Any,
    n: int,
    *,
    alpha: float,
    min_total: int,
) -> np.ndarray | None:
    try:
        counts = np.asarray(values, dtype=np.float64).reshape(-1)
    except Exception:
        return None
    if counts.size != n:
        return None
    counts = np.clip(counts, 0.0, None)
    total = float(np.sum(counts))
    if (not np.isfinite(total)) or total < float(max(0, int(min_total))):
        return None
    a = max(float(alpha), 1e-12)
    probs = counts + a
    probs = probs / np.clip(float(np.sum(probs)), 1e-12, None)
    return probs


class LayeredMeshScorer:
    """
    Scorer por capas para Tris.

    Todos los componentes son soft y combinables por pesos.
    """

    def __init__(self, weights: dict | None = None):
        cfg = dict(weights or {})
        self.weights = {
            "positional_logp": float(cfg.get("positional_logp", 1.0)),
            "hamming_memory": float(cfg.get("hamming_memory", 0.25)),
            "cross_turbulence": float(cfg.get("cross_turbulence", 0.10)),
            "camera_repeat_penalty": float(cfg.get("camera_repeat_penalty", 0.35)),
        }

        self.repeat_penalty_per_pos = self._coerce_pos_array(
            cfg.get("camera_repeat_penalty_per_pos", np.ones(_N_POS, dtype=np.float64)),
            default=1.0,
        )
        self.hamming_min_history = int(cfg.get("hamming_min_history", 30))
        self.hamming_alpha = float(cfg.get("hamming_alpha", 1e-3))
        self.cross_min_history = int(cfg.get("cross_min_history", 30))
        self.cross_alpha = float(cfg.get("cross_alpha", 1e-3))

    @staticmethod
    def _coerce_pos_array(values: Any, default: float = 0.0) -> np.ndarray:
        try:
            arr = np.asarray(values, dtype=np.float64).reshape(-1)
        except Exception:
            arr = np.full(_N_POS, float(default), dtype=np.float64)
        if arr.size < _N_POS:
            arr = np.pad(
                arr,
                (0, _N_POS - arr.size),
                mode="constant",
                constant_values=float(default),
            )
        return arr[:_N_POS].astype(np.float64, copy=False)

    @staticmethod
    def _coerce_tickets(tickets: np.ndarray) -> np.ndarray:
        arr = np.asarray(tickets)
        if arr.ndim != 2 or arr.shape[1] < _N_POS:
            raise ValueError("tickets debe tener shape (N,5).")
        arr = np.mod(arr[:, :_N_POS].astype(np.int16, copy=False), _N_DIGITS)
        return arr.astype(np.int16, copy=False)

    @staticmethod
    def _coerce_pmf(pmf_pos: np.ndarray) -> np.ndarray:
        pmf = np.asarray(pmf_pos, dtype=np.float64)
        if pmf.shape != (_N_POS, _N_DIGITS):
            raise ValueError("pmf_pos debe tener shape (5,10).")
        pmf = np.clip(pmf, 1e-12, None)
        pmf = pmf / np.clip(np.sum(pmf, axis=1, keepdims=True), 1e-12, None)
        return pmf

    @staticmethod
    def _coerce_prev(prev_digits: list[int] | None) -> np.ndarray | None:
        if prev_digits is None:
            return None
        try:
            prev = np.asarray(prev_digits, dtype=np.int16).reshape(-1)
        except Exception:
            return None
        if prev.size < _N_POS:
            return None
        prev = np.mod(prev[:_N_POS], _N_DIGITS)
        return prev.astype(np.int16, copy=False)

    def _component_positional_logp(self, tickets: np.ndarray, pmf: np.ndarray) -> np.ndarray:
        return (
            np.log(np.clip(pmf[0, tickets[:, 0]], 1e-12, None))
            + np.log(np.clip(pmf[1, tickets[:, 1]], 1e-12, None))
            + np.log(np.clip(pmf[2, tickets[:, 2]], 1e-12, None))
            + np.log(np.clip(pmf[3, tickets[:, 3]], 1e-12, None))
            + np.log(np.clip(pmf[4, tickets[:, 4]], 1e-12, None))
        ).astype(np.float64, copy=False)

    def _component_camera_repeat_penalty(
        self,
        tickets: np.ndarray,
        prev: np.ndarray | None,
        camera_diag: dict | None,
    ) -> np.ndarray:
        n = int(tickets.shape[0])
        if prev is None:
            return np.zeros(n, dtype=np.float64)

        per_pos = self.repeat_penalty_per_pos
        if isinstance(camera_diag, dict):
            diag_pen = camera_diag.get("camera_repeat_penalty_per_pos")
            if diag_pen is None:
                diag_pen = camera_diag.get("repeat_penalty_per_pos")
            if diag_pen is not None:
                per_pos = self._coerce_pos_array(diag_pen, default=1.0)

        repeats = (tickets == prev[None, :]).astype(np.float64, copy=False)
        return -np.sum(repeats * per_pos[None, :], axis=1, dtype=np.float64)

    def _extract_hamming_empirical(self, camera_diag: dict | None) -> np.ndarray | None:
        if not isinstance(camera_diag, dict):
            return None

        for k in ("hamming_empirical_probs", "hamming_distance_probs", "hamming_probs"):
            probs = _safe_prob_vector(camera_diag.get(k), _N_HAMMING)
            if probs is not None:
                return probs

        for k in ("hamming_hist_counts", "hamming_distance_counts", "hamming_counts"):
            probs = _safe_counts_to_prob(
                camera_diag.get(k),
                _N_HAMMING,
                alpha=self.hamming_alpha,
                min_total=self.hamming_min_history,
            )
            if probs is not None:
                return probs

        return None

    def _component_hamming_memory(
        self,
        tickets: np.ndarray,
        prev: np.ndarray | None,
        camera_diag: dict | None,
    ) -> np.ndarray:
        n = int(tickets.shape[0])
        if prev is None:
            return np.zeros(n, dtype=np.float64)

        emp = self._extract_hamming_empirical(camera_diag)
        if emp is None:
            return np.zeros(n, dtype=np.float64)

        dist = np.sum(tickets != prev[None, :], axis=1, dtype=np.int16)
        dist = np.clip(dist.astype(np.int64, copy=False), 0, _N_POS)
        return np.log(np.clip(emp[dist], 1e-12, None)) - np.log(
            np.clip(_NULL_HAMMING_PMF[dist], 1e-12, None)
        )

    def _extract_cross_empirical(self, camera_diag: dict | None) -> np.ndarray:
        if not isinstance(camera_diag, dict):
            return _DEFAULT_CROSS_EMP_PMF.copy()

        for k in (
            "cross_turbulence_probs",
            "low_count_probs",
            "low_high_balance_probs",
        ):
            probs = _safe_prob_vector(camera_diag.get(k), _N_HAMMING)
            if probs is not None:
                return probs

        for k in (
            "cross_turbulence_counts",
            "low_count_counts",
            "low_high_balance_counts",
        ):
            probs = _safe_counts_to_prob(
                camera_diag.get(k),
                _N_HAMMING,
                alpha=self.cross_alpha,
                min_total=self.cross_min_history,
            )
            if probs is not None:
                return probs

        return _DEFAULT_CROSS_EMP_PMF.copy()

    def _component_cross_turbulence(
        self,
        tickets: np.ndarray,
        camera_diag: dict | None,
        slot_context: str | None,
    ) -> np.ndarray:
        _ = slot_context  # reservado para capas futuras por contexto de slot.
        low_count = np.sum(tickets <= 4, axis=1, dtype=np.int16).astype(np.int64, copy=False)
        low_count = np.clip(low_count, 0, _N_POS)
        emp = self._extract_cross_empirical(camera_diag)
        return np.log(np.clip(emp[low_count], 1e-12, None)) - np.log(
            np.clip(_NULL_LOWCOUNT_PMF[low_count], 1e-12, None)
        )

    def score_all(
        self,
        tickets: np.ndarray,
        pmf_pos: np.ndarray,
        prev_digits: list[int] | None = None,
        camera_diag: dict | None = None,
        slot_context: str | None = None,
    ) -> dict:
        """
        Returns:
          {
            "total_score": np.ndarray (N,),
            "components": {
              "positional_logp": np.ndarray,
              "hamming_memory": np.ndarray,
              "cross_turbulence": np.ndarray,
              "camera_repeat_penalty": np.ndarray,
            }
          }
        """
        t = self._coerce_tickets(tickets)
        pmf = self._coerce_pmf(pmf_pos)
        prev = self._coerce_prev(prev_digits)
        n = int(t.shape[0])
        if n == 0:
            empty = np.zeros(0, dtype=np.float64)
            return {
                "total_score": empty.copy(),
                "components": {
                    "positional_logp": empty.copy(),
                    "hamming_memory": empty.copy(),
                    "cross_turbulence": empty.copy(),
                    "camera_repeat_penalty": empty.copy(),
                },
            }

        components = {
            "positional_logp": self._component_positional_logp(t, pmf),
            "hamming_memory": self._component_hamming_memory(t, prev, camera_diag),
            "cross_turbulence": self._component_cross_turbulence(
                t, camera_diag, slot_context
            ),
            "camera_repeat_penalty": self._component_camera_repeat_penalty(
                t, prev, camera_diag
            ),
        }

        total = np.zeros(n, dtype=np.float64)
        for name, values in components.items():
            w = float(self.weights.get(name, 0.0))
            total += w * np.asarray(values, dtype=np.float64)

        return {
            "total_score": total.astype(np.float64, copy=False),
            "components": {
                k: np.asarray(v, dtype=np.float64, copy=False) for k, v in components.items()
            },
        }
