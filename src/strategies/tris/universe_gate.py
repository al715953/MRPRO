from __future__ import annotations

from typing import Sequence

import numpy as np


def _coerce_row(row: Sequence[int]) -> np.ndarray:
    if row is None or len(row) < 5:
        raise ValueError("Cada fila debe contener al menos 5 digitos.")
    out = np.asarray(row[:5], dtype=np.int16).reshape(5)
    return np.mod(out, 10).astype(np.int16, copy=False)


def _to_index(row: np.ndarray) -> int:
    d0, d1, d2, d3, d4 = [int(v) for v in row[:5]]
    return d0 * 10000 + d1 * 1000 + d2 * 100 + d3 * 10 + d4


def _coerce_history(history_digits: Sequence[Sequence[int]]) -> np.ndarray:
    rows = []
    for row in history_digits or []:
        if row is None or len(row) < 5:
            continue
        rows.append(_coerce_row(row))
    if not rows:
        return np.empty((0, 5), dtype=np.int16)
    return np.vstack(rows).astype(np.int16, copy=False)


def should_use_topk(
    history_digits,
    *,
    gate_calib_size: int = 300,
    K: int = 2000,
    alpha: float = 1.0,
    threshold_z: float = 1.0,
) -> bool:
    """
    Gate prequential sin leakage para decidir si usar universo top-K.

    Para cada t en la ventana de calibracion:
    - train usa solo historia <= t-1
    - construye top-K con conteos acumulados
    - y_t=1 si winner_t cae en ese top-K
    """
    rows = _coerce_history(history_digits)
    n_total = int(rows.shape[0])
    if n_total < 2:
        return False

    k_eff = int(max(0, min(int(K), 100000)))
    if k_eff <= 0:
        return False

    calib_size = int(max(1, min(int(gate_calib_size), n_total - 1)))
    start_t = n_total - calib_size

    counts = np.full(100000, float(max(alpha, 0.0)), dtype=np.float64)
    for i in range(start_t):
        idx = _to_index(rows[i])
        counts[idx] += 1.0

    y_vals = np.zeros(calib_size, dtype=np.float64)
    for j, t in enumerate(range(start_t, n_total)):
        if k_eff >= 100000:
            in_topk = 1.0
        else:
            top_idx = np.argpartition(counts, -k_eff)[-k_eff:]
            winner_idx = _to_index(rows[t])
            in_topk = float(np.any(top_idx == winner_idx))
        y_vals[j] = in_topk

        counts[_to_index(rows[t])] += 1.0

    p = float(k_eff / 100000.0)
    edge_vals = y_vals - p
    edge_mean = float(np.mean(edge_vals))

    n = int(edge_vals.size)
    var_mean = float(np.mean(np.full(n, p * (1.0 - p), dtype=np.float64)))
    denom = float(np.sqrt(var_mean / max(n, 1)))
    if denom <= 0.0:
        z = float("inf") if edge_mean > 0.0 else float("-inf")
    else:
        z = float(edge_mean / denom)

    return bool((z >= float(threshold_z)) and (edge_mean > 0.0))

