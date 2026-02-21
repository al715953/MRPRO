import numpy as np


def _normalize_pos_probs(pos_probs: np.ndarray) -> np.ndarray:
    probs = np.asarray(pos_probs, dtype=np.float64)
    probs = np.clip(probs, 1e-12, None)
    row_sums = probs.sum(axis=-1, keepdims=True)
    return probs / np.clip(row_sums, 1e-12, None)


def logloss_positional(pos_probs, y_digits) -> float:
    """
    LogLoss medio por posición para 5 dígitos Tris.
    """
    probs = _normalize_pos_probs(pos_probs)
    y = [int(d) for d in y_digits[:5]]
    eps = 1e-12
    losses = []
    for i in range(5):
        p = float(probs[i, y[i]])
        losses.append(-np.log(max(eps, p)))
    return float(np.mean(losses))


def brier_positional(pos_probs, y_digits) -> float:
    """
    Brier score medio por posición para clasificación multiclase (10 dígitos).
    """
    probs = _normalize_pos_probs(pos_probs)
    y = [int(d) for d in y_digits[:5]]
    per_pos = []
    for i in range(5):
        target = np.zeros(10, dtype=np.float64)
        target[y[i]] = 1.0
        per_pos.append(float(np.sum((probs[i] - target) ** 2)))
    return float(np.mean(per_pos))


def ece_positional(pos_probs, y_digits, n_bins: int = 10) -> float:
    """
    Expected Calibration Error (ECE) promedio sobre las 5 posiciones.
    Definición estándar:
    - conf = max(p)
    - pred = argmax(p)
    - correct = 1[pred == y]
    Agrupa por bins de confianza y pondera |acc_bin - conf_bin| por fracción del bin.

    Soporta:
    - una sola muestra: pos_probs shape (5, 10), y_digits shape (5,)
    - lote de muestras: pos_probs shape (N, 5, 10), y_digits shape (N, 5)
    """
    probs = _normalize_pos_probs(pos_probs)
    y = np.asarray(y_digits, dtype=np.int64)
    n_bins = max(1, int(n_bins))

    if probs.ndim == 2:
        n_pos = probs.shape[0]
        probs_batch = probs.reshape(1, n_pos, probs.shape[1])
        if y.ndim == 1:
            y_batch = y[:n_pos].reshape(1, n_pos)
        else:
            y_batch = y.reshape(1, -1)[:, :n_pos]
    elif probs.ndim == 3:
        n_samples, n_pos, _ = probs.shape
        probs_batch = probs
        if y.ndim == 1:
            if y.shape[0] == n_pos:
                y_batch = np.tile(y.reshape(1, n_pos), (n_samples, 1))
            else:
                raise ValueError(
                    "y_digits 1D debe tener longitud igual al número de posiciones."
                )
        elif y.ndim == 2:
            if y.shape[0] != n_samples or y.shape[1] < n_pos:
                raise ValueError(
                    "y_digits 2D debe tener shape (N, posiciones) compatible con pos_probs."
                )
            y_batch = y[:, :n_pos]
        else:
            raise ValueError("y_digits debe ser 1D o 2D.")
    else:
        raise ValueError("pos_probs debe tener shape (posiciones, clases) o (N, posiciones, clases).")

    n_samples, n_pos, _ = probs_batch.shape
    if n_samples == 0:
        return 0.0

    pos_eces = []
    for i in range(n_pos):
        p_i = probs_batch[:, i, :]
        conf = np.max(p_i, axis=1)
        pred = np.argmax(p_i, axis=1)
        correct = (pred == y_batch[:, i]).astype(np.float64)

        # Map conf in [0,1] to bins [0, n_bins-1], including conf==1.0 in the last bin.
        bin_ids = np.minimum((conf * n_bins).astype(np.int64), n_bins - 1)
        ece = 0.0
        for b in range(n_bins):
            mask = bin_ids == b
            if not np.any(mask):
                continue
            acc_bin = float(np.mean(correct[mask]))
            conf_bin = float(np.mean(conf[mask]))
            weight = float(np.mean(mask))
            ece += weight * abs(acc_bin - conf_bin)
        pos_eces.append(ece)

    return float(np.mean(pos_eces))
