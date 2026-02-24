from __future__ import annotations

from typing import Dict, Tuple

import numpy as np

from src.core.prob_metrics import logloss_positional
from src.strategies.tris.v1a_model import TrisV1AModel, _extract_tris_series


def _get_override(cfg: Dict, key: str, default):
    return cfg.get(key, default) if isinstance(cfg, dict) else default


def choose_strategy(history, config) -> Tuple[str, Dict]:
    """
    Selecciona entre V1-A y baseline uniforme usando una ventana de calibración
    exclusivamente con datos pasados (sin leakage).
    """
    overrides = config.filter_overrides or {}
    gate_calib_size = int(_get_override(overrides, "gate_calib_size", 300))
    gate_margin = float(_get_override(overrides, "gate_margin", 0.0005))

    short_window = int(_get_override(overrides, "short_window", 200))
    long_window = int(_get_override(overrides, "long_window", 2000))
    alpha_bayes = float(_get_override(overrides, "alpha_bayes", 0.5))
    mix_lambda = float(_get_override(overrides, "mix_lambda", 0.7))
    markov_window = int(_get_override(overrides, "markov_window", 2000))
    alpha_markov = float(_get_override(overrides, "alpha_markov", 0.2))
    blend_markov = float(_get_override(overrides, "blend_markov", 0.35))
    uniform_mix = float(_get_override(overrides, "uniform_mix", 0.0))
    uniform_floor_mu = float(_get_override(overrides, "uniform_floor_mu", 0.35))
    peak_max_prob = float(_get_override(overrides, "peak_max_prob", 0.22))
    peak_mu_boost = float(_get_override(overrides, "peak_mu_boost", 0.20))
    temperature = float(_get_override(overrides, "temperature", 1.4))

    digits_list, mult_list = _extract_tris_series(history)
    n_draws = len(digits_list)
    uniform_logloss = float(-np.log(0.1))

    if n_draws < 2:
        report = {
            "calib_size": 0,
            "model_logloss": uniform_logloss,
            "uniform_logloss": uniform_logloss,
            "margin": gate_margin,
            "chosen": "uniform",
            "reason": "insufficient_history",
        }
        return "uniform", report

    calib_size = max(1, min(gate_calib_size, n_draws - 1))
    calib_start = n_draws - calib_size
    train_digits = digits_list[:calib_start]
    train_mult = mult_list[:calib_start]
    losses = []

    if len(train_digits) < 50:
        model = None
    else:
        model = TrisV1AModel(
            blend_markov=blend_markov,
            uniform_mix=uniform_mix,
            uniform_floor_mu=uniform_floor_mu,
            peak_max_prob=peak_max_prob,
            peak_mu_boost=peak_mu_boost,
            temperature=temperature,
            bayes_params={
                "alpha": alpha_bayes,
                "short_window": short_window,
                "long_window": long_window,
                "mix_lambda": mix_lambda,
            },
            markov_params={
                "alpha": alpha_markov,
                "window": markov_window,
            },
        )
        model.fit(train_digits, train_mult)

    for idx in range(calib_start, n_draws):
        if model is None:
            pos_probs = np.full((5, 10), 0.1, dtype=np.float64)
        else:
            context_last_digits = digits_list[idx - 1] if idx > 0 else [0, 0, 0, 0, 0]
            pos_probs, _, _, _, _ = model.predict(context_last_digits)
        y_digits = digits_list[idx][:5]
        losses.append(float(logloss_positional(pos_probs, y_digits)))

    model_logloss = float(np.mean(losses)) if losses else uniform_logloss
    chosen = (
        "v1a" if model_logloss <= (uniform_logloss - gate_margin) else "uniform"
    )
    report = {
        "calib_size": calib_size,
        "model_logloss": model_logloss,
        "uniform_logloss": uniform_logloss,
        "margin": gate_margin,
        "chosen": chosen,
    }
    return chosen, report
