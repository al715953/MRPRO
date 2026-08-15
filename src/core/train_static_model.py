import os
from typing import Iterable, Optional

import numpy as np
import pandas as pd
import xgboost as xgb

from src.data_access.config import (
    BACKTEST_MODEL_FILE_PATH,
    CSV_FILE_PATH,
    MODEL_FILE_PATH,
)


BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DATA_FILE = CSV_FILE_PATH
MODEL_OUTPUT = MODEL_FILE_PATH
BACKTEST_MODEL_OUTPUT = BACKTEST_MODEL_FILE_PATH
TOTAL_BALLS = 39
NEGATIVE_RATIO = 10
HOLDOUT_FRACTION = 0.20
VALIDATION_FRACTION = 0.15
RANDOM_SEED = 42
MAX_BOOST_ROUNDS = 1000


def _ticket_key(ticket: Iterable[int]) -> tuple[int, ...]:
    return tuple(sorted(int(value) for value in ticket))


def generate_valid_negative_draws(
    n_samples: int,
    *,
    rng: np.random.Generator,
    forbidden: Optional[set[tuple[int, ...]]] = None,
) -> np.ndarray:
    """Generate valid Melate tickets not present in the positive label set."""
    forbidden_keys = set(forbidden or ())
    generated = set()
    rows = []

    while len(rows) < int(n_samples):
        ticket = tuple(
            sorted(
                int(value)
                for value in rng.choice(
                    np.arange(1, TOTAL_BALLS + 1), size=6, replace=False
                )
            )
        )
        if ticket in forbidden_keys or ticket in generated:
            continue
        generated.add(ticket)
        rows.append(ticket)

    return np.asarray(rows, dtype=np.uint8)


def _classification_dataset(
    positive_draws: np.ndarray,
    *,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray]:
    positives = np.sort(np.asarray(positive_draws, dtype=np.uint8), axis=1)
    forbidden = {_ticket_key(row) for row in positives}
    negatives = generate_valid_negative_draws(
        len(positives) * NEGATIVE_RATIO,
        rng=rng,
        forbidden=forbidden,
    )
    features = np.vstack((positives, negatives))
    labels = np.concatenate(
        (
            np.ones(len(positives), dtype=np.float32),
            np.zeros(len(negatives), dtype=np.float32),
        )
    )
    order = rng.permutation(len(features))
    return features[order], labels[order]


def _params() -> dict:
    return {
        "objective": "binary:logistic",
        "tree_method": "hist",
        "eval_metric": "logloss",
        "max_depth": 12,
        "eta": 0.02,
        "subsample": 0.9,
        "colsample_bytree": 0.9,
        "seed": RANDOM_SEED,
    }


def _fit_booster(
    positive_draws: np.ndarray,
    *,
    rng: np.random.Generator,
    rounds: int,
) -> xgb.Booster:
    features, labels = _classification_dataset(positive_draws, rng=rng)
    return xgb.train(
        _params(),
        xgb.DMatrix(features, label=labels),
        num_boost_round=max(1, int(rounds)),
        verbose_eval=False,
    )


def _select_round_count(
    train_draws: np.ndarray,
    validation_draws: np.ndarray,
    *,
    rng: np.random.Generator,
) -> int:
    train_x, train_y = _classification_dataset(train_draws, rng=rng)
    val_x, val_y = _classification_dataset(validation_draws, rng=rng)
    tuning_model = xgb.train(
        _params(),
        xgb.DMatrix(train_x, label=train_y),
        num_boost_round=MAX_BOOST_ROUNDS,
        evals=[(xgb.DMatrix(val_x, label=val_y), "temporal_validation")],
        early_stopping_rounds=50,
        verbose_eval=False,
    )
    best_iteration = getattr(tuning_model, "best_iteration", MAX_BOOST_ROUNDS - 1)
    return max(1, int(best_iteration) + 1)


def _temporal_auc(positive_scores: np.ndarray, negative_scores: np.ndarray) -> float:
    comparisons = positive_scores[:, None] - negative_scores[None, :]
    return float(np.mean(comparisons > 0) + 0.5 * np.mean(comparisons == 0))


def _set_model_metadata(
    model: xgb.Booster,
    *,
    role: str,
    trained_through,
    training_rows: int,
    holdout_start=None,
    holdout_end=None,
) -> None:
    attrs = {
        "model_role": str(role),
        "trained_through_concurso": str(int(trained_through)),
        "training_rows": str(int(training_rows)),
        "negative_sampling": "valid_unique_without_replacement",
        "random_seed": str(RANDOM_SEED),
    }
    if holdout_start is not None:
        attrs["holdout_start_concurso"] = str(int(holdout_start))
    if holdout_end is not None:
        attrs["holdout_end_concurso"] = str(int(holdout_end))
    model.set_attr(**attrs)


def train_master_brain():
    print("🧠 INICIANDO ENTRENAMIENTO TEMPORAL DEL CEREBRO MELATE RETRO...")
    print(f"   📂 Buscando datos en: {DATA_FILE}")

    if not os.path.exists(DATA_FILE):
        fallback = os.path.join(BASE_DIR, "Melate-Retro.csv")
        if not os.path.exists(fallback):
            raise FileNotFoundError(f"No se encontró el histórico: {DATA_FILE}")
        data_path = fallback
    else:
        data_path = DATA_FILE

    df = pd.read_csv(data_path)
    required = ["CONCURSO", "F1", "F2", "F3", "F4", "F5", "F6"]
    missing = [column for column in required if column not in df.columns]
    if missing:
        raise ValueError(f"Histórico sin columnas requeridas: {', '.join(missing)}")

    df = df.copy()
    df["CONCURSO"] = pd.to_numeric(df["CONCURSO"], errors="raise").astype(int)
    df = df.sort_values("CONCURSO", kind="mergesort").reset_index(drop=True)
    real_draws = np.sort(
        df[[f"F{i}" for i in range(1, 7)]].to_numpy(dtype=np.uint8), axis=1
    )
    contests = df["CONCURSO"].to_numpy(dtype=np.int64)

    if len(real_draws) < 100:
        raise ValueError("Se requieren al menos 100 sorteos para reservar futuro temporal.")

    holdout_start_idx = int(np.floor(len(real_draws) * (1.0 - HOLDOUT_FRACTION)))
    holdout_start_idx = min(max(1, holdout_start_idx), len(real_draws) - 1)
    historical_draws = real_draws[:holdout_start_idx]
    holdout_draws = real_draws[holdout_start_idx:]

    validation_start_idx = int(
        np.floor(len(historical_draws) * (1.0 - VALIDATION_FRACTION))
    )
    validation_start_idx = min(max(1, validation_start_idx), len(historical_draws) - 1)
    tuning_train = historical_draws[:validation_start_idx]
    tuning_validation = historical_draws[validation_start_idx:]

    best_rounds = _select_round_count(
        tuning_train,
        tuning_validation,
        rng=np.random.default_rng(RANDOM_SEED),
    )
    print(
        f"   🧪 Ventana honesta: entrenamiento hasta #{contests[holdout_start_idx - 1]}, "
        f"reserva #{contests[holdout_start_idx]}-#{contests[-1]}."
    )
    print(f"   🏋️ Rondas seleccionadas sin mirar la reserva: {best_rounds}")

    backtest_model = _fit_booster(
        historical_draws,
        rng=np.random.default_rng(RANDOM_SEED + 1),
        rounds=best_rounds,
    )
    _set_model_metadata(
        backtest_model,
        role="temporal_backtest",
        trained_through=contests[holdout_start_idx - 1],
        training_rows=len(historical_draws),
        holdout_start=contests[holdout_start_idx],
        holdout_end=contests[-1],
    )

    holdout_negatives = generate_valid_negative_draws(
        len(holdout_draws) * NEGATIVE_RATIO,
        rng=np.random.default_rng(RANDOM_SEED + 2),
        forbidden={_ticket_key(row) for row in holdout_draws},
    )
    positive_scores = backtest_model.predict(xgb.DMatrix(holdout_draws))
    negative_scores = backtest_model.predict(xgb.DMatrix(holdout_negatives))
    holdout_auc = _temporal_auc(positive_scores, negative_scores)
    backtest_model.set_attr(temporal_holdout_auc=f"{holdout_auc:.8f}")
    backtest_model.save_model(BACKTEST_MODEL_OUTPUT)
    print(
        f"   📊 AUC temporal fuera de muestra: {holdout_auc:.4f} "
        "(0.5000 equivale a azar)."
    )
    print(f"   ✅ Modelo temporal guardado en: {BACKTEST_MODEL_OUTPUT}")

    production_model = _fit_booster(
        real_draws,
        rng=np.random.default_rng(RANDOM_SEED + 3),
        rounds=best_rounds,
    )
    _set_model_metadata(
        production_model,
        role="production",
        trained_through=contests[-1],
        training_rows=len(real_draws),
    )
    production_model.set_attr(temporal_holdout_auc=f"{holdout_auc:.8f}")
    production_model.save_model(MODEL_OUTPUT)
    print(
        f"   ✅ Modelo de producción entrenado hasta #{contests[-1]} y guardado en: "
        f"{MODEL_OUTPUT}"
    )
    print(
        "   ℹ️ El AUC temporal mide generalización; no demuestra que un sorteo justo "
        "sea predecible ni cambia la probabilidad matemática de cada combinación."
    )


if __name__ == "__main__":
    train_master_brain()
