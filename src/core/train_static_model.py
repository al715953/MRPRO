import os
from typing import Iterable, Optional

import numpy as np
import pandas as pd
import xgboost as xgb

from src.core.melate_features import FEATURE_NAMES, FEATURE_SCHEMA, build_candidate_features
from src.core.melate_number_model import (
    NUMBER_FEATURE_NAMES,
    NUMBER_FEATURE_SCHEMA,
    build_number_walk_forward_dataset,
    number_topk_metrics,
)
from src.data_access.config import (
    BACKTEST_NUMBER_MODEL_FILE_PATH,
    BACKTEST_MODEL_FILE_PATH,
    CSV_FILE_PATH,
    MODEL_FILE_PATH,
    NUMBER_MODEL_FILE_PATH,
)


DATA_FILE = CSV_FILE_PATH
MODEL_OUTPUT = MODEL_FILE_PATH
BACKTEST_MODEL_OUTPUT = BACKTEST_MODEL_FILE_PATH
NUMBER_MODEL_OUTPUT = NUMBER_MODEL_FILE_PATH
BACKTEST_NUMBER_MODEL_OUTPUT = BACKTEST_NUMBER_MODEL_FILE_PATH
TOTAL_BALLS = 39
NEGATIVE_RATIO = 10
MIN_CONTEXT_DRAWS = 100
HOLDOUT_FRACTION = 0.20
VALIDATION_FRACTION = 0.15
RANDOM_SEED = 42
MAX_BOOST_ROUNDS = 800


def _ticket_key(ticket: Iterable[int]) -> tuple[int, ...]:
    return tuple(sorted(int(value) for value in ticket))


def generate_valid_negative_draws(
    n_samples: int,
    *,
    rng: np.random.Generator,
    forbidden: Optional[set[tuple[int, ...]]] = None,
) -> np.ndarray:
    """Generate valid Melate tickets not present in the supplied label set."""
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

    if not rows:
        return np.empty((0, 6), dtype=np.uint8)
    return np.asarray(rows, dtype=np.uint8)


def build_walk_forward_dataset(
    draws: np.ndarray,
    *,
    start_idx: int,
    end_idx: int,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray]:
    """Create examples for draw ``t`` using exclusively draws before ``t``."""
    ordered_draws = np.sort(np.asarray(draws, dtype=np.uint8), axis=1)
    start = max(MIN_CONTEXT_DRAWS, int(start_idx))
    end = min(int(end_idx), len(ordered_draws))
    if end <= start:
        return (
            np.empty((0, len(FEATURE_NAMES)), dtype=np.float32),
            np.empty(0, dtype=np.float32),
        )

    feature_parts = []
    label_parts = []
    for target_idx in range(start, end):
        positive = ordered_draws[target_idx : target_idx + 1]
        negatives = generate_valid_negative_draws(
            NEGATIVE_RATIO,
            rng=rng,
            forbidden={_ticket_key(positive[0])},
        )
        candidates = np.vstack((positive, negatives))
        features = build_candidate_features(candidates, ordered_draws[:target_idx])
        labels = np.zeros(len(candidates), dtype=np.float32)
        labels[0] = 1.0
        feature_parts.append(features)
        label_parts.append(labels)

    all_features = np.vstack(feature_parts)
    all_labels = np.concatenate(label_parts)
    order = rng.permutation(len(all_features))
    return all_features[order], all_labels[order]


def _params() -> dict:
    return {
        "objective": "binary:logistic",
        "tree_method": "hist",
        "eval_metric": "logloss",
        "max_depth": 6,
        "min_child_weight": 8.0,
        "eta": 0.025,
        "subsample": 0.85,
        "colsample_bytree": 0.85,
        "reg_alpha": 0.20,
        "reg_lambda": 2.0,
        "seed": RANDOM_SEED,
    }


def _fit_matrix(features: np.ndarray, labels: np.ndarray, rounds: int) -> xgb.Booster:
    return xgb.train(
        _params(),
        xgb.DMatrix(features, label=labels, feature_names=list(FEATURE_NAMES)),
        num_boost_round=max(1, int(rounds)),
        verbose_eval=False,
    )


def _select_round_count(
    train_features: np.ndarray,
    train_labels: np.ndarray,
    validation_features: np.ndarray,
    validation_labels: np.ndarray,
) -> int:
    dtrain = xgb.DMatrix(
        train_features, label=train_labels, feature_names=list(FEATURE_NAMES)
    )
    dvalidation = xgb.DMatrix(
        validation_features,
        label=validation_labels,
        feature_names=list(FEATURE_NAMES),
    )
    tuning_model = xgb.train(
        _params(),
        dtrain,
        num_boost_round=MAX_BOOST_ROUNDS,
        evals=[(dvalidation, "temporal_validation")],
        early_stopping_rounds=60,
        verbose_eval=False,
    )
    best_iteration = getattr(tuning_model, "best_iteration", MAX_BOOST_ROUNDS - 1)
    return max(1, int(best_iteration) + 1)


def _temporal_auc(positive_scores: np.ndarray, negative_scores: np.ndarray) -> float:
    comparisons = positive_scores[:, None] - negative_scores[None, :]
    return float(np.mean(comparisons > 0) + 0.5 * np.mean(comparisons == 0))


def _auc_from_labeled_scores(scores: np.ndarray, labels: np.ndarray) -> float:
    positives = np.asarray(scores)[np.asarray(labels) == 1]
    negatives = np.asarray(scores)[np.asarray(labels) == 0]
    return _temporal_auc(positives, negatives)


def _number_params() -> dict:
    return {
        "objective": "binary:logistic",
        "tree_method": "hist",
        "eval_metric": "auc",
        "max_depth": 4,
        "min_child_weight": 12.0,
        "eta": 0.025,
        "subsample": 0.85,
        "colsample_bytree": 0.90,
        "reg_alpha": 0.25,
        "reg_lambda": 2.5,
        "scale_pos_weight": 33.0 / 6.0,
        "seed": RANDOM_SEED,
    }


def _number_slice(start_idx: int, end_idx: int) -> slice:
    start = (int(start_idx) - MIN_CONTEXT_DRAWS) * TOTAL_BALLS
    end = (int(end_idx) - MIN_CONTEXT_DRAWS) * TOTAL_BALLS
    return slice(start, end)


def _fit_number_model(
    features: np.ndarray,
    labels: np.ndarray,
    rounds: int,
) -> xgb.Booster:
    return xgb.train(
        _number_params(),
        xgb.DMatrix(
            features,
            label=labels,
            feature_names=list(NUMBER_FEATURE_NAMES),
        ),
        num_boost_round=max(1, int(rounds)),
        verbose_eval=False,
    )


def _select_number_rounds(
    train_x: np.ndarray,
    train_y: np.ndarray,
    validation_x: np.ndarray,
    validation_y: np.ndarray,
) -> int:
    model = xgb.train(
        _number_params(),
        xgb.DMatrix(
            train_x,
            label=train_y,
            feature_names=list(NUMBER_FEATURE_NAMES),
        ),
        num_boost_round=MAX_BOOST_ROUNDS,
        evals=[
            (
                xgb.DMatrix(
                    validation_x,
                    label=validation_y,
                    feature_names=list(NUMBER_FEATURE_NAMES),
                ),
                "temporal_validation",
            )
        ],
        early_stopping_rounds=60,
        maximize=True,
        verbose_eval=False,
    )
    return max(1, int(getattr(model, "best_iteration", 0)) + 1)


def _set_number_model_metadata(
    model: xgb.Booster,
    *,
    role: str,
    trained_through,
    training_examples: int,
    holdout_auc: float,
) -> None:
    model.set_attr(
        model_role=str(role),
        feature_schema=NUMBER_FEATURE_SCHEMA,
        feature_count=str(len(NUMBER_FEATURE_NAMES)),
        trained_through_concurso=str(int(trained_through)),
        training_examples=str(int(training_examples)),
        temporal_holdout_auc=f"{float(holdout_auc):.8f}",
        random_seed=str(RANDOM_SEED),
    )


def _print_number_walk_forward_windows(
    all_x: np.ndarray,
    all_y: np.ndarray,
    contests: np.ndarray,
    rounds: int,
) -> None:
    print("   📈 Evaluación por ventanas del modelo por número:")
    for cutoff in (1200, 1300, 1400, 1500):
        eval_end = min(cutoff + 100, len(contests))
        if cutoff <= MIN_CONTEXT_DRAWS or eval_end <= cutoff:
            continue
        train_slice = _number_slice(MIN_CONTEXT_DRAWS, cutoff)
        eval_slice = _number_slice(cutoff, eval_end)
        model = _fit_number_model(all_x[train_slice], all_y[train_slice], rounds)
        eval_scores = model.predict(
            xgb.DMatrix(
                all_x[eval_slice], feature_names=list(NUMBER_FEATURE_NAMES)
            )
        )
        auc = _auc_from_labeled_scores(eval_scores, all_y[eval_slice])
        topk = number_topk_metrics(eval_scores, all_y[eval_slice])
        print(
            f"      #{contests[cutoff - 1]} -> #{contests[eval_end - 1]} | "
            f"AUC={auc:.4f} | Top6={topk['mean_hits_at_6']:.3f}/6 | "
            f"Top10={topk['mean_hits_at_10']:.3f}/6"
        )


def _set_model_metadata(
    model: xgb.Booster,
    *,
    role: str,
    trained_through,
    training_rows: int,
    training_examples: int = 0,
    holdout_start=None,
    holdout_end=None,
) -> None:
    attrs = {
        "model_role": str(role),
        "feature_schema": FEATURE_SCHEMA,
        "feature_count": str(len(FEATURE_NAMES)),
        "trained_through_concurso": str(int(trained_through)),
        "training_rows": str(int(training_rows)),
        "training_examples": str(int(training_examples)),
        "negative_sampling": "walk_forward_valid_unique",
        "random_seed": str(RANDOM_SEED),
    }
    if holdout_start is not None:
        attrs["holdout_start_concurso"] = str(int(holdout_start))
    if holdout_end is not None:
        attrs["holdout_end_concurso"] = str(int(holdout_end))
    model.set_attr(**attrs)


def train_master_brain():
    print("🧠 INICIANDO ENTRENAMIENTO WALK-FORWARD DE MELATE RETRO...")
    print(f"   📂 Buscando datos en: {DATA_FILE}")

    if not os.path.exists(DATA_FILE):
        raise FileNotFoundError(f"No se encontró el histórico: {DATA_FILE}")
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

    if len(real_draws) < MIN_CONTEXT_DRAWS + 100:
        raise ValueError(
            f"Se requieren al menos {MIN_CONTEXT_DRAWS + 100} sorteos para "
            "entrenamiento walk-forward."
        )

    holdout_start_idx = int(np.floor(len(real_draws) * (1.0 - HOLDOUT_FRACTION)))
    holdout_start_idx = min(
        max(MIN_CONTEXT_DRAWS + 1, holdout_start_idx), len(real_draws) - 1
    )
    eligible_historical = holdout_start_idx - MIN_CONTEXT_DRAWS
    validation_examples = max(1, int(np.floor(eligible_historical * VALIDATION_FRACTION)))
    validation_start_idx = holdout_start_idx - validation_examples

    train_x, train_y = build_walk_forward_dataset(
        real_draws,
        start_idx=MIN_CONTEXT_DRAWS,
        end_idx=validation_start_idx,
        rng=np.random.default_rng(RANDOM_SEED),
    )
    validation_x, validation_y = build_walk_forward_dataset(
        real_draws,
        start_idx=validation_start_idx,
        end_idx=holdout_start_idx,
        rng=np.random.default_rng(RANDOM_SEED + 1),
    )
    best_rounds = _select_round_count(
        train_x,
        train_y,
        validation_x,
        validation_y,
    )

    print(
        "   🧪 Validación general 80/20: modelo diagnóstico hasta "
        f"#{contests[holdout_start_idx - 1]}; reserva futura "
        f"#{contests[holdout_start_idx]}-#{contests[-1]}."
    )
    print(
        f"   🧩 Esquema: {FEATURE_SCHEMA} ({len(FEATURE_NAMES)} variables) | "
        f"rondas seleccionadas: {best_rounds}."
    )

    historical_x, historical_y = build_walk_forward_dataset(
        real_draws,
        start_idx=MIN_CONTEXT_DRAWS,
        end_idx=holdout_start_idx,
        rng=np.random.default_rng(RANDOM_SEED + 2),
    )
    backtest_model = _fit_matrix(historical_x, historical_y, best_rounds)
    _set_model_metadata(
        backtest_model,
        role="temporal_backtest",
        trained_through=contests[holdout_start_idx - 1],
        training_rows=holdout_start_idx,
        training_examples=len(historical_x),
        holdout_start=contests[holdout_start_idx],
        holdout_end=contests[-1],
    )

    holdout_x, holdout_y = build_walk_forward_dataset(
        real_draws,
        start_idx=holdout_start_idx,
        end_idx=len(real_draws),
        rng=np.random.default_rng(RANDOM_SEED + 3),
    )
    holdout_scores = backtest_model.predict(
        xgb.DMatrix(holdout_x, feature_names=list(FEATURE_NAMES))
    )
    holdout_auc = _auc_from_labeled_scores(holdout_scores, holdout_y)
    backtest_model.set_attr(temporal_holdout_auc=f"{holdout_auc:.8f}")
    backtest_model.save_model(BACKTEST_MODEL_OUTPUT)
    print(
        f"   📊 AUC walk-forward fuera de muestra: {holdout_auc:.4f} "
        "(0.5000 equivale a azar)."
    )
    print(f"   ✅ Modelo diagnóstico temporal guardado en: {BACKTEST_MODEL_OUTPUT}")

    production_x, production_y = build_walk_forward_dataset(
        real_draws,
        start_idx=MIN_CONTEXT_DRAWS,
        end_idx=len(real_draws),
        rng=np.random.default_rng(RANDOM_SEED + 4),
    )
    production_model = _fit_matrix(production_x, production_y, best_rounds)
    _set_model_metadata(
        production_model,
        role="production",
        trained_through=contests[-1],
        training_rows=len(real_draws),
        training_examples=len(production_x),
    )
    production_model.set_attr(temporal_holdout_auc=f"{holdout_auc:.8f}")
    production_model.save_model(MODEL_OUTPUT)
    print(
        f"   ✅ Modelo de producción entrenado hasta #{contests[-1]} y guardado en: "
        f"{MODEL_OUTPUT}"
    )

    print("   🎯 Entrenando modelo probabilístico por número...")
    number_x, number_y = build_number_walk_forward_dataset(
        real_draws,
        start_idx=MIN_CONTEXT_DRAWS,
        end_idx=len(real_draws),
    )
    number_train_slice = _number_slice(MIN_CONTEXT_DRAWS, validation_start_idx)
    number_validation_slice = _number_slice(validation_start_idx, holdout_start_idx)
    number_rounds = _select_number_rounds(
        number_x[number_train_slice],
        number_y[number_train_slice],
        number_x[number_validation_slice],
        number_y[number_validation_slice],
    )

    number_historical_slice = _number_slice(MIN_CONTEXT_DRAWS, holdout_start_idx)
    number_holdout_slice = _number_slice(holdout_start_idx, len(real_draws))
    backtest_number_model = _fit_number_model(
        number_x[number_historical_slice],
        number_y[number_historical_slice],
        number_rounds,
    )
    number_holdout_scores = backtest_number_model.predict(
        xgb.DMatrix(
            number_x[number_holdout_slice],
            feature_names=list(NUMBER_FEATURE_NAMES),
        )
    )
    number_holdout_auc = _auc_from_labeled_scores(
        number_holdout_scores,
        number_y[number_holdout_slice],
    )
    number_topk = number_topk_metrics(
        number_holdout_scores,
        number_y[number_holdout_slice],
    )
    _set_number_model_metadata(
        backtest_number_model,
        role="temporal_backtest_number",
        trained_through=contests[holdout_start_idx - 1],
        training_examples=len(number_x[number_historical_slice]),
        holdout_auc=number_holdout_auc,
    )
    backtest_number_model.set_attr(
        holdout_start_concurso=str(int(contests[holdout_start_idx])),
        holdout_end_concurso=str(int(contests[-1])),
        holdout_mean_hits_at_6=f"{number_topk['mean_hits_at_6']:.8f}",
        holdout_mean_hits_at_10=f"{number_topk['mean_hits_at_10']:.8f}",
    )
    backtest_number_model.save_model(BACKTEST_NUMBER_MODEL_OUTPUT)

    production_number_model = _fit_number_model(number_x, number_y, number_rounds)
    _set_number_model_metadata(
        production_number_model,
        role="production_number",
        trained_through=contests[-1],
        training_examples=len(number_x),
        holdout_auc=number_holdout_auc,
    )
    production_number_model.set_attr(
        holdout_mean_hits_at_6=f"{number_topk['mean_hits_at_6']:.8f}",
        holdout_mean_hits_at_10=f"{number_topk['mean_hits_at_10']:.8f}",
    )
    production_number_model.save_model(NUMBER_MODEL_OUTPUT)

    print(
        f"   📊 Números fuera de muestra: AUC={number_holdout_auc:.4f} | "
        f"Top6={number_topk['mean_hits_at_6']:.3f}/6 | "
        f"Top10={number_topk['mean_hits_at_10']:.3f}/6 | rondas={number_rounds}."
    )
    _print_number_walk_forward_windows(
        number_x,
        number_y,
        contests,
        number_rounds,
    )
    print(f"   ✅ Modelo por número temporal: {BACKTEST_NUMBER_MODEL_OUTPUT}")
    print(f"   ✅ Modelo por número de producción: {NUMBER_MODEL_OUTPUT}")


if __name__ == "__main__":
    train_master_brain()
