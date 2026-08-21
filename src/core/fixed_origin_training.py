from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
import os
from pathlib import Path
from uuid import uuid4

import numpy as np
import xgboost as xgb

from src.core.melate_features import FEATURE_NAMES, FEATURE_SCHEMA
from src.core.melate_number_model import (
    NUMBER_FEATURE_NAMES,
    NUMBER_FEATURE_SCHEMA,
    build_number_walk_forward_dataset,
    number_topk_metrics,
)
from src.core.train_static_model import (
    MIN_CONTEXT_DRAWS,
    RANDOM_SEED,
    VALIDATION_FRACTION,
    _auc_from_labeled_scores,
    _fit_matrix,
    _fit_number_model,
    _number_slice,
    _select_number_rounds,
    _select_round_count,
    _set_model_metadata,
    _set_number_model_metadata,
    build_walk_forward_dataset,
)
from src.data_access.config import BACKTEST_MODEL_CACHE_PATH
from src.domain.dtos import DrawHistoryDTO, sort_history_chronologically


CACHE_SCHEMA_VERSION = 1
TRAINER_SIGNATURE = "melate_fixed_origin_v1"


@dataclass(frozen=True)
class FixedOriginArtifacts:
    context_model_path: str
    number_model_path: str
    manifest_path: str
    dataset_hash: str
    requested_backtest_size: int
    training_rows: int
    training_cutoff_contest: int
    test_start_contest: int
    test_end_contest: int
    internal_validation_start_contest: int
    internal_context_auc: float
    internal_number_auc: float
    context_rounds: int
    number_rounds: int
    reused_cache: bool

    def to_dict(self) -> dict:
        return asdict(self)


def _history_arrays(history: DrawHistoryDTO):
    ordered = sort_history_chronologically(history)
    if len(ordered.concursos) != len(ordered.winning_numbers):
        raise ValueError("Histórico inconsistente: concursos y resultados no coinciden")
    if any(len(draw) < 6 for draw in ordered.winning_numbers):
        raise ValueError("Cada sorteo de Melate debe contener al menos seis números")
    draws = np.sort(
        np.asarray([draw[:6] for draw in ordered.winning_numbers], dtype=np.uint8),
        axis=1,
    )
    contests = np.asarray([int(value) for value in ordered.concursos], dtype=np.int64)
    return ordered, draws, contests


def _history_hash(contests: np.ndarray, draws: np.ndarray) -> str:
    digest = hashlib.sha256()
    digest.update(TRAINER_SIGNATURE.encode("utf-8"))
    digest.update(np.asarray(contests, dtype=np.int64).tobytes())
    digest.update(np.asarray(draws, dtype=np.uint8).tobytes())
    return digest.hexdigest()


def _atomic_save_model(model: xgb.Booster, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.stem}.{uuid4().hex}.json")
    try:
        model.save_model(str(temporary))
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def _atomic_save_json(payload: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{uuid4().hex}.tmp")
    try:
        temporary.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def _cache_is_valid(
    manifest_path: Path,
    context_path: Path,
    number_path: Path,
    expected: dict,
) -> dict | None:
    if not (manifest_path.exists() and context_path.exists() and number_path.exists()):
        return None
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        for key, value in expected.items():
            if manifest.get(key) != value:
                return None
        context = xgb.Booster()
        context.load_model(str(context_path))
        number = xgb.Booster()
        number.load_model(str(number_path))
        if context.attr("trained_through_concurso") != str(
            expected["training_cutoff_contest"]
        ):
            return None
        if number.attr("trained_through_concurso") != str(
            expected["training_cutoff_contest"]
        ):
            return None
        return manifest
    except (OSError, ValueError, TypeError, xgb.core.XGBoostError):
        return None


def _artifacts_from_manifest(manifest: dict, *, reused: bool) -> FixedOriginArtifacts:
    return FixedOriginArtifacts(
        context_model_path=str(manifest["context_model_path"]),
        number_model_path=str(manifest["number_model_path"]),
        manifest_path=str(manifest["manifest_path"]),
        dataset_hash=str(manifest["dataset_hash"]),
        requested_backtest_size=int(manifest["requested_backtest_size"]),
        training_rows=int(manifest["training_rows"]),
        training_cutoff_contest=int(manifest["training_cutoff_contest"]),
        test_start_contest=int(manifest["test_start_contest"]),
        test_end_contest=int(manifest["test_end_contest"]),
        internal_validation_start_contest=int(
            manifest["internal_validation_start_contest"]
        ),
        internal_context_auc=float(manifest["internal_context_auc"]),
        internal_number_auc=float(manifest["internal_number_auc"]),
        context_rounds=int(manifest["context_rounds"]),
        number_rounds=int(manifest["number_rounds"]),
        reused_cache=bool(reused),
    )


def prepare_fixed_origin_models(
    history: DrawHistoryDTO,
    backtest_size: int,
    *,
    cache_directory: str | Path = BACKTEST_MODEL_CACHE_PATH,
    force_retrain: bool = False,
) -> FixedOriginArtifacts:
    """Train/cache models using only rows before the requested final test window."""

    _, draws, contests = _history_arrays(history)
    requested = int(backtest_size)
    if requested <= 0:
        raise ValueError("backtest_size debe ser positivo")
    if requested >= len(draws):
        raise ValueError("El backtest debe dejar sorteos anteriores para entrenamiento")
    cutoff_idx = len(draws) - requested
    minimum_training_rows = MIN_CONTEXT_DRAWS + 20
    if cutoff_idx < minimum_training_rows:
        maximum = len(draws) - minimum_training_rows
        raise ValueError(
            f"El backtest solicitado deja muy poco entrenamiento; máximo permitido: {maximum}"
        )

    eligible = cutoff_idx - MIN_CONTEXT_DRAWS
    validation_examples = max(1, int(np.floor(eligible * VALIDATION_FRACTION)))
    validation_start_idx = cutoff_idx - validation_examples
    dataset_hash = _history_hash(contests, draws)
    short_hash = dataset_hash[:12]
    cutoff_contest = int(contests[cutoff_idx - 1])
    test_start = int(contests[cutoff_idx])
    test_end = int(contests[-1])
    cache_dir = Path(cache_directory)
    stem = f"melate_fixed_cutoff_{cutoff_contest}_b{requested}_{short_hash}"
    context_path = cache_dir / f"{stem}_context.json"
    number_path = cache_dir / f"{stem}_number.json"
    manifest_path = cache_dir / f"{stem}_manifest.json"
    expected = {
        "cache_schema_version": CACHE_SCHEMA_VERSION,
        "trainer_signature": TRAINER_SIGNATURE,
        "dataset_hash": dataset_hash,
        "requested_backtest_size": requested,
        "training_rows": cutoff_idx,
        "training_cutoff_contest": cutoff_contest,
        "test_start_contest": test_start,
        "test_end_contest": test_end,
    }
    if not force_retrain:
        cached = _cache_is_valid(
            manifest_path,
            context_path,
            number_path,
            expected,
        )
        if cached is not None:
            return _artifacts_from_manifest(cached, reused=True)

    train_x, train_y = build_walk_forward_dataset(
        draws,
        start_idx=MIN_CONTEXT_DRAWS,
        end_idx=validation_start_idx,
        rng=np.random.default_rng(RANDOM_SEED),
    )
    validation_x, validation_y = build_walk_forward_dataset(
        draws,
        start_idx=validation_start_idx,
        end_idx=cutoff_idx,
        rng=np.random.default_rng(RANDOM_SEED + 1),
    )
    context_rounds = _select_round_count(
        train_x,
        train_y,
        validation_x,
        validation_y,
    )
    context_validation_model = _fit_matrix(train_x, train_y, context_rounds)
    context_validation_scores = context_validation_model.predict(
        xgb.DMatrix(validation_x, feature_names=list(FEATURE_NAMES))
    )
    context_auc = _auc_from_labeled_scores(
        context_validation_scores,
        validation_y,
    )
    full_context_x, full_context_y = build_walk_forward_dataset(
        draws,
        start_idx=MIN_CONTEXT_DRAWS,
        end_idx=cutoff_idx,
        rng=np.random.default_rng(RANDOM_SEED + 2),
    )
    context_model = _fit_matrix(full_context_x, full_context_y, context_rounds)
    _set_model_metadata(
        context_model,
        role="fixed_origin_backtest",
        trained_through=cutoff_contest,
        training_rows=cutoff_idx,
        training_examples=len(full_context_x),
        holdout_start=test_start,
        holdout_end=test_end,
    )
    context_model.set_attr(
        temporal_holdout_auc=f"{context_auc:.8f}",
        validation_start_concurso=str(int(contests[validation_start_idx])),
        validation_end_concurso=str(cutoff_contest),
        requested_backtest_size=str(requested),
        dataset_hash=dataset_hash,
        split_strategy="fixed_origin",
        trainer_signature=TRAINER_SIGNATURE,
        selected_rounds=str(context_rounds),
    )

    number_x, number_y = build_number_walk_forward_dataset(
        draws,
        start_idx=MIN_CONTEXT_DRAWS,
        end_idx=cutoff_idx,
    )
    number_train_slice = _number_slice(MIN_CONTEXT_DRAWS, validation_start_idx)
    number_validation_slice = _number_slice(validation_start_idx, cutoff_idx)
    number_rounds = _select_number_rounds(
        number_x[number_train_slice],
        number_y[number_train_slice],
        number_x[number_validation_slice],
        number_y[number_validation_slice],
    )
    number_validation_model = _fit_number_model(
        number_x[number_train_slice],
        number_y[number_train_slice],
        number_rounds,
    )
    number_validation_scores = number_validation_model.predict(
        xgb.DMatrix(
            number_x[number_validation_slice],
            feature_names=list(NUMBER_FEATURE_NAMES),
        )
    )
    number_auc = _auc_from_labeled_scores(
        number_validation_scores,
        number_y[number_validation_slice],
    )
    number_topk = number_topk_metrics(
        number_validation_scores,
        number_y[number_validation_slice],
    )
    number_model = _fit_number_model(number_x, number_y, number_rounds)
    _set_number_model_metadata(
        number_model,
        role="fixed_origin_backtest_number",
        trained_through=cutoff_contest,
        training_examples=len(number_x),
        holdout_auc=number_auc,
    )
    number_model.set_attr(
        feature_schema=NUMBER_FEATURE_SCHEMA,
        holdout_start_concurso=str(test_start),
        holdout_end_concurso=str(test_end),
        validation_start_concurso=str(int(contests[validation_start_idx])),
        validation_end_concurso=str(cutoff_contest),
        holdout_mean_hits_at_6=f"{number_topk['mean_hits_at_6']:.8f}",
        holdout_mean_hits_at_10=f"{number_topk['mean_hits_at_10']:.8f}",
        requested_backtest_size=str(requested),
        dataset_hash=dataset_hash,
        split_strategy="fixed_origin",
        trainer_signature=TRAINER_SIGNATURE,
        selected_rounds=str(number_rounds),
    )

    _atomic_save_model(context_model, context_path)
    _atomic_save_model(number_model, number_path)
    manifest = {
        **expected,
        "feature_schema": FEATURE_SCHEMA,
        "number_feature_schema": NUMBER_FEATURE_SCHEMA,
        "context_model_path": str(context_path.resolve()),
        "number_model_path": str(number_path.resolve()),
        "manifest_path": str(manifest_path.resolve()),
        "internal_validation_start_contest": int(contests[validation_start_idx]),
        "internal_validation_end_contest": cutoff_contest,
        "internal_context_auc": float(context_auc),
        "internal_number_auc": float(number_auc),
        "context_rounds": int(context_rounds),
        "number_rounds": int(number_rounds),
        "negative_sampling": "walk_forward_valid_unique",
        "random_seed": RANDOM_SEED,
        "final_test_used_for_training_or_tuning": False,
    }
    _atomic_save_json(manifest, manifest_path)
    return _artifacts_from_manifest(manifest, reused=False)
