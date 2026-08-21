from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

import numpy as np

from src.strategies.combinatorial.statistics import (
    exact_mcnemar,
    paired_block_bootstrap_ci,
    paired_sign_permutation_test,
)


STATUS_REFERENCE = "REFERENCE"
STATUS_BENCHMARK = "BENCHMARK"
STATUS_UNMATCHED = "UNMATCHED_BUDGET"
STATUS_INSUFFICIENT = "INSUFFICIENT_SAMPLE"
STATUS_COLLECTING = "COLLECTING"
STATUS_PROMISING = "PROMISING"
STATUS_ELIGIBLE = "ELIGIBLE_FOR_PILOT"
STATUS_NO_ADVANTAGE = "NO_ADVANTAGE"
STATUS_REJECTED = "REJECTED"


@dataclass(frozen=True)
class PromotionRules:
    """Precommitted rules for shadow evaluation; never changes production itself."""

    minimum_draws: int = 20
    pilot_draws: int = 50
    alpha: float = 0.05
    max_hit_rate_ge_5_drop: float = 0.01
    required_stable_windows: int = 2
    temporal_windows: int = 3
    permutation_trials: int = 10_000
    bootstrap_trials: int = 5_000
    random_seed: int = 20260819

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


DEFAULT_PROMOTION_RULES = PromotionRules()


def _variant_series(payload: dict[str, Any]) -> dict[str, dict[str, Any]]:
    series: dict[str, dict[str, Any]] = {}
    for portfolio in payload.get("portfolios", []):
        contest = int(portfolio.get("concurso", 0))
        for variant in portfolio.get("variants", []):
            validation = variant.get("validation")
            if not isinstance(validation, dict):
                continue
            key = str(variant.get("key", "unknown"))
            row = series.setdefault(
                key,
                {
                    "key": key,
                    "label": str(variant.get("label", key)),
                    "official": bool(variant.get("official", False)),
                    "settings": dict(variant.get("settings") or {}),
                    "records": {},
                },
            )
            distribution = validation.get("hit_distribution", {})
            row["records"][contest] = {
                "contest": contest,
                "ticket_count": int(validation.get("ticket_count", 0)),
                "max_hits": int(validation.get("max_hits", 0)),
                "success_ge_4": int(validation.get("max_hits", 0)) >= 4,
                "success_ge_5": int(validation.get("max_hits", 0)) >= 5,
                "success_eq_6": int(validation.get("max_hits", 0)) == 6,
                "high_hit_tickets": sum(
                    int(distribution.get(str(hit), 0)) for hit in (4, 5, 6)
                ),
            }
    return series


def _representative_budget(row: dict[str, Any]) -> int:
    counts = [
        int(record["ticket_count"]) for record in row.get("records", {}).values()
    ]
    return int(round(float(np.median(counts)))) if counts else 0


def _reference_key(
    candidate: dict[str, Any],
    series: dict[str, dict[str, Any]],
) -> str | None:
    explicit = candidate.get("settings", {}).get("promotion_reference_key")
    if explicit and str(explicit) in series:
        return str(explicit)
    budget = _representative_budget(candidate)
    benchmark_key = f"benchmark_mrpro_native_m{budget}"
    if benchmark_key in series:
        return benchmark_key
    for key, row in series.items():
        if row.get("official") and _representative_budget(row) == budget:
            return key
    return None


def _paired_arrays(candidate: dict[str, Any], reference: dict[str, Any]):
    contests = sorted(
        set(candidate.get("records", {})).intersection(reference.get("records", {}))
    )
    candidate_hits = np.asarray(
        [candidate["records"][contest]["max_hits"] for contest in contests],
        dtype=np.float64,
    )
    reference_hits = np.asarray(
        [reference["records"][contest]["max_hits"] for contest in contests],
        dtype=np.float64,
    )
    return contests, candidate_hits, reference_hits


def _stable_windows(
    candidate_hits: np.ndarray,
    reference_hits: np.ndarray,
    windows: int,
) -> dict[str, Any]:
    count = min(max(1, int(windows)), int(candidate_hits.size))
    rows = []
    stable = 0
    for idx, indices in enumerate(np.array_split(np.arange(candidate_hits.size), count)):
        delta = float(np.mean(candidate_hits[indices] - reference_hits[indices]))
        candidate_ge_4 = float(np.mean(candidate_hits[indices] >= 4))
        reference_ge_4 = float(np.mean(reference_hits[indices] >= 4))
        favorable = bool(delta >= 0.0 and candidate_ge_4 >= reference_ge_4)
        stable += int(favorable)
        rows.append(
            {
                "window": idx + 1,
                "draws": int(indices.size),
                "avg_max_hits_delta": delta,
                "hit_rate_ge_4_delta": candidate_ge_4 - reference_ge_4,
                "favorable": favorable,
            }
        )
    return {"favorable_windows": stable, "total_windows": count, "windows": rows}


def _promotion_status(
    *,
    draws: int,
    avg_delta: float,
    ge_4_delta: float,
    ge_5_delta: float,
    bootstrap: dict[str, Any],
    permutation: dict[str, Any],
    stability: dict[str, Any],
    rules: PromotionRules,
) -> tuple[str, list[str]]:
    reasons = []
    if draws < rules.minimum_draws:
        return STATUS_INSUFFICIENT, [
            f"Faltan {rules.minimum_draws - draws} sorteos pareados para la primera revisión."
        ]

    point_favorable = (
        avg_delta > 0.0
        and ge_4_delta > 0.0
        and ge_5_delta >= -rules.max_hit_rate_ge_5_drop
    )
    stable = stability["favorable_windows"] >= min(
        rules.required_stable_windows, stability["total_windows"]
    )
    ci_low = bootstrap.get("ci_low")
    ci_high = bootstrap.get("ci_high")
    p_value = permutation.get("p_two_sided")

    if draws < rules.pilot_draws:
        if point_favorable and stable:
            return STATUS_PROMISING, [
                f"Señal favorable, pero se requieren {rules.pilot_draws} sorteos para piloto."
            ]
        return STATUS_COLLECTING, [
            f"Muestra inicial completa; faltan {rules.pilot_draws - draws} sorteos para el gate de piloto."
        ]

    statistically_supported = (
        ci_low is not None
        and ci_low >= 0.0
        and p_value is not None
        and p_value <= rules.alpha
    )
    if point_favorable and stable and statistically_supported:
        reasons.append("Cumple muestra, efecto, estabilidad y evidencia pareada.")
        return STATUS_ELIGIBLE, reasons
    if ci_high is not None and ci_high < 0.0 and ge_4_delta <= 0.0:
        return STATUS_REJECTED, [
            "El intervalo pareado indica deterioro del máximo de aciertos."
        ]
    if point_favorable:
        return STATUS_PROMISING, [
            "La estimación puntual es favorable, pero la incertidumbre aún cruza cero."
        ]
    return STATUS_NO_ADVANTAGE, [
        "No cumple simultáneamente mejora promedio, ≥4 y tolerancia de ≥5."
    ]


def evaluate_shadow_promotions(
    payload: dict[str, Any],
    rules: PromotionRules = DEFAULT_PROMOTION_RULES,
) -> dict[str, Any]:
    """Evaluate shadows against an equal-budget reference using paired draws."""

    series = _variant_series(payload)
    evaluations = {}
    for key, candidate in series.items():
        budget = _representative_budget(candidate)
        if candidate.get("official"):
            evaluations[key] = {
                "key": key,
                "label": candidate["label"],
                "status": STATUS_REFERENCE,
                "ticket_budget": budget,
                "paired_draws": len(candidate["records"]),
                "reference_key": None,
                "reasons": ["Cartera oficial usada como referencia cuando coincide el presupuesto."],
            }
            continue
        if key.startswith("benchmark_mrpro_native_m"):
            evaluations[key] = {
                "key": key,
                "label": candidate["label"],
                "status": STATUS_BENCHMARK,
                "ticket_budget": budget,
                "paired_draws": len(candidate["records"]),
                "reference_key": None,
                "reasons": ["Benchmark MRPRO para challengers con el mismo presupuesto."],
            }
            continue

        reference_key = _reference_key(candidate, series)
        if reference_key is None:
            evaluations[key] = {
                "key": key,
                "label": candidate["label"],
                "status": STATUS_UNMATCHED,
                "ticket_budget": budget,
                "paired_draws": 0,
                "reference_key": None,
                "reasons": ["No existe referencia MRPRO con el mismo número de boletos."],
            }
            continue

        reference = series[reference_key]
        contests, candidate_hits, reference_hits = _paired_arrays(candidate, reference)
        draws = int(candidate_hits.size)
        candidate_ge_4 = candidate_hits >= 4
        reference_ge_4 = reference_hits >= 4
        candidate_ge_5 = candidate_hits >= 5
        reference_ge_5 = reference_hits >= 5
        avg_delta = float(np.mean(candidate_hits - reference_hits)) if draws else 0.0
        ge_4_delta = (
            float(np.mean(candidate_ge_4) - np.mean(reference_ge_4)) if draws else 0.0
        )
        ge_5_delta = (
            float(np.mean(candidate_ge_5) - np.mean(reference_ge_5)) if draws else 0.0
        )
        permutation = paired_sign_permutation_test(
            candidate_hits,
            reference_hits,
            trials=rules.permutation_trials,
            seed=rules.random_seed,
        )
        bootstrap = paired_block_bootstrap_ci(
            candidate_hits,
            reference_hits,
            trials=rules.bootstrap_trials,
            seed=rules.random_seed + 1,
        )
        stability = _stable_windows(
            candidate_hits,
            reference_hits,
            rules.temporal_windows,
        ) if draws else {"favorable_windows": 0, "total_windows": 0, "windows": []}
        status, reasons = _promotion_status(
            draws=draws,
            avg_delta=avg_delta,
            ge_4_delta=ge_4_delta,
            ge_5_delta=ge_5_delta,
            bootstrap=bootstrap,
            permutation=permutation,
            stability=stability,
            rules=rules,
        )
        evaluations[key] = {
            "key": key,
            "label": candidate["label"],
            "status": status,
            "ticket_budget": budget,
            "reference_key": reference_key,
            "reference_label": reference["label"],
            "paired_draws": draws,
            "draw_range": [contests[0], contests[-1]] if contests else None,
            "avg_max_hits_delta": avg_delta,
            "hit_rate_ge_4_delta": ge_4_delta,
            "hit_rate_ge_5_delta": ge_5_delta,
            "mcnemar_ge_4": exact_mcnemar(candidate_ge_4, reference_ge_4),
            "mcnemar_ge_5": exact_mcnemar(candidate_ge_5, reference_ge_5),
            "permutation_max_hits": permutation,
            "bootstrap_max_hits": bootstrap,
            "temporal_stability": stability,
            "reasons": reasons,
        }
    return {
        "rules": rules.to_dict(),
        "automatic_production_change": False,
        "evaluations": evaluations,
    }
