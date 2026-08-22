"""Curva reproducible de presupuesto para el selector de Melate Retro."""

from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime
import json
import os
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any
from uuid import uuid4

import numpy as np

from src.core.analytics import PerformanceTracker
from src.core.backtester import BacktestEngine
from src.core.fixed_origin_training import prepare_fixed_origin_models
from src.core.rules import MelateRetroRules
from src.data_access.config import (
    BEST_SETTINGS,
    DATA_FOLDER_PATH,
    LOTTERY_PROFILES,
    TICKET_SIZE,
    TOTAL_BALLS,
)
from src.data_access.loader import LotteryLoader
from src.domain.dtos import PredictionConfigDTO
from src.strategies.combinatorial.statistics import (
    exact_mcnemar,
    paired_block_bootstrap_ci,
    paired_sign_permutation_test,
)
from src.strategies.genetic_selector import GeneticSelectorStrategy
from src.strategies.universe_reduction import UniverseReductionStrategy


DEFAULT_BUDGETS = tuple(range(24, 37, 2))


def _atomic_save_json(payload: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{uuid4().hex}.tmp")
    try:
        temporary.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _temporal_windows(delta: np.ndarray, windows: int = 3) -> list[dict[str, Any]]:
    result = []
    for number, indices in enumerate(
        np.array_split(np.arange(delta.size), min(windows, delta.size)), start=1
    ):
        values = delta[indices]
        result.append(
            {
                "window": number,
                "draws": int(values.size),
                "mean_max_hits_delta": float(values.mean()),
                "improved_draws": int(np.sum(values > 0)),
            }
        )
    return result


def summarize_budget_curve(
    forensic_rows: list[dict[str, Any]],
    budgets: tuple[int, ...],
    *,
    ticket_cost: float,
    seed: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    if not forensic_rows:
        raise ValueError("El barrido no produjo telemetría forense")
    max_budget = max(budgets)
    hit_sequences = []
    prize_sequences = []
    prize_category_sequences = []
    draw_ids = []
    rules = MelateRetroRules()
    for row in forensic_rows:
        metrics = row.get("metrics_json")
        if not isinstance(metrics, dict):
            raise ValueError("Falta metrics_json en la telemetría")
        hits = [int(value) for value in metrics.get("selected_ticket_hits", [])]
        prizes = [
            float(value) for value in metrics.get("selected_ticket_prizes", [])
        ]
        categories = [
            str(value)
            for value in metrics.get("selected_ticket_prize_categories", [])
        ]
        if len(hits) != max_budget or len(prizes) != max_budget:
            raise ValueError(
                f"El sorteo {row.get('draw_id')} no contiene {max_budget} boletos"
            )
        if len(categories) != max_budget:
            categories = [
                rules.category_from_recorded_result(hit, prize)
                for hit, prize in zip(hits, prizes)
            ]
        hit_sequences.append(hits)
        prize_sequences.append(prizes)
        prize_category_sequences.append(categories)
        draw_ids.append(int(row["draw_id"]))

    hits_matrix = np.asarray(hit_sequences, dtype=np.int16)
    prizes_matrix = np.asarray(prize_sequences, dtype=np.float64)
    categories_matrix = np.asarray(prize_category_sequences, dtype=object)
    summaries = []
    max_hits_series: dict[int, np.ndarray] = {}
    previous = None
    for budget in budgets:
        prefix_hits = hits_matrix[:, :budget]
        prefix_prizes = prizes_matrix[:, :budget]
        prefix_categories = categories_matrix[:, :budget]
        max_hits = prefix_hits.max(axis=1)
        max_hits_series[budget] = max_hits
        ticket_distribution = Counter(int(value) for value in prefix_hits.ravel())
        investment = float(len(forensic_rows) * budget * ticket_cost)
        earnings = float(prefix_prizes.sum())
        prize_breakdown = {}
        for category in rules.PRIZE_CATEGORY_ORDER:
            category_mask = prefix_categories == category
            category_count = int(np.sum(category_mask))
            if category_count <= 0:
                continue
            prize_breakdown[category] = {
                "tickets": category_count,
                "earnings": float(prefix_prizes[category_mask].sum()),
            }
        gross_return_ratio = earnings / investment if investment else 0.0
        net_balance = earnings - investment
        summary = {
            "budget": int(budget),
            "draws": len(forensic_rows),
            "tickets_total": int(prefix_hits.size),
            "investment": investment,
            "earnings": earnings,
            "net_balance": net_balance,
            # Compatibilidad con reportes v1: `roi` era recuperación bruta.
            "roi": gross_return_ratio,
            "gross_return_ratio": gross_return_ratio,
            "net_roi": net_balance / investment if investment else 0.0,
            "prize_breakdown": prize_breakdown,
            "ticket_hit_distribution": {
                str(hit): int(ticket_distribution.get(hit, 0)) for hit in range(7)
            },
            "max_hits_distribution": {
                str(hit): int(np.sum(max_hits == hit)) for hit in range(7)
            },
            "avg_max_hits": float(max_hits.mean()),
            "draws_ge_4": int(np.sum(max_hits >= 4)),
            "draws_ge_5": int(np.sum(max_hits >= 5)),
            "draws_eq_6": int(np.sum(max_hits == 6)),
            "hit_rate_ge_4": float(np.mean(max_hits >= 4)),
            "hit_rate_ge_5": float(np.mean(max_hits >= 5)),
            "hit_rate_eq_6": float(np.mean(max_hits == 6)),
        }
        if previous is None:
            summary["marginal_vs_previous"] = None
        else:
            added = budget - int(previous["budget"])
            marginal_investment = investment - float(previous["investment"])
            marginal_earnings = earnings - float(previous["earnings"])
            marginal_prize_breakdown = {}
            added_categories = categories_matrix[:, int(previous["budget"]) : budget]
            added_prizes = prizes_matrix[:, int(previous["budget"]) : budget]
            for category in rules.PRIZE_CATEGORY_ORDER:
                category_mask = added_categories == category
                category_count = int(np.sum(category_mask))
                if category_count <= 0:
                    continue
                marginal_prize_breakdown[category] = {
                    "tickets": category_count,
                    "earnings": float(added_prizes[category_mask].sum()),
                }
            summary["marginal_vs_previous"] = {
                "tickets_added_per_draw": added,
                "investment_added": marginal_investment,
                "earnings_added": marginal_earnings,
                "net_added": marginal_earnings - marginal_investment,
                "marginal_roi": (
                    marginal_earnings / marginal_investment
                    if marginal_investment
                    else 0.0
                ),
                "prize_breakdown": marginal_prize_breakdown,
                "additional_draws_ge_4": int(
                    summary["draws_ge_4"] - previous["draws_ge_4"]
                ),
                "additional_draws_ge_5": int(
                    summary["draws_ge_5"] - previous["draws_ge_5"]
                ),
                "additional_draws_eq_6": int(
                    summary["draws_eq_6"] - previous["draws_eq_6"]
                ),
            }
        summaries.append(summary)
        previous = summary

    baseline_budget = min(budgets)
    baseline = max_hits_series[baseline_budget]
    comparisons = []
    for offset, budget in enumerate(budgets[1:], start=1):
        challenger = max_hits_series[budget]
        delta = challenger.astype(float) - baseline.astype(float)
        comparisons.append(
            {
                "reference_budget": baseline_budget,
                "challenger_budget": int(budget),
                "paired_draws": int(delta.size),
                "improved_draws": int(np.sum(delta > 0)),
                "unchanged_draws": int(np.sum(delta == 0)),
                "worsened_draws": int(np.sum(delta < 0)),
                "permutation_max_hits": paired_sign_permutation_test(
                    challenger,
                    baseline,
                    trials=10_000,
                    seed=seed + offset,
                ),
                "bootstrap_max_hits": paired_block_bootstrap_ci(
                    challenger,
                    baseline,
                    trials=5_000,
                    seed=seed + 100 + offset,
                ),
                "mcnemar_ge_4": exact_mcnemar(
                    challenger >= 4,
                    baseline >= 4,
                ),
                "mcnemar_ge_5": exact_mcnemar(
                    challenger >= 5,
                    baseline >= 5,
                ),
                "mcnemar_eq_6": exact_mcnemar(
                    challenger == 6,
                    baseline == 6,
                ),
                "temporal_windows": _temporal_windows(delta),
                "improved_draw_ids": [
                    draw_ids[index]
                    for index in np.flatnonzero(delta > 0).tolist()
                ],
            }
        )
    return summaries, comparisons


def _sweet_spot(summaries: list[dict[str, Any]]) -> dict[str, Any]:
    best_roi = max(summaries, key=lambda row: (row["roi"], -row["budget"]))
    best_ge_4_efficiency = max(
        summaries,
        key=lambda row: (row["hit_rate_ge_4"] / row["budget"], -row["budget"]),
    )
    first_ge_5 = next((row for row in summaries if row["draws_ge_5"] > 0), None)
    positive_marginal = [
        row
        for row in summaries
        if row["marginal_vs_previous"] is not None
        and row["marginal_vs_previous"]["net_added"] > 0
    ]
    return {
        "best_roi_budget": int(best_roi["budget"]),
        "best_ge_4_efficiency_budget": int(best_ge_4_efficiency["budget"]),
        "first_budget_with_ge_5": (
            int(first_ge_5["budget"]) if first_ge_5 is not None else None
        ),
        "budgets_with_positive_incremental_net": [
            int(row["budget"]) for row in positive_marginal
        ],
        "interpretation": (
            "Un punto dulce requiere una mejora repetible de premios altos o retorno "
            "marginal; aumentar boletos por sí solo no demuestra ventaja predictiva."
        ),
    }


def run_budget_sweep(
    *,
    draws: int,
    budgets: tuple[int, ...] = DEFAULT_BUDGETS,
    seed: int = 20260821,
) -> dict[str, Any]:
    budgets = tuple(sorted({int(value) for value in budgets}))
    if not budgets or min(budgets) < 5:
        raise ValueError("Los presupuestos deben contener al menos cinco boletos")

    history = LotteryLoader(LOTTERY_PROFILES["melate_retro"]).load_data()
    artifacts = prepare_fixed_origin_models(history, draws)
    settings = dict(BEST_SETTINGS)
    settings.update(
        {
            "seed": int(seed),
            "backtest_model_mode": "fixed_origin",
            "fixed_origin_training_cutoff": artifacts.training_cutoff_contest,
            "fixed_origin_test_start": artifacts.test_start_contest,
            "fixed_origin_test_end": artifacts.test_end_contest,
            "fixed_origin_dataset_hash": artifacts.dataset_hash,
        }
    )
    config = PredictionConfigDTO(
        TOTAL_BALLS,
        TICKET_SIZE,
        max(budgets),
        backtest_size=draws,
        filter_overrides=settings,
    )

    with TemporaryDirectory(prefix="mrpro_budget_sweep_") as ledger_directory:
        root = Path(ledger_directory)
        engine = BacktestEngine()
        engine.tracker = PerformanceTracker(
            log_path=root / "detailed_forensic_log.csv",
            json_path=root / "backtest_results.json",
            archive_directory=root / "forensic_log_archive",
        )
        result = engine.run(
            GeneticSelectorStrategy(
                model_path=artifacts.context_model_path,
                number_model_path=artifacts.number_model_path,
            ),
            history,
            config,
            verbose=False,
            pre_process_strategy=UniverseReductionStrategy(),
        )
        summaries, comparisons = summarize_budget_curve(
            engine.forensic_data,
            budgets,
            ticket_cost=float(engine.rules.ticket_cost),
            seed=seed,
        )

    return {
        "generated_at": datetime.now().isoformat(),
        "experiment": "melate_ticket_budget_sweep_v1",
        "budgets": list(budgets),
        "draws": int(draws),
        "draw_range": [
            int(engine.forensic_data[0]["draw_id"]),
            int(engine.forensic_data[-1]["draw_id"]),
        ],
        "seed": int(seed),
        "fixed_origin": artifacts.to_dict(),
        "maximum_budget_run": {
            "budget": max(budgets),
            "investment": float(result.investment),
            "earnings": float(result.earnings),
            "net_balance": float(result.net_balance),
            "gross_return_ratio": (
                float(result.earnings / result.investment)
                if result.investment
                else 0.0
            ),
            "net_roi": (
                float(result.net_balance / result.investment)
                if result.investment
                else 0.0
            ),
            "prize_breakdown": result.prize_breakdown,
        },
        "financial_metric_definitions": {
            "gross_return_ratio": "earnings / investment",
            "net_roi": "(earnings - investment) / investment",
            "roi": "legacy alias of gross_return_ratio",
        },
        "nested_portfolios": True,
        "ledger_isolated": True,
        "summaries": summaries,
        "paired_vs_base": comparisons,
        "sweet_spot": _sweet_spot(summaries),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--draws", type=int, default=216)
    parser.add_argument("--base", type=int, default=24)
    parser.add_argument("--maximum", type=int, default=36)
    parser.add_argument("--step", type=int, default=2)
    parser.add_argument("--seed", type=int, default=20260821)
    parser.add_argument(
        "--output",
        type=Path,
        default=DATA_FOLDER_PATH / "melate_ticket_budget_sweep_24_36.json",
    )
    args = parser.parse_args()
    if args.step <= 0 or args.maximum < args.base:
        parser.error("El rango de presupuestos es inválido")
    budgets = tuple(range(args.base, args.maximum + 1, args.step))
    if budgets[-1] != args.maximum:
        parser.error("El máximo debe ser alcanzable exactamente por el paso")
    payload = run_budget_sweep(draws=args.draws, budgets=budgets, seed=args.seed)
    _atomic_save_json(payload, args.output)
    print(f"Reporte de presupuestos guardado en {args.output}")


if __name__ == "__main__":
    main()
