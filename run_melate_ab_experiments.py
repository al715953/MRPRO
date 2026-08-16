"""Ejecuta experimentos A/B reproducibles del scorer de Melate Retro."""

from __future__ import annotations

import argparse
from datetime import datetime
import json
from pathlib import Path
from typing import Any

import numpy as np

from src.core.backtester import BacktestEngine
from src.data_access.config import (
    BACKTEST_MODEL_FILE_PATH,
    BEST_SETTINGS,
    DATA_FOLDER_PATH,
    LOTTERY_PROFILES,
    TICKET_SIZE,
    TOTAL_BALLS,
)
from src.data_access.loader import LotteryLoader
from src.domain.dtos import PredictionConfigDTO
from src.strategies.genetic_selector import GeneticSelectorStrategy
from src.strategies.universe_reduction import UniverseReductionStrategy


CORE_VARIANTS = (
    {
        "name": "A_contextual_geo_adaptive",
        "description": "IA contextual + Geo con safety net adaptativo actual",
        "overrides": {
            "ai_context_weight": 1.0,
            "ai_number_weight": 0.0,
            "resonance_blend_mode": "adaptive",
        },
    },
    {
        "name": "B_geo_only",
        "description": "Control Geo puro; la IA no participa en radar ni ranking",
        "overrides": {
            "ai_context_weight": 1.0,
            "ai_number_weight": 0.0,
            "resonance_blend_mode": "fixed",
            "hybrid_alpha": 0.0,
            "hybrid_beta": 1.0,
        },
    },
    {
        "name": "C_contextual_number15_geo_adaptive",
        "description": "IA contextual 85% + IA por número 15%, combinada con Geo",
        "overrides": {
            "ai_context_weight": 0.85,
            "ai_number_weight": 0.15,
            "resonance_blend_mode": "adaptive",
        },
    },
)

BLEND_SWEEP_VARIANTS = (
    CORE_VARIANTS[0],
    CORE_VARIANTS[1],
    {
        "name": "D_ai10_geo90_fixed",
        "description": "IA contextual 10% + Geo 90% con mezcla fija",
        "overrides": {
            "ai_context_weight": 1.0,
            "ai_number_weight": 0.0,
            "resonance_blend_mode": "fixed",
            "hybrid_alpha": 0.10,
            "hybrid_beta": 0.90,
        },
    },
    {
        "name": "E_ai25_geo75_fixed",
        "description": "IA contextual 25% + Geo 75% con mezcla fija",
        "overrides": {
            "ai_context_weight": 1.0,
            "ai_number_weight": 0.0,
            "resonance_blend_mode": "fixed",
            "hybrid_alpha": 0.25,
            "hybrid_beta": 0.75,
        },
    },
    {
        "name": "F_ai40_geo60_fixed",
        "description": "IA contextual 40% + Geo 60% con mezcla fija",
        "overrides": {
            "ai_context_weight": 1.0,
            "ai_number_weight": 0.0,
            "resonance_blend_mode": "fixed",
            "hybrid_alpha": 0.40,
            "hybrid_beta": 0.60,
        },
    },
)

VARIANT_SUITES = {
    "core": CORE_VARIANTS,
    "blend-sweep": BLEND_SWEEP_VARIANTS,
}


def _plain_number(value: Any) -> Any:
    if isinstance(value, (np.integer, np.floating)):
        return value.item()
    return value


def _summarize(result, forensic_rows: list[dict[str, Any]]) -> dict[str, Any]:
    ranks = np.asarray([row.get("rank", 0) for row in forensic_rows], dtype=float)
    proximity = np.asarray(
        [row.get("proximity", 999) for row in forensic_rows], dtype=float
    )
    ai_scores = np.asarray(
        [row.get("ai_score", 0.0) for row in forensic_rows], dtype=float
    )
    geo_scores = np.asarray(
        [row.get("geo_score", 0.0) for row in forensic_rows], dtype=float
    )
    jackpots = [row for row in forensic_rows if int(row.get("hits", 0)) == 6]
    hit_distribution = {
        str(key): int(value) for key, value in result.hit_distribution.items()
    }
    return {
        "draws": int(result.total_draws_tested),
        "draw_ids": [int(row["draw_id"]) for row in forensic_rows],
        "event_id": forensic_rows[0].get("event_id", "") if forensic_rows else "",
        "investment": float(result.investment),
        "earnings": float(result.earnings),
        "net_balance": float(result.net_balance),
        "ticket_hit_distribution": hit_distribution,
        "selected_hits_4": hit_distribution.get("4", 0),
        "selected_hits_5": hit_distribution.get("5", 0),
        "selected_jackpots": hit_distribution.get("6", 0),
        "universe_jackpots": len(jackpots),
        "universe_hits_5": sum(
            int(row.get("hits", 0)) == 5 for row in forensic_rows
        ),
        "universe_hits_4": sum(
            int(row.get("hits", 0)) == 4 for row in forensic_rows
        ),
        "rank_median": float(np.median(ranks)) if ranks.size else None,
        "rank_mean": float(np.mean(ranks)) if ranks.size else None,
        "proximity_median": float(np.median(proximity)) if proximity.size else None,
        "proximity_mean": float(np.mean(proximity)) if proximity.size else None,
        "proximity_zero": int(np.sum(proximity == 0)),
        "proximity_le_50": int(np.sum(proximity <= 50)),
        "ai_score_mean": float(np.mean(ai_scores)) if ai_scores.size else None,
        "geo_score_mean": float(np.mean(geo_scores)) if geo_scores.size else None,
        "jackpot_details": [
            {
                "draw_id": int(row["draw_id"]),
                "rank": int(row["rank"]),
                "proximity": int(row["proximity"]),
                "ai_score": float(row.get("ai_score", 0.0)),
                "geo_score": float(row.get("geo_score", 0.0)),
            }
            for row in jackpots
        ],
    }


def run_experiments(
    backtest_size: int,
    tickets: int,
    seed: int,
    variants=CORE_VARIANTS,
    experiment_name: str = "melate_resonance_ab_v1",
) -> dict[str, Any]:
    profile = LOTTERY_PROFILES["melate_retro"]
    history = LotteryLoader(profile).load_data()
    results: list[dict[str, Any]] = []
    reference_draw_ids: list[int] | None = None

    for variant in variants:
        overrides = dict(BEST_SETTINGS)
        overrides.update(variant["overrides"])
        overrides["seed"] = int(seed)

        config = PredictionConfigDTO(
            TOTAL_BALLS,
            TICKET_SIZE,
            tickets,
            backtest_size=backtest_size,
            filter_overrides=overrides,
        )
        engine = BacktestEngine()
        result = engine.run(
            GeneticSelectorStrategy(model_path=BACKTEST_MODEL_FILE_PATH),
            history,
            config,
            verbose=False,
            pre_process_strategy=UniverseReductionStrategy(),
        )
        summary = _summarize(result, engine.forensic_data)
        summary["name"] = variant["name"]
        summary["description"] = variant["description"]
        summary["overrides"] = variant["overrides"]

        draw_ids = summary.pop("draw_ids")
        if reference_draw_ids is None:
            reference_draw_ids = draw_ids
        elif draw_ids != reference_draw_ids:
            raise RuntimeError(
                f"Ventanas distintas: {variant['name']} no evaluó los mismos sorteos"
            )
        results.append(summary)

    return {
        "generated_at": datetime.now().isoformat(),
        "experiment": experiment_name,
        "requested_backtest_size": int(backtest_size),
        "tickets_per_draw": int(tickets),
        "seed": int(seed),
        "evaluated_draw_ids": reference_draw_ids or [],
        "variants": results,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--draws", type=int, default=108)
    parser.add_argument("--tickets", type=int, default=24)
    parser.add_argument("--seed", type=int, default=20260816)
    parser.add_argument(
        "--suite",
        choices=tuple(VARIANT_SUITES),
        default="core",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
    )
    args = parser.parse_args()

    output = args.output or Path(
        DATA_FOLDER_PATH / "melate_blend_sweep.json"
        if args.suite == "blend-sweep"
        else DATA_FOLDER_PATH / "melate_ab_experiment.json"
    )
    payload = run_experiments(
        args.draws,
        args.tickets,
        args.seed,
        variants=VARIANT_SUITES[args.suite],
        experiment_name=f"melate_resonance_{args.suite}_v1",
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, default=_plain_number),
        encoding="utf-8",
    )
    print(f"Reporte A/B guardado en {output}")


if __name__ == "__main__":
    main()
