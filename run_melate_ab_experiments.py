"""Ejecuta experimentos A/B reproducibles del scorer de Melate Retro."""

from __future__ import annotations

import argparse
from datetime import datetime
import json
import math
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any

import numpy as np

from src.core.backtester import BacktestEngine
from src.core.analytics import PerformanceTracker
from src.core.fixed_origin_training import prepare_fixed_origin_models
from src.data_access.config import (
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

SELECTOR_SHADOW_VARIANTS = (
    CORE_VARIANTS[0],
    {
        "name": "G_context50_number50_adaptive",
        "description": "IA contextual 50% + IA por número 50%, con Geo adaptativo",
        "overrides": {
            "ai_context_weight": 0.50,
            "ai_number_weight": 0.50,
            "resonance_blend_mode": "adaptive",
        },
    },
    {
        "name": "H_deep_rank_5000_same_budget",
        "description": "Selector estratificado hasta rank 5000 con igual presupuesto",
        "overrides": {
            "ai_context_weight": 1.0,
            "ai_number_weight": 0.0,
            "resonance_blend_mode": "adaptive",
            "fitness_focus_max_rank": 5000,
            "fitness_candidate_max_rank": 5000,
            "fitness_rank_edges": [5, 20, 100, 300, 750, 1500, 3000, 5000],
            "fitness_bucket_plan": [
                [6, 20, 2],
                [21, 100, 3],
                [101, 300, 3],
                [301, 750, 3],
                [751, 1500, 3],
                [1501, 3000, 3],
                [3001, 5000, 2],
            ],
        },
    },
)

CONTROLLED_WEIGHT_VARIANTS = (
    CORE_VARIANTS[0],
    CORE_VARIANTS[2],
    SELECTOR_SHADOW_VARIANTS[1],
)

DEEP_DISPERSION_VARIANTS = (
    {
        "name": "I_native_30_reference",
        "description": "MRPRO nativo con 30 tickets como control de presupuesto",
        "overrides": {
            "ai_context_weight": 1.0,
            "ai_number_weight": 0.0,
            "resonance_blend_mode": "adaptive",
        },
    },
    {
        "name": "J_core20_deep10_equal_population",
        "description": (
            "20 tickets del núcleo nativo + 10 estratos profundos de igual población"
        ),
        "overrides": {
            "ai_context_weight": 1.0,
            "ai_number_weight": 0.0,
            "resonance_blend_mode": "adaptive",
            "fitness_selector_mode": "core_plus_deep",
            "deep_dispersion_core_tickets": 20,
            "deep_dispersion_tickets": 10,
            "deep_dispersion_min_rank": 501,
            "deep_dispersion_max_overlap": 3,
            "deep_dispersion_pair_novelty_weight": 0.40,
            "deep_dispersion_number_rarity_weight": 0.25,
            "deep_dispersion_dissimilarity_weight": 0.20,
            "deep_dispersion_local_quality_weight": 0.15,
        },
    },
)

ELITE_COVERAGE_DEEP_VARIANTS = (
    DEEP_DISPERSION_VARIANTS[1],
    {
        "name": "K_elite10_cover10_deep10",
        "description": (
            "10 ranks élite exactos + 10 de cobertura 2/3/4 + 10 profundos"
        ),
        "overrides": {
            "ai_context_weight": 1.0,
            "ai_number_weight": 0.0,
            "resonance_blend_mode": "adaptive",
            "fitness_selector_mode": "elite_coverage_deep",
            "portfolio_elite_tickets": 10,
            "portfolio_coverage_tickets": 10,
            "portfolio_deep_tickets": 10,
            "portfolio_coverage_max_rank": 500,
            "portfolio_min_deep_rank": 501,
            "portfolio_max_overlap": 3,
            "portfolio_pair_novelty_weight": 0.15,
            "portfolio_triple_novelty_weight": 0.30,
            "portfolio_quad_novelty_weight": 0.30,
            "portfolio_number_rarity_weight": 0.05,
            "portfolio_dissimilarity_weight": 0.05,
            "portfolio_local_quality_weight": 0.15,
        },
    },
    {
        "name": "L_elite15_cover10_deep5",
        "description": (
            "15 ranks élite exactos + 10 de cobertura 2/3/4 + 5 profundos"
        ),
        "overrides": {
            "ai_context_weight": 1.0,
            "ai_number_weight": 0.0,
            "resonance_blend_mode": "adaptive",
            "fitness_selector_mode": "elite_coverage_deep",
            "portfolio_elite_tickets": 15,
            "portfolio_coverage_tickets": 10,
            "portfolio_deep_tickets": 5,
            "portfolio_coverage_max_rank": 500,
            "portfolio_min_deep_rank": 501,
            "portfolio_max_overlap": 3,
            "portfolio_pair_novelty_weight": 0.15,
            "portfolio_triple_novelty_weight": 0.30,
            "portfolio_quad_novelty_weight": 0.30,
            "portfolio_number_rarity_weight": 0.05,
            "portfolio_dissimilarity_weight": 0.05,
            "portfolio_local_quality_weight": 0.15,
        },
    },
    {
        "name": "M_elite10_cover15_deep5",
        "description": (
            "10 ranks élite exactos + 15 de cobertura 2/3/4 + 5 profundos"
        ),
        "overrides": {
            "ai_context_weight": 1.0,
            "ai_number_weight": 0.0,
            "resonance_blend_mode": "adaptive",
            "fitness_selector_mode": "elite_coverage_deep",
            "portfolio_elite_tickets": 10,
            "portfolio_coverage_tickets": 15,
            "portfolio_deep_tickets": 5,
            "portfolio_coverage_max_rank": 500,
            "portfolio_min_deep_rank": 501,
            "portfolio_max_overlap": 3,
            "portfolio_pair_novelty_weight": 0.15,
            "portfolio_triple_novelty_weight": 0.30,
            "portfolio_quad_novelty_weight": 0.30,
            "portfolio_number_rarity_weight": 0.05,
            "portfolio_dissimilarity_weight": 0.05,
            "portfolio_local_quality_weight": 0.15,
        },
    },
)

VARIANT_SUITES = {
    "core": CORE_VARIANTS,
    "blend-sweep": BLEND_SWEEP_VARIANTS,
    "selector-shadows": SELECTOR_SHADOW_VARIANTS,
    "controlled-weights": CONTROLLED_WEIGHT_VARIANTS,
    "deep-dispersion": DEEP_DISPERSION_VARIANTS,
    "elite-coverage-deep": ELITE_COVERAGE_DEEP_VARIANTS,
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
    selected_max_hits = []
    earnings_by_draw = []
    prefix_20_max_hits = []
    prefix_20_earnings = []
    selected_ticket_observations = 0
    selected_ranks_by_draw = []
    deep_ranks_by_draw = []
    portfolio_elite_ranks_by_draw = []
    portfolio_coverage_ranks_by_draw = []
    portfolio_deep_ranks_by_draw = []
    portfolio_unique_pairs = []
    portfolio_unique_triples = []
    portfolio_unique_quads = []
    selected_unique_pairs = []
    selected_unique_triples = []
    selected_unique_quads = []
    winner_selected_max_overlap = []
    radar_jackpot_diagnostics = []
    for row in forensic_rows:
        metrics = row.get("metrics_json", {})
        if not isinstance(metrics, dict) or "selected_max_hits" not in metrics:
            selected_max_hits = []
            break
        selected_max_hits.append(int(metrics["selected_max_hits"]))
        ticket_hits = [int(value) for value in metrics.get("selected_ticket_hits", [])]
        ticket_prizes = [
            float(value) for value in metrics.get("selected_ticket_prizes", [])
        ]
        selected_ticket_observations += len(ticket_hits)
        if ticket_prizes:
            earnings_by_draw.append(float(sum(ticket_prizes)))
        if len(ticket_hits) >= 20 and len(ticket_prizes) >= 20:
            prefix_20_max_hits.append(int(max(ticket_hits[:20])))
            prefix_20_earnings.append(float(sum(ticket_prizes[:20])))
        selected_ranks_by_draw.extend(
            int(rank) for rank in metrics.get("selected_ranks", [])
        )
        deep_ranks_by_draw.extend(
            int(rank) for rank in metrics.get("deep_dispersion_ranks", [])
        )
        portfolio_elite_ranks_by_draw.extend(
            int(rank) for rank in metrics.get("portfolio_elite_ranks", [])
        )
        portfolio_coverage_ranks_by_draw.extend(
            int(rank) for rank in metrics.get("portfolio_coverage_ranks", [])
        )
        portfolio_deep_ranks_by_draw.extend(
            int(rank) for rank in metrics.get("portfolio_deep_ranks", [])
        )
        for metric_name, destination in (
            ("portfolio_unique_pairs", portfolio_unique_pairs),
            ("portfolio_unique_triples", portfolio_unique_triples),
            ("portfolio_unique_quads", portfolio_unique_quads),
        ):
            if metrics.get(metric_name) is not None:
                destination.append(int(metrics[metric_name]))
        for metric_name, destination in (
            ("selected_unique_pairs", selected_unique_pairs),
            ("selected_unique_triples", selected_unique_triples),
            ("selected_unique_quads", selected_unique_quads),
        ):
            if metrics.get(metric_name) is not None:
                destination.append(int(metrics[metric_name]))
        if metrics.get("winner_selected_max_overlap") is not None:
            winner_selected_max_overlap.append(
                int(metrics["winner_selected_max_overlap"])
            )
        if int(metrics.get("winner_in_universe", 0)) == 1:
            radar_jackpot_diagnostics.append(
                {
                    "draw_id": int(row["draw_id"]),
                    "winner_rank": int(row.get("rank", 0)),
                    "winner_stable_rank": (
                        int(metrics["winner_stable_rank"])
                        if metrics.get("winner_stable_rank") is not None
                        else None
                    ),
                    "winner_score_tie_size": int(
                        metrics.get("winner_score_tie_size", 0)
                    ),
                    "winner_stable_rank_proximity": int(
                        metrics.get("winner_stable_rank_proximity", 999)
                    ),
                    "selected_max_overlap": int(
                        metrics.get("winner_selected_max_overlap", 0)
                    ),
                    "selected_missing_numbers": int(
                        metrics.get("winner_selected_min_missing", 6)
                    ),
                    "selected_count_ge_4": int(
                        metrics.get("winner_selected_count_ge_4", 0)
                    ),
                    "selected_count_ge_5": int(
                        metrics.get("winner_selected_count_ge_5", 0)
                    ),
                    "selected_best_ranks": [
                        int(rank)
                        for rank in metrics.get("winner_selected_best_ranks", [])
                    ],
                    "selected_best_stable_ranks": [
                        int(rank)
                        for rank in metrics.get(
                            "winner_selected_best_stable_ranks", []
                        )
                    ],
                }
            )
    selected_ranks_array = np.asarray(selected_ranks_by_draw, dtype=float)
    deep_ranks_array = np.asarray(deep_ranks_by_draw, dtype=float)
    portfolio_elite_ranks_array = np.asarray(
        portfolio_elite_ranks_by_draw, dtype=float
    )
    portfolio_coverage_ranks_array = np.asarray(
        portfolio_coverage_ranks_by_draw, dtype=float
    )
    portfolio_deep_ranks_array = np.asarray(
        portfolio_deep_ranks_by_draw, dtype=float
    )
    winner_overlap_array = np.asarray(winner_selected_max_overlap, dtype=int)
    prefix_20_array = np.asarray(prefix_20_max_hits, dtype=int)
    prefix_20_earnings_array = np.asarray(prefix_20_earnings, dtype=float)
    inferred_ticket_cost = (
        float(result.investment / selected_ticket_observations)
        if selected_ticket_observations
        else 0.0
    )
    prefix_20_investment = float(
        len(prefix_20_max_hits) * 20 * inferred_ticket_cost
    )
    return {
        "draws": int(result.total_draws_tested),
        "draw_ids": [int(row["draw_id"]) for row in forensic_rows],
        "event_id": forensic_rows[0].get("event_id", "") if forensic_rows else "",
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
        "prize_breakdown": getattr(result, "prize_breakdown", {}),
        "ticket_hit_distribution": hit_distribution,
        "selected_hits_4": hit_distribution.get("4", 0),
        "selected_hits_5": hit_distribution.get("5", 0),
        "selected_jackpots": hit_distribution.get("6", 0),
        "selected_max_hits_by_draw": selected_max_hits,
        "earnings_by_draw": earnings_by_draw,
        "prefix_20": {
            "draws": int(prefix_20_array.size),
            "tickets_per_draw": 20,
            "investment": prefix_20_investment,
            "earnings": float(prefix_20_earnings_array.sum()),
            "net_balance": float(
                prefix_20_earnings_array.sum() - prefix_20_investment
            ),
            "gross_return_ratio": (
                float(prefix_20_earnings_array.sum() / prefix_20_investment)
                if prefix_20_investment
                else 0.0
            ),
            "avg_max_hits": (
                float(prefix_20_array.mean()) if prefix_20_array.size else None
            ),
            "draws_ge_4": int(np.sum(prefix_20_array >= 4)),
            "draws_ge_5": int(np.sum(prefix_20_array >= 5)),
            "draws_eq_6": int(np.sum(prefix_20_array == 6)),
            "max_hits_distribution": {
                str(hit): int(np.sum(prefix_20_array == hit)) for hit in range(7)
            },
        },
        "selected_rank_min": (
            int(np.min(selected_ranks_array)) if selected_ranks_array.size else None
        ),
        "selected_rank_median": (
            float(np.median(selected_ranks_array))
            if selected_ranks_array.size
            else None
        ),
        "selected_rank_max": (
            int(np.max(selected_ranks_array)) if selected_ranks_array.size else None
        ),
        "deep_rank_min": (
            int(np.min(deep_ranks_array)) if deep_ranks_array.size else None
        ),
        "deep_rank_median": (
            float(np.median(deep_ranks_array)) if deep_ranks_array.size else None
        ),
        "deep_rank_max": (
            int(np.max(deep_ranks_array)) if deep_ranks_array.size else None
        ),
        "portfolio_elite_rank_min": (
            int(np.min(portfolio_elite_ranks_array))
            if portfolio_elite_ranks_array.size
            else None
        ),
        "portfolio_elite_rank_max": (
            int(np.max(portfolio_elite_ranks_array))
            if portfolio_elite_ranks_array.size
            else None
        ),
        "portfolio_coverage_rank_median": (
            float(np.median(portfolio_coverage_ranks_array))
            if portfolio_coverage_ranks_array.size
            else None
        ),
        "portfolio_deep_rank_median": (
            float(np.median(portfolio_deep_ranks_array))
            if portfolio_deep_ranks_array.size
            else None
        ),
        "portfolio_unique_pairs_mean": (
            float(np.mean(portfolio_unique_pairs))
            if portfolio_unique_pairs
            else None
        ),
        "portfolio_unique_triples_mean": (
            float(np.mean(portfolio_unique_triples))
            if portfolio_unique_triples
            else None
        ),
        "portfolio_unique_quads_mean": (
            float(np.mean(portfolio_unique_quads))
            if portfolio_unique_quads
            else None
        ),
        "selected_unique_pairs_mean": (
            float(np.mean(selected_unique_pairs)) if selected_unique_pairs else None
        ),
        "selected_unique_triples_mean": (
            float(np.mean(selected_unique_triples))
            if selected_unique_triples
            else None
        ),
        "selected_unique_quads_mean": (
            float(np.mean(selected_unique_quads)) if selected_unique_quads else None
        ),
        "winner_selected_max_overlap_distribution": {
            str(hit): int(np.sum(winner_overlap_array == hit)) for hit in range(7)
        },
        "radar_jackpot_diagnostics": radar_jackpot_diagnostics,
        "jackpots_in_selector_radar": int(
            sum(
                row.get("winner_stable_rank") is not None
                for row in radar_jackpot_diagnostics
            )
        ),
        "jackpots_outside_selector_radar": int(
            sum(
                row.get("winner_stable_rank") is None
                for row in radar_jackpot_diagnostics
            )
        ),
        "draws_with_max_hits_at_least_4": sum(
            value >= 4 for value in selected_max_hits
        ),
        "draws_with_max_hits_at_least_5": sum(
            value >= 5 for value in selected_max_hits
        ),
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


def _exact_mcnemar(reference: np.ndarray, challenger: np.ndarray) -> dict[str, Any]:
    reference_only = int(np.sum(reference & ~challenger))
    challenger_only = int(np.sum(~reference & challenger))
    both = int(np.sum(reference & challenger))
    neither = int(np.sum(~reference & ~challenger))
    discordant = reference_only + challenger_only
    if discordant == 0:
        p_value = 1.0
    else:
        tail = sum(
            math.comb(discordant, value)
            for value in range(min(reference_only, challenger_only) + 1)
        ) / (2**discordant)
        p_value = min(1.0, 2.0 * tail)
    return {
        "reference_only": reference_only,
        "challenger_only": challenger_only,
        "both": both,
        "neither": neither,
        "discordant": discordant,
        "exact_two_sided_p": float(p_value),
    }


def _paired_comparisons(
    results: list[dict[str, Any]], seed: int, resamples: int = 20_000
) -> list[dict[str, Any]]:
    if not results:
        return []
    reference = np.asarray(results[0].get("selected_max_hits_by_draw", []), dtype=int)
    if reference.size == 0:
        return []

    comparisons = []
    for offset, challenger_summary in enumerate(results[1:], start=1):
        challenger = np.asarray(
            challenger_summary.get("selected_max_hits_by_draw", []), dtype=int
        )
        if challenger.shape != reference.shape:
            raise RuntimeError("Resultados A/B sin pares completos por sorteo")
        differences = challenger.astype(float) - reference.astype(float)
        rng = np.random.default_rng(int(seed) + 10_000 + offset)
        bootstrap_idx = rng.integers(
            0, differences.size, size=(int(resamples), differences.size)
        )
        bootstrap_means = differences[bootstrap_idx].mean(axis=1)
        ci_low, ci_high = np.percentile(bootstrap_means, [2.5, 97.5])
        signs = rng.choice(
            np.asarray([-1.0, 1.0]),
            size=(int(resamples), differences.size),
        )
        observed = abs(float(differences.mean()))
        permuted = np.abs((signs * differences).mean(axis=1))
        permutation_p = float((np.sum(permuted >= observed) + 1) / (resamples + 1))
        reference_earnings = np.asarray(
            results[0].get("earnings_by_draw", []), dtype=float
        )
        challenger_earnings = np.asarray(
            challenger_summary.get("earnings_by_draw", []), dtype=float
        )
        earnings_comparison = None
        if (
            reference_earnings.shape == challenger_earnings.shape
            and reference_earnings.size == reference.size
        ):
            earnings_delta = challenger_earnings - reference_earnings
            earnings_bootstrap = earnings_delta[bootstrap_idx].mean(axis=1)
            earnings_observed = abs(float(earnings_delta.mean()))
            earnings_permuted = np.abs((signs * earnings_delta).mean(axis=1))
            earnings_comparison = {
                "wins": int(np.sum(earnings_delta > 0)),
                "losses": int(np.sum(earnings_delta < 0)),
                "ties": int(np.sum(earnings_delta == 0)),
                "total_reference": float(reference_earnings.sum()),
                "total_challenger": float(challenger_earnings.sum()),
                "total_delta": float(earnings_delta.sum()),
                "mean_delta_per_draw": float(earnings_delta.mean()),
                "bootstrap_95_ci_mean_delta": [
                    float(value)
                    for value in np.percentile(earnings_bootstrap, [2.5, 97.5])
                ],
                "paired_permutation_two_sided_p": float(
                    (np.sum(earnings_permuted >= earnings_observed) + 1)
                    / (resamples + 1)
                ),
            }
        comparisons.append(
            {
                "reference": results[0]["name"],
                "challenger": challenger_summary["name"],
                "paired_draws": int(reference.size),
                "max_hits_wins": int(np.sum(differences > 0)),
                "max_hits_losses": int(np.sum(differences < 0)),
                "max_hits_ties": int(np.sum(differences == 0)),
                "mean_max_hits_reference": float(reference.mean()),
                "mean_max_hits_challenger": float(challenger.mean()),
                "mean_paired_delta": float(differences.mean()),
                "bootstrap_95_ci_mean_delta": [float(ci_low), float(ci_high)],
                "paired_permutation_two_sided_p": permutation_p,
                "mcnemar_ge_4": _exact_mcnemar(reference >= 4, challenger >= 4),
                "mcnemar_ge_5": _exact_mcnemar(reference >= 5, challenger >= 5),
                "mcnemar_eq_6": _exact_mcnemar(reference == 6, challenger == 6),
                "paired_earnings": earnings_comparison,
            }
        )
    return comparisons


def run_experiments(
    backtest_size: int,
    tickets: int,
    seed: int,
    variants=CORE_VARIANTS,
    experiment_name: str = "melate_resonance_ab_v1",
    isolate_ledger: bool = True,
) -> dict[str, Any]:
    profile = LOTTERY_PROFILES["melate_retro"]
    history = LotteryLoader(profile).load_data()
    artifacts = prepare_fixed_origin_models(history, backtest_size)
    results: list[dict[str, Any]] = []
    reference_draw_ids: list[int] | None = None

    with TemporaryDirectory(prefix="mrpro_ab_ledger_") as ledger_directory:
        ledger_root = Path(ledger_directory)
        for variant in variants:
            overrides = dict(BEST_SETTINGS)
            overrides.update(variant["overrides"])
            overrides.update(
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
                tickets,
                backtest_size=backtest_size,
                filter_overrides=overrides,
            )
            engine = BacktestEngine()
            if isolate_ledger:
                engine.tracker = PerformanceTracker(
                    log_path=ledger_root / "detailed_forensic_log.csv",
                    json_path=ledger_root / "backtest_results.json",
                    archive_directory=ledger_root / "forensic_log_archive",
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
        "fixed_origin": artifacts.to_dict(),
        "evaluated_draw_ids": reference_draw_ids or [],
        "variants": results,
        "paired_comparisons": _paired_comparisons(results, seed),
        "ledger_isolated": bool(isolate_ledger),
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
        "--variant",
        action="append",
        default=None,
        help="Nombre exacto de variante; puede repetirse para ejecutar un subconjunto",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
    )
    args = parser.parse_args()

    default_names = {
        "core": "melate_ab_experiment.json",
        "blend-sweep": "melate_blend_sweep.json",
        "selector-shadows": "melate_selector_shadows.json",
        "controlled-weights": "melate_controlled_weights.json",
        "deep-dispersion": "melate_deep_dispersion_20_10.json",
        "elite-coverage-deep": "melate_elite_coverage_deep.json",
    }
    output = args.output or Path(DATA_FOLDER_PATH / default_names[args.suite])
    selected_variants = VARIANT_SUITES[args.suite]
    if args.variant:
        requested = set(args.variant)
        selected_variants = tuple(
            variant for variant in selected_variants if variant["name"] in requested
        )
        missing = requested.difference(
            variant["name"] for variant in selected_variants
        )
        if missing:
            parser.error(f"Variantes desconocidas para {args.suite}: {sorted(missing)}")
    payload = run_experiments(
        args.draws,
        args.tickets,
        args.seed,
        variants=selected_variants,
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
