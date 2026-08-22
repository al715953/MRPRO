"""Benchmark MRPRO contra boletos uniformes del universo completo de Melate.

El baseline no usa historia, filtros, IA, ranking ni reglas de reducción. Para cada
sorteo toma ``m`` combinaciones distintas de las C(39, 6) posibles. La simulación
trabaja con las particiones hipergeométricas por aciertos; esto es exactamente
equivalente a enumerar el universo completo y muestrear boletos sin reemplazo,
pero evita materializar más de tres millones de combinaciones.
"""

from __future__ import annotations

import argparse
from datetime import datetime
import json
import math
from pathlib import Path
from typing import Any

import numpy as np

from src.core.rules import MelateRetroRules
from src.data_access.config import DATA_FOLDER_PATH, TICKET_SIZE, TOTAL_BALLS
from src.strategies.combinatorial.baselines import (
    score_against_random,
    summarize_distribution,
)


DEFAULT_VARIANT = "J_core20_deep10_equal_population"
DEFAULT_SOURCE = DATA_FOLDER_PATH / "melate_deep_dispersion_20_10.json"
DEFAULT_OUTPUT = DATA_FOLDER_PATH / "melate_full_random_30_vs_deep_108.json"


def build_outcome_population(
    total_balls: int = TOTAL_BALLS,
    ticket_size: int = TICKET_SIZE,
) -> list[dict[str, Any]]:
    """Partition the full ticket universe by natural hits and additional ball.

    Melate has six natural winners, one additional ball and ``N - 7`` neutral
    balls. Each returned population is a disjoint set of tickets, so drawing from
    these populations with a multivariate hypergeometric distribution is uniform
    sampling without replacement from the complete ticket universe.
    """
    if ticket_size != 6:
        raise ValueError("Este benchmark está definido para boletos Melate de 6 números")
    neutral_balls = int(total_balls) - ticket_size - 1
    if neutral_balls < ticket_size:
        raise ValueError("El universo no contiene suficientes números neutrales")

    rules = MelateRetroRules()
    cells: list[dict[str, Any]] = []
    for hits in range(ticket_size + 1):
        for has_additional in (False, True):
            neutral_needed = ticket_size - hits - int(has_additional)
            if neutral_needed < 0 or neutral_needed > neutral_balls:
                continue
            population = math.comb(ticket_size, hits) * math.comb(
                neutral_balls, neutral_needed
            )
            if population <= 0:
                continue
            cells.append(
                {
                    "natural_hits": hits,
                    "has_additional": has_additional,
                    "population": population,
                    "prize": rules.calculate_prize(hits, has_additional),
                    "prize_category": rules.prize_category(hits, has_additional),
                }
            )

    expected = math.comb(int(total_balls), int(ticket_size))
    observed = sum(int(cell["population"]) for cell in cells)
    if observed != expected:
        raise AssertionError(f"Partición inválida: {observed:,} != {expected:,}")
    return cells


def _allocate_metric_arrays(trials: int) -> dict[str, np.ndarray]:
    integer_metrics = (
        "draws_ge_4",
        "draws_ge_5",
        "draws_eq_6",
        "tickets_hits_4",
        "tickets_hits_5",
        "tickets_hits_6",
    )
    arrays = {
        name: np.empty(int(trials), dtype=np.int32) for name in integer_metrics
    }
    arrays["earnings"] = np.empty(int(trials), dtype=np.float64)
    arrays["earnings_without_jackpot"] = np.empty(int(trials), dtype=np.float64)
    arrays["avg_max_hits"] = np.empty(int(trials), dtype=np.float64)
    return arrays


def simulate_random_portfolios(
    *,
    draws: int,
    tickets: int,
    trials: int,
    seed: int,
    chunk_size: int = 2_000,
    total_balls: int = TOTAL_BALLS,
    ticket_size: int = TICKET_SIZE,
) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray], list[dict[str, Any]]]:
    """Simulate fresh, unique, uniform tickets for every historical draw.

    Returns trial-level metrics, aggregate prize-category counts per trial and the
    exact universe partition. Keeping only trial summaries makes 100k repetitions
    inexpensive while preserving exact without-replacement sampling per draw.
    """
    draws, tickets, trials = int(draws), int(tickets), int(trials)
    if draws <= 0 or tickets <= 0 or trials <= 0:
        raise ValueError("draws, tickets y trials deben ser positivos")
    cells = build_outcome_population(total_balls, ticket_size)
    populations = np.asarray([cell["population"] for cell in cells], dtype=np.int64)
    universe_size = int(populations.sum())
    if tickets > universe_size:
        raise ValueError("No se pueden tomar más boletos que el universo completo")

    hits = np.asarray([cell["natural_hits"] for cell in cells], dtype=np.int8)
    prizes = np.asarray([cell["prize"] for cell in cells], dtype=np.float64)
    categories = tuple(dict.fromkeys(cell["prize_category"] for cell in cells))
    category_masks = {
        category: np.asarray(
            [cell["prize_category"] == category for cell in cells], dtype=bool
        )
        for category in categories
    }
    metrics = _allocate_metric_arrays(trials)
    category_counts = {
        category: np.empty(trials, dtype=np.int32) for category in categories
    }
    rng = np.random.default_rng(int(seed))

    for start in range(0, trials, max(1, int(chunk_size))):
        stop = min(trials, start + max(1, int(chunk_size)))
        size = stop - start
        counts = rng.multivariate_hypergeometric(
            populations,
            tickets,
            size=(size, draws),
        )
        if not np.all(counts.sum(axis=2) == tickets):
            raise AssertionError("El muestreo no conservó exactamente m boletos")

        earnings_by_draw = counts @ prizes
        max_hits_by_draw = np.where(counts > 0, hits, -1).max(axis=2)
        jackpot_count = counts[:, :, hits == ticket_size].sum(axis=(1, 2))
        total_earnings = earnings_by_draw.sum(axis=1)

        metrics["earnings"][start:stop] = total_earnings
        metrics["earnings_without_jackpot"][start:stop] = (
            total_earnings - jackpot_count * MelateRetroRules().pay_table[(6, False)]
        )
        metrics["avg_max_hits"][start:stop] = max_hits_by_draw.mean(axis=1)
        metrics["draws_ge_4"][start:stop] = np.sum(max_hits_by_draw >= 4, axis=1)
        metrics["draws_ge_5"][start:stop] = np.sum(max_hits_by_draw >= 5, axis=1)
        metrics["draws_eq_6"][start:stop] = np.sum(max_hits_by_draw == 6, axis=1)
        for hit_value in (4, 5, 6):
            metrics[f"tickets_hits_{hit_value}"][start:stop] = counts[
                :, :, hits == hit_value
            ].sum(axis=(1, 2))
        for category, mask in category_masks.items():
            category_counts[category][start:stop] = counts[:, :, mask].sum(
                axis=(1, 2)
            )

    return metrics, category_counts, cells


def _comparison(value: float, random_values: np.ndarray) -> dict[str, Any]:
    array = np.asarray(random_values, dtype=np.float64)
    result = score_against_random(float(value), array)
    result.update(
        {
            "mrpro_value": float(value),
            "random_probability_ge_mrpro": float(np.mean(array >= value)),
            "random_probability_gt_mrpro": float(np.mean(array > value)),
            "random_probability_eq_mrpro": float(np.mean(array == value)),
        }
    )
    return result


def _load_mrpro_variant(
    report_path: Path,
    variant_name: str,
    *,
    expected_draws: int,
    expected_tickets: int,
) -> tuple[dict[str, Any], dict[str, Any]]:
    source = json.loads(Path(report_path).read_text(encoding="utf-8"))
    variant = next(
        (row for row in source.get("variants", []) if row.get("name") == variant_name),
        None,
    )
    if variant is None:
        raise ValueError(f"No existe la variante {variant_name!r} en {report_path}")
    if int(variant.get("draws", -1)) != int(expected_draws):
        raise ValueError("El reporte MRPRO no tiene la misma cantidad de sorteos")
    observed_tickets = int(source.get("tickets_per_draw", -1))
    if observed_tickets != int(expected_tickets):
        raise ValueError(
            f"El reporte MRPRO usa {observed_tickets} boletos, no {expected_tickets}"
        )
    if len(variant.get("selected_max_hits_by_draw", [])) != int(expected_draws):
        raise ValueError("El reporte MRPRO no contiene telemetría completa por sorteo")
    return source, variant


def build_benchmark_report(
    *,
    mrpro_report_path: Path = DEFAULT_SOURCE,
    variant_name: str = DEFAULT_VARIANT,
    draws: int = 108,
    tickets: int = 30,
    trials: int = 100_000,
    seed: int = 20260821,
    chunk_size: int = 2_000,
) -> dict[str, Any]:
    source, mrpro = _load_mrpro_variant(
        Path(mrpro_report_path),
        variant_name,
        expected_draws=draws,
        expected_tickets=tickets,
    )
    metrics, category_counts, cells = simulate_random_portfolios(
        draws=draws,
        tickets=tickets,
        trials=trials,
        seed=seed,
        chunk_size=chunk_size,
    )
    rules = MelateRetroRules()
    universe_size = math.comb(TOTAL_BALLS, TICKET_SIZE)
    total_ticket_observations = int(draws) * int(tickets)
    investment = float(total_ticket_observations * rules.ticket_cost)
    expected_payout_per_ticket = sum(
        cell["population"] * cell["prize"] for cell in cells
    ) / universe_size
    expected_jackpot_earnings = (
        total_ticket_observations
        * rules.pay_table[(6, False)]
        / universe_size
    )
    probability_any_jackpot = 1.0 - (1.0 - tickets / universe_size) ** draws

    mrpro_max = np.asarray(mrpro["selected_max_hits_by_draw"], dtype=np.int8)
    mrpro_values = {
        "earnings": float(mrpro["earnings"]),
        "earnings_without_jackpot": float(mrpro["earnings"])
        - float(mrpro.get("selected_jackpots", 0)) * rules.pay_table[(6, False)],
        "avg_max_hits": float(mrpro_max.mean()),
        "draws_ge_4": int(np.sum(mrpro_max >= 4)),
        "draws_ge_5": int(np.sum(mrpro_max >= 5)),
        "draws_eq_6": int(np.sum(mrpro_max == 6)),
        "tickets_hits_4": int(mrpro.get("ticket_hit_distribution", {}).get("4", 0)),
        "tickets_hits_5": int(mrpro.get("ticket_hit_distribution", {}).get("5", 0)),
        "tickets_hits_6": int(mrpro.get("ticket_hit_distribution", {}).get("6", 0)),
    }

    random_summaries = {
        name: summarize_distribution(values) for name, values in metrics.items()
    }
    comparisons = {
        name: _comparison(value, metrics[name]) for name, value in mrpro_values.items()
    }
    mrpro_prize_breakdown = mrpro.get("prize_breakdown", {})
    category_comparisons = {
        category: _comparison(
            float(mrpro_prize_breakdown.get(category, {}).get("tickets", 0)),
            category_counts[category],
        )
        for category in category_counts
        if category != "SIN_PREMIO"
    }
    no_jackpot_mask = metrics["draws_eq_6"] == 0
    random_no_jackpot_earnings = metrics["earnings"][no_jackpot_mask]
    example_index = 0
    example = {name: float(values[example_index]) for name, values in metrics.items()}
    for name in (
        "draws_ge_4",
        "draws_ge_5",
        "draws_eq_6",
        "tickets_hits_4",
        "tickets_hits_5",
        "tickets_hits_6",
    ):
        example[name] = int(example[name])

    return {
        "generated_at": datetime.now().isoformat(),
        "experiment": "melate_full_universe_random_vs_mrpro_deep_v1",
        "scientific_question": (
            "Con 30 boletos por sorteo, ¿MRPRO core20+deep10 supera a seleccionar "
            "30 combinaciones uniformes sin reglas desde el universo completo?"
        ),
        "configuration": {
            "draws": int(draws),
            "tickets_per_draw": int(tickets),
            "random_trials": int(trials),
            "random_seed": int(seed),
            "random_portfolio_policy": "fresh_per_draw",
            "sampling": "uniform_without_replacement",
            "uses_history_or_rules": False,
            "simulation_method": "exact_multivariate_hypergeometric_partition",
            "total_balls": TOTAL_BALLS,
            "ticket_size": TICKET_SIZE,
            "full_universe_size": universe_size,
            "ticket_cost": rules.ticket_cost,
            "investment_per_108_draw_run": investment,
        },
        "mrpro_source": {
            "path": str(Path(mrpro_report_path)),
            "experiment": source.get("experiment"),
            "generated_at": source.get("generated_at"),
            "variant": variant_name,
            "evaluated_draw_ids": source.get("evaluated_draw_ids", []),
            "fixed_origin": source.get("fixed_origin", {}),
        },
        "mathematical_validation": {
            "outcome_population": cells,
            "partition_population_sum": sum(cell["population"] for cell in cells),
            "expected_payout_per_random_ticket": expected_payout_per_ticket,
            "expected_gross_return_ratio": expected_payout_per_ticket / rules.ticket_cost,
            "expected_earnings_per_run": expected_payout_per_ticket
            * total_ticket_observations,
            "expected_jackpot_earnings_per_run": expected_jackpot_earnings,
            "expected_earnings_per_run_excluding_jackpot_prize": (
                expected_payout_per_ticket * total_ticket_observations
                - expected_jackpot_earnings
            ),
            "expected_jackpots_per_run": total_ticket_observations / universe_size,
            "probability_at_least_one_jackpot_per_run": probability_any_jackpot,
        },
        "mrpro": {
            "name": variant_name,
            "values": mrpro_values,
            "investment": float(mrpro["investment"]),
            "gross_return_ratio": float(mrpro["gross_return_ratio"]),
            "net_roi": float(mrpro["net_roi"]),
            "prize_breakdown": mrpro_prize_breakdown,
        },
        "random": {
            "one_reproducible_run": example,
            "distribution": random_summaries,
            "earnings_conditional_no_jackpot": {
                "trials": int(np.sum(no_jackpot_mask)),
                "distribution": summarize_distribution(random_no_jackpot_earnings),
            },
            "prize_category_ticket_counts": {
                category: summarize_distribution(values)
                for category, values in category_counts.items()
            },
        },
        "mrpro_vs_random": comparisons,
        "mrpro_vs_random_prize_categories": category_comparisons,
        "interpretation_guardrail": (
            "Este experimento mide selección de cartera sobre históricos. No cambia "
            "la probabilidad física del sorteo ni prueba capacidad predictiva futura."
        ),
    }


def _print_summary(report: dict[str, Any]) -> None:
    config = report["configuration"]
    mrpro = report["mrpro"]["values"]
    random = report["random"]["distribution"]
    comparisons = report["mrpro_vs_random"]
    print(
        f"Universo completo: {config['full_universe_size']:,} | "
        f"{config['draws']} sorteos x {config['tickets_per_draw']} boletos | "
        f"{config['random_trials']:,} repeticiones"
    )
    print("\nMÉTRICA                 MRPRO      RANDOM MEDIA   P50      P95   PCT MRPRO")
    for name in ("earnings", "avg_max_hits", "draws_ge_4", "draws_ge_5", "draws_eq_6"):
        summary = random[name]
        print(
            f"{name:22s} {mrpro[name]:9.3f} {summary['mean']:13.3f} "
            f"{summary['p50']:8.3f} {summary['p95']:8.3f} "
            f"{comparisons[name]['percentile_rank']:10.2f}"
        )
    no_jackpot = report["random"]["earnings_conditional_no_jackpot"]["distribution"]
    print(
        "\nGanancia random sin jackpot: "
        f"media={no_jackpot['mean']:.2f}, p50={no_jackpot['p50']:.2f}, "
        f"p95={no_jackpot['p95']:.2f}"
    )
    print(
        "Probabilidad simulada de que random iguale/supere a MRPRO en ganancias: "
        f"{comparisons['earnings']['random_probability_ge_mrpro']:.2%}"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--draws", type=int, default=108)
    parser.add_argument("--tickets", type=int, default=30)
    parser.add_argument("--trials", type=int, default=100_000)
    parser.add_argument("--seed", type=int, default=20260821)
    parser.add_argument("--chunk-size", type=int, default=2_000)
    parser.add_argument("--mrpro-report", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--variant", default=DEFAULT_VARIANT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()

    report = build_benchmark_report(
        mrpro_report_path=args.mrpro_report,
        variant_name=args.variant,
        draws=args.draws,
        tickets=args.tickets,
        trials=args.trials,
        seed=args.seed,
        chunk_size=args.chunk_size,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    _print_summary(report)
    print(f"\nReporte guardado en {args.output}")


if __name__ == "__main__":
    main()
