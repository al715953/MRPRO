"""CLI reproducible for combinatorial covering experiments."""

from __future__ import annotations

import argparse
from datetime import datetime
import itertools
from pathlib import Path
import time

from src.data_access.config import (
    DATA_FOLDER_PATH,
    LOTTERY_PROFILES,
    TICKET_SIZE,
    TOTAL_BALLS,
    VERSION_TAG,
)
from src.data_access.loader import LotteryLoader
from src.strategies.combinatorial.config import CoveringExperimentConfig
from src.strategies.combinatorial.covering import ProblemTooLargeError, estimate_problem
from src.strategies.combinatorial.experiment import (
    build_design_bundle,
    design_bundle_to_dict,
    run_historical_experiment,
)
from src.strategies.combinatorial.reporting import (
    add_incremental_efficiency,
    generate_plots,
    print_experiment_summary,
    write_json_report,
    write_summary_csv,
)


def _parse_int_list(raw: str) -> list[int]:
    return [int(value.strip()) for value in str(raw).split(",") if value.strip()]


def _resolve_t_values(raw: str, ticket_size: int) -> list[int]:
    values = []
    for value in str(raw).split(","):
        token = value.strip().lower().replace(" ", "")
        if not token:
            continue
        if token.startswith("k-"):
            values.append(int(ticket_size) - int(token[2:]))
        elif token == "k":
            values.append(int(ticket_size))
        else:
            values.append(int(token))
    return values


def execute_sweep(
    *,
    v_values: list[int],
    t_values: list[int],
    budgets: list[int],
    candidate_method: str,
    random_trials: int,
    seed: int,
    local_iterations: int,
    coverage_target: float,
    backtest_draws: int,
    include_current: bool,
    current_tickets: int,
    explicit_candidates: list[int] | None,
    mode: str,
) -> dict:
    profile = LOTTERY_PROFILES["melate_retro"]
    history = LotteryLoader(profile).load_data() if mode in {"both", "historical"} else None
    payload = {
        "generated_at": datetime.now().isoformat(),
        "experiment": "combinatorial_covering_design_v1",
        "version": VERSION_TAG,
        "interpretation": {
            "combinatorial_advantage": "Coverage dentro de un V fijo.",
            "predictive_advantage": "Calidad fuera de muestra al construir V antes del sorteo.",
            "warning": "Una ventaja combinatoria no demuestra ventaja predictiva ni cambia la física del sorteo.",
        },
        "experiments": [],
    }

    for v, t, budget in itertools.product(v_values, t_values, budgets):
        config = CoveringExperimentConfig(
            candidate_pool_size=int(v),
            target_subset_size=int(t),
            ticket_budget=int(budget),
            random_trials=int(random_trials),
            random_seed=int(seed),
            local_search_iterations=int(local_iterations),
            coverage_target=float(coverage_target),
            candidate_method=str(candidate_method),
            explicit_candidates=explicit_candidates,
            backtest_draws=int(backtest_draws),
            include_current_mrpro=bool(include_current),
            current_mrpro_ticket_count=int(current_tickets),
        )
        row = {
            "status": "pending",
            "config": config.to_dict(),
            "ticket_size": int(TICKET_SIZE),
            "total_balls": int(TOTAL_BALLS),
            "target_subset_size_resolved": int(t),
        }
        started = time.perf_counter()
        try:
            row["estimate"] = estimate_problem(v, TICKET_SIZE, t).to_dict()
            config.validate(TOTAL_BALLS, TICKET_SIZE)
            bundle = build_design_bundle(config, ticket_size=TICKET_SIZE)
            row["mathematical"] = (
                design_bundle_to_dict(bundle) if mode in {"both", "math"} else None
            )
            row["historical"] = (
                run_historical_experiment(
                    history,
                    config,
                    total_balls=TOTAL_BALLS,
                    ticket_size=TICKET_SIZE,
                    design_bundle=bundle,
                )
                if mode in {"both", "historical"}
                else None
            )
            row["status"] = "completed"
        except ProblemTooLargeError as exc:
            row["status"] = "skipped_guardrail"
            row["skip_reason"] = str(exc)
        except ValueError as exc:
            row["status"] = "skipped_invalid"
            row["skip_reason"] = str(exc)
        row["elapsed_seconds"] = float(time.perf_counter() - started)
        payload["experiments"].append(row)
        print_experiment_summary(row)

    add_incremental_efficiency(payload)
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--v", default="15", help="Lista: 10,12,15,18,20")
    parser.add_argument("--t", default="k-1", help="Lista: k-1,k-2")
    parser.add_argument("--budget", default="300", help="Lista: 50,100,200,300")
    parser.add_argument(
        "--candidate-method",
        choices=(
            "mrpro_candidate_set",
            "random_candidate_set",
            "oracle_candidate_set",
            "explicit_candidate_set",
        ),
        default="oracle_candidate_set",
    )
    parser.add_argument("--explicit", default="")
    parser.add_argument("--random-trials", type=int, default=100)
    parser.add_argument("--seed", type=int, default=20260816)
    parser.add_argument("--local-iterations", type=int, default=100)
    parser.add_argument("--coverage-target", type=float, default=1.0)
    parser.add_argument("--draws", type=int, default=108)
    parser.add_argument("--no-current", action="store_true")
    parser.add_argument("--current-tickets", type=int, default=24)
    parser.add_argument("--mode", choices=("math", "historical", "both"), default="both")
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()

    v_values = _parse_int_list(args.v)
    t_values = _resolve_t_values(args.t, TICKET_SIZE)
    budgets = _parse_int_list(args.budget)
    explicit = _parse_int_list(args.explicit) if args.explicit else None
    payload = execute_sweep(
        v_values=v_values,
        t_values=t_values,
        budgets=budgets,
        candidate_method=args.candidate_method,
        random_trials=args.random_trials,
        seed=args.seed,
        local_iterations=args.local_iterations,
        coverage_target=args.coverage_target,
        backtest_draws=args.draws,
        include_current=not args.no_current,
        current_tickets=args.current_tickets,
        explicit_candidates=explicit,
        mode=args.mode,
    )

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output = args.output or DATA_FOLDER_PATH / f"covering_experiment_{stamp}.json"
    write_json_report(payload, output)
    csv_path = output.with_suffix(".csv")
    write_summary_csv(payload, csv_path)
    plots = generate_plots(payload, output.with_suffix(""))
    print(f"Reporte JSON: {output}")
    print(f"Resumen CSV: {csv_path}")
    for plot in plots:
        print(f"Gráfico: {plot}")


if __name__ == "__main__":
    main()
