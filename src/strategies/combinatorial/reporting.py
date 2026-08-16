from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from rich.console import Console
from rich.table import Table

from .experiment import (
    METHOD_CURRENT,
    METHOD_CURRENT_RESTRICTED,
    METHOD_CURRENT_SAME_M,
    METHOD_EXHAUSTIVE,
    METHOD_GREEDY,
    METHOD_LOCAL,
    METHOD_RANDOM,
)


console = Console()


def _json_default(value):
    if isinstance(value, (np.integer, np.floating)):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    raise TypeError(f"No serializable: {type(value)!r}")


def write_json_report(payload: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, default=_json_default),
        encoding="utf-8",
    )


def _method_rows(experiment: dict[str, Any]) -> list[dict[str, Any]]:
    config = experiment["config"]
    math_result = experiment.get("mathematical") or {}
    historical = experiment.get("historical") or {}
    rows = []
    math_methods = math_result.get("methods", {})
    history_methods = historical.get("methods", {})
    method_names = set(math_methods) | set(history_methods)
    for method in sorted(method_names):
        math_row = math_methods.get(method, {})
        history_row = history_methods.get(method, {})
        rows.append(
            {
                "candidate_method": config["candidate_method"],
                "v": config["candidate_pool_size"],
                "k": experiment["ticket_size"],
                "t": experiment["target_subset_size_resolved"],
                "budget": config["ticket_budget"],
                "method": method,
                "m": history_row.get("ticket_count", math_row.get("ticket_count")),
                "coverage_t": math_row.get("coverage_t"),
                "avg_max_hits": history_row.get("avg_max_hits"),
                "hit_rate_ge_4": history_row.get("hit_rate_ge_4"),
                "hit_rate_ge_5": history_row.get("hit_rate_ge_5"),
                "hit_rate_eq_6": history_row.get("hit_rate_eq_6"),
            }
        )
    random_math = math_result.get("random_same_size", {}).get(METHOD_GREEDY, {})
    random_history = historical.get("random_same_size", {})
    if random_math or random_history:
        distributions = random_history.get("metric_distributions", {})
        rows.append(
            {
                "candidate_method": config["candidate_method"],
                "v": config["candidate_pool_size"],
                "k": experiment["ticket_size"],
                "t": experiment["target_subset_size_resolved"],
                "budget": config["ticket_budget"],
                "method": "RANDOM_MEAN",
                "m": random_history.get("ticket_count", random_math.get("ticket_count")),
                "coverage_t": random_math.get("coverage", {}).get("mean"),
                "avg_max_hits": distributions.get("avg_max_hits", {}).get("mean"),
                "hit_rate_ge_4": distributions.get("hit_rate_ge_4", {}).get("mean"),
                "hit_rate_ge_5": distributions.get("hit_rate_ge_5", {}).get("mean"),
                "hit_rate_eq_6": distributions.get("hit_rate_eq_6", {}).get("mean"),
            }
        )
        rows.append(
            {
                "candidate_method": config["candidate_method"],
                "v": config["candidate_pool_size"],
                "k": experiment["ticket_size"],
                "t": experiment["target_subset_size_resolved"],
                "budget": config["ticket_budget"],
                "method": "RANDOM_P95",
                "m": random_history.get("ticket_count", random_math.get("ticket_count")),
                "coverage_t": random_math.get("coverage", {}).get("p95"),
                "avg_max_hits": distributions.get("avg_max_hits", {}).get("p95"),
                "hit_rate_ge_4": distributions.get("hit_rate_ge_4", {}).get("p95"),
                "hit_rate_ge_5": distributions.get("hit_rate_ge_5", {}).get("p95"),
                "hit_rate_eq_6": distributions.get("hit_rate_eq_6", {}).get("p95"),
            }
        )
    return rows


def write_summary_csv(payload: dict[str, Any], path: Path) -> list[dict[str, Any]]:
    rows = []
    for experiment in payload.get("experiments", []):
        if experiment.get("status") == "completed":
            rows.extend(_method_rows(experiment))
    if not rows:
        return rows
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as target:
        writer = csv.DictWriter(target, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    return rows


def add_incremental_efficiency(payload: dict[str, Any]) -> None:
    groups: dict[tuple, list[dict[str, Any]]] = {}
    for experiment in payload.get("experiments", []):
        if experiment.get("status") != "completed":
            continue
        cfg = experiment["config"]
        key = (
            cfg["candidate_method"],
            cfg["candidate_pool_size"],
            experiment["target_subset_size_resolved"],
        )
        groups.setdefault(key, []).append(experiment)
    for experiments in groups.values():
        experiments.sort(key=lambda row: int(row["config"]["ticket_budget"]))
        previous_hits: dict[str, tuple[int, float]] = {}
        previous_coverage: dict[str, tuple[int, float]] = {}
        for experiment in experiments:
            historical = experiment.get("historical") or {}
            increments = {}
            for method, metrics in historical.get("methods", {}).items():
                current = (
                    int(metrics.get("ticket_count", 0)),
                    float(metrics.get("hit_rate_ge_4") or 0.0),
                )
                if method in previous_hits:
                    delta_m = current[0] - previous_hits[method][0]
                    delta_rate = current[1] - previous_hits[method][1]
                    increments.setdefault(method, {}).update({
                        "incremental_tickets": delta_m,
                        "incremental_hit_rate_ge_4": delta_rate,
                        "incremental_hit_rate_per_ticket": (
                            float(delta_rate / delta_m) if delta_m > 0 else None
                        ),
                    })
                previous_hits[method] = current
            mathematical = experiment.get("mathematical") or {}
            for method, metrics in mathematical.get("methods", {}).items():
                current = (
                    int(metrics.get("ticket_count", 0)),
                    float(metrics.get("coverage_t") or 0.0),
                )
                if method in previous_coverage:
                    delta_m = current[0] - previous_coverage[method][0]
                    delta_coverage = current[1] - previous_coverage[method][1]
                    increments.setdefault(method, {}).update(
                        {
                            "incremental_tickets_coverage": delta_m,
                            "incremental_coverage_t": delta_coverage,
                            "incremental_coverage_per_ticket": (
                                float(delta_coverage / delta_m) if delta_m > 0 else None
                            ),
                        }
                    )
                previous_coverage[method] = current
            experiment["incremental_efficiency"] = increments


def print_experiment_summary(experiment: dict[str, Any]) -> None:
    if experiment.get("status") != "completed":
        console.print(
            f"[yellow]SKIP[/] {experiment.get('skip_reason', 'configuración omitida')}"
        )
        return
    cfg = experiment["config"]
    title = (
        f"Covering {cfg['candidate_method']} | "
        f"v={cfg['candidate_pool_size']} t={experiment['target_subset_size_resolved']} "
        f"budget={cfg['ticket_budget']}"
    )
    table = Table(title=title)
    table.add_column("Método")
    table.add_column("M", justify="right")
    table.add_column("Coverage", justify="right")
    table.add_column("Avg max hits", justify="right")
    table.add_column(">=4", justify="right")
    table.add_column(">=5", justify="right")
    table.add_column("6/6", justify="right")
    for row in _method_rows(experiment):
        table.add_row(
            str(row["method"]),
            str(row.get("m", "")),
            _fmt(row.get("coverage_t")),
            _fmt(row.get("avg_max_hits")),
            _fmt(row.get("hit_rate_ge_4")),
            _fmt(row.get("hit_rate_ge_5")),
            _fmt(row.get("hit_rate_eq_6")),
        )
    console.print(table)


def _fmt(value) -> str:
    if value is None:
        return "—"
    return f"{float(value):.4f}"


def generate_plots(payload: dict[str, Any], output_prefix: Path) -> list[str]:
    completed = [
        row for row in payload.get("experiments", []) if row.get("status") == "completed"
    ]
    if not completed:
        return []
    generated = []
    groups: dict[tuple, list[dict[str, Any]]] = {}
    for experiment in completed:
        cfg = experiment["config"]
        key = (
            int(cfg["candidate_pool_size"]),
            int(experiment["target_subset_size_resolved"]),
            str(cfg["candidate_method"]),
        )
        groups.setdefault(key, []).append(experiment)

    for (v, t, candidate_method), group in groups.items():
        fig, ax = plt.subplots(figsize=(10, 6))
        for method, color in ((METHOD_GREEDY, "#00a6d6"), (METHOD_LOCAL, "#7a4dd8")):
            points = []
            for experiment in group:
                metrics = (experiment.get("mathematical") or {}).get("methods", {}).get(
                    method
                )
                if metrics:
                    points.append((metrics["ticket_count"], metrics["coverage_t"]))
            if points:
                points.sort()
                ax.plot(*zip(*points), marker="o", label=method, color=color)
        random_points = []
        for experiment in group:
            random = (experiment.get("mathematical") or {}).get(
                "random_same_size", {}
            ).get(METHOD_GREEDY, {})
            if random:
                random_points.append(
                    (
                        random["ticket_count"],
                        random["coverage"]["mean"],
                        random["coverage"]["p05"],
                        random["coverage"]["p95"],
                    )
                )
        if random_points:
            random_points.sort()
            x, mean, low, high = (np.asarray(values) for values in zip(*random_points))
            ax.plot(x, mean, marker="o", label="RANDOM mean", color="#777777")
            ax.fill_between(
                x,
                low,
                high,
                alpha=0.2,
                color="#777777",
                label="RANDOM P5-P95",
            )
        ax.set_xlabel("Número de boletos")
        ax.set_ylabel("Coverage t")
        ax.set_ylim(0, 1.02)
        ax.grid(alpha=0.25)
        ax.legend()
        ax.set_title(f"v={v}, t={t}, candidatos={candidate_method}")
        fig.tight_layout()
        suffix = f"_coverage_v{v}_t{t}.png"
        coverage_path = output_prefix.with_name(output_prefix.name + suffix)
        fig.savefig(coverage_path, dpi=160)
        plt.close(fig)
        generated.append(str(coverage_path))

        historical_group = [row for row in group if row.get("historical")]
        if historical_group:
            fig, ax = plt.subplots(figsize=(10, 6))
            for method, color in (
                (METHOD_GREEDY, "#00a6d6"),
                (METHOD_LOCAL, "#7a4dd8"),
                (METHOD_CURRENT, "#e67e22"),
                (METHOD_CURRENT_SAME_M, "#d62728"),
                (METHOD_CURRENT_RESTRICTED, "#2ca02c"),
                (METHOD_EXHAUSTIVE, "#222222"),
            ):
                points = []
                for experiment in historical_group:
                    metrics = experiment["historical"].get("methods", {}).get(method)
                    if metrics and metrics.get("hit_rate_ge_4") is not None:
                        points.append(
                            (metrics["ticket_count"], metrics["hit_rate_ge_4"])
                        )
                if points:
                    points.sort()
                    ax.plot(*zip(*points), marker="o", label=method, color=color)
            ax.set_xlabel("Número de boletos")
            ax.set_ylabel("P(maxHits >= 4)")
            ax.set_ylim(0, 1.02)
            ax.grid(alpha=0.25)
            ax.legend()
            ax.set_title(f"v={v}, t={t}, candidatos={candidate_method}")
            fig.tight_layout()
            hit_path = output_prefix.with_name(
                output_prefix.name + f"_hits_v{v}_t{t}.png"
            )
            fig.savefig(hit_path, dpi=160)
            plt.close(fig)
            generated.append(str(hit_path))
    return generated
