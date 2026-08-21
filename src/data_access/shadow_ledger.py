"""Persistencia y liquidación de carteras sombra de Melate Retro."""

from __future__ import annotations

from datetime import datetime
import json
import os
from typing import Any

from rich import box
from rich.console import Console
from rich.table import Table

from src.core.rules import MelateRetroRules
from src.core.shadow_promotion import evaluate_shadow_promotions
from src.data_access.config import (
    FILE_CARTERAS_SOMBRA,
    FILE_TABLERO_SOMBRA,
    VERSION_TAG,
)


console = Console()
SCHEMA_VERSION = 1
VALIDATION_TARGET_DRAWS = 20


def _empty_ledger() -> dict[str, Any]:
    return {"schema_version": SCHEMA_VERSION, "portfolios": []}


def _load_ledger(path: str = FILE_CARTERAS_SOMBRA) -> dict[str, Any]:
    if not os.path.exists(path):
        return _empty_ledger()
    with open(path, mode="r", encoding="utf-8") as source:
        payload = json.load(source)
    if not isinstance(payload, dict) or not isinstance(payload.get("portfolios"), list):
        raise ValueError("El ledger de carteras sombra tiene un formato inválido")
    return payload


def _atomic_save(payload: dict[str, Any], path: str = FILE_CARTERAS_SOMBRA) -> None:
    directory = os.path.dirname(path)
    os.makedirs(directory, exist_ok=True)
    temporary_path = f"{path}.tmp"
    with open(temporary_path, mode="w", encoding="utf-8") as target:
        json.dump(payload, target, ensure_ascii=False, indent=2)
        target.flush()
        os.fsync(target.fileno())
    os.replace(temporary_path, path)


def _clean_tickets(tickets) -> list[list[int]]:
    cleaned = []
    for ticket in tickets or []:
        values = sorted(int(number) for number in ticket[:6])
        if len(values) != 6:
            raise ValueError("Cada boleto sombra debe contener exactamente 6 números")
        cleaned.append(values)
    return cleaned


def _compact_metadata(metadata: dict[str, Any] | None) -> dict[str, Any]:
    source = metadata if isinstance(metadata, dict) else {}
    keys = (
        "ai_signal_enabled",
        "ai_signal_validated",
        "temporal_holdout_auc",
        "feature_schema",
        "number_model_enabled",
        "number_model_applied",
        "number_temporal_holdout_auc",
        "ai_context_weight",
        "ai_number_weight",
        "resonance_blend_mode",
        "hybrid_alpha",
        "hybrid_beta",
        "fitness_focus_max_rank",
        "fitness_candidate_max_rank",
        "fitness_rank_edges",
        "fitness_bucket_plan",
        "selector_debug_ranks",
        "selected_ranks",
        "shadow_family",
        "promotion_reference_key",
        "candidate_method",
        "candidate_pool_size",
        "candidate_rank_depth",
        "candidate_numbers",
        "ticket_budget",
        "ticket_count",
        "coverage_algorithm",
        "target_subset_sizes",
        "target_weights",
        "local_search_iterations",
        "coverage_by_t",
        "weighted_coverage",
        "universe_variant",
        "raw_universe_size",
        "universe_ticket_limit",
        "reduction_stage_stats",
        "sniper_mode",
        "sniper_candidates",
        "hard_excluded_numbers",
        "sniper_soft_numbers",
        "sniper_soft_penalty",
        "sniper_soft_candidate_count",
        "sniper_soft_reserve_target",
        "sniper_soft_reserve_actual",
        "sniper_soft_reserve_replacements",
    )
    compact = {}
    for key in keys:
        value = source.get(key)
        if value is None:
            continue
        if hasattr(value, "tolist"):
            value = value.tolist()
        compact[key] = value
    return compact


def guardar_carteras_sombra(
    concurso_id: int,
    variants: list[dict[str, Any]],
    dataset_through_concurso: int,
    path: str = FILE_CARTERAS_SOMBRA,
) -> bool:
    """Guarda una comparación pre-sorteo sin tocar el ledger de apuestas reales."""

    payload = _load_ledger(path)
    contest = int(concurso_id)
    if any(int(row.get("concurso", -1)) == contest for row in payload["portfolios"]):
        return False

    stored_variants = []
    for variant in variants:
        tickets = _clean_tickets(variant.get("tickets"))
        if not tickets:
            raise ValueError(f"La variante {variant.get('key', '')} no generó boletos")
        stored_variants.append(
            {
                "key": str(variant["key"]),
                "label": str(variant.get("label", variant["key"])),
                "official": bool(variant.get("official", False)),
                "settings": dict(variant.get("settings") or {}),
                "tickets": tickets,
                "metadata": _compact_metadata(variant.get("metadata")),
                "status": "Pendiente",
                "validation": None,
            }
        )

    payload["portfolios"].append(
        {
            "concurso": contest,
            "generated_at": datetime.now().isoformat(timespec="seconds"),
            "dataset_through_concurso": int(dataset_through_concurso),
            "version": VERSION_TAG,
            "variants": stored_variants,
        }
    )
    _atomic_save(payload, path)
    return True


def _validate_variant(variant: dict[str, Any], winning_draw, rules) -> dict[str, Any]:
    distribution = {str(hits): 0 for hits in range(7)}
    ticket_results = []
    total_prize = 0.0
    max_hits = 0
    for ticket in variant.get("tickets", []):
        hits, additional = rules.validate_ticket(ticket, winning_draw)
        prize = float(rules.calculate_prize(hits, additional))
        distribution[str(hits)] += 1
        total_prize += prize
        max_hits = max(max_hits, int(hits))
        ticket_results.append(
            {
                "ticket": [int(number) for number in ticket],
                "hits": int(hits),
                "additional": bool(additional),
                "prize": prize,
            }
        )

    simulated_investment = len(ticket_results) * float(rules.ticket_cost)
    return {
        "validated_at": datetime.now().isoformat(timespec="seconds"),
        "winning_draw": [int(number) for number in winning_draw],
        "ticket_count": len(ticket_results),
        "hit_distribution": distribution,
        "max_hits": max_hits,
        "simulated_investment": simulated_investment,
        "simulated_prize": total_prize,
        "simulated_net": total_prize - simulated_investment,
        "ticket_results": ticket_results,
    }


def _aggregate(payload: dict[str, Any]) -> dict[str, dict[str, Any]]:
    summary: dict[str, dict[str, Any]] = {}
    for portfolio in payload.get("portfolios", []):
        for variant in portfolio.get("variants", []):
            validation = variant.get("validation")
            if not isinstance(validation, dict):
                continue
            key = str(variant.get("key", "unknown"))
            row = summary.setdefault(
                key,
                {
                    "key": key,
                    "label": str(variant.get("label", key)),
                    "official": bool(variant.get("official", False)),
                    "contests": 0,
                    "tickets": 0,
                    "hits_4": 0,
                    "hits_5": 0,
                    "hits_6": 0,
                    "max_hits_sum": 0,
                    "contests_ge_4": 0,
                    "contests_ge_5": 0,
                    "contests_eq_6": 0,
                    "simulated_investment": 0.0,
                    "simulated_prize": 0.0,
                    "simulated_net": 0.0,
                },
            )
            distribution = validation.get("hit_distribution", {})
            row["contests"] += 1
            row["tickets"] += int(validation.get("ticket_count", 0))
            row["hits_4"] += int(distribution.get("4", 0))
            row["hits_5"] += int(distribution.get("5", 0))
            row["hits_6"] += int(distribution.get("6", 0))
            max_hits = int(validation.get("max_hits", 0))
            row["max_hits_sum"] += max_hits
            row["contests_ge_4"] += int(max_hits >= 4)
            row["contests_ge_5"] += int(max_hits >= 5)
            row["contests_eq_6"] += int(max_hits == 6)
            row["simulated_investment"] += float(
                validation.get("simulated_investment", 0.0)
            )
            row["simulated_prize"] += float(validation.get("simulated_prize", 0.0))
            row["simulated_net"] += float(validation.get("simulated_net", 0.0))
    for row in summary.values():
        contests = int(row["contests"])
        tickets = int(row["tickets"])
        high_hit_tickets = int(row["hits_4"] + row["hits_5"] + row["hits_6"])
        row["tickets_per_contest"] = (
            float(tickets / contests) if contests else 0.0
        )
        row["avg_max_hits"] = (
            float(row["max_hits_sum"] / contests) if contests else 0.0
        )
        row["contest_rate_ge_4"] = (
            float(row["contests_ge_4"] / contests) if contests else 0.0
        )
        row["contest_rate_ge_5"] = (
            float(row["contests_ge_5"] / contests) if contests else 0.0
        )
        row["high_hit_tickets_per_1000"] = (
            float(1000.0 * high_hit_tickets / tickets) if tickets else 0.0
        )
    return summary


def liquidar_carteras_sombra(
    history,
    path: str = FILE_CARTERAS_SOMBRA,
    dashboard_path: str | None = None,
) -> dict[str, Any] | None:
    """Valida carteras con resultados disponibles y devuelve el acumulado simulado."""

    if not os.path.exists(path):
        return None

    payload = _load_ledger(path)
    results = {
        str(int(contest)): numbers
        for contest, numbers in zip(history.concursos, history.winning_numbers)
    }
    rules = MelateRetroRules()
    updated_contests = []
    pending_contests = []
    changed = False

    for portfolio in payload.get("portfolios", []):
        contest = int(portfolio["concurso"])
        winning_draw = results.get(str(contest))
        if winning_draw is None:
            pending_contests.append(contest)
            continue
        contest_updated = False
        for variant in portfolio.get("variants", []):
            if variant.get("status") == "Validado" and isinstance(
                variant.get("validation"), dict
            ):
                continue
            variant["validation"] = _validate_variant(variant, winning_draw, rules)
            variant["status"] = "Validado"
            changed = True
            contest_updated = True
        if contest_updated:
            updated_contests.append(contest)

    if changed:
        _atomic_save(payload, path)

    variants = _aggregate(payload)
    promotion = evaluate_shadow_promotions(payload)
    summary = {
        "updated_contests": updated_contests,
        "pending_contests": sorted(set(pending_contests)),
        "target_draws": VALIDATION_TARGET_DRAWS,
        "variants": variants,
        "promotion": promotion,
    }
    resolved_dashboard_path = dashboard_path
    if resolved_dashboard_path is None:
        resolved_dashboard_path = (
            FILE_TABLERO_SOMBRA
            if os.path.abspath(path) == os.path.abspath(FILE_CARTERAS_SOMBRA)
            else os.path.join(os.path.dirname(path), "Tablero_Sombra.json")
        )
    export_payload = {
        "schema_version": 1,
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "source_ledger": os.path.basename(path),
        "version": VERSION_TAG,
        "target_draws": VALIDATION_TARGET_DRAWS,
        "pending_contests": summary["pending_contests"],
        "variants": variants,
        "promotion": promotion,
    }
    _atomic_save(export_payload, resolved_dashboard_path)
    summary["dashboard_path"] = resolved_dashboard_path
    return summary


def mostrar_resumen_sombra(summary: dict[str, Any] | None) -> None:
    if not summary:
        console.print("[dim]Aún no hay carteras sombra registradas.[/]")
        return

    variants = summary.get("variants", {})
    if not variants:
        pending = summary.get("pending_contests", [])
        if pending:
            contests = ", ".join(f"#{contest}" for contest in pending)
            console.print(f"[yellow]Carteras sombra pendientes: {contests}[/]")
        else:
            console.print("[dim]Aún no hay carteras sombra validadas.[/]")
        return

    table = Table(
        title="🌓 VALIDACIÓN PROSPECTIVA — CARTERAS SOMBRA",
        box=box.DOUBLE_EDGE,
    )
    table.add_column("Variante", style="cyan")
    table.add_column("Sorteos", justify="right")
    table.add_column("M/S", justify="right")
    table.add_column("Max prom.", justify="right")
    table.add_column("S≥4", justify="right")
    table.add_column("S≥5", justify="right")
    table.add_column("6/6", justify="right")
    table.add_column("≥4/1k", justify="right")
    table.add_column("Premio sim.", justify="right")
    table.add_column("Neto sim.", justify="right")

    for row in variants.values():
        label = row["label"] + (" [OFICIAL]" if row.get("official") else "")
        table.add_row(
            label,
            f"{row['contests']}/{summary.get('target_draws', VALIDATION_TARGET_DRAWS)}",
            f"{row['tickets_per_contest']:.0f}",
            f"{row['avg_max_hits']:.2f}",
            f"{row['contests_ge_4']} ({row['contest_rate_ge_4']:.1%})",
            f"{row['contests_ge_5']} ({row['contest_rate_ge_5']:.1%})",
            str(row["contests_eq_6"]),
            f"{row['high_hit_tickets_per_1000']:.2f}",
            f"${row['simulated_prize']:,.2f}",
            f"${row['simulated_net']:,.2f}",
        )
    console.print(table)
    console.print(
        "[dim]Los importes sombra son hipotéticos y no se suman al ROI real.[/]"
    )

    promotion = summary.get("promotion", {})
    evaluations = promotion.get("evaluations", {})
    if evaluations:
        promotion_table = Table(
            title="🚦 GATE DE PROMOCIÓN — COMPARACIONES PAREADAS MISMO PRESUPUESTO",
            box=box.DOUBLE_EDGE,
        )
        promotion_table.add_column("Variante", style="cyan")
        promotion_table.add_column("Referencia")
        promotion_table.add_column("N", justify="right")
        promotion_table.add_column("ΔMax", justify="right")
        promotion_table.add_column("Δ≥4", justify="right")
        promotion_table.add_column("Δ≥5", justify="right")
        promotion_table.add_column("IC95% ΔMax", justify="right")
        promotion_table.add_column("p", justify="right")
        promotion_table.add_column("Ventanas", justify="right")
        promotion_table.add_column("Estado")
        status_colors = {
            "REFERENCE": "bold blue",
            "BENCHMARK": "blue",
            "UNMATCHED_BUDGET": "bold red",
            "INSUFFICIENT_SAMPLE": "yellow",
            "COLLECTING": "yellow",
            "PROMISING": "bold cyan",
            "ELIGIBLE_FOR_PILOT": "bold green",
            "NO_ADVANTAGE": "magenta",
            "REJECTED": "bold red",
        }
        for evaluation in evaluations.values():
            bootstrap = evaluation.get("bootstrap_max_hits", {})
            permutation = evaluation.get("permutation_max_hits", {})
            stability = evaluation.get("temporal_stability", {})
            ci_low = bootstrap.get("ci_low")
            ci_high = bootstrap.get("ci_high")
            ci_text = (
                f"[{ci_low:+.3f}, {ci_high:+.3f}]"
                if ci_low is not None and ci_high is not None
                else "—"
            )
            p_value = permutation.get("p_two_sided")
            windows_text = (
                f"{stability.get('favorable_windows', 0)}/"
                f"{stability.get('total_windows', 0)}"
                if stability
                else "—"
            )
            status = str(evaluation.get("status", ""))
            style = status_colors.get(status, "white")
            promotion_table.add_row(
                str(evaluation.get("label", evaluation.get("key", ""))),
                str(evaluation.get("reference_label") or "—"),
                str(evaluation.get("paired_draws", 0)),
                _fmt_signed(evaluation.get("avg_max_hits_delta")),
                _fmt_percent_delta(evaluation.get("hit_rate_ge_4_delta")),
                _fmt_percent_delta(evaluation.get("hit_rate_ge_5_delta")),
                ci_text,
                f"{float(p_value):.4f}" if p_value is not None else "—",
                windows_text,
                f"[{style}]{status}[/]",
            )
        console.print(promotion_table)
        for evaluation in evaluations.values():
            if evaluation.get("status") in {"REFERENCE", "BENCHMARK"}:
                continue
            reasons = " ".join(str(value) for value in evaluation.get("reasons", []))
            console.print(
                f"[dim]• {evaluation.get('label', evaluation.get('key'))}: {reasons}[/]"
            )
        console.print(
            "[dim]El gate solo recomienda; nunca modifica producción automáticamente. "
            f"Exportado en {summary.get('dashboard_path', FILE_TABLERO_SOMBRA)}[/]"
        )


def _fmt_signed(value: Any) -> str:
    return f"{float(value):+.3f}" if value is not None else "—"


def _fmt_percent_delta(value: Any) -> str:
    return f"{float(value):+.1%}" if value is not None else "—"
