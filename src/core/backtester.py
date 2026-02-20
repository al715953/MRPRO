# src/core/backtester.py

import time
import os
from uuid import uuid4
import numpy as np
from rich.console import Console
from rich.table import Table
from rich.progress import (
    Progress,
    SpinnerColumn,
    BarColumn,
    TextColumn,
    TimeElapsedColumn,
)

try:
    import cupy as cp

    HAS_CUPY = True
except ImportError:
    HAS_CUPY = False

from src.domain.dtos import DrawHistoryDTO, BacktestResultDTO
from src.core.rules import MelateRetroRules
from src.core.analytics import PerformanceTracker
from src.core.forensics import LotteryForensics
from src.data_access.config import VERSION_TAG, DATA_FOLDER, get_lottery_profile
from src.data_access.dataset_version import compute_dataset_version


class BacktestEngine:
    """Motor Sniper V14.10: Full Data Capture (Visual + CSV)."""

    def __init__(self, rules=None):
        self.rules = rules or MelateRetroRules()
        self.console = Console()
        self.tracker, self.forensic_data = PerformanceTracker(), []

    def _infer_profile_code(self, config, history: DrawHistoryDTO) -> str:
        overrides = (
            config.filter_overrides
            if hasattr(config, "filter_overrides")
            and isinstance(config.filter_overrides, dict)
            else {}
        )
        profile_code = overrides.get("profile_code")
        if profile_code:
            return str(profile_code)

        if config.ticket_size == 5 and getattr(config, "total_balls", None) == 10:
            return "tris_multiplicador"
        if config.ticket_size == 6 and getattr(config, "total_balls", None) == 39:
            return "melate_retro"

        if history.winning_numbers:
            first = history.winning_numbers[0]
            if len(first) <= 6 and config.ticket_size == 5:
                return "tris_multiplicador"

        return "melate_retro"

    def _build_tracking_context(self, config, history: DrawHistoryDTO, test_size: int):
        overrides = (
            config.filter_overrides
            if hasattr(config, "filter_overrides")
            and isinstance(config.filter_overrides, dict)
            else {}
        )
        profile_code = self._infer_profile_code(config, history)
        csv_path = ""
        try:
            profile = get_lottery_profile(profile_code)
            csv_path = os.path.join(DATA_FOLDER, profile.csv_filename)
        except Exception:
            csv_path = ""

        if csv_path:
            try:
                dataset_info = compute_dataset_version(csv_path)
            except Exception:
                dataset_info = {
                    "dataset_hash": "",
                    "row_count": 0,
                    "max_concurso": None,
                }
        else:
            dataset_info = {
                "dataset_hash": "",
                "row_count": 0,
                "max_concurso": None,
            }
        return {
            "event_id": str(uuid4()),
            "profile_code": profile_code,
            "dataset_hash": dataset_info.get("dataset_hash", ""),
            "seed": overrides.get("seed", ""),
            "split_id": f"bt_last_{test_size}",
        }

    def run(
        self,
        strategy,
        history: DrawHistoryDTO,
        config,
        verbose=True,
        pre_process_strategy=None,
    ):
        self.forensic_data = []
        total_inv, total_earn, jackpot_coverage = 0.0, 0.0, 0
        max_hits = getattr(self.rules, "max_hits", config.ticket_size)
        hits_dist = {i: 0 for i in range(max_hits + 1)}
        reduced_sizes = []
        high_hit_levels = list(range(max(0, max_hits - 2), max_hits + 1))
        max_hits_by_draw = {h: 0 for h in high_hit_levels}
        expects_universe_coverage = pre_process_strategy is not None
        has_universe_data = False

        # --- LOG: print once per run ---
        sniper_header_printed = False
        sniper_header_msg = ""
        # -----------------------------

        is_reduction_only = (
            pre_process_strategy is None
            and strategy.__class__.__name__ == "UniverseReductionStrategy"
        )

        full_h = sorted(
            zip(history.dates, history.winning_numbers, history.concursos),
            key=lambda x: x[2],
        )
        test_size = min(config.backtest_size, len(full_h))
        start_idx = len(full_h) - test_size
        tracking_ctx = self._build_tracking_context(config, history, test_size)
        strategy_model_version = getattr(strategy, "model_version", "")

        self.console.print(
            f"\n[bold magenta]🚀 INICIANDO MISIÓN ALPHA GLOBAL ({VERSION_TAG})[/bold magenta]"
        )

        with Progress(
            SpinnerColumn(),
            TextColumn("[bold blue]📡 Sniper Lab:[/][white] Analizando Malla...[/]"),
            BarColumn(bar_width=20),
            TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
            TimeElapsedColumn(),
            console=self.console,
            disable=(not verbose) or is_reduction_only,
        ) as progress:
            task = progress.add_task("Misión", total=test_size)

            for i in range(start_idx, len(full_h)):
                t_start = time.time()
                _, target, t_id = full_h[i]

                past = full_h[:i]
                d_past, n_past, ids_past = zip(*past)
                curr_h = DrawHistoryDTO(list(d_past), list(n_past), list(ids_past))

                # --- FASE 1: REDUCCIÓN ---
                sniper_msg = ""
                sniper_msg_for_line = ""  # <- no repetimos el log por sorteo
                if pre_process_strategy:
                    res_univ = pre_process_strategy.predict(
                        curr_h, config, verbose=False
                    )
                    config.raw_universe_ptr = res_univ.metadata.get("raw_ndarray")
                    sniper_msg = res_univ.metadata.get("sniper_log", "")

                    # --- LOG: imprimir solo 1 vez en toda la corrida ---
                    if verbose and (not sniper_header_printed) and sniper_msg:
                        sniper_header_msg = sniper_msg
                        self.console.print(
                            f"[cyan]🧷 SNIPER (run):[/] {sniper_header_msg}"
                        )
                        sniper_header_printed = True
                    # ---------------------------------------------------

                    if config.raw_universe_ptr is not None:
                        has_universe_data = True
                        xp = (
                            cp
                            if (HAS_CUPY and hasattr(config.raw_universe_ptr, "get"))
                            else np
                        )
                        universe_ptr = config.raw_universe_ptr[:, : config.ticket_size]
                        target_slice = target[: config.ticket_size]

                        if isinstance(self.rules, MelateRetroRules):
                            t_xp = xp.asarray(target_slice, dtype=xp.uint8)
                            matches = xp.sum(xp.isin(universe_ptr, t_xp), axis=1)
                            if int(xp.max(matches)) == max_hits:
                                jackpot_coverage += 1
                        else:
                            # Modo agnóstico: deferimos la semántica de acierto a las reglas del juego.
                            univ_cpu = (
                                universe_ptr.get()
                                if hasattr(universe_ptr, "get")
                                else np.asarray(universe_ptr)
                            )
                            best_hits = 0
                            for candidate in univ_cpu:
                                h_n, _ = self.rules.validate_ticket(
                                    candidate.tolist(), target
                                )
                                if h_n > best_hits:
                                    best_hits = h_n
                                if best_hits == max_hits:
                                    break
                            if best_hits == max_hits:
                                jackpot_coverage += 1

                # --- FASE 2: ESTRATEGIA ---
                if is_reduction_only:
                    prediction = strategy.predict(curr_h, config, verbose=False)
                else:
                    prediction = strategy.predict(curr_h, config)
                snapshot = prediction.metadata
                if is_reduction_only:
                    reduced_sizes.append(
                        int(snapshot.get("final_size", len(prediction.tickets)))
                    )

                # Auditoría Forense
                if is_reduction_only:
                    audit = None
                else:
                    audit_snapshot = dict(snapshot) if snapshot else {}
                    audit_snapshot["_pred_tickets"] = [
                        [int(x) for x in t]
                        for t in prediction.tickets
                    ]
                    xp_audit = (
                        cp
                        if (HAS_CUPY and hasattr(config.raw_universe_ptr, "get"))
                        else np
                    )
                    audit = LotteryForensics.audit_winner(
                        audit_snapshot, target, xp_audit
                    )

                if audit:
                    audit["draw_id"] = int(t_id)

                    # 1. Guardamos el Tamaño del Universo
                    audit["univ_size"] = (
                        len(config.raw_universe_ptr)
                        if config.raw_universe_ptr is not None
                        else len(prediction.tickets)
                    )

                    # 2. Guardamos el Log del Sniper (sin cambios; se guarda en CSV igual)
                    audit["sniper_log"] = sniper_msg
                    audit["event_id"] = tracking_ctx["event_id"]
                    audit["profile_code"] = tracking_ctx["profile_code"]
                    audit["dataset_hash"] = tracking_ctx["dataset_hash"]
                    audit["model_version"] = (
                        snapshot.get("model_version", strategy_model_version)
                        if isinstance(snapshot, dict)
                        else strategy_model_version
                    )
                    audit["seed"] = tracking_ctx["seed"]
                    audit["split_id"] = tracking_ctx["split_id"]
                    audit["metrics_json"] = {
                        "hits_pos": int(audit.get("hits", 0)),
                        "winner_in_topk": "",
                    }

                    self.forensic_data.append(audit)

                # --- FASE 3: VALIDACIÓN FINANCIERA ---
                max_hit_this_draw = 0
                high_hits_this_draw = {h: 0 for h in high_hit_levels}
                for tkt in prediction.tickets:
                    total_inv += self.rules.ticket_cost
                    h_n, h_a = self.rules.validate_ticket(tkt, target)
                    total_earn += self.rules.calculate_prize(h_n, h_a)
                    hits_dist[h_n] += 1
                    if h_n > max_hit_this_draw:
                        max_hit_this_draw = h_n
                    if h_n in high_hits_this_draw:
                        high_hits_this_draw[h_n] += 1

                if is_reduction_only and max_hit_this_draw in max_hits_by_draw:
                    max_hits_by_draw[max_hit_this_draw] += 1

                if verbose and is_reduction_only:
                    self._render_reduction_telemetry(
                        t_id=t_id,
                        univ_size=reduced_sizes[-1] if reduced_sizes else 0,
                        max_hit=max_hit_this_draw,
                        high_hits=high_hits_this_draw,
                        high_hit_levels=high_hit_levels,
                        max_hits=max_hits,
                        elapsed=time.time() - t_start,
                    )

                if verbose and audit and not is_reduction_only:
                    # <- log por sorteo apagado (solo se imprimió 1 vez arriba)
                    self._render_telemetry(
                        audit, t_id, t_start, max_hits, sniper_msg_for_line
                    )

                if HAS_CUPY:
                    cp.get_default_memory_pool().free_all_blocks()

                progress.advance(task)

        # Reporte Final
        res = BacktestResultDTO(
            f"Sniper Global (Dynamic Hybrid)",
            test_size,
            total_inv,
            total_earn,
            total_earn - total_inv,
            hits_dist,
        )
        if is_reduction_only:
            self._print_reduction_summary(
                res, reduced_sizes, max_hits_by_draw, max_hits
            )
        else:
            self._print_final_report(
                res,
                jackpot_coverage,
                max_hits,
                expects_universe_coverage,
                has_universe_data,
            )
        self.tracker.log_run(res, VERSION_TAG, self.forensic_data)
        return res

    def _render_telemetry(self, audit, t_id, t_s, max_hits, sniper_msg=""):
        d, r, h = (
            audit.get("proximity", 999),
            audit.get("rank", 0),
            audit.get("hits", 0),
        )
        ai_s = audit.get("ai_score", 0.0)
        geo_s = audit.get("geo_score", 0.0)
        u_s = audit.get("univ_size", 0)

        st_c = "bold green" if d == 0 else "bold red"
        h_c = (
            "bold green"
            if h == max_hits
            else "bold yellow"
            if h == max(0, max_hits - 1)
            else "cyan"
            if h == max(0, max_hits - 2)
            else "white"
        )
        d_c = "bold green" if d == 0 else "bold yellow" if d < 50 else "white"
        status = "🎯 HIT" if d == 0 else "❌"

        line = (
            f"[bold blue]#{t_id}[/] | "
            f"U: {u_s:>6,d} | "
            f"[{h_c}]{h}/{max_hits}[/] | "
            f"AI: [bold yellow]{ai_s:.4f}[/] | "
            f"Geo: [bold cyan]{geo_s:.4f}[/] | "
            f"Rank: #{r:<5} | "
            f"Dist: [{d_c}]{d:<4}[/] | "
            f"[{st_c}]{status}[/] | [dim]{time.time()-t_s:.2f}s[/dim]"
        )

        if sniper_msg:
            line += f" | [cyan]{sniper_msg}[/]"

        self.console.print(line)

    def _print_final_report(
        self, res, jackpot_coverage, max_hits, expects_universe_coverage, has_universe_data
    ):
        self.console.print("\n[bold green]📊 REPORTE FINAL DE MISIÓN[/bold green]")
        summary = Table(show_header=True, header_style="bold magenta")
        summary.add_column("Métrica Sniper", style="dim", width=20)
        summary.add_column("Valor", justify="right", width=15)
        summary.add_row("Sorteos Analizados", str(res.total_draws_tested))
        summary.add_row(
            "Balance Neto",
            f"[{'green' if res.net_balance >= 0 else 'red'}]${res.net_balance:,.2f}[/]",
        )
        if expects_universe_coverage and not has_universe_data:
            jackpot_value = "[bold yellow]N/A[/]"
        elif expects_universe_coverage:
            jackpot_value = f"[bold yellow]{jackpot_coverage}[/]"
        else:
            jackpot_value = "[bold yellow]0[/]"
        summary.add_row("Jackpots en Universo", jackpot_value)
        self.console.print(summary)

        dist_table = Table(
            title="Distribución de Aciertos",
            show_header=True,
            header_style="bold cyan",
        )
        dist_table.add_column("Rango", justify="center")
        dist_table.add_column("Tickets", justify="right")
        for h in range(max_hits + 1):
            count = res.hit_distribution.get(h, 0)
            style = "bold yellow" if h >= max(0, max_hits - 2) else "white"
            dist_table.add_row(f"{h}/{max_hits} aciertos", f"[{style}]{count}[/]")
        self.console.print(dist_table)

    def _print_reduction_summary(self, res, reduced_sizes, max_hits_by_draw, max_hits):
        final_universe = int(reduced_sizes[-1]) if reduced_sizes else 0
        sorted_levels = sorted(max_hits_by_draw.keys())

        self.console.print(
            "\n[bold green]📊 RESUMEN REDUCCIÓN DE UNIVERSO[/bold green]"
        )
        summary = Table(show_header=True, header_style="bold magenta")
        summary.add_column("Métrica", style="dim", width=30)
        summary.add_column("Valor", justify="right", width=15)
        summary.add_row("Universo final reducido", f"[bold cyan]{final_universe:,}[/]")
        for h in sorted_levels:
            summary.add_row(
                f"Hits {h}/{max_hits}",
                f"[bold yellow]{int(max_hits_by_draw.get(h, 0))}[/]",
            )
        self.console.print(summary)

    def _render_reduction_telemetry(
        self, t_id, univ_size, max_hit, high_hits, high_hit_levels, max_hits, elapsed
    ):
        hits_line = " ".join(
            [
                f"{h}/{max_hits}: [yellow]{high_hits.get(h, 0)}[/]"
                for h in high_hit_levels
            ]
        )
        self.console.print(
            f"[bold blue]#{t_id}[/] | "
            f"U_final: [bold cyan]{int(univ_size):,}[/] | "
            f"Hits -> {hits_line} | "
            f"Max: [bold]{int(max_hit)}/{max_hits}[/] | "
            f"[dim]{elapsed:.2f}s[/dim]"
        )
