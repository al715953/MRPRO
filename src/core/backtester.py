# src/core/backtester.py

import time
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
from src.data_access.config import VERSION_TAG

class BacktestEngine:
    """Motor Sniper V14.10: Full Data Capture (Visual + CSV)."""

    def __init__(self):
        self.rules, self.console = MelateRetroRules(), Console()
        self.tracker, self.forensic_data = PerformanceTracker(), []

    def run(
        self,
        strategy,
        history: DrawHistoryDTO,
        config,
        verbose=True,
        pre_process_strategy=None,
    ):
        total_inv, total_earn, coverage_6 = 0.0, 0.0, 0
        hits_dist = {i: 0 for i in range(7)}
        reduced_sizes = []
        max_hits_by_draw = {4: 0, 5: 0, 6: 0}
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
                if pre_process_strategy:
                    res_univ = pre_process_strategy.predict(
                        curr_h, config, verbose=False
                    )
                    config.raw_universe_ptr = res_univ.metadata.get("raw_ndarray")
                    sniper_msg = res_univ.metadata.get("sniper_log", "")

                    if config.raw_universe_ptr is not None:
                        xp = (
                            cp
                            if (HAS_CUPY and hasattr(config.raw_universe_ptr, "get"))
                            else np
                        )
                        t_xp = xp.asarray(target[:6], dtype=xp.uint8)
                        matches = xp.sum(xp.isin(config.raw_universe_ptr, t_xp), axis=1)
                        if int(xp.max(matches)) == 6:
                            coverage_6 += 1

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
                    xp_audit = (
                        cp
                        if (HAS_CUPY and hasattr(config.raw_universe_ptr, "get"))
                        else np
                    )
                    audit = LotteryForensics.audit_winner(snapshot, target, xp_audit)

                if audit:
                    audit["draw_id"] = int(t_id)
                    
                    # 1. Guardamos el Tamaño del Universo
                    audit["univ_size"] = (
                        len(config.raw_universe_ptr)
                        if config.raw_universe_ptr is not None
                        else 0
                    )
                    
                    # 2. Guardamos el Log del Sniper (¡LA PIEZA FALTANTE!)
                    audit["sniper_log"] = sniper_msg
                    
                    self.forensic_data.append(audit)

                # --- FASE 3: VALIDACIÓN FINANCIERA ---
                max_hit_this_draw = 0
                high_hits_this_draw = {4: 0, 5: 0, 6: 0}
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
                        elapsed=time.time() - t_start,
                    )

                if verbose and audit and not is_reduction_only:
                    self._render_telemetry(audit, t_id, t_start, sniper_msg)

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
            self._print_reduction_summary(res, reduced_sizes, max_hits_by_draw)
        else:
            self._print_final_report(res, coverage_6)
        self.tracker.log_run(res, VERSION_TAG, self.forensic_data)
        return res

    def _render_telemetry(self, audit, t_id, t_s, sniper_msg=""):
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
            "bold green" if h == 6 else 
            "bold yellow" if h == 5 else 
            "cyan" if h == 4 else "white"
        )
        d_c = "bold green" if d == 0 else "bold yellow" if d < 50 else "white"
        status = "🎯 HIT" if d == 0 else "❌"
        
        line = (
            f"[bold blue]#{t_id}[/] | "
            f"U: {u_s:>6,d} | "
            f"[{h_c}]{h}/6[/] | "
            f"AI: [bold yellow]{ai_s:.4f}[/] | "
            f"Geo: [bold cyan]{geo_s:.4f}[/] | "
            f"Rank: #{r:<5} | "
            f"Dist: [{d_c}]{d:<4}[/] | "
            f"[{st_c}]{status}[/] | [dim]{time.time()-t_s:.2f}s[/dim]"
        )
        
        if sniper_msg:
            line += f" | [cyan]{sniper_msg}[/]"

        self.console.print(line)

    def _print_final_report(self, res, coverage_6):
        self.console.print("\n[bold green]📊 REPORTE FINAL DE MISIÓN[/bold green]")
        summary = Table(show_header=True, header_style="bold magenta")
        summary.add_column("Métrica Sniper", style="dim", width=20)
        summary.add_column("Valor", justify="right", width=15)
        summary.add_row("Sorteos Analizados", str(res.total_draws_tested))
        summary.add_row("Balance Neto", f"[{'green' if res.net_balance >= 0 else 'red'}]${res.net_balance:,.2f}[/]")
        summary.add_row("Jackpots en Universo", f"[bold yellow]{coverage_6}[/]")
        self.console.print(summary)
        
        dist_table = Table(title="Distribución de Aciertos", show_header=True, header_style="bold cyan")
        dist_table.add_column("Rango", justify="center")
        dist_table.add_column("Tickets", justify="right")
        for h in range(7):
            count = res.hit_distribution.get(h, 0)
            style = "bold yellow" if h >= 4 else "white"
            dist_table.add_row(f"{h}/6 aciertos", f"[{style}]{count}[/]")
        self.console.print(dist_table)

    def _print_reduction_summary(self, res, reduced_sizes, max_hits_by_draw):
        final_universe = int(reduced_sizes[-1]) if reduced_sizes else 0
        hits_4 = int(max_hits_by_draw.get(4, 0))
        hits_5 = int(max_hits_by_draw.get(5, 0))
        hits_6 = int(max_hits_by_draw.get(6, 0))

        self.console.print("\n[bold green]📊 RESUMEN REDUCCIÓN DE UNIVERSO[/bold green]")
        summary = Table(show_header=True, header_style="bold magenta")
        summary.add_column("Métrica", style="dim", width=30)
        summary.add_column("Valor", justify="right", width=15)
        summary.add_row("Universo final reducido", f"[bold cyan]{final_universe:,}[/]")
        summary.add_row("Hits 4/6", f"[bold yellow]{hits_4}[/]")
        summary.add_row("Hits 5/6", f"[bold yellow]{hits_5}[/]")
        summary.add_row("Hits 6/6", f"[bold yellow]{hits_6}[/]")
        self.console.print(summary)

    def _render_reduction_telemetry(self, t_id, univ_size, max_hit, high_hits, elapsed):
        self.console.print(
            f"[bold blue]#{t_id}[/] | "
            f"U_final: [bold cyan]{int(univ_size):,}[/] | "
            f"Hits -> 4/6: [yellow]{high_hits.get(4, 0)}[/] "
            f"5/6: [yellow]{high_hits.get(5, 0)}[/] "
            f"6/6: [yellow]{high_hits.get(6, 0)}[/] | "
            f"Max: [bold]{int(max_hit)}/6[/] | "
            f"[dim]{elapsed:.2f}s[/dim]"
        )
