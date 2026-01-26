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
from src.data_access.config import VERSION_TAG


class BacktestEngine:
    """Motor Sniper V14.9: Integración de Telemetría y Gestión de Memoria."""

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
        """
        Ejecuta la misión de backtest con arquitectura desacoplada.
        Delegación total de IA a la estrategia y validación financiera estricta.
        """
        total_inv, total_earn, coverage_6 = 0.0, 0.0, 0
        hits_dist = {i: 0 for i in range(7)}  # Registro de impacto 0/6 a 6/6

        # Preparación del historial cronológico
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
            disable=not verbose,
        ) as progress:
            task = progress.add_task("Misión", total=test_size)

            for i in range(start_idx, len(full_h)):
                t_start = time.time()
                _, target, t_id = full_h[i]  # Sorteo real a batir

                # Ventana de datos históricos hasta el sorteo actual
                past = full_h[:i]
                d_past, n_past, ids_past = zip(*past)
                curr_h = DrawHistoryDTO(list(d_past), list(n_past), list(ids_past))

                # --- FASE 1: REDUCCIÓN (Handshake V14.1) ---
                if pre_process_strategy:
                    res_univ = pre_process_strategy.predict(
                        curr_h, config, verbose=False
                    )
                    # Sincronización del puntero físico para la Fase 2
                    config.raw_universe_ptr = res_univ.metadata.get("raw_ndarray")

                    # Radar de Cobertura (Jackpot Tracker)
                    if config.raw_universe_ptr is not None:
                        xp = (
                            cp.get_array_module(config.raw_universe_ptr)
                            if HAS_CUPY
                            else np
                        )
                        t_xp = xp.asarray(target[:6], dtype=xp.uint8)
                        if (
                            int(
                                xp.max(
                                    xp.sum(
                                        xp.isin(config.raw_universe_ptr, t_xp), axis=1
                                    )
                                )
                            )
                            == 6
                        ):
                            coverage_6 += 1

                # --- FASE 2: ESTRATEGIA (IA & Selección) ---
                # El entrenamiento ahora es interno: strategy se auto-gestiona
                prediction = strategy.predict(curr_h, config)

                # Auditoría Forense: Rank y Distancia
                audit = (
                    strategy.audit_winner(curr_h, config, target)
                    if hasattr(strategy, "audit_winner")
                    else {}
                )

                if audit:
                    audit["draw_id"] = int(t_id)
                    self.forensic_data.append(audit)

                # --- FASE 3: VALIDACIÓN FINANCIERA ---
                for tkt in prediction.tickets:
                    total_inv += self.rules.ticket_cost
                    h_n, h_a = self.rules.validate_ticket(
                        tkt, target
                    )  # Conteo de hits reales
                    total_earn += self.rules.calculate_prize(h_n, h_a)
                    hits_dist[h_n] += 1

                # Telemetría en tiempo real
                if verbose and audit:
                    self._render_telemetry(audit, t_id, t_start)

                # Gestión de memoria GPU (RTX 4070 Ti)
                if HAS_CUPY:
                    cp.get_default_memory_pool().free_all_blocks()

                progress.advance(task)

        # Empaquetado de resultados finales
        res = BacktestResultDTO(
            f"Sniper Global (Dynamic Hybrid)",
            test_size,
            total_inv,
            total_earn,
            total_earn - total_inv,
            hits_dist,
        )

        self._print_final_report(res, coverage_6)
        self.tracker.log_run(res, VERSION_TAG, self.forensic_data)
        return res

    def _render_telemetry(self, audit, t_id, t_s):
        """Renderizado con Geo-Score integrado y paleta Sniper."""
        d, r, h = (
            audit.get("proximity", 999),
            audit.get("rank", 0),
            audit.get("hits", 0),
        )
        ai_s = audit.get("ai_score", 0.0)
        geo_s = audit.get("geo_score", 0.0)  # Recuperamos el Geo
        u_s = audit.get("univ_size", 0)

        st_c = "bold green" if d == 0 else "bold red"
        h_c = "bold yellow" if h >= 5 else "cyan" if h == 4 else "white"
        d_c = "bold green" if d == 0 else "bold yellow" if d < 50 else "white"

        status = "🎯 HIT" if d == 0 else "❌"

        # Formato de log extendido con Geo
        self.console.print(
            f"[bold blue]#{t_id}[/] | "
            f"U: {u_s:>6,d} | "
            f"[{h_c}]{h}/6[/] | "
            f"AI: [bold yellow]{ai_s:.4f}[/] | "
            f"Geo: [bold cyan]{geo_s:.4f}[/] | "
            f"Rank: #{r:<5} | "
            f"Dist: [{d_c}]{d:<4}[/] | "
            f"[{st_c}]{status}[/] | [dim]{time.time()-t_s:.2f}s[/dim]"
        )

    def _print_final_report(self, res, coverage_6):
        """Reporte sin cuadros negros: Encabezados definidos."""
        self.console.print("\n[bold green]📊 REPORTE FINAL DE MISIÓN[/bold green]")

        # Tabla de Resumen Financiero
        summary = Table(show_header=True, header_style="bold magenta")
        summary.add_column("Métrica Sniper", style="dim", width=20)
        summary.add_column("Valor", justify="right", width=15)

        summary.add_row("Sorteos Analizados", str(res.total_draws_tested))
        summary.add_row(
            "Balance Neto",
            f"[{'green' if res.net_balance >= 0 else 'red'}]${res.net_balance:,.2f}[/]",
        )
        summary.add_row("Jackpots en Universo", f"[bold yellow]{coverage_6}[/]")
        self.console.print(summary)

        # Tabla de Distribución de Aciertos
        dist_table = Table(
            title="Distribución de Aciertos", show_header=True, header_style="bold cyan"
        )
        dist_table.add_column("Rango", justify="center")
        dist_table.add_column("Tickets", justify="right")

        for h in range(7):
            count = res.hit_distribution.get(h, 0)
            style = "bold yellow" if h >= 4 else "white"
            dist_table.add_row(f"{h}/6 aciertos", f"[{style}]{count}[/]")

        self.console.print(dist_table)
