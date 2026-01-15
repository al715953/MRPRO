from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.text import Text
from rich import box

console = Console()


class SniperReport:
    """Reportero V32: Transparencia de Captura Real y Semántica Cromática."""

    @staticmethod
    def render_global_header(config_settings: dict, num_tickets: int):
        header = Text.assemble(
            ("🚀 MRPRO SNIPER V32 ", "bold cyan"),
            ("| ", "white"),
            (f"TICKETS/SORTEO: {num_tickets} ", "bold yellow"),
            ("| ", "white"),
            (f"MODO: Backtest de Realidad", "magenta"),
        )
        console.print(Panel(header, style="blue", expand=False))

    @staticmethod
    def render_phase1_summary(target_id: int, ac_survivors: int, final_univ: int):
        msg = Text.assemble(
            (f"#{target_id:4d} ", "bold white"),
            ("| ", "white"),
            ("🔍 FASE 1: ", "dim cyan"),
            (f"Pool AC: {ac_survivors:,} ", "dim green"),
            ("| ", "dim"),
            (f"Univ. Final: {final_univ:,}", "bold cyan"),
        )
        console.print(msg)

    @staticmethod
    def render_draw_summary(metadata: dict, audit_data: dict):
        found = audit_data.get("found", False)
        hits = audit_data.get("hits", 0)
        captured = audit_data.get("actually_captured", False)

        # 1. ESCALA CROMÁTICA DE HITS (Potencial del Universo)
        if hits == 6:
            type_tag = (" 6/6 ", "bold green")
        elif hits == 5:
            type_tag = (" 5/6 ", "bold blue")
        elif hits == 4:
            type_tag = (" 4/6 ", "bold yellow")
        else:
            type_tag = (f" {hits}/6 " if hits > 0 else " <4/6 ", "bold red")

        # 2. MÉTRICAS Y STATUS REAL
        if not found and hits < 4:
            status_text = Text("FUERA F1", style="bold red")
            metrics = Text("AI: ---- | Geo: ----", style="dim")
            rank_display = Text("")
        else:
            ai, geo = audit_data.get("ai_score", 0), audit_data.get("geo_score", 0)
            rank, pctl = audit_data.get("rank", 0), audit_data.get("percentile", 0)

            # EL VEREDICTO: Solo CAPTURED si está en los tickets comprados
            if captured:
                status_text = Text("✅ CAPTURED", style="bold green")
            else:
                status_text = Text("❌ MISSED", style="bold red")

            metrics = Text.assemble(
                ("AI: ", ""),
                (f"{ai:.4f} ", "yellow"),
                ("| Geo: ", ""),
                (f"{geo:.4f}", "yellow"),
            )
            rank_display = Text(f" | Rank: #{rank:,} (P{pctl:.1f})", style="dim cyan")

        line = Text.assemble(
            ("       | ", "white"),
            (f"AI Cut: {metadata.get('ai_threshold', 0):.2f} ", "cyan"),
            ("|", "white"),
            type_tag,
            ("| ", "white"),
            metrics,
            rank_display,
            (" | ", "white"),
            status_text,
        )
        console.print(line)

    @staticmethod
    def render_final_dashboard(size, invest, earn, funnel, dist):
        balance = earn - invest
        color_bal = "green" if balance >= 0 else "red"
        console.print(f"\n[bold yellow]{'='*85}[/]")

        dist_table = Table(
            title="[bold cyan]🎯 PREMIOS REALES COBRADOS[/]", box=box.SIMPLE
        )
        dist_table.add_column("Hits"), dist_table.add_column(
            "Cantidad", justify="right"
        )
        for h in range(6, 2, -1):  # Mostramos solo premios cobrables
            count = dist.get(h, 0)
            dist_table.add_row(
                f"{h} hits", f"[{'green' if h >= 5 else 'yellow'}]{count}[/]"
            )

        kpi_table = Table(
            title="[bold magenta]KPI DE EFECTIVIDAD SNIPER[/]", box=box.ROUNDED
        )
        kpi_table.add_column("Métrica"), kpi_table.add_column("Valor", justify="right")
        kpi_table.add_row(
            "Recall Univ. (4+)",
            f"{funnel['opp_gold'] + funnel['opp_silver']}/{funnel['total_draws']}",
        )
        kpi_table.add_row(
            "Premios Atrapados",
            f"{funnel['captured_gold'] + funnel['captured_silver']}",
            style="bold green",
        )
        kpi_table.add_row("Balance Neto", f"${balance:,.2f}", style=f"bold {color_bal}")

        console.print(dist_table)
        console.print(kpi_table)
