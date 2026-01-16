from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.text import Text
from rich import box

console = Console()


class SniperReport:
    """Reportero V33.1: Restauración de Datos Forenses y Auditoría de Proximidad."""

    @staticmethod
    def render_global_header(config_settings: dict, num_tickets: int):
        header = Text.assemble(
            ("🚀 MRPRO SNIPER V33.1 ", "bold cyan"),
            ("| ", "white"),
            (f"PRECISION: {num_tickets} TICKETS ", "bold yellow"),
            ("| ", "white"),
            (
                f"AI CUT: {config_settings.get('threshold_ai_override', 0.72)}",
                "magenta",
            ),
        )
        console.print(Panel(header, style="blue", expand=False))

    @staticmethod
    def render_phase1_summary(target_id: int, ac_survivors: int, final_univ: int):
        msg = Text.assemble(
            (f"#{target_id:4d} ", "bold white"),
            ("| ", "white"),
            ("🔍 F1 (P88): ", "dim cyan"),
            (f"Univ: {final_univ:,}", "bold cyan"),
        )
        console.print(msg)

    @staticmethod
    def render_draw_summary(metadata: dict, audit_data: dict):
        hits = audit_data.get("hits", 0)
        captured = audit_data.get("actually_captured", False)
        prox = audit_data.get("proximity", 0)
        ai_val = audit_data.get("ai_score", 0)
        geo_val = audit_data.get("geo_score", 0)
        pctl = audit_data.get("percentile", 0)

        color = "green" if hits >= 5 else "yellow" if hits == 4 else "white"
        status = (
            Text("✅ CAPTURED", style="bold green")
            if captured
            else Text("❌ MISSED", style="bold red")
        )

        # Restauración de métricas de scoring para análisis de frontera
        metrics = Text.assemble(
            ("AI: ", "dim"),
            (f"{ai_val:.4f} ", "yellow"),
            ("| Geo: ", "dim"),
            (f"{geo_val:.4f}", "yellow"),
        )

        line = Text.assemble(
            ("       | ", "white"),
            (f" {hits}/6 ", f"bold {color}"),
            ("| ", "white"),
            metrics,
            (f" | Rank: #{audit_data.get('rank', 0):,}", "cyan"),
            (f" (P{pctl:.1f}%) ", "dim"),
            (f" | Dist: {prox:,} ", "bold magenta" if prox < 500 else "dim"),
            ("| ", "white"),
            status,
        )
        console.print(line)

    @staticmethod
    def render_final_dashboard(size, invest, earn, funnel, dist):
        balance = earn - invest
        color_bal = "green" if balance >= 0 else "red"

        dist_table = Table(
            title="[bold cyan]🎯 DISTRIBUCIÓN DE HITS REALES (20 TKT)[/]",
            box=box.SIMPLE,
            expand=True,
        )
        dist_table.add_column("Hits"), dist_table.add_column(
            "Cantidad", justify="right"
        )
        for h in range(6, -1, -1):
            count = dist.get(h, 0)
            style = "bold green" if h >= 5 else "yellow" if h == 4 else "dim"
            dist_table.add_row(f"{h} hits", f"[{style}]{count}[/]")

        univ_table = Table(
            title="[bold magenta]🌊 EFICIENCIA UNIVERSO (RECALL)[/]",
            box=box.ROUNDED,
            expand=True,
        )
        univ_table.add_column("Premio"), univ_table.add_column(
            "En Univ.", justify="right"
        ), univ_table.add_column("Recall %", justify="right")

        def fmt_rec(opp, cap):
            return f"{(cap/opp*100 if opp>0 else 0):.1f}%"

        univ_table.add_row(
            "JACKPOT (6/6)",
            str(funnel.get("opp_gold", 0)),
            fmt_rec(funnel.get("opp_gold", 0), funnel.get("captured_gold", 0)),
        )
        univ_table.add_row(
            "ORO (5/6)",
            str(funnel.get("opp_silver", 0)),
            fmt_rec(funnel.get("opp_silver", 0), funnel.get("captured_silver", 0)),
        )
        univ_table.add_row(
            "PLATA (4/6)",
            str(funnel.get("opp_bronze", 0)),
            fmt_rec(funnel.get("opp_bronze", 0), funnel.get("captured_bronze", 0)),
        )

        console.print(f"\n[bold yellow]{'='*85}[/]")
        console.print(dist_table)
        console.print(univ_table)

        fin_panel = Panel(
            Text.assemble(
                ("💰 RESUMEN FINANCIERO\n", "bold white underline"),
                (f"Inversión: ${invest:,.2f}\n", "white"),
                (f"Balance:   ", "white"),
                (f"${balance:,.2f}", f"bold {color_bal}"),
            ),
            border_style=color_bal,
            expand=False,
        )
        console.print(fin_panel)
