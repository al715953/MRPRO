from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.text import Text
from rich import box

console = Console()


class SniperReport:
    """Reportero V33.2: Sistema de Estatus por Impacto y Colores de Alta Fidelidad."""

    @staticmethod
    def render_global_header(config_settings: dict, num_tickets: int):
        header = Text.assemble(
            ("🚀 MRPRO SNIPER V9.8.4 ", "bold cyan"),
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
        """Versión V9.8.6: Colores de Potencial + Estatus de Captura Real."""
        hits = audit_data.get("hits", 0)
        prox = audit_data.get("proximity", 0)
        ai_val = audit_data.get("ai_score", 0)
        geo_val = audit_data.get("geo_score", 0)
        pctl = audit_data.get("percentile", 0)
        d_id = audit_data.get("draw_id", "####")
        univ_size = audit_data.get("univ_size", 0)

        # 1. COLOR DEL POTENCIAL (Siempre visible, basado en el universo)
        if hits == 6:
            potential_color = "bold cyan"  # Azul Diamante
        elif hits == 5:
            potential_color = "bold green"  # Verde
        elif hits == 4:
            potential_color = "bold yellow"  # Amarillito
        else:
            potential_color = "white"  # 3/6 o menos en blanco

        # 2. LÓGICA DE CAPTURA REAL (Estatus)
        is_hit = prox == 0
        if is_hit:
            if hits == 6:
                status = Text("💎 JACKPOT!!", style="bold cyan")
            elif hits == 5:
                status = Text("🔥 HIT (5/6)", style="bold green")
            else:
                status = Text("🎯 HIT (4/6)", style="bold yellow")
        else:
            status = Text("❌ MISSED", style="bold red")

        # 3. COLOR DE DISTANCIA (Resaltar "escoltas" cercanos)
        prox_color = (
            "bold cyan" if prox == 0 else "bold magenta" if prox < 15 else "dim"
        )

        line = Text.assemble(
            (f"#{d_id:4d} ", "bold white"),
            ("| ", "white"),
            (f"U: {univ_size:7,d} ", "dim cyan"),
            ("| ", "white"),
            (
                f" {hits}/6 ",
                potential_color,
            ),  # <--- Aquí devolvemos el color al potencial
            ("| ", "white"),
            ("AI: ", "dim"),
            (f"{ai_val:.4f} ", "yellow"),
            ("| Geo: ", "dim"),
            (f"{geo_val:.4f}", "yellow"),
            (f" | Rank: #{audit_data.get('rank', 0):,}", "cyan"),
            (f" (P{pctl:.1f}%) ", "dim"),
            (f" | Dist: {prox:,} ", prox_color),
            ("| ", "white"),
            status,
        )
        console.print(line)

    @staticmethod
    def render_final_dashboard(size, invest, earn, funnel, dist):
        # ... (Se mantiene igual que la versión anterior)
        balance = earn - invest
        color_bal = "green" if balance >= 0 else "red"

        dist_table = Table(
            title="[bold cyan]🎯 DISTRIBUCIÓN DE HITS REALES[/]",
            box=box.SIMPLE,
            expand=True,
        )
        dist_table.add_column("Hits"), dist_table.add_column(
            "Cantidad", justify="right"
        )
        for h in range(6, -1, -1):
            count = dist.get(h, 0)
            style = (
                "bold cyan"
                if h == 6
                else "bold green" if h == 5 else "bold yellow" if h == 4 else "dim"
            )
            dist_table.add_row(f"{h} hits", f"[{style}]{count}[/]")

        # (Resto de la tabla de eficiencia...)
        console.print(dist_table)
