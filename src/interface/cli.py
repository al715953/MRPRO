# src/interface/cli.py

import os
import sys
from collections import Counter
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.text import Text
from rich.columns import Columns
from rich import box

from src.domain.dtos import PredictionResultDTO, DrawHistoryDTO
from src.data_access.config import VERSION_TAG
from src.core.health import get_model_status


def _ensure_utf8_console():
    """Evita errores con emojis/unicode en terminales Windows cp1252."""
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            try:
                stream.reconfigure(encoding="utf-8", errors="replace")
            except Exception:
                pass


_ensure_utf8_console()
console = Console()


class ConsoleUI:
    """
    Interface de Misión Crítica MRPRO V15.
    Maneja la visualización forense y operativa del sistema.
    """

    def __init__(self):
        self.console = console

    def clear_screen(self):
        os.system("cls" if os.name == "nt" else "clear")

    def show_status_bar(
        self, history: DrawHistoryDTO, tiene_apuestas: bool = False, profile=None
    ):
        """HUD (Heads-Up Display) Refinado."""
        ultimo_id = max(history.concursos)
        proximo_id = ultimo_id + 1
        game_label = profile.display_name if profile else "MRPRO"
        status_ledg = (
            "[white on red] 🔒 BLOQUEADO [/]"
            if tiene_apuestas
            else "[black on green] 🔓 LIBRE [/]"
        )

        grid = Table.grid(expand=True)
        grid.add_column(justify="left", ratio=1)
        grid.add_column(justify="center", ratio=2)
        grid.add_column(justify="right", ratio=1)

        grid.add_row(
            f"[bold blue]📦 {game_label}:[/] {ultimo_id}",
            f"[bold cyan]🎯 TARGET:[/] {proximo_id}  {status_ledg}",
            f"[dim]{VERSION_TAG}[/]",
        )

        self.console.print(Panel(grid, style="dim white", box=box.HORIZONTALS))

    def show_welcome(self):
        self.clear_screen()
        self.console.print(
            Text(
                " MRPRO TERMINAL :: SISTEMA DE PREDICCIÓN ESTRATÉGICA ",
                style="bold cyan",
            ),
            justify="left",
        )

    def show_main_menu(self, profile=None) -> str:
        if profile and profile.code == "tris_multiplicador":
            return self._show_main_menu_tris()
        return self._show_main_menu_melate()

    def _show_main_menu_melate(self) -> str:
        dias, color = get_model_status()
        status_brain = f"[{color}]({dias})[/]"

        menu_table = Table(box=None, show_header=False, padding=(0, 2))
        menu_table.add_column("ID", style="bold cyan", width=4)
        menu_table.add_column("Acción", style="white")
        menu_table.add_column("ID ", style="bold cyan", width=4)
        menu_table.add_column("Acción ", style="white")

        # Capa 1: Inteligencia
        menu_table.add_row(
            "[dim]--[/]", "[dim]INTELIGENCIA[/]", "[dim]--[/]", "[dim]DATA PREP[/]"
        )

        menu_table.add_row(
            "1", "Ver Historial", "4", f"Reentrenar Cerebro {status_brain}"
        )
        menu_table.add_row("2", "Análisis Frecuencia", "5", "Sincronizar Scraper")
        menu_table.add_row("3", "Optimizador (Lab)", "", "")
        menu_table.add_row("P", "Reporte Plot (Forense)", "", "")
        menu_table.add_row("C", "Lab Covering Design", "", "")

        # Capa 2: Operaciones
        menu_table.add_row("", "", "", "")
        menu_table.add_row(
            "[dim]--[/]",
            "[bold green]OPERACIONES [/]",
            "[dim]--[/]",
            "[dim]SISTEMA[/]",
        )
        menu_table.add_row("6", "Lab Backtest (Simulación)", "0", "Finalizar Sesión")
        menu_table.add_row("7", "[bold green]EJECUTAR OMEGA STRIDE[/]", "", "")
        menu_table.add_row(
            "8", "[bold yellow]LIQUIDAR ROI & TABLEROS SOMBRA[/]", "", ""
        )

        self.console.print(menu_table)
        return self.console.input(f"\n[bold cyan]MRPRO[/] > ")

    def _show_main_menu_tris(self) -> str:
        menu_table = Table(box=None, show_header=False, padding=(0, 2))
        menu_table.add_column("ID", style="bold cyan", width=4)
        menu_table.add_column("Accion", style="white")
        menu_table.add_column("ID ", style="bold cyan", width=4)
        menu_table.add_column("Accion ", style="white")

        menu_table.add_row(
            "[dim]--[/]", "[dim]INTELIGENCIA[/]", "[dim]--[/]", "[dim]DATOS[/]"
        )
        menu_table.add_row("1", "Ver Historial", "4", "Calibrador Tris [yellow](beta)[/]")
        menu_table.add_row("2", "Analisis Frecuencia", "5", "Sincronizar Historico")
        menu_table.add_row("3", "Optimizador Tris [yellow](beta)[/]", "", "")
        menu_table.add_row("P", "Plot Forense [yellow](beta)[/]", "", "")

        menu_table.add_row("", "", "", "")
        menu_table.add_row(
            "[dim]--[/]",
            "[bold green]OPERACIONES[/]",
            "[dim]--[/]",
            "[dim]SISTEMA[/]",
        )
        menu_table.add_row("6", "Backtest Tris", "0", "Finalizar Sesion")
        menu_table.add_row("7", "[bold green]EJECUTAR MODO TRIS[/]", "", "")
        menu_table.add_row("8", "[bold yellow]LIQUIDAR CARTERA TRIS [yellow](beta)[/][/]", "", "")

        self.console.print(menu_table)
        return self.console.input(f"\n[bold cyan]TRIS[/] > ")

    def show_history(self, history: DrawHistoryDTO, profile=None):
        """Visualización compacta del historial (Opción 1)."""
        is_tris = bool(profile and profile.code == "tris_multiplicador")
        title = "HISTORIAL TRIS RECIENTE" if is_tris else "HISTORIAL RECIENTE"
        table = Table(
            title=title, box=box.SIMPLE, header_style="bold blue"
        )
        table.add_column("Concurso", justify="center")
        table.add_column("Fecha", justify="center")
        table.add_column("Combinacion Ganadora", justify="center")
        if is_tris:
            table.add_column("Multiplicador", justify="center")

        # ZIP de datos y ordenamiento por ID descendente
        data = sorted(
            zip(history.concursos, history.dates, history.winning_numbers),
            key=lambda x: x[0],
            reverse=True,
        )[
            :15
        ]  # Mostramos los últimos 15

        for conc, date, nums in data:
            date_text = self._format_date_only(date)
            if is_tris:
                nums_str = "".join(str(int(n)) for n in nums[:5])
                has_multiplier = bool(nums[5]) if len(nums) > 5 else False
                mult_text = (
                    "[bold green]SI[/]" if has_multiplier else "[bold red]NO[/]"
                )
                table.add_row(str(conc), date_text, nums_str, mult_text)
            else:
                nums_str = "-".join(f"{n:02d}" for n in nums[:6])
                if len(nums) > 6:
                    nums_str += f" [bold yellow]({nums[6]:02d})[/]"
                table.add_row(str(conc), date_text, nums_str)

        self.console.print(table)

    def _format_date_only(self, value) -> str:
        if hasattr(value, "strftime"):
            try:
                return value.strftime("%d/%m/%Y")
            except Exception:
                pass
        text = str(value)
        if " " in text:
            text = text.split(" ", 1)[0]
        if "T" in text:
            text = text.split("T", 1)[0]
        return text

    def show_frequency_analysis(self, history: DrawHistoryDTO, profile=None):
        """Dashboard de frecuencias Hot/Cold (Opción 2)."""
        is_tris = bool(profile and profile.code == "tris_multiplicador")
        limit = 5 if is_tris else 6
        all_nums = [n for draw in history.winning_numbers for n in draw[:limit]]
        counts = Counter(all_nums)

        # Asegurar cobertura completa de dominio por juego.
        number_range = range(0, 10) if is_tris else range(1, 40)
        for n in number_range:
            if n not in counts:
                counts[n] = 0

        hot_title = "HOT DIGITOS" if is_tris else "HOT (Frecuentes)"
        cold_title = "COLD DIGITOS" if is_tris else "COLD (Rezagados)"

        hot_table = Table(title=f"🔥 {hot_title}", box=box.SIMPLE)
        hot_table.add_column("Num", style="bold yellow")
        hot_table.add_column("Hits", justify="right")

        cold_table = Table(title=f"❄️ {cold_title}", box=box.SIMPLE)
        cold_table.add_column("Num", style="bold blue")
        cold_table.add_column("Hits", justify="right")

        for num, freq in counts.most_common(5):
            hot_table.add_row(f"{num:02d}", str(freq))

        sorted_cold = sorted(counts.items(), key=lambda x: x[1])
        for num, freq in sorted_cold[:5]:
            cold_table.add_row(f"{num:02d}", str(freq))

        self.console.print(Columns([hot_table, cold_table]))

    def show_prediction_results(self, result: PredictionResultDTO):
        """Visualización de tickets generados (Opción 7)."""
        self.console.print(
            f"\n[bold green]🎫 SELECCIÓN ESTRATÉGICA: {result.strategy_name}[/bold green]"
        )

        if not result.tickets:
            self.console.print(
                Panel("❌ [bold red]FALLO DE GENERACIÓN[/]", border_style="red")
            )
            return

        ranks = result.metadata.get("selected_ranks", [])

        table = Table(box=box.ROUNDED, header_style="bold magenta")
        table.add_column("#", justify="right", style="dim")
        table.add_column("Combinación Sugerida", justify="center")
        table.add_column("Zona", justify="center")
        table.add_column("Rank", justify="right")

        for i, ticket in enumerate(result.tickets):
            t_str = " ".join(
                [f"[bold black on white] {n:02d} [/]" for n in sorted(ticket)]
            )

            # Lógica de Zona (Nucleus vs Stride)
            current_rank = ranks[i] if i < len(ranks) else "?"
            if isinstance(current_rank, int) and current_rank <= 10:
                zona, style = "NUCLEUS", "bold cyan"
            else:
                zona, style = "STRIDE", "blue"

            table.add_row(
                f"{(i+1):02d}",
                t_str,
                f"[{style}]{zona}[/]",
                f"[{style}]#{current_rank}[/]",
            )

        self.console.print(table)
