import os
import time
from typing import Tuple
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.text import Text
from rich import box
from rich.align import Align
from rich.columns import Columns
from collections import Counter

from src.domain.dtos import PredictionResultDTO, DrawHistoryDTO

# Instancia global de consola
console = Console()


class ConsoleUI:
    """
    Interfaz de Usuario por Consola (CLI) potenciada con Rich.
    Maneja toda la entrada/salida visual del sistema MRPRO.
    """

    def __init__(self):
        self.console = console

    def clear_screen(self):
        os.system("cls" if os.name == "nt" else "clear")

    def show_welcome(self):
        self.clear_screen()

        # Título Estilizado
        title_text = Text(
            "🎱 MRPRO SYSTEM V4", style="bold white on blue", justify="center"
        )
        subtitle = Text(
            "Clean Architecture | AI-Powered | Numpy Accelerated", style="cyan"
        )

        panel = Panel(
            Align.center(subtitle),
            title=title_text,
            border_style="blue",
            padding=(1, 2),
        )
        self.console.print(panel)
        self.console.print("\n")

    def show_main_menu(self) -> str:
        """Menú principal con tabla invisible para alineación."""
        menu_table = Table(show_header=False, box=None, padding=(0, 2))
        menu_table.add_column("Opción", style="bold cyan", justify="right")
        menu_table.add_column("Descripción", style="white")

        menu_table.add_row("1.", "📜 Ver Historial de Sorteos")
        menu_table.add_row("2.", "📊 Análisis de Frecuencia (Hot/Cold)")
        menu_table.add_row("3.", "🎲 Simulación Monte Carlo (Baseline)")
        menu_table.add_row("4.", "🧠 Optimizador de Estrategia (Laboratorio)")
        menu_table.add_row("5.", "🌌 Generar Universo Reducido (Fase 1)")
        menu_table.add_row("6.", "📡 Laboratorio de Pruebas (Backtest & QA)")
        menu_table.add_row("7.", "🎯 SELECTOR GENÉTICO FINAL (Producción)")
        menu_table.add_row("", "")
        menu_table.add_row("0.", "🚪 Salir")

        panel = Panel(
            menu_table,
            title="[bold yellow]Menú Principal[/bold yellow]",
            border_style="yellow",
            expand=False,
        )
        self.console.print(panel)
        return self.console.input(
            "\n[bold green]>> Tu orden, Arquitecto:[/bold green] "
        )

    def show_optimizer_menu(self) -> Tuple[str, int]:
        """
        Sub-menú específico para el Optimizador de Parámetros.
        Retorna: (Opción seleccionada, Cantidad de sorteos)
        """
        self.console.print(
            f"\n[bold magenta]🧠 OPTIMIZADOR DE ESTRATEGIA (LABORATORIO)[/]"
        )

        table = Table(show_header=False, box=box.SIMPLE, padding=(0, 2))
        table.add_row(
            "1.",
            "🔧 Solo Filtros (Topología Universo)",
            "[dim]Rápido. Ajusta Suma, Pares, Primos.[/]",
        )
        table.add_row(
            "2.",
            "⚖️  Solo Pesos (Sintonización Fina)",
            "[dim]Rápido. Ajusta Balanza IA vs Heurística.[/]",
        )
        table.add_row(
            "3.", "🧱 Solo Cuotas", "[dim]Alineación Táctica (E-M-L).[/]"
        )  # <--- NUEVO
        table.add_row("4.", "🚀 FULL STACK", "[bold cyan]Todo el pipeline.[/]")

        self.console.print(table)

        sub_opt = self.console.input("\n   👉 Selecciona modo (4): ") or "4"

        # Selección de Profundidad
        self.console.print(f"\n[bold cyan]📅 PROFUNDIDAD DEL ANÁLISIS:[/]")
        depth_table = Table(show_header=False, box=None, padding=(0, 2))
        depth_table.add_row("• [20]", "Sorteos", "Pruebas rápidas (Dev Mode)")
        depth_table.add_row("• [50]", "Sorteos", "Calibración Semestral (Standard)")
        depth_table.add_row(
            "• [108]", "Sorteos", "Calibración Anual (Hardcore - Lento)"
        )

        self.console.print(depth_table)

        try:
            n_draws_input = (
                self.console.input("   👉 Cantidad de sorteos a analizar (20): ")
                or "20"
            )
            n_draws = int(n_draws_input)
        except ValueError:
            n_draws = 20

        return sub_opt, n_draws

    def show_history(self, history: DrawHistoryDTO):
        self.console.print("\n[bold cyan]📜 ÚLTIMOS 10 SORTEOS REGISTRADOS[/bold cyan]")

        table = Table(box=box.SIMPLE_HEAD)
        table.add_column("Fecha", style="dim")
        table.add_column("Sorteo #", justify="right", style="cyan")
        table.add_column("Combinación Ganadora", justify="center", style="bold white")

        total = len(history.winning_numbers)
        start = max(0, total - 10)

        for i in range(start, total):
            date = str(history.dates[i]) if i < len(history.dates) else "??"
            concurso = str(history.concursos[i]) if i < len(history.concursos) else "??"
            nums = history.winning_numbers[i]
            # Formateo visual de las bolas
            nums_styled = "  ".join([f"[bold white on blue] {n:02d} [/]" for n in nums])

            table.add_row(date, concurso, nums_styled)

        self.console.print(table)

    def analyze_frequency(self, history: DrawHistoryDTO, total_balls: int):
        self.console.print(
            "\n[bold cyan]📊 ANÁLISIS DE FRECUENCIA (Top & Flop)[/bold cyan]"
        )

        all_nums = [n for draw in history.winning_numbers for n in draw[:6]]
        counts = Counter(all_nums)
        full_counts = {n: 0 for n in range(1, total_balls + 1)}
        full_counts.update(counts)

        # Crear tablas lado a lado
        hot_table = Table(
            title="🔥 Calientes (Top 5)", box=box.ROUNDED, border_style="red"
        )
        hot_table.add_column("Bola", justify="center", style="bold red")
        hot_table.add_column("Veces", justify="right")

        cold_table = Table(
            title="❄️ Fríos (Top 5)", box=box.ROUNDED, border_style="blue"
        )
        cold_table.add_column("Bola", justify="center", style="bold blue")
        cold_table.add_column("Veces", justify="right")

        # Llenar datos
        for num, freq in counts.most_common(5):
            hot_table.add_row(f"{num:02d}", str(freq))

        sorted_cold = sorted(full_counts.items(), key=lambda x: x[1])
        for num, freq in sorted_cold[:5]:
            cold_table.add_row(f"{num:02d}", str(freq))

        self.console.print(Columns([hot_table, cold_table]))

    def show_prediction_results(self, result: PredictionResultDTO):
        self.console.print(
            f"\n[bold green]🎫 RESULTADOS: {result.strategy_name}[/bold green]"
        )

        if not result.tickets:
            self.console.print(
                Panel(
                    "❌ [bold red]No se generaron tickets.[/] Relaja los filtros.",
                    border_style="red",
                )
            )
            return

        table = Table(box=box.DOUBLE_EDGE, show_lines=True)
        table.add_column("#", justify="right", style="dim")
        table.add_column("Ticket Sugerido", justify="center")

        for i, ticket in enumerate(result.tickets, 1):
            t_str = " ".join(
                [f"[bold black on white] {n:02d} [/]" for n in sorted(ticket)]
            )
            table.add_row(f"{i:02d}", t_str)

        self.console.print(table)
        self.console.print(f"[dim]Total generados: {len(result.tickets)}[/dim]")
        self.console.print("\n[bold green]🍀 ¡Buena suerte, Arquitecto![/bold green]")
