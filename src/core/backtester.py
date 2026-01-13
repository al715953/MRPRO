from rich.progress import (
    Progress,
    SpinnerColumn,
    TimeElapsedColumn,
    BarColumn,
    TextColumn,
)
from rich.console import Console
from rich.table import Table
from rich import box
from src.domain.interfaces import ILotteryStrategy
from src.domain.dtos import DrawHistoryDTO, PredictionConfigDTO, BacktestResultDTO
from src.core.rules import MelateRetroRules


class BacktestEngine:
    """
    Motor de simulación histórica con capacidad FORENSE.
    Incluye:
    - Radar de Cobertura (Fase 1).
    - Análisis comparativo (ADN del Ganador vs Predicciones).
    """

    def __init__(self):
        self.rules = MelateRetroRules()
        self.console = Console()

    def _calculate_dna(self, ticket):
        """Calcula propiedades estructurales rápidas para diagnóstico."""
        s = sum(ticket)
        evens = sum(1 for n in ticket if n % 2 == 0)
        primes_set = {2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37}
        primes = sum(1 for n in ticket if n in primes_set)
        return {"sum": s, "evens": evens, "primes": primes}

    def run(
        self,
        strategy: ILotteryStrategy,
        history: DrawHistoryDTO,
        config: PredictionConfigDTO,
        verbose: bool = True,
        pre_process_strategy: ILotteryStrategy = None,
        debug_deep: bool = False,  # <--- NUEVO MODO FORENSE
    ) -> BacktestResultDTO:

        strategy_name = strategy.__class__.__name__
        if verbose:
            self.console.print(
                f"\n[bold yellow]⚙️  Iniciando Backtest Forense para:[/bold yellow] [cyan]{strategy_name}[/cyan]"
            )

        total_investment = 0.0
        total_earnings = 0.0
        hits_distribution = {0: 0, 1: 0, 2: 0, 3: 0, 4: 0, 5: 0, 6: 0}

        # Stats cobertura
        coverage_stats = {
            "missed": 0,
            "captured_6": 0,
            "captured_5": 0,
            "captured_4": 0,
        }

        # Preparar historia
        full_history = list(
            zip(history.dates, history.winning_numbers, history.concursos)
        )
        full_history.sort(key=lambda x: x[2])

        total_draws = len(full_history)
        test_size = min(config.backtest_size, total_draws)
        start_index = total_draws - test_size

        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
            TimeElapsedColumn(),
            console=self.console,
            transient=True,  # Ocultar barra al terminar para dejar limpio el log
        ) as progress:

            task_id = progress.add_task(
                f"[cyan]Analizando {test_size} sorteos...", total=test_size
            )

            for i in range(start_index, total_draws):
                target_date, target_draw, target_id = full_history[i]
                target_set = set(target_draw[:6])

                # Contexto Histórico
                past_data = full_history[:i]
                if not past_data:
                    current_history = DrawHistoryDTO([], [], [])
                else:
                    p_dates, p_nums, p_ids = zip(*past_data)
                    current_history = DrawHistoryDTO(
                        list(p_dates), list(p_nums), list(p_ids)
                    )

                # --- FASE 1: RADAR DE COBERTURA ---
                universe_info = ""
                univ_hits_max = 0

                if pre_process_strategy:
                    # Silenciar verbose interno de la estrategia
                    old_ov = getattr(config, "filter_overrides", {})
                    config.filter_overrides = {**old_ov, "verbose": False}

                    univ_res = pre_process_strategy.predict(current_history, config)
                    config.filter_overrides = old_ov  # Restaurar

                    if univ_res.tickets:
                        hits_list = [len(set(t) & target_set) for t in univ_res.tickets]
                        univ_hits_max = max(hits_list) if hits_list else 0

                        # KPI Radar
                        if univ_hits_max == 6:
                            coverage_stats["captured_6"] += 1
                        elif univ_hits_max == 5:
                            coverage_stats["captured_5"] += 1
                        elif univ_hits_max == 4:
                            coverage_stats["captured_4"] += 1
                        else:
                            coverage_stats["missed"] += 1

                        icon = (
                            "💎"
                            if univ_hits_max == 6
                            else ("⚠️" if univ_hits_max == 5 else "❌")
                        )
                        universe_info = f"[{icon} Red: {univ_hits_max} Hits]"

                # --- FASE 2: ESTRATEGIA PRINCIPAL ---
                prediction = strategy.predict(current_history, config)

                # --- FASE 3: EVALUACIÓN Y FORENSE ---
                draw_earnings = 0.0
                max_hit = 0

                # Tabla Forense del Sorteo
                forensic_table = Table(
                    box=box.SIMPLE, show_header=True, header_style="bold magenta"
                )
                forensic_table.add_column("Ticket", width=24)
                forensic_table.add_column("Aciertos", justify="center")
                forensic_table.add_column("Premio", justify="right")
                forensic_table.add_column("Suma", justify="center")
                forensic_table.add_column("Pares", justify="center")
                forensic_table.add_column("Primos", justify="center")

                # 1. Analizar GANADOR REAL
                dna_winner = self._calculate_dna(target_draw[:6])
                w_str = ", ".join(f"{n:02d}" for n in sorted(target_draw[:6]))
                forensic_table.add_row(
                    f"[bold yellow]{w_str}[/]",
                    "🏆",
                    "GANADOR",
                    str(dna_winner["sum"]),
                    str(dna_winner["evens"]),
                    str(dna_winner["primes"]),
                    style="on black",
                )
                forensic_table.add_row("---", "-", "-", "-", "-", "-")

                # 2. Analizar PREDICCIONES
                for t_idx, ticket in enumerate(prediction.tickets, 1):
                    total_investment += self.rules.ticket_cost
                    hits_nat, has_add = self.rules.validate_ticket(ticket, target_draw)
                    prize = self.rules.calculate_prize(hits_nat, has_add)

                    draw_earnings += prize
                    hits_distribution[hits_nat] = hits_distribution.get(hits_nat, 0) + 1
                    max_hit = max(max_hit, hits_nat)

                    # DNA predicción
                    dna = self._calculate_dna(ticket)
                    t_str = ", ".join(f"{n:02d}" for n in sorted(ticket))

                    # Estilo visual según aciertos
                    style = "dim"
                    hit_str = str(hits_nat)
                    if hits_nat >= 3:
                        style = "bold green"
                        hit_str = f"★ {hits_nat}"
                    if hits_nat >= 4:
                        style = "bold cyan"

                    forensic_table.add_row(
                        f"[{style}]{t_str}[/]",
                        f"[{style}]{hit_str}[/]",
                        f"[{style}]${prize:,.0f}[/]" if prize > 0 else "",
                        str(dna["sum"]),
                        str(dna["evens"]),
                        str(dna["primes"]),
                    )

                total_earnings += draw_earnings

                # --- IMPRESIÓN CONDICIONAL ---
                # Si debug_deep está activo, mostramos la autopsia SIEMPRE,
                # o si hubo premio relevante.
                should_print = debug_deep or (verbose and draw_earnings > 0)

                if should_print:
                    progress.console.print(
                        f"\n[bold white]🔎 SORTEO #{target_id} ({target_date})[/] {universe_info}"
                    )
                    progress.console.print(forensic_table)

                    if draw_earnings > 0:
                        progress.console.print(
                            f"   💰 GANANCIA: [green]${draw_earnings:,.2f}[/]"
                        )
                    else:
                        progress.console.print(
                            f"   📉 Resultado: -${len(prediction.tickets)*self.rules.ticket_cost:,.2f}"
                        )

                progress.advance(task_id)

        # --- REPORTE FINAL ---
        net = total_earnings - total_investment
        self.console.print(f"\n[bold yellow]{'='*60}[/]")
        self.console.print(f"📊 BALANCE FINAL ({test_size} Sorteos)")
        self.console.print(f"   Inversión: ${total_investment:,.2f}")
        self.console.print(f"   Ganancia:  ${total_earnings:,.2f}")
        c = "green" if net >= 0 else "red"
        self.console.print(f"   Neto:      [{c}]${net:,.2f}[/]")

        # Puntería
        self.console.print("\n🎯 Distribución de Aciertos:")
        for h in range(7):
            cnt = hits_distribution.get(h, 0)
            if cnt > 0:
                bar = "█" * (cnt // 2 + 1)
                self.console.print(f"   {h} Hits: {cnt:3d} {bar}")

        if pre_process_strategy:
            self.console.print(f"\n🕸️  Eficacia Fase 1 (Radar):")
            self.console.print(f"   6 Hits en Red: {coverage_stats['captured_6']}")
            self.console.print(f"   5 Hits en Red: {coverage_stats['captured_5']}")
            self.console.print(f"   4 Hits en Red: {coverage_stats['captured_4']}")
            self.console.print(f"   Red Rota:      {coverage_stats['missed']}")

        return BacktestResultDTO(
            strategy_name,
            test_size,
            total_investment,
            total_earnings,
            net,
            hits_distribution,
        )
