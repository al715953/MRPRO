from rich.progress import (
    Progress,
    SpinnerColumn,
    TimeElapsedColumn,
    BarColumn,
    TextColumn,
)
from rich.console import Console
from rich.table import Table
from src.domain.interfaces import ILotteryStrategy
from src.domain.dtos import DrawHistoryDTO, PredictionConfigDTO, BacktestResultDTO
from src.core.rules import MelateRetroRules


class BacktestEngine:
    """
    Motor de simulación histórica.
    Incluye:
    - Barra de progreso visual (Rich).
    - Radar de Cobertura (Fase 1).
    - Reporte detallado de tickets ganadores.
    """

    def __init__(self):
        self.rules = MelateRetroRules()
        self.console = Console()

    def run(
        self,
        strategy: ILotteryStrategy,
        history: DrawHistoryDTO,
        config: PredictionConfigDTO,
        verbose: bool = True,
        pre_process_strategy: ILotteryStrategy = None,
    ) -> BacktestResultDTO:

        strategy_name = strategy.__class__.__name__
        if verbose:
            self.console.print(
                f"\n[bold yellow]⚙️  Iniciando Backtest para:[/bold yellow] [cyan]{strategy_name}[/cyan]"
            )

        total_investment = 0.0
        total_earnings = 0.0
        hits_distribution = {0: 0, 1: 0, 2: 0, 3: 0, 4: 0, 5: 0, 6: 0}

        # Estadísticas de Cobertura
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
            transient=False,
        ) as progress:

            task_id = progress.add_task(
                f"[cyan]Simulando {test_size} sorteos...", total=test_size
            )

            for i in range(start_index, total_draws):
                target_date, target_draw, target_id = full_history[i]
                target_set = set(target_draw[:6])  # Solo naturales para el radar

                past_data = full_history[:i]
                if not past_data:
                    current_history = DrawHistoryDTO([], [], [])
                else:
                    p_dates, p_nums, p_ids = zip(*past_data)
                    current_history = DrawHistoryDTO(
                        list(p_dates), list(p_nums), list(p_ids)
                    )

                universe_info_str = ""

                # --- FASE 1: PRE-PROCESO (LA PESCA) ---
                if pre_process_strategy:
                    old_overrides = getattr(config, "filter_overrides", {})
                    config.filter_overrides = {**old_overrides, "verbose": False}
                    universe_result = pre_process_strategy.predict(
                        current_history, config
                    )
                    config.filter_overrides = old_overrides

                    # Radar de Cobertura
                    univ_size = len(universe_result.tickets)
                    max_potential_hit = 0
                    if univ_size > 0:
                        for t in universe_result.tickets:
                            h = len(set(t) & target_set)
                            if h > max_potential_hit:
                                max_potential_hit = h
                            if h == 6:
                                break

                    qa_icon = "❌"
                    if max_potential_hit == 6:
                        qa_icon = "💎"
                        coverage_stats["captured_6"] += 1
                    elif max_potential_hit == 5:
                        qa_icon = "⚠️"
                        coverage_stats["captured_5"] += 1
                    elif max_potential_hit == 4:
                        qa_icon = "📉"
                        coverage_stats["captured_4"] += 1
                    else:
                        coverage_stats["missed"] += 1

                    universe_info_str = f" | [magenta]Univ: {univ_size//1000}k[/] | [bold]{qa_icon} MaxPot: {max_potential_hit}[/]"

                # --- FASE 2: ESTRATEGIA ---
                prediction = strategy.predict(current_history, config)

                # --- FASE 3: EVALUACIÓN ---
                draw_earnings = 0.0
                max_hit = 0
                best_label = "0"
                winning_tickets_log = []  # Restauramos el log detallado

                for idx, ticket in enumerate(prediction.tickets, 1):
                    total_investment += self.rules.ticket_cost
                    hits_nat, has_add = self.rules.validate_ticket(ticket, target_draw)
                    prize = self.rules.calculate_prize(hits_nat, has_add)

                    if prize > 0:
                        draw_earnings += prize
                        # Guardamos el detalle del ticket ganador
                        t_str = ", ".join([f"{n:02d}" for n in sorted(ticket)])

                        # Etiqueta especial para premios con Adicional
                        type_str = f"{hits_nat} hits"
                        if has_add:
                            type_str += " + Bola Adicional"

                        winning_tickets_log.append(
                            f"   🎫 Ticket #{idx:02d}: [{t_str}] -> [bold green]${prize:,.2f}[/] ({type_str})"
                        )

                    # Actualizar estadísticas globales
                    if hits_nat > max_hit:
                        max_hit = hits_nat
                        best_label = f"{max_hit}"
                        if has_add:
                            best_label += "+B"
                    elif hits_nat == max_hit and has_add:
                        best_label = f"{max_hit}+B"

                    hits_distribution[hits_nat] = hits_distribution.get(hits_nat, 0) + 1

                total_earnings += draw_earnings

                # --- LOGGING ---
                # Si ganamos dinero, imprimimos el detalle antes de avanzar la barra
                if draw_earnings > 0:
                    progress.console.print(
                        f"\n[bold green]✨ ¡PREMIO EN SORTEO #{target_id} ({target_date})! ✨[/]"
                    )
                    for log_line in winning_tickets_log:
                        progress.console.print(log_line)

                    status_icon = "🟢"
                    msg = (
                        f"   RESUMEN: "
                        f"{universe_info_str} "
                        f"| {status_icon} Ganancia Total Sorteo: [bold green]${draw_earnings:,.2f}[/] (Best: {best_label})"
                    )
                    progress.console.print(msg)
                    progress.console.print("-" * 50)  # Separador visual

                elif verbose and max_hit >= 4:
                    # Si hubo un buen hit (4) pero no premio (por reglas) o falló algo, avisamos
                    progress.console.print(
                        f"[dim]   Sorteo #{target_id}: Max Hit {max_hit} (Sin premio)[/dim]"
                    )

                progress.advance(task_id)

        # --- RESUMEN FINAL ---
        net_balance = total_earnings - total_investment
        self.console.print(f"\n[bold yellow]{'='*60}[/bold yellow]")
        self.console.print(
            f"[bold]📊 RESUMEN FINAL DEL BACKTEST ({test_size} Sorteos)[/bold]"
        )
        self.console.print(f"{'='*60}")

        self.console.print(f"💰 Inversión Total:  ${total_investment:,.2f}")
        self.console.print(f"💵 Ganancia Total:   ${total_earnings:,.2f}")
        color_bal = "green" if net_balance >= 0 else "red"
        self.console.print(
            f"📉 Balance Neto:     [bold {color_bal}]${net_balance:,.2f}[/]"
        )

        # Tabla de Aciertos
        self.console.print(f"\n[bold]🎯 Puntería Final (IA/Heurística):[/bold]")
        dist_table = Table(show_header=False, box=None)
        max_count = max(hits_distribution.values()) if hits_distribution else 1
        for hits, count in sorted(hits_distribution.items(), reverse=True):
            if count > 0:
                bar = "█" * int((count / max_count) * 20)
                color = "green" if hits >= 3 else "dim white"
                dist_table.add_row(
                    f"{hits} Aciertos", f"{count:3d}", f"[{color}]{bar}[/]"
                )
        self.console.print(dist_table)

        # Estadísticas de Cobertura
        if pre_process_strategy:
            self.console.print(
                f"\n[bold magenta]🕸️  Calidad de la Red (Universo Pre-IA):[/bold magenta]"
            )
            self.console.print(
                f"   💎 6 Hits Disponibles: {coverage_stats['captured_6']} veces"
            )
            self.console.print(
                f"   ⚠️ 5 Hits Disponibles: {coverage_stats['captured_5']} veces"
            )
            self.console.print(
                f"   📉 4 Hits Disponibles: {coverage_stats['captured_4']} veces"
            )
            if coverage_stats["missed"] > 0:
                self.console.print(
                    f"   ❌ Red Rota (<4 Hits): {coverage_stats['missed']} veces"
                )

        self.console.print(f"[bold yellow]{'='*60}[/bold yellow]\n")

        return BacktestResultDTO(
            strategy_name=strategy.__class__.__name__,
            total_draws_tested=test_size,
            investment=total_investment,
            earnings=total_earnings,
            net_balance=net_balance,
            hit_distribution=hits_distribution,
        )
