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


class DummyProgress:
    """Clase auxiliar para silenciar la barra de progreso en modo optimización."""
    def __enter__(self): return self
    def __exit__(self, *args): pass
    def add_task(self, *args, **kwargs): return 0
    def advance(self, *args, **kwargs): pass
    @property
    def console(self): return self
    def print(self, *args, **kwargs): pass


class BacktestEngine:
    """
    Motor de simulación histórica V4 (Forensic Mode Enabled).
    """

    def __init__(self):
        self.rules = MelateRetroRules()
        self.console = Console()

    def run(
        self,
        strategy: ILotteryStrategy,
        history: DrawHistoryDTO,
        config: PredictionConfigDTO,
        verbose: bool = False,
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

        coverage_stats = {
            "missed": 0,
            "captured_6": 0,
            "captured_5": 0,
            "captured_4": 0,
        }

        full_history = list(
            zip(history.dates, history.winning_numbers, history.concursos)
        )
        full_history.sort(key=lambda x: x[2])

        total_draws = len(full_history)
        test_size = min(config.backtest_size, total_draws)
        start_index = total_draws - test_size

        # Contexto de barra
        progress_ctx = DummyProgress()
        if verbose:
            progress_ctx = Progress(
                SpinnerColumn(),
                TextColumn("[progress.description]{task.description}"),
                BarColumn(),
                TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
                TimeElapsedColumn(),
                console=self.console,
                transient=True, 
            )

        with progress_ctx as progress:
            task_id = progress.add_task(
                f"[cyan]Procesando {test_size} sorteos...[/cyan]", total=test_size
            )

            for i in range(start_index, total_draws):
                target_date, target_draw, target_id = full_history[i]
                target_set = set(target_draw[:6])

                past_data = full_history[:i]
                if not past_data:
                    current_history = DrawHistoryDTO([], [], [])
                else:
                    p_dates, p_nums, p_ids = zip(*past_data)
                    current_history = DrawHistoryDTO(
                        list(p_dates), list(p_nums), list(p_ids)
                    )

                # --- FASE 1: PRE-PROCESO (LA PESCA) ---
                universe_has_winner = False
                universe_info_str = ""

                if pre_process_strategy:
                    old_overrides = getattr(config, "filter_overrides", {})
                    config.filter_overrides = {**old_overrides, "verbose": False}
                    
                    universe_result = pre_process_strategy.predict(
                        current_history, config
                    )
                    config.filter_overrides = old_overrides

                    # Cálculo silencioso de cobertura
                    univ_size = len(universe_result.tickets)
                    max_potential_hit = 0
                    if univ_size > 0:
                        for t in universe_result.tickets:
                            h = len(set(t) & target_set)
                            if h > max_potential_hit: max_potential_hit = h
                            if h == 6: break

                    if max_potential_hit == 6:
                        universe_has_winner = True
                        coverage_stats["captured_6"] += 1
                        qa_icon = "💎"
                    elif max_potential_hit == 5:
                        coverage_stats["captured_5"] += 1
                        qa_icon = "⚠️"
                    elif max_potential_hit == 4:
                        coverage_stats["captured_4"] += 1
                        qa_icon = "📉"
                    else:
                        coverage_stats["missed"] += 1
                        qa_icon = "❌"
                    
                    if verbose:
                        universe_info_str = f" | [magenta]Univ: {univ_size//1000}k[/] | [bold]{qa_icon} MaxPot: {max_potential_hit}[/]"

                # --- FASE 2: ESTRATEGIA ---
                prediction = strategy.predict(current_history, config)

                # --- FASE 3: EVALUACIÓN ---
                draw_earnings = 0.0
                max_hit = 0
                best_label = "0"
                winning_tickets_log = []

                for idx, ticket in enumerate(prediction.tickets, 1):
                    total_investment += self.rules.ticket_cost
                    hits_nat, has_add = self.rules.validate_ticket(ticket, target_draw)
                    prize = self.rules.calculate_prize(hits_nat, has_add)

                    if prize > 0:
                        draw_earnings += prize
                        if verbose:
                            t_str = ", ".join([f"{n:02d}" for n in sorted(ticket)])
                            type_str = f"{hits_nat} hits"
                            if has_add: type_str += " + Bola Adicional"
                            winning_tickets_log.append(
                                f"   🎫 Ticket #{idx:02d}: [{t_str}] -> [bold green]${prize:,.2f}[/] ({type_str})"
                            )

                    if hits_nat > max_hit:
                        max_hit = hits_nat
                        best_label = f"{max_hit}"
                        if has_add: best_label += "+B"
                    elif hits_nat == max_hit and has_add:
                        best_label = f"{max_hit}+B"

                    hits_distribution[hits_nat] = hits_distribution.get(hits_nat, 0) + 1

                total_earnings += draw_earnings

                # --- LOGGING & FORENSICS ---
                progress.advance(task_id)

                if verbose:
                    # 1. Si ganamos
                    if draw_earnings > 0:
                        progress.console.print(f"\n[bold green]✨ ¡PREMIO EN SORTEO #{target_id}! ✨[/]")
                        for log_line in winning_tickets_log:
                            progress.console.print(log_line)
                        progress.console.print(f"   RESUMEN: {universe_info_str} | 🟢 Total: [bold green]${draw_earnings:,.2f}[/]")
                        progress.console.print("-" * 50)

                    # 2. CASO CRÍTICO: Diamante perdido (Estaba en universo, pero no ganamos premio mayor)
                    # Si universe_has_winner es True, pero max_hit < 6, ejecutamos forense
                    if universe_has_winner and max_hit < 6 and hasattr(strategy, "audit_winner"):
                        progress.console.print(f"\n[bold magenta]🔍 ALERTA FORENSE (Sorteo #{target_id})[/]")
                        progress.console.print(f"   El ganador {target_draw[:6]} estaba en el Universo, pero no fue seleccionado.")
                        
                        # Llamada al detective
                        report = strategy.audit_winner(current_history, config, target_draw)
                        progress.console.print(report)
                        progress.console.print("-" * 50)

        # --- RESUMEN FINAL ---
        net_balance = total_earnings - total_investment
        
        if verbose:
            self.console.print(f"\n[bold yellow]{'='*60}[/bold yellow]")
            self.console.print(f"[bold]📊 RESUMEN FINAL DEL BACKTEST ({test_size} Sorteos)[/bold]")
            self.console.print(f"{'='*60}")
            self.console.print(f"💰 Inversión: ${total_investment:,.2f} | 💵 Ganancia: ${total_earnings:,.2f}")
            color_bal = "green" if net_balance >= 0 else "red"
            self.console.print(f"📉 Balance:    [bold {color_bal}]${net_balance:,.2f}[/]")

            # Tabla Simple
            self.console.print(f"\n[bold]🎯 Puntería Final:[/bold]")
            dist_table = Table(show_header=False, box=None)
            for hits, count in sorted(hits_distribution.items(), reverse=True):
                if count > 0:
                    color = "green" if hits >= 3 else "white"
                    dist_table.add_row(f"{hits} Aciertos", f"{count:3d}", f"[{color}]{'█'*count}[/]")
            self.console.print(dist_table)

            if pre_process_strategy:
                self.console.print(f"\n[bold magenta]🕸️  Calidad Universo (Fase 1):[/bold magenta]")
                self.console.print(f"   💎 6 Hits: {coverage_stats['captured_6']} veces")
                self.console.print(f"   ⚠️ 5 Hits: {coverage_stats['captured_5']} veces")

        return BacktestResultDTO(
            strategy_name=strategy.__class__.__name__,
            total_draws_tested=test_size,
            investment=total_investment,
            earnings=total_earnings,
            net_balance=net_balance,
            hit_distribution=hits_distribution,
        )