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


class DummyProgress:
    """Clase auxiliar para silenciar la barra de progreso en modo optimización."""

    def __enter__(self):
        return self

    def __exit__(self, *args):
        pass

    def add_task(self, *args, **kwargs):
        return 0

    def advance(self, *args, **kwargs):
        pass

    @property
    def console(self):
        return self

    def print(self, *args, **kwargs):
        pass


class BacktestEngine:
    """
    Motor de simulación histórica V5.1 (High Sensitivity Forensics).
    Ahora rastrea explícitamente la pérdida de premios de 5 aciertos.
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
                f"\n[bold yellow]⚙️  Iniciando Backtest Forense V5.1 (Target: 5+ Hits) para:[/bold yellow] [cyan]{strategy_name}[/cyan]"
            )

        total_investment = 0.0
        total_earnings = 0.0
        hits_distribution = {0: 0, 1: 0, 2: 0, 3: 0, 4: 0, 5: 0, 6: 0}

        # --- METRICAS DE AUTOPSIA (FUNNEL 5+) ---
        # Rastreamos cuántas veces tuvimos "Oro" (6) o "Plata" (5) en la mano y lo dejamos ir.
        funnel_stats = {
            "total_draws": 0,
            # Fase 1: Potencial
            "opp_gold": 0,  # Universo contenía el 6
            "opp_silver": 0,  # Universo contenía el 5 (pero no 6)
            "opp_trash": 0,  # Universo no tenía ni 5 ni 6
            # Fase 3: Conversión
            "captured_gold": 0,  # Teníamos 6 y cobramos 6
            "captured_silver": 0,  # Teníamos 5+ y cobramos 5+
            "lost_opportunity": 0,  # Teníamos 5+ y cobramos 4 o menos
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
                TextColumn("[bold cyan]{task.description}"),
                BarColumn(bar_width=40, style="dim white", complete_style="green"),
                TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
                TimeElapsedColumn(),
                console=self.console,
                transient=True,
            )

        with progress_ctx as progress:
            task_id = progress.add_task(
                f"Simulando {test_size} sorteos...", total=test_size
            )

            for i in range(start_index, total_draws):
                funnel_stats["total_draws"] += 1
                target_date, target_draw, target_id = full_history[i]
                target_set = set(target_draw[:6])
                target_tuple = tuple(sorted(target_draw[:6]))

                past_data = full_history[:i]
                if not past_data:
                    current_history = DrawHistoryDTO([], [], [])
                else:
                    p_dates, p_nums, p_ids = zip(*past_data)
                    current_history = DrawHistoryDTO(
                        list(p_dates), list(p_nums), list(p_ids)
                    )

                # --- FASE 1: PRE-PROCESO (LA PESCA) ---
                universe_info_str = ""
                max_potential_hit = 0

                # Banderas para autopsia
                has_gold_potential = False
                has_silver_potential = False

                if pre_process_strategy:
                    old_overrides = getattr(config, "filter_overrides", {})
                    # Silenciar Fase 1
                    config.filter_overrides = {**old_overrides, "verbose": False}

                    universe_result = pre_process_strategy.predict(
                        current_history, config
                    )
                    config.filter_overrides = old_overrides

                    univ_size = len(universe_result.tickets)

                    # Verificación de Potencial (Optimizada)
                    # 1. Buscamos Jackpot exacto
                    universe_set = set(tuple(t) for t in universe_result.tickets)
                    if target_tuple in universe_set:
                        max_potential_hit = 6
                        has_gold_potential = True
                    else:
                        # 2. Si no, buscamos el max hit (necesario para saber si hay 5)
                        for t in universe_result.tickets:
                            h = len(set(t) & target_set)
                            if h > max_potential_hit:
                                max_potential_hit = h
                            if (
                                h == 5
                            ):  # Suficiente para marcar silver potential si no hay gold
                                break

                        if max_potential_hit == 5:
                            has_silver_potential = True

                    # Actualizar Stats Fase 1
                    qa_icon = "💀"
                    if has_gold_potential:
                        funnel_stats["opp_gold"] += 1
                        qa_icon = "💎"
                    elif max_potential_hit == 5:
                        funnel_stats["opp_silver"] += 1
                        qa_icon = "🥈"
                    else:
                        funnel_stats["opp_trash"] += 1
                        qa_icon = "📉"

                    if verbose:
                        universe_info_str = f" | [magenta]Univ: {univ_size//1000}k[/] | {qa_icon} Potencial: {max_potential_hit}/6"

                # --- FASE 2 y 3: ESTRATEGIA Y SELECCIÓN ---
                prediction = strategy.predict(current_history, config)

                # --- FASE 4: EVALUACIÓN DE RESULTADOS ---
                draw_earnings = 0.0
                max_hit = 0
                best_label = "0"
                winning_tickets_log = []

                captured_gold = False
                captured_silver = False

                for idx, ticket in enumerate(prediction.tickets, 1):
                    total_investment += self.rules.ticket_cost
                    hits_nat, has_add = self.rules.validate_ticket(ticket, target_draw)
                    prize = self.rules.calculate_prize(hits_nat, has_add)

                    if hits_nat == 6:
                        captured_gold = True
                    if hits_nat == 5:
                        captured_silver = True

                    if prize > 0:
                        draw_earnings += prize
                        if verbose:
                            t_str = ", ".join([f"{n:02d}" for n in sorted(ticket)])
                            type_str = f"{hits_nat} hits"
                            if has_add:
                                type_str += " + Bola Adicional"
                            color = "green" if hits_nat < 5 else "bold yellow"
                            winning_tickets_log.append(
                                f"   🎫 Ticket #{idx:02d}: [{t_str}] -> [{color}]${prize:,.2f}[/] ({type_str})"
                            )

                    if hits_nat > max_hit:
                        max_hit = hits_nat
                        best_label = f"{max_hit}"

                    hits_distribution[hits_nat] = hits_distribution.get(hits_nat, 0) + 1

                total_earnings += draw_earnings

                # --- ACTUALIZAR EMBUDO (CONVERSIÓN) ---
                # Definimos "Oportunidad Valiosa" como tener 5 o 6 en el universo
                had_opportunity = has_gold_potential or has_silver_potential

                # Definimos "Éxito" como capturar 5 o 6
                success = captured_gold or captured_silver

                status_icon = "❓"
                if had_opportunity:
                    if captured_gold:
                        funnel_stats["captured_gold"] += 1
                        status_icon = "✅ JACKPOT"
                    elif captured_silver:
                        funnel_stats["captured_silver"] += 1
                        status_icon = "✅ 5 HITS"
                    else:
                        funnel_stats["lost_opportunity"] += 1
                        status_icon = "📉 FAILED SELECTION"
                else:
                    status_icon = "💀 BAD FILTER"

                progress.advance(task_id)

                # --- REPORTING VERBOSE ---
                if verbose:
                    # A. Si ganamos algo decente (3+) o dinero
                    if draw_earnings > 0:
                        header_color = "green" if max_hit < 5 else "bold yellow"
                        progress.console.print(
                            f"\n[{header_color}]✨ PREMIO SORTEO #{target_id} | Max: {max_hit} hits[/]"
                        )
                        for log_line in winning_tickets_log:
                            progress.console.print(log_line)
                        progress.console.print(
                            f"   RESUMEN: {universe_info_str} | 💰 Cash: [{header_color}]${draw_earnings:,.2f}[/]"
                        )
                        progress.console.print("-" * 50)

                    # B. TRIGGER FORENSE DE "PLATA O MEJOR"
                    # Si había potencial de 5+, y sacamos menos de eso.
                    if max_potential_hit >= 5 and max_hit < 5:
                        target_type = (
                            "JACKPOT (6/6)"
                            if max_potential_hit == 6
                            else "PREMIO MAYOR (5/6)"
                        )
                        progress.console.print(
                            f"\n[bold red]🔍 AUTOPSIA FORENSE #{target_id} ({status_icon})[/]"
                        )
                        progress.console.print(
                            f"   El Universo tenía {target_type}. Fallamos en la Pesca Final (Max: {max_hit})."
                        )

                        if hasattr(strategy, "audit_winner"):
                            # El audit winner suele buscar el 6, pero sigue siendo útil ver el ranking
                            report = strategy.audit_winner(
                                current_history, config, target_draw
                            )
                            progress.console.print(report)
                        progress.console.print("-" * 50)

        # --- RESUMEN FINAL EJECUTIVO ---
        net_balance = total_earnings - total_investment

        if verbose:
            self.console.print(f"\n\n[bold yellow]{'='*60}[/bold yellow]")
            self.console.print(
                f"[bold white on blue] 📊 REPORTE DE RENDIMIENTO V5.1 (Target: 5+ Hits) [/]"
            )
            self.console.print(f"[bold yellow]{'='*60}[/bold yellow]")

            # 1. Tabla Financiera
            fin_table = Table(title="💵 Finanzas", box=box.SIMPLE)
            fin_table.add_column("Concepto", style="cyan")
            fin_table.add_column("Valor", justify="right")

            fin_table.add_row("Sorteos Analizados", f"{test_size}")
            fin_table.add_row("Inversión Total", f"${total_investment:,.2f}")
            fin_table.add_row("Ganancias", f"${total_earnings:,.2f}")

            color_bal = "green" if net_balance >= 0 else "red"
            fin_table.add_row("Balance Neto", f"[{color_bal}]${net_balance:,.2f}[/]")
            fin_table.add_row(
                "ROI",
                (
                    f"[{color_bal}]{(net_balance/total_investment)*100:.1f}%[/]"
                    if total_investment
                    else "0%"
                ),
            )

            self.console.print(fin_table)
            self.console.print("\n")

            # 2. Tabla de Embudo (Funnel de 5+ Aciertos)
            funnel_table = Table(
                title="📉 Embudo de Calidad (Premios 5 y 6)", box=box.ROUNDED
            )
            funnel_table.add_column("Métrica", style="magenta")
            funnel_table.add_column("Conteo", justify="right")
            funnel_table.add_column("Tasa", justify="right", style="cyan")

            total = funnel_stats["total_draws"]

            # Oportunidades Totales (Univ tiene 5 o 6)
            total_opps = funnel_stats["opp_gold"] + funnel_stats["opp_silver"]

            # Capturas Totales (Sel tiene 5 o 6 dado que había oportunidad)
            total_caps = funnel_stats["captured_gold"] + funnel_stats["captured_silver"]
            lost_caps = funnel_stats[
                "lost_opportunity"
            ]  # Debería ser total_opps - total_caps si logic is perfect

            # Porcentajes
            pct_opps = (total_opps / total * 100) if total else 0
            pct_conv = (total_caps / total_opps * 100) if total_opps else 0

            funnel_table.add_row(
                "Fase 1: Universos con 5+ Hits",
                f"{total_opps}/{total}",
                f"{pct_opps:.1f}%",
            )
            funnel_table.add_row(
                "   ├─ Potencial Jackpot (6)", f"{funnel_stats['opp_gold']}", ""
            )
            funnel_table.add_row(
                "   └─ Potencial Plata (5)", f"{funnel_stats['opp_silver']}", ""
            )

            color_conv = "green" if pct_conv > 10 else "red"
            funnel_table.add_section()
            funnel_table.add_row(
                "Fase 3: Conversión Final (5+ Hits)",
                f"{total_caps}/{total_opps}",
                f"[{color_conv}]{pct_conv:.1f}%[/]",
            )
            funnel_table.add_row(
                "   ├─ Jackpots Cobrados", f"{funnel_stats['captured_gold']}", "🏆"
            )
            funnel_table.add_row(
                "   └─ Platas Cobradas", f"{funnel_stats['captured_silver']}", "🥈"
            )

            self.console.print(funnel_table)

            # Diagnóstico
            self.console.print("\n[bold]🩺 Diagnóstico de Fuga:[/bold]")
            if pct_opps < 20:
                self.console.print(
                    "   🔴 [red]PROBLEMA FASE 1:[/red] Pocos universos tienen potencial de 5+. Relaja filtros."
                )
            elif pct_conv < 5:
                self.console.print(
                    "   🔴 [red]PROBLEMA FASE 3:[/red] Muchos premios de 5+ entran al universo pero NO se seleccionan."
                )
                self.console.print(
                    "      -> Revisa: Score IA muy bajo para ganadores o Cuotas insuficientes."
                )
            else:
                self.console.print(
                    "   🟢 [green]SISTEMA SANO:[/green] Buena tasa de generación y captura."
                )

            # 3. Puntería
            self.console.print(f"\n[bold]🎯 Distribución de Aciertos:[/bold]")
            dist_table = Table(show_header=False, box=None)
            for hits, count in sorted(hits_distribution.items(), reverse=True):
                if count > 0:
                    color = "green" if hits >= 3 else "dim white"
                    bar = "█" * min(count, 50)
                    dist_table.add_row(
                        f"{hits} Aciertos", f"{count:3d}", f"[{color}]{bar}[/]"
                    )
            self.console.print(dist_table)

        return BacktestResultDTO(
            strategy_name=strategy.__class__.__name__,
            total_draws_tested=test_size,
            investment=total_investment,
            earnings=total_earnings,
            net_balance=net_balance,
            hit_distribution=hits_distribution,
        )
