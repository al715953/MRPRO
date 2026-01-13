from colorama import Fore, Style
from src.domain.interfaces import ILotteryStrategy
from src.domain.dtos import DrawHistoryDTO, PredictionConfigDTO, BacktestResultDTO
from src.core.rules import MelateRetroRules


class BacktestEngine:
    """
    Motor de simulación histórica.
    Permite validar estrategias 'rebobinando' el tiempo.
    """

    def __init__(self):
        self.rules = MelateRetroRules()

    def run(
        self,
        strategy: ILotteryStrategy,
        history: DrawHistoryDTO,
        config: PredictionConfigDTO,
        verbose: bool = True,
        pre_process_strategy: ILotteryStrategy = None,  # <--- NUEVO: Inyección de Dependencia
    ) -> BacktestResultDTO:

        strategy_name = strategy.__class__.__name__
        if verbose:
            print(f"⚙️ Iniciando Backtest Profesional para: {strategy_name}")
            if pre_process_strategy:
                print(
                    f"   ↳ Con Pre-Proceso: {pre_process_strategy.__class__.__name__} (Regeneración de Universo)"
                )

        total_investment = 0.0
        total_earnings = 0.0
        hits_distribution = {0: 0, 1: 0, 2: 0, 3: 0, 4: 0, 5: 0, 6: 0}

        # Preparar historia cronológica
        full_history = list(
            zip(history.dates, history.winning_numbers, history.concursos)
        )
        full_history.sort(key=lambda x: x[2])

        total_draws = len(full_history)
        test_size = config.backtest_size
        if total_draws < test_size:
            test_size = total_draws

        start_index = total_draws - test_size
        step_count = 1

        # --- BUCLE DE SIMULACIÓN (VIAJE EN EL TIEMPO) ---
        for i in range(start_index, total_draws):
            target_date, target_draw, target_id = full_history[i]

            # Cortamos la historia justo antes del sorteo objetivo
            past_data = full_history[:i]

            if not past_data:
                current_history = DrawHistoryDTO([], [], [])
            else:
                p_dates, p_nums, p_ids = zip(*past_data)
                current_history = DrawHistoryDTO(
                    list(p_dates), list(p_nums), list(p_ids)
                )

            # 1. EJECUTAR PRE-PROCESO (Si existe)
            # Esto es vital para la IA: Regenerar el universo con datos DE ESE MOMENTO
            if pre_process_strategy:
                if verbose:
                    print(f"\n   🔄 Regenerando Universo para Sorteo {target_id}...")
                # Ejecutamos la reducción (esto guarda el CSV 'universo_reducido.csv' con datos del pasado)
                # Forzamos verbose=False en el pre-proceso para no ensuciar la consola
                old_overrides = getattr(config, "filter_overrides", {})
                config.filter_overrides = {**old_overrides, "verbose": False}

                pre_process_strategy.predict(current_history, config)

                # Restauramos config
                config.filter_overrides = old_overrides

            # 2. EJECUTAR ESTRATEGIA PRINCIPAL (IA / Selector)
            # La estrategia leerá el CSV recién generado y entrenará la IA con 'current_history'
            prediction = strategy.predict(current_history, config)

            # 3. EVALUACIÓN DE RESULTADOS
            draw_earnings = 0.0
            max_hit = 0
            tickets_ganadores = []

            for idx, ticket in enumerate(prediction.tickets, 1):
                total_investment += self.rules.ticket_cost
                hits_nat, has_add = self.rules.validate_ticket(ticket, target_draw)
                prize = self.rules.calculate_prize(hits_nat, has_add)

                if prize > 0:
                    t_str = ", ".join([f"{n:02d}" for n in sorted(ticket)])
                    tickets_ganadores.append(
                        f"   Ticket #{idx:02d}: [{t_str}] -> ${prize:,.2f}"
                    )

                draw_earnings += prize
                if hits_nat > max_hit:
                    max_hit = hits_nat
                total_earnings += prize
                hits_distribution[hits_nat] = hits_distribution.get(hits_nat, 0) + 1

            # Reporte del Paso
            if verbose:
                print(f"\n" + "─" * 60)
                print(
                    f"🎫 SORTEO: #{target_id} | FECHA: {target_date} | ({step_count}/{test_size})"
                )
                print(f"🎱 Reales: {target_draw}")
                if tickets_ganadores:
                    print("\n✨ ACUMULADO GANADOR:")
                    for t in tickets_ganadores:
                        print(t)
                else:
                    print("\n   (Sin tickets premiados)")

                bal_icon = "🟢" if draw_earnings > 0 else "⚪"
                print(
                    f"\n✅ RESULTADO: Max Hit: {max_hit} | Ganancia Sorteo: ${draw_earnings:,.2f} {bal_icon}"
                )
                print("─" * 60)

            step_count += 1

        # --- RESUMEN FINAL ---
        net_balance = total_earnings - total_investment
        if verbose:
            print(f"\n{Fore.YELLOW}" + "═" * 60)
            print(f"📊 RESUMEN FINAL DEL BACKTEST ({test_size} Sorteos)")
            print("═" * 60 + f"{Style.RESET_ALL}")
            print(f"💰 Inversión Total:  ${total_investment:,.2f}")
            print(f"💵 Ganancia Total:   ${total_earnings:,.2f}")

            color_bal = Fore.GREEN if net_balance >= 0 else Fore.RED
            print(
                f"📉 Balance Neto:     {color_bal}${net_balance:,.2f}{Style.RESET_ALL}"
            )

            print(f"\n🎯 Distribución de Aciertos:")
            for hits, count in sorted(hits_distribution.items(), reverse=True):
                if count > 0:
                    bar = "█" * count
                    print(f"   {hits} Aciertos: {count:2d} {bar}")
            print(f"{Fore.YELLOW}" + "═" * 60 + f"{Style.RESET_ALL}\n")

        return BacktestResultDTO(
            strategy_name=strategy.__class__.__name__,
            total_draws_tested=test_size,
            investment=total_investment,
            earnings=total_earnings,
            net_balance=net_balance,
            hit_distribution=hits_distribution,
        )
