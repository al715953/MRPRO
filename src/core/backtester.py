from src.domain.interfaces import ILotteryStrategy
from src.domain.dtos import DrawHistoryDTO, PredictionConfigDTO, BacktestResultDTO
from src.core.rules import MelateRetroRules


class BacktestEngine:
    def __init__(self):
        self.rules = MelateRetroRules()

    def run(
        self,
        strategy: ILotteryStrategy,
        history: DrawHistoryDTO,
        config: PredictionConfigDTO,
        verbose: bool = True,
    ) -> BacktestResultDTO:

        if verbose:
            print(f"⚙️ Iniciando Backtest para: {strategy.__class__.__name__}")

        total_investment = 0.0
        total_earnings = 0.0
        hits_distribution = {0: 0, 1: 0, 2: 0, 3: 0, 4: 0, 5: 0, 6: 0}

        # 1. Preparar la historia completa ordenada cronológicamente
        # Es crucial ordenar por ID para simular la línea de tiempo real.
        full_history = list(
            zip(history.dates, history.winning_numbers, history.concursos)
        )
        full_history.sort(key=lambda x: x[2])  # Ordenar por ID de concurso

        # 2. Definir el rango de pruebas
        total_draws = len(full_history)
        test_size = config.backtest_size

        if total_draws < test_size:
            test_size = total_draws

        # Índice de inicio: Calculamos desde dónde empezar a testear
        start_index = total_draws - test_size

        step_count = 1

        # 3. Bucle de Simulación "Viaje en el Tiempo"
        for i in range(start_index, total_draws):
            # El sorteo "Futuro" que intentamos predecir hoy
            target_date, target_draw, target_id = full_history[i]

            # --- CORRECCIÓN CRÍTICA ---
            # Recortamos la historia: Solo lo que pasó ANTES de 'i'.
            past_data = full_history[:i]

            if not past_data:
                # Caso borde: Primer sorteo de la historia (sin datos previos)
                current_history = DrawHistoryDTO([], [], [])
            else:
                # Reconstruimos el DTO con la historia parcial
                p_dates, p_nums, p_ids = zip(*past_data)
                current_history = DrawHistoryDTO(
                    list(p_dates), list(p_nums), list(p_ids)
                )

            # Ahora la estrategia recibe 'current_history', que termina justo ayer.
            # Por tanto, history.winning_numbers[-1] será diferente en cada vuelta.
            prediction = strategy.predict(current_history, config)

            # --- Evaluación de Resultados (Igual que antes) ---
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

            # --- SALIDA VISUAL (Solo si verbose=True) ---
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

                balance_icon = "🟢" if draw_earnings > 0 else "⚪"
                print(
                    f"\n✅ RESULTADO: Max Hit: {max_hit} | Total Sorteo: ${draw_earnings:,.2f} {balance_icon}"
                )
                print("─" * 60)

            step_count += 1

        return BacktestResultDTO(
            strategy_name=strategy.__class__.__name__,
            total_draws_tested=test_size,
            investment=total_investment,
            earnings=total_earnings,
            net_balance=total_earnings - total_investment,
            hit_distribution=hits_distribution,
        )
