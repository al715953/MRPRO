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
    ) -> BacktestResultDTO:
        print(f"⚙️ Iniciando Backtest para: {strategy.__class__.__name__}")

        total_investment = 0.0
        total_earnings = 0.0
        hits_distribution = {0: 0, 1: 0, 2: 0, 3: 0, 4: 0, 5: 0, 6: 0}

        # 1. Ordenar historia por concurso
        full_history = list(
            zip(history.dates, history.winning_numbers, history.concursos)
        )
        full_history.sort(key=lambda x: x[2])

        # 2. Definir rango
        test_range = config.backtest_size
        if len(full_history) < test_range:
            test_range = len(full_history)

        test_data = full_history[-test_range:]
        step_count = 1

        # 3. Bucle de ejecución
        for date, real_draw, id_concurso in test_data:

            prediction = strategy.predict(history, config)

            draw_earnings = 0.0
            max_hit = 0
            tickets_ganadores = []  # Para almacenar solo los que ganan

            # -----------------------------Nueva lógica que imprime los ganadores-----------------------------
            for i, ticket in enumerate(prediction.tickets, 1):
                total_investment += self.rules.ticket_cost

                # Obtenemos aciertos (naturales y adicional)
                hits_nat, has_add = self.rules.validate_ticket(ticket, real_draw)
                prize = self.rules.calculate_prize(hits_nat, has_add)

                if prize > 0:
                    # Guardamos formato: [ 01, 02... ] -> $10.00
                    t_str = ", ".join([f"{n:02d}" for n in sorted(ticket)])
                    tickets_ganadores.append(
                        f"   Ticket #{i:02d}: [{t_str}] -> ${prize:,.2f}"
                    )

                draw_earnings += prize
                if hits_nat > max_hit:
                    max_hit = hits_nat

                total_earnings += prize
                hits_distribution[hits_nat] = hits_distribution.get(hits_nat, 0) + 1

            # --- NUEVA SALIDA VISUAL ---
            print(f"\n" + "─" * 60)
            print(
                f"🎫 SORTEO: #{id_concurso} | FECHA: {date} | ({step_count}/{test_range})"
            )
            print(f"🎱 Reales: {real_draw}")

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
            total_draws_tested=test_range,
            investment=total_investment,
            earnings=total_earnings,
            net_balance=total_earnings - total_investment,
            hit_distribution=hits_distribution,
        )
