from src.domain.interfaces import ILotteryStrategy
from src.domain.dtos import DrawHistoryDTO, PredictionConfigDTO, BacktestResultDTO
from src.core.rules import MelateRetroRules

class BacktestEngine:
    def __init__(self):
        self.rules = MelateRetroRules()

    def run(self, strategy: ILotteryStrategy, history: DrawHistoryDTO, config: PredictionConfigDTO) -> BacktestResultDTO:
        print(f"⚙️ Iniciando Backtest para: {strategy.__class__.__name__}")
        
        total_investment = 0.0
        total_earnings = 0.0
        hits_distribution = {0:0, 1:0, 2:0, 3:0, 4:0, 5:0, 6:0}
        
        # 1. Ordenar historia por concurso
        full_history = list(zip(history.dates, history.winning_numbers, history.concursos))
        full_history.sort(key=lambda x: x[2]) 

        # 2. Definir rango
        test_range = config.backtest_size
        if len(full_history) < test_range:
            test_range = len(full_history)

        test_data = full_history[-test_range:]
        step_count = 1

        # 3. Bucle de ejecución
        for date, real_draw, id_concurso in test_data:
            print(f"\n" + "─"*60)
            print(f"🎫 SORTEO: #{id_concurso} | FECHA: {date} | ({step_count}/{test_range})")
            print(f"🎱 Reales: {real_draw}")

            prediction = strategy.predict(history, config)
            
            draw_earnings = 0.0
            max_hit = 0
            
            for ticket in prediction.tickets:
                total_investment += self.rules.ticket_cost
                hits = self.rules.validate_ticket(ticket, real_draw)
                prize = self.rules.calculate_prize(hits)
                
                draw_earnings += prize
                if hits > max_hit: max_hit = hits
                
                total_earnings += prize
                hits_distribution[hits] = hits_distribution.get(hits, 0) + 1

            # --- UX MEJORADA: RESULTADOS EN UNA SOLA LÍNEA ---
            balance_icon = "🟢" if draw_earnings > 0 else "⚪"
            print(f"✅ RESULTADO: Max Hit: {max_hit} | Premios: ${draw_earnings:,.2f} {balance_icon}")
            print("─"*60)
            
            step_count += 1

        return BacktestResultDTO(
            strategy_name=strategy.__class__.__name__,
            total_draws_tested=test_range,
            investment=total_investment,
            earnings=total_earnings,
            net_balance=total_earnings - total_investment,
            hit_distribution=hits_distribution
        )