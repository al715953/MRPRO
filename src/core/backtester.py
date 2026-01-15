from rich.console import Console
from src.domain.interfaces import ILotteryStrategy
from src.domain.dtos import DrawHistoryDTO, PredictionConfigDTO, BacktestResultDTO
from src.core.rules import MelateRetroRules
from src.data_access.report import SniperReport
from src.data_access.config import BEST_SETTINGS


class BacktestEngine:
    """
    Motor V5.8: Corregido para iteración limpia y reportes acumulados.
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

        # 1. ENCABEZADO GLOBAL
        if verbose:
            SniperReport.render_global_header(BEST_SETTINGS, config.num_tickets)

        total_investment = 0.0
        total_earnings = 0.0
        hits_distribution = {i: 0 for i in range(7)}

        funnel_stats = {
            "total_draws": 0,
            "opp_gold": 0,
            "opp_silver": 0,
            "opp_trash": 0,
            "captured_gold": 0,
            "captured_silver": 0,
        }

        full_history = list(
            zip(history.dates, history.winning_numbers, history.concursos)
        )
        full_history.sort(key=lambda x: x[2])
        test_size = min(config.backtest_size, len(full_history))
        start_index = len(full_history) - test_size

        # --- INICIO DEL BUCLE DE SORTEOS ---
        for i in range(start_index, len(full_history)):
            funnel_stats["total_draws"] += 1
            _, target_draw, target_id = full_history[i]
            target_tuple = tuple(sorted(target_draw[:6]))
            target_set = set(target_draw[:6])

            # Datos históricos previos al sorteo actual
            past_data = full_history[:i]
            p_dates, p_nums, p_ids = zip(*past_data)
            current_history = DrawHistoryDTO(list(p_dates), list(p_nums), list(p_ids))

            # FASE 1: REDUCCIÓN
            has_gold_potential = False
            has_silver_potential = False

            if pre_process_strategy:
                config.filter_overrides["verbose"] = False
                univ_result = pre_process_strategy.predict(current_history, config)

                if verbose:
                    ac_s = getattr(univ_result, "metadata", {}).get("ac_survivors", 0)
                    f_univ = getattr(univ_result, "metadata", {}).get("final_size", 0)
                    SniperReport.render_phase1_summary(target_id, ac_s, f_univ)

                universe_set = set(tuple(sorted(t)) for t in univ_result.tickets)
                if target_tuple in universe_set:
                    has_gold_potential = True
                    funnel_stats["opp_gold"] += 1
                else:
                    max_hit_in_univ = 0
                    for t in univ_result.tickets:
                        h = len(set(t) & target_set)
                        if h > max_hit_in_univ:
                            max_hit_in_univ = h
                        if h == 5:
                            break

                    if max_hit_in_univ == 5:
                        has_silver_potential = True
                        funnel_stats["opp_silver"] += 1
                    else:
                        funnel_stats["opp_trash"] += 1

            # FASE 3: SELECCIÓN
            prediction = strategy.predict(current_history, config)
            prediction_set = set(tuple(sorted(t)) for t in prediction.tickets)
            actually_captured = target_tuple in prediction_set

            # EVALUACIÓN FINANCIERA
            for ticket in prediction.tickets:
                total_investment += self.rules.ticket_cost
                h_nat, h_add = self.rules.validate_ticket(ticket, target_draw)
                prize = self.rules.calculate_prize(h_nat, h_add)
                total_earnings += prize
                hits_distribution[h_nat] += 1

            if actually_captured:
                if has_gold_potential:
                    funnel_stats["captured_gold"] += 1
                else:
                    funnel_stats["captured_silver"] += 1

            # TELEMETRÍA (Dentro del bucle, uno por sorteo)
            if verbose:
                meta = getattr(prediction, "metadata", {})
                audit = {}
                if hasattr(strategy, "audit_winner"):
                    audit = strategy.audit_winner(current_history, config, target_draw)

                audit["actually_captured"] = actually_captured
                SniperReport.render_draw_summary(meta, audit)

        # --- FIN DEL BUCLE ---

        # 2. DASHBOARD FINAL (FUERA DEL BUCLE)
        if verbose:
            SniperReport.render_final_dashboard(
                test_size,
                total_investment,
                total_earnings,
                funnel_stats,
                hits_distribution,
            )

        return BacktestResultDTO(
            strategy_name=strategy_name,
            total_draws_tested=test_size,
            investment=total_investment,
            earnings=total_earnings,
            net_balance=total_earnings - total_investment,
            hit_distribution=hits_distribution,
        )
