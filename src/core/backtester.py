import json
import os
from rich.console import Console
from src.domain.interfaces import ILotteryStrategy
from src.domain.dtos import DrawHistoryDTO, PredictionConfigDTO, BacktestResultDTO
from src.core.rules import MelateRetroRules
from src.data_access.report import SniperReport


class BacktestEngine:
    """
    Motor V6.0: Backtest con persistencia de Auditoría Forense para Visualización.
    """

    def __init__(self):
        self.rules = MelateRetroRules()
        self.console = Console()
        self.audit_history = []  # Memoria de auditoría para visualización

    def run(
        self,
        strategy: ILotteryStrategy,
        history: DrawHistoryDTO,
        config: PredictionConfigDTO,
        verbose: bool = False,
        pre_process_strategy: ILotteryStrategy = None,
    ) -> BacktestResultDTO:
        self.audit_history = []  # Reset al inicio
        total_investment = 0.0
        total_earnings = 0.0
        hits_distribution = {i: 0 for i in range(7)}
        funnel_stats = {
            "total_draws": 0,
            "opp_gold": 0,
            "opp_silver": 0,
            "opp_bronze": 0,
            "captured_gold": 0,
            "captured_silver": 0,
            "captured_bronze": 0,
        }

        full_history = list(
            zip(history.dates, history.winning_numbers, history.concursos)
        )
        full_history.sort(key=lambda x: x[2])
        test_size = min(config.backtest_size, len(full_history))
        start_index = len(full_history) - test_size

        for i in range(start_index, len(full_history)):
            funnel_stats["total_draws"] += 1
            _, target_draw, target_id = full_history[i]
            target_tuple = tuple(sorted(target_draw[:6]))
            target_set = set(target_draw[:6])

            past_data = full_history[:i]
            p_dates, p_nums, p_ids = zip(*past_data)
            current_history = DrawHistoryDTO(list(p_dates), list(p_nums), list(p_ids))

            # FASE 1: REDUCCIÓN
            has_gold = has_silver = has_bronze = False
            if pre_process_strategy:
                config.filter_overrides["verbose"] = False
                univ_result = pre_process_strategy.predict(current_history, config)
                if (
                    hasattr(univ_result, "metadata")
                    and "raw_ndarray" in univ_result.metadata
                ):
                    config.raw_universe_ptr = univ_result.metadata["raw_ndarray"]

                # Verificación de Potencial
                max_hit = 0
                for t in univ_result.tickets:
                    h = len(set(t) & target_set)
                    if h > max_hit:
                        max_hit = h
                    if h == 6:
                        break

                if max_hit == 6:
                    funnel_stats["opp_gold"] += 1
                    has_gold = True
                elif max_hit == 5:
                    funnel_stats["opp_silver"] += 1
                    has_silver = True
                elif max_hit == 4:
                    funnel_stats["opp_bronze"] += 1
                    has_bronze = True

            # FASE 3: SELECCIÓN
            prediction = strategy.predict(current_history, config)
            prediction_set = set(tuple(sorted(t)) for t in prediction.tickets)
            captured = target_tuple in prediction_set

            # Auditoría y Guardado
            if hasattr(strategy, "audit_winner"):
                audit = strategy.audit_winner(current_history, config, target_draw)
                audit["draw_id"] = int(target_id)
                audit["actually_captured"] = captured
                self.audit_history.append(audit)  # Guardamos para el JSON

            for ticket in prediction.tickets:
                total_investment += self.rules.ticket_cost
                h_nat, h_add = self.rules.validate_ticket(ticket, target_draw)
                total_earnings += self.rules.calculate_prize(h_nat, h_add)
                hits_distribution[h_nat] += 1

            if captured:
                if has_gold:
                    funnel_stats["captured_gold"] += 1
                elif has_silver:
                    funnel_stats["captured_silver"] += 1
                elif has_bronze:
                    funnel_stats["captured_bronze"] += 1

            if verbose:
                SniperReport.render_draw_summary(
                    getattr(prediction, "metadata", {}), audit
                )

        # PERSISTENCIA: Guardar resultados para visualización
        os.makedirs("data", exist_ok=True)
        with open("data/backtest_results.json", "w") as f:
            json.dump(self.audit_history, f, indent=4)

        if verbose:
            SniperReport.render_final_dashboard(
                test_size,
                total_investment,
                total_earnings,
                funnel_stats,
                hits_distribution,
            )

        return BacktestResultDTO(
            strategy.__class__.__name__,
            test_size,
            total_investment,
            total_earnings,
            total_earnings - total_investment,
            hits_distribution,
        )
