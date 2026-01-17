import json
import os
from rich.console import Console
from src.domain.interfaces import ILotteryStrategy
from src.domain.dtos import DrawHistoryDTO, PredictionConfigDTO, BacktestResultDTO
from src.core.rules import MelateRetroRules
from src.data_access.report import SniperReport


class BacktestEngine:
    """Motor V6.1.1: Estable, Secuencial y libre de errores de RAM."""

    def __init__(self):
        self.rules = MelateRetroRules()
        self.console = Console()
        self.audit_history = []

    def run(self, strategy, history, config, verbose=False, pre_process_strategy=None):
        self.audit_history = []
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
            target_set = set(target_draw[:6])

            past_data = full_history[:i]
            p_dates, p_nums, p_ids = zip(*past_data)
            current_history = DrawHistoryDTO(list(p_dates), list(p_nums), list(p_ids))

            # FASE 1: REDUCCIÓN (POTENCIAL)
            current_univ_size = 0
            if pre_process_strategy:
                config.filter_overrides["verbose"] = False
                univ_result = pre_process_strategy.predict(current_history, config)

                if hasattr(univ_result, "metadata"):
                    meta = univ_result.metadata
                    current_univ_size = meta.get("final_size", 0)
                    config.raw_universe_ptr = meta.get("raw_ndarray")

                # CORRECCIÓN DE LA VARIABLE NameError: max_hit_univ
                max_hit_univ = 0
                for t in univ_result.tickets:
                    h = len(set(t) & target_set)
                    if h > max_hit_univ:
                        max_hit_univ = h
                    if h == 6:
                        break

                if max_hit_univ == 6:
                    funnel_stats["opp_gold"] += 1
                elif max_hit_univ == 5:
                    funnel_stats["opp_silver"] += 1
                elif max_hit_univ == 4:
                    funnel_stats["opp_bronze"] += 1

            # FASE 3: SELECCIÓN
            prediction = strategy.predict(current_history, config)

            audit = {}
            if hasattr(strategy, "audit_winner"):
                audit = strategy.audit_winner(current_history, config, target_draw)
                audit["draw_id"] = int(target_id)
                audit["univ_size"] = current_univ_size

            # Cálculo de premios real
            max_hits_captured = 0
            for ticket in prediction.tickets:
                total_investment += self.rules.ticket_cost
                h_nat, h_add = self.rules.validate_ticket(ticket, target_draw)
                total_earnings += self.rules.calculate_prize(h_nat, h_add)
                hits_distribution[h_nat] += 1
                if h_nat > max_hits_captured:
                    max_hits_captured = h_nat

            # Política de Honestidad (Proximity 0)
            if audit.get("proximity", -1) == 0:
                if max_hits_captured == 6:
                    funnel_stats["captured_gold"] += 1
                elif max_hits_captured == 5:
                    funnel_stats["captured_silver"] += 1
                elif max_hits_captured == 4:
                    funnel_stats["captured_bronze"] += 1

            if verbose:
                SniperReport.render_draw_summary(
                    getattr(prediction, "metadata", {}), audit
                )

            self.audit_history.append(audit)

        # PERSISTENCIA
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
