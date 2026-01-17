import json
import os
import time
from rich.console import Console
from src.domain.interfaces import ILotteryStrategy
from src.domain.dtos import DrawHistoryDTO, PredictionConfigDTO, BacktestResultDTO
from src.core.rules import MelateRetroRules
from src.data_access.report import SniperReport


class BacktestEngine:
    """Motor V6.1.5: Telemetría Sniper Refinada (Contraste de Proximidad)."""

    def __init__(self):
        self.rules = MelateRetroRules()
        self.console = Console()
        self.audit_history = []

    def run(self, strategy, history, config, verbose=False, pre_process_strategy=None):
        start_time_total = time.time()
        self.audit_history = []
        total_investment, total_earnings = 0.0, 0.0
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
            draw_start = time.time()
            funnel_stats["total_draws"] += 1
            _, target_draw, target_id = full_history[i]
            target_set = set(target_draw[:6])

            past_data = full_history[:i]
            p_dates, p_nums, p_ids = zip(*past_data)
            current_history = DrawHistoryDTO(list(p_dates), list(p_nums), list(p_ids))

            # FASE 1: REDUCCIÓN
            current_univ_size = 0
            if pre_process_strategy:
                config.filter_overrides["verbose"] = False
                univ_result = pre_process_strategy.predict(current_history, config)
                if hasattr(univ_result, "metadata"):
                    meta = univ_result.metadata
                    current_univ_size = meta.get("final_size", 0)
                    config.raw_universe_ptr = meta.get("raw_ndarray")

            # FASE 3-4: SELECCIÓN
            prediction = strategy.predict(current_history, config)
            audit = {}
            if hasattr(strategy, "audit_winner"):
                audit = strategy.audit_winner(current_history, config, target_draw)
                audit["draw_id"] = int(target_id)
                audit["univ_size"] = current_univ_size

            # Validación de premios
            max_hits_captured = 0
            for ticket in prediction.tickets:
                total_investment += self.rules.ticket_cost
                h_nat, h_add = self.rules.validate_ticket(ticket, target_draw)
                total_earnings += self.rules.calculate_prize(h_nat, h_add)
                hits_distribution[h_nat] += 1
                if h_nat > max_hits_captured:
                    max_hits_captured = h_nat

            if audit.get("proximity", -1) == 0:
                if max_hits_captured == 6:
                    funnel_stats["captured_gold"] += 1
                elif max_hits_captured == 5:
                    funnel_stats["captured_silver"] += 1
                elif max_hits_captured == 4:
                    funnel_stats["captured_bronze"] += 1

            draw_elapsed = time.time() - draw_start

            # --- RENDERIZADO SNIPER V6.1.5 (Limpieza de decimales y Foco en Distancia) ---
            if verbose:
                is_hit = audit.get("proximity") == 0
                st_col = "bold green" if is_hit else "bold red"
                emoji = "🎯 HIT" if is_hit else "❌ MISSED"

                # Formateo de Hits
                h_val = audit.get("hits", 0)
                h_col = (
                    "bold yellow" if h_val >= 5 else "cyan" if h_val == 4 else "white"
                )

                # Lógica de Distancia: Gris claro por defecto, Amarillo si < 10
                dist_val = audit.get("proximity", 0)
                dist_col = "bold yellow" if dist_val < 10 else "grey70"

                # Construcción de la línea limpia
                self.console.print(
                    f"[bold blue]#{target_id}[/] | "
                    f"U: [bold white]{audit.get('univ_size', 0):>6,d}[/] | "
                    f"[{h_col}]{h_val}/6[/] | "
                    f"AI: [white]{audit.get('ai_score', 0):.4f}[/] | "
                    f"Geo: [white]{audit.get('geo_score', 0):.4f}[/] | "
                    f"Rank: [bold white]#{audit.get('rank', 0):<5,d}[/] "
                    f"([dim]P{audit.get('percentile', 0):.1f}%[/]) | "
                    f"Dist: [{dist_col}]{dist_val:<4,d}[/] | "
                    f"[{st_col}]{emoji}[/] | [dim]⏱ {draw_elapsed:.2f}s[/dim]"
                )

            self.audit_history.append(audit)

        # Resumen Final
        end_time_total = time.time()
        total_seconds = end_time_total - start_time_total
        minutes, seconds = divmod(total_seconds, 60)

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
            self.console.print(
                f"\n🚀 [bold cyan]BENCHMARK:[/bold cyan] {int(minutes)}m {seconds:.1f}s total ([white]{total_seconds/test_size:.2f}s/sorteo[/white])\n"
            )

        return BacktestResultDTO(
            strategy.__class__.__name__,
            test_size,
            total_investment,
            total_earnings,
            total_earnings - total_investment,
            hits_distribution,
        )
