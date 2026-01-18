import itertools
import sys
import time
import traceback
import numpy as np
from typing import Dict, Any, List, Tuple
from colorama import Fore, Style

from src.domain.dtos import DrawHistoryDTO, PredictionConfigDTO
from src.data_access.config import TOTAL_BALLS, TICKET_SIZE, BEST_SETTINGS
from src.strategies.universe_reduction import UniverseReductionStrategy
from src.strategies.genetic_selector import GeneticSelectorStrategy
from src.strategies.heuristic_selector import HeuristicSelectorStrategy
from src.core.backtester import BacktestEngine

CYAN = Fore.CYAN
GREEN = Fore.GREEN
RED = Fore.RED
YELLOW = Fore.YELLOW
RESET = Style.RESET_ALL


class StrategyOptimizer:
    """
    Optimizador Híbrido V7.1 (UX Improved).
    Pasa contexto de iteración a las estrategias para mejorar las barras de progreso.
    """

    def __init__(self):
        self.backtester = BacktestEngine()

    def _print_progress(
        self, current, total, value, label="Iter", extra_info="", is_money=True
    ):
        if total == 0:
            total = 1
        percent = int((current + 1) / total * 100)

        if is_money:
            val_str = f"${value:,.0f}"
            status_color = GREEN if value > 0 else RED
        else:
            val_str = f"{value:.4f}"
            status_color = YELLOW

        bar_len = 20
        filled_len = int(bar_len * (current + 1) // total)
        bar = "█" * filled_len + "-" * (bar_len - filled_len)

        sys.stdout.write(
            f"\r   {CYAN}[{bar}] {percent}%{RESET} | {label} {current+1}/{total} | {extra_info} | Val: {status_color}{val_str}{RESET}"
        )
        sys.stdout.flush()

    def optimize_filters(
        self, history: DrawHistoryDTO, draws_to_test: int = 20
    ) -> Dict[str, Any]:
        print(f"\n{CYAN}🔬 FASE 1: Calibración de Filtros (Topología)...{RESET}")

        sums = [(110, 180), (120, 170), (122, 168)]
        evens = [(2, 4)]
        acs = [4, 5]
        combinations = list(itertools.product(sums, evens, acs))

        best_roi = -float("inf")
        best_params = BEST_SETTINGS.copy()
        found_better = False
        strategy = HeuristicSelectorStrategy()

        config = PredictionConfigDTO(
            TOTAL_BALLS, TICKET_SIZE, 15, draws_to_test, {"verbose": False}
        )

        total_iter = len(combinations)
        for i, (sum_r, even_r, ac_val) in enumerate(combinations):
            params = {
                "sum_min": sum_r[0],
                "sum_max": sum_r[1],
                "even_min": even_r[0],
                "even_max": even_r[1],
                "ac_min": ac_val,
                "verbose": False,
                # --- CONTEXTO DE ITERACIÓN ---
                "opt_iter": i + 1,
                "opt_total": total_iter,
                "opt_phase": "Fase 1",
            }
            config.filter_overrides = params

            try:
                res = self.backtester.run(
                    strategy,
                    history,
                    config,
                    pre_process_strategy=UniverseReductionStrategy(),
                    verbose=False,
                )
                self._print_progress(i, total_iter, res.net_balance, "Filtros")

                if res.net_balance > best_roi:
                    best_roi = res.net_balance
                    best_params = params.copy()
                    found_better = True
            except Exception as e:
                print(f"\n{RED}❌ Error en iteración {i}: {e}{RESET}")

        if not found_better:
            print(f"\n   ⚠️ Sin mejoras en Fase 1.")
        else:
            print(f"\n   ✅ Mejor ROI Fase 1: ${best_roi:,.0f}")
        return best_params

    def optimize_heuristics(
        self, history: DrawHistoryDTO, base_params: Dict, draws_to_test: int = 20
    ) -> Dict[str, Any]:
        print(f"\n{CYAN}⚖️  FASE 2: Sintonización de Pesos (Deep Search)...{RESET}")

        weights_grid = [
            (0.6, 0.4, 0.3),
            (0.8, 0.2, 0.0),
            (0.2, 0.8, 0.1),
            (0.1, 0.1, 0.9),
            (0.3, 0.3, 0.3),
            (0.9, 0.0, 0.1),
            (0.0, 0.9, 0.0),
            (0.5, 0.0, 0.5),
            (0.0, 0.5, 0.5),
        ]

        best_roi = -float("inf")
        best_params = base_params.copy()
        found_better = False
        strategy = GeneticSelectorStrategy()

        config = PredictionConfigDTO(
            TOTAL_BALLS, TICKET_SIZE, 15, draws_to_test, {"verbose": False}
        )

        total_iter = len(weights_grid)
        for i, (wc, wh, wa) in enumerate(weights_grid):
            params = base_params.copy()
            params.update(
                {
                    "w_cluster": wc,
                    "w_hotness": wh,
                    "w_ai": wa,
                    "verbose": False,
                    # --- CONTEXTO DE ITERACIÓN ---
                    "opt_iter": i + 1,
                    "opt_total": total_iter,
                    "opt_phase": "Fase 2",
                }
            )
            config.filter_overrides = params

            try:
                res = self.backtester.run(
                    strategy,
                    history,
                    config,
                    pre_process_strategy=UniverseReductionStrategy(),
                    verbose=False,
                )
                self._print_progress(i, total_iter, res.net_balance, "Pesos")

                if res.net_balance > best_roi:
                    best_roi = res.net_balance
                    best_params = params.copy()
                    found_better = True
            except Exception as e:
                print(f"\n{RED}❌ Error en iteración {i}: {e}{RESET}")

        if not found_better:
            print(f"\n   ⚠️ Sin mejoras en Fase 2.")
        else:
            print(f"\n   ✅ Mejor ROI Fase 2: ${best_roi:,.0f}")
        return best_params

    def optimize_quotas(
        self, history: DrawHistoryDTO, base_params: Dict, draws_to_test: int = 20
    ) -> Dict[str, Any]:
        # Fase 3 no usa reducción masiva repetitiva, así que no necesita cambios de UX críticos
        print(
            f"\n{CYAN}🧱 FASE 3: Calibración Estructural (Cuotas & Umbrales)...{RESET}"
        )
        analyze_depth = max(draws_to_test, 50)
        print(f"   📊 Analizando últimos {analyze_depth} sorteos...")

        strategy = GeneticSelectorStrategy()
        strategy._train_model(history, TOTAL_BALLS)

        scores_log = []
        scoring_weights = {
            "w_cluster": base_params.get("w_cluster", 0.6),
            "w_hotness": base_params.get("w_hotness", 0.4),
            "w_ai": base_params.get("w_ai", 0.3),
        }

        start_idx = max(0, len(history.winning_numbers) - analyze_depth)
        winners = history.winning_numbers[start_idx:]

        for i, w_draw in enumerate(winners):
            target = tuple(sorted(w_draw[:6]))
            ai_score = strategy.ai_model.score_tickets([target])[0]
            if hasattr(strategy, "_compute_v7_score"):
                try:
                    s, _, _ = strategy._compute_v7_score(
                        target, ai_score, scoring_weights
                    )
                except:
                    s, _, _ = strategy._compute_v7_score(target, ai_score)
            else:
                s = 0.5
            scores_log.append(s)
            self._print_progress(
                i,
                len(winners),
                s,
                label="Scan",
                extra_info=f"S:{s:.2f}",
                is_money=False,
            )

        scores_log.sort(reverse=True)
        if not scores_log:
            return base_params

        idx_elite = int(len(scores_log) * 0.20)
        suggested_elite_th = (
            scores_log[idx_elite] if idx_elite < len(scores_log) else 0.70
        )
        idx_mid = int(len(scores_log) * 0.80)
        suggested_mid_th = scores_log[idx_mid] if idx_mid < len(scores_log) else 0.55

        suggested_elite_th = float(round(suggested_elite_th, 2))
        suggested_mid_th = float(round(suggested_mid_th, 2))

        if suggested_elite_th > 0.85:
            suggested_elite_th = 0.75
        if suggested_mid_th < 0.40:
            suggested_mid_th = 0.45
        if suggested_elite_th <= suggested_mid_th:
            suggested_elite_th = suggested_mid_th + 0.05

        print(
            f"\n   📏 Umbrales Sugeridos: Elite >= {suggested_elite_th}, Mid >= {suggested_mid_th}"
        )

        zone_hits = {"Elite": 0, "Mid": 0, "Low": 0}
        for s in scores_log:
            if s >= suggested_elite_th:
                zone_hits["Elite"] += 1
            elif s >= suggested_mid_th:
                zone_hits["Mid"] += 1
            elif s >= (suggested_mid_th - 0.10):
                zone_hits["Low"] += 1

        total = sum(zone_hits.values()) or 1
        target_tickets = 15
        q_e = max(1, int(round((zone_hits["Elite"] / total) * target_tickets)))
        q_m = max(1, int(round((zone_hits["Mid"] / total) * target_tickets)))
        q_l = max(1, int(round((zone_hits["Low"] / total) * target_tickets)))

        while (q_e + q_m + q_l) > 15:
            if q_l > 1:
                q_l -= 1
            elif q_m > 1:
                q_m -= 1
            else:
                q_e -= 1
        while (q_e + q_m + q_l) < 15:
            q_m += 1

        best_quotas = {
            "quota_elite": q_e,
            "quota_mid": q_m,
            "quota_low": q_l,
            "threshold_elite": suggested_elite_th,
            "threshold_mid": suggested_mid_th,
        }

        print(f"   ✅ Estructura Optimizada: {best_quotas}")
        final_params = base_params.copy()
        final_params.update(best_quotas)
        return final_params

    def optimize_full_stack(self, history: DrawHistoryDTO, draws_to_test: int = 20):
        print(f"\n{YELLOW}🚀 INICIANDO OPTIMIZACIÓN FULL STACK (V7 Robust GPU){RESET}")
        start_time = time.time()
        cfg_1 = self.optimize_filters(history, draws_to_test)
        cfg_2 = self.optimize_heuristics(history, cfg_1, draws_to_test)
        final_cfg = self.optimize_quotas(history, cfg_2, draws_to_test)
        elapsed = time.time() - start_time
        print(f"\n\n{GREEN}📊 CONFIGURACIÓN MAESTRA RECOMENDADA:{RESET}")
        print(f"⏱️  Tiempo Total: {elapsed/60:.1f} min")
        # Print final report...
        return final_cfg
