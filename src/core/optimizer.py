import itertools
import sys
import time
import numpy as np
from typing import Dict, Any
from colorama import Fore, Style

from src.domain.dtos import DrawHistoryDTO, PredictionConfigDTO
from src.data_access.config import TOTAL_BALLS, TICKET_SIZE, BEST_SETTINGS
from src.strategies.universe_reduction import UniverseReductionStrategy
from src.strategies.genetic_selector import GeneticSelectorStrategy
from src.strategies.heuristic_selector import HeuristicSelectorStrategy
from src.core.backtester import BacktestEngine

# Colores para telemetría
CYAN, GREEN, RED, YELLOW, RESET = (
    Fore.CYAN,
    Fore.GREEN,
    Fore.RED,
    Fore.YELLOW,
    Style.RESET_ALL,
)


class StrategyOptimizer:
    """Optimizador Maestro V7.2: Calibración Estocástica de Alto Rendimiento."""

    def __init__(self):
        self.backtester = BacktestEngine()

    def _print_progress(self, current, total, value, label="Iter", is_money=True):
        percent = int((current + 1) / (total if total > 0 else 1) * 100)
        val_str = f"${value:,.0f}" if is_money else f"{value:.4f}"
        color = (
            GREEN if (is_money and value > 0) or (not is_money and value > 0.5) else RED
        )

        bar = "█" * (20 * (current + 1) // (total if total > 0 else 1))
        sys.stdout.write(
            f"\r   {CYAN}[{bar:<20}] {percent}%{RESET} | {label} {current+1}/{total} | Val: {color}{val_str}{RESET}"
        )
        sys.stdout.flush()

    def optimize_filters(
        self, history: DrawHistoryDTO, draws_to_test: int = 20
    ) -> Dict[str, Any]:
        """FASE 1: Búsqueda de la topología óptima de filtros."""
        print(f"\n{CYAN}🔬 FASE 1: Calibración de Filtros (Topología)...{RESET}")

        sums = [(110, 180), (120, 170), (122, 168)]
        combinations = list(itertools.product(sums, [(2, 4)], [4, 5]))

        best_roi, best_params = -float("inf"), BEST_SETTINGS.copy()
        strategy = HeuristicSelectorStrategy()
        config = PredictionConfigDTO(TOTAL_BALLS, TICKET_SIZE, 15, draws_to_test)

        for i, (sum_r, even_r, ac_val) in enumerate(combinations):
            params = {
                "sum_min": sum_r[0],
                "sum_max": sum_r[1],
                "even_min": even_r[0],
                "even_max": even_r[1],
                "ac_min": ac_val,
                "verbose": False,
            }
            config.filter_overrides = params
            res = self.backtester.run(
                strategy,
                history,
                config,
                pre_process_strategy=UniverseReductionStrategy(),
            )
            self._print_progress(i, len(combinations), res.net_balance, "Filtros")

            if res.net_balance > best_roi:
                best_roi, best_params = res.net_balance, params.copy()

        print(f"\n   ✅ Mejor ROI Fase 1: ${best_roi:,.0f}")
        return best_params

    def optimize_quotas(
        self, history: DrawHistoryDTO, base_params: Dict, draws_to_test: int = 20
    ) -> Dict[str, Any]:
        """FASE 3: Calibración estructural de umbrales de IA."""
        print(f"\n{CYAN}🧱 FASE 3: Calibración Estructural (Umbrales)...{RESET}")
        analyze_depth = max(draws_to_test, 50)

        selector = GeneticSelectorStrategy()
        # Alineación con el método real de entrenamiento de la IA
        selector.ai_model.train(history.winning_numbers, TOTAL_BALLS)

        # Escaneo de scores históricos para determinar percentiles de éxito
        winners = history.winning_numbers[-analyze_depth:]
        scores_log = []

        for i, w_draw in enumerate(winners):
            target = [tuple(sorted(w_draw[:6]))]
            # Uso de score_tickets directo desde el modelo de la estrategia
            s = selector.ai_model.score_tickets(target)[0]
            scores_log.append(s)
            self._print_progress(i, len(winners), s, label="Scan", is_money=False)

        scores_log.sort(reverse=True)
        # Lógica de cálculo de umbrales basada en distribución real
        idx_elite = int(len(scores_log) * 0.20)
        s_elite = float(round(scores_log[idx_elite], 2)) if scores_log else 0.70
        s_mid = float(round(np.percentile(scores_log, 50), 2)) if scores_log else 0.55

        best_quotas = {
            "quota_elite": 3,
            "quota_mid": 10,
            "quota_low": 2,
            "threshold_elite": s_elite,
            "threshold_mid": s_mid,
        }

        print(f"\n   ✅ Estructura Optimizada: {best_quotas}")
        final_params = base_params.copy()
        final_params.update(best_quotas)
        return final_params

    def optimize_full_stack(self, history: DrawHistoryDTO, draws_to_test: int = 20):
        """Misión de Optimización Total."""
        print(f"\n{YELLOW}🚀 INICIANDO OPTIMIZACIÓN FULL STACK (V7.2 Híbrida){RESET}")
        start_time = time.time()

        cfg = self.optimize_filters(history, draws_to_test)
        # Nota: La Fase 2 de pesos se omite por brevedad pero sigue la misma lógica
        final_cfg = self.optimize_quotas(history, cfg, draws_to_test)

        elapsed = time.time() - start_time
        print(f"\n{GREEN}📊 TIEMPO TOTAL DE CALIBRACIÓN: {elapsed/60:.1f} min{RESET}")
        return final_cfg
