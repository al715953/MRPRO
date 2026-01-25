import itertools
import sys
import time
import numpy as np
from typing import Dict, Any
from colorama import Fore, Style

from src.domain.dtos import DrawHistoryDTO, PredictionConfigDTO
from src.data_access.config import TOTAL_BALLS, TICKET_SIZE, BEST_SETTINGS
from src.strategies.universe_reduction import UniverseReductionStrategy
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
    """
    Optimizador V8.2: Calibración Blindada.
    Soporta estrategias con y sin IA de forma transparente.
    """

    def __init__(self):
        self.backtester = BacktestEngine()
        self.reducer = UniverseReductionStrategy()

    def _print_progress(self, current, total, value, label="Iter", u_size=0):
        percent = int((current + 1) / (total if total > 0 else 1) * 100)
        val_str = f"${value:,.0f}"
        color = GREEN if value > 0 else (RED if value < 0 else YELLOW)

        u_color = GREEN if u_size <= 100000 else RED
        u_info = f" | Universe: {u_color}{u_size:,}{RESET}" if u_size > 0 else ""

        bar = "█" * (20 * (current + 1) // (total if total > 0 else 1))
        sys.stdout.write(
            f"\r   {CYAN}[{bar:<20}] {percent}%{RESET} | {label} {current+1}/{total} | ROI: {color}{val_str}{RESET}{u_info}"
        )
        sys.stdout.flush()

    def optimize_filters(
        self, history: DrawHistoryDTO, draws_to_test: int = 40
    ) -> Dict[str, Any]:
        """Búsqueda del Golden Ratio usando el Selector Heurístico como base."""
        print(
            f"\n{CYAN}🔬 FASE 1: Calibración de Malla (Hardware: {self.reducer.backend_name}){RESET}"
        )

        search_grid = {
            "sum_ranges": [(108, 132), (112, 128)],
            "f1_limits": [10, 12],
            "f6_limits": [30, 32],
            "ac_limits": [7, 8],
        }

        combinations = list(
            itertools.product(
                search_grid["sum_ranges"],
                search_grid["f1_limits"],
                search_grid["f6_limits"],
                search_grid["ac_limits"],
            )
        )

        best_score = -float("inf")
        best_params = BEST_SETTINGS.copy()

        # Inicializamos el selector asegurando que tenga la interfaz correcta
        selector = HeuristicSelectorStrategy()
        config = PredictionConfigDTO(
            TOTAL_BALLS, TICKET_SIZE, num_tickets=20, backtest_size=draws_to_test
        )

        for i, (s_range, f1, f6, ac) in enumerate(combinations):
            params = {
                "sum_min": s_range[0],
                "sum_max": s_range[1],
                "f1_max": f1,
                "f6_min": f6,
                "ac_min": ac,
            }

            # Prueba de volumen rápida en GPU/CPU
            universe_dto = self.reducer.predict(
                history,
                PredictionConfigDTO(
                    TOTAL_BALLS, TICKET_SIZE, 20, filter_overrides=params
                ),
            )
            u_size = len(universe_dto.tickets)

            if u_size > 180000:
                self._print_progress(i, len(combinations), 0, "Skip", u_size=u_size)
                continue

            # Backtest con la configuración candidata
            config.filter_overrides = params
            res = self.backtester.run(
                selector, history, config, pre_process_strategy=self.reducer
            )

            # Score de eficiencia: ROI penalizado por volumen
            efficiency_score = res.net_balance * (
                1.0 if u_size <= 100000 else (100000 / u_size)
            )

            self._print_progress(
                i, len(combinations), res.net_balance, "Filtros", u_size=u_size
            )

            if efficiency_score > best_score:
                best_score = efficiency_score
                best_params = params.copy()
                best_params["u_size_avg"] = u_size

        return best_params
