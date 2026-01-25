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
CYAN, GREEN, RED, YELLOW, RESET = (Fore.CYAN, Fore.GREEN, Fore.RED, Fore.YELLOW, Style.RESET_ALL)

class StrategyOptimizer:
    """Optimizador V8.0: Calibración de Precisión para el Golden Ratio."""

    def __init__(self):
        self.backtester = BacktestEngine()
        self.reducer = UniverseReductionStrategy()

    def _print_progress(self, current, total, value, label="Iter", is_money=True, u_size=0):
        percent = int((current + 1) / (total if total > 0 else 1) * 100)
        val_str = f"${value:,.0f}" if is_money else f"{value:.4f}"
        color = GREEN if (is_money and value > 0) else (RED if value < 0 else YELLOW)
        
        u_color = GREEN if u_size <= 100000 else RED
        u_info = f" | Universe: {u_color}{u_size:,}{RESET}" if u_size > 0 else ""

        bar = "█" * (20 * (current + 1) // (total if total > 0 else 1))
        sys.stdout.write(
            f"\r   {CYAN}[{bar:<20}] {percent}%{RESET} | {label} {current+1}/{total} | ROI: {color}{val_str}{RESET}{u_info}"
        )
        sys.stdout.flush()

    def optimize_filters(self, history: DrawHistoryDTO, draws_to_test: int = 20) -> Dict[str, Any]:
        """FASE 1: Calibración de la Malla Posicional y Agregación."""
        print(f"\n{CYAN}🔬 FASE 1: Calibración de Filtros (Búsqueda del Golden Ratio)...{RESET}")

        # Espacio de búsqueda refinado basado en el histórico real
        sum_ranges = [(108, 132), (110, 130), (112, 128)]
        f1_limits = [10, 12, 14]
        f6_limits = [28, 30, 32]
        ac_limits = [7, 8]

        combinations = list(itertools.product(sum_ranges, f1_limits, f6_limits, ac_limits))
        
        best_score = -float("inf")
        best_params = BEST_SETTINGS.copy()
        
        strategy = HeuristicSelectorStrategy()
        config = PredictionConfigDTO(TOTAL_BALLS, TICKET_SIZE, 15, draws_to_test)

        for i, (s_range, f1, f6, ac) in enumerate(combinations):
            params = {
                "sum_min": s_range[0],
                "sum_max": s_range[1],
                "f1_max": f1,
                "f6_min": f6,
                "ac_min": ac,
                "verbose": False
            }
            
            # Prueba rápida de volumen sobre el último sorteo
            # Si el volumen es > 150k, descartamos por ineficiencia antes de backtestear
            test_universe = self.reducer.reduce(history, params, verbose=False)
            u_size = len(test_universe)
            
            if u_size > 150000:
                self._print_progress(i, len(combinations), 0, "Skip", u_size=u_size)
                continue

            config.filter_overrides = params
            res = self.backtester.run(
                strategy,
                history,
                config,
                pre_process_strategy=self.reducer,
            )
            
            # MÉTRICA DE ÉXITO: ROI ponderado por el inverso del volumen
            # Castiga universos grandes y premia los que caben en < 100k
            efficiency_multiplier = 1.0 if u_size <= 100000 else (100000 / u_size)
            current_score = res.net_balance * efficiency_multiplier

            self._print_progress(i, len(combinations), res.net_balance, "Filtros", u_size=u_size)

            if current_score > best_score:
                best_score = current_score
                best_params = params.copy()

        print(f"\n   ✅ Mejor Configuración Encontrada:")
        for k, v in best_params.items():
            print(f"      {k}: {v}")
        
        return best_params

    def optimize_full_stack(self, history: DrawHistoryDTO, draws_to_test: int = 20):
        """Misión de Optimización Total."""
        print(f"\n{YELLOW}🚀 INICIANDO OPTIMIZACIÓN FULL STACK (V8.0 Precision){RESET}")
        start_time = time.time()

        # En esta sesión nos enfocamos en filtros para ganar el reto del universo
        cfg = self.optimize_filters(history, draws_to_test)

        elapsed = time.time() - start_time
        print(f"\n{GREEN}📊 TIEMPO TOTAL DE CALIBRACIÓN: {elapsed/60:.1f} min{RESET}")
        return cfg