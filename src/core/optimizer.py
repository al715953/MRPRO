import itertools
import sys
import time
from typing import Dict, Any, List
from src.domain.dtos import DrawHistoryDTO, PredictionConfigDTO
from src.strategies.universe_reduction import UniverseReductionStrategy
from src.strategies.heuristic_selector import HeuristicSelectorStrategy
from src.core.backtester import BacktestEngine
from src.data_access.config import TOTAL_BALLS, TICKET_SIZE

# Colores ANSI
CYAN = "\033[96m"
GREEN = "\033[92m"
RED = "\033[91m"
RESET = "\033[0m"

class StrategyOptimizer:
    """
    Optimizador Bimodal V3.
    Modo 1: Filtros de Reducción (Universo).
    Modo 2: Pesos Heurísticos (Ranking).
    """

    def __init__(self):
        self.backtester = BacktestEngine()

    def optimize_filters(self, history: DrawHistoryDTO) -> Dict[str, Any]:
        """
        Busca los mejores rangos de Suma, Pares y Primos.
        """
        print(f"\n{CYAN}🚀 Iniciando Optimización de FILTROS (Fase 1)...{RESET}")
        
        # Grid reducido para demostración (ampliar si tienes tiempo)
        sums = [(100, 180), (120, 160)]
        evens = [(2, 4), (3, 3)] # (min, max)
        acs = [4, 5]
        
        combinations = list(itertools.product(sums, evens, acs))
        total_tests = len(combinations)
        
        best_roi = -float('inf')
        best_params = {}

        strategy = UniverseReductionStrategy()
        
        # Configuración base SILENCIOSA
        config_base = PredictionConfigDTO(
            total_balls=TOTAL_BALLS,
            ticket_size=TICKET_SIZE,
            num_tickets=10,
            backtest_size=15, # Test rápido de 15 sorteos
            filter_overrides={"verbose": False} # IMPORTANTE: Silenciar logs
        )

        for i, (sum_r, even_r, ac_val) in enumerate(combinations):
            params = {
                "sum_min": sum_r[0], "sum_max": sum_r[1],
                "even_min": even_r[0], "even_max": even_r[1],
                "ac_min": ac_val,
                "verbose": False # Refuerzo
            }
            config_base.filter_overrides = params

            try:
                # Ejecutamos Backtest
                # Nota: UniverseReduction no genera tickets finales, solo reduce.
                # Para medir ROI, necesitamos una estrategia de selección tonta o usar CoverageTester.
                # Pero asumiremos que BacktestEngine maneja Universe como estrategia única
                # devolviendo los primeros N del universo.
                result = self.backtester.run(
                    strategy, history, config_base, verbose=False
                )
                
                self._print_progress(i, total_tests, result.net_balance)

                if result.net_balance > best_roi:
                    best_roi = result.net_balance
                    best_params = params.copy()

            except Exception as e:
                pass

        print("\n")
        return best_params

    def optimize_heuristics(self, history: DrawHistoryDTO) -> Dict[str, Any]:
        """
        Busca la mejor distribución de pesos para la Heurística.
        w_cluster + w_hotness + w_balance = 1.0
        """
        print(f"\n{CYAN}🚀 Iniciando Optimización de PESOS HEURÍSTICOS (Fase 2)...{RESET}")
        
        # Generar combinaciones de pesos que sumen 1.0
        # Steps de 0.1
        weights = []
        for w1 in range(1, 9): # 0.1 a 0.8
            for w2 in range(1, 9):
                if w1 + w2 < 10:
                    w3 = 10 - (w1 + w2)
                    weights.append((w1/10.0, w2/10.0, w3/10.0))
        
        total_tests = len(weights)
        best_roi = -float('inf')
        best_params = {}

        strategy = HeuristicSelectorStrategy()
        
        # Necesitamos ejecutar primero la reducción de universo una vez para no repetirla
        # pero por simplicidad de código, dejaremos que corra todo.
        config_base = PredictionConfigDTO(
            total_balls=TOTAL_BALLS,
            ticket_size=TICKET_SIZE,
            num_tickets=10,
            backtest_size=20,
            filter_overrides={"verbose": False} 
        )

        for i, (wc, wh, wb) in enumerate(weights):
            params = {
                "w_cluster": wc,
                "w_hotness": wh,
                "w_balance": wb,
                "verbose": False
            }
            # Combinamos con los mejores filtros conocidos (o defaults)
            params.update({"sum_min": 108, "sum_max": 180}) # Hardcode de seguridad
            
            config_base.filter_overrides = params

            try:
                # Usamos HeuristicSelectorStrategy que ahora lee estos pesos
                result = self.backtester.run(
                    strategy, history, config_base, 
                    pre_process_strategy=UniverseReductionStrategy(), # Pipeline completo
                    verbose=False
                )
                
                self._print_progress(i, total_tests, result.net_balance)

                if result.net_balance > best_roi:
                    best_roi = result.net_balance
                    best_params = params.copy()

            except Exception as e:
                pass

        print("\n")
        return best_params

    def _print_progress(self, current, total, roi):
        percent = int((current + 1) / total * 100)
        status_color = GREEN if roi > 0 else RED
        bar_len = 20
        filled_len = int(bar_len * (current + 1) // total)
        bar = '█' * filled_len + '-' * (bar_len - filled_len)
        
        sys.stdout.write(f"\r{CYAN}[{bar}] {percent}%{RESET} | Iter {current+1}/{total} | ROI: {status_color}${roi:,.0f}{RESET}")
        sys.stdout.flush()