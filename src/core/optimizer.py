import itertools
from typing import Dict, Any
from src.domain.dtos import DrawHistoryDTO, PredictionConfigDTO
from src.domain.interfaces import ILotteryStrategy
from src.core.backtester import BacktestEngine

# Colores ANSI para la terminal
CYAN = "\033[96m"
GREEN = "\033[92m"
YELLOW = "\033[93m"
RED = "\033[91m"
RESET = "\033[0m"


class StrategyOptimizer:
    """
    Motor de búsqueda de hiperparámetros (Grid Search).
    """

    def __init__(self, strategy: ILotteryStrategy, history: DrawHistoryDTO):
        self.strategy = strategy
        self.history = history
        self.backtester = BacktestEngine()

    def run_grid_search(self, config_base: PredictionConfigDTO) -> Dict[str, Any]:
        print(f"{CYAN}🔎 INICIANDO BÚSQUEDA DE ESTRATEGIA ÓPTIMA (GRID SEARCH){RESET}")

        # 1. Definimos los parámetros a probar
        # NOTA: Puedes editar estos rangos para probar más o menos cosas
        param_grid = {
            "sum_min": [90, 100, 108, 115],  # ¿Campana Gauss ajustada o amplia?
            "ac_min": [4, 5, 6],  # ¿Complejidad aritmética?
            "even_min": [2, 3],  # ¿Pares mínimos?
            "inertia_min": [0, 1],  # ¿Forzar número del sorteo anterior?
        }

        keys, values = zip(*param_grid.items())
        permutations = [dict(zip(keys, v)) for v in itertools.product(*values)]

        total_tests = len(permutations)
        print(f"⚙️  Se probarán {total_tests} configuraciones distintas.\n")

        best_roi = -float("inf")
        best_params = {}

        # 2. Bucle de prueba
        for i, params in enumerate(permutations):
            print(f"🧪 Test {i+1}/{total_tests}: {params} ... ", end="", flush=True)

            # Clonamos la config base y le inyectamos los overrides
            current_config = PredictionConfigDTO(
                total_balls=config_base.total_balls,
                ticket_size=config_base.ticket_size,
                num_tickets=config_base.num_tickets,
                backtest_size=config_base.backtest_size,
                filter_overrides=params,  # <--- Aquí ocurre la magia
            )

            # Ejecutamos Backtest SILENCIOSO
            try:
                result = self.backtester.run(
                    self.strategy, self.history, current_config, verbose=False
                )
                roi = result.net_balance

                # Visualización rápida de resultados
                if roi > best_roi:
                    best_roi = roi
                    best_params = params
                    print(f"{GREEN}¡NUEVO RÉCORD! ROI: ${roi:,.2f}{RESET}")
                elif roi > 0:
                    print(f"{YELLOW}PROFIT: ${roi:,.2f}{RESET}")
                else:
                    print(f"{RED}${roi:,.2f}{RESET}")

            except Exception as e:
                print(f"{RED}ERROR: {e}{RESET}")

        # 3. Reporte Final
        print(f"\n{YELLOW}🏆 MEJOR CONFIGURACIÓN ENCONTRADA:{RESET}")
        print(f"   Config: {best_params}")
        print(f"   Balance Neto: ${best_roi:,.2f}")

        return best_params
