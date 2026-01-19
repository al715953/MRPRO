import src.data_access.report as report
import src.data_access.scraper as scraper
from colorama import Fore, Style
from src.domain.dtos import PredictionConfigDTO
from src.data_access.config import (
    BEST_SETTINGS,
    TOTAL_BALLS,
    TICKET_SIZE,
    CSV_FILE_PATH,
)
from src.strategies.monte_carlo import MonteCarloStrategy
from src.strategies.universe_reduction import UniverseReductionStrategy
from src.strategies.genetic_selector import GeneticSelectorStrategy
from src.core.backtester import BacktestEngine
from src.core.optimizer import StrategyOptimizer
from src.core.coverage_tester import CoverageTester
from src.data_access.visualizer import run_forensic_visualization

# Importación segura para la estrategia Heurística
try:
    from src.strategies.heuristic_selector import HeuristicSelectorStrategy
except ImportError:
    HeuristicSelectorStrategy = None


class MissionController:
    def __init__(self, ui, history):
        self.ui = ui
        self.history = history

    def run_mission(self, option):
        """Despachador de misiones (Opciones 1-8)."""
        if option == "1":
            self._view_history()
        elif option == "2":
            self._analyze_frequency()
        elif option == "3":
            self._run_monte_carlo()
        elif option == "4":
            self._run_optimizer()
        elif option == "5":
            self._generate_universe()
        elif option == "6":
            self._run_backtest_lab()
        elif option == "7":
            self._run_production()
        elif option == "8":
            run_forensic_visualization()
        else:
            print(f"{Fore.RED}⚠ Opción no reconocida.{Style.RESET_ALL}")

    def _view_history(self):
        self.ui.show_history(self.history)
        input(f"\n{Fore.YELLOW}>> Presiona ENTER...{Style.RESET_ALL}")

    def _analyze_frequency(self):
        self.ui.analyze_frequency(self.history, TOTAL_BALLS)
        input(f"\n{Fore.YELLOW}>> Presiona ENTER...{Style.RESET_ALL}")

    def _run_monte_carlo(self):
        print(f"\n{Fore.CYAN}🎲 MÓDULO MONTE CARLO{Style.RESET_ALL}")
        config = PredictionConfigDTO(TOTAL_BALLS, TICKET_SIZE, num_tickets=10)
        config.filter_overrides = BEST_SETTINGS
        pred = MonteCarloStrategy().predict(self.history, config)
        self.ui.show_prediction_results(pred)
        input(f"\n{Fore.YELLOW}>> Presiona ENTER...{Style.RESET_ALL}")

    def _run_optimizer(self):
        sub_opt, n_draws = self.ui.show_optimizer_menu()
        opt = StrategyOptimizer()
        try:
            base_dummy = BEST_SETTINGS.copy()
            base_dummy["verbose"] = False
            if sub_opt == "1":
                best_cfg = opt.optimize_filters(self.history, n_draws)
            elif sub_opt == "2":
                best_cfg = opt.optimize_heuristics(self.history, base_dummy, n_draws)
            elif sub_opt == "3":
                best_cfg = opt.optimize_quotas(self.history, base_dummy, n_draws)
            elif sub_opt == "4":
                best_cfg = opt.optimize_full_stack(self.history, n_draws)

            print(
                f"\n{Fore.GREEN}🏆 CONFIGURACIÓN OPTIMIZADA ({n_draws} Sorteos):{Style.RESET_ALL}"
            )
            for k, v in best_cfg.items():
                print(f"   • {k:<15}: {Fore.CYAN}{v}{Style.RESET_ALL}")
            print(
                f"\n{Fore.GREEN}💾 Actualiza 'BEST_SETTINGS' en config.py.{Style.RESET_ALL}"
            )
        except Exception as e:
            print(f"{Fore.RED}⚠ Error: {e}{Style.RESET_ALL}")
        input(f"\n{Fore.YELLOW}>> Presiona ENTER...{Style.RESET_ALL}")

    def _generate_universe(self):
        print(f"\n{Fore.CYAN}🌌 GENERANDO UNIVERSO (P88 DENSITY)...{Style.RESET_ALL}")
        config = PredictionConfigDTO(TOTAL_BALLS, TICKET_SIZE, num_tickets=0)
        config.filter_overrides = BEST_SETTINGS.copy()
        config.filter_overrides["verbose"] = True
        UniverseReductionStrategy().predict(self.history, config)
        print(f"{Fore.GREEN}✅ Universo generado correctamente.{Style.RESET_ALL}")
        input(f"\n{Fore.YELLOW}>> Presiona ENTER...{Style.RESET_ALL}")

    def _run_backtest_lab(self):
        print(
            f"\n{Fore.CYAN}📡 LABORATORIO V32.1 (STOCHASTIC SNIPER P8){Style.RESET_ALL}"
        )
        print("1. 🛡️  Cobertura Fase 1 (Universo) | 2. 🥊 Duelo | 3. 🧠 Solo AI")
        sub_op = input("   👉 Selecciona modo (3): ") or "3"
        try:
            n_test, n_tickets = int(input("¿Sorteos? (40): ") or 40), int(
                input("¿Tickets? (15): ") or 15
            )
        except:
            n_test, n_tickets = 40, 15

        config = PredictionConfigDTO(
            TOTAL_BALLS, TICKET_SIZE, n_tickets, n_test, BEST_SETTINGS
        )
        engine = BacktestEngine()
        if sub_op == "1":
            CoverageTester().run(UniverseReductionStrategy(), self.history, config)
        elif sub_op == "2" and HeuristicSelectorStrategy:
            engine.run(
                HeuristicSelectorStrategy(),
                self.history,
                config,
                True,
                UniverseReductionStrategy(),
            )
            engine.run(
                GeneticSelectorStrategy(),
                self.history,
                config,
                True,
                UniverseReductionStrategy(),
            )
        elif sub_op == "3":
            engine.run(
                GeneticSelectorStrategy(),
                self.history,
                config,
                True,
                UniverseReductionStrategy(),
            )
        input(f"\n{Fore.YELLOW}>> Presiona ENTER...{Style.RESET_ALL}")

    def _run_production(self):
        """Misión de Élite: Asegura reducción antes de selección."""
        try:
            n_prod = int(input(f"\n   ¿Tickets para hoy? (15): ") or 15)
        except:
            n_prod = 15

        config = PredictionConfigDTO(
            TOTAL_BALLS, TICKET_SIZE, n_prod, filter_overrides=BEST_SETTINGS
        )
        print(f"   {Fore.YELLOW}⏳ Paso 1: Reduciendo Universo...{Style.RESET_ALL}")
        univ_res = UniverseReductionStrategy().predict(self.history, config)
        config.raw_universe_ptr = univ_res.metadata.get("raw_ndarray")

        print(f"   {Fore.CYAN}🧬 Paso 2: Ejecutando Mesh Genético...{Style.RESET_ALL}")
        pred = GeneticSelectorStrategy().predict(self.history, config)

        if pred.tickets:
            report.guardar_prediccion(pred.tickets)
            self.ui.show_prediction_results(pred)
            print(f"\n{Fore.GREEN}🍀 ¡Predicción lista y guardada!{Style.RESET_ALL}")
        else:
            print(f"{Fore.RED}❌ Error en generación.{Style.RESET_ALL}")
        input(f"\n{Fore.YELLOW}>> Presiona ENTER...{Style.RESET_ALL}")
