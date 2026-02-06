# src/core/mission_controller.py

import src.data_access.report as report
import src.data_access.scraper as scraper
from colorama import Fore, Style
from rich.panel import Panel
from src.domain.dtos import PredictionConfigDTO
from src.data_access.config import (
    BEST_SETTINGS,
    TOTAL_BALLS,
    TICKET_SIZE,
    VERSION_TAG
)
from src.strategies.monte_carlo import MonteCarloStrategy
from src.strategies.universe_reduction import UniverseReductionStrategy
from src.strategies.genetic_selector import GeneticSelectorStrategy
from src.core.backtester import BacktestEngine
from src.core.optimizer import StrategyOptimizer
from src.core.coverage_tester import CoverageTester
from src.data_access.visualizer import run_forensic_visualization

class MissionController:
    def __init__(self, ui, history):
        self.ui = ui
        self.history = history

    def run_mission(self, option):
        option = option.upper()
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
            self._update_history()
        elif option == "9":
            self._validate_bets()
        elif option == "P":
            self._run_forensic_plot()

    def _view_history(self):
        self.ui.show_history(self.history)
        input(f"\n{Fore.YELLOW}>> Presiona ENTER para volver...{Style.RESET_ALL}")

    def _analyze_frequency(self):
        self.ui.show_frequency_analysis(self.history)
        input(f"\n{Fore.YELLOW}>> Presiona ENTER para volver...{Style.RESET_ALL}")

    def _run_monte_carlo(self):
        try:
            n_tkt = int(input(f"\n   ¿Cuántos tickets generar? (10): ") or 10)
        except: n_tkt = 10
        config = PredictionConfigDTO(TOTAL_BALLS, TICKET_SIZE, n_tkt)
        pred = MonteCarloStrategy().predict(self.history, config)
        self.ui.show_prediction_results(pred)
        input(f"\n{Fore.YELLOW}>> Presiona ENTER para volver...{Style.RESET_ALL}")

    def _run_optimizer(self):
        optimizer = StrategyOptimizer(self.history)
        optimizer.optimize()
        input(f"\n{Fore.YELLOW}>> Presiona ENTER para volver...{Style.RESET_ALL}")

    def _generate_universe(self):
        tester = CoverageTester(self.history)
        tester.test_reduction()
        input(f"\n{Fore.YELLOW}>> Presiona ENTER para volver...{Style.RESET_ALL}")

    def _run_backtest_lab(self):
        """Restaurada la funcionalidad de personalización de Backtest."""
        self.ui.clear_screen()
        print(f"\n{Fore.MAGENTA}🧪 LABORATORIO DE PRUEBAS (V15 OMEGA STRIDE){Style.RESET_ALL}")
        print("1. Sniper Mode (Solo Reducción)")
        print("2. Hybrid Mode (Resonancia + Genético)")
        print("3. Full Omega Stride (Producción Sim)")
        
        sub_op = input(f"\n{Fore.CYAN}Selecciona modo: {Style.RESET_ALL}")
        
        # RESTAURACIÓN DE INPUTS FUNCIONALES
        try:
            b_size = int(input(f"   ¿Cuántos sorteos hacia atrás probar? (50): ") or 50)
            n_tkt = int(input(f"   ¿Cuántos tickets por sorteo? (20): ") or 20)
        except:
            b_size, n_tkt = 50, 20

        engine = BacktestEngine()
        config = PredictionConfigDTO(TOTAL_BALLS, TICKET_SIZE, n_tkt, backtest_size=b_size)

        if sub_op == "1":
            engine.run(UniverseReductionStrategy(), self.history, config)
        elif sub_op in ["2", "3"]:
            engine.run(GeneticSelectorStrategy(), self.history, config, True, UniverseReductionStrategy())
            
        input(f"\n{Fore.YELLOW}>> Presiona ENTER...{Style.RESET_ALL}")

    def _run_production(self):
        """Producción V15: Con inputs funcionales y Ledger Lock."""
        ultimo_id = max(self.history.concursos)
        proximo_id = ultimo_id + 1
        
        if report.tiene_apuestas_pendientes(proximo_id):
            self.ui.console.print(Panel(
                f"[bold red]🚫 BLOQUEO DE SEGURIDAD[/]\n\nYa existen apuestas para el sorteo [bold cyan]#{proximo_id}[/].",
                border_style="red"
            ))
            input(f"\n{Fore.YELLOW}>> Presiona ENTER para volver...{Style.RESET_ALL}")
            return

        # RESTAURACIÓN DE INPUT FUNCIONAL
        try:
            n_prod = int(input(f"\n   ¿Cuántos tickets generar para el sorteo #{proximo_id}? (20): ") or 20)
        except:
            n_prod = 20

        config = PredictionConfigDTO(TOTAL_BALLS, TICKET_SIZE, n_prod, filter_overrides=BEST_SETTINGS)
        
        print(f"   {Fore.YELLOW}⏳ Paso 1: Filtrado Titanium...{Style.RESET_ALL}")
        univ_res = UniverseReductionStrategy().predict(self.history, config)
        config.raw_universe_ptr = univ_res.metadata.get("raw_ndarray")

        print(f"   {Fore.CYAN}🧬 Paso 2: Ejecutando Omega Stride...{Style.RESET_ALL}")
        pred = GeneticSelectorStrategy().predict(self.history, config)

        if pred.tickets:
            report.guardar_prediccion(pred.tickets, proximo_id)
            report.generar_ticket_limpio(pred.tickets, proximo_id)
            self.ui.show_prediction_results(pred)
            print(f"\n{Fore.GREEN}🍀 Tickets bloqueados y listos en archivo .txt{Style.RESET_ALL}")
        
        input(f"\n{Fore.YELLOW}>> Presiona ENTER para volver...{Style.RESET_ALL}")

    def _update_history(self):
        print(f"\n{Fore.YELLOW}🌐 Sincronizando datos...{Style.RESET_ALL}")
        if scraper.actualizar_csv():
            print(f"{Fore.GREEN}✅ Historial actualizado.{Style.RESET_ALL}")
        input(f"\n{Fore.YELLOW}>> Presiona ENTER para volver...{Style.RESET_ALL}")

    def _validate_bets(self):
        self.ui.clear_screen()
        print(f"\n{Fore.CYAN}💰 LIQUIDACIÓN DE CARTERA (ROI REAL){Style.RESET_ALL}")
        totales = report.liquidar_cartera(self.history)
        if totales:
            report.mostrar_resumen_roi(totales)
        input(f"\n{Fore.YELLOW}>> Presiona ENTER para volver...{Style.RESET_ALL}")

    def _run_forensic_plot(self):
        print(f"\n{Fore.CYAN}📊 Generando visualización forense...{Style.RESET_ALL}")
        run_forensic_visualization()
        input(f"\n{Fore.YELLOW}>> Reporte generado. Presiona ENTER...{Style.RESET_ALL}")