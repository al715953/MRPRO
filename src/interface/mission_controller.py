# src/core/mission_controller.py

import src.data_access.report as report
import src.data_access.scraper as scraper
import subprocess
from colorama import Fore, Style
from rich.panel import Panel
from src.domain.dtos import PredictionConfigDTO
from src.data_access.config import BEST_SETTINGS, TOTAL_BALLS, TICKET_SIZE, VERSION_TAG
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
            self._run_optimizer()
        elif option == "4":
            self._retrain_model()
        elif option == "5":
            self._update_history()
        elif option == "6":
            self._run_backtest_lab()
        elif option == "7":
            self._run_production()
        elif option == "8":
            self._validate_bets()
        elif option == "P":
            self._run_forensic_plot()

    def _view_history(self):
        self.ui.show_history(self.history)
        input(f"\n{Fore.YELLOW}>> Presiona ENTER para volver...{Style.RESET_ALL}")

    def _analyze_frequency(self):
        self.ui.show_frequency_analysis(self.history)
        input(f"\n{Fore.YELLOW}>> Presiona ENTER para volver...{Style.RESET_ALL}")

    def _run_optimizer(self):
        self.ui.clear_screen()
        print(f"\n{Fore.MAGENTA}🔬 MÓDULO DE OPTIMIZACIÓN MRPRO V15{Style.RESET_ALL}")
        print("1. Calibración Forense (Filtros de Reducción)")
        print("2. Optimizar Pesos de Votantes (Sniper E1)")

        op = input(f"\n{Fore.CYAN}Selecciona opción: {Style.RESET_ALL}")

        try:
            n_draws = int(input(f"   ¿Cuántos sorteos analizar? (200): ") or 200)
        except:
            n_draws = 200

        opt = StrategyOptimizer()

        if op == "1":
            best_cfg = opt.optimize_filters(self.history, n_draws)
        elif op == "2":
            best_cfg = opt.optimize_voter_weights(self.history, n_draws)

            print(f"\n{Fore.GREEN}🏆 PESOS SUGERIDOS:{Style.RESET_ALL}")
            for k, v in best_cfg.items():
                print(f"   • {k:<10}: {Fore.CYAN}{v}{Style.RESET_ALL}")
            print(
                f"\n{Fore.YELLOW}ℹ️  Actualiza estos valores en src/strategies/universe/filters.py{Style.RESET_ALL}"
            )

        input(f"\n{Fore.YELLOW}>> Presiona ENTER para volver...{Style.RESET_ALL}")

    def _generate_universe(self):
        tester = CoverageTester()
        config = PredictionConfigDTO(TOTAL_BALLS, TICKET_SIZE, num_tickets=20)
        tester.run(UniverseReductionStrategy(), self.history, config)
        input(f"\n{Fore.YELLOW}>> Presiona ENTER para volver...{Style.RESET_ALL}")

    def _run_backtest_lab(self):
        """Restaurada la funcionalidad de personalización de Backtest."""
        self.ui.clear_screen()
        print(
            f"\n{Fore.MAGENTA}🧪 LABORATORIO DE PRUEBAS (V15 OMEGA STRIDE){Style.RESET_ALL}"
        )
        print("1. Sniper Mode (Solo Reducción)")
        print("2. Full Omega Stride (Producción Sim)")

        sub_op = input(f"\n{Fore.CYAN}Selecciona modo: {Style.RESET_ALL}")

        # RESTAURACIÓN DE INPUTS FUNCIONALES
        try:
            b_size = int(
                input(f"   ¿Cuántos sorteos hacia atrás probar? (108): ") or 108
            )
            n_tkt = int(input(f"   ¿Cuántos tickets por sorteo? (20): ") or 20)
        except:
            b_size, n_tkt = 108, 20

        engine = BacktestEngine()
        config = PredictionConfigDTO(
            TOTAL_BALLS, TICKET_SIZE, n_tkt, backtest_size=b_size
        )

        if sub_op == "1":
            engine.run(UniverseReductionStrategy(), self.history, config)
        elif sub_op == "2":
            engine.run(
                GeneticSelectorStrategy(),
                self.history,
                config,
                True,
                UniverseReductionStrategy(),
            )

        input(f"\n{Fore.YELLOW}>> Presiona ENTER...{Style.RESET_ALL}")

    def _run_production(self):
        """Producción V15: Con inputs funcionales y Ledger Lock."""
        ultimo_id = max(self.history.concursos)
        proximo_id = ultimo_id + 1

        if report.tiene_apuestas_pendientes(proximo_id):
            self.ui.console.print(
                Panel(
                    f"[bold red]🚫 BLOQUEO DE SEGURIDAD[/]\n\nYa existen apuestas para el sorteo [bold cyan]#{proximo_id}[/].",
                    border_style="red",
                )
            )
            input(f"\n{Fore.YELLOW}>> Presiona ENTER para volver...{Style.RESET_ALL}")
            return

        # RESTAURACIÓN DE INPUT FUNCIONAL
        try:
            n_prod = int(
                input(
                    f"\n   ¿Cuántos tickets generar para el sorteo #{proximo_id}? (20): "
                )
                or 20
            )
        except:
            n_prod = 20

        config = PredictionConfigDTO(
            TOTAL_BALLS, TICKET_SIZE, n_prod, filter_overrides=BEST_SETTINGS
        )

        print(f"   {Fore.YELLOW}⏳ Paso 1: Filtrado Titanium...{Style.RESET_ALL}")
        univ_res = UniverseReductionStrategy().predict(self.history, config)
        config.raw_universe_ptr = univ_res.metadata.get("raw_ndarray")

        print(f"   {Fore.CYAN}🧬 Paso 2: Ejecutando Omega Stride...{Style.RESET_ALL}")
        pred = GeneticSelectorStrategy().predict(self.history, config)

        if pred.tickets:
            report.guardar_prediccion(pred.tickets, proximo_id)
            report.generar_ticket_limpio(pred.tickets, proximo_id)
            self.ui.show_prediction_results(pred)
            print(
                f"\n{Fore.GREEN}🍀 Tickets bloqueados y listos en archivo .txt{Style.RESET_ALL}"
            )

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

    def _retrain_model(self):
        """Módulo de Calibración de Neuronas V15."""
        self.ui.clear_screen()
        # Corregido: Fore para color, Style para efectos (DIM)
        print(
            f"\n{Fore.MAGENTA}☢️  PROTOCOLO DE RECALIBRACIÓN CEREBRAL V8{Style.RESET_ALL}"
        )
        print(
            f"{Style.DIM}Iniciando XGBoost Engine sobre hardware detectado...{Style.RESET_ALL}\n"
        )

        try:
            # Ejecutamos el script de entrenamiento como proceso independiente
            import subprocess

            subprocess.run(["python", "src/core/train_static_model.py"], check=True)
            print(
                f"\n{Fore.GREEN}✅ MODELO ACTUALIZADO: Los pesos han sido sincronizados.{Style.RESET_ALL}"
            )
        except Exception as e:
            print(
                f"\n{Fore.RED}❌ ERROR CRÍTICO EN ENTRENAMIENTO: {e}{Style.RESET_ALL}"
            )

        input(f"\n{Fore.YELLOW}>> Presiona ENTER para volver...{Style.RESET_ALL}")
