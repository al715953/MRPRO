# src/core/mission_controller.py

import src.data_access.report as report
import src.data_access.scraper as scraper
import subprocess
from colorama import Fore, Style
from rich.panel import Panel
from rich.table import Table
from src.domain.dtos import PredictionConfigDTO
from src.data_access.config import (
    BEST_SETTINGS,
    BEST_SETTINGS_TRIS,
    TOTAL_BALLS,
    TICKET_SIZE,
    VERSION_TAG,
)
from src.strategies.universe_reduction import UniverseReductionStrategy
from src.strategies.genetic_selector import GeneticSelectorStrategy
from src.strategies.tris.tris_forecast import TrisForecastV1A
from src.core.backtester import BacktestEngine
from src.core.optimizer import StrategyOptimizer
from src.core.coverage_tester import CoverageTester
from src.core.rules import TrisMultiplicadorRules


class MissionController:
    def __init__(self, ui, history, profile=None):
        self.ui = ui
        self.history = history
        self.profile = profile

    def run_mission(self, option):
        if self._is_tris():
            self._run_mission_tris(option)
            return
        self._run_mission_melate(option)

    def _run_mission_melate(self, option):
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
        else:
            self.ui.console.print("[red]Opcion no valida.[/]")
            self._pause()

    def _run_mission_tris(self, option):
        option = option.upper()
        if option == "1":
            self._view_history()
        elif option == "2":
            self._analyze_frequency()
        elif option == "3":
            self._notify_beta_feature(
                "Optimizador Tris",
                "La logica de optimizacion para secuencias (00000-99999) aun no esta integrada.",
            )
        elif option == "4":
            self._notify_beta_feature(
                "Reentrenamiento Tris",
                "El pipeline de entrenamiento para Tris Multiplicador sigue en desarrollo.",
            )
        elif option == "5":
            self._update_history()
        elif option == "6":
            self._run_tris_backtest()
        elif option == "7":
            self._run_tris_production()
        elif option == "8":
            self._notify_beta_feature(
                "Liquidacion Tris",
                "La liquidacion de cartera para Tris requiere ledger separado y tabla oficial de premios.",
            )
        elif option == "P":
            self._notify_beta_feature(
                "Plot Forense Tris",
                "El reporte forense para Tris aun no tiene esquema final.",
            )
        else:
            self.ui.console.print("[red]Opcion no valida.[/]")
            self._pause()

    def _view_history(self):
        self.ui.show_history(self.history, self.profile)
        self._pause()

    def _analyze_frequency(self):
        self.ui.show_frequency_analysis(self.history, self.profile)
        self._pause()

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
                f"\n{Fore.YELLOW}ℹ️  Actualiza estos valores en src/data_access/config.py{Style.RESET_ALL}"
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
            n_tkt = int(input(f"   ¿Cuántos tickets por sorteo? (24): ") or 24)
        except:
            b_size, n_tkt = 108, 24

        engine = BacktestEngine()
        config = PredictionConfigDTO(
            TOTAL_BALLS, TICKET_SIZE, n_tkt, backtest_size=b_size
        )

        if sub_op == "1":
            engine.run(UniverseReductionStrategy(), self.history, config, verbose=True)
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
                    f"\n   ¿Cuántos tickets generar para el sorteo #{proximo_id}? (24): "
                )
                or 24
            )
        except:
            n_prod = 24

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
        game_code = self.profile.code if self.profile else "melate_retro"
        if scraper.actualizar_csv(game_code):
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
        try:
            from src.data_access.visualizer import run_forensic_visualization

            run_forensic_visualization()
        except Exception as e:
            print(
                f"{Fore.RED}❌ No se pudo cargar el módulo de visualización: {e}{Style.RESET_ALL}"
            )
        input(f"\n{Fore.YELLOW}>> Reporte generado. Presiona ENTER...{Style.RESET_ALL}")

    def _run_tris_backtest(self):
        self.ui.clear_screen()
        self.ui.console.print(
            "\n[bold magenta]🧪 BACKTEST TRIS V1-A (BAYES + MARKOV)[/]"
        )

        settings = BEST_SETTINGS_TRIS.copy()
        config = PredictionConfigDTO(
            total_balls=self.profile.total_balls,
            ticket_size=self.profile.ticket_size,
            num_tickets=int(settings["num_tickets"]),
            backtest_size=int(settings["backtest_size"]),
            filter_overrides=settings.copy(),
        )

        engine = BacktestEngine(rules=TrisMultiplicadorRules())
        engine.run(
            strategy=TrisForecastV1A(),
            history=self.history,
            config=config,
            verbose=True,
            pre_process_strategy=None,
        )
        self._pause()

    def _run_tris_production(self):
        self.ui.clear_screen()
        self.ui.console.print(
            "\n[bold green]🎯 PRODUCCION TRIS V1-A (ONE-SHOT)[/]"
        )

        settings = BEST_SETTINGS_TRIS.copy()
        config = PredictionConfigDTO(
            total_balls=self.profile.total_balls,
            ticket_size=self.profile.ticket_size,
            num_tickets=int(settings["num_tickets"]),
            backtest_size=int(settings["backtest_size"]),
            filter_overrides=settings.copy(),
        )

        predictor = TrisForecastV1A()
        pred = predictor.predict(self.history, config)

        preview_n = min(10, len(pred.tickets))
        self.ui.console.print(
            f"[cyan]Estrategia:[/] {pred.strategy_name} | [cyan]Tickets generados:[/] {len(pred.tickets)}"
        )
        self.ui.console.print(f"[cyan]Top {preview_n} tickets:[/]")
        for idx, t in enumerate(pred.tickets[:preview_n], start=1):
            t_str = "".join(str(int(d)) for d in t[:5])
            self.ui.console.print(f"  [bold]{idx:02d}.[/] {t_str}")

        pos_probs = pred.metadata.get("pos_probs", [])
        if pos_probs and len(pos_probs) == 5:
            table = Table(title="Top-3 digitos por posicion", show_header=True)
            table.add_column("Pos", justify="center")
            table.add_column("Top-1", justify="center")
            table.add_column("Top-2", justify="center")
            table.add_column("Top-3", justify="center")

            for pos in range(5):
                row = pos_probs[pos]
                ranked = sorted(
                    enumerate(row),
                    key=lambda x: x[1],
                    reverse=True,
                )[:3]
                top_cells = [f"{d} ({p:.3f})" for d, p in ranked]
                table.add_row(str(pos + 1), top_cells[0], top_cells[1], top_cells[2])

            self.ui.console.print(table)

        p_multiplier = float(pred.metadata.get("p_multiplier", 0.0))
        entropy_mean = float(pred.metadata.get("entropy_mean", 0.0))
        self.ui.console.print(
            f"[yellow]p_multiplier:[/] {p_multiplier:.4f} | [yellow]entropy_mean:[/] {entropy_mean:.4f}"
        )
        self._pause()

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

    def _notify_beta_feature(self, title: str, details: str):
        self.ui.console.print(
            Panel(
                f"[bold yellow]MODULO EN BETA: {title}[/]\n\n{details}",
                border_style="yellow",
            )
        )
        self._pause()

    def _pause(self):
        self.ui.console.input(
            f"\n[yellow]>> Presiona ENTER para volver...[/]"
        )

    def _is_tris(self) -> bool:
        return bool(self.profile and self.profile.code == "tris_multiplicador")
