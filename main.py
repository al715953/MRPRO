import sys
import os
from colorama import Fore, Style

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from src.domain.dtos import PredictionConfigDTO
from src.data_access.loader import MelateLoader
from src.data_access.config import (
    CSV_FILE_PATH,
    TICKET_SIZE,
    TOTAL_BALLS,
    BEST_SETTINGS,
)
from src.data_access import scraper
import src.data_access.report as report
from src.interface.cli import ConsoleUI
from src.strategies.monte_carlo import MonteCarloStrategy
from src.strategies.universe_reduction import UniverseReductionStrategy
from src.strategies.genetic_selector import GeneticSelectorStrategy
from src.core.backtester import BacktestEngine
from src.core.optimizer import StrategyOptimizer
from src.core.coverage_tester import CoverageTester


def main():
    ui = ConsoleUI()
    ui.show_welcome()

    print(
        f"{Fore.CYAN}📂 Cargando histórico desde: {CSV_FILE_PATH}...{Style.RESET_ALL}"
    )
    loader = MelateLoader(CSV_FILE_PATH)
    history = loader.load_data()

    if not history.dates:
        print(
            f"{Fore.RED}❌ Error: No hay datos históricos. Intentando actualizar...{Style.RESET_ALL}"
        )
        try:
            scraper.actualizar_base_datos()
            history = loader.load_data()
        except Exception as e:
            print(f"Error actualizando: {e}")

    while True:
        ui.mostrar_logo()
        opcion = ui.get_main_menu_option()

        # 1. GENERACIÓN MAESTRA
        if opcion == "1":
            print(
                f"\n{Fore.YELLOW}🚀 INICIANDO SISTEMA DE PREDICCIÓN MAESTRA...{Style.RESET_ALL}"
            )
            print(f"\n{Fore.MAGENTA}🔹 FASE 1: Generando Universo...{Style.RESET_ALL}")

            current_settings = BEST_SETTINGS.copy()
            current_settings["verbose"] = False

            config_universe = PredictionConfigDTO(
                total_balls=TOTAL_BALLS,
                ticket_size=TICKET_SIZE,
                num_tickets=10,
                filter_overrides=current_settings,
            )
            strategy_universe = UniverseReductionStrategy()
            strategy_universe.predict(history, config_universe)
            print("✅ Universo generado.")

            print(f"\n{Fore.GREEN}🔹 FASE 2: Selección Genética...{Style.RESET_ALL}")
            config_final = PredictionConfigDTO(
                total_balls=TOTAL_BALLS,
                ticket_size=TICKET_SIZE,
                num_tickets=15,
            )
            strategy_selector = GeneticSelectorStrategy()
            prediction = strategy_selector.predict(history, config_final)

            if prediction.tickets:
                report.guardar_prediccion(prediction.tickets)
                ui.show_prediction_results(prediction)
                print(
                    f"\n{Fore.YELLOW}✨ ¡PROCESO COMPLETADO! Boletos en 'data/Mis_Apuestas.csv'{Style.RESET_ALL}"
                )
            else:
                print(f"\n{Fore.RED}❌ Error en selección.{Style.RESET_ALL}")
            input(f"\n{Fore.YELLOW}>> Presiona ENTER para volver...{Style.RESET_ALL}")

        # 2. BACKTESTING
        elif opcion == "2":
            print(
                f"\n{Fore.GREEN}🧪 INICIANDO BACKTESTING FINANCIERO...{Style.RESET_ALL}"
            )

            # --- INPUT DE USUARIO ---
            try:
                raw_input = input(
                    f"¿Cuántos sorteos pasados quieres simular? (Default=10): "
                )
                n_draws = int(raw_input) if raw_input.strip() else 10
            except ValueError:
                n_draws = 10

            print(f"⚙️  Simulando los últimos {n_draws} sorteos...")

            config = PredictionConfigDTO(
                total_balls=TOTAL_BALLS,
                ticket_size=TICKET_SIZE,
                num_tickets=15,
                backtest_size=n_draws,  # Usamos el input
                filter_overrides=BEST_SETTINGS,
            )

            engine = BacktestEngine()
            engine.run(MonteCarloStrategy(), history, config)

            input(f"\n{Fore.YELLOW}>> Presiona ENTER para volver...{Style.RESET_ALL}")

        # 3. VALIDAR RESULTADOS
        elif opcion == "3":
            print(f"\n{Fore.BLUE}🎫 VALIDANDO RESULTADOS...{Style.RESET_ALL}")
            try:
                scraper.validar_apuestas()
            except Exception as e:
                print(f"Error: {e}")
            input(f"\n{Fore.YELLOW}>> Presiona ENTER...{Style.RESET_ALL}")

        # 4. OPTIMIZADOR
        elif opcion == "4":
            print(f"\n{Fore.MAGENTA}🧠 AI TRAINER{Style.RESET_ALL}")
            print("1. Filtros Duros (Monte Carlo)\n2. Calidad Universo")
            sub_op = input("\n>> Opción: ")

            if sub_op == "1":
                base_config = PredictionConfigDTO(
                    total_balls=TOTAL_BALLS,
                    ticket_size=TICKET_SIZE,
                    num_tickets=15,
                    backtest_size=50,
                )
                optimizer = StrategyOptimizer(MonteCarloStrategy(), history)
                best = optimizer.run_grid_search(base_config)
                print(f"\n✅ MEJOR CONFIG: {best}")
            elif sub_op == "2":
                try:
                    from src.core.universe_optimizer import UniverseOptimizer

                    optimizer = UniverseOptimizer(history)
                    best_p = optimizer.optimize(
                        PredictionConfigDTO(
                            total_balls=TOTAL_BALLS,
                            ticket_size=TICKET_SIZE,
                            num_tickets=1,
                        )
                    )
                    print(f"Cambia QUALITY_PERCENTILE a: {best_p}")
                except:
                    print("Falta módulo universe_optimizer")
            input(f"\n{Fore.YELLOW}>> Presiona ENTER...{Style.RESET_ALL}")

        # 5. GENERAR UNIVERSO
        elif opcion == "5":
            print(f"\n{Fore.MAGENTA}🌌 GENERANDO UNIVERSO (MANUAL)...{Style.RESET_ALL}")
            config = PredictionConfigDTO(
                total_balls=TOTAL_BALLS,
                ticket_size=TICKET_SIZE,
                num_tickets=10,
                filter_overrides=BEST_SETTINGS,
            )
            UniverseReductionStrategy().predict(history, config)
            print(
                f"\n{Fore.GREEN}✅ Generado 'data/universo_reducido.csv'{Style.RESET_ALL}"
            )
            input(f"\n{Fore.YELLOW}>> Presiona ENTER...{Style.RESET_ALL}")

        # 6. COVERAGE TESTER
        elif opcion == "6":
            print(f"\n{Fore.CYAN}📡 COVERAGE TESTER...{Style.RESET_ALL}")
            try:
                n_test = int(input("¿Sorteos? (5): ") or 5)
            except:
                n_test = 5
            settings = BEST_SETTINGS.copy()
            settings["verbose"] = False
            config = PredictionConfigDTO(
                total_balls=TOTAL_BALLS,
                ticket_size=TICKET_SIZE,
                num_tickets=10,
                backtest_size=n_test,
                filter_overrides=settings,
            )
            CoverageTester().run(UniverseReductionStrategy(), history, config)
            input(f"\n{Fore.YELLOW}>> Presiona ENTER...{Style.RESET_ALL}")

        # 7. SELECTOR
        elif opcion == "7":
            print(f"\n{Fore.GREEN}🎯 SELECTOR FINAL (MANUAL)...{Style.RESET_ALL}")
            config = PredictionConfigDTO(
                total_balls=TOTAL_BALLS, ticket_size=TICKET_SIZE, num_tickets=15
            )
            pred = GeneticSelectorStrategy().predict(history, config)
            if pred.tickets:
                report.guardar_prediccion(pred.tickets)
                ui.show_prediction_results(pred)
            else:
                print(f"{Fore.RED}❌ Error.{Style.RESET_ALL}")
            input(f"\n{Fore.YELLOW}>> Presiona ENTER...{Style.RESET_ALL}")

        elif opcion == "0":
            print("👋 Bye.")
            sys.exit()
        else:
            print("⚠️ Opción inválida.")


if __name__ == "__main__":
    main()
