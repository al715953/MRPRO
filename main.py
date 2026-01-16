import sys
import os
from datetime import datetime
from colorama import Fore, Style

# Configuración de rutas para importaciones absolutas
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# --- DOMAIN & DATA ACCESS ---
from src.data_access.visualizer import run_forensic_visualization
from src.domain.dtos import PredictionConfigDTO
from src.data_access.loader import MelateLoader
from src.data_access.config import (
    CSV_FILE_PATH,
    TICKET_SIZE,
    TOTAL_BALLS,
    BEST_SETTINGS,
)
import src.data_access.scraper as scraper
import src.data_access.report as report

# --- INTERFACE ---
from src.interface.cli import ConsoleUI

# --- STRATEGIES ---
from src.strategies.monte_carlo import MonteCarloStrategy
from src.strategies.universe_reduction import UniverseReductionStrategy
from src.strategies.genetic_selector import GeneticSelectorStrategy

# Importación segura para la estrategia Heurística
try:
    from src.strategies.heuristic_selector import HeuristicSelectorStrategy
except ImportError:
    HeuristicSelectorStrategy = None

# --- CORE ENGINES ---
from src.core.backtester import BacktestEngine
from src.core.optimizer import StrategyOptimizer
from src.core.coverage_tester import CoverageTester


def check_and_update_database(loader: MelateLoader, verbose: bool = True):
    """Verifica si la base de datos local requiere actualización."""
    if verbose:
        print(
            f"{Fore.CYAN}🔍 Verificando integridad de la base de datos...{Style.RESET_ALL}"
        )

    needs_update = False
    try:
        history = loader.load_data()
        if not history.dates:
            needs_update = True
        else:
            last_item = history.dates[-1]
            last_date = (
                datetime.strptime(last_item, "%d/%m/%Y").date()
                if isinstance(last_item, str)
                else last_item
            )
            if (datetime.now().date() - last_date).days > 4:
                needs_update = True
    except:
        needs_update = True

    if needs_update:
        print(
            f"\n{Fore.CYAN}📥 Iniciando actualización desde Lotería Nacional...{Style.RESET_ALL}"
        )
        scraper.descargar_datos(CSV_FILE_PATH)


def main():
    ui = ConsoleUI()
    ui.show_welcome()

    loader = MelateLoader(CSV_FILE_PATH)
    check_and_update_database(loader)
    history = loader.load_data()

    if not history.dates:
        print(f"{Fore.RED}❌ ERROR CRÍTICO: No se cargaron datos.{Style.RESET_ALL}")
        return

    print(
        f"{Fore.GREEN}✅ Sistema listo. {len(history.winning_numbers)} sorteos cargados.{Style.RESET_ALL}"
    )

    while True:
        opcion = ui.show_main_menu()

        if opcion == "0":
            print(f"{Fore.CYAN}👋 ¡Hasta la próxima, Arquitecto!{Style.RESET_ALL}")
            break

        # 1. VER HISTORIAL
        elif opcion == "1":
            ui.show_history(history)
            input(f"\n{Fore.YELLOW}>> Presiona ENTER...{Style.RESET_ALL}")

        # 2. ANÁLISIS FRECUENCIA
        elif opcion == "2":
            ui.analyze_frequency(history, TOTAL_BALLS)
            input(f"\n{Fore.YELLOW}>> Presiona ENTER...{Style.RESET_ALL}")

        # 3. MONTE CARLO (SIMULACIÓN)
        elif opcion == "3":
            print(f"\n{Fore.CYAN}🎲 MÓDULO MONTE CARLO{Style.RESET_ALL}")
            config = PredictionConfigDTO(TOTAL_BALLS, TICKET_SIZE, num_tickets=10)
            config.filter_overrides = BEST_SETTINGS
            pred = MonteCarloStrategy().predict(history, config)
            ui.show_prediction_results(pred)
            input(f"\n{Fore.YELLOW}>> Presiona ENTER...{Style.RESET_ALL}")

        # 4. OPTIMIZADOR (IA PARÁMETROS)
        elif opcion == "4":
            sub_opt, n_draws = ui.show_optimizer_menu()
            opt = StrategyOptimizer()
            try:
                base_dummy = BEST_SETTINGS.copy()
                base_dummy["verbose"] = False

                if sub_opt == "1":
                    best_cfg = opt.optimize_filters(history, n_draws)
                elif sub_opt == "2":
                    best_cfg = opt.optimize_heuristics(history, base_dummy, n_draws)
                elif sub_opt == "3":
                    best_cfg = opt.optimize_quotas(history, base_dummy, n_draws)
                elif sub_opt == "4":
                    best_cfg = opt.optimize_full_stack(history, n_draws)

                print(
                    f"\n{Fore.GREEN}🏆 CONFIGURACIÓN OPTIMIZADA ({n_draws} Sorteos):{Style.RESET_ALL}"
                )
                for k, v in best_cfg.items():
                    print(f"   • {k:<15}: {Fore.CYAN}{v}{Style.RESET_ALL}")
                print(
                    f"\n{Fore.GREEN}💾 Actualiza 'BEST_SETTINGS' en config.py con estos valores.{Style.RESET_ALL}"
                )
            except Exception as e:
                print(f"{Fore.RED}⚠ Error en optimización: {e}{Style.RESET_ALL}")
            input(f"\n{Fore.YELLOW}>> Presiona ENTER...{Style.RESET_ALL}")

        # 5. GENERAR UNIVERSO (REDUCCIÓN)
        elif opcion == "5":
            print(
                f"\n{Fore.CYAN}🌌 GENERANDO UNIVERSO (P88 DENSITY)...{Style.RESET_ALL}"
            )
            config = PredictionConfigDTO(TOTAL_BALLS, TICKET_SIZE, num_tickets=0)
            config.filter_overrides = BEST_SETTINGS.copy()
            config.filter_overrides["verbose"] = True  # Ver progreso de GPU
            UniverseReductionStrategy().predict(history, config)
            print(
                f"{Fore.GREEN}✅ Universo persistido en data/universo_reducido.csv{Style.RESET_ALL}"
            )
            input(f"\n{Fore.YELLOW}>> Presiona ENTER...{Style.RESET_ALL}")

        # 6. LABORATORIO (BACKTEST DINÁMICO)
        elif opcion == "6":
            print(
                f"\n{Fore.CYAN}📡 LABORATORIO V32.1 (STOCHASTIC SNIPER P8){Style.RESET_ALL}"
            )
            print("1. 🛡️  Cobertura Fase 1 (Universo)")
            print("2. 🥊 Duelo (IA Sniper vs Clásica)")
            print("3. 🧠 Solo AI (Centauro V7 - Stochastic)")

            sub_op = input("   👉 Selecciona modo (3): ") or "3"
            try:
                n_test = int(input(f"   ¿Sorteos a simular? (40): ") or 40)
                n_tickets = int(input(f"   ¿Tickets a comprar? (15): ") or 15)
            except:
                n_test, n_tickets = 40, 15

            config = PredictionConfigDTO(
                total_balls=TOTAL_BALLS,
                ticket_size=TICKET_SIZE,
                num_tickets=n_tickets,
                backtest_size=n_test,
                filter_overrides=BEST_SETTINGS,
            )
            engine = BacktestEngine()

            if sub_op == "1":
                CoverageTester().run(UniverseReductionStrategy(), history, config)

            elif sub_op == "2" and HeuristicSelectorStrategy:
                print(f"\n{Fore.YELLOW}🥊 INICIANDO DUELO...{Style.RESET_ALL}")
                # Corrección: Usamos nombres de parámetros para evitar errores de orden
                engine.run(
                    strategy=HeuristicSelectorStrategy(),
                    history=history,
                    config=config,
                    verbose=True,
                    pre_process_strategy=UniverseReductionStrategy(),
                )
                engine.run(
                    strategy=GeneticSelectorStrategy(),
                    history=history,
                    config=config,
                    verbose=True,
                    pre_process_strategy=UniverseReductionStrategy(),
                )

            elif sub_op == "3":
                print(f"\n{Fore.MAGENTA}🧠 EJECUTANDO MISIÓN IA...{Style.RESET_ALL}")
                # Corrección: Argumentos nombrados para seguridad total
                try:
                    engine.run(
                        strategy=GeneticSelectorStrategy(),
                        history=history,
                        config=config,
                        verbose=True,
                        pre_process_strategy=UniverseReductionStrategy(),
                    )
                except Exception as e:
                    print(
                        f"{Fore.RED}❌ ERROR CRÍTICO EN BACKTEST: {e}{Style.RESET_ALL}"
                    )
                    import traceback

                    traceback.print_exc()

            input(f"\n{Fore.YELLOW}>> Presiona ENTER...{Style.RESET_ALL}")

        # 7. SELECTOR FINAL (PRODUCCIÓN)
        elif opcion == "7":
            try:
                n_prod = int(input(f"\n   ¿Cuántos tickets comprar hoy? (15): ") or 15)
            except:
                n_prod = 15

            config = PredictionConfigDTO(
                TOTAL_BALLS,
                TICKET_SIZE,
                num_tickets=n_prod,
                filter_overrides=BEST_SETTINGS,
            )
            pred = GeneticSelectorStrategy().predict(history, config)

            if pred.tickets:
                report.guardar_prediccion(pred.tickets)
                ui.show_prediction_results(pred)
                print(
                    f"\n{Fore.GREEN}🍀 Predicción guardada. ¡Buena suerte!{Style.RESET_ALL}"
                )
            else:
                print(
                    f"{Fore.RED}❌ Error: Generación fallida. Revisa el Paso 5.{Style.RESET_ALL}"
                )
            input(f"\n{Fore.YELLOW}>> Presiona ENTER...{Style.RESET_ALL}")

        elif opcion == "8":
            run_forensic_visualization()
            input("\nPresione ENTER para volver al menú...")

        else:
            print(f"{Fore.RED}Opción inválida.{Style.RESET_ALL}")


if __name__ == "__main__":
    main()
