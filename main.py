import sys
import os
import time
from datetime import datetime, date
from colorama import Fore, Style

# Añadimos el directorio raíz al path para importaciones absolutas
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# --- DOMAIN & DATA ACCESS ---
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

# Importación segura para la estrategia Heurística (si falla, no rompe todo el programa hasta usarlo)
try:
    from src.strategies.heuristic_selector import HeuristicSelectorStrategy
except ImportError:
    HeuristicSelectorStrategy = None

# --- CORE ENGINES ---
from src.core.backtester import BacktestEngine
from src.core.optimizer import StrategyOptimizer
from src.core.coverage_tester import CoverageTester


def check_and_update_database(loader: MelateLoader, verbose: bool = True):
    """
    Verifica si la base de datos local está actualizada.
    Si el último sorteo tiene > 4 días, ejecuta el scraper.
    """
    if verbose:
        print(
            f"{Fore.CYAN}🔍 Verificando integridad de la base de datos...{Style.RESET_ALL}"
        )

    needs_update = False
    try:
        # Intentamos cargar la data actual
        history = loader.load_data()

        if not history.dates:
            print(
                f"{Fore.YELLOW}⚠ Archivo local vacío o no encontrado.{Style.RESET_ALL}"
            )
            needs_update = True
        else:
            last_item = history.dates[-1]
            if isinstance(last_item, str):
                last_date = datetime.strptime(last_item, "%d/%m/%Y").date()
            else:
                last_date = last_item

            days_diff = (datetime.now().date() - last_date).days

            if days_diff > 4:
                print(
                    f"{Fore.YELLOW}⚠ Base de datos desactualizada (Último: {last_date}).{Style.RESET_ALL}"
                )
                needs_update = True
            else:
                if verbose:
                    print(
                        f"{Fore.GREEN}✅ Base de datos al día ({last_date}).{Style.RESET_ALL}"
                    )

    except Exception as e:
        print(f"{Fore.RED}⚠ Error verificando fecha: {e}{Style.RESET_ALL}")
        needs_update = True

    if needs_update:
        print(
            f"\n{Fore.CYAN}📥 Iniciando actualización automática desde Lotería Nacional...{Style.RESET_ALL}"
        )
        try:
            exito, mensaje = scraper.descargar_datos(CSV_FILE_PATH)
            if exito:
                print(f"{Fore.GREEN}{mensaje}{Style.RESET_ALL}\n")
            else:
                print(f"{Fore.YELLOW}{mensaje}{Style.RESET_ALL}\n")
        except AttributeError:
            print(
                f"{Fore.RED}❌ Error: Función del scraper no encontrada.{Style.RESET_ALL}"
            )
        except Exception as e:
            print(
                f"{Fore.RED}❌ Falló la actualización automática: {e}{Style.RESET_ALL}"
            )
            print(
                f"{Fore.YELLOW}   Continuando con datos locales...{Style.RESET_ALL}\n"
            )


def main():
    ui = ConsoleUI()
    ui.show_welcome()

    # --- 1. VALIDACIÓN E INTEGRIDAD DE DATOS ---
    loader = MelateLoader(CSV_FILE_PATH)
    check_and_update_database(loader)

    # --- 2. CARGA DE DATOS ---
    print(f"{Fore.CYAN}📂 Cargando histórico oficial...{Style.RESET_ALL}")
    history = loader.load_data()

    if not history.dates:
        print(
            f"{Fore.RED}❌ ERROR CRÍTICO: No se pudieron cargar datos para operar.{Style.RESET_ALL}"
        )
        return

    print(
        f"{Fore.GREEN}✅ Sistema listo. {len(history.winning_numbers)} sorteos cargados.{Style.RESET_ALL}"
    )

    # --- 3. BUCLE PRINCIPAL ---
    while True:
        opcion = ui.show_main_menu()

        if opcion == "0":
            print(f"{Fore.CYAN}👋 ¡Hasta la próxima, Arquitecto!{Style.RESET_ALL}")
            break

        # 1. VER HISTORIAL
        elif opcion == "1":
            ui.show_history(history)
            input(f"\n{Fore.YELLOW}>> Presiona ENTER...{Style.RESET_ALL}")

        # 2. ANALISIS FRECUENCIA
        elif opcion == "2":
            ui.analyze_frequency(history, TOTAL_BALLS)
            input(f"\n{Fore.YELLOW}>> Presiona ENTER...{Style.RESET_ALL}")

        # 3. MONTE CARLO (SIMULACIÓN)
        elif opcion == "3":
            print(f"\n{Fore.CYAN}🎲 MÓDULO MONTE CARLO{Style.RESET_ALL}")
            config = PredictionConfigDTO(
                total_balls=TOTAL_BALLS, ticket_size=TICKET_SIZE, num_tickets=10
            )
            config.filter_overrides = BEST_SETTINGS
            pred = MonteCarloStrategy().predict(history, config)
            ui.show_prediction_results(pred)
            input(f"\n{Fore.YELLOW}>> Presiona ENTER...{Style.RESET_ALL}")

        # 4. OPTIMIZADOR (IA PARAMETROS)
        elif opcion == "4":
            print(
                f"\n{Fore.MAGENTA}🧠 OPTIMIZADOR DE PARÁMETROS (GRID SEARCH){Style.RESET_ALL}"
            )
            opt = StrategyOptimizer()
            best_cfg = opt.optimize(history)
            print(
                f"\n{Fore.GREEN}💾 Guarda estos valores en data_access/config.py!{Style.RESET_ALL}"
            )
            input(f"\n{Fore.YELLOW}>> Presiona ENTER...{Style.RESET_ALL}")

        # 5. GENERAR UNIVERSO (REDUCCIÓN)
        elif opcion == "5":
            print(f"\n{Fore.CYAN}🌌 GENERANDO UNIVERSO (MANUAL)...{Style.RESET_ALL}")
            config = PredictionConfigDTO(
                total_balls=TOTAL_BALLS,
                ticket_size=TICKET_SIZE,
                filter_overrides=BEST_SETTINGS,
            )
            UniverseReductionStrategy().predict(history, config)
            input(f"\n{Fore.YELLOW}>> Presiona ENTER...{Style.RESET_ALL}")

        # 6. BACKTESTING (MENU AVANZADO)
        elif opcion == "6":
            print(f"\n{Fore.CYAN}⚙️  CONFIGURACIÓN DE BACKTESTING{Style.RESET_ALL}")
            print("1. 🏛️  Estrategia Clásica (Solo Heurística)")
            print("2. 🤖  Estrategia Centauro (Solo IA)")
            print("3. ⚖️  Comparativa (IA vs Clásica)")

            sub_opcion = input(
                f"\n{Fore.YELLOW}>> Elige una opción (1-3): {Style.RESET_ALL}"
            )

            # Configuración común
            test_size = 24  # O el número que prefieras por defecto
            try:
                ts_input = input(f"Cuantos sorteos simular? (Default {test_size}): ")
                if ts_input.strip():
                    test_size = int(ts_input)
            except ValueError:
                pass

            config = PredictionConfigDTO(
                total_balls=TOTAL_BALLS,
                ticket_size=TICKET_SIZE,
                num_tickets=15,
                backtest_size=test_size,
                filter_overrides=BEST_SETTINGS,
            )

            # Instancias
            backtester = BacktestEngine()
            universe_reduction = (
                UniverseReductionStrategy()
            )  # Fase 1 (necesaria para Radar)

            # --- OPCIÓN 1: SOLO CLÁSICA ---
            if sub_opcion == "1":
                if HeuristicSelectorStrategy:
                    heuristic = HeuristicSelectorStrategy()
                    backtester.run(
                        strategy=heuristic,
                        history=history,
                        config=config,
                        verbose=True,
                        pre_process_strategy=universe_reduction,
                    )
                else:
                    print("Estrategia Heurística no disponible.")

            # --- OPCIÓN 2: SOLO IA (CENTAURO) ---
            elif sub_opcion == "2":

                print("MODO FORENSE ACTIVADO: Verás la autopsia de cada sorteo.")
                genetic = GeneticSelectorStrategy()
                backtester.run(
                    strategy=genetic,
                    history=history,
                    config=config,
                    verbose=True,  # Verás el detalle sorteo a sorteo
                    pre_process_strategy=universe_reduction,  # Activa el Radar de Cobertura
                    debug_deep=True,
                )

            # --- OPCIÓN 3: COMPARATIVA (LO QUE TENÍAS ANTES) ---
            elif sub_opcion == "3":
                print(
                    f"\n{Fore.CYAN}⚔️  INICIANDO DUELO DE ESTRATEGIAS...{Style.RESET_ALL}"
                )

                # 1. Correr Clásico (Silencioso para no ensuciar consola)
                res_classic = None
                if HeuristicSelectorStrategy:
                    heuristic = HeuristicSelectorStrategy()
                    res_classic = backtester.run(
                        heuristic,
                        history,
                        config,
                        verbose=False,
                        pre_process_strategy=universe_reduction,
                    )

                # 2. Correr IA (Verbose para ver progreso)
                genetic = GeneticSelectorStrategy()
                res_ai = backtester.run(
                    genetic,
                    history,
                    config,
                    verbose=True,
                    pre_process_strategy=universe_reduction,
                )

                # 3. Tabla Comparativa Final
                print(f"\n{Fore.MAGENTA}📊 REPORTE DE BATALLA (Final){Style.RESET_ALL}")
                print(f"{'Métrica':<20} | {'Clásica':<15} | {'Centauro (IA)':<15}")
                print("-" * 56)

                c_inv = res_classic.earnings if res_classic else 0
                print(
                    f"{'Ganancias':<20} | ${c_inv:,.2f}       | ${res_ai.earnings:,.2f}"
                )

                c_bal = res_classic.net_balance if res_classic else 0
                print(
                    f"{'Balance Neto':<20} | ${c_bal:,.2f}       | ${res_ai.net_balance:,.2f}"
                )

                # Comparar aciertos
                if res_classic:
                    c3 = res_classic.hit_distribution.get(3, 0)
                    c4p = sum(
                        [res_classic.hit_distribution.get(k, 0) for k in [4, 5, 6]]
                    )
                else:
                    c3, c4p = 0, 0

                a3 = res_ai.hit_distribution.get(3, 0)
                a4p = sum([res_ai.hit_distribution.get(k, 0) for k in [4, 5, 6]])

                print(f"{'Aciertos (3)':<20} | {c3:<15} | {a3:<15}")
                print(f"{'Aciertos (4+)':<20} | {c4p:<15} | {a4p:<15}")

            input(f"\n{Fore.YELLOW}>> Presiona ENTER para volver...{Style.RESET_ALL}")

        # 7. SELECTOR FINAL (PREDICCIÓN)
        elif opcion == "7":
            print(
                f"\n{Fore.GREEN}🎯 SELECTOR GENÉTICO (PRODUCCIÓN)...{Style.RESET_ALL}"
            )
            config = PredictionConfigDTO(
                total_balls=TOTAL_BALLS,
                ticket_size=TICKET_SIZE,
                num_tickets=15,
                filter_overrides=BEST_SETTINGS,
            )

            pred = GeneticSelectorStrategy().predict(history, config)

            if pred.tickets:
                report.guardar_prediccion(pred.tickets)
                ui.show_prediction_results(pred)
                print(f"\n{Fore.GREEN}🍀 ¡Buena suerte!{Style.RESET_ALL}")
            else:
                print(
                    f"{Fore.RED}❌ No se generaron tickets. Verifica si ejecutaste el paso 5.{Style.RESET_ALL}"
                )

            input(f"\n{Fore.YELLOW}>> Presiona ENTER...{Style.RESET_ALL}")

        else:
            print(f"{Fore.RED}Opción inválida.{Style.RESET_ALL}")


if __name__ == "__main__":
    main()
