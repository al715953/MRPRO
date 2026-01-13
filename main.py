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

        # 6. LABORATORIO (BACKTEST & DUELO)
        elif opcion == "6":
            print(
                f"\n{Fore.CYAN}📡 LABORATORIO DE PRUEBAS (QA & COMPARATIVAS){Style.RESET_ALL}"
            )
            print("1. Test de Cobertura (Solo Fase 1 - Universo)")
            print("2. 🥊 DUELO: AI Sniper vs Heurística Clásica")

            sub_op = input("   👉 Selecciona modo (2): ") or "2"

            try:
                n_test = int(input(f"   ¿Cuántos sorteos simular? (10): ") or 10)
            except:
                n_test = 10

            config = PredictionConfigDTO(
                total_balls=TOTAL_BALLS,
                ticket_size=TICKET_SIZE,
                num_tickets=15,
                backtest_size=n_test,
                filter_overrides=BEST_SETTINGS,
            )

            if sub_op == "1":
                print(
                    f"\n{Fore.BLUE}ℹ️  Verificando calidad de filtros en 'UniverseReduction'...{Style.RESET_ALL}"
                )
                CoverageTester().run(UniverseReductionStrategy(), history, config)

            else:
                # --- MODO DUELO ---
                if HeuristicSelectorStrategy is None:
                    print(
                        f"{Fore.RED}❌ Error: No se encontró 'src/strategies/heuristic_selector.py'.{Style.RESET_ALL}"
                    )
                    input(f"\n{Fore.YELLOW}>> Presiona ENTER...{Style.RESET_ALL}")
                    continue

                engine = BacktestEngine()

                # 1. Corremos la Clásica
                print(
                    f"\n{Fore.YELLOW}🥊 ROUND 1: Lógica Clásica (Sin IA)...{Style.RESET_ALL}"
                )
                res_classic = engine.run(
                    strategy=HeuristicSelectorStrategy(),
                    history=history,
                    config=config,
                    pre_process_strategy=UniverseReductionStrategy(),
                    verbose=True,
                )

                # 2. Corremos la IA
                print(
                    f"\n{Fore.MAGENTA}🥊 ROUND 2: Inteligencia Artificial (Sniper V6)...{Style.RESET_ALL}"
                )
                res_ai = engine.run(
                    strategy=GeneticSelectorStrategy(),
                    history=history,
                    config=config,
                    pre_process_strategy=UniverseReductionStrategy(),
                    verbose=True,
                )

                # 3. COMPARATIVA FINAL
                print(f"\n{Fore.GREEN}🏆 RESULTADO DEL DUELO{Style.RESET_ALL}")
                print(f"{'METRICA':<20} | {'CLÁSICA':<15} | {'IA (V6)':<15}")
                print("-" * 55)
                print(
                    f"{'Ganancia Total':<20} | ${res_classic.earnings:<14,.2f} | ${res_ai.earnings:<14,.2f}"
                )
                print(
                    f"{'Balance Neto':<20} | ${res_classic.net_balance:<14,.2f} | ${res_ai.net_balance:<14,.2f}"
                )

                # Extracción segura de datos
                c3 = res_classic.hit_distribution.get(3, 0)
                a3 = res_ai.hit_distribution.get(3, 0)
                c4p = sum([res_classic.hit_distribution.get(k, 0) for k in [4, 5, 6]])
                a4p = sum([res_ai.hit_distribution.get(k, 0) for k in [4, 5, 6]])

                print(f"{'Aciertos (3)':<20} | {c3:<15} | {a3:<15}")
                print(f"{'Aciertos (4+)':<20} | {c4p:<15} | {a4p:<15}")

            input(f"\n{Fore.YELLOW}>> Presiona ENTER...{Style.RESET_ALL}")

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
