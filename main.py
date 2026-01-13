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
            # CORRECCIÓN DE FECHAS:
            # El loader ya devuelve objetos datetime.date, no strings.
            last_item = history.dates[-1]

            if isinstance(last_item, str):
                # Fallback por si acaso viniera como string
                last_date = datetime.strptime(last_item, "%d/%m/%Y").date()
            else:
                # Ya es un objeto date
                last_date = last_item

            # Comparar fechas (date vs date)
            days_diff = (datetime.now().date() - last_date).days

            # Si pasaron más de 4 días (frecuencia Melate Retro es Mar/Sab)
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
        # Ante la duda, intentamos actualizar
        needs_update = True

    # Ejecutar Scraper si es necesario
    if needs_update:
        print(
            f"\n{Fore.CYAN}📥 Iniciando actualización automática desde Lotería Nacional...{Style.RESET_ALL}"
        )
        try:
            # CORRECCIÓN SCRAPER: Usamos la función exacta de tu archivo scraper.py
            exito, mensaje = scraper.descargar_datos(CSV_FILE_PATH)

            if exito:
                print(f"{Fore.GREEN}{mensaje}{Style.RESET_ALL}\n")
            else:
                print(f"{Fore.YELLOW}{mensaje}{Style.RESET_ALL}\n")

        except AttributeError:
            # Fallback por si cambió el nombre
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
        # ui.show_main_menu() debe existir en tu CLI actualizado
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
            # Override con settings optimizados
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
            # Asumimos que optimize retorna un dict
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

        # 6. BACKTEST & COVERAGE (MOTOR VALIDACIÓN)
        elif opcion == "6":
            print(f"\n{Fore.CYAN}📡 LABORATORIO DE PRUEBAS (QA){Style.RESET_ALL}")
            print(
                "1. Test de Cobertura de Universo (Rápido - Verifica si el ganador entra en la red)"
            )
            print("2. Backtest de Estrategia Completa (AI-Ready - Simulación Realista)")

            sub_op = input("   👉 Selecciona modo (1): ") or "1"

            try:
                default_n = 5 if sub_op == "2" else 10
                n_test = int(
                    input(f"   ¿Cuántos sorteos pasados simular? ({default_n}): ")
                    or default_n
                )
            except:
                n_test = 5

            # Configuración base para tests
            test_settings = BEST_SETTINGS.copy()

            config = PredictionConfigDTO(
                total_balls=TOTAL_BALLS,
                ticket_size=TICKET_SIZE,
                num_tickets=15,  # Simulamos comprar 15 boletos
                backtest_size=n_test,
                filter_overrides=test_settings,
            )

            if sub_op == "1":
                # --- MODO 1: COBERTURA ---
                print(
                    f"\n{Fore.BLUE}ℹ️  Verificando calidad de filtros en 'UniverseReduction'...{Style.RESET_ALL}"
                )
                CoverageTester().run(UniverseReductionStrategy(), history, config)

            else:
                # --- MODO 2: BACKTEST COMPLETO (CON REGENERACIÓN) ---
                print(
                    f"\n{Fore.YELLOW}⚠️  MODO INTENSIVO: Se regenerará el universo para cada sorteo.{Style.RESET_ALL}"
                )
                print(
                    f"{Fore.YELLOW}⏳ Esto tomará tiempo (aprox. 5-8 seg por sorteo)...{Style.RESET_ALL}"
                )

                engine = BacktestEngine()

                # Ejecutamos el motor inyectando la estrategia de Pre-Proceso
                engine.run(
                    strategy=GeneticSelectorStrategy(),  # Estrategia a probar (Sniper + AI)
                    history=history,
                    config=config,
                    pre_process_strategy=UniverseReductionStrategy(),  # <--- LA CLAVE: Regenera el entorno
                )

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

            # Ejecutamos la estrategia principal
            # (Esta leerá el universo generado en la opción 5)
            pred = GeneticSelectorStrategy().predict(history, config)

            if pred.tickets:
                report.guardar_prediccion(pred.tickets)
                ui.show_prediction_results(pred)
                print(f"\n{Fore.GREEN}🍀 ¡Buena suerte, Arquitecto!{Style.RESET_ALL}")
            else:
                print(
                    f"{Fore.RED}❌ No se generaron tickets. Verifica si ejecutaste el paso 5.{Style.RESET_ALL}"
                )

            input(f"\n{Fore.YELLOW}>> Presiona ENTER...{Style.RESET_ALL}")

        else:
            print(f"{Fore.RED}Opción inválida.{Style.RESET_ALL}")


if __name__ == "__main__":
    main()
