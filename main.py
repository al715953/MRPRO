import sys
import os
from colorama import Fore, Style

# --- 1. AJUSTE DE RUTA ---
# Asegura que Python encuentre los módulos sin importar desde dónde se ejecute
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# --- 2. IMPORTACIONES ---
# Datos y Configuración
from src.domain.dtos import PredictionConfigDTO
from src.data_access.loader import MelateLoader
from src.data_access.config import (
    CSV_FILE_PATH,
    TICKET_SIZE,
    TOTAL_BALLS,
    BEST_SETTINGS,
)

# Acceso a Datos y Reportes
from src.data_access import scraper
import src.data_access.report as report  # Módulo para guardar CSVs

# Interfaz
from src.interface.cli import ConsoleUI

# --- ESTRATEGIAS (EL ARSENAL) ---
from src.strategies.monte_carlo import MonteCarloStrategy
from src.strategies.universe_reduction import UniverseReductionStrategy
from src.strategies.genetic_selector import GeneticSelectorStrategy

# --- MOTORES DE CÁLCULO ---
from src.core.backtester import BacktestEngine
from src.core.optimizer import StrategyOptimizer
from src.core.coverage_tester import CoverageTester


def main():
    # A. Inicialización de Interfaz
    ui = ConsoleUI()
    ui.show_welcome()

    # B. Carga de Datos Históricos
    print(
        f"{Fore.CYAN}📂 Cargando histórico desde: {CSV_FILE_PATH}...{Style.RESET_ALL}"
    )
    loader = MelateLoader(CSV_FILE_PATH)
    history = loader.load_data()

    # Validación básica de carga
    if not history.dates:
        print(
            f"{Fore.RED}❌ Error: No hay datos históricos. Intentando actualizar...{Style.RESET_ALL}"
        )
        # Intentamos descargar si falla la carga local
        try:
            scraper.actualizar_base_datos()
            history = loader.load_data()
        except Exception as e:
            print(f"Error actualizando: {e}")

    # --- BUCLE PRINCIPAL ---
    while True:
        ui.mostrar_logo()
        opcion = ui.get_main_menu_option()

        # ========================================================
        # OPCIÓN 1: GENERACIÓN MAESTRA (FLUJO AUTOMÁTICO 5 -> 7)
        # ========================================================
        if opcion == "1":
            print(
                f"\n{Fore.YELLOW}🚀 INICIANDO SISTEMA DE PREDICCIÓN MAESTRA...{Style.RESET_ALL}"
            )
            print(
                "Se ejecutarán 2 fases: Generación de Universo -> Selección Genética."
            )

            # --- FASE 1: CREAR EL LAGO (UNIVERSO) ---
            print(
                f"\n{Fore.MAGENTA}🔹 FASE 1: Generando Universo de Alta Probabilidad...{Style.RESET_ALL}"
            )

            # Usamos los mejores filtros conocidos
            current_settings = BEST_SETTINGS.copy()
            current_settings["verbose"] = False  # Silencioso para no ensuciar pantalla

            # Configuración para el Universo
            config_universe = PredictionConfigDTO(
                total_balls=TOTAL_BALLS,
                ticket_size=TICKET_SIZE,
                num_tickets=10,  # Irrelevante aquí, genera miles
                filter_overrides=current_settings,
            )

            # Ejecutar Estrategia 1 (Generar el CSV grande)
            strategy_universe = UniverseReductionStrategy()
            strategy_universe.predict(history, config_universe)
            print("✅ Universo generado y optimizado.")

            # --- FASE 2: PESCA DE ÉLITE (SELECTOR) ---
            print(
                f"\n{Fore.GREEN}🔹 FASE 2: Selección Genética y Diversificación...{Style.RESET_ALL}"
            )

            # Configuración Final (Tus 15 boletos reales)
            config_final = PredictionConfigDTO(
                total_balls=TOTAL_BALLS,
                ticket_size=TICKET_SIZE,
                num_tickets=15,  # <--- AQUÍ DECIDES CUÁNTOS JUGAR
            )

            # Ejecutar Estrategia 2 (Leer CSV y filtrar)
            strategy_selector = GeneticSelectorStrategy()
            prediction = strategy_selector.predict(history, config_final)

            # --- RESULTADOS ---
            if prediction.tickets:
                # 1. Guardar en CSV de apuestas
                report.guardar_prediccion(prediction.tickets)

                # 2. Mostrar en pantalla
                ui.show_prediction_results(prediction)

                print(
                    f"\n{Fore.YELLOW}✨ ¡PROCESO MAESTRO COMPLETADO! ✨{Style.RESET_ALL}"
                )
                print(
                    f"Tus mejores {len(prediction.tickets)} jugadas están listas en 'data/Mis_Apuestas.csv'"
                )
            else:
                print(
                    f"\n{Fore.RED}❌ Algo falló en la selección. Revisa si se generó el universo.{Style.RESET_ALL}"
                )

            input(f"\n{Fore.YELLOW}>> Presiona ENTER para volver...{Style.RESET_ALL}")

        # ==========================================
        # OPCIÓN 2: BACKTESTING (SIMULACIÓN $)
        # ==========================================
        elif opcion == "2":
            print(
                f"\n{Fore.GREEN}🧪 INICIANDO BACKTESTING FINANCIERO...{Style.RESET_ALL}"
            )

            config = PredictionConfigDTO(
                total_balls=TOTAL_BALLS,
                ticket_size=TICKET_SIZE,
                num_tickets=15,
                backtest_size=10,  # Prueba rápida de 10 sorteos
                filter_overrides=BEST_SETTINGS,
            )

            engine = BacktestEngine()
            # Probamos la estrategia Monte Carlo por defecto
            engine.run(MonteCarloStrategy(), history, config)

            input(f"\n{Fore.YELLOW}>> Presiona ENTER para volver...{Style.RESET_ALL}")

        # ==========================================
        # OPCIÓN 3: VALIDAR RESULTADOS (WEB)
        # ==========================================
        elif opcion == "3":
            print(
                f"\n{Fore.BLUE}🎫 VALIDANDO RESULTADOS CON LOTENAL...{Style.RESET_ALL}"
            )
            try:
                # Esta función debe existir en scraper o report según tu implementación
                scraper.validar_apuestas()
            except AttributeError:
                print("Función de validación en construcción.")
            except Exception as e:
                print(f"Error en validación: {e}")

            input(f"\n{Fore.YELLOW}>> Presiona ENTER para volver...{Style.RESET_ALL}")

        # ==========================================
        # OPCIÓN 4: OPTIMIZADOR (AI TRAINER)
        # ==========================================
        elif opcion == "4":
            print(
                f"\n{Fore.MAGENTA}🧠 ENTRENANDO ESTRATEGIA (GRID SEARCH)...{Style.RESET_ALL}"
            )

            base_config = PredictionConfigDTO(
                total_balls=TOTAL_BALLS,
                ticket_size=TICKET_SIZE,
                num_tickets=15,
                backtest_size=50,  # 50 sorteos para una buena optimización
            )

            optimizer = StrategyOptimizer(MonteCarloStrategy(), history)
            best_params = optimizer.run_grid_search(base_config)

            print(
                f"\n{Fore.GREEN}✅ MEJOR CONFIGURACIÓN:{Style.RESET_ALL} {best_params}"
            )
            print(
                "Copia esto en src/data_access/config.py si deseas guardarlo permanentemente."
            )

            input(f"\n{Fore.YELLOW}>> Presiona ENTER para volver...{Style.RESET_ALL}")

        # ==========================================
        # OPCIÓN 5: GENERADOR DE UNIVERSO (BIG DATA)
        # ==========================================
        elif opcion == "5":
            print(
                f"\n{Fore.MAGENTA}🌌 GENERANDO UNIVERSO REDUCIDO (MANUAL)...{Style.RESET_ALL}"
            )
            print("Creando 'lago de pesca' de alta probabilidad...")

            current_settings = BEST_SETTINGS.copy()
            config = PredictionConfigDTO(
                total_balls=TOTAL_BALLS,
                ticket_size=TICKET_SIZE,
                num_tickets=10,
                filter_overrides=current_settings,
            )

            strategy = UniverseReductionStrategy()
            strategy.predict(history, config)

            print(
                f"\n{Fore.GREEN}✅ Archivo 'data/universo_reducido.csv' generado.{Style.RESET_ALL}"
            )
            input(f"\n{Fore.YELLOW}>> Presiona ENTER para volver...{Style.RESET_ALL}")

        # ==========================================
        # OPCIÓN 6: COVERAGE TESTER (CALIDAD)
        # ==========================================
        elif opcion == "6":
            print(
                f"\n{Fore.CYAN}📡 VALIDANDO COBERTURA (HIT RATIO)...{Style.RESET_ALL}"
            )

            try:
                dias = input("¿Sorteos a validar? (Enter=5): ")
                n_test = int(dias) if dias.strip() else 5
            except:
                n_test = 5

            current_settings = BEST_SETTINGS.copy()
            current_settings["verbose"] = False  # Silenciar logs

            config = PredictionConfigDTO(
                total_balls=TOTAL_BALLS,
                ticket_size=TICKET_SIZE,
                num_tickets=10,
                backtest_size=n_test,
                filter_overrides=current_settings,
            )

            tester = CoverageTester()
            tester.run(UniverseReductionStrategy(), history, config)

            input(f"\n{Fore.YELLOW}>> Presiona ENTER para volver...{Style.RESET_ALL}")

        # ==========================================
        # OPCIÓN 7: SELECTOR FINAL (MANUAL)
        # ==========================================
        elif opcion == "7":
            print(
                f"\n{Fore.GREEN}🎯 EJECUTANDO SELECTOR FINAL (MANUAL)...{Style.RESET_ALL}"
            )
            print("Analizando 'universo_reducido.csv' existente...")

            config = PredictionConfigDTO(
                total_balls=TOTAL_BALLS,
                ticket_size=TICKET_SIZE,
                num_tickets=15,
            )

            strategy = GeneticSelectorStrategy()
            prediction = strategy.predict(history, config)

            if prediction.tickets:
                report.guardar_prediccion(prediction.tickets)
                ui.show_prediction_results(prediction)
                print(
                    f"\n{Fore.YELLOW}✅ Boletos listos en 'data/Mis_Apuestas.csv'.{Style.RESET_ALL}"
                )
            else:
                print(
                    f"\n{Fore.RED}❌ Error: No se generaron tickets. ¿Ejecutaste la opción 5 antes?{Style.RESET_ALL}"
                )

            input(f"\n{Fore.YELLOW}>> Presiona ENTER para volver...{Style.RESET_ALL}")

        # ==========================================
        # SALIR
        # ==========================================
        elif opcion == "0":
            print("👋 ¡Hasta luego! Que la probabilidad esté a tu favor.")
            sys.exit()

        else:
            print("⚠️ Opción no válida.")


if __name__ == "__main__":
    main()
