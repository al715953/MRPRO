import sys
import os

# --- 1. AJUSTE DE RUTA ---
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# --- 2. IMPORTACIONES ---
from src.domain.dtos import PredictionConfigDTO
from src.data_access.loader import MelateLoader
from src.data_access import scraper
from src.data_access.config import CSV_FILE_PATH, TICKET_SIZE, TOTAL_BALLS
from src.data_access.config import CYAN, RESET
from src.interface.cli import ConsoleUI


# CORRECCIÓN: Importamos BacktestEngine (no Service)
from src.core.backtester import BacktestEngine
from src.strategies.monte_carlo import MonteCarloStrategy


def obtener_estrategia():
    print("\n--- SELECCIONA UNA ESTRATEGIA ---")
    print("1. Monte Carlo (Simulación Ponderada)")
    opcion = input(">> Elige una opción: ")
    if opcion == "1":
        return MonteCarloStrategy()
    else:
        print("⚠ Opción no válida. Usando Monte Carlo por defecto.")
        return MonteCarloStrategy()


def main():
    try:
        # Actualizo archivo de resultados de ser necesario
        actualizado, mensaje = scraper.descargar_datos(CSV_FILE_PATH)

        # Actualizo información
        loader = MelateLoader(CSV_FILE_PATH)
        history = loader.load_data()

        if not history.winning_numbers:
            print("⚠ Advertencia: No se encontraron sorteos en el archivo.")

        # Instancia del motor correcto
        backtester = BacktestEngine()

    except FileNotFoundError:
        print(f"❌ ERROR: No se encontró el archivo en: {CSV_FILE_PATH}")
        return
    except Exception as e:
        print(f"❌ ERROR INESPERADO AL INICIAR: {e}")
        return

    # 1. Instanciar la interfaz
    ui = ConsoleUI()

    while True:
        # 2. Pantalla de inicio
        ui.show_welcome()
        print(f"{CYAN}>> Estado del sistema: {mensaje}{RESET}\n")
        print(f"📂 Sorteos cargados: {len(history.winning_numbers)}")

        opcion = ui.get_main_menu_option()

        if opcion == "1":
            print("\nMODO GENERAR")
            strategy = obtener_estrategia()
            try:
                n_tickets = int(input("\n¿Cuántos tickets quieres generar? (Ej. 5): "))
            except ValueError:
                n_tickets = 5

            config = PredictionConfigDTO(
                total_balls=TOTAL_BALLS, ticket_size=TICKET_SIZE, num_tickets=n_tickets
            )

            print(f"\n⚙ Ejecutando estrategia: {strategy.__class__.__name__}...")
            resultado = strategy.predict(history, config)

            print("\n" + "=" * 40)
            print(f"🎫 RESULTADOS: {resultado.strategy_name}")
            print("=" * 40)
            for i, ticket in enumerate(resultado.tickets, 1):
                print(f"Ticket #{i}: {ticket}")
            print("=" * 40)
            input("\nPresiona ENTER para continuar...")

        elif opcion == "2":
            print("\n🧪 MODO BACKTESTING")
            # 2. Obtenemos el NOMBRE (string) de la estrategia
            strategy_key = ui.get_strategy_selection()
            # 3. TRADUCTOR: Mapeamos el string a una INSTANCIA real
            strategies_map = {
                "MONTE_CARLO": MonteCarloStrategy(),
                # Aquí añadirás: "GENETICO": GeneticStrategy(), etc.
            }
            # Obtenemos el objeto real
            selected_strategy = strategies_map.get(strategy_key)

            try:
                raw_input = input("\n¿Cuántos sorteos probar? (Ej. 20): ")
                test_size = int(raw_input) if raw_input else 10
            except ValueError:
                test_size = 10

            config = PredictionConfigDTO(
                total_balls=TOTAL_BALLS,
                ticket_size=TICKET_SIZE,
                num_tickets=20,  # Simulamos comprar 5 boletos por sorteo
                backtest_size=test_size,
            )

            print(f"\n⏳ Ejecutando Backtest en los últimos {test_size} sorteos...")
            # Ejecutamos y guardamos el resultado
            report = backtester.run(selected_strategy, history, config)
            print("-" * 40)
            # --- IMPRIMIR REPORTE ---
            print("\n" + "█" * 40)
            print(f"📊 REPORTE FINAL: {report.strategy_name}")
            print("█" * 40)
            print(f"📅 Sorteos analizados: {report.total_draws_tested}")
            print(f"💰 Inversión Total:   ${report.investment:,.2f}")
            print(f"🏆 Ganancias Totales: ${report.earnings:,.2f}")
            print("-" * 40)

            balance_color = "🟢" if report.net_balance >= 0 else "🔴"
            print(f"⚖  BALANCE NETO:      {balance_color} ${report.net_balance:,.2f}")
            print("-" * 40)
            print("🎯 Aciertos:")
            for hits, count in sorted(report.hit_distribution.items(), reverse=True):
                if count > 0:
                    print(f"   {hits} aciertos: {count} veces")
            print("█" * 40)

            input("\nPresiona ENTER para volver al menú...")

        elif opcion == "3":
            history = loader.load_data()
            print("¡Datos actualizados!")
            input("Enter...")

        elif opcion == "4":
            sys.exit()


if __name__ == "__main__":
    main()
