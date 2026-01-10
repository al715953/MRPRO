import os
from src.data_access.config import CYAN, RESET, VERDE, BLANCO_B
from src.domain.dtos import PredictionResultDTO


class ConsoleUI:
    def clear_screen(self):
        os.system("cls" if os.name == "nt" else "clear")

    def show_welcome(self):
        self.clear_screen()
        print("══════════════════════════════════════════════════")
        print(f"            🎱  MRPRO {VERDE}- {BLANCO_B}SYSTEM V2  🎱       ")
        print("══════════════════════════════════════════════════")

    def mostrar_logo(self):
        linea = "*" * 50
        # print(f"{VERDE}{linea}")
        print(f"{BLANCO_B}  MRPRO {VERDE}| {BLANCO_B}Retro Pro Analyzer")
        print(f"{VERDE}{linea}{RESET}")

    def get_main_menu_option(self) -> str:
        print("\nSelecciona una opción:")
        print("1. 🔮 Generar Predicción")
        print("2. 🧪 Backtesting (Prueba Histórica)")
        print("3 🎫 Validar Resultados (Manual o CSV)")
        print("4.🧠 Entrenar/Optimizar Estrategia (AI Trainer)")
        print("0. 🚪 Salir")
        return input(">> ")

    def get_strategy_selection(self) -> str:
        print("\n🧠 SELECCIÓN DE INTELIGENCIA:")
        print("1. Monte Carlo (Aleatoriedad Pura - Baseline)")
        print("2. [Próximamente] Genética")
        print("3. [Próximamente] Híbrida")

        selection = input(">> Elige una estrategia (1-3): ")

        if selection == "1":
            return "MONTE_CARLO"
        # Aquí agregaremos más opciones luego
        else:
            print("⚠️ Selección inválida, usando Monte Carlo por defecto.")
            return "MONTE_CARLO"

    def show_prediction_results(self, result: PredictionResultDTO):
        print(f"\n🎫 RESULTADOS GENERADOS POR: {result.strategy_name}")
        print("=" * 40)

        for i, ticket in enumerate(result.tickets, 1):
            # Formato visual: [ 01, 05, 12, ... ]
            # Usamos :02d para que el 5 se vea como '05'
            ticket_str = ", ".join([f"{num:02d}" for num in ticket])
            print(f"Ticket #{i}: [ {ticket_str} ]")

        print("=" * 40)
        print("✅ Proceso finalizado.")
