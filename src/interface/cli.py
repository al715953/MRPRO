import os
from src.domain.dtos import PredictionResultDTO

class ConsoleUI:
    def clear_screen(self):
        os.system('cls' if os.name == 'nt' else 'clear')

    def show_welcome(self):
        self.clear_screen()
        print("══════════════════════════════════════════════════")
        print("       🎱  MELATE RETRO PRO - SYSTEM V2  🎱       ")
        print("══════════════════════════════════════════════════")

    def get_main_menu_option(self) -> str:
        print("\nSelecciona una opción:")
        print("1. 🔄 Actualizar Base de Datos (Scraper)")
        print("2. 🎫 Validar Resultados (Manual o CSV)")
        print("3. 🧠 Ejecutar Predicción (Generar Números)")
        print("4. 🧪 Correr Backtest (Prueba Histórica)")
        print("5. 🚪 Salir")
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